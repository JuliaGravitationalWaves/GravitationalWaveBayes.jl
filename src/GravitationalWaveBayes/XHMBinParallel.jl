# ------------------------------------------------------------------------------------------------
# Phenom models with a mode-sum factorization on the device, bin-parallel with forward tangents: IMRPhenomXHM
# and IMRPhenomXP (`XPBinParallel.jl`). The XAS `:fwd` factorization, per "mode":
#
#     h_d(f) = sum_k c_kd H_k(f) e^{-2 pi i f tau_d},
#
# where every H_k depends only on NT intrinsic coordinates theta (a function of the sampled parameters) and c_kd
# carries everything else. XHM: the 5 modes (2,2), (2,1), (3,3), (3,2), (4,4) at dL = 1 Gpc and phase 0, theta =
# (mc, q, chi1, chi2), c_kd = (Fp fp_k(theta_jn) + Fc fc_k(theta_jn)) e^{i m_k phase} / dL (checked exactly: each
# mode depends on dL and phase only through that factor). XP: see `XPBinParallel.jl`. tau_d = t_c - t_shift(theta) +
# detector delay. A model is a "spec" type with the methods `_mbp_nm` (modes), `_mbp_nt` (tangents), `_mbp_np`
# (sampled rows), `_mbp_setup_type`, `_mbp_setup_store!` (or staged: `_mbp_setup_stage!`), `_mbp_modes`, `_mbp_theta`, `_mbp_ext`. Per walker:
#   (a) one thread per walker: the setup (flattened from its TYPE, `_xhm_flatten`) and the time shift, with forward
#       tangents along theta (one Enzyme width-NT forward pass);
#   (b) c_kd and tau_d + t_shift (`_mbp_ext`);
#   (c) device, one work-group: one pass over the bins; each bin evaluates the modes with their NT tangents (Enzyme
#       forward through `_mbp_modes` on lazy views of the flat setup) and accumulates, per detector,
#           <h,h>, <d,h>, sum f conj(d) h w, Q_k = sum w conj(d) H_k S, R_k = sum w conj(h) H_k S,
#           d<h,h>/dtheta_j, d<d,h>/dtheta_j                       (S = e^{-2 pi i f tau_d}, w = 1/PSD)
#       (`_mbp_nv(NM, NT)` = 5 + 4 NM + 3 NT numbers per detector);
#   (d) the log-likelihood and its gradient through a FIRST-ORDER surrogate in the sampled parameters,
#           <d,h>(x) = <d,h> + sum_k Q_k dc_k - 2 pi i R dtau + sum_j d<d,h>/dtheta_j dtheta_j,
#           <h,h>(x) = <h,h> + 2 Re sum_k R_k dc_k + sum_j d<h,h>/dtheta_j dtheta_j,
#       exact in value and gradient at the walker (hand-written `_marg_combine_grad` + the surrogate's derivative).
# `setup = :host`: the per-walker stages (theta, kappa and the raw constants, (b), (d) with Enzyme reverse over the
# seeded pre-sums) on host threads and the setup (a) on one host thread per walker. `setup = :device`:
# everything on the backend -- per walker, a primal pass for theta, kraw, (b) and kappa; forward Jacobians of (b) and
# kappa along the sampled rows (`_mbp_jac_kernel!`, tangent groups); (a) in tangent groups; and (d) as the contraction
# of those Jacobians with the surrogate's (linear) first-order part (`_mbp_post_kernel!`). Only the positions go to
# the device and the log-likelihoods and gradients come back.
# Same likelihood options and time frame as the models' CPU likelihoods (no phase marginalization).
# ------------------------------------------------------------------------------------------------

# ---- flat layout of the XHM setup (sys, ms21, ms33, ms44, m32), generated from its TYPE ----
# Float -> 1 slot, Complex -> 2, Integer (mode labels l, m) -> 1 (stored as Float64, rounded back), DataType (`T`
# fields) -> none (rebuilt as the element type of the flat array).
_xhm_flatn(::Type{<:AbstractFloat}) = 1
_xhm_flatn(::Type{<:Complex}) = 2
_xhm_flatn(::Type{<:Integer}) = 1
_xhm_flatn(::Type{DataType}) = 0
_xhm_flatn(::Type{<:Type}) = 0
_xhm_flatn(::Type{T}) where {T<:Union{Tuple,NamedTuple}} = sum(_xhm_flatn(fieldtype(T, i)) for i in 1:fieldcount(T); init=0)

function _xhm_flatexprs(T, ex)
    T <: AbstractFloat && return Any[ex]
    T <: Complex && return Any[:(real($ex)), :(imag($ex))]
    T <: Integer && return Any[:(Float64($ex))]
    (T === DataType || T <: Type) && return Any[]
    out = Any[]
    for i in 1:fieldcount(T)
        append!(out, _xhm_flatexprs(fieldtype(T, i), :(getfield($ex, $i))))
    end
    return out
end
@generated _xhm_flatten(x) = Expr(:block, Expr(:meta, :inline), Expr(:tuple, _xhm_flatexprs(x, :x)...))

function _xhm_unflatexpr(T, off::Ref{Int})
    if T <: AbstractFloat
        off[] += 1
        return :(@inbounds K[$(off[]), w])
    elseif T <: Complex
        off[] += 2
        return :(Complex(@inbounds(K[$(off[] - 1), w]), @inbounds(K[$(off[]), w])))
    elseif T <: Integer
        off[] += 1
        return :(unsafe_trunc($T, @inbounds(K[$(off[]), w]) + 0.5))
    elseif T === DataType || T <: Type
        return :(eltype(K))
    end
    vals = [_xhm_unflatexpr(fieldtype(T, i), off) for i in 1:fieldcount(T)]
    return T <: NamedTuple ? :(NamedTuple{$(fieldnames(T))}(($(vals...),))) : Expr(:tuple, vals...)
end
# The setup of type `ST` from column `w` of the flat array `K` (`_xhm_flatten`'s order).
@generated function _xhm_unflatten(::Val{ST}, K, w) where {ST}
    Expr(:block, Expr(:meta, :inline), _xhm_unflatexpr(ST, Ref(0)))
end

# Lazy view of a flattened (sub)structure of type T, at offset `off` in column `w` of K: `getproperty` / `getindex` /
# destructuring read K at the point of use, so device code never materializes the whole setup (the unflattened
# NamedTuples and their 4 forward-mode shadows were an ~18 KB stack frame per thread in the sums kernel).
struct _XhmFV{T,A}
    a::A
    w::Int
    off::Int
end
@inline _xhm_fv(::Type{T}, a, w, off) where {T} = _XhmFV{T,typeof(a)}(a, Int(w), off)
_xhm_fieldoff(T, i) = sum(_xhm_flatn(fieldtype(T, j)) for j in 1:i-1; init=0)
function _xhm_leafexpr(FT, o)                       # value of leaf type FT at relative offset o (1-based first slot)
    if FT <: AbstractFloat
        return :(@inbounds getfield(v, :a)[getfield(v, :off) + $o, getfield(v, :w)])
    elseif FT <: Complex
        return :(Complex(@inbounds(getfield(v, :a)[getfield(v, :off) + $o, getfield(v, :w)]),
                         @inbounds(getfield(v, :a)[getfield(v, :off) + $(o + 1), getfield(v, :w)])))
    elseif FT <: Integer
        return :(unsafe_trunc($FT, @inbounds(getfield(v, :a)[getfield(v, :off) + $o, getfield(v, :w)]) + 0.5))
    elseif FT === DataType || FT <: Type
        return :(eltype(getfield(v, :a)))
    end
    return :(_xhm_fv($FT, getfield(v, :a), getfield(v, :w), getfield(v, :off) + $(o - 1)))
end
@generated function _xhm_fvfield(v::_XhmFV{T}, ::Val{i}) where {T,i}
    Expr(:block, Expr(:meta, :inline), _xhm_leafexpr(fieldtype(T, i), _xhm_fieldoff(T, i) + 1))
end
@inline Base.getproperty(v::_XhmFV{T}, s::Symbol) where {T} = _xhm_fvfield(v, Val(Base.fieldindex(T, s)))
@inline Base.getindex(v::_XhmFV{T}, i::Int) where {T<:Tuple} = _xhm_fvindex(v, i)
# Tuples: literal-position access for heterogeneous ones; homogeneous leaf tuples also by a runtime index.
@generated function _xhm_fvindex(v::_XhmFV{T}, i::Int) where {T}
    FT = fieldtype(T, 1)
    if all(j -> fieldtype(T, j) === FT, 1:fieldcount(T)) && (FT <: AbstractFloat || FT <: Complex)
        n = _xhm_flatn(FT)
        o = :(getfield(v, :off) + $n * (i - 1) + 1)
        val = FT <: AbstractFloat ? :(@inbounds getfield(v, :a)[$o, getfield(v, :w)]) :
              :(Complex(@inbounds(getfield(v, :a)[$o, getfield(v, :w)]), @inbounds(getfield(v, :a)[$o + 1, getfield(v, :w)])))
        return Expr(:block, Expr(:meta, :inline), val)
    end
    branches = foldr(1:fieldcount(T); init=:(throw(BoundsError()))) do j, rest
        :(i == $j ? _xhm_fvfield(v, Val($j)) : $rest)
    end
    return Expr(:block, Expr(:meta, :inline), branches)
end
@inline Base.length(::_XhmFV{T}) where {T<:Tuple} = fieldcount(T)
@inline Base.indexed_iterate(v::_XhmFV{T}, i::Int, state=1) where {T<:Tuple} = (v[i], i + 1)
@inline Base.iterate(v::_XhmFV{T}, i::Int=1) where {T<:Tuple} = i > fieldcount(T) ? nothing : (v[i], i + 1)
# The 5 setup parts (sys, ms21, ms33, ms44, m32) of column w as views.
@generated function _xhm_views(::Val{ST}, K, w) where {ST}
    offs = cumsum([0; [_xhm_flatn(fieldtype(ST, j)) for j in 1:fieldcount(ST) - 1]])
    Expr(:block, Expr(:meta, :inline),
         Expr(:tuple, (:(_xhm_fv($(fieldtype(ST, j)), K, w, $(offs[j]))) for j in 1:fieldcount(ST))...))
end

# Setup stages by part: XHM mode setup k (2..5: ms21, ms33, ms44, m32) from the system setup at offset 0
# (a lazy view: its tangents through Enzyme's shadow of K), stored at part k of the flat setup type ST. In one function
# the XHM setup was a 40 KB stack frame per thread even at one tangent (sm_70).
@inline _xhm_mode_part(::Val{2}, sys) = KernelFused._xhm_mode_setup(2, 1, sys)
@inline _xhm_mode_part(::Val{3}, sys) = KernelFused._xhm_mode_setup(3, 3, sys)
@inline _xhm_mode_part(::Val{4}, sys) = KernelFused._xhm_mode_setup(4, 4, sys)
@inline _xhm_mode_part(::Val{5}, sys) = KernelFused._xhm_mode32_setup(sys)
@generated _xhm_part_off(::Val{ST}, ::Val{j}) where {ST,j} = sum(_xhm_flatn(fieldtype(ST, i)) for i in 1:j-1; init=0)
@inline function _xhm_mode_stage!(KT, w, stv::Val{ST}, kv::Val) where {ST}
    ms = _xhm_mode_part(kv, _xhm_fv(fieldtype(ST, 1), KT, w, 0))
    _xasbp_store_col!(KT, w, _xhm_flatten(ms), _xhm_part_off(stv, kv))
    return nothing
end

# Type of the Float64 XHM setup (the layout key).
_xhm_setup_type() = typeof(KernelFused._xhm_setup(36.0, 29.0, 0.1, 0.1, 1.0, 0.0, 20.0))


# ---- pow-free per-frequency XHM modes ----
# `KernelFused._xhm_modes` (GravitationalWaveforms' evaluators) selects regions with `ifelse`, so every bin evaluates the inspiral,
# intermediate AND ringdown forms of every mode, and the inspiral forms use `Mf^(p/q)`: ptxas showed 216 calls to the
# FP64 `pow` per bin in the sums kernel (with the tangents). Here: real branches, every Mf^(k/3) and Mf^(-7/6) from one
# cbrt and a sqrt, sin and cos separately (Enzyme has no device rule for `__nv_sincos`), the (2,2) mode through XAS's
# pow-free `_xas_pa_core` (no-op accumulator). Same values to rounding (`scripts/check_xhm_binparallel.jl`).
struct _NoAcc end
@inline _acc(a::_NoAcc, j::Int, v) = a

# Inspiral amplitude (`_xhm_insp_amp_ansatz` = PN amplitude + pseudo-PN terms); c13 = Mf^(1/3), m76 = Mf^(-7/6).
@inline function _xhmf_insp_amp(Mf, c13, m76, co)
    pn = co.pn
    c23 = c13 * c13
    series = pn[1] + c13 * pn[2] + c23 * pn[3] + Mf * pn[4] + Mf * c13 * pn[5] + Mf * c23 * pn[6] + Mf * Mf * pn[7]
    # |series| with a safe sqrt: the (complex) series is exactly 0 for the odd modes at equal mass + equal chi_z, where
    # the device derivative of `abs` (hypot) gave 0/0 tangents (CUDA)
    a2 = abs2(series)
    pnAmp = ifelse(a2 > 0, sqrt(ifelse(a2 > 0, a2, one(a2))), zero(a2)) * co.PNglobalfactor * m76 * co.ampNorm
    r = Mf / co.fcutInsp
    r13 = cbrt(r)
    r2 = r * r
    IC0, IC1, IC2 = co.IC
    pseudo = (r2 * r13 * IC0 + r2 * r13 * r13 * IC1 + r2 * r * IC2) * m76 * co.PNdominant
    return pnAmp + pseudo
end
# Inspiral phase integral (`_xhm_insp_phase_ansatz_int`): sum_{i=0}^{14} (phi_i + phiL_i log Mf) Mf^((i-7)/3).
@generated function _xhmf_insp_phase(Mf, c13, lMf, phi, phiL)
    ex = [:(p = 1 / (Mf * Mf * c13)), :(s = (phi[1] + phiL[1] * lMf) * p)]
    for i1 in 2:15
        push!(ex, :(p *= c13), :(s += (phi[$i1] + phiL[$i1] * lMf) * p))
    end
    Expr(:block, Expr(:meta, :inline), ex..., :(return s))
end
@inline function _xhmf_inter_amp(Mf, m76, c, fmid, invfhalf)
    x = (Mf - fmid) * invfhalf
    return (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x * (c[5] + x * (c[6] + x * (c[7] + x * c[8]))))))) * m76
end
@inline function _xhmf_nonmixing(sgn, Mf, c13, lMf, m76, amp0, ms)
    if Mf < ms.fAmpMatchIN
        amp = _xhmf_insp_amp(Mf, c13, m76, ms.inspcoeffs)
        phi = _xhmf_insp_phase(Mf, c13, lMf, ms.phi_lm, ms.phiL_lm) + ms.C1INSP * Mf + ms.CINSP
    elseif Mf < ms.fBoundIM
        amp = _xhmf_inter_amp(Mf, m76, ms.intamp, ms.fmid, ms.invfhalf)
        phi = GravitationalWaveforms._xhm_inter_phase_ansatzint(Mf, ms.intphase, ms.fring, ms.fdamp)
    else
        amp = GravitationalWaveforms._xhm_rd_amp_ansatz(Mf, ms.rdamp)
        phi = GravitationalWaveforms._xhm_rd_phase_ansatzint(Mf, ms.rdphase, ms.fring, ms.fdamp) + ms.C1RD * Mf + ms.CRD
    end
    phi += ms.deltaphiLM
    s = sin(phi); c = cos(phi)
    return (sgn * amp0 * amp) * Complex(c, s)
end
@inline function _xhmf_32(Mf, c13, lMf, m76, amp0, phi22, sys, m32)
    core = m32.core
    if Mf < m32.fAmpMatchIN
        amp = _xhmf_insp_amp(Mf, c13, m76, m32.inspcoeffs)
        phi = _xhmf_insp_phase(Mf, c13, lMf, m32.phi_lm, m32.phiL_lm) + m32.C1INSP * Mf + m32.CINSP + m32.deltaphiLM
        s = sin(phi); c = cos(phi)
        h = amp * Complex(c, s)
    elseif Mf < m32.fAmpMatchIM
        amp = _xhmf_inter_amp(Mf, m76, m32.intamp, m32.fmid, m32.invfhalf)
        phi = GravitationalWaveforms._xhm_inter_phase32_ansatzint(Mf, m32.intphase32, core.fring32, core.fdamp32) + m32.deltaphiLM
        s = sin(phi); c = cos(phi)
        h = amp * Complex(c, s)
    else
        # `_xhm_s2s` with the (2,2) phase from the caller and Mf^(-7/6) from m76
        a22 = core.amp22coeffs
        amp22 = KernelFused._xas_amp_rd_val(Mf, a22.amp_rd.gamma1, a22.amp_rd.gamma2, a22.amp_rd.gamma3,
                                            sys.coeffs22.fdamp, sys.coeffs22.fRD)
        s22 = sin(phi22); c22 = cos(phi22)
        wf22R = (amp22 * a22.amp0 * m76) * Complex(c22, s22)
        amplm = GravitationalWaveforms._xhm_rd_amp32_ansatz(Mf, core.rdamp32)
        philm = GravitationalWaveforms._xhm_spheroidal32_ansatzint(Mf, core.sph32.sph, core.sph32.phi0_S, core.fring32, core.fdamp32)
        slm = sin(philm); clm = cos(philm)
        z = conj(core.mc2) * wf22R + conj(core.mc3) * (amplm * Complex(clm, slm))
        pc = m32.C1RD * Mf + m32.CRD + m32.deltaphiLM
        sc = sin(pc); cc = cos(pc)
        h = z * Complex(cc, sc)
    end
    return -amp0 * h
end
# The five modes (`KernelFused._xhm_modes` order), pow-free.
@inline function _xhm_modes_fast(f, sys, ms21, ms33, ms44, m32)
    Mf = sys.coeffs22.M * sys.coeffs22.GMsun_over_c3 * f
    c13 = cbrt(Mf)
    m76 = 1 / (Mf * sqrt(c13))
    lMf = log(Mf)
    _, phis, A = _xas_pa_core(_NoAcc(), _XasFlatLazy(sys.coeffs22, sys.amplcoeffs22), f)
    phi22 = phis + sys.timeshift * Mf + sys.phiref22
    s22 = sin(phi22); c22 = cos(phi22)
    h22 = (KernelFused._XP_HAS_MODE_NORMALIZATION * A) * Complex(c22, s22)
    amp0 = sys.amplcoeffs22.Overallamp * KernelFused._XP_HAS_MODE_NORMALIZATION
    return (h22, _xhmf_nonmixing(1.0, Mf, c13, lMf, m76, amp0, ms21), _xhmf_nonmixing(-1.0, Mf, c13, lMf, m76, amp0, ms33),
            _xhmf_32(Mf, c13, lMf, m76, amp0, phi22, sys, m32), _xhmf_nonmixing(1.0, Mf, c13, lMf, m76, amp0, ms44))
end

# ---- generic core (any spec) ----
# Optional per spec: NK extra coordinate directions kappa(x) (`_mbp_nk`, `_mbp_kappa`) along which `_mbp_modes`
# returns hand-written derivatives (not Enzyme tangents: they do not widen the forward pass), and NKR raw per-walker
# constants (`_mbp_nkr`, `_mbp_kraw`) the setup stores with zero tangents. With NK > 0, `_mbp_modes` returns
# (H, Hk) with Hk[i] = dH/dkappa_i, and the sums and the surrogate treat kappa like further tangent directions.
_mbp_nk(spec) = 0
_mbp_nkr(spec) = 0
_mbp_kappa(spec, x, g) = ()
_mbp_kraw(spec, x, g) = ()
_mbp_nv(nm, nt) = 5 + 4nm + 3nt                      # values per detector (layout: see `_mbp_terms`)
const _MBP_FG = 32                                   # default work-group size of the sums kernel
# Work-group size per spec (a power of two <= 32): smaller for specs whose staged setup + accumulators would exceed
# 48 KB of shared memory (IMRPhenomXPHM 223: 7 tangents).
_mbp_fg(spec) = _MBP_FG
# Rows of per-walker data after the staged setup and its time shift (XPHM 223's angle table): written by
# `_mbp_setup_store!` with tangents like the setup, NOT staged in shared memory; `_mbp_modes(spec, KT, w, f, stv, KG, wg)`
# reads them from the global setup array KG (column wg) through a second BatchDuplicated pair.
_mbp_naux(spec) = 0
# Optional per spec: a second setup stage on the backend after (a), on the device KT / JT (XPHM 223: its angle table).
_mbp_post_setup!(spec, backend, KT, JT, nflat, nwalkers, gpu, fx) = nothing
@generated _mbp_flatn(::Val{ST}) where {ST} = _xhm_flatn(ST)
@inline _mbp_take(::Val{N}, t) where {N} = ntuple(i -> t[i], Val(N))

# (a) column w of KT: the flat setup and the time shift (`_mbp_setup_store!`), and of JT[j] their tangents along
# theta_j for j in JS: one Enzyme forward pass of width length(JS) (the other directions Const). The host runs all NT
# at once; the device in groups of at most `_mbp_setup_width(spec)` (`GRAVITATIONALWAVEBAYES_MBP_SETUP_WIDTH`): the stack frame
# grows with the width, and CUDA reserves it for every resident thread (XPHM 223 at width 7: out of memory at launch).
@inline _mbp_unit(::Val{NT}, i) where {NT} = ntuple(j -> ifelse(i == j, 1.0, 0.0), Val(NT))
_mbp_setup_width(spec) = _mbp_nt(spec)
# Setup stages (one pass each; a later stage may read what an earlier one stored, with its tangents): XPHM 223 builds
# its MSA block first, then XHM from it -- one inlined function had a 44 KB frame even at width 1.
_mbp_setup_nstage(spec) = 1
# (`::Val`, not `::Val{1}`: an unstaged spec only sees stage 1, and a staged spec's `::Val{s}` catch-all is then strictly
# more specific -- with `::Val{1}` the two were ambiguous, Aqua)
@inline _mbp_setup_stage!(spec, ::Val, KT, w, args...) = _mbp_setup_store!(spec, KT, w, args...)
@generated function _mbp_setup_fwd!(spec, sv, KT, JT, x::NTuple{N,Float64}, kr, w, fref, lik, gpu, ::Val{JS}) where {N,JS}
    W = length(JS)
    # (a width-1 BatchDuplicated fails in the NVPTX back-end)
    sh = W == 1 ? :(Enzyme.Duplicated(KT, JT[$(JS[1])])) :
         :(Enzyme.BatchDuplicated(KT, $(Expr(:tuple, (:(JT[$j]) for j in JS)...)), false))
    xs = [!(i in JS) ? :(Enzyme.Const(x[$i])) : W == 1 ? :(Enzyme.Duplicated(x[$i], 1.0)) :
          :(Enzyme.BatchDuplicated(x[$i], $(Tuple(Float64(i == j) for j in JS)))) for i in 1:N]
    return :(_xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(_mbp_setup_stage!), Enzyme.Const, Enzyme.Const(spec),
        Enzyme.Const(sv), $sh, Enzyme.Const(w), $(xs...), Enzyme.Const(kr), Enzyme.Const(fref), Enzyme.Const(lik)))
end
# POS[:, w] = (theta, kraw).
@kernel function _mbp_setup_kernel!(KT, JT, @Const(POS), spec, fref, lik, ::Val{NT}, ::Val{NKR}, gpu, sv, js) where {NT,NKR}
    w = @index(Global, Linear)
    x = ntuple(k -> @inbounds(POS[k, w]), Val(NT))
    kr = ntuple(k -> @inbounds(POS[NT + k, w]), Val(NKR))
    _mbp_setup_fwd!(spec, sv, KT, JT, x, kr, w, fref, lik, gpu, js)
end
# Tangent groups 1:NT in chunks of at most mw.
_mbp_groups(nt::Int, mw::Int) = Tuple(Tuple(j0:min(j0 + mw - 1, nt)) for j0 in 1:mw:nt)
# (a), every tangent group in ONE launch: thread (w, g) runs the width-W pass of group g, the direction
# chosen at run time (seeds Float64(i == j), shadows JT[j]; padding j = 0 -> the scratch column block JX). The
# compile-time groups needed one launch of nwalkers threads each (XPHM 223: 6 stages x 4 groups = 24 launches of 4096
# threads, ~1-2 warps per V100 SM, latency-bound). GS: the groups, each padded with 0 to the width W.
@generated function _mbp_setup_fwd_rt!(spec, sv, KT, JW::NTuple{W}, x::NTuple{N,Float64}, kr, w, fref, lik, gpu,
                                       js::NTuple{W,Int}) where {W,N}
    sh = W == 1 ? :(Enzyme.Duplicated(KT, JW[1])) : :(Enzyme.BatchDuplicated(KT, JW, false))
    xs = [W == 1 ? :(Enzyme.Duplicated(x[$i], Float64($i == js[1]))) :
          :(Enzyme.BatchDuplicated(x[$i], $(Expr(:tuple, (:(Float64($i == js[$k])) for k in 1:W)...)))) for i in 1:N]
    return :(_xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(_mbp_setup_stage!), Enzyme.Const, Enzyme.Const(spec),
        Enzyme.Const(sv), $sh, Enzyme.Const(w), $(xs...), Enzyme.Const(kr), Enzyme.Const(fref), Enzyme.Const(lik)))
end
@kernel function _mbp_setup_rt_kernel!(KT, JT, JX, @Const(POS), spec, fref, lik, ::Val{NT}, ::Val{NKR}, gpu, sv,
                                       ::Val{GS}) where {NT,NKR,GS}
    w, g = @index(Global, NTuple)
    x = ntuple(k -> @inbounds(POS[k, w]), Val(NT))
    kr = ntuple(k -> @inbounds(POS[NT + k, w]), Val(NKR))
    js = GS[g]
    JW = ntuple(k -> (j = js[k]; j == 0 ? JX : JT[max(j, 1)]), Val(length(GS[1])))
    _mbp_setup_fwd_rt!(spec, sv, KT, JW, x, kr, w, fref, lik, gpu, js)
end
_mbp_pad_groups(gs, W) = Tuple(ntuple(k -> k <= length(g) ? g[k] : 0, W) for g in gs)
# value-only calls: the setup stages without tangents (JT untouched; the value-only sums do not read it)
@kernel function _mbp_setup_val_kernel!(KT, @Const(POS), spec, fref, lik, ::Val{NT}, ::Val{NKR}, sv) where {NT,NKR}
    w = @index(Global, Linear)
    x = ntuple(k -> @inbounds(POS[k, w]), Val(NT))
    kr = ntuple(k -> @inbounds(POS[NT + k, w]), Val(NKR))
    _mbp_setup_stage!(spec, sv, KT, w, x..., kr, fref, lik)
end
# ... and the post-setup stage's value (XPHM 223: the angle table without tangents)
_mbp_post_setup_value!(spec, backend, KT, nflat, nwalkers, fx) = nothing

# (c) one bin's contribution for one detector (`_mbp_nv(NM, NT)` numbers):
#   1 <h,h>; 2-3 <d,h>; 4-5 f <d,h>; then Q_k (re, im; k = 1:NM); R_k (re, im); d<h,h>/dtheta_j (j = 1:NT);
#   d<d,h>/dtheta_j (re, im).
@generated function _mbp_terms(H::NTuple{NM}, dH::NTuple{NT}, c, tau, f, wt, cd, Ca=true, Cz=true) where {NM,NT}
    hs(k) = Symbol(:hs, k)
    dh(j) = Symbol(:dh, j)
    # Ca (real) weights the |h|^2-type terms, Cz (complex) the <d,h>-type terms (`true` = 1). Calibration factor C:
    # (Ca, Cz) = (|C|^2, C), the calibrated template exactly; calibration marginalization: the curve-averaged per-bin
    # weights (a, b) of `_mbp_device`'s second pass.
    ex = Any[:((s, co) = sincos(-2 * pi * f * tau)), :(S = Complex(co, s)),
             :(h = $(foldl((a, k) -> :($a + c[$k] * H[$k]), 2:NM; init=:(c[1] * H[1]))) * S),
             :(cdz = cd * Cz),                       # cd = conj(d) / PSD already carries the weight
             :(z = cdz * h),
             :(wch = wt * Ca * conj(h))]
    for k in 1:NM
        push!(ex, :($(hs(k)) = H[$k] * S))
    end
    for j in 1:NT
        push!(ex, :($(dh(j)) = $(foldl((a, k) -> :($a + c[$k] * dH[$j][$k]), 2:NM; init=:(c[1] * dH[$j][1]))) * S))
    end
    out = Any[:(wt * Ca * abs2(h)), :(real(z)), :(imag(z)), :(f * real(z)), :(f * imag(z))]
    for k in 1:NM
        push!(out, :(real(cdz * $(hs(k)))), :(imag(cdz * $(hs(k)))))
    end
    for k in 1:NM
        push!(out, :(real(wch * $(hs(k)))), :(imag(wch * $(hs(k)))))
    end
    for j in 1:NT
        push!(out, :(2 * real(wch * $(dh(j)))))
    end
    for j in 1:NT
        push!(out, :(real(cdz * $(dh(j)))), :(imag(cdz * $(dh(j)))))
    end
    Expr(:block, Expr(:meta, :inline), ex..., Expr(:tuple, out...))
end

# Mode factors and tau of detector d for walker w from EXT[(d-1) NE + (1:NE), w], NE = 2 NM + 1 (2 NM numbers c_kd,
# then tau_d + t_shift; t_shift is subtracted here).
@inline _mbp_c(EXT, w, d, ::Val{NM}) where {NM} =
    ntuple(k -> Complex(@inbounds(EXT[(d - 1) * (2NM + 1) + 2k - 1, w]), @inbounds(EXT[(d - 1) * (2NM + 1) + 2k, w])), Val(NM))
# L[(d-1) NV + v, lid] += term v of detector d, unrolled.
# Calibration: with NN > 0 spline nodes per detector, C_d(f_i) = (1 + dA)(2 + i dphi) / (2 - i dphi),
# dA = sum_k WC[i, k] CAL[(d-1) 2NN + k, w], dphi likewise from the next NN rows (bilby's CubicSpline), and the
# per-bin (w |h|^2, Re z, Im z) of the calibrated template go to UZ[(d-1) 3 + 1:3, i, w] for the node gradients.
@inline function _mbp_calfac(WC, CAL, i, w, o, ::Val{NN}) where {NN}
    dA = 0.0; dP = 0.0
    for k in 1:NN
        wk = @inbounds WC[i, k]
        dA += wk * @inbounds(CAL[o + k, w])
        dP += wk * @inbounds(CAL[o + NN + k, w])
    end
    return (1 + dA) * Complex(2.0, dP) / Complex(2.0, -dP)
end
@generated function _mbp_bin_add!(L, lid, H, dH, EXT, w, ts, f, W, CDr, CDi, i, ::Val{D}, ::Val{NM}, ::Val{NT},
                                  CAL=nothing, WC=nothing, UZ=nothing, ::Val{NN}=Val(0), AW=nothing, BWr=nothing,
                                  BWi=nothing, ::Val{MG}=Val(false)) where {D,NM,NT,NN,MG}
    NV = _mbp_nv(NM, NT)
    NE = 2NM + 1
    ex = Any[]
    for d in 1:D
        if MG                     # calibration marginalization, second pass: per-bin weights (a, b), rows (d - 1) nb + i
            push!(ex, :(jj = $(d - 1) * size(CDr, 1) + i),
                      :(Ca = @inbounds AW[jj, w]), :(Cz = Complex(@inbounds(BWr[jj, w]), @inbounds(BWi[jj, w]))))
        elseif NN > 0
            push!(ex, :(Cz = _mbp_calfac(WC, CAL, i, w, $((d - 1) * 2NN), Val($NN))), :(Ca = abs2(Cz)))
        else
            push!(ex, :(Ca = true), :(Cz = true))
        end
        push!(ex, :(t = _mbp_terms(H, dH, _mbp_c(EXT, w, $d, Val($NM)), @inbounds(EXT[$((d - 1) * NE + NE), w]) - ts, f,
                                   @inbounds(W[i, $d]), Complex(@inbounds(CDr[i, $d]), @inbounds(CDi[i, $d])), Ca, Cz)))
        if NN > 0
            push!(ex, :(@inbounds UZ[$(3d - 2), i, w] = t[1]), :(@inbounds UZ[$(3d - 1), i, w] = t[2]),
                      :(@inbounds UZ[$(3d), i, w] = t[3]))
        end
        for v in 1:NV
            push!(ex, :(@inbounds L[$((d - 1) * NV + v), lid] += t[$v]))
        end
    end
    Expr(:block, Expr(:meta, :inline), ex..., :(return nothing))
end

# The NM mode values and their NT tangents at f: one Enzyme forward pass through the flat setup that STORES the modes
# into this thread's scratch column `t` of HS (2 NM rows: re, im per mode), the tangents into dHS[j] (a function
# returning the mode tuple needs its return type inferred for the deferred device call, which fails for the XHM setup).
@generated function _mbp_modes_store!(HS, KS, t, spec, KT, w, f, stv, ::Val{NM}, ::Val{NK}) where {NM,NK}
    st = Any[]
    for k in 1:NM
        push!(st, :(@inbounds HS[$(2k - 1), t] = real(h[$k])), :(@inbounds HS[$(2k), t] = imag(h[$k])))
    end
    for i in 1:NK, k in 1:NM                         # kappa derivatives: primal values only (KS is Const)
        push!(st, :(@inbounds KS[$(2NM * (i - 1) + 2k - 1), t] = real(hk[$i][$k])),
                  :(@inbounds KS[$(2NM * (i - 1) + 2k), t] = imag(hk[$i][$k])))
    end
    call = NK == 0 ? :(h = _mbp_modes(spec, KT, w, f, stv)) : :((h, hk) = _mbp_modes(spec, KT, w, f, stv))
    Expr(:block, Expr(:meta, :inline), call, st..., :(return nothing))
end
# (with aux rows: `_mbp_modes(spec, KT, w, f, stv, KG, wg)`)
@generated function _mbp_modes_store_aux!(HS, KS, t, spec, KT, w, f, stv, ::Val{NM}, ::Val{NK}, KG, wg) where {NM,NK}
    st = Any[]
    for k in 1:NM
        push!(st, :(@inbounds HS[$(2k - 1), t] = real(h[$k])), :(@inbounds HS[$(2k), t] = imag(h[$k])))
    end
    for i in 1:NK, k in 1:NM
        push!(st, :(@inbounds KS[$(2NM * (i - 1) + 2k - 1), t] = real(hk[$i][$k])),
                  :(@inbounds KS[$(2NM * (i - 1) + 2k), t] = imag(hk[$i][$k])))
    end
    call = NK == 0 ? :(h = _mbp_modes(spec, KT, w, f, stv, KG, wg)) : :((h, hk) = _mbp_modes(spec, KT, w, f, stv, KG, wg))
    Expr(:block, Expr(:meta, :inline), call, st..., :(return nothing))
end
@inline _mbp_ld(A, t, ::Val{NM}) where {NM} = ntuple(k -> Complex(@inbounds(A[2k - 1, t]), @inbounds(A[2k, t])), Val(NM))
@inline function _mbp_modes_tan(HS, dHS, KS, t, spec, KT, JT, w, f, stv, nmv::Val{NM}, nkv::Val{NK}, gpu,
                                KG = nothing, JG = nothing, wg = 0) where {NM,NK}
    if _mbp_naux(spec) == 0
        _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(_mbp_modes_store!), Enzyme.Const,
                  Enzyme.BatchDuplicated(HS, dHS, false), Enzyme.Const(KS), Enzyme.Const(t), Enzyme.Const(spec),
                  Enzyme.BatchDuplicated(KT, JT, false), Enzyme.Const(w), Enzyme.Const(f), Enzyme.Const(stv),
                  Enzyme.Const(nmv), Enzyme.Const(nkv))
    else
        _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(_mbp_modes_store_aux!), Enzyme.Const,
                  Enzyme.BatchDuplicated(HS, dHS, false), Enzyme.Const(KS), Enzyme.Const(t), Enzyme.Const(spec),
                  Enzyme.BatchDuplicated(KT, JT, false), Enzyme.Const(w), Enzyme.Const(f), Enzyme.Const(stv),
                  Enzyme.Const(nmv), Enzyme.Const(nkv), Enzyme.BatchDuplicated(KG, JG, false), Enzyme.Const(wg))
    end
    hk = ntuple(i -> ntuple(k -> Complex(@inbounds(KS[2NM * (i - 1) + 2k - 1, t]), @inbounds(KS[2NM * (i - 1) + 2k, t])),
                            Val(NM)), Val(NK))
    return _mbp_ld(HS, t, nmv), (map(A -> _mbp_ld(A, t, nmv), dHS)..., hk...)
end
@generated function _mbp_tail!(SUMS, w, ::Val{NL}, K0, KJ, ::Val{NF}, ::Val{NT}) where {NL,NF,NT}
    ex = Any[:(@inbounds SUMS[$(NL + 1), w] = K0[$NF, 1])]
    for j in 1:NT
        push!(ex, :(@inbounds SUMS[$(NL + 1 + j), w] = KJ[$j][$NF, 1]))
    end
    Expr(:block, Expr(:meta, :inline), ex..., :(return nothing))
end

# Work-group = walker. SUMS[(d-1) NV + v, w] (NL = NV D rows), then SUMS[NL + 1, w] = t_shift and SUMS[NL + 1 + j, w]
# its tangents. The walker's flat setup and its NT tangent columns (NF = nflat + 1 numbers each) are first staged in
# shared memory: read from global memory inside the per-bin forward pass, every bin re-loaded all of them (XHM:
# 156 of 171 ms per call at 1024 walkers on the V100). The setup and each of its NT tangent columns get their own
# shared array (views of one parent as BatchDuplicated shadows give wrong tangents -- the in-kernel Enzyme pitfall);
# columns beyond NT are 1 x 1 dummies, so shared memory is NF (NT + 1) (NT up to 7, FG per spec).
@inline _mbp_nf(NF, NT, j) = NT >= j ? NF : 1
@kernel function _mbp_sums_kernel!(SUMS, KT, JT, HS, dHS, KS, @Const(EXT), @Const(F1), @Const(W), @Const(CDr),
                                   @Const(CDi), spec, stv, ::Val{NF}, ::Val{D}, ::Val{NM}, ::Val{NT}, ::Val{NK}, ::Val{NL},
                                   ::Val{FG}, gpu, @Const(CAL), @Const(WC), UZ, ::Val{NN}, @Const(AW), @Const(BWr),
                                   @Const(BWi), ::Val{MG}) where {NF,D,NM,NT,NK,NL,FG,NN,MG}
    w = @index(Group, Linear)
    lid = @index(Local, Linear)
    t = @index(Global, Linear)
    L = @localmem Float64 (NL, FG)
    K0 = @localmem Float64 (NF, 1)
    K1 = @localmem Float64 (_mbp_nf(NF, NT, 1), 1)
    K2 = @localmem Float64 (_mbp_nf(NF, NT, 2), 1)
    K3 = @localmem Float64 (_mbp_nf(NF, NT, 3), 1)
    K4 = @localmem Float64 (_mbp_nf(NF, NT, 4), 1)
    K5 = @localmem Float64 (_mbp_nf(NF, NT, 5), 1)
    K6 = @localmem Float64 (_mbp_nf(NF, NT, 6), 1)
    K7 = @localmem Float64 (_mbp_nf(NF, NT, 7), 1)
    j = lid
    while j <= NF
        @inbounds K0[j, 1] = KT[j, w]
        _mbp_stage_tan!(_mbp_take(Val(NT), (K1, K2, K3, K4, K5, K6, K7)), JT, j, w, Val(NT))
        j += FG
    end
    @synchronize
    # (rebuilt after every @synchronize: KA's CPU lowering splits the kernel there, only @localmem arrays carry over)
    KJ = _mbp_take(Val(NT), (K1, K2, K3, K4, K5, K6, K7))
    nb = length(F1)
    ts = @inbounds K0[NF, 1]
    for v in 1:NL                                    # accumulate in this thread's shared column, not in registers
        @inbounds L[v, lid] = 0.0
    end
    i = lid
    while i <= nb
        f = @inbounds F1[i]
        H, dH = _mbp_modes_tan(HS, dHS, KS, t, spec, K0, KJ, 1, f, stv, Val(NM), Val(NK), gpu, KT, JT, w)
        _mbp_bin_add!(L, lid, H, dH, EXT, w, ts, f, W, CDr, CDi, i, Val(D), Val(NM), Val(NT + NK), CAL, WC, UZ, Val(NN),
                      AW, BWr, BWi, Val(MG))
        i += FG
    end
    @synchronize
    _mbp_red_step!(L, lid, 16, NL, FG)                # steps s >= FG are no-ops (FG <= 32, a power of two)
    @synchronize
    _mbp_red_step!(L, lid, 8, NL, FG)
    @synchronize
    _mbp_red_step!(L, lid, 4, NL, FG)
    @synchronize
    _mbp_red_step!(L, lid, 2, NL, FG)
    @synchronize
    _mbp_red_step!(L, lid, 1, NL, FG)
    @synchronize
    v = lid
    while v <= NL
        @inbounds SUMS[v, w] = L[v, 1]
        v += FG
    end
    if lid == 1                                      # (K0, not `ts`: KA's CPU lowering splits at @synchronize)
        _mbp_tail!(SUMS, w, Val(NL), K0, _mbp_take(Val(NT), (K1, K2, K3, K4, K5, K6, K7)), Val(NF), Val(NT))
    end
end
@generated function _mbp_stage_tan!(KJ, JT, j, w, ::Val{NT}) where {NT}
    ex = [:(@inbounds KJ[$k][j, 1] = JT[$k][j, w]) for k in 1:NT]
    Expr(:block, Expr(:meta, :inline), ex..., :(return nothing))
end
@inline function _mbp_red_step!(L, lid, s, nv, fg)
    if lid <= s && s < fg
        for v in 1:nv
            @inbounds L[v, lid] += L[v, lid + s]
        end
    end
    return nothing
end

# ---- Value-only sums (`value_only` likelihood calls: RW stages, reweighting, reconstruction) ----
# Per detector only the three sums the log-likelihood needs, SUMS[(d-1) NV + (1:3), w] = (sum w Ca |h|^2, Re, Im of
# sum cd Cz h) as `_mbp_terms`' first three (other rows untouched), from the primal modes: no Enzyme forward pass, no
# tangent columns. Sampled calibration (NN > 0) enters as in `_mbp_bin_add!`. Work-group = walker, setup in shared memory.
@kernel function _mbp_val_kernel!(SUMS, KT, @Const(EXT), @Const(F1), @Const(W), @Const(CDr), @Const(CDi), spec, stv,
                                  ::Val{NF}, ::Val{D}, ::Val{NM}, ::Val{NK}, ::Val{NV}, ::Val{FG}, @Const(CAL), @Const(WC),
                                  ::Val{NN}) where {NF,D,NM,NK,NV,FG,NN}
    w = @index(Group, Linear)
    lid = @index(Local, Linear)
    L = @localmem Float64 (3D, FG)
    K0 = @localmem Float64 (NF, 1)
    j = lid
    while j <= NF
        @inbounds K0[j, 1] = KT[j, w]
        j += FG
    end
    for v in 1:3D
        @inbounds L[v, lid] = 0.0
    end
    @synchronize
    nb = length(F1)
    ts = @inbounds K0[NF, 1]
    i = lid
    while i <= nb
        f = @inbounds F1[i]
        r = _mbp_naux(spec) == 0 ? _mbp_modes(spec, K0, 1, f, stv) : _mbp_modes(spec, K0, 1, f, stv, KT, w)
        H = NK == 0 ? r : r[1]
        _mbp_val_bin!(L, lid, H, EXT, w, ts, f, W, CDr, CDi, i, Val(D), Val(NM), CAL, WC, Val(NN))
        i += FG
    end
    @synchronize
    _mbp_red_step!(L, lid, 16, 3D, FG)
    @synchronize
    _mbp_red_step!(L, lid, 8, 3D, FG)
    @synchronize
    _mbp_red_step!(L, lid, 4, 3D, FG)
    @synchronize
    _mbp_red_step!(L, lid, 2, 3D, FG)
    @synchronize
    _mbp_red_step!(L, lid, 1, 3D, FG)
    @synchronize
    if lid == 1
        for d in 1:D, v in 1:3
            @inbounds SUMS[(d - 1) * NV + v, w] = L[3 * (d - 1) + v, 1]
        end
    end
end
@generated function _mbp_val_bin!(L, lid, H, EXT, w, ts, f, W, CDr, CDi, i, ::Val{D}, ::Val{NM}, CAL, WC,
                                  ::Val{NN}) where {D,NM,NN}
    NE = 2NM + 1
    ex = Any[]
    for d in 1:D
        if NN > 0
            push!(ex, :(Cz = _mbp_calfac(WC, CAL, i, w, $((d - 1) * 2NN), Val($NN))), :(Ca = abs2(Cz)))
        else
            push!(ex, :(Ca = true), :(Cz = true))
        end
        push!(ex, :(c = _mbp_c(EXT, w, $d, Val($NM))),
                  :((s, co) = sincos(-2 * pi * f * (@inbounds(EXT[$((d - 1) * NE + NE), w]) - ts))),
                  :(h = $(foldl((a, k) -> :($a + c[$k] * H[$k]), 2:NM; init=:(c[1] * H[1]))) * Complex(co, s)),
                  :(z = Complex(@inbounds(CDr[i, $d]), @inbounds(CDi[i, $d])) * Cz * h),
                  :(@inbounds L[$(3d - 2), lid] += W[i, $d] * Ca * abs2(h)),
                  :(@inbounds L[$(3d - 1), lid] += real(z)),
                  :(@inbounds L[$(3d), lid] += imag(z)))
    end
    Expr(:block, Expr(:meta, :inline), ex..., :(return nothing))
end

# ---- Calibration marginalization (as the SEOBNRv5PHM device path, `KernelFused._PHMCalMarg`) ----
# First pass (value only, no tangents): V[(d-1) nb + i, w] = w_i |h_d(f_i)|^2, U = conj(d) h / PSD (uncalibrated).
# Work-group = walker; the flat setup staged in shared memory as in the sums kernel.
@kernel function _mbp_vu_kernel!(V, U, KT, @Const(EXT), @Const(F1), @Const(W), @Const(CDr), @Const(CDi), spec, stv,
                                 ::Val{NF}, ::Val{D}, ::Val{NM}, ::Val{NK}, ::Val{FG}) where {NF,D,NM,NK,FG}
    w = @index(Group, Linear)
    lid = @index(Local, Linear)
    K0 = @localmem Float64 (NF, 1)
    j = lid
    while j <= NF
        @inbounds K0[j, 1] = KT[j, w]
        j += FG
    end
    @synchronize
    nb = length(F1)
    ts = @inbounds K0[NF, 1]
    i = lid
    while i <= nb
        f = @inbounds F1[i]
        r = _mbp_naux(spec) == 0 ? _mbp_modes(spec, K0, 1, f, stv) : _mbp_modes(spec, K0, 1, f, stv, KT, w)
        H = NK == 0 ? r : r[1]
        _mbp_vu_bin!(V, U, H, EXT, w, ts, f, W, CDr, CDi, i, nb, Val(D), Val(NM))
        i += FG
    end
end
@generated function _mbp_vu_bin!(V, U, H, EXT, w, ts, f, W, CDr, CDi, i, nb, ::Val{D}, ::Val{NM}) where {D,NM}
    NE = 2NM + 1
    ex = Any[]
    for d in 1:D
        push!(ex, :(c = _mbp_c(EXT, w, $d, Val($NM))),
                  :((s, co) = sincos(-2 * pi * f * (@inbounds(EXT[$((d - 1) * NE + NE), w]) - ts))),
                  :(h = $(foldl((a, k) -> :($a + c[$k] * H[$k]), 2:NM; init=:(c[1] * H[1]))) * Complex(co, s)),
                  :(@inbounds V[$(d - 1) * nb + i, w] = W[i, $d] * abs2(h)),
                  :(@inbounds U[$(d - 1) * nb + i, w] = Complex(CDr[i, $d], CDi[i, $d]) * h))
    end
    Expr(:block, Expr(:meta, :inline), ex..., :(return nothing))
end
# Per walker: logL = log-mean-exp over the curves of the distance-marginalized l_k (ll_plain_k = dd - hh_k/2 + zr_k,
# hh_k = 4 df HH[k, w], z_k = 4 df ZZ[k, w]); G1 = -2 df p_k E_k[u^2], G2 = 4 df p_k E_k[u] (the second pass's weights
# a = A' G1, b = B' G2), GDL = sum p_k E_k[(zr - hh u) u] / dL. One thread per walker.
@kernel function _mbp_cm_kernel!(LL, G1, G2, GDL, LK, EU, EU2, ED, @Const(HH), @Const(ZZ), @Const(DL), dd, df, nc::Int,
                                 lik_arr)
    w = @index(Global, Linear)
    @inbounds begin
        T = Float64
        lik = KernelFused._lik_from_array(lik_arr)
        dL = DL[w]
        m = T(-Inf)
        for k in 1:nc
            hh = 4 * df * HH[k, w]; zr = 4 * df * real(ZZ[k, w])
            lp = dd - hh / 2 + zr
            v, eu, eu2, ed = lik.margdist ? KernelFused._distance_marg_moments(lp, hh, zr, dL, lik) : (lp, one(T), one(T), zero(T))
            LK[k, w] = v; EU[k, w] = eu; EU2[k, w] = eu2; ED[k, w] = ed
            m = max(m, v)
        end
        s = zero(T)
        for k in 1:nc
            s += exp(LK[k, w] - m)
        end
        lse = m + log(s)
        ok = isfinite(lse)
        LL[w] = ifelse(ok, lse - log(T(nc)), T(-Inf))
        gdl = zero(T)
        for k in 1:nc
            p = ifelse(ok, exp(LK[k, w] - lse), zero(T))
            G1[k, w] = -2 * df * p * EU2[k, w]
            G2[k, w] = Complex(4 * df * p * EU[k, w], zero(T))
            gdl += ifelse(ok, p * ED[k, w], zero(T))
        end
        GDL[w] = gdl
    end
end

# Calibration node sums: thread (k, d, w) -> GC[5 (k - 1 + NN (d - 1)) + 1:5, w] =
#   sum_i WC[i, k] (2 u_i / (1 + dA_i), z_i / (1 + dA_i), z_i 4i / (4 + dphi_i^2))   (u, z of the calibrated template),
# i.e. d<h,h>/da_k (d<h,h>/dphi_k = 0), d<d,h>/da_k and d<d,h>/dphi_k for the unnormalized bin sums.
@kernel function _mbp_cal_kernel!(GC, @Const(UZ), @Const(CAL), @Const(WC), ::Val{NN}, ::Val{D}) where {NN,D}
    t = @index(Global, Linear)
    k = (t - 1) % NN + 1
    d = ((t - 1) ÷ NN) % D + 1
    w = (t - 1) ÷ (NN * D) + 1
    o = (d - 1) * 2NN
    g = 0.0; zar = 0.0; zai = 0.0; zpr = 0.0; zpi = 0.0
    for i in 1:size(WC, 1)
        dA = 0.0; dP = 0.0
        for kk in 1:NN
            wk = @inbounds WC[i, kk]
            dA += wk * @inbounds(CAL[o + kk, w])
            dP += wk * @inbounds(CAL[o + NN + kk, w])
        end
        wik = @inbounds WC[i, k]
        ia = wik / (1 + dA)
        u = @inbounds UZ[3d - 2, i, w]
        zr = @inbounds UZ[3d - 1, i, w]; zi = @inbounds UZ[3d, i, w]
        g += 2 * u * ia
        zar += zr * ia; zai += zi * ia
        fp = 4 * wik / (4 + dP * dP)                 # z * 4i fp / wik  ->  (-4 zi, 4 zr) fp / 4
        zpr -= zi * fp; zpi += zr * fp
    end
    q = 5 * (k - 1 + NN * (d - 1))
    @inbounds GC[q + 1, w] = g; @inbounds GC[q + 2, w] = zar; @inbounds GC[q + 3, w] = zai
    @inbounds GC[q + 4, w] = zpr; @inbounds GC[q + 5, w] = zpi
end

# (d) the first-order surrogate (see the header), split as in the XAS path: `_mbp_pre` = the `_marg_combine` inputs
# (ll_plain, <h,h>, Re z, Im z) of the first-order sums around x0 (th0 = theta(x0) and E0 = `_mbp_ext` at x0, both
# already computed in stages (a) and (b)), `_marg_combine_grad` (hand-written) their outer derivatives, and Enzyme
# reverse only over the seeded scalar s . pre(x) (Enzyme through the distance-marginalization quadrature was ~200 us
# per walker). Value and gradient at x = x0 are the likelihood's.
@inline _cplx(s, o) = Complex(s[o], s[o + 1])
# Coordinates of the sums' directions: theta (NT, with t_shift tangents) then kappa (NK).
@inline _mbp_coords(spec, x, g) = (_mbp_theta(spec, x)..., _mbp_kappa(spec, x, g)...)
# The coordinates and `_mbp_ext` together (a spec whose kappa and extrinsic factors share work overrides it: XPHM's
# precession setup).
@inline _mbp_coords_ext(spec, x, g, dv) = (_mbp_coords(spec, x, g), _mbp_ext(spec, x, g, dv))
# The first-order part of the surrogate (linear in its arguments): the changes of `_mbp_pre0`'s (ll_plain, <h,h>,
# Re z, Im z) for coordinate changes dth (NT + NK) and per detector mode-factor / time changes dE[d] = (dc, dtau).
function _mbp_pre_dir(spec, dth, dE, S, g, ::Val{D}) where {D}
    NM = _mbp_nm(spec); NT = _mbp_nt(spec); NTT = NT + _mbp_nk(spec)
    NV = _mbp_nv(NM, NTT)
    T = typeof(real(dE[1][2]))
    dts = zero(T)
    for j in 1:NT
        dts += S[NV * D + 1 + j] * dth[j]
    end
    ll_plain = zero(T); hh = zero(T); zr = zero(T); zi = zero(T)
    for d in 1:D
        o = (d - 1) * NV
        dc, dtau = dE[d]
        z = -2 * pi * im * (dtau - dts) * _cplx(S, o + 4)
        h2 = zero(T)
        for k in 1:NM
            z += _cplx(S, o + 4 + 2k) * dc[k]
            h2 += 2 * real(_cplx(S, o + 4 + 2NM + 2k) * dc[k])
        end
        for j in 1:NTT
            z += _cplx(S, o + 4 + 4NM + NTT + 2j) * dth[j]
            h2 += S[o + 5 + 4NM + j] * dth[j]
        end
        df = g.df[d]
        ll_plain += -2 * df * h2 + 4 * df * real(z)
        hh += 4 * df * h2
        zr += 4 * df * real(z)
        zi += 4 * df * imag(z)
    end
    return ll_plain, hh, zr, zi
end
function _mbp_pre(spec, x, th0, E0, S, g, dv::Val{D}) where {D}
    NM = _mbp_nm(spec); NV = _mbp_nv(NM, _mbp_nt(spec) + _mbp_nk(spec))
    th, E = _mbp_coords_ext(spec, x, g, dv)
    dth = ntuple(j -> th[j] - th0[j], Val(length(th0)))
    dE = ntuple(d -> (ntuple(k -> E[d][1][k] - E0[d][1][k], Val(NM)), E[d][2] - E0[d][2]), Val(D))
    a, b, c, e = _mbp_pre0(S, g, dv, Val(NV))
    da, db, dc, de = _mbp_pre_dir(spec, dth, dE, S, g, dv)
    return a + da, b + db, c + dc, e + de
end
function _mbp_pre_seeded(spec, x, th0, E0, S, g, dv, sd)
    a, b, c, d = _mbp_pre(spec, x, th0, E0, S, g, dv)
    return sd[1] * a + sd[2] * b + sd[3] * c + sd[4] * d
end
# `_mbp_pre` at x = x0 itself: the sums as they are.
function _mbp_pre0(S, g, ::Val{D}, ::Val{NV}) where {D,NV}
    ll_plain = 0.0; hh = 0.0; zr = 0.0; zi = 0.0
    for d in 1:D
        o = (d - 1) * NV
        df = g.df[d]
        ll_plain += g.dd[d] - 2 * df * S[o + 1] + 4 * df * S[o + 2]
        hh += 4 * df * S[o + 1]
        zr += 4 * df * S[o + 2]
        zi += 4 * df * S[o + 3]
    end
    return ll_plain, hh, zr, zi
end
# (log-likelihood, gradient) of one walker from its sums.
function _mbp_post(spec, x::NTuple{NP,Float64}, th0, E0, S, g, lik, dv) where {NP}
    a, b, c, d = _mbp_pre0(S, g, dv, Val(_mbp_nv(_mbp_nm(spec), _mbp_nt(spec) + _mbp_nk(spec))))
    v, gl, gh, gzr, gzi, gdL = _marg_combine_grad(a, b, c, d, x[5], lik)
    gr = Enzyme.autodiff(Enzyme.Reverse, _mbp_pre_seeded, Enzyme.Active, Enzyme.Const(spec), Enzyme.Active(x),
                         Enzyme.Const(th0), Enzyme.Const(E0), Enzyme.Const(S), Enzyme.Const(g), Enzyme.Const(dv),
                         Enzyme.Const((gl, gh, gzr, gzi)))[1][2]
    return v, ntuple(k -> gr[k] + ifelse(k == 5, gdL, 0.0), Val(NP)), (gl, gh, gzr, gzi)
end
# Calibration marginalization: the sums S were taken with the curve-averaged weights (a, b) (`_mbp_device`'s
# `margll!`), so d logL / dx = d/dx [h2~ + Re z~] = the surrogate seeded with (0, 1/(4 df), 1/(4 df), 0), plus the
# explicit d_L term of the distance marginalization.
function _mbp_post_cm(spec, x::NTuple{NP,Float64}, th0, E0, S, g, dv, gdl) where {NP}
    sd = (0.0, 1 / (4 * g.df[1]), 1 / (4 * g.df[1]), 0.0)
    gr = Enzyme.autodiff(Enzyme.Reverse, _mbp_pre_seeded, Enzyme.Active, Enzyme.Const(spec), Enzyme.Active(x),
                         Enzyme.Const(th0), Enzyme.Const(E0), Enzyme.Const(S), Enzyme.Const(g), Enzyme.Const(dv),
                         Enzyme.Const(sd))[1][2]
    return ntuple(k -> gr[k] + ifelse(k == 5, gdl, 0.0), Val(NP))
end

# E = `_mbp_ext` of walker w back from the host EXT array (the same numbers).
@inline _mbp_ext_load(exth, w, ::Val{D}, ::Val{NM}) where {D,NM} =
    ntuple(d -> (ntuple(k -> Complex(exth[(d - 1) * (2NM + 1) + 2k - 1, w], exth[(d - 1) * (2NM + 1) + 2k, w]), Val(NM)),
                 exth[(d - 1) * (2NM + 1) + 2NM + 1, w]), Val(D))

# ---- the per-walker stages on the backend (`setup = :device`) ----
# F[:, w]: the EXT layout of (b) (D (2 NM + 1) rows: the sums read EXT = F), then the NT + NK coordinates; JF[k] its
# derivatives along sampled row k. Stores unrolled (`_xasbp_store_col!`: a runtime-indexed store loop writes primal
# constants into Enzyme's forward shadows).
struct _MbpCol{A}                                    # column w of a matrix, indexed like a vector
    a::A
    w::Int
end
@inline Base.getindex(c::_MbpCol, i::Int) = @inbounds c.a[i, c.w]
@generated function _mbp_F_store!(F, w, x, spec, g, ::Val{D}, ::Val{NM}) where {D,NM}
    NE = 2NM + 1
    ex = Any[:((cv, E) = _mbp_coords_ext(spec, x, g, Val($D)))]
    for d in 1:D
        o = (d - 1) * NE
        for k in 1:NM
            push!(ex, :(@inbounds F[$(o + 2k - 1), w] = real(E[$d][1][$k])), :(@inbounds F[$(o + 2k), w] = imag(E[$d][1][$k])))
        end
        push!(ex, :(@inbounds F[$(o + NE), w] = E[$d][2]))
    end
    push!(ex, :(_xasbp_store_col!(F, w, cv, $(D * NE))))
    Expr(:block, Expr(:meta, :inline), ex..., :(return nothing))
end
# Primal pass: POS[:, w] = (theta, kraw) for (a), F[:, w].
@kernel function _mbp_prep_kernel!(POS, F, @Const(X), spec, g, dvv, nmv, ::Val{NP}) where {NP}
    w = @index(Global, Linear)
    x = ntuple(k -> @inbounds(X[k, w]), Val(NP))
    _xasbp_store_col!(POS, w, _mbp_theta(spec, x), 0)   # (two stores: splatting the 51-tuple went dynamic on the device)
    _xasbp_store_col!(POS, w, _mbp_kraw(spec, x, g), _mbp_nt(spec))
    _mbp_F_store!(F, w, x, spec, g, dvv, nmv)
end
# JF[k] for the sampled rows k in JS (F rewritten with the same values): one Enzyme forward pass of width length(JS).
@kernel function _mbp_jac_kernel!(F, JF, @Const(X), spec, g, dvv, nmv, ::Val{NP}, ::Val{JS}, gpu) where {NP,JS}
    w = @index(Global, Linear)
    x = ntuple(k -> @inbounds(X[k, w]), Val(NP))
    if length(JS) == 1                                # (a width-1 BatchDuplicated fails in the NVPTX back-end)
        _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(_mbp_F_store!), Enzyme.Const, Enzyme.Duplicated(F, JF[JS[1]]),
                  Enzyme.Const(w), Enzyme.Duplicated(x, _mbp_unit(Val(NP), JS[1])), Enzyme.Const(spec), Enzyme.Const(g),
                  Enzyme.Const(dvv), Enzyme.Const(nmv))
    else
        _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(_mbp_F_store!), Enzyme.Const,
                  Enzyme.BatchDuplicated(F, map(k -> JF[k], JS), false), Enzyme.Const(w),
                  Enzyme.BatchDuplicated(x, map(k -> _mbp_unit(Val(NP), k), JS)), Enzyme.Const(spec), Enzyme.Const(g),
                  Enzyme.Const(dvv), Enzyme.Const(nmv))
    end
end
# (dth, dE) of `_mbp_pre_dir` from an F-layout column J (F's values, or its derivative along one sampled row).
@inline function _mbp_F_dir(J, ::Val{D}, ::Val{NM}, ::Val{NTT}) where {D,NM,NTT}
    NE = 2NM + 1
    dE = ntuple(d -> (ntuple(k -> Complex(J[(d - 1) * NE + 2k - 1], J[(d - 1) * NE + 2k]), Val(NM)), J[(d - 1) * NE + NE]),
                Val(D))
    return ntuple(j -> J[D * NE + j], Val(NTT)), dE
end
# (d) per walker: LL[w] (log-likelihood; MG: calibration-marginalized, from `margll!`'s LLM / GDLM) and G[:, w] (its
# gradient in the NP sampled rows, then the NN-node calibration rows from the node sums GC; `VO`: value only, G
# untouched).
@kernel function _mbp_post_kernel!(LL, G, @Const(SUMS), @Const(X), JF, @Const(LLM), @Const(GDLM), @Const(GC), @Const(CS),
                                   spec, g, lik, dvv::Val{D}, nmv, nttv, ::Val{NP}, ::Val{NN}, ::Val{MG},
                                   ::Val{VO}) where {D,NP,NN,MG,VO}
    w = @index(Global, Linear)
    S = _MbpCol(SUMS, w)
    a, b, c, e = _mbp_pre0(S, g, dvv, Val(_mbp_nv(_mbp_nm(spec), _mbp_nt(spec) + _mbp_nk(spec))))
    dL = @inbounds X[5, w]
    if VO
        @inbounds LL[w] = MG ? LLM[w] : KernelFused._marg_combine(a, b, c, e, dL, lik)
    else
        v, gl, gh, gzr, gzi, gdL = MG ? (@inbounds(LLM[w]), 0.0, 1 / (4 * g.df[1]), 1 / (4 * g.df[1]), 0.0, @inbounds(GDLM[w])) :
                                        _marg_combine_grad(a, b, c, e, dL, lik)
        @inbounds LL[w] = v
        gr = ntuple(Val(NP)) do k
            da, db, dc, de = _mbp_pre_dir(spec, _mbp_F_dir(_MbpCol(JF[k], w), dvv, nmv, nttv)..., S, g, dvv)
            gl * da + gh * db + gzr * dc + gzi * de + ifelse(k == 5, gdL, 0.0)
        end
        _xasbp_store_col!(G, w, gr, 0)
        # calibration nodes (as the host path): ll_plain = dd - 2 df h2 + 4 df Re z, hh = 4 df h2, zr, zi = 4 df z
        for d in 1:D, k in 1:NN
            q = 5 * (k - 1 + NN * (d - 1)); df = g.df[d]
            @inbounds begin
                dh2 = GC[q + 1, w]; dzr = GC[q + 2, w]; dzi = GC[q + 3, w]
                pzr = GC[q + 4, w]; pzi = GC[q + 5, w]
                ga = gl * (-2df * dh2 + 4df * dzr) + gh * 4df * dh2 + gzr * 4df * dzr + gzi * 4df * dzi
                gp = gl * 4df * pzr + gzr * 4df * pzr + gzi * 4df * pzi
                ra = (d - 1) * 2NN + k
                G[NP + ra, w] = CS[ra] * ga
                G[NP + ra + NN, w] = CS[ra + NN] * gp
            end
        end
    end
end

# Device side of stages (a) and (c) for the flat-setup specs (XHM, XP, XPHM): `setup!(posh)` fills the setup and its
# tangents from the theta (+ kraw) rows `posh`; `sums!(sumsh, EXTd)` runs the bin sums into the host array `sumsh`
# (`_mbp_nv` rows per detector, then t_shift and its NT tangents); `dead` (nothing, or a host Bool vector `sums!`
# fills) marks walkers whose log-likelihood is the dead-zone constant with zero gradient. A spec whose setup is not a
# per-walker flat record (SEOBNRv5HM: multi-stage device tables) has its own `_mbp_device` method.
# ---- (c) as batched GEMMs (`bin_sums = :gemm`; the SEOBNRv5HM path, `SEOBNRHMBinParallel.jl`) ----
# With |S| = 1 every |h|^2-type row of `_mbp_terms` is an entry of the walker's weighted Gram matrix of the modes and
# their tangent / kappa columns, every <d,h>-type row a dot with v = cd Cz S. Left matrix per detector (bins x (NM + 2)):
# [wt Ca H_1..NM, conj(v), f conj(v)]; right: [H | dH_1..NT | dH/dkappa_1..NK] at X[bin, walker, mode, block], each
# block its own dense array (`GPU.block_alias`: separate same-typed Enzyme shadows); one strided-batched ZGEMM over the
# walkers of a chunk (`GPU.batched_adj_mul_mid!`), then `_mbp_gemm_combine!` writes SUMS as `_mbp_sums_kernel!`.
# Calibration marginalization's second pass (Ca, Cz per walker and bin) folds into L; sampled calibration (per-bin UZ
# rows) stays on the kernel.
@generated function _mbp_gemm_store!(XH, XK, i, w, w0, nw, spec, KT, F1, stv, ::Val{NM}, ::Val{NK}) where {NM,NK}
    st = Any[]
    for k in 1:NM
        push!(st, :(@inbounds XH[i, w, $k] = H[$k]))
    end
    for q in 1:NK, k in 1:NM                         # kappa derivatives: primal values (XK Const)
        push!(st, :(@inbounds XK[$q][i, w, $k] = hk[$q][$k]))
    end
    zs = Any[:(@inbounds XH[i, w, $k] = zero(ComplexF64)) for k in 1:NM]
    for q in 1:NK, k in 1:NM
        push!(zs, :(@inbounds XK[$q][i, w, $k] = zero(ComplexF64)))
    end
    call = :(r = _mbp_naux(spec) == 0 ? _mbp_modes(spec, KT, wg, f, stv) : _mbp_modes(spec, KT, wg, f, stv, KT, wg))
    split = NK == 0 ? :(H = r) : :((H, hk) = r)
    quote
        $(Expr(:meta, :inline))
        wg = w0 + w
        if wg > nw
            $(zs...)
            return nothing
        end
        f = @inbounds F1[i]
        $call
        $split
        $(st...)
        return nothing
    end
end
@kernel function _mbp_gemm_modes_kernel!(XH, XHT, XK, KT, JT, @Const(F1), w0, nw, spec, stv, nmv, nkv, gpu)
    i, w = @index(Global, NTuple)
    if XHT === nothing
        _mbp_gemm_store!(XH, XK, i, w, w0, nw, spec, KT, F1, stv, nmv, nkv)
    else
        _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(_mbp_gemm_store!), Enzyme.Const, Enzyme.BatchDuplicated(XH, XHT, false),
                  Enzyme.Const(XK), Enzyme.Const(i), Enzyme.Const(w), Enzyme.Const(w0), Enzyme.Const(nw), Enzyme.Const(spec),
                  Enzyme.BatchDuplicated(KT, JT, false), Enzyme.Const(F1), Enzyme.Const(stv), Enzyme.Const(nmv),
                  Enzyme.Const(nkv))
    end
end
# The GEMM blocks XH (and with `grad` XHT, XK) of walkers w0 + 1 : w0 + wc. A spec may split this stage
# (`_mbp_gemm_nmb(spec)` > 0: buffers XM, XMT of that many intermediate modes per bin, see `XPHMBinParallel.jl`).
_mbp_gemm_nmb(spec) = 0
_mbp_gemm_modes!(spec, backend, gb, KT, JT, F1, w0, nw, stv, nmv, nkv, grad, gpu) =
    _mbp_gemm_modes_kernel!(backend)(gb.XH, grad ? gb.XHT : nothing, gb.XK, KT, JT, F1, w0, nw, spec, stv, nmv, nkv, gpu;
                                     ndrange=(length(F1), gb.wc))
# L[i, (d - 1)(NM + 2) + (1:NM + 2), w]; MG: calibration-marginalization weights (rows (d - 1) nb + i of AW, BW).
@kernel function _mbp_gemm_lhs_kernel!(L, XH, @Const(KT), @Const(EXT), @Const(F1), @Const(W), @Const(CDr), @Const(CDi),
                                       w0, nw, nf, ::Val{D}, ::Val{NM}, @Const(AW), @Const(BWr), @Const(BWi),
                                       ::Val{MG}) where {D,NM,MG}
    i, w = @index(Global, NTuple)
    @inbounds begin
        wg = min(w0 + w, nw)                          # (past nw: XH is zero)
        f = F1[i]; nb = length(F1)
        ts = KT[nf, wg]
        for d in 1:D
            r0 = (d - 1) * (NM + 2)
            jj = (d - 1) * nb + i
            Ca = MG ? AW[jj, wg] : 1.0
            Cz = MG ? Complex(BWr[jj, wg], BWi[jj, wg]) : complex(1.0)
            wt = W[i, d] * Ca
            for k in 1:NM
                L[i, r0 + k, w] = wt * XH[i, w, k]
            end
            ph = -2pi * f * (EXT[(d - 1) * (2NM + 1) + 2NM + 1, wg] - ts)
            v = conj(Complex(CDr[i, d], CDi[i, d]) * Cz * Complex(cos(ph), sin(ph)))
            L[i, r0 + NM + 1, w] = v; L[i, r0 + NM + 2, w] = f * v
        end
    end
end
# M[:, :, w] = L[:, :, w]^H X[:, w, :] by hand (`GRAVITATIONALWAVEBAYES_MBP_LHX=kernel`, GPU backends only: @synchronize in a loop),
# for the tall-skinny shapes here (k = bins, m = D (NM + 2) = 8, n <= 24) that the cuBLAS strided-batched ZGEMM runs
# with a 32 x 32 tile. One work-group of 256 per walker: warp g (0..7) owns columns g + 1, g + 9, g + 17, its 32 lanes
# stride over the bins (coalesced loads of X's columns and of L, which the 8 warps share through L1), then each of the
# NCL x CPT accumulators is reduced over the lanes in shared memory.
@generated function _mbp_lhx_zero(::Val{N}) where {N}
    Expr(:tuple, (:(zero(ComplexF64)) for _ in 1:N)...)
end
@generated function _mbp_lhx_step(acc::NTuple{N,ComplexF64}, GL, XA, i, w, g, ::Val{NCL}, ::Val{CPT},
                                  ::Val{NCOL}) where {N,NCL,CPT,NCOL}
    ex = Any[Expr(:meta, :inline)]
    for r in 1:NCL
        push!(ex, :($(Symbol(:l, r)) = conj(@inbounds GL[i, $r, w])))
    end
    for q in 0:CPT-1
        push!(ex, :($(Symbol(:c, q)) = g + $(8q + 1)),
              :($(Symbol(:x, q)) = $(Symbol(:c, q)) <= $NCOL ? @inbounds(XA[i, w, $(Symbol(:c, q))]) : zero(ComplexF64)))
    end
    terms = Any[:(acc[$(q * NCL + r)] + $(Symbol(:l, r)) * $(Symbol(:x, q))) for q in 0:CPT-1 for r in 1:NCL]
    push!(ex, Expr(:tuple, terms...))
    return Expr(:block, ex...)
end
@kernel function _mbp_lhx_kernel!(GM, @Const(GL), @Const(XA), ::Val{NCL}, ::Val{NCOL}) where {NCL,NCOL}
    w = @index(Group, Linear)
    t = @index(Local, Linear) - 1
    S = @localmem ComplexF64 (256,)
    lane = t % 32; g = t ÷ 32
    nb = size(GL, 1)
    CPT = cld(NCOL, 8)
    acc = _mbp_lhx_zero(Val(NCL * cld(NCOL, 8)))
    i = lane + 1
    while i <= nb
        acc = _mbp_lhx_step(acc, GL, XA, i, w, g, Val(NCL), Val(cld(NCOL, 8)), Val(NCOL))
        i += 32
    end
    for s in 1:(NCL * CPT)
        @inbounds S[t + 1] = acc[s]
        @synchronize
        off = 16
        while off >= 1
            if lane < off
                @inbounds S[t + 1] += S[t + 1 + off]
            end
            @synchronize
            off ÷= 2
        end
        q = (s - 1) ÷ NCL; r = s - q * NCL; c = g + 8q + 1
        if lane == 0 && c <= NCOL
            @inbounds GM[r, c, w] = S[t + 1]
        end
        @synchronize
    end
end

# M[:, :, w] = L^H [H | blocks] ((D (NM + 2)) x (NM nblk)) and c_kd -> SUMS rows (layout `_mbp_terms`), tail as `_mbp_tail!`.
@kernel function _mbp_gemm_combine_kernel!(SUMS, @Const(M), @Const(EXT), @Const(KT), JT, w0, nw, nf, ::Val{D}, ::Val{NM},
                                           ::Val{NTK}, ::Val{NT}, ::Val{NL}, ::Val{G}) where {D,NM,NTK,NT,NL,G}
    w = @index(Global, Linear)
    @inbounds if w0 + w <= nw
        wg = w0 + w
        NV = _mbp_nv(NM, NTK)
        for d in 1:D
            c = _mbp_c(EXT, wg, d, Val(NM))
            r0 = (d - 1) * (NM + 2); o = (d - 1) * NV
            hh = zero(ComplexF64); z = zero(ComplexF64); fz = zero(ComplexF64)
            for l in 1:NM
                cg = zero(ComplexF64)                  # (c^H G)_l = R_l
                for a in 1:NM
                    cg += conj(c[a]) * M[r0 + a, l, w]
                end
                hh += cg * c[l]
                q = M[r0 + NM + 1, l, w]
                z += c[l] * q; fz += c[l] * M[r0 + NM + 2, l, w]
                SUMS[o + 5 + 2l - 1, wg] = real(q); SUMS[o + 5 + 2l, wg] = imag(q)
                SUMS[o + 5 + 2NM + 2l - 1, wg] = real(cg); SUMS[o + 5 + 2NM + 2l, wg] = imag(cg)
            end
            SUMS[o + 1, wg] = real(hh); SUMS[o + 2, wg] = real(z); SUMS[o + 3, wg] = imag(z)
            SUMS[o + 4, wg] = real(fz); SUMS[o + 5, wg] = imag(fz)
            if G
                for j in 1:NTK
                    cb = j * NM                        # block j + 1's first column - 1
                    t = zero(ComplexF64); zt = zero(ComplexF64)
                    for l in 1:NM
                        cg = zero(ComplexF64)
                        for a in 1:NM
                            cg += conj(c[a]) * M[r0 + a, cb + l, w]
                        end
                        t += cg * c[l]; zt += c[l] * M[r0 + NM + 1, cb + l, w]
                    end
                    SUMS[o + 5 + 4NM + j, wg] = 2real(t)
                    SUMS[o + 5 + 4NM + NTK + 2j - 1, wg] = real(zt); SUMS[o + 5 + 4NM + NTK + 2j, wg] = imag(zt)
                end
            end
        end
        if G
            SUMS[NL + 1, wg] = KT[nf, wg]
            for j in 1:NT
                SUMS[NL + 1 + j, wg] = JT[j][nf, wg]
            end
        end
    end
end
# Default bin-sum path per spec (1024 walkers, V100, GW150914 20-896 Hz): XPHM 223 logL+grad 1177 -> 857 ms
# with GEMMs; XHM no gain (41.8 vs 41.0 ms) and value-only calls slower (13 -> 21 ms); XPHM 102 even.
_mbp_bin_sums(spec) = :kernel
# Work buffers for chunks of `wc` walkers within `bytes`; nblk = 1 (value-only) or 1 + NT + NK; NMB intermediate
# modes (`_mbp_gemm_nmb`) with their NT tangent blocks.
function _mbp_gemm_buffers(backend, nb, nw, D, NM, NT, NK, vb, bytes, NMB)
    nblk = vb ? 1 : 1 + NT + NK; ncl = D * (NM + 2); nmt = vb ? 1 : 1 + NT
    zc(dims...) = KernelAbstractions.zeros(backend, ComplexF64, dims...)
    perw = 16 * nb * (nblk * NM + ncl + NMB * nmt) + 16 * ncl * nblk * NM
    wc = cld(nw, cld(nw, clamp(bytes ÷ perw, 1, nw)))
    X = zc(nb * wc * NM * nblk)
    blk(b) = GPU.block_alias(X, (b - 1) * nb * wc * NM, (nb, wc, NM))
    XMB = NMB > 0 ? zc(nb * wc * NMB * nmt) : nothing
    mblk(b) = GPU.block_alias(XMB, b * nb * wc * NMB, (nb, wc, NMB))
    return (; wc, X, XA = GPU.block_alias(X, 0, (nb, wc, nblk * NM)), XH = blk(1),
            XHT = vb ? nothing : ntuple(j -> blk(1 + j), NT), XK = vb ? () : ntuple(q -> blk(1 + NT + q), NK),
            XMB, XM = NMB > 0 ? mblk(0) : nothing, XMT = NMB > 0 && !vb ? ntuple(mblk, NT) : nothing,
            GL = zc(nb, ncl, wc), GM = zc(ncl, nblk * NM, wc), GMv = zc(ncl, NM, wc))
end

function _mbp_device(spec, backend, nwalkers::Int, a)
    (; NM, NT, NK, NKR, NL, D, W, CDr, CDi, F1, fref, lik, gpu, setup) = a
    cal = get(a, :cal, nothing)
    NN = cal === nothing ? 0 : cal.nn
    devc(x) = (b = KernelAbstractions.allocate(backend, Float64, size(x)); copyto!(b, x); b)
    WCd = NN > 0 ? devc(cal.W) : KernelAbstractions.zeros(backend, Float64, 1, 1)
    CALd = KernelAbstractions.zeros(backend, Float64, max(2NN * D, 1), nwalkers)
    UZd = NN > 0 ? KernelAbstractions.zeros(backend, Float64, 3D, length(F1), nwalkers) :
          KernelAbstractions.zeros(backend, Float64, 1, 1, 1)
    GCd = KernelAbstractions.zeros(backend, Float64, max(5NN * D, 1), nwalkers)
    cal_dev! = _mbp_cal_kernel!(backend, 64)
    # calibration marginalization (`calmarg`: response curves, ncurves x D nbins): A = |C|^2, B = C on the device
    CM = get(a, :calmarg, nothing)
    MG = CM !== nothing
    nb = length(F1); J = D * nb; nc = MG ? size(CM, 1) : 1
    if MG
        size(CM, 2) == J || throw(ArgumentError("calibration curves have $(size(CM, 2)) columns, the data $J bins"))
        Ad = KernelAbstractions.allocate(backend, Float64, nc, J); copyto!(Ad, abs2.(CM))
        Bd = KernelAbstractions.allocate(backend, ComplexF64, nc, J); copyto!(Bd, ComplexF64.(CM))
        Vd = KernelAbstractions.allocate(backend, Float64, J, nwalkers)          # V, then a
        Ud = KernelAbstractions.allocate(backend, ComplexF64, J, nwalkers)       # U, then b
        BWr = KernelAbstractions.allocate(backend, Float64, J, nwalkers); BWi = KernelAbstractions.allocate(backend, Float64, J, nwalkers)
        HHd = KernelAbstractions.allocate(backend, Float64, nc, nwalkers); ZZd = KernelAbstractions.allocate(backend, ComplexF64, nc, nwalkers)
        cmA = ntuple(_ -> KernelAbstractions.allocate(backend, Float64, nc, nwalkers), 5)   # G1, LK, EU, EU2, ED
        G2d = KernelAbstractions.allocate(backend, ComplexF64, nc, nwalkers)
        LLd = KernelAbstractions.allocate(backend, Float64, nwalkers); GDLd = KernelAbstractions.allocate(backend, Float64, nwalkers)
        DLd = KernelAbstractions.allocate(backend, Float64, nwalkers)
        ddtot = sum(a.dd); dfc = a.dfc
        la = KernelFused._lik_array(a.cfg, backend)
        vu_dev! = _mbp_vu_kernel!(backend, _mbp_fg(spec)); cm_dev! = _mbp_cm_kernel!(backend, 32)
    else
        Vd = KernelAbstractions.zeros(backend, Float64, 1, 1); BWr = Vd; BWi = Vd
    end
    ST = _mbp_setup_type(spec)
    stv = Val(ST)
    nflat = _xhm_flatn(ST)
    nk = nflat + 1 + _mbp_naux(spec)                 # setup, time shift, aux rows (`_mbp_naux`)
    KT = KernelAbstractions.zeros(backend, Float64, nk, nwalkers)
    JT = ntuple(_ -> KernelAbstractions.zeros(backend, Float64, nk, nwalkers), NT)
    POSd = KernelAbstractions.allocate(backend, Float64, NT + NKR, nwalkers)
    FG = _mbp_fg(spec)
    KS = KernelAbstractions.zeros(backend, Float64, max(2NM * NK, 1), FG * nwalkers)
    HS = KernelAbstractions.zeros(backend, Float64, 2NM, FG * nwalkers)
    dHS = ntuple(_ -> KernelAbstractions.zeros(backend, Float64, 2NM, FG * nwalkers), NT)
    KTh = zeros(nk, nwalkers); JTh = ntuple(_ -> zeros(nk, nwalkers), NT)
    setup_host! = _mbp_setup_kernel!(KernelAbstractions.CPU())
    SUMSd = KernelAbstractions.zeros(backend, Float64, NL + 1 + NT, nwalkers)
    ntv = Val(NT); nkrv = Val(NKR)
    setup_dev! = _mbp_setup_kernel!(backend)
    sums_dev! = _mbp_sums_kernel!(backend, FG)
    val_dev! = _mbp_val_kernel!(backend, FG)
    # `bin_sums` (`:kernel` | `:gemm`, default `_mbp_bin_sums(spec)`), `gemm_bytes` (default 2 GiB): see
    # `_mbp_gemm_store!`. Sampled calibration (NN > 0) needs the kernel's per-bin rows.
    bs = get(a, :bin_sums, GPU.tuning(:MBP_BIN_SUMS, _mbp_bin_sums(spec)))   # env override of the per-spec default
    bs in (:kernel, :gemm) || throw(ArgumentError("bin_sums must be :kernel or :gemm, got $bs"))
    haskey(a, :bin_sums) && bs === :gemm && NN > 0 &&
        throw(ArgumentError("bin_sums = :gemm does not support sampled calibration"))
    gemm = bs === :gemm && NN == 0
    gb = gemm ? _mbp_gemm_buffers(backend, nb, nwalkers, D, NM, NT, NK, false, Int(get(a, :gemm_bytes, 2^31)),
                                  _mbp_gemm_nmb(spec)) : nothing
    gl_dev! = _mbp_gemm_lhs_kernel!(backend)
    gc_dev! = _mbp_gemm_combine_kernel!(backend)
    lhx_dev! = _mbp_lhx_kernel!(backend, 256)
    function sums_gemm!(EXTd, grad)
        # the L^H X stage by hand (`GRAVITATIONALWAVEBAYES_MBP_LHX=kernel`, the GPU default: XPHM 223, V100, 4096 walkers, bin sums
        # 110.2 -> 107.8 ms) or cuBLAS (`gemm`)
        lhx = !(backend isa KernelAbstractions.CPU) && GPU.tuning(:MBP_LHX, :kernel) === :kernel
        (; wc, XA, XH, GL, GM, GMv) = gb
        for w0 in 0:wc:(nwalkers - 1)
            _mbp_gemm_modes!(spec, backend, gb, KT, JT, F1, w0, nwalkers, stv, Val(NM), Val(NK), grad, gpu)
            gl_dev!(GL, XH, KT, EXTd, F1, W, CDr, CDi, w0, nwalkers, nflat + 1, Val(D), Val(NM), Vd, BWr, BWi,
                    Val(MG && grad); ndrange=(nb, wc))
            if lhx
                grad ? lhx_dev!(GM, GL, XA, Val(D * (NM + 2)), Val((1 + NT + NK) * NM); ndrange=256 * wc) :
                       lhx_dev!(GMv, GL, XH, Val(D * (NM + 2)), Val(NM); ndrange=256 * wc)
            else
                grad ? GPU.batched_adj_mul_mid!(GM, GL, XA) : GPU.batched_adj_mul_mid!(GMv, GL, XH)
            end
            gc_dev!(SUMSd, grad ? GM : GMv, EXTd, KT, JT, w0, nwalkers, nflat + 1, Val(D), Val(NM), Val(NT + NK), ntv,
                    Val(NL), Val(grad); ndrange=wc)
        end
        return nothing
    end
    swidth = GPU.tuning(:MBP_SETUP_WIDTH, _mbp_setup_width(spec))
    sgroups = _mbp_groups(NT, swidth)
    # every tangent group in one launch per stage (`GRAVITATIONALWAVEBAYES_MBP_SETUP_RT`, default on: `_mbp_setup_rt_kernel!`)
    srt = GPU.tuning(:MBP_SETUP_RT, true)
    sgv = Val(_mbp_pad_groups(sgroups, min(swidth, NT)))
    JX = srt && length(sgroups[end]) < min(swidth, NT) ? KernelAbstractions.zeros(backend, Float64, nk, nwalkers) : KT
    setup_rt! = _mbp_setup_rt_kernel!(backend)
    setup_val! = _mbp_setup_val_kernel!(backend)
    fx = (; fref, fmax = Float64(maximum(Array(F1))))   # (the post-setup stage's frequency range: XPHM 223's table)
    # `posh`: the host (theta, kraw) rows (`setup = :host`), or nothing: POSd already filled on the device
    function setup!(posh)
        if setup === :device
            posh === nothing || copyto!(POSd, posh)
            for s in 1:_mbp_setup_nstage(spec)
                if srt
                    setup_rt!(KT, JT, JX, POSd, spec, fref, lik, ntv, nkrv, gpu, Val(s), sgv;
                              ndrange=(nwalkers, length(sgroups)), workgroupsize=(32, 1))
                else
                    for js in sgroups
                        setup_dev!(KT, JT, POSd, spec, fref, lik, ntv, nkrv, gpu, Val(s), Val(js); ndrange=nwalkers,
                                   workgroupsize=32)
                    end
                end
            end
        else
            for s in 1:_mbp_setup_nstage(spec)
                setup_host!(KTh, JTh, posh, spec, fref, lik, ntv, nkrv, Val(false), Val(s), Val(Tuple(1:NT));
                            ndrange=nwalkers, workgroupsize=1)
            end
            copyto!(KT, KTh)
            foreach(j -> copyto!(JT[j], JTh[j]), 1:NT)
        end
        _mbp_post_setup!(spec, backend, KT, JT, nflat, nwalkers, gpu, fx)
        return nothing
    end
    # value-only calls (`setup = :device`; `GRAVITATIONALWAVEBAYES_MBP_SETUP_VALUE`, default on): no tangents
    sval = setup === :device && GPU.tuning(:MBP_SETUP_VALUE, true)
    function setup_value!(posh)
        sval || return setup!(posh)
        posh === nothing || copyto!(POSd, posh)
        for s in 1:_mbp_setup_nstage(spec)
            setup_val!(KT, POSd, spec, fref, lik, ntv, nkrv, Val(s); ndrange=nwalkers, workgroupsize=32)
        end
        _mbp_post_setup_value!(spec, backend, KT, nflat, nwalkers, fx)
        return nothing
    end
    # `sumsh`: a host array the sums are copied to, or nothing (`setup = :device`: read on the device as `SUMS`)
    function sums!(sumsh, EXTd)
        if gemm
            sums_gemm!(EXTd, true)
        else
            sums_dev!(SUMSd, KT, JT, HS, dHS, KS, EXTd, F1, W, CDr, CDi, spec, stv,
                      Val(nflat + 1), Val(D), Val(NM), ntv, Val(NK), Val(NL), Val(FG), gpu, CALd, WCd, UZd, Val(NN),
                      Vd, BWr, BWi, Val(MG); ndrange=FG * nwalkers)
        end
        sumsh === nothing || copyto!(sumsh, SUMSd)
        return nothing
    end
    # value-only calls: the three log-likelihood sums per detector (`_mbp_val_kernel!`; the other rows of `sumsh` are stale)
    function sums_value!(sumsh, EXTd)
        if gemm
            sums_gemm!(EXTd, false)
        else
            val_dev!(SUMSd, KT, EXTd, F1, W, CDr, CDi, spec, stv, Val(nflat + 1), Val(D), Val(NM), Val(NK),
                     Val(_mbp_nv(NM, NT + NK)), Val(FG), CALd, WCd, Val(NN); ndrange=FG * nwalkers)
        end
        sumsh === nothing || copyto!(sumsh, SUMSd)
        return nothing
    end
    # calibration marginalization: value pass, GEMMs, per-walker log-mean-exp (llh, gdlh host vectors, or nothing: kept
    # on the device as LLM, GDLM), then the second-pass weights a = A' G1 (into Vd), b = B' G2 (into BWr, BWi) for
    # `sums!`. dL: host vector, or a device row (`setup = :device`: the positions' row 5).
    function margll!(llh, gdlh, EXTd, dL)
        vu_dev!(Vd, Ud, KT, EXTd, F1, W, CDr, CDi, spec, stv, Val(nflat + 1), Val(D), Val(NM), Val(NK), Val(FG);
                ndrange=FG * nwalkers)
        LinearAlgebra.mul!(HHd, Ad, Vd)
        LinearAlgebra.mul!(ZZd, Bd, Ud)
        dL isa Vector ? copyto!(DLd, dL) : (DLd .= dL)
        cm_dev!(LLd, cmA[1], G2d, GDLd, cmA[2], cmA[3], cmA[4], cmA[5], HHd, ZZd, DLd, ddtot, dfc, nc, la; ndrange=nwalkers)
        LinearAlgebra.mul!(Vd, transpose(Ad), cmA[1])
        LinearAlgebra.mul!(Ud, transpose(Bd), G2d)
        BWr .= real.(Ud); BWi .= imag.(Ud)
        llh === nothing || (copyto!(llh, LLd); copyto!(gdlh, GDLd))
        return nothing
    end
    setcal!(calh) = (copyto!(CALd, calh); nothing)
    function calsums!(gch)
        cal_dev!(GCd, UZd, CALd, WCd, Val(NN), Val(D); ndrange=NN * D * nwalkers)
        gch === nothing || copyto!(gch, GCd)
        return nothing
    end
    # device arrays of the `setup = :device` per-walker stages: POS (the setup's input rows), SUMS, the node sums GC,
    # the calibration-marginalized LLM, GDLM (placeholders without)
    devs = (; POS=POSd, SUMS=SUMSd, GC=GCd, LLM=MG ? LLd : SUMSd, GDLM=MG ? GDLd : SUMSd)
    return MG ? (; setup!, setup_value!, sums!, sums_value!, dead=nothing, margll!, curve_ll=cmA[2], devs) :
           NN > 0 ? (; setup!, setup_value!, sums!, sums_value!, dead=nothing, setcal!, calsums!, devs) :
                    (; setup!, setup_value!, sums!, sums_value!, dead=nothing, devs)
end

# The closure (host arrays in and out) for any spec; see `xhm_binparallel_logp_and_grad`. `dev_kw` go to the spec's
# `_mbp_device`.
function _mbp_logp_and_grad(spec, data, cfg::PEConfig{T}, freqs; backend, nwalkers::Int, setup::Symbol=:host,
                            timers=nothing, calibration=nothing, calmarg=nothing, box=_call_log_prior_vec,
                            dev_kw...) where {T}
    setup in (:host, :device) || throw(ArgumentError("setup must be :host or :device"))
    dets = data isa AbstractVector ? collect(data) : [data]
    D = length(dets)
    model = _mbp_model(spec)
    all(d -> d.freqs == dets[1].freqs, dets) ||
        throw(ArgumentError("bin-parallel $(nameof(typeof(model))) needs identical frequency grids"))
    KernelFused._check_likelihood_options(cfg, model, D)
    NM = _mbp_nm(spec); NT = _mbp_nt(spec); NP = _mbp_np(spec); NK = _mbp_nk(spec); NKR = _mbp_nkr(spec)
    NTT = NT + NK
    NV = _mbp_nv(NM, NTT); NE = 2NM + 1; NL = NV * D
    # the waveform's phase reference frequency: the data's lowest frequency, or (IMRPhenomXHM) `phase_ref_frequency` -- a
    # phase defined at 20 Hz is coupled to the masses / spins by the inspiral phase up to the band (the thin direction
    # that pins NUTS at small steps); the sampled phase is then the phase at that frequency (uniform prior unchanged)
    fphi = Float64(cfg.phase_ref_frequency)
    # (SEOBNRv5HM: a phase offset from its own tables, `_sdh_phase_shift`; `fref` stays the lowest data frequency)
    (fphi > 0 && !(spec isa Union{_XHMSpec,_SEOBHMSpec})) &&
        throw(ArgumentError("phase_ref_frequency: on the mode-sum device path IMRPhenomXHM / SEOBNRv5HM only"))
    fref = fphi > 0 && spec isa _XHMSpec ? fphi : Float64(dets[1].freqs[1])
    g = merge(_xasbp_geometry(dets, cfg), (; fref))
    lik = KernelFused._lik_opts(cfg)
    dev(a) = (b = KernelAbstractions.allocate(backend, Float64, size(a)); copyto!(b, a); b)
    W = dev(reduce(hcat, [[p == 0 ? 0.0 : 1 / p for p in d.psd] for d in dets]))
    CD = reduce(hcat, [[d.psd[i] == 0 ? 0.0im : conj(d.strain[i]) / d.psd[i] for i in eachindex(d.psd)] for d in dets])
    CDr = dev(real.(CD)); CDi = dev(imag.(CD))
    F1 = dev(Float64.(dets[1].freqs))
    gpu = Val(!(backend isa KernelAbstractions.CPU))
    # calibration (`Calibration.calibration_prior`): rows NP + 1 : NP + NCAL of the positions are standard normals z, the
    # node values mu + sigma z (per detector: NN amplitude then NN phase nodes, detectors in `data` order)
    cal = calibration
    NCAL = cal === nothing ? 0 : 2 * cal.nn * D
    cal === nothing || size(cal.W, 1) == length(dets[1].freqs) ||
        throw(ArgumentError("calibration weights built for $(size(cal.W, 1)) bins, data has $(length(dets[1].freqs))"))
    cal === nothing || length(cal.mu) == NCAL || throw(ArgumentError("calibration prior has $(length(cal.mu)) nodes, need $NCAL"))
    # `calmarg`: response curves (`calibration_curves(...).C`) for calibration MARGINALIZATION (no sampled rows)
    (calmarg === nothing || cal === nothing) || throw(ArgumentError("calibration: sample OR marginalize, not both"))
    (calmarg === nothing || !cfg.marginalize_phase) || throw(ArgumentError("calibration marginalization: no phase marginalization"))
    MG = calmarg !== nothing
    dv = _mbp_device(spec, backend, nwalkers, (; NM, NT, NK, NKR, NL, D, W, CDr, CDi, F1, fref, lik, gpu, setup,
                                               cfg, cal, calmarg, dd=g.dd, dfc=g.df[1], dev_kw...))
    (MG && !all(==(g.df[1]), g.df)) && throw(ArgumentError("calibration marginalization needs one frequency spacing"))
    llmh = zeros(nwalkers); gdlh = zeros(nwalkers); dLh = zeros(nwalkers)
    (cal === nothing || haskey(dv, :calsums!)) || throw(ArgumentError("this spec's device path has no calibration support"))
    calh = zeros(max(NCAL, 1), nwalkers)
    gch = zeros(max(5 * NCAL ÷ 2, 1), nwalkers)
    EXTd = KernelAbstractions.allocate(backend, Float64, NE * D, nwalkers)
    NS = NL + 1 + NT
    posh = zeros(NT + NKR, nwalkers); crdh = zeros(NTT, nwalkers); exth = zeros(NE * D, nwalkers)
    sumsh = zeros(NS, nwalkers)
    nsv = Val(NS); npv = Val(NP); dvv = Val(D); nmv = Val(NM); nttv = Val(NTT)
    if setup === :device
        dv.dead === nothing || throw(ArgumentError("setup = :device: not for specs with dead-zone walkers"))
        return _mbp_device_closure(spec, backend, dv, g, lik, cal, cfg, box, timers; nwalkers, NP, NCAL, NE, NTT, D, NM, MG)
    end
    # `curve_ll` (calibration marginalization only): receives the per-curve log-likelihoods (ncurves x nwalkers).
    # `value_only`: the log-likelihood alone (`grad_out` zeroed): no surrogate reverse on the host, and specs whose device
    # path has primal-only stages (`setup_value!` / `sums_value!`, SEOBNRv5HM) skip their forward tangents.
    return function (grad_out::AbstractMatrix{T}, logp_out::AbstractVector{T}, positions::AbstractMatrix{T}, active_mask;
                     curve_ll = nothing, value_only::Bool = false)
        size(positions) == (NP + NCAL, nwalkers) ||
            throw(ArgumentError("built for $(NP + NCAL) x $nwalkers positions, got $(size(positions))"))
        if NCAL > 0
            @inbounds for w in 1:nwalkers, r in 1:NCAL
                calh[r, w] = cal.mu[r] + cal.sigma[r] * positions[NP + r, w]
            end
            dv.setcal!(calh)
        end
        tick = time_ns()
        lap!(k) = timers === nothing || (KernelAbstractions.synchronize(backend); t = time_ns();
                                          timers[k] += (t - tick) / 1e9; tick = t)
        Threads.@threads for w in 1:nwalkers
            x = ntuple(k -> Float64(positions[k, w]), npv)
            pv = (_mbp_theta(spec, x)..., _mbp_kraw(spec, x, g)...)
            cv = _mbp_coords(spec, x, g)
            for j in 1:NT + NKR
                posh[j, w] = pv[j]
            end
            for j in 1:NTT
                crdh[j, w] = cv[j]
            end
        end
        value_only && haskey(dv, :setup_value!) ? dv.setup_value!(posh) : dv.setup!(posh)
        lap!(1)
        Threads.@threads for w in 1:nwalkers
            E = _mbp_ext(spec, ntuple(k -> Float64(positions[k, w]), npv), g, dvv)
            for d in 1:D
                c, tau = E[d]
                for k in 1:NM
                    exth[(d - 1) * NE + 2k - 1, w] = real(c[k]); exth[(d - 1) * NE + 2k, w] = imag(c[k])
                end
                exth[(d - 1) * NE + NE, w] = tau
            end
        end
        copyto!(EXTd, exth)
        lap!(2)
        if MG
            @inbounds for w in 1:nwalkers
                dLh[w] = positions[5, w]
            end
            dv.margll!(llmh, gdlh, EXTd, dLh)
            curve_ll === nothing || copyto!(curve_ll, dv.curve_ll)
        end
        if !value_only
            dv.sums!(sumsh, EXTd)
            NCAL > 0 && dv.calsums!(gch)
        elseif !MG
            # (marginalized calibration: the value came from `margll!`; sampled calibration enters the sums' weights)
            haskey(dv, :sums_value!) ? dv.sums_value!(sumsh, EXTd) : dv.sums!(sumsh, EXTd)
        end
        lap!(3)
        dead = dv.dead
        Threads.@threads for w in 1:nwalkers
            if dead !== nothing && dead[w]
                logp_out[w] = T(KernelFused._SEOBV5F2_DEADZONE_LOGLIKELIHOOD)
                @inbounds for k in 1:NP + NCAL
                    grad_out[k, w] = zero(T)
                end
                continue
            end
            x = ntuple(k -> Float64(positions[k, w]), npv)
            if value_only
                logp_out[w] = T(MG ? llmh[w] :
                                KernelFused._marg_combine(_mbp_pre0(ntuple(v -> sumsh[v, w], nsv), g, dvv,
                                                                    Val(NV))..., x[5], lik))
                @inbounds for k in 1:NP + NCAL
                    grad_out[k, w] = zero(T)
                end
                continue
            end
            S = ntuple(v -> sumsh[v, w], nsv)
            th0 = ntuple(j -> crdh[j, w], nttv)
            v, gr, (gl, gh, gzr, gzi) = MG ?
                (llmh[w], _mbp_post_cm(spec, x, th0, _mbp_ext_load(exth, w, dvv, nmv), S, g, dvv, gdlh[w]), (0.0, 0.0, 0.0, 0.0)) :
                _mbp_post(spec, x, th0, _mbp_ext_load(exth, w, dvv, nmv), S, g, lik, dvv)
            logp_out[w] = T(v)
            @inbounds for k in 1:NP
                grad_out[k, w] = T(gr[k])
            end
            if NCAL > 0
                # dlogL/d(node) from the node sums: ll_plain = dd - 2 df h2 + 4 df Re z, hh = 4 df h2, zr, zi = 4 df z
                NN = cal.nn
                @inbounds for d in 1:D, k in 1:NN
                    q = 5 * (k - 1 + NN * (d - 1)); df = g.df[d]
                    dh2 = gch[q + 1, w]; dzr = gch[q + 2, w]; dzi = gch[q + 3, w]
                    pzr = gch[q + 4, w]; pzi = gch[q + 5, w]
                    ga = gl * (-2df * dh2 + 4df * dzr) + gh * 4df * dh2 + gzr * 4df * dzr + gzi * 4df * dzi
                    gp = gl * 4df * pzr + gzr * 4df * pzr + gzi * 4df * pzi
                    ra = (d - 1) * 2NN + k
                    grad_out[NP + ra, w] = T(cal.sigma[ra] * ga)
                    grad_out[NP + ra + NN, w] = T(cal.sigma[ra + NN] * gp)
                end
            end
        end
        lap!(4)
        logp_out .+= box(positions)    # support: the package box by default, or a configured prior's (`prior_support_vec`)
        return nothing
    end
end

# `setup = :device`: the closure with every stage on the backend (see the header). Timers: (a) prep + setup,
# (b) the Jacobians, (c) bin sums, (d) the surrogate kernel and the copies back.
_mbp_jac_width(spec) = 4
function _mbp_device_closure(spec, backend, dv, g, lik, cal, cfg::PEConfig{T}, box, timers; nwalkers, NP, NCAL, NE, NTT,
                             D, NM, MG) where {T}
    alloc(dims...) = KernelAbstractions.zeros(backend, Float64, dims...)
    NF = NE * D + NTT
    Xd = alloc(NP + NCAL, nwalkers); xh = zeros(NP + NCAL, nwalkers)
    Fd = alloc(NF, nwalkers)
    JF = ntuple(_ -> alloc(NF, nwalkers), NP)
    LLd = alloc(nwalkers); Gd = alloc(NP + NCAL, nwalkers)
    llh = zeros(nwalkers); gh = zeros(NP + NCAL, nwalkers)
    CSd = alloc(max(NCAL, 1)); NCAL > 0 && copyto!(CSd, Float64.(cal.sigma))
    calh = zeros(max(NCAL, 1), nwalkers)
    (; POS, SUMS, GC, LLM, GDLM) = dv.devs
    gpu = Val(!(backend isa KernelAbstractions.CPU))
    dvv = Val(D); nmv = Val(NM); npv = Val(NP); nttv = Val(NTT)
    prep! = _mbp_prep_kernel!(backend, 32); jac! = _mbp_jac_kernel!(backend, 32); post! = _mbp_post_kernel!(backend, 32)
    jgroups = _mbp_groups(NP, GPU.tuning(:MBP_JAC_WIDTH, _mbp_jac_width(spec)))
    NN = cal === nothing ? 0 : cal.nn
    return function (grad_out::AbstractMatrix{T}, logp_out::AbstractVector{T}, positions::AbstractMatrix{T}, active_mask;
                     curve_ll = nothing, value_only::Bool = false)
        size(positions) == (NP + NCAL, nwalkers) ||
            throw(ArgumentError("built for $(NP + NCAL) x $nwalkers positions, got $(size(positions))"))
        tick = time_ns()
        lap!(k) = timers === nothing || (KernelAbstractions.synchronize(backend); t = time_ns();
                                          timers[k] += (t - tick) / 1e9; tick = t)
        xh .= positions
        copyto!(Xd, xh)
        if NCAL > 0
            @inbounds for w in 1:nwalkers, r in 1:NCAL
                calh[r, w] = cal.mu[r] + cal.sigma[r] * xh[NP + r, w]
            end
            dv.setcal!(calh)
        end
        prep!(POS, Fd, Xd, spec, g, dvv, nmv, npv; ndrange=nwalkers)
        value_only && haskey(dv, :setup_value!) ? dv.setup_value!(nothing) : dv.setup!(nothing)
        lap!(1)
        if !value_only
            for js in jgroups
                jac!(Fd, JF, Xd, spec, g, dvv, nmv, npv, Val(js), gpu; ndrange=nwalkers)
            end
        end
        lap!(2)
        if MG
            dv.margll!(nothing, nothing, Fd, view(Xd, 5, :))
            curve_ll === nothing || copyto!(curve_ll, dv.curve_ll)
        end
        if !value_only
            dv.sums!(nothing, Fd)
            NCAL > 0 && dv.calsums!(nothing)
        elseif !MG
            haskey(dv, :sums_value!) ? dv.sums_value!(nothing, Fd) : dv.sums!(nothing, Fd)
        end
        lap!(3)
        post!(LLd, Gd, SUMS, Xd, JF, LLM, GDLM, GC, CSd, spec, g, lik, dvv, nmv, nttv, npv, Val(NN), Val(MG),
              Val(value_only); ndrange=nwalkers)
        copyto!(llh, LLd)
        logp_out .= llh
        if value_only
            fill!(grad_out, zero(T))
        else
            copyto!(gh, Gd)
            grad_out .= gh
        end
        lap!(4)
        logp_out .+= box(positions)    # support: the package box by default, or a configured prior's (`prior_support_vec`)
        return nothing
    end
end

# ---- IMRPhenomXHM spec ----
struct _XHMSpec end
_mbp_model(::_XHMSpec) = IMRPhenomXHM()
_mbp_nm(::_XHMSpec) = 5
_mbp_nt(::_XHMSpec) = 4                              # mc, q, chi1, chi2
_mbp_np(::_XHMSpec) = 11
_mbp_setup_type(::_XHMSpec) = _xhm_setup_type()
const _XHM_MM = (2, 1, 3, 2, 4)                      # azimuthal number m of each mode, `_xhm_modes` order

@inline _mbp_modes(::_XHMSpec, KT, w, f, stv::Val) = _xhm_modes_fast(f, _xhm_views(stv, KT, w)...)
@inline _mbp_theta(::_XHMSpec, x) = (x[1], x[2], x[3], x[4])
# Column w of KT: the flat setup at dL = 1, phase = 0, then the time shift, in setup stages: 1 = the system setup and
# the time shift, 2-5 = the mode setups (`_xhm_mode_stage!`).
const _XHM_ST = _xhm_setup_type()
const _XHM_O = _xhm_flatn(_XHM_ST) + 1                # the time-shift row (a constant: device code)
_mbp_setup_nstage(::_XHMSpec) = 5
@inline function _mbp_setup_stage!(::_XHMSpec, ::Val{1}, KT, w, mc, q, chi1, chi2, kr, fref, lik)
    m1, m2, _, _ = _eobimr_physical(mc, q, zero(mc), zero(mc))
    mcc, eta = KernelFused._chirp_mass_eta(m1, m2)
    sys = KernelFused._xhm_system_setup(mcc, eta, chi1, chi2, one(mc), fref, zero(mc);
                                        GMsun_over_c3 = KernelFused.GMsun_over_c3,
                                        GMsun_over_c2_Gpc = KernelFused.GMsun_over_c2_Gpc)
    _xasbp_store_col!(KT, w, _xhm_flatten(sys), 0)
    @inbounds KT[_XHM_O, w] = KernelFused._xhm_time_shift(sys, lik)
    return nothing
end
@inline _mbp_setup_stage!(::_XHMSpec, ::Val{s}, KT, w, args...) where {s} = _xhm_mode_stage!(KT, w, Val(_XHM_ST), Val(s))

# Per detector the mode factors c_kd and tau_d + t_shift (generic element type: also the surrogate's), for aligned-spin
# modes h_{l,-m} at dL = 1 Gpc and phase 0 summed with `_xhm_add_mode!`'s polarization factors `fac` (azimuthal numbers
# `mm`): c_kd = (Fp fp_k + Fc fc_k) e^{i m_k phase} / dL.
@inline function _mbp_mode_ext(fac::NTuple{N}, mm::NTuple{N}, dL, t_c, phase, ra, dec, psi, g, d) where {N}
    T = typeof(dL)
    Fp, Fc = antenna_response_gmst(g.dets[d], ra, dec, psi, g.gmst)
    c = ntuple(k -> (Fp * fac[k][1] + Fc * fac[k][2]) * Complex(cos(mm[k] * phase), sin(mm[k] * phase)) / dL, Val(N))
    dt_ref = ifelse(g.geo & (g.tref > 0), time_delay_from_geocenter_gmst(g.dets[max(g.tref, 1)], ra, dec, g.gmst), zero(T))
    tau = t_c + ifelse(g.geo, time_delay_from_geocenter_gmst(g.dets[d], ra, dec, g.gmst) - dt_ref, zero(T))
    return c, tau
end
@inline function _mbp_ext(::_XHMSpec, x, g, ::Val{D}) where {D}
    _, _, dec, th = _eobimr_physical(x[1], x[2], x[9], x[10])
    fac = KernelFused._xhm_mode_factors(th)
    return ntuple(d -> _mbp_mode_ext(fac, _XHM_MM, x[5], x[6], x[7], x[8], dec, x[11], g, d), Val(D))
end

"""
    xhm_binparallel_logp_and_grad(data, cfg, freqs; backend, nwalkers) -> logp_and_grad!

IMRPhenomXHM log-likelihood and gradient with the setup and the frequency-bin sums on `backend` (`CUDABackend()`, or
`KernelAbstractions.CPU()` for testing) and the extrinsic factors and the final combination on the host (see the
header of `XHMBinParallel.jl`). Same likelihood options, time frame and 11-row sampled parametrization as
`gw_logp_and_grad(...; model=IMRPhenomXHM())`; all detectors must share one frequency grid. The closure takes HOST
`grad_out`, `logp_out`, `positions` and returns the log-likelihood plus the package's box prior, untempered.
`setup`: `:host` (default: the per-walker stages on host threads, the setup on the host) or `:device` (every stage on
`backend`; only positions in and log-likelihoods / gradients out). `timers`: optional `Vector{Float64}` of length 4
accumulating seconds in (a) setup, (b) extrinsic factors (`:device`: their Jacobians), (c) bin sums, (d) surrogate
(synchronizes).
"""
xhm_binparallel_logp_and_grad(data, cfg::PEConfig, freqs; kw...) = _mbp_logp_and_grad(_XHMSpec(), data, cfg, freqs; kw...)
