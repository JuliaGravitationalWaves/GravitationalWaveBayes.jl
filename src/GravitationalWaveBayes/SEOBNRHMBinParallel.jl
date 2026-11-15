# ------------------------------------------------------------------------------------------------
# SEOBNRv5HM log-likelihood and gradient on a KernelAbstractions device : the
# mode-sum likelihood of `XHMBinParallel.jl`, h_d(f) = sum_k c_kd H_k(f) e^{-2 pi i f tau_d}, with its stages (b)
# (c_kd, tau_d on the host) and (d) (first-order surrogate, `_marg_combine_grad`) shared, and stages (a) and (c) on
# the SEOBNRv5HM device tables (`SEOBNRHMDeviceTable.jl`) instead of a flat per-walker setup:
#   (a) every walker's tables with their forward tangents along theta = (mc, q, chi1, chi2) (`TT`), and the f_ref time
#       shift from the (2,2) mode's SPA table (with its tangents);
#   (c) the bin sums as batched GEMMs (see `_sdh_b_modes!`): per walker the 6 modes H_k(f) =
#       M GMsun/c^2 / (1 Gpc) `_sdh_fd_eval` with their forward tangents along the tables' tangents (d<h,h>/dtheta_j,
#       d<d,h>/dtheta_j at fixed c_kd and tau_d: tau_d's t_shift is a constant there, its tangent enters through the
#       surrogate as in the flat specs), and f <d,h>, Q_k, R_k as plain values, laid out as `_mbp_terms`' rows.
# The tables are too large to stage per walker in shared memory (Fourier tables alone 196 KB), hence (bin x walker)
# threads reading global memory as the (2,2) device likelihood. Walkers whose tables failed or end below the first bin
# get the dead-zone log-likelihood with zero gradient, as there.
# ------------------------------------------------------------------------------------------------

struct _SEOBHMSpec end
_mbp_model(::_SEOBHMSpec) = GravitationalWaveforms.SEOBNRv5HM()
_mbp_nm(::_SEOBHMSpec) = _SDH_NM
_mbp_nt(::_SEOBHMSpec) = 4                           # mc, q, chi1, chi2
_mbp_np(::_SEOBHMSpec) = 11
@inline _mbp_theta(::_SEOBHMSpec, x) = (x[1], x[2], x[3], x[4])

# `_xhm_add_mode!`'s (fp, fc) of the six modes (`GravitationalWaveforms._SEOBV5HM_L/_M` order; `hphc(::SEOBNRv5HM, ...)` sums them
# at azimuth pi/2 - phase, i.e. e^{i m phase} times these).
@assert GravitationalWaveforms._SEOBV5HM_L == (2, 2, 3, 3, 4, 4) && _SDH_M == (2, 1, 3, 2, 4, 3)
@inline _sdh_mode_factors(th) = (KernelFused._xhm_mode_factor(th, 2, 2), KernelFused._xhm_mode_factor(th, 2, 1),
    KernelFused._xhm_mode_factor(th, 3, 3), KernelFused._xhm_mode_factor(th, 3, 2), KernelFused._xhm_mode_factor(th, 4, 4),
    KernelFused._xhm_mode_factor(th, 4, 3))
@inline function _mbp_ext(::_SEOBHMSpec, x, g, ::Val{D}) where {D}
    _, _, dec, th = _eobimr_physical(x[1], x[2], x[9], x[10])
    fac = _sdh_mode_factors(th)
    return ntuple(d -> _mbp_mode_ext(fac, _SDH_M, x[5], x[6], x[7], x[8], dec, x[11], g, d), Val(D))
end

# (a) theta rows -> table inputs (chirp mass, eta, chi1, chi2).
@inline function _sdh_b_phys!(w, POS4, TH)
    @inbounds begin
        m1, m2, _, _ = _eobimr_physical(TH[1, w], TH[2, w], 0.0, 0.0)
        mcc, eta = KernelFused._chirp_mass_eta(m1, m2)
        POS4[1, w] = mcc; POS4[2, w] = eta; POS4[3, w] = TH[3, w]; POS4[4, w] = TH[4, w]
    end
    return nothing
end

# The f_ref time shift (`_sdt_time_shift`) from the (2,2) mode's SPA table.
@inline function _sdh_time_shift(c, lik)
    (lik.fref > 0 && c.success) || return 0.0
    fr = clamp(lik.fref, c.fmin[1], c.fspa_max[1])
    r = _SDT_W._seobv5f2_dense_to_freq(c.ts, view(c.phis, :, 1), view(c.omegas, :, 1), c.nreal, c.M_sec, fr)
    return (r.t_star - c.t_a) * c.M_sec + lik.toff
end
@inline function _sdh_b_tshift!(w, TS, HS, HT, FT, BLD, B, lik)
    @inbounds TS[w] = _sdh_time_shift(_sdh_tab(HS, HT, FT, BLD, B, w), lik)
    return nothing
end

# `phase_ref_frequency` fphi > 0: the orbital-phase offset delta(theta) of the (2,2) mode at fphi in the likelihood's
# own convention -- mode k's SPA phase is m phi*(t*) - 2 pi f M (t* - t_a) + pi/4 and the bins carry e^{+2 pi i f t_shift},
# so delta = phi* - pi fphi M (t* - t_a) + pi fphi t_shift. The sums rotate mode k by e^{-i m_k delta}: the sampled phase
# is the phase at fphi (the phase at the lowest frequency is coupled to mc / q / chi by the inspiral phase up to the
# band -- the thin direction that holds NUTS at small steps). Uniform periodic prior unchanged.
@inline function _sdh_phase_shift(c, lik, fphi)
    (fphi > 0 && c.success) || return 0.0
    fr = clamp(fphi, c.fmin[1], c.fspa_max[1])
    r = _SDT_W._seobv5f2_dense_to_freq(c.ts, view(c.phis, :, 1), view(c.omegas, :, 1), c.nreal, c.M_sec, fr)
    return r.phi_star - pi * fr * c.M_sec * (r.t_star - c.t_a) + pi * fr * _sdh_time_shift(c, lik)
end
@inline function _sdh_b_pshift!(w, PO, HS, HT, FT, BLD, B, lik, fphi)
    @inbounds PO[w] = _sdh_phase_shift(_sdh_tab(HS, HT, FT, BLD, B, w), lik, fphi)
    return nothing
end

const _SDH_NQ = 2 + 4 * _SDH_NM                    # plain-value rows per detector (f <d,h>, Q_k, R_k)

# (c) as batched GEMMs (the hand-written per-bin kernel it replaced: 44.6 -> 17.7 ms per 1024-walker
# call in the bench band, run wall -10%). With S = e^{-2 pi i f tau_d} (|S| = 1) every
# <h,h>-type sum is an entry of the walker's weighted Gram matrix of the 6 modes and their tangents, independent of tau_d:
#   <h,h> = c^H G c, d<h,h>/dtheta_i = 2 Re c^H G_i c, R_k = (c^H G)_k,  G = H^H diag(wt) H, G_i = H^H diag(wt) dH_i,
# and the <d,h>-type sums are dots with the walker's data vector v = cd S: Q_k = v^T H_k, <d,h> = c^T Q,
# d<d,h>/dtheta_i = c^T (v^T dH_i), f <d,h> = c^T ((f v)^T H). Per walker and detector the left matrix (bins x 8) is
# [wt H_1..6, conj(v), f conj(v)], so M = L^H [H | dH_1 .. dH_4] is ONE strided-batched ZGEMM over walkers per chunk
# (value-only calls: L^H H). Layout X[bin, walker, mode, block] (block 1 = H, 2..5 = dH_i): every block is a dense
# array of its own (`GPU.block_alias`, separate same-typed Enzyme shadows -- views of one buffer as shadows gave wrong
# tangents) and walker w's 30 columns sit at the uniform stride bins x walkers, so cuBLAS reads them in
# place (`GPU.batched_adj_mul_mid!`). Walkers are processed in chunks of `wc` (16 B x bins x (6 x 5 + 8 D) per
# walker); walkers past `nw` in the last chunk are zero.
# Modes with their forward tangents (shadows of H): H[j, w, k] = M GMsun/c^2/Gpc H_k(f_j) e^{-i m_k delta}.
@inline function _sdh_b_modes!(j, w, H, w0, nw, PO, F, HS, HT, FT, BLD, B)
    @inbounds begin
        wg = w0 + w
        if wg > nw
            Base.Cartesian.@nexprs 6 k -> H[j, w, k] = zero(ComplexF64)
            return nothing
        end
        tab = _sdh_tab(HS, HT, FT, BLD, B, wg)
        scale = B[wg].M * _SDT_GMSUN_C2_GPC
        f = F[j]
        ok = tab.success & (minimum(tab.fmin) <= f) & (f <= maximum(tab.fmax))
        dl = PO[wg]
        e1 = Complex(cos(dl), -sin(dl)); e2 = e1 * e1; e3 = e2 * e1; e4 = e2 * e2
        rot = (e2, e1, e3, e2, e4, e3)                  # e^{-i m_k delta}, m = `_SDH_M`
        Base.Cartesian.@nexprs 6 k -> begin
            re_k, im_k = ok ? _sdh_fd_eval(tab, k, f) : (0.0, 0.0)
            H[j, w, k] = Complex(scale * re_k, scale * im_k) * rot[k]
        end
    end
    return nothing
end
@assert _SDH_NM == 6
# Left matrices L[j, 8 (d - 1) + (1:8), w] (plain values).
@inline function _sdh_b_lhs!(j, w, L, H, w0, nw, EXT, TS, F, WT, CDr, CDi, ::Val{D}) where {D}
    NE = 2_SDH_NM + 1
    @inbounds begin
        wg = min(w0 + w, nw)                            # (past nw: H is zero)
        f = F[j]
        for d in 1:D
            r0 = 8(d - 1); wt = WT[j, d]
            for k in 1:_SDH_NM
                L[j, r0 + k, w] = wt * H[j, w, k]
            end
            ph = -2pi * f * (EXT[(d - 1) * NE + NE, wg] - TS[wg])
            v = conj(Complex(CDr[j, d], CDi[j, d]) * Complex(cos(ph), sin(ph)))
            L[j, r0 + 7, w] = v; L[j, r0 + 8, w] = f * v
        end
    end
    return nothing
end
# The GEMM outputs M0 = L^H H and MT[i] = L^H dH_i ((8 D) x 6 per walker) combined with c_kd into `sums!`'s reduced rows.
@inline function _sdh_b_combine!(w, R, RQ, RT, M0, MT, w0, nw, EXT, ::Val{D}) where {D}
    NM = _SDH_NM; NE = 2NM + 1
    @inbounds begin
        wg = w0 + w
        wg > nw && return nothing
        for d in 1:D
            o = (d - 1) * NE; r0 = 8(d - 1)
            c = ntuple(k -> Complex(EXT[o + 2k - 1, wg], EXT[o + 2k, wg]), Val(NM))
            hh = zero(ComplexF64); z = zero(ComplexF64); fz = zero(ComplexF64)
            for l in 1:NM
                cg = zero(ComplexF64)                   # (c^H G)_l = R_l
                for a in 1:NM
                    cg += conj(c[a]) * M0[r0 + a, l, w]
                end
                hh += cg * c[l]
                q = M0[r0 + 7, l, w]
                z += c[l] * q; fz += c[l] * M0[r0 + 8, l, w]
                RQ[2 + 2l - 1, 1, d, wg] = real(q); RQ[2 + 2l, 1, d, wg] = imag(q)
                RQ[2 + 2NM + 2l - 1, 1, d, wg] = real(cg); RQ[2 + 2NM + 2l, 1, d, wg] = imag(cg)
            end
            R[1, 1, d, wg] = real(hh); R[2, 1, d, wg] = real(z); R[3, 1, d, wg] = imag(z)
            RQ[1, 1, d, wg] = real(fz); RQ[2, 1, d, wg] = imag(fz)
            if RT !== nothing
                for i in 1:4
                    Mi = MT[i]; t = zero(ComplexF64); zt = zero(ComplexF64)
                    for l in 1:NM
                        cg = zero(ComplexF64)
                        for a in 1:NM
                            cg += conj(c[a]) * Mi[r0 + a, l, w]
                        end
                        t += cg * c[l]; zt += c[l] * Mi[r0 + 7, l, w]
                    end
                    RT[i][1, 1, d, wg] = 2real(t); RT[i][2, 1, d, wg] = real(zt); RT[i][3, 1, d, wg] = imag(zt)
                end
            end
        end
    end
    return nothing
end

# Work buffers of (c) for chunks of `wc` walkers within `bytes` (layout: see `_sdh_b_modes!`).
function _sdh_gemm_buffers(backend, nb, nw, D, vb, bytes)
    NM = _SDH_NM; nblk = vb ? 1 : 5
    zc(dims...) = KernelAbstractions.zeros(backend, ComplexF64, dims...)
    perw = 16 * nb * (nblk * NM + 8D) + 16 * 8D * nblk * NM
    wc0 = clamp(bytes ÷ perw, 1, nw)
    wc = cld(nw, cld(nw, wc0))                          # equal chunks
    X = zc(nb * wc * NM * nblk)
    blk(b) = GPU.block_alias(X, (b - 1) * nb * wc * NM, (nb, wc, NM))
    GMA = zc(8D, nblk * NM, wc)
    return (; wc, X, XA = GPU.block_alias(X, 0, (nb, wc, nblk * NM)), GH = blk(1),
            GHT = vb ? nothing : ntuple(i -> blk(i + 1), 4), GL = zc(nb, 8D, wc), GMA, GM = view(GMA, :, 1:NM, :),
            GMT = vb ? nothing : ntuple(i -> view(GMA, :, i * NM + 1:(i + 1) * NM, :), 4), GMv = zc(8D, NM, wc))
end

# Stages (a) and (c) for `_mbp_logp_and_grad`. Keywords (through `dev_kw`): `f_min` (table start, Hz), `dirpar_nodes`
# (one thread per tangent direction in the node-parallel stages and (c)'s modes), `table_timers`
# (a `Dict{Symbol,Float64}` accumulating `seobnr_hm_device_tables!`'s per-stage seconds; synchronizes), `table_kw` (a
# NamedTuple of more `seobnr_hm_device_tables!` keywords for the forward pass, e.g. `dirpar`, `plunge_split`),
# `value_build` (no tangent tables: value-only calls only, ~1/5 of the device memory), `gemm_bytes` (device memory
# budget of (c)'s work buffers, default 2 GiB; sets the walker chunk).
function _mbp_device(spec::_SEOBHMSpec, backend, nw::Int, a)
    (; NT, NL, D, W, CDr, CDi, F1, fref, lik) = a
    f_min = Float64(get(a, :f_min, 20.0)); dirpar = get(a, :dirpar_nodes, false)
    ttm = get(a, :table_timers, nothing)
    tkw = get(a, :table_kw, (;))                     # more keywords for `seobnr_hm_device_tables!` (forward pass)
    vb = get(a, :value_build, false)                 # value-only closure: no tangent tables (gradient calls throw)
    NM = _SDH_NM; NV = _mbp_nv(NM, NT)
    z(dims...) = KernelAbstractions.zeros(backend, Float64, dims...)
    T = SEOBNRHMDeviceTables(backend, nw)
    TT = vb ? nothing : ntuple(_ -> SEOBNRHMDeviceTables(backend, nw), 4)
    tg(f) = ntuple(k -> f(TT[k]), 4)
    TH = z(4, nw)
    SEED = vb ? nothing : ntuple(k -> (s = zeros(4, nw); s[k, :] .= 1.0; d = z(4, nw); copyto!(d, s); d), 4)
    TS = z(nw); TST = vb ? nothing : ntuple(_ -> z(nw), 4)
    fphi = Float64(a.cfg.phase_ref_frequency)
    PO = z(nw); POT = vb ? nothing : ntuple(_ -> z(nw), 4)        # phase_ref_frequency offset (and tangents)
    nb = length(F1)
    # (single assignment: the closures below capture it -- Julia 1.13 boxed-capture bug)
    gb = _sdh_gemm_buffers(backend, nb, nw, D, vb, Int(get(a, :gemm_bytes, 2^31)))
    R = z(3, 1, D, nw); RT = vb ? nothing : ntuple(_ -> z(3, 1, D, nw), 4); RQ = z(_SDH_NQ, 1, D, nw)
    Rh = zeros(3, 1, D, nw); RTh = ntuple(_ -> zeros(3, 1, D, nw), 4); RQh = zeros(_SDH_NQ, 1, D, nw)
    TSh = zeros(nw); TSTh = ntuple(_ -> zeros(nw), 4)
    dead = fill(false, nw)
    dv = Val(D)
    first_grad = Ref(true)
    function setup!(posh)
        vb && throw(ArgumentError("SEOBNRv5HM closure built value-only (value_build = true): no gradient"))
        # The first forward launch reserves the forward kernels' stack (~42 KB x every resident thread = 6.8 GB on a
        # V100), which the driver cannot take from CUDA.jl's pool: uncollected arrays of the value-only stages before
        # it made that launch fail on a 16 GB card (13.1 GB in the pool at the first NUTS stage).
        if first_grad[]
            GC.gc(true); GPU.reclaim()
            first_grad[] = false
        end
        copyto!(TH, posh)
        _sdt_run!(backend, _sdh_b_phys!, nw, (T.T22.POS, TH), (tg(t -> t.T22.POS), SEED))
        seobnr_hm_device_tables!(T, nothing; f_min = f_min, TT = TT, dirpar_nodes = dirpar, timers = ttm, tkw...)
        _sdt_run!(backend, _sdh_b_tshift!, nw, (TS, T.HS, T.HT, T.FT, T.T22.BLD, T.T22.B, lik),
            (TST, tg(t -> t.HS), tg(t -> t.HT), tg(t -> t.FT), tg(t -> t.T22.BLD), tg(t -> t.T22.B), nothing))
        fphi > 0 && _sdt_run!(backend, _sdh_b_pshift!, nw, (PO, T.HS, T.HT, T.FT, T.T22.BLD, T.T22.B, lik, fphi),
            (POT, tg(t -> t.HS), tg(t -> t.HT), tg(t -> t.FT), tg(t -> t.T22.BLD), tg(t -> t.T22.B), nothing, nothing))
        return nothing
    end
    # the log-likelihood alone (`value_only`): the same stages without forward tangents (their rows of `sumsh` are 0)
    function setup_value!(posh)
        copyto!(TH, posh)
        _sdt_run!(backend, _sdh_b_phys!, nw, (T.T22.POS, TH), nothing)
        seobnr_hm_device_tables!(T, nothing; f_min = f_min, timers = ttm)
        _sdt_run!(backend, _sdh_b_tshift!, nw, (TS, T.HS, T.HT, T.FT, T.T22.BLD, T.T22.B, lik), nothing)
        fphi > 0 && _sdt_run!(backend, _sdh_b_pshift!, nw, (PO, T.HS, T.HT, T.FT, T.T22.BLD, T.T22.B, lik, fphi), nothing)
        return nothing
    end
    # (c) by GEMMs, walker chunk by chunk, into the reduced rows R, RQ (and RT)
    function sums_gemm!(EXTd, grad)
        (; wc, XA, GH, GHT, GL, GMA, GM, GMT, GMv) = gb
        t0 = Ref(time_ns())
        lap!(k) = ttm === nothing || (KernelAbstractions.synchronize(backend); t = time_ns();
            ttm[k] = get(ttm, k, 0.0) + (t - t0[]) / 1e9; t0[] = t)
        for w0 in 0:wc:(nw - 1)
            if grad
                _sdt_run!(backend, _sdh_b_modes!, (nb, wc), (GH, w0, nw, PO, F1, T.HS, T.HT, T.FT, T.T22.BLD, T.T22.B),
                    (GHT, nothing, nothing, fphi > 0 ? POT : nothing, nothing, tg(t -> t.HS), tg(t -> t.HT),
                     tg(t -> t.FT), tg(t -> t.T22.BLD), tg(t -> t.T22.B)); dirpar = dirpar)
            else
                _sdt_run!(backend, _sdh_b_modes!, (nb, wc), (GH, w0, nw, PO, F1, T.HS, T.HT, T.FT, T.T22.BLD, T.T22.B),
                    nothing)
            end
            lap!(:gemm_modes)
            _sdt_run!(backend, _sdh_b_lhs!, (nb, wc), (GL, GH, w0, nw, EXTd, TS, F1, W, CDr, CDi, dv), nothing)
            lap!(:gemm_lhs)
            if grad
                GPU.batched_adj_mul_mid!(GMA, GL, XA)
            else
                GPU.batched_adj_mul_mid!(GMv, GL, GH)
            end
            lap!(:gemm_mul)
            _sdt_run!(backend, _sdh_b_combine!, wc, (R, RQ, grad ? RT : nothing, grad ? GM : GMv, grad ? GMT : nothing,
                w0, nw, EXTd, dv), nothing)
            lap!(:gemm_comb)
        end
        return nothing
    end
    function sums!(sumsh, EXTd; grad::Bool = true)
        sums_gemm!(EXTd, grad)
        copyto!(Rh, R); copyto!(RQh, RQ); copyto!(TSh, TS)
        if grad
            foreach(k -> copyto!(RTh[k], RT[k]), 1:4)
            foreach(k -> copyto!(TSTh[k], TST[k]), 1:4)
        else
            foreach(k -> fill!(RTh[k], 0.0), 1:4); foreach(k -> fill!(TSTh[k], 0.0), 1:4)
        end
        hs = Array(T.HS)
        @inbounds for w in 1:nw
            dead[w] = !hs[w].success | (hs[w].fmax[1] < fref)
            for d in 1:D
                o = (d - 1) * NV
                for r in 1:3
                    sumsh[o + r, w] = Rh[r, 1, d, w]
                end
                for r in 1:_SDH_NQ
                    sumsh[o + 3 + r, w] = RQh[r, 1, d, w]
                end
                for j in 1:NT
                    sumsh[o + 5 + 4NM + j, w] = RTh[j][1, 1, d, w]
                    sumsh[o + 5 + 4NM + NT + 2j - 1, w] = RTh[j][2, 1, d, w]
                    sumsh[o + 5 + 4NM + NT + 2j, w] = RTh[j][3, 1, d, w]
                end
            end
            sumsh[NL + 1, w] = TSh[w]
            for j in 1:NT
                sumsh[NL + 1 + j, w] = TSTh[j][w]
            end
        end
        return nothing
    end
    sums_value!(sumsh, EXTd) = sums!(sumsh, EXTd; grad = false)
    return (; setup!, sums!, setup_value!, sums_value!, dead)
end

"""
    seobnrhm_binparallel_logp_and_grad(data, cfg, freqs; backend, nwalkers, f_min = 20.0, dirpar_nodes = false,
                                       timers = nothing) -> logp_and_grad!

SEOBNRv5HM (modes (2,2), (2,1), (3,3), (3,2), (4,4), (4,3)) log-likelihood and gradient with the tables and the
frequency-bin sums on `backend` (`CUDABackend()`, or `KernelAbstractions.CPU()` for testing), the mode factors and the
final combination on the host (see the header of `SEOBNRHMBinParallel.jl`). 11 sampled rows as the IMR-table models;
distance marginalization, f_ref timing and the time reference as IMRPhenomXHM's; all detectors must share one
frequency grid. The closure takes HOST `grad_out`, `logp_out`, `positions` and returns the log-likelihood plus the
package's box prior, untempered. `timers`: optional `Vector{Float64}` of length 4 ((a) tables, (b) mode factors, (c) bin
sums, (d) surrogate seconds; synchronizes); `table_timers`: optional `Dict{Symbol,Float64}` of the table stages' seconds.
"""
seobnrhm_binparallel_logp_and_grad(data, cfg::PEConfig, freqs; f_min = 20.0, dirpar_nodes = false, kw...) =
    _mbp_logp_and_grad(_SEOBHMSpec(), data, cfg, freqs; f_min = f_min, dirpar_nodes = dirpar_nodes, kw...)
