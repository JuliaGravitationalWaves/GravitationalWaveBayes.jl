# ------------------------------------------------------------------------------------------------
# SEOBNRv5 IMR log-likelihood and gradient on a KernelAbstractions device from the device tables of
# `SEOBNRDeviceTable.jl`. Included inside `BatchedNUTS`.
#
# Same factorisation as the XAS bin-parallel path. With H0(f) = A(f) e^{-i p0(f)} the table's waveform at unit
# distance (amplitude taper included) and tau_d the time applied in detector d, the CPU kernels'
# (`_seobv5f2_ll_given_coeffs[_multi]`: phase_val = -(p0 - tshift f - phiRef), tshift = -2 pi tau_d) strain is
#     h_d(f) = combo_d e^{i phase} e^{-2 pi i tau_d f} H0(f) / dL,
# so with w = 1/S (0 where S = 0) and, over the table band,
#     P_d = sum w |H0|^2,      Q_d = sum w conj(d) H0 e^{-2 pi i tau_d f},
# <h,h>_d = 4 df |combo_d|^2 P_d / dL^2 and <d,h>_d = 4 df combo_d e^{i phase} Q_d / dL. One call:
#   sampled rows -> table inputs (walker) -> device tables -> tau_d (walker) -> chunked bin sums (chunk x walker)
#   -> reduction -> log-likelihood from the sums: `_marg_combine`, dead-zone reject (walker).
#
# Gradient (`grad = true`): the four intrinsic rows (mc, q, chi1, chi2) by forward mode through every stage above
# (4 tangent directions, `seobnr_device_tables!(...; TT)`; tau_d's intrinsic part is the f_ref time shift), so the
# reduced sums arrive with their tangents dP, dQ. Per walker, `_marg_combine`'s hand-written gradient plus an
# in-kernel reverse pass over the small loop-free `_sdt_pre` give dlogL/d(extrinsics) at fixed sums and the sums'
# cotangents; dQ_d/dtau_d (accumulated with the sums) carries those into t_c, ra and sin(dec) through tau_d, and
# the intrinsic rows are the cotangents contracted with the forward tangents.
# ------------------------------------------------------------------------------------------------

const _SDT_GMSUN_C2_GPC = GravitationalWaveforms.GMsun_over_c2_Gpc
const _SDT_NSUM = 5          # rows of the bin sums: P, Re Q, Im Q, Re dQ/dtau, Im dQ/dtau

# `KernelFused._seobv5f2_time_shift` for a device table: its `_spa_fmax`/`_spa_t0` dispatch on the
# `EOBIMRTable` type, so a NamedTuple table must spell out the IMR choices (fspa_max, t_attach).
@inline function _sdt_time_shift(tab, lik)
    (lik.fref > 0 && tab.success) || return 0.0
    fr = clamp(lik.fref, tab.fmin, tab.fspa_max)
    r = _SDT_W._seobv5f2_dense_to_freq(tab.ts, tab.phis, tab.omegas, tab.nreal, tab.M_sec, fr)
    return (r.t_star - tab.t_a) * tab.M_sec + lik.toff
end

# Sampled rows -> table inputs (chirp mass, eta, chi1, chi2), as `_eobimr_walker_ll[_multi]`.
@inline function _sdt_b_phys!(w, POS4, POS)
    @inbounds begin
        m1, m2, _, _ = _eobimr_physical(POS[1, w], POS[2, w], POS[9, w], POS[10, w])
        mcc, eta = KernelFused._chirp_mass_eta(m1, m2)
        POS4[1, w] = mcc; POS4[2, w] = eta; POS4[3, w] = POS[3, w]; POS4[4, w] = POS[4, w]
    end
    return nothing
end

# The extrinsic part of tau_d: t_c plus the geometric delay relative to the reference detector.
@inline function _sdt_tau_ext(d, t_c, ra, sdec, g)
    dec = asin(_clamp_unit(sdec))
    dt_ref = ifelse(g.geo & (g.tref > 0), time_delay_from_geocenter_gmst(g.dets[max(g.tref, 1)], ra, dec, g.gmst), 0.0)
    return t_c + ifelse(g.geo, time_delay_from_geocenter_gmst(g.dets[d], ra, dec, g.gmst) - dt_ref, 0.0)
end

# tau_d = (extrinsic part) - (f_ref time shift of the table).
@inline function _sdt_b_tau!(w, TAU, POS, TABS, TB, g, lik, ::Val{D}) where {D}
    @inbounds begin
        sh = _sdt_time_shift(_sdt_table(TB, w, TABS[w]), lik)
        for d in 1:D
            TAU[d, w] = _sdt_tau_ext(d, POS[6, w], POS[8, w], POS[9, w], g) - sh
        end
    end
    return nothing
end

# Partial sums over one chunk of bins of the frequency grid shared by all detectors: P_d, Re/Im Q_d, and
# Re/Im dQ_d/dtau_d (= sum w conj(d) H0 (-2 pi i f) e^{-2 pi i tau_d f}).
@inline function _sdt_b_sums!(c, w, part, TAU, F, WT, CDr, CDi, TABS, TB, taper, ch, ::Val{D}) where {D}
    @inbounds begin
        s = TABS[w]
        tab = _sdt_table(TB, w, s)
        nb = length(F)
        amp_scale = s.M * _SDT_GMSUN_C2_GPC
        for d in 1:D, r in 1:_SDT_NSUM
            part[r, c, d, w] = 0.0
        end
        for j in ((c - 1) * ch + 1):min(c * ch, nb)
            f = F[j]
            if s.success && s.fmin <= f <= s.fmax
                a0, p0 = _SDT_W._eobimr_eval(tab, f)
                A = a0 * amp_scale * KernelFused._end_taper(f, s.fmax, taper)
                hr = A * cos(p0); hi = -A * sin(p0)
                h2 = hr * hr + hi * hi
                for d in 1:D
                    wt = WT[j, d]
                    th = -2pi * TAU[d, w] * f
                    ct, st = cos(th), sin(th)
                    xr = CDr[j, d] * hr + CDi[j, d] * hi
                    xi = CDr[j, d] * hi - CDi[j, d] * hr
                    qr = wt * (xr * ct - xi * st)
                    qi = wt * (xr * st + xi * ct)
                    part[1, c, d, w] += wt * h2
                    part[2, c, d, w] += qr
                    part[3, c, d, w] += qi
                    part[4, c, d, w] += 2pi * f * qi
                    part[5, c, d, w] -= 2pi * f * qr
                end
            end
        end
    end
    return nothing
end

# The `_marg_combine` inputs (ll_plain, <h,h>, Re/Im <d,h>) from the extrinsic rows and the sums.
@inline function _sdt_pre(dL, phase, ra, sdec, cth, psi, P::NTuple{D}, Qr::NTuple{D}, Qi::NTuple{D}, g) where {D}
    dec = asin(_clamp_unit(sdec))
    th = acos(_clamp_unit(cth))
    pfac = 0.5 * (1 + cos(th) * cos(th))
    cfac = cos(th)
    ep = Complex(cos(phase), sin(phase)) / dL
    ll_plain = 0.0; hh = 0.0; zr = 0.0; zi = 0.0
    for d in 1:D
        Fp, Fc = antenna_response_gmst(g.dets[d], ra, dec, psi, g.gmst)
        combo = Fp * pfac - im * Fc * cfac
        z = combo * ep * Complex(Qr[d], Qi[d])
        h2 = abs2(combo) * P[d] / (dL * dL)
        ll_plain += g.dd[d] - 2 * g.df[d] * (h2 - 2 * real(z))
        hh += 4 * g.df[d] * h2
        zr += 4 * g.df[d] * real(z)
        zi += 4 * g.df[d] * imag(z)
    end
    return ll_plain, hh, zr, zi
end

@inline function _sdt_ll_from_sums(dL, phase, ra, sdec, cth, psi, P, Qr, Qi, g, lik, success, fmax, f1)
    ll = KernelFused._marg_combine(_sdt_pre(dL, phase, ra, sdec, cth, psi, P, Qr, Qi, g)..., dL, lik)
    dead = !success | (fmax < f1)
    return ifelse(dead, KernelFused._SEOBV5F2_DEADZONE_LOGLIKELIHOOD, ll)
end

# Seeded stores for the in-kernel reverse passes (Const{Nothing} return).
@inline function _sdt_pre_seed!(out, w, dL, phase, ra, sdec, cth, psi, P, Qr, Qi, g, s)
    a, b, c, d = _sdt_pre(dL, phase, ra, sdec, cth, psi, P, Qr, Qi, g)
    @inbounds out[w] = s[1] * a + s[2] * b + s[3] * c + s[4] * d
    return nothing
end
@inline function _sdt_tau_seed!(out, w, t_c, ra, sdec, g, bt::NTuple{D}) where {D}
    acc = 0.0
    for d in 1:D
        acc += bt[d] * _sdt_tau_ext(d, t_c, ra, sdec, g)
    end
    @inbounds out[w] = acc
    return nothing
end

@kernel function _sdt_ll_kernel!(LL, @Const(POS), @Const(RED), @Const(TABS), g, lik, f1, ::Val{D}) where {D}
    w = @index(Global, Linear)
    @inbounds begin
        s = TABS[w]
        P = ntuple(d -> RED[1, 1, d, w], Val(D))
        Qr = ntuple(d -> RED[2, 1, d, w], Val(D))
        Qi = ntuple(d -> RED[3, 1, d, w], Val(D))
        LL[w] = _sdt_ll_from_sums(POS[5, w], POS[7, w], POS[8, w], POS[9, w], POS[10, w], POS[11, w], P, Qr, Qi,
            g, lik, s.success, s.fmax, f1)
    end
end

# Log-likelihood and its gradient for walker w (rows: mc, q, chi1, chi2, dL, t_c, phase, ra, sin dec, cos th, psi).
@kernel function _sdt_grad_kernel!(LL, G, SC, dSC, @Const(POS), @Const(RED), RT, @Const(TABS), g, lik, f1, ::Val{D},
        gpu) where {D}
    w = @index(Global, Linear)
    @inbounds begin
        s = TABS[w]
        dL, t_c, phase, ra, sdec, cth, psi = POS[5, w], POS[6, w], POS[7, w], POS[8, w], POS[9, w], POS[10, w], POS[11, w]
        P = ntuple(d -> RED[1, 1, d, w], Val(D))
        Qr = ntuple(d -> RED[2, 1, d, w], Val(D))
        Qi = ntuple(d -> RED[3, 1, d, w], Val(D))
        llp, hh, zr, zi = _sdt_pre(dL, phase, ra, sdec, cth, psi, P, Qr, Qi, g)
        val, gl, gh, gzr, gzi, gdL = _marg_combine_grad(llp, hh, zr, zi, dL, lik)
        dSC[w] = 1.0
        gr = _xasbp_ad(gpu, Enzyme.Reverse, Enzyme.Const(_sdt_pre_seed!), Enzyme.Const{Nothing},
            Enzyme.Duplicated(SC, dSC), Enzyme.Const(w), Enzyme.Active(dL), Enzyme.Active(phase), Enzyme.Active(ra),
            Enzyme.Active(sdec), Enzyme.Active(cth), Enzyme.Active(psi), Enzyme.Active(P), Enzyme.Active(Qr),
            Enzyme.Active(Qi), Enzyme.Const(g), Enzyme.Const((gl, gh, gzr, gzi)))[1]
        bP = gr[9]; bQr = gr[10]; bQi = gr[11]
        bt = ntuple(d -> bQr[d] * RED[4, 1, d, w] + bQi[d] * RED[5, 1, d, w], Val(D))
        dSC[w] = 1.0
        gt = _xasbp_ad(gpu, Enzyme.Reverse, Enzyme.Const(_sdt_tau_seed!), Enzyme.Const{Nothing},
            Enzyme.Duplicated(SC, dSC), Enzyme.Const(w), Enzyme.Active(t_c), Enzyme.Active(ra), Enzyme.Active(sdec),
            Enzyme.Const(g), Enzyme.Const(bt))[1]
        gint = ntuple(Val(4)) do k
            R = RT[k]
            acc = 0.0
            for d in 1:D
                acc += bP[d] * R[1, 1, d, w] + bQr[d] * R[2, 1, d, w] + bQi[d] * R[3, 1, d, w]
            end
            acc
        end
        dead = !s.success | (s.fmax < f1)
        LL[w] = ifelse(dead, KernelFused._SEOBV5F2_DEADZONE_LOGLIKELIHOOD, val)
        z = ifelse(dead, 0.0, 1.0)
        G[1, w] = z * gint[1]; G[2, w] = z * gint[2]; G[3, w] = z * gint[3]; G[4, w] = z * gint[4]
        G[5, w] = z * (gr[3] + gdL)
        G[6, w] = z * gt[3]
        G[7, w] = z * gr[4]
        G[8, w] = z * (gr[5] + gt[4])
        G[9, w] = z * (gr[6] + gt[5])
        G[10, w] = z * gr[7]
        G[11, w] = z * gr[8]
    end
end

"""
    SEOBNRDeviceLikelihood(backend, data, cfg; nwalkers, chunk = 16, f_min = 20.0, grad = true)

Device buffers and data for [`seobnr_device_logp!`](@ref) / [`seobnr_device_logp_and_grad!`](@ref): the SEOBNRv5
IMR log-likelihood (and gradient, if `grad`) of `nwalkers` walkers (the 11 sampled rows of the IMR-table models)
on a KernelAbstractions `backend`; all detectors must share one frequency grid.
"""
struct SEOBNRDeviceLikelihood{TT,TG,A1,A2,A4,G,L}
    T::TT
    TT::TG                       # nothing, or 4 tangent tables
    D::Int; nb::Int; nch::Int; ch::Int
    F::A1; WT::A2; CDr::A2; CDi::A2
    POS::A2; SEED::NTuple{4,A2}; TAU::A2; TAUT::NTuple{4,A2}
    part::A4; partT::NTuple{4,A4}; RED::A4; REDT::NTuple{4,A4}
    LL::A1; G::A2; SC::A1; dSC::A1
    g::G; lik::L; f1::Float64; taper::Float64; f_min::Float64
end

function SEOBNRDeviceLikelihood(backend, data, cfg::PEConfig; nwalkers::Int, chunk::Int = 16, f_min = 20.0, grad = true)
    dets = data isa AbstractVector ? data : [data]
    D = length(dets)
    F0 = dets[1].freqs
    all(d -> d.freqs == F0, dets) || throw(ArgumentError("SEOBNRDeviceLikelihood needs one frequency grid for all detectors"))
    nb = length(F0)
    nw = nwalkers
    dev(a) = (b = KernelAbstractions.allocate(backend, Float64, size(a)); copyto!(b, a); b)
    wt = [d.psd[i] == 0 ? 0.0 : 1.0 / d.psd[i] for i in 1:nb, d in dets]
    cdr = [Float64(real(d.strain[i])) for i in 1:nb, d in dets]
    cdi = [Float64(imag(d.strain[i])) for i in 1:nb, d in dets]
    nch = cld(nb, chunk)
    z(dims...) = KernelAbstractions.zeros(backend, Float64, dims...)
    g = _xasbp_geometry(dets, cfg)
    # forward seeds: direction k is sampled row k (mc, q, chi1, chi2)
    seed = ntuple(k -> (a = zeros(11, nw); a[k, :] .= 1.0; dev(a)), 4)
    TT = grad ? ntuple(_ -> SEOBNRDeviceTables(backend, nw), 4) : nothing
    ng = grad ? 1 : 0
    return SEOBNRDeviceLikelihood(SEOBNRDeviceTables(backend, nw), TT, D, nb, nch, chunk,
        dev(Float64.(collect(F0))), dev(wt), dev(cdr), dev(cdi), z(11, nw), seed, z(D, nw),
        ntuple(_ -> z(D, nw * ng), 4), z(_SDT_NSUM, nch, D, nw), ntuple(_ -> z(_SDT_NSUM, nch, D, nw * ng), 4),
        z(_SDT_NSUM, 1, D, nw), ntuple(_ -> z(_SDT_NSUM, 1, D, nw * ng), 4), z(nw), z(11, nw), z(nw), z(nw),
        g, KernelFused._lik_opts(cfg), Float64(F0[1]), Float64(cfg.fmax_taper), Float64(f_min))
end

function _sdt_logp_stages!(L::SEOBNRDeviceLikelihood, pos, grad::Bool; timers = nothing, table_kw...)
    T = L.T; be = T.backend; nw = T.nw
    copyto!(L.POS, pos)
    TT = grad ? L.TT : nothing
    sh(xs...) = grad ? xs : nothing
    _sdt_run!(be, _sdt_b_phys!, nw, (T.POS, L.POS), grad ? (ntuple(k -> TT[k].POS, 4), L.SEED) : nothing)
    seobnr_device_tables!(T, nothing; f_min = L.f_min, timers = timers, TT = TT, table_kw...)
    t0 = time_ns()
    TABS = grad ? ntuple(k -> TT[k].TABS, 4) : nothing
    TB = grad ? ntuple(k -> TT[k].TB, 4) : nothing
    _sdt_run!(be, _sdt_b_tau!, nw, (L.TAU, L.POS, T.TABS, T.TB, L.g, L.lik, Val(L.D)),
        sh(L.TAUT, L.SEED, TABS, TB, nothing, nothing, nothing))
    _sdt_run!(be, _sdt_b_sums!, (L.nch, nw), (L.part, L.TAU, L.F, L.WT, L.CDr, L.CDi, T.TABS, T.TB, L.taper, L.ch, Val(L.D)),
        sh(L.partT, L.TAUT, nothing, nothing, nothing, nothing, TABS, TB, nothing, nothing, nothing);
        dirpar = get(get(table_kw, :dirpar, (;)), :nodes, false))
    sum!(L.RED, L.part)
    if grad
        for k in 1:4
            sum!(L.REDT[k], L.partT[k])
        end
    end
    return t0
end

"""
    seobnr_device_logp!(L::SEOBNRDeviceLikelihood, pos; timers = nothing) -> device vector of log-likelihoods

`pos`: 11 x nwalkers (mc, q, chi1, chi2, dL, t_c, phase, ra, sin dec, cos theta_jn, psi), host or device.
`timers`, if a `Dict{Symbol,Float64}`, accumulates per-stage seconds (synchronizing); other keywords go to
[`seobnr_device_tables!`](@ref).
"""
function seobnr_device_logp!(L::SEOBNRDeviceLikelihood, pos; timers = nothing, table_kw...)
    be = L.T.backend
    t0 = _sdt_logp_stages!(L, pos, false; timers = timers, table_kw...)
    _sdt_ll_kernel!(be)(L.LL, L.POS, L.RED, L.T.TABS, L.g, L.lik, L.f1, Val(L.D); ndrange = L.T.nw)
    KernelAbstractions.synchronize(be)
    timers === nothing || (timers[:likelihood] = get(timers, :likelihood, 0.0) + (time_ns() - t0) / 1e9)
    return L.LL
end

"""
    seobnr_device_logp_and_grad!(L::SEOBNRDeviceLikelihood, pos; timers = nothing) -> (LL, G) device arrays

Log-likelihood (nwalkers) and its gradient (11 x nwalkers) with respect to the sampled rows. `L` must have been
built with `grad = true`.
"""
function seobnr_device_logp_and_grad!(L::SEOBNRDeviceLikelihood, pos; timers = nothing, table_kw...)
    L.TT === nothing && throw(ArgumentError("SEOBNRDeviceLikelihood built with grad = false"))
    be = L.T.backend
    t0 = _sdt_logp_stages!(L, pos, true; timers = timers, table_kw...)
    gpu = Val(!(be isa KernelAbstractions.CPU))
    _sdt_grad_kernel!(be)(L.LL, L.G, L.SC, L.dSC, L.POS, L.RED, L.REDT, L.T.TABS, L.g, L.lik, L.f1, Val(L.D), gpu;
        ndrange = L.T.nw)
    KernelAbstractions.synchronize(be)
    timers === nothing || (timers[:likelihood] = get(timers, :likelihood, 0.0) + (time_ns() - t0) / 1e9)
    return L.LL, L.G
end

"""
    seobnr_device_closures(data, cfg; backend, nwalkers, chunk = 16) -> (logp_and_grad!, loglik!)

Host-array closures around one [`SEOBNRDeviceLikelihood`](@ref) (SEOBNRv5 IMR, `nwalkers` walkers), for the samplers:
`logp_and_grad!(grad, logp, positions, mask)` = log-likelihood plus the package's box prior (0 inside, `-Inf` with zero
gradient outside), like [`gw_logp_and_grad_gpu_staged`](@ref)'s other models; `loglik!(ll, positions)` = the untempered
log-likelihood (SMC reweighting; `-Inf` outside the box). Both evaluate all `nwalkers` columns on the device.
"""
function seobnr_device_closures(data, cfg::PEConfig{T}; backend, nwalkers::Int, chunk::Int = 16) where {T}
    L = SEOBNRDeviceLikelihood(backend, data, cfg; nwalkers = nwalkers, chunk = chunk)
    gh = Matrix{Float64}(undef, 11, nwalkers); llh = Vector{Float64}(undef, nwalkers)
    chk(positions) = size(positions) == (11, nwalkers) ||
        throw(ArgumentError("built for 11 x $nwalkers positions, got $(size(positions))"))
    lpg! = function (grad_out::AbstractMatrix{T}, logp_out::AbstractVector{T}, positions::AbstractMatrix{T}, active_mask)
        chk(positions)
        LL, G = seobnr_device_logp_and_grad!(L, positions)
        copyto!(gh, G); copyto!(llh, LL)
        box = _call_log_prior_vec(positions)
        @inbounds for w in 1:nwalkers
            if isfinite(box[w])
                logp_out[w] = llh[w] + box[w]
                for k in 1:11
                    grad_out[k, w] = gh[k, w]
                end
            else
                logp_out[w] = T(-Inf)
                for k in 1:11
                    grad_out[k, w] = zero(T)
                end
            end
        end
        return nothing
    end
    ll! = function (ll_out::AbstractVector, positions::AbstractMatrix)
        chk(positions)
        copyto!(llh, seobnr_device_logp!(L, positions))
        box = _call_log_prior_vec(positions)
        @inbounds for w in 1:nwalkers
            ll_out[w] = isfinite(box[w]) ? llh[w] : -Inf
        end
        return ll_out
    end
    return lpg!, ll!
end
