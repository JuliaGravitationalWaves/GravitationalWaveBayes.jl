# ------------------------------------------------------------------------------------------------
# IMRPhenomXAS likelihood + gradient, parallel over frequency bins. Included inside
# `BatchedNUTS`. Runs on any KernelAbstractions backend; written for CUDA, where Enzyme through the XAS
# kernel launch fails (tape contains `Any`) and one thread per walker is 40-140x slower than the CPU.
#
# Factorisation. With H0(f) = A(f) e^{i Phi(f)} the intrinsic waveform (zero above XAS's cutoff) and
# tau_d the time applied in detector d, the dense kernels' strain is
#     h_d(f) = combo_d e^{-i phase} e^{-2 pi i tau_d f} H0(f),
# so with w = 1/S (0 where S = 0) and the per-detector sums
#     P_d = sum w |H0|^2,      Q_d = sum w conj(d) H0 e^{-2 pi i tau_d f},
# `<h,h>` = 4 df |combo_d|^2 P_d and `<d,h>` = 4 df combo_d e^{-i phase} Q_d. One call:
#   (a) host, per walker: flat coefficient tuple K (phase + amplitude) and tau_d        [_xasbp_coeffs_tau]
#   (b) device: partial sums of (P_d, Re Q_d, Im Q_d) per bin chunk, reduced           [_xasbp_sums_kernel!]
#   (c) host, per walker: Enzyme on logL(x, P, Q) -> logL, dL/dx at fixed sums, bar_P, bar_Q
#   (d) device: per bin, bar_A / bar_Phi in closed form, then one in-kernel Enzyme VJP of (Phi, A)(K, f)
#       (no loop inside the AD call: no device-heap caches) -> bar_K, bar_tau_d, reduced   [_xasbp_adjoint_kernel!]
#   (e) host, per walker: Enzyme VJP of (K(x), tau(x)) with (bar_K, bar_tau), added to dL/dx.
# ------------------------------------------------------------------------------------------------

const _XASBP_NK = 61

# Flat coefficient tuple: everything `_xas_phase_amp` reads, in a fixed order (`_xas_phase_amp_flat`).
# A literal tuple, not `(_xas_flat_phase(pc)..., _xas_flat_amp(ac)...)`: Enzyme's forward pass of the CUDA coefficient
# kernels asserts on the splat concatenation (`invertPointerM` on a store). The halves below must list the
# same fields in the same order.
@inline function _xas_flat(pc, ac)
    p = pc.phi_ins; b = pc.phi_int; c = pc.phi_mrd
    ai = ac.amp_ins; an = ac.amp_int; ar = ac.amp_rd
    return (pc.M * pc.GMsun_over_c3, pc.f1, pc.f2, pc.fcutPar, pc.eta, pc.fdamp, pc.fRD,                  # 1-7
            p.phi0, p.phi1, p.phi2, p.phi3, p.phi4, p.phi5, p.phi5L, p.phi6, p.phi6L,                     # 8-16
            p.phi7, p.phi8, p.phi8L, p.sigma1, p.sigma2, p.sigma3, p.sigma4, p.phiN,                      # 17-24
            b.b0, b.b1, b.b2, b.b3, b.b4, b.alpha0, b.alpha1,                                             # 25-31
            c.c0, c.c1, c.c2, c.c4, c.cL, c.beta0, c.beta1,                                               # 32-38
            ac.M * ac.GMsun_over_c3, ac.fA1, ac.fA4, ac.fcutPar, ac.fdamp, ac.fRD,                        # 39-44
            ai.A2, ai.A3, ai.A4, ai.A5, ai.A6, ai.rho1, ai.rho2, ai.rho3,                                 # 45-52
            an.delta0, an.delta1, an.delta2, an.delta3, an.delta4,                                        # 53-57
            ar.gamma1, ar.gamma2, ar.gamma3,                                                              # 58-60
            ac.Overallamp * ac.amp0)                                                                      # 61
end

# `_xas_flat` read lazily: k[j] (a literal j at every `_xas_pa_core` use) evaluates entry j of the tuple
# above at its use, so with lazy setup views (`_XhmFV`) the 61 coefficients and their forward tangents are not all
# loaded into registers up front (XPHM 223's (2,2) mode kernel: 255 registers, 2.1 KB stack at 7 tangents).
struct _XasFlatLazy{P,A}
    pc::P
    ac::A
end
const _XAS_FLAT_EXPRS = let p = :(pc.phi_ins), b = :(pc.phi_int), c = :(pc.phi_mrd), ai = :(ac.amp_ins),
                            an = :(ac.amp_int), ar = :(ac.amp_rd)
    fl(o, s) = Expr(:., o, QuoteNode(s))
    Any[:(pc.M * pc.GMsun_over_c3), :(pc.f1), :(pc.f2), :(pc.fcutPar), :(pc.eta), :(pc.fdamp), :(pc.fRD),
        (fl(p, s) for s in (:phi0, :phi1, :phi2, :phi3, :phi4, :phi5, :phi5L, :phi6, :phi6L, :phi7, :phi8, :phi8L,
                            :sigma1, :sigma2, :sigma3, :sigma4, :phiN))...,
        (fl(b, s) for s in (:b0, :b1, :b2, :b3, :b4, :alpha0, :alpha1))...,
        (fl(c, s) for s in (:c0, :c1, :c2, :c4, :cL, :beta0, :beta1))...,
        :(ac.M * ac.GMsun_over_c3), :(ac.fA1), :(ac.fA4), :(ac.fcutPar), :(ac.fdamp), :(ac.fRD),
        (fl(ai, s) for s in (:A2, :A3, :A4, :A5, :A6, :rho1, :rho2, :rho3))...,
        (fl(an, s) for s in (:delta0, :delta1, :delta2, :delta3, :delta4))...,
        (fl(ar, s) for s in (:gamma1, :gamma2, :gamma3))...,
        :(ac.Overallamp * ac.amp0)]
end
length(_XAS_FLAT_EXPRS) == 61 || error("_XAS_FLAT_EXPRS: $(length(_XAS_FLAT_EXPRS)) entries")
@eval @inline function _xas_flat_entry(pc, ac, i::Int)
    $(foldr((j, rest) -> :(i == $j ? $(_XAS_FLAT_EXPRS[j]) : $rest), 1:61; init = :(throw(BoundsError()))))
end
@inline Base.getindex(k::_XasFlatLazy, i::Int) = _xas_flat_entry(getfield(k, :pc), getfield(k, :ac), i)

const _XASBP_NKP = 38                               # rows 1:38 phase (from pc), 39:61 amplitude (from ac)

@inline function _xas_flat_phase(pc)
    p = pc.phi_ins; b = pc.phi_int; c = pc.phi_mrd
    return (pc.M * pc.GMsun_over_c3, pc.f1, pc.f2, pc.fcutPar, pc.eta, pc.fdamp, pc.fRD,                  # 1-7
            p.phi0, p.phi1, p.phi2, p.phi3, p.phi4, p.phi5, p.phi5L, p.phi6, p.phi6L,                     # 8-16
            p.phi7, p.phi8, p.phi8L, p.sigma1, p.sigma2, p.sigma3, p.sigma4, p.phiN,                      # 17-24
            b.b0, b.b1, b.b2, b.b3, b.b4, b.alpha0, b.alpha1,                                             # 25-31
            c.c0, c.c1, c.c2, c.c4, c.cL, c.beta0, c.beta1)                                               # 32-38
end

@inline function _xas_flat_amp(ac)
    ai = ac.amp_ins; an = ac.amp_int; ar = ac.amp_rd
    return (ac.M * ac.GMsun_over_c3, ac.fA1, ac.fA4, ac.fcutPar, ac.fdamp, ac.fRD,                        # 39-44
            ai.A2, ai.A3, ai.A4, ai.A5, ai.A6, ai.rho1, ai.rho2, ai.rho3,                                 # 45-52
            an.delta0, an.delta1, an.delta2, an.delta3, an.delta4,                                        # 53-57
            ar.gamma1, ar.gamma2, ar.gamma3,                                                              # 58-60
            ac.Overallamp * ac.amp0)                                                                      # 61
end

# `KernelFused._xas_phase_amp(pc, ac, f)` from the flat tuple (same operations, same order).
@inline function _xas_phase_amp_flat(k, f)
    fs = max(f, eps(typeof(f)))
    x = k[1] * fs
    phis = if x < k[2]
        KernelFused._xas_phi_ins_val(x, k[8], k[9], k[10], k[11], k[12], k[13], k[14], k[15], k[16],
                                     k[17], k[18], k[19], k[20], k[21], k[22], k[23], k[24])
    elseif x < k[3]
        KernelFused._xas_phi_int_val(x, k[25], k[26], k[27], k[28], k[29], k[36], k[6], k[7]) + k[31] * x + k[30]
    elseif x < k[4]
        KernelFused._xas_phi_mrd_val(x, k[32], k[33], k[34], k[35], k[36], k[6], k[7]) + k[37] + k[38] * x
    else
        zero(x)
    end / k[5]
    xa = k[39] * fs
    aimr = if xa < k[40]
        KernelFused._xas_amp_ins_val(xa, k[45], k[46], k[47], k[48], k[49], k[50], k[51], k[52])
    elseif xa < k[41]
        xa^(7.0 / 6.0) / (k[53] + xa * (k[54] + xa * (k[55] + xa * (k[56] + xa * k[57]))))
    elseif xa < k[42]
        KernelFused._xas_amp_rd_val(xa, k[58], k[59], k[60], k[43], k[44])
    else
        zero(xa)
    end
    return phis, k[61] * (xa^(-7.0 / 6.0)) * aimr
end

# Detector geometry and data constants for the host-side functions. `geo = false`: one detector, t_c is its
# arrival time (the single-detector kernels); `geo = true`: t_c geocentric (`tref = 0`) or at detector `tref`.
function _xasbp_geometry(dets, cfg::PEConfig)
    geo = length(dets) > 1
    dd = Tuple(-2 * (d.freqs[2] - d.freqs[1]) *
               sum(d.psd[i] == 0 ? 0.0 : abs2(d.strain[i]) / d.psd[i] for i in eachindex(d.freqs)) for d in dets)
    return (dets=Tuple(geometry(d.detector) for d in dets),
            gmst=gmst(dets[1].gps_time), geo=geo, tref=geo ? cfg.time_reference : 0,
            df=Tuple(Float64(d.freqs[2] - d.freqs[1]) for d in dets), dd=dd)
end

# (a)/(e): flat coefficients and per-detector times from the SAMPLED parameters.
@inline function _xasbp_coeffs_tau(mc, q, chi1, chi2, dL, t_c, phase, ra, sdec, cth, psi, model, g, lik)
    m1, m2, dec, th = _eobimr_physical(mc, q, sdec, cth)
    mcc, eta = KernelFused._chirp_mass_eta(m1, m2)
    pc = KernelFused._xas_phi_coeffs(model, mcc, eta, chi1, chi2)
    ac = KernelFused._xas_ampl_coeffs(model, mcc, eta, chi1, chi2, dL)
    tc = t_c - KernelFused._xas_time_shift(pc, lik)
    dt_ref = ifelse(g.geo & (g.tref > 0), time_delay_from_geocenter_gmst(g.dets[max(g.tref, 1)], ra, dec, g.gmst), zero(tc))
    tau = ntuple(d -> tc + ifelse(g.geo, time_delay_from_geocenter_gmst(g.dets[d], ra, dec, g.gmst) - dt_ref, zero(tc)),
                 Val(length(g.dets)))
    return _xas_flat(pc, ac), tau
end

# Elementwise tuple sum (a function argument, not a closure over a reassigned local: a captured,
# reassigned variable is boxed, which is a dynamic call in a GPU kernel).
@inline _tadd(a::NTuple{N,Float64}, b::NTuple{N,Float64}) where {N} = ntuple(i -> a[i] + b[i], Val(N))
# Unrolled left fold, same order as `sum` of the tuple. NOT `sum(ntuple(...))`: for N > 32 Base folds with a loop,
# and Enzyme's reverse pass caches that loop in an exponentially growing device-heap buffer whose old blocks are never
# freed -- this is the "coefficient VJP leak" (seen in the kernel IR; same class as the
# searchsortedlast leak).
@generated function _tdot(a::NTuple{N}, b::NTuple{N}) where {N}
    foldl((ex, j) -> :($ex + a[$j] * b[$j]), 2:N; init=:(a[1] * b[1]))
end

function _xasbp_seed(mc, q, chi1, chi2, dL, t_c, phase, ra, sdec, cth, psi, model, g, lik, barK, bartau)
    K, tau = _xasbp_coeffs_tau(mc, q, chi1, chi2, dL, t_c, phase, ra, sdec, cth, psi, model, g, lik)
    return _tdot(K, barK) + _tdot(tau, bartau)
end

# (c): log-likelihood from the sums (the dense kernels' `_marg_combine` inputs).
function _xasbp_ll_from_sums(mc, q, chi1, chi2, dL, t_c, phase, ra, sdec, cth, psi,
                             P::NTuple{D}, Qr::NTuple{D}, Qi::NTuple{D}, g, lik) where {D}
    ll_plain, hh, zr, zi = _xasbp_pre(mc, q, chi1, chi2, dL, t_c, phase, ra, sdec, cth, psi, P, Qr, Qi, g)
    return KernelFused._marg_combine(ll_plain, hh, zr, zi, dL, lik)
end

# The `_marg_combine` inputs (ll_plain, <h,h>, Re/Im <d,h>) from the sampled parameters and the bin sums.
@inline function _xasbp_pre(mc, q, chi1, chi2, dL, t_c, phase, ra, sdec, cth, psi,
                            P::NTuple{D}, Qr::NTuple{D}, Qi::NTuple{D}, g) where {D}
    dec = asin(_clamp_unit(sdec))
    th = acos(_clamp_unit(cth))
    T = typeof(dL)
    pfac = T(0.5) * (1 + cos(th) * cos(th))
    cfac = cos(th)
    ep = Complex(cos(phase), -sin(phase))
    ll_plain = zero(T); hh = zero(T); zr = zero(T); zi = zero(T)
    for d in 1:D
        Fp, Fc = antenna_response_gmst(g.dets[d], ra, dec, psi, g.gmst)
        combo = Fp * pfac - im * Fc * cfac
        z = combo * ep * Complex(Qr[d], Qi[d])
        h2 = abs2(combo) * P[d]
        ll_plain += g.dd[d] - 2 * g.df[d] * (h2 - 2 * real(z))
        hh += 4 * g.df[d] * h2
        zr += 4 * g.df[d] * real(z)
        zi += 4 * g.df[d] * imag(z)
    end
    return ll_plain, hh, zr, zi
end

# ---- Hand-written gradient of `KernelFused._marg_combine` ----------------------------------------
# The in-kernel reverse pass over the 64-node distance quadrature (two log-sum-exp passes with a log-Bessel per node)
# was the most expensive stage of the resident path. Here the gradient of the log-sum-exp is the softmax-weighted
# average of the per-node derivatives, accumulated in the same second pass, plus the chain through the node placement
# (clamp/min/max of ustar +- 12 sigma inside [dL/dhi, dL/dlo]). Same branch choices as the primal; validated against
# Enzyme in test/test_xas_binparallel.jl.

# The distance quadrature's Gauss-Legendre nodes/weights as constants of this module: referenced as
# `KernelFused._GL_X` from here, the GPU compiler could not resolve the (destructured-const) binding.
const _XASBP_NDIST = KernelFused._NDIST
const _XASBP_GL_X = KernelFused._GL_X
const _XASBP_GL_W = KernelFused._GL_W

# d/dx of `KernelFused._log_besseli0` (same polynomial fits, differentiated).
@inline function _dlog_besseli0(x::T) where {T<:Real}
    xs = min(x, T(3.75))
    t = (xs / T(3.75))^2
    s = 1 + t * (T(3.5156229) + t * (T(3.0899424) + t * (T(1.2067492) + t * (T(0.2659732) + t * (T(0.0360768) +
        t * T(0.0045813))))))
    ds = T(3.5156229) + t * (2 * T(3.0899424) + t * (3 * T(1.2067492) + t * (4 * T(0.2659732) + t * (5 * T(0.0360768) +
         t * 6 * T(0.0045813)))))
    dsmall = ds / s * 2 * xs / (T(3.75) * T(3.75))
    xl = max(x, T(3.75))
    u = T(3.75) / xl
    Q = T(0.39894228) + u * (T(0.01328592) + u * (T(0.00225319) + u * (T(-0.00157565) + u * (T(0.00916281) +
        u * (T(-0.02057706) + u * (T(0.02635537) + u * (T(-0.01647633) + u * T(0.00392377))))))))
    dQ = T(0.01328592) + u * (2 * T(0.00225319) + u * (3 * T(-0.00157565) + u * (4 * T(0.00916281) +
         u * (5 * T(-0.02057706) + u * (6 * T(0.02635537) + u * (7 * T(-0.01647633) + u * 8 * T(0.00392377)))))))
    dlarge = 1 - T(0.5) / xl - dQ / Q * T(3.75) / (xl * xl)
    return ifelse(x < T(3.75), dsmall, dlarge)
end

# Tangent bookkeeping over the three node-placement drivers (hh, zeff, dL) as 3-tuples.
@inline _t3(a, b, c) = (a, b, c)
@inline _tsc(s, t) = (s * t[1], s * t[2], s * t[3])
@inline _tad(a, b) = (a[1] + b[1], a[2] + b[2], a[3] + b[3])
# Julia's float max/min choose y when y > x (resp. y < x); clamp(x, lo, hi) = x > hi ? hi : x < lo ? lo : x.
@inline _tmax(x, tx, y, ty) = y > x ? (y, ty) : (x, tx)
@inline _tmin(x, tx, y, ty) = y < x ? (y, ty) : (x, tx)
@inline _tclamp(x, tx, lo, tlo, hi, thi) = x > hi ? (hi, thi) : (x < lo ? (lo, tlo) : (x, tx))

# (value, d/dll_plain, d/dhh, d/dzr, d/dabsz, d/ddL) of `KernelFused._distance_marg`: node placement, per-node
# terms and the final combination are separate functions, so the device path can evaluate the nodes in parallel
# (`_xasbp_llnodes_kernel!`) with the same arithmetic and summation order.
@inline function _distance_marg_grad(ll_plain::T, hh::T, zr::T, absz::T, dL::T, lik) where {T<:Real}
    pl = _dm_place(hh, zr, absz, dL, lik)
    m = T(-Inf)
    @inbounds for i in 1:_XASBP_NDIST
        m = max(m, _dm_node(i, pl, zr, absz, dL, lik)[1])
    end
    acc = zero(T)
    S = _t3(zero(T), zero(T), zero(T))
    @inbounds for i in 1:_XASBP_NDIST
        e, de = _dm_node(i, pl, zr, absz, dL, lik)
        we = T(_XASBP_GL_W[i]) * exp(e - m)
        acc += we
        S = _tad(S, _tsc(we, de))
    end
    return _dm_finish(ll_plain, hh, zr, pl, m, acc, S, lik)
end

@inline function _dm_place(hh::T, zr::T, absz::T, dL::T, lik) where {T<:Real}
    mp = lik.margphase
    zeff = ifelse(mp, absz, zr)
    hhs = max(hh, T(1e-30))
    dhhs = ifelse(T(1e-30) > hh, zero(T), one(T))
    ustar = zeff / hhs
    sig = 1 / sqrt(hhs)
    t_ustar = _t3(-zeff / (hhs * hhs) * dhhs, 1 / hhs, zero(T))
    t_sig = _t3(-sig / (2 * hhs) * dhhs, zero(T), zero(T))
    umin = dL / T(lik.dhi); t_umin = _t3(zero(T), zero(T), 1 / T(lik.dhi))
    umax = dL / T(lik.dlo); t_umax = _t3(zero(T), zero(T), 1 / T(lik.dlo))
    a0, t_a0 = _tclamp(ustar - 12 * sig, _tad(t_ustar, _tsc(-12, t_sig)), umin, t_umin, umax, t_umax)
    b0, t_b0 = _tclamp(ustar + 12 * sig, _tad(t_ustar, _tsc(12, t_sig)), umin, t_umin, umax, t_umax)
    m3, t_m3 = _tmax(umin, t_umin, b0 - 24 * sig, _tad(t_b0, _tsc(-24, t_sig)))
    ua, t_ua = _tmin(a0, t_a0, m3, t_m3)
    m4, t_m4 = _tmin(umax, t_umax, a0 + 24 * sig, _tad(t_a0, _tsc(24, t_sig)))
    ub, t_ub = _tmax(b0, t_b0, m4, t_m4)
    e3 = _t3(zero(T), zero(T), one(T))
    Da = dL / ub; t_Da = _tad(_tsc(1 / ub, e3), _tsc(-dL / (ub * ub), t_ub))
    Db = dL / ua; t_Db = _tad(_tsc(1 / ua, e3), _tsc(-dL / (ua * ua), t_ua))
    half = (Db - Da) / 2; t_half = _tsc(T(0.5), _tad(t_Db, _tsc(-1, t_Da)))
    mid = (Db + Da) / 2; t_mid = _tsc(T(0.5), _tad(t_Db, t_Da))
    return (; hhs, dhhs, half, t_half, mid, t_mid)
end

# Node i: the exponent e and its derivative along the 3 placement drivers (hh, zeff, dL).
@inline function _dm_node(i, pl, zr::T, absz::T, dL::T, lik) where {T<:Real}
    mp = lik.margphase
    k = lik.dpow
    hhs = pl.hhs
    xi = T(@inbounds _XASBP_GL_X[i])
    D = pl.mid + pl.half * xi
    u = dL / D
    e = k * log(D) - hhs * u * u / 2 + ifelse(mp, KernelFused._log_besseli0(absz * u), zr * u)
    dbes = ifelse(mp, _dlog_besseli0(absz * u), zero(T))
    E_u = -hhs * u + ifelse(mp, absz * dbes, zr)          # de/du
    E_D = k / D - E_u * dL / (D * D)                       # de/dD at fixed dL
    E_zeff = ifelse(mp, u * dbes, u)                       # explicit de/dzeff
    de = _tad(_tsc(E_D, _tad(pl.t_mid, _tsc(xi, pl.t_half))),
              _t3(-u * u / 2 * pl.dhhs, E_zeff, E_u / D))
    return e, de
end

@inline function _dm_finish(ll_plain::T, hh::T, zr::T, pl, m::T, acc::T, S, lik) where {T<:Real}
    mp = lik.margphase
    k = lik.dpow
    half = pl.half; t_half = pl.t_half
    minus_dd_half = ll_plain + hh / 2 - zr
    lognorm = log((T(lik.dhi)^(k + 1) - T(lik.dlo)^(k + 1)) / (k + 1))
    val = minus_dd_half + m + log(acc * half) - lognorm
    gH = T(0.5) + t_half[1] / half + S[1] / acc
    gZ = t_half[2] / half + S[2] / acc
    gD = t_half[3] / half + S[3] / acc
    return val, one(T), gH, -1 + ifelse(mp, zero(T), gZ), ifelse(mp, gZ, zero(T)), gD
end

# (value, d/dll_plain, d/dhh, d/dzr, d/dzi, d/ddL) of `KernelFused._marg_combine`.
@inline function _marg_combine_grad(ll_plain::T, hh::T, zr::T, zi::T, dL::T, lik) where {T<:Real}
    absz = sqrt(zr * zr + zi * zi + T(1e-300))
    if lik.margdist
        v, gl, gh, gzr, gab, gdL = _distance_marg_grad(ll_plain, hh, zr, absz, dL, lik)
        return v, gl, gh, gzr + gab * zr / absz, gab * zi / absz, gdL
    elseif lik.margphase
        db = _dlog_besseli0(absz)
        return ll_plain - zr + KernelFused._log_besseli0(absz), one(T), zero(T), -1 + db * zr / absz, db * zi / absz, zero(T)
    else
        return ll_plain, one(T), zero(T), zero(T), zero(T), zero(T)
    end
end

# Seeded scalar for the reverse pass over `_xasbp_pre`: s . (ll_plain, hh, zr, zi), stored (Const{Nothing} return).
@inline function _xasbp_pre_seed!(out, w, mc, q, chi1, chi2, dL, t_c, phase, ra, sdec, cth, psi, P, Qr, Qi, g, s)
    a, b, c, d = _xasbp_pre(mc, q, chi1, chi2, dL, t_c, phase, ra, sdec, cth, psi, P, Qr, Qi, g)
    @inbounds out[w] = s[1] * a + s[2] * b + s[3] * c + s[4] * d
    return nothing
end

# (b): thread (chunk, detector, walker) -> partial (P, Re Q, Im Q) over `ch` bins.
@kernel function _xasbp_sums_kernel!(part, @Const(K), @Const(TAU), @Const(F), @Const(W), @Const(CDr), @Const(CDi), ch)
    c, d, w = @index(Global, NTuple)
    nb = size(F, 1)
    k = ntuple(j -> @inbounds(K[j, w]), Val(_XASBP_NK))
    tau = @inbounds TAU[d, w]
    P = 0.0; Qr = 0.0; Qi = 0.0
    lo = (c - 1) * ch + 1
    hi = min(c * ch, nb)
    @inbounds for i in lo:hi
        f = F[i, d]
        phi, A = _xas_phase_amp_flat(k, f)
        th = phi - 2 * pi * tau * f
        s, co = sincos(th)
        P += W[i, d] * A * A
        # conj(d) w * A e^{i th}
        ar = A * co; ai = A * s
        Qr += CDr[i, d] * ar - CDi[i, d] * ai
        Qi += CDr[i, d] * ai + CDi[i, d] * ar
    end
    @inbounds part[1, c, d, w] = P
    @inbounds part[2, c, d, w] = Qr
    @inbounds part[3, c, d, w] = Qi
end

# In-kernel AD: `autodiff_deferred` on a GPU, plain `autodiff` on the KernelAbstractions CPU backend (Enzyme refuses
# `autodiff_deferred` outside GPU kernels). `GPU` is a type parameter, so the other branch is pruned at compile time.
@inline _xasbp_ad(::Val{true}, mode, f, ret, args...) = Enzyme.autodiff_deferred(mode, f, ret, args...)
@inline _xasbp_ad(::Val{false}, mode, f, ret, args...) = Enzyme.autodiff(mode, f, ret, args...)

# One bin's contribution to the adjoint seed, as a store (Const{Nothing} return: Enzyme 0.13's
# `autodiff_deferred` in a GPU kernel cannot return an Active scalar, see
# scripts/experiments_xas_inkernel_cuda_grad.jl). `out[j]`'s shadow is seeded with 1 before each call.
@inline function _xasbp_bin_seed!(out, j, k, f, bA, bPhi)
    phi, A = _xas_phase_amp_flat(k, f)
    @inbounds out[j] = bA * A + bPhi * phi
    return nothing
end

# (d): thread (chunk, detector, walker) -> partial bar_K (rows 1:NK) and bar_tau_d (row NK + 1).
@kernel function _xasbp_adjoint_kernel!(partk, S, dS, @Const(K), @Const(TAU), @Const(BAR), @Const(F), @Const(W),
                                        @Const(CDr), @Const(CDi), ch, gpu)
    c, d, w = @index(Global, NTuple)
    j = @index(Global, Linear)
    nb = size(F, 1)
    k = ntuple(jj -> @inbounds(K[jj, w]), Val(_XASBP_NK))
    tau = @inbounds TAU[d, w]
    bP = @inbounds BAR[1, d, w]
    bQr = @inbounds BAR[2, d, w]
    bQi = @inbounds BAR[3, d, w]
    acc = ntuple(_ -> 0.0, Val(_XASBP_NK))
    btau = 0.0
    lo = (c - 1) * ch + 1
    hi = min(c * ch, nb)
    for i in lo:hi
        f = @inbounds F[i, d]
        wt = @inbounds W[i, d]
        cr = @inbounds CDr[i, d]
        ci = @inbounds CDi[i, d]
        phi, A = _xas_phase_amp_flat(k, f)
        th = phi - 2 * pi * tau * f
        s, co = sincos(th)
        # g = bP w A^2 + Re(Bbar q), Bbar = bQr - i bQi, q = cw A e^{i th}, cw = cr + i ci
        # Bbar cw = (bQr cr + bQi ci) + i (bQr ci - bQi cr)
        br = bQr * cr + bQi * ci
        bi = bQr * ci - bQi * cr
        # Re(Bbar cw e^{i th}) = br co - bi s ; Im(...) = br s + bi co
        re_ = br * co - bi * s
        im_ = br * s + bi * co
        bA = 2 * bP * wt * A + re_
        bPhi = -A * im_
        btau += 2 * pi * f * A * im_
        @inbounds dS[j] = 1.0
        res = _xasbp_ad(gpu, Enzyme.Reverse, Enzyme.Const(_xasbp_bin_seed!), Enzyme.Const{Nothing},
            Enzyme.Duplicated(S, dS), Enzyme.Const(j), Enzyme.Active(k), Enzyme.Const(f),
            Enzyme.Const(bA), Enzyme.Const(bPhi))
        acc = _tadd(acc, res[1][3])
    end
    for jj in 1:_XASBP_NK
        @inbounds partk[jj, c, d, w] = acc[jj]
    end
    @inbounds partk[_XASBP_NK + 1, c, d, w] = btau
end

# Shared frequency grid (every detector has the same `freqs`, the usual case): thread (chunk, walker), the
# intrinsic H0 is evaluated once per bin for all detectors, and the adjoint needs ONE VJP per bin (the
# detectors' bar_A / bar_Phi are summed first). Same output layout as the kernels above; the adjoint writes
# bar_K into detector slot 1 only (the host sums bar_K over detectors; the other slots stay zero).
@kernel function _xasbp_sums_shared_kernel!(part, @Const(K), @Const(TAU), @Const(F1), @Const(W), @Const(CDr),
                                            @Const(CDi), ch, ::Val{D}) where {D}
    c, w = @index(Global, NTuple)
    nb = length(F1)
    k = ntuple(j -> @inbounds(K[j, w]), Val(_XASBP_NK))
    tau = ntuple(d -> @inbounds(TAU[d, w]), Val(D))
    P = ntuple(_ -> 0.0, Val(D)); Qr = P; Qi = P
    lo = (c - 1) * ch + 1
    hi = min(c * ch, nb)
    for i in lo:hi
        f = @inbounds F1[i]
        phi, A = _xas_phase_amp_flat(k, f)
        A2 = A * A
        dP = ntuple(d -> @inbounds(W[i, d]) * A2, Val(D))
        sc = ntuple(d -> sincos(phi - 2 * pi * tau[d] * f), Val(D))
        dQr = ntuple(d -> @inbounds(CDr[i, d] * A * sc[d][2] - CDi[i, d] * A * sc[d][1]), Val(D))
        dQi = ntuple(d -> @inbounds(CDr[i, d] * A * sc[d][1] + CDi[i, d] * A * sc[d][2]), Val(D))
        P = _tadd(P, dP); Qr = _tadd(Qr, dQr); Qi = _tadd(Qi, dQi)
    end
    for d in 1:D
        @inbounds part[1, c, d, w] = P[d]
        @inbounds part[2, c, d, w] = Qr[d]
        @inbounds part[3, c, d, w] = Qi[d]
    end
end

# One bin's full contribution to the linearised objective for all detectors,
#   sum_d  bP_d w_d A^2 + Re( conj(bQ_d) cw_d A e^{i (Phi - 2 pi tau_d f)} ),
# whose gradient with respect to (K, tau) IS the bin's (bar_K, bar_tau): one Enzyme call per bin, no
# separate primal pass. Stored into `out[j]` (Const{Nothing} return, see `_xasbp_bin_seed!`).
@inline function _xasbp_bin_full!(out, j, k, tau::NTuple{D}, f, wt::NTuple{D}, cr::NTuple{D}, ci::NTuple{D},
                                  bP::NTuple{D}, bQr::NTuple{D}, bQi::NTuple{D}) where {D}
    phi, A = _xas_phase_amp_flat(k, f)
    acc = 0.0
    for d in 1:D
        th = phi - 2 * pi * tau[d] * f
        s = sin(th); co = cos(th)          # not sincos: Enzyme has no GPU rule for __nv_sincos
        br = bQr[d] * cr[d] + bQi[d] * ci[d]
        bi = bQr[d] * ci[d] - bQi[d] * cr[d]
        acc += bP[d] * wt[d] * A * A + A * (br * co - bi * s)
    end
    @inbounds out[j] = acc
    return nothing
end

@kernel function _xasbp_adjoint_shared_kernel!(partk, S, dS, @Const(K), @Const(TAU), @Const(BAR), @Const(F1), @Const(W),
                                               @Const(CDr), @Const(CDi), ch, ::Val{D}, gpu) where {D}
    c, w = @index(Global, NTuple)
    j = @index(Global, Linear)
    nb = length(F1)
    k = ntuple(jj -> @inbounds(K[jj, w]), Val(_XASBP_NK))
    tau = ntuple(d -> @inbounds(TAU[d, w]), Val(D))
    bP = ntuple(d -> @inbounds(BAR[1, d, w]), Val(D))
    bQr = ntuple(d -> @inbounds(BAR[2, d, w]), Val(D))
    bQi = ntuple(d -> @inbounds(BAR[3, d, w]), Val(D))
    acc = ntuple(_ -> 0.0, Val(_XASBP_NK))
    btau = ntuple(_ -> 0.0, Val(D))
    lo = (c - 1) * ch + 1
    hi = min(c * ch, nb)
    for i in lo:hi
        f = @inbounds F1[i]
        wt = ntuple(d -> @inbounds(W[i, d]), Val(D))
        cr = ntuple(d -> @inbounds(CDr[i, d]), Val(D))
        ci = ntuple(d -> @inbounds(CDi[i, d]), Val(D))
        @inbounds dS[j] = 1.0
        res = _xasbp_ad(gpu, Enzyme.Reverse, Enzyme.Const(_xasbp_bin_full!), Enzyme.Const{Nothing},
            Enzyme.Duplicated(S, dS), Enzyme.Const(j), Enzyme.Active(k), Enzyme.Active(tau), Enzyme.Const(f),
            Enzyme.Const(wt), Enzyme.Const(cr), Enzyme.Const(ci), Enzyme.Const(bP), Enzyme.Const(bQr), Enzyme.Const(bQi))
        acc = _tadd(acc, res[1][3])
        btau = _tadd(btau, res[1][4])
    end
    for jj in 1:_XASBP_NK
        @inbounds partk[jj, c, 1, w] = acc[jj]
    end
    for d in 1:D
        @inbounds partk[_XASBP_NK + 1, c, d, w] = btau[d]
    end
end

# Hand-written VJP of `_xas_phase_amp_flat`: `acc + gPhi * d(phi)/dk + gA * d(A)/dk` (acc is the running
# coefficient cotangent). Only the active phase and amplitude segments contribute; the segment thresholds
# (f1, f2, fcutPar, fA1, fA4) get zero, as with AD. d/dx of each segment uses GravitationalWaveforms' own analytic
# derivatives (`_xas_*_deriv`, exact derivatives of the `_val` forms). Validated against the Enzyme VJP
# (`adjoint = :enzyme`) in test/test_xas_binparallel.jl.
# (`Base.setindex` on a 61-tuple is a generic splatting call, not GPU-compilable; this unrolls and folds, as `j`
# is a literal at every call site.)
@inline _acc(t::NTuple{N,Float64}, j::Int, v) where {N} = ntuple(i -> ifelse(i == j, t[i] + v, t[i]), Val(N))

@inline function _xas_phase_amp_flat_vjp(acc, k, f, gPhi, gA)
    fs = max(f, eps(typeof(f)))
    x = k[1] * fs
    # ---- phase: phis = seg(x) / eta ----
    if x < k[4]
        gs = gPhi / k[5]
        if x < k[2]
            lx = log(x); x13 = cbrt(x); x23 = x13 * x13; x43 = x * x13; x53 = x * x23; x2 = x * x
            x73 = x2 * x13; x83 = x2 * x23; x3 = x2 * x; x103 = x3 * x13; x113 = x3 * x23
            phiN = k[24]
            phi_Ins = k[8] + k[9] * x13 + k[10] * x23 + k[11] * x + k[12] * x43 + k[13] * x53 + k[14] * x53 * lx +
                      k[15] * x2 + k[16] * x2 * lx + k[17] * x73 + k[18] * x83 + k[19] * x83 * lx +
                      k[20] * x83 + k[21] * x3 + k[22] * x103 + k[23] * x113
            xm53 = 1 / x53
            seg = phiN * xm53 * phi_Ins
            c = gs * phiN * xm53
            acc = _acc(acc, 8, c); acc = _acc(acc, 9, c * x13); acc = _acc(acc, 10, c * x23); acc = _acc(acc, 11, c * x)
            acc = _acc(acc, 12, c * x43); acc = _acc(acc, 13, c * x53); acc = _acc(acc, 14, c * x53 * lx)
            acc = _acc(acc, 15, c * x2); acc = _acc(acc, 16, c * x2 * lx); acc = _acc(acc, 17, c * x73)
            acc = _acc(acc, 18, c * x83); acc = _acc(acc, 19, c * x83 * lx); acc = _acc(acc, 20, c * x83)
            acc = _acc(acc, 21, c * x3); acc = _acc(acc, 22, c * x103); acc = _acc(acc, 23, c * x113)
            acc = _acc(acc, 24, gs * xm53 * phi_Ins)
            dsdx = KernelFused._xas_phi_ins_deriv(x, k[8], k[9], k[10], k[11], k[12], k[13], k[14], k[15], k[16],
                                                  k[17], k[18], k[19], k[20], k[21], k[22], k[23], k[24])
        elseif x < k[3]
            fd = k[6]; fRD = k[7]; cL = k[36]
            u = x - fRD
            z = u / (2 * fd)
            at = atan(z)
            dz = 1 / (1 + z * z)
            seg = k[25] * x + k[26] * log(x) - k[27] / x - k[28] / (2 * x * x) - k[29] / (3 * x * x * x) +
                  (2 * cL / fd) * at + k[31] * x + k[30]
            acc = _acc(acc, 25, gs * x); acc = _acc(acc, 26, gs * log(x)); acc = _acc(acc, 27, -gs / x)
            acc = _acc(acc, 28, -gs / (2 * x * x)); acc = _acc(acc, 29, -gs / (3 * x * x * x))
            acc = _acc(acc, 36, gs * (2 / fd) * at)
            acc = _acc(acc, 6, gs * (-(2 * cL / (fd * fd)) * at - (2 * cL / fd) * dz * u / (2 * fd * fd)))
            acc = _acc(acc, 7, -gs * (2 * cL / fd) * dz / (2 * fd))
            acc = _acc(acc, 30, gs); acc = _acc(acc, 31, gs * x)
            dsdx = KernelFused._xas_phi_int_deriv(x, k[25], k[26], k[27], k[28], k[29], cL, fd, fRD) + k[31]
        else
            fd = k[6]; fRD = k[7]; cL = k[36]
            u = x - fRD
            z = u / fd
            at = atan(z)
            dz = 1 / (1 + z * z)
            x23 = cbrt(x)^2
            seg = k[32] * x + 1.5 * k[33] * x23 - k[34] / x - (k[35] / 3) / (x * x * x) + (cL / fd) * at + k[37] + k[38] * x
            acc = _acc(acc, 32, gs * x); acc = _acc(acc, 33, gs * 1.5 * x23); acc = _acc(acc, 34, -gs / x)
            acc = _acc(acc, 35, -gs / (3 * x * x * x)); acc = _acc(acc, 36, gs * at / fd)
            acc = _acc(acc, 6, gs * (-(cL / (fd * fd)) * at - (cL / fd) * dz * u / (fd * fd)))
            acc = _acc(acc, 7, -gs * (cL / fd) * dz / fd)
            acc = _acc(acc, 37, gs); acc = _acc(acc, 38, gs * x)
            dsdx = KernelFused._xas_phi_mrd_deriv(x, k[32], k[33], k[34], k[35], cL, fd, fRD) + k[38]
        end
        acc = _acc(acc, 5, -gs * seg / k[5])
        acc = _acc(acc, 1, gs * dsdx * fs)
    end
    # ---- amplitude: A = k61 * xa^(-7/6) * aimr(xa) ----
    xa = k[39] * fs
    if xa < k[42]
        p = xa^(-7.0 / 6.0)
        if xa < k[40]
            x13 = cbrt(xa); x23 = x13 * x13; x43 = xa * x13; x53 = xa * x23; x2 = xa * xa
            x73 = x2 * x13; x83 = x2 * x23; x3 = x2 * xa
            aimr = 1 + k[45] * x23 + k[46] * xa + k[47] * x43 + k[48] * x53 + k[49] * x2 + k[50] * x73 + k[51] * x83 + k[52] * x3
            ga = gA * k[61] * p
            acc = _acc(acc, 45, ga * x23); acc = _acc(acc, 46, ga * xa); acc = _acc(acc, 47, ga * x43)
            acc = _acc(acc, 48, ga * x53); acc = _acc(acc, 49, ga * x2); acc = _acc(acc, 50, ga * x73)
            acc = _acc(acc, 51, ga * x83); acc = _acc(acc, 52, ga * x3)
            daimr = GravitationalWaveforms._xas_amp_ins_deriv(xa, k[45], k[46], k[47], k[48], k[49], k[50], k[51], k[52])
        elseif xa < k[41]
            Pn = k[53] + xa * (k[54] + xa * (k[55] + xa * (k[56] + xa * k[57])))
            dPn = k[54] + xa * (2 * k[55] + xa * (3 * k[56] + xa * 4 * k[57]))
            x76 = xa^(7.0 / 6.0)
            aimr = x76 / Pn
            ga = gA * k[61] * p
            c = -ga * aimr / Pn
            acc = _acc(acc, 53, c); acc = _acc(acc, 54, c * xa); acc = _acc(acc, 55, c * xa * xa)
            acc = _acc(acc, 56, c * xa * xa * xa); acc = _acc(acc, 57, c * xa * xa * xa * xa)
            daimr = (7.0 / 6.0) * aimr / xa - aimr * dPn / Pn
        else
            g1 = k[58]; g2 = k[59]; g3 = k[60]; fd = k[43]; fRD = k[44]
            u = xa - fRD
            gR = g2 / (fd * g3)
            Dn = u * u + (g3 * fd) * (g3 * fd)
            E = exp(-u * gR)
            aimr = E * (fd * g1 * g3) / Dn
            ga = gA * k[61] * p
            acc = _acc(acc, 58, ga * E * fd * g3 / Dn)
            acc = _acc(acc, 59, ga * aimr * (-u / (fd * g3)))
            acc = _acc(acc, 60, ga * (aimr * (u * g2 / (fd * g3 * g3)) + E * fd * g1 / Dn - aimr * 2 * g3 * fd * fd / Dn))
            acc = _acc(acc, 43, ga * (aimr * (u * g2 / (fd * fd * g3)) + E * g1 * g3 / Dn - aimr * 2 * g3 * g3 * fd / Dn))
            acc = _acc(acc, 44, ga * aimr * (gR + 2 * u / Dn))
            daimr = aimr * (-gR - 2 * u / Dn)
        end
        acc = _acc(acc, 61, gA * p * aimr)
        gxa = gA * k[61] * ((-7.0 / 6.0) * p / xa * aimr + p * daimr)
        acc = _acc(acc, 39, gxa * fs)
    end
    return acc
end

@kernel function _xasbp_adjoint_hand_kernel!(partk, @Const(K), @Const(TAU), @Const(BAR), @Const(F1), @Const(W),
                                             @Const(CDr), @Const(CDi), ch, ::Val{D}) where {D}
    c, w = @index(Global, NTuple)
    nb = length(F1)
    k = ntuple(jj -> @inbounds(K[jj, w]), Val(_XASBP_NK))
    tau = ntuple(d -> @inbounds(TAU[d, w]), Val(D))
    bP = ntuple(d -> @inbounds(BAR[1, d, w]), Val(D))
    bQr = ntuple(d -> @inbounds(BAR[2, d, w]), Val(D))
    bQi = ntuple(d -> @inbounds(BAR[3, d, w]), Val(D))
    acc = ntuple(_ -> 0.0, Val(_XASBP_NK))
    btau = ntuple(_ -> 0.0, Val(D))
    lo = (c - 1) * ch + 1
    hi = min(c * ch, nb)
    for i in lo:hi
        f = @inbounds F1[i]
        phi, A = _xas_phase_amp_flat(k, f)
        gA = 0.0; gPhi = 0.0
        dtau = ntuple(_ -> 0.0, Val(D))
        for d in 1:D
            cr = @inbounds CDr[i, d]; ci = @inbounds CDi[i, d]
            s, co = sincos(phi - 2 * pi * tau[d] * f)
            br = bQr[d] * cr + bQi[d] * ci
            bi = bQr[d] * ci - bQi[d] * cr
            re_ = br * co - bi * s
            im_ = br * s + bi * co
            gA += 2 * bP[d] * @inbounds(W[i, d]) * A + re_
            gPhi -= A * im_
            dtau = Base.setindex(dtau, 2 * pi * f * A * im_, d)
        end
        btau = _tadd(btau, dtau)
        acc = _xas_phase_amp_flat_vjp(acc, k, f, gPhi, gA)
    end
    for jj in 1:_XASBP_NK
        @inbounds partk[jj, c, 1, w] = acc[jj]
    end
    for d in 1:D
        @inbounds partk[_XASBP_NK + 1, c, d, w] = btau[d]
    end
end

# ------------------------------------------------------------------------------------------------
# Device-resident per-walker stages (`resident = true`): (a), (c) and (e) as one-thread-per-walker
# kernels, so a call moves only positions in and logp / gradients out.
#   (a+e') forward mode: K and tau depend on 8 of the 11 sampled rows (`_XASBP_JROWS`), so ONE width-8
#          `BatchDuplicated` forward pass gives K, tau and their Jacobian. Forward mode keeps no tape, so none of
#          the reverse-mode device-heap caches that made the in-kernel reverse VJP of the coefficient build leak
#          on the device.                                                        [_xasbp_coeffs_jac_kernel!]
#   (c)    in-kernel reverse Enzyme on `_xasbp_ll_from_sums` (small, constant-trip loops only). [_xasbp_ll_kernel!]
#   (e)    gradient = dL/dx at fixed sums + J' (bar_K, bar_tau).                               [_xasbp_grad_kernel!]
# ------------------------------------------------------------------------------------------------

const _XASBP_JROWS = (1, 2, 3, 4, 5, 6, 8, 9)      # mc, q, chi1, chi2, dL, t_c, ra, sin_dec
const _XASBP_NJ = length(_XASBP_JROWS)

# K (rows 1:NK) and tau (rows NK+1:NK+D) of walker `w` into column `w` of `KT`. The stores are unrolled
# (`_xasbp_store_col!`): with a runtime-indexed loop the tuple goes through memory and Enzyme's forward pass then
# writes the PRIMAL of the constant entries (e.g. fcutPar = 0.3, phi0 = 1) into the shadows (CPU and CUDA,
# width 1 and 8; reverse mode is unaffected).
@generated function _xasbp_store_col!(KT, w, v::NTuple{N}, off) where {N}
    Expr(:block, (:(@inbounds KT[off + $j, w] = v[$j]) for j in 1:N)..., :(return nothing))
end

@inline function _xasbp_coeffs_tau_store!(KT, w, mc, q, chi1, chi2, dL, t_c, ra, sdec, phase, cth, psi,
                                          model, g, lik)
    K, tau = _xasbp_coeffs_tau(mc, q, chi1, chi2, dL, t_c, phase, ra, sdec, cth, psi, model, g, lik)
    _xasbp_store_col!(KT, w, K, 0)
    _xasbp_store_col!(KT, w, tau, _XASBP_NK)
    return nothing
end

@inline _xasbp_unit(i) = ntuple(j -> ifelse(i == j, 1.0, 0.0), Val(_XASBP_NJ))

# KT[:, w] = (K, tau); JT[i][:, w] = d(K, tau)/d x[_XASBP_JROWS[i]]. Separate arrays of one type for the primal and
# the 8 shadows (`BatchDuplicated` with SubArray views of one array left the shadows zero on CUDA).
@kernel function _xasbp_coeffs_jac_kernel!(KT, JT, @Const(POS), model, g, lik, gpu)
    w = @index(Global, Linear)
    x = ntuple(k -> @inbounds(POS[k, w]), Val(11))
    _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(_xasbp_coeffs_tau_store!), Enzyme.Const,
        Enzyme.BatchDuplicated(KT, JT, false), Enzyme.Const(w),
        Enzyme.BatchDuplicated(x[1], _xasbp_unit(1)), Enzyme.BatchDuplicated(x[2], _xasbp_unit(2)),
        Enzyme.BatchDuplicated(x[3], _xasbp_unit(3)), Enzyme.BatchDuplicated(x[4], _xasbp_unit(4)),
        Enzyme.BatchDuplicated(x[5], _xasbp_unit(5)), Enzyme.BatchDuplicated(x[6], _xasbp_unit(6)),
        Enzyme.BatchDuplicated(x[8], _xasbp_unit(7)), Enzyme.BatchDuplicated(x[9], _xasbp_unit(8)),
        Enzyme.Const(x[7]), Enzyme.Const(x[10]), Enzyme.Const(x[11]),
        Enzyme.Const(model), Enzyme.Const(g), Enzyme.Const(lik))
end

# Same result split by what each sampled row reaches (`jac = :split`): K depends on mc, q, chi1, chi2, dL
# only, so the coefficient build is pushed forward along those 5 directions (width 5, not 8); t_c, ra and sin_dec
# move tau alone, through t_c + detector delays, whose width-3 forward pass is tiny. The tau rows of JT[6:8] are
# written by the second pass (into scratch `TS` for the primal); their K rows stay zero.
@inline function _xasbp_tau_extr_store!(TS, w, t_c, ra, sdec, g)
    dec = asin(_clamp_unit(sdec))
    dt_ref = ifelse(g.geo & (g.tref > 0), time_delay_from_geocenter_gmst(g.dets[max(g.tref, 1)], ra, dec, g.gmst), zero(t_c))
    tau = ntuple(d -> t_c + ifelse(g.geo, time_delay_from_geocenter_gmst(g.dets[d], ra, dec, g.gmst) - dt_ref, zero(t_c)),
                 Val(length(g.dets)))
    _xasbp_store_col!(TS, w, tau, _XASBP_NK)
    return nothing
end

@inline _xasbp_unit5(i) = ntuple(j -> ifelse(i == j, 1.0, 0.0), Val(5))
@inline _xasbp_unit3(i) = ntuple(j -> ifelse(i == j, 1.0, 0.0), Val(3))
@inline _xasbp_unit4(i) = ntuple(j -> ifelse(i == j, 1.0, 0.0), Val(4))

# `jac = :split3`: as `:split`, with the coefficient build split as well -- thread (part, walker): part 1 the phase
# coefficients and tau (the f_ref time shift needs the phase fit), part 2 the amplitude coefficients, part 3 the
# extrinsic tau directions. A shorter serial chain and less per-thread state per walker, for the latency at small
# walker counts, where one thread per walker otherwise runs the whole build.
@inline function _xasbp_phase_tau_store!(KT, w, mc, q, chi1, chi2, t_c, ra, sdec, cth, model, g, lik)
    m1, m2, dec, _ = _eobimr_physical(mc, q, sdec, cth)
    mcc, eta = KernelFused._chirp_mass_eta(m1, m2)
    pc = KernelFused._xas_phi_coeffs(model, mcc, eta, chi1, chi2)
    tc = t_c - KernelFused._xas_time_shift(pc, lik)
    dt_ref = ifelse(g.geo & (g.tref > 0), time_delay_from_geocenter_gmst(g.dets[max(g.tref, 1)], ra, dec, g.gmst), zero(tc))
    tau = ntuple(d -> tc + ifelse(g.geo, time_delay_from_geocenter_gmst(g.dets[d], ra, dec, g.gmst) - dt_ref, zero(tc)),
                 Val(length(g.dets)))
    _xasbp_store_col!(KT, w, _xas_flat_phase(pc), 0)
    _xasbp_store_col!(KT, w, tau, _XASBP_NK)
    return nothing
end

@inline function _xasbp_amp_store!(KT, w, mc, q, chi1, chi2, dL, sdec, cth, model)
    m1, m2, _, _ = _eobimr_physical(mc, q, sdec, cth)
    mcc, eta = KernelFused._chirp_mass_eta(m1, m2)
    ac = KernelFused._xas_ampl_coeffs(model, mcc, eta, chi1, chi2, dL)
    _xasbp_store_col!(KT, w, _xas_flat_amp(ac), _XASBP_NKP)
    return nothing
end

@kernel function _xasbp_coeffs_jac_split3_kernel!(KT, JT, TS, @Const(POS), model, g, lik, gpu)
    w, part = @index(Global, NTuple)                 # part slowest: each warp runs one part (no divergence)
    x = ntuple(k -> @inbounds(POS[k, w]), Val(11))
    J5 = (JT[1], JT[2], JT[3], JT[4], JT[5])
    if part == 1                                     # width 4: the phase fit does not depend on dL
        _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(_xasbp_phase_tau_store!), Enzyme.Const,
            Enzyme.BatchDuplicated(KT, (JT[1], JT[2], JT[3], JT[4]), false), Enzyme.Const(w),
            Enzyme.BatchDuplicated(x[1], _xasbp_unit4(1)), Enzyme.BatchDuplicated(x[2], _xasbp_unit4(2)),
            Enzyme.BatchDuplicated(x[3], _xasbp_unit4(3)), Enzyme.BatchDuplicated(x[4], _xasbp_unit4(4)),
            Enzyme.Const(x[6]), Enzyme.Const(x[8]), Enzyme.Const(x[9]), Enzyme.Const(x[10]),
            Enzyme.Const(model), Enzyme.Const(g), Enzyme.Const(lik))
    elseif part == 2
        _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(_xasbp_amp_store!), Enzyme.Const,
            Enzyme.BatchDuplicated(KT, J5, false), Enzyme.Const(w),
            Enzyme.BatchDuplicated(x[1], _xasbp_unit5(1)), Enzyme.BatchDuplicated(x[2], _xasbp_unit5(2)),
            Enzyme.BatchDuplicated(x[3], _xasbp_unit5(3)), Enzyme.BatchDuplicated(x[4], _xasbp_unit5(4)),
            Enzyme.BatchDuplicated(x[5], _xasbp_unit5(5)), Enzyme.Const(x[9]), Enzyme.Const(x[10]), Enzyme.Const(model))
    else
        _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(_xasbp_tau_extr_store!), Enzyme.Const,
            Enzyme.BatchDuplicated(TS, (JT[6], JT[7], JT[8]), false), Enzyme.Const(w),
            Enzyme.BatchDuplicated(x[6], _xasbp_unit3(1)), Enzyme.BatchDuplicated(x[8], _xasbp_unit3(2)),
            Enzyme.BatchDuplicated(x[9], _xasbp_unit3(3)), Enzyme.Const(g))
    end
end

# `jac = :split5`: five narrower passes per walker -- phase+tau along (mc, q) and along (chi1, chi2), amplitude along
# (mc, q) and along (chi1, chi2, dL), extrinsic tau -- each on its own warps (the phase/amplitude primal is recomputed).
@inline _xasbp_unit2(i) = ntuple(j -> ifelse(i == j, 1.0, 0.0), Val(2))
@kernel function _xasbp_coeffs_jac_split5_kernel!(KT, JT, TS, @Const(POS), model, g, lik, gpu)
    w, part = @index(Global, NTuple)                 # part slowest: each warp runs one part
    x = ntuple(k -> @inbounds(POS[k, w]), Val(11))
    if part == 1
        _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(_xasbp_phase_tau_store!), Enzyme.Const,
            Enzyme.BatchDuplicated(KT, (JT[1], JT[2]), false), Enzyme.Const(w),
            Enzyme.BatchDuplicated(x[1], _xasbp_unit2(1)), Enzyme.BatchDuplicated(x[2], _xasbp_unit2(2)),
            Enzyme.Const(x[3]), Enzyme.Const(x[4]),
            Enzyme.Const(x[6]), Enzyme.Const(x[8]), Enzyme.Const(x[9]), Enzyme.Const(x[10]),
            Enzyme.Const(model), Enzyme.Const(g), Enzyme.Const(lik))
    elseif part == 2
        _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(_xasbp_phase_tau_store!), Enzyme.Const,
            Enzyme.BatchDuplicated(KT, (JT[3], JT[4]), false), Enzyme.Const(w),
            Enzyme.Const(x[1]), Enzyme.Const(x[2]),
            Enzyme.BatchDuplicated(x[3], _xasbp_unit2(1)), Enzyme.BatchDuplicated(x[4], _xasbp_unit2(2)),
            Enzyme.Const(x[6]), Enzyme.Const(x[8]), Enzyme.Const(x[9]), Enzyme.Const(x[10]),
            Enzyme.Const(model), Enzyme.Const(g), Enzyme.Const(lik))
    elseif part == 3
        _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(_xasbp_amp_store!), Enzyme.Const,
            Enzyme.BatchDuplicated(KT, (JT[1], JT[2]), false), Enzyme.Const(w),
            Enzyme.BatchDuplicated(x[1], _xasbp_unit2(1)), Enzyme.BatchDuplicated(x[2], _xasbp_unit2(2)),
            Enzyme.Const(x[3]), Enzyme.Const(x[4]), Enzyme.Const(x[5]), Enzyme.Const(x[9]), Enzyme.Const(x[10]),
            Enzyme.Const(model))
    elseif part == 4
        _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(_xasbp_amp_store!), Enzyme.Const,
            Enzyme.BatchDuplicated(KT, (JT[3], JT[4], JT[5]), false), Enzyme.Const(w),
            Enzyme.Const(x[1]), Enzyme.Const(x[2]),
            Enzyme.BatchDuplicated(x[3], _xasbp_unit3(1)), Enzyme.BatchDuplicated(x[4], _xasbp_unit3(2)),
            Enzyme.BatchDuplicated(x[5], _xasbp_unit3(3)), Enzyme.Const(x[9]), Enzyme.Const(x[10]), Enzyme.Const(model))
    else
        _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(_xasbp_tau_extr_store!), Enzyme.Const,
            Enzyme.BatchDuplicated(TS, (JT[6], JT[7], JT[8]), false), Enzyme.Const(w),
            Enzyme.BatchDuplicated(x[6], _xasbp_unit3(1)), Enzyme.BatchDuplicated(x[8], _xasbp_unit3(2)),
            Enzyme.BatchDuplicated(x[9], _xasbp_unit3(3)), Enzyme.Const(g))
    end
end

@kernel function _xasbp_coeffs_jac_split_kernel!(KT, JT, TS, @Const(POS), model, g, lik, gpu)
    w = @index(Global, Linear)
    x = ntuple(k -> @inbounds(POS[k, w]), Val(11))
    _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(_xasbp_coeffs_tau_store!), Enzyme.Const,
        Enzyme.BatchDuplicated(KT, (JT[1], JT[2], JT[3], JT[4], JT[5]), false), Enzyme.Const(w),
        Enzyme.BatchDuplicated(x[1], _xasbp_unit5(1)), Enzyme.BatchDuplicated(x[2], _xasbp_unit5(2)),
        Enzyme.BatchDuplicated(x[3], _xasbp_unit5(3)), Enzyme.BatchDuplicated(x[4], _xasbp_unit5(4)),
        Enzyme.BatchDuplicated(x[5], _xasbp_unit5(5)), Enzyme.Const(x[6]),
        Enzyme.Const(x[8]), Enzyme.Const(x[9]),
        Enzyme.Const(x[7]), Enzyme.Const(x[10]), Enzyme.Const(x[11]),
        Enzyme.Const(model), Enzyme.Const(g), Enzyme.Const(lik))
    _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(_xasbp_tau_extr_store!), Enzyme.Const,
        Enzyme.BatchDuplicated(TS, (JT[6], JT[7], JT[8]), false), Enzyme.Const(w),
        Enzyme.BatchDuplicated(x[6], _xasbp_unit3(1)), Enzyme.BatchDuplicated(x[8], _xasbp_unit3(2)),
        Enzyme.BatchDuplicated(x[9], _xasbp_unit3(3)), Enzyme.Const(g))
end

# (c): LL[w] = logL, GX[:, w] = dlogL/dx at fixed sums, BAR[:, :, w] = dlogL/d(P, Re Q, Im Q). `_marg_combine`'s
# gradient is hand-written (`_marg_combine_grad`); only the small, loop-free `_xasbp_pre` is reversed by Enzyme,
# seeded with it. `SC` is scratch for the seeded store.
@kernel function _xasbp_ll_kernel!(LL, SC, dSC, GX, BAR, @Const(POS), @Const(RED), g, lik, ::Val{D}, gpu) where {D}
    w = @index(Global, Linear)
    x = ntuple(k -> @inbounds(POS[k, w]), Val(11))
    P = ntuple(d -> @inbounds(RED[1, d, w]), Val(D))
    Qr = ntuple(d -> @inbounds(RED[2, d, w]), Val(D))
    Qi = ntuple(d -> @inbounds(RED[3, d, w]), Val(D))
    llp, hh, zr, zi = _xasbp_pre(x..., P, Qr, Qi, g)
    val, gl, gh, gzr, gzi, gdL = _marg_combine_grad(llp, hh, zr, zi, x[5], lik)
    @inbounds LL[w] = val
    @inbounds dSC[w] = 1.0
    res = _xasbp_ad(gpu, Enzyme.Reverse, Enzyme.Const(_xasbp_pre_seed!), Enzyme.Const{Nothing},
        Enzyme.Duplicated(SC, dSC), Enzyme.Const(w),
        Enzyme.Active(x[1]), Enzyme.Active(x[2]), Enzyme.Active(x[3]), Enzyme.Active(x[4]), Enzyme.Active(x[5]),
        Enzyme.Active(x[6]), Enzyme.Active(x[7]), Enzyme.Active(x[8]), Enzyme.Active(x[9]), Enzyme.Active(x[10]),
        Enzyme.Active(x[11]), Enzyme.Active(P), Enzyme.Active(Qr), Enzyme.Active(Qi), Enzyme.Const(g),
        Enzyme.Const((gl, gh, gzr, gzi)))
    gr = res[1]
    gx = ntuple(k -> gr[k + 2] + ifelse(k == 5, gdL, 0.0), Val(11))     # unrolled: gr is heterogeneous
    for k in 1:11
        @inbounds GX[k, w] = gx[k]
    end
    for d in 1:D
        @inbounds BAR[1, d, w] = gr[14][d]
        @inbounds BAR[2, d, w] = gr[15][d]
        @inbounds BAR[3, d, w] = gr[16][d]
    end
end

# Distance-marginalization nodes in parallel (`_XASBP_NDIST` threads per walker; the per-walker ll kernel otherwise
# evaluates 2 x 64 nodes serially, ~0.24 ms of latency whatever the walker count). EN[:, i, w] = (e, de...).
@kernel function _xasbp_llnodes_kernel!(EN, @Const(POS), @Const(RED), g, lik, ::Val{D}) where {D}
    i, w = @index(Global, NTuple)
    x = ntuple(k -> @inbounds(POS[k, w]), Val(11))
    P = ntuple(d -> @inbounds(RED[1, d, w]), Val(D))
    Qr = ntuple(d -> @inbounds(RED[2, d, w]), Val(D))
    Qi = ntuple(d -> @inbounds(RED[3, d, w]), Val(D))
    _, hh, zr, zi = _xasbp_pre(x..., P, Qr, Qi, g)
    absz = sqrt(zr * zr + zi * zi + 1e-300)
    pl = _dm_place(hh, zr, absz, x[5], lik)
    e, de = _dm_node(i, pl, zr, absz, x[5], lik)
    @inbounds EN[1, i, w] = e
    @inbounds EN[2, i, w] = de[1]
    @inbounds EN[3, i, w] = de[2]
    @inbounds EN[4, i, w] = de[3]
end

# `_marg_combine_grad` with distance marginalization, from precomputed nodes (same order and arithmetic).
@inline function _marg_combine_grad_nodes(ll_plain::T, hh::T, zr::T, zi::T, dL::T, lik, EN, w) where {T<:Real}
    absz = sqrt(zr * zr + zi * zi + T(1e-300))
    pl = _dm_place(hh, zr, absz, dL, lik)
    m = T(-Inf)
    for i in 1:_XASBP_NDIST
        m = max(m, @inbounds EN[1, i, w])
    end
    acc = zero(T)
    S = _t3(zero(T), zero(T), zero(T))
    for i in 1:_XASBP_NDIST
        we = T(@inbounds _XASBP_GL_W[i]) * exp(@inbounds(EN[1, i, w]) - m)
        acc += we
        S = _tad(S, _tsc(we, _t3(@inbounds(EN[2, i, w]), @inbounds(EN[3, i, w]), @inbounds(EN[4, i, w]))))
    end
    v, gl, gh, gzr, gab, gdL = _dm_finish(ll_plain, hh, zr, pl, m, acc, S, lik)
    return v, gl, gh, gzr + gab * zr / absz, gab * zi / absz, gdL
end

@kernel function _xasbp_ll2_kernel!(LL, SC, dSC, GX, BAR, @Const(POS), @Const(RED), @Const(EN), g, lik, ::Val{D},
                                    gpu) where {D}
    w = @index(Global, Linear)
    x = ntuple(k -> @inbounds(POS[k, w]), Val(11))
    P = ntuple(d -> @inbounds(RED[1, d, w]), Val(D))
    Qr = ntuple(d -> @inbounds(RED[2, d, w]), Val(D))
    Qi = ntuple(d -> @inbounds(RED[3, d, w]), Val(D))
    llp, hh, zr, zi = _xasbp_pre(x..., P, Qr, Qi, g)
    val, gl, gh, gzr, gzi, gdL = _marg_combine_grad_nodes(llp, hh, zr, zi, x[5], lik, EN, w)
    @inbounds LL[w] = val
    @inbounds dSC[w] = 1.0
    res = _xasbp_ad(gpu, Enzyme.Reverse, Enzyme.Const(_xasbp_pre_seed!), Enzyme.Const{Nothing},
        Enzyme.Duplicated(SC, dSC), Enzyme.Const(w),
        Enzyme.Active(x[1]), Enzyme.Active(x[2]), Enzyme.Active(x[3]), Enzyme.Active(x[4]), Enzyme.Active(x[5]),
        Enzyme.Active(x[6]), Enzyme.Active(x[7]), Enzyme.Active(x[8]), Enzyme.Active(x[9]), Enzyme.Active(x[10]),
        Enzyme.Active(x[11]), Enzyme.Active(P), Enzyme.Active(Qr), Enzyme.Active(Qi), Enzyme.Const(g),
        Enzyme.Const((gl, gh, gzr, gzi)))
    gr = res[1]
    gx = ntuple(k -> gr[k + 2] + ifelse(k == 5, gdL, 0.0), Val(11))
    for k in 1:11
        @inbounds GX[k, w] = gx[k]
    end
    for d in 1:D
        @inbounds BAR[1, d, w] = gr[14][d]
        @inbounds BAR[2, d, w] = gr[15][d]
        @inbounds BAR[3, d, w] = gr[16][d]
    end
end

# (e): G[:, w] = GX[:, w] + J' (bar_K summed over detector slots, bar_tau).
@kernel function _xasbp_grad_kernel!(G, @Const(GX), JT, @Const(REDK), ::Val{D}) where {D}
    w = @index(Global, Linear)
    for k in 1:11
        @inbounds G[k, w] = GX[k, w]
    end
    for i in 1:_XASBP_NJ
        J = JT[i]
        s = 0.0
        for j in 1:_XASBP_NK
            bk = 0.0
            for d in 1:D
                bk += @inbounds REDK[j, d, w]
            end
            s += bk * @inbounds(J[j, w])
        end
        for d in 1:D
            s += @inbounds(REDK[_XASBP_NK + 1, d, w]) * @inbounds(J[_XASBP_NK + d, w])
        end
        @inbounds G[_XASBP_JROWS[i], w] += s
    end
end

# ------------------------------------------------------------------------------------------------
# Relative binning on the device (`relative_binning = rb`, with `resident`). With r_e = h(f_e)/h_ref(f_e)
# at the bin edges and r linear in each bin (`KernelFused._xas_ll_rb`), the RB sums are, exactly,
#     <d,h>-part:  sum_b A0_b conj(r0_b) + A1_b conj(r1_b)        = sum_e conj(r_e) C_e
#     <h,h>-part:  sum_b B0_b |r0_b|^2 + 2 B1_b Re(r0_b conj r1_b) = sum_e |r_e|^2 D_e + sum_b (B0_b/2) Re(r_b conj r_{b+1})
# (the imaginary cross terms of Re(r0 conj r1) cancel). With r_e = combo_d e^{-i phase} u_e,
# u_e = H0(f_e) e^{-2 pi i tau_d f_e} / h_ref(f_e), these are the dense path's (P_d, Q_d) with
#     P_d = [sum_e |u_e|^2 D_e + sum_b (B0_b/2) Re(u_b conj u_{b+1})] / (4 df),   Q_d = sum_e u_e conj(C_e) / (4 df),
# so the resident logL / gradient kernels are reused unchanged. Edges are shared by the detectors (they come from the
# reference's intrinsic phase on a common grid), so H0 is evaluated once per edge.
# ------------------------------------------------------------------------------------------------

# Host: per-edge coefficient arrays (ne x D) from `build_xas_relative_binning`'s summary data.
function _xasrb_coeffs(rb, g)
    D = length(rb.edges)
    E = rb.edges[1]
    all(e -> e == E, rb.edges) || throw(ArgumentError("device relative binning needs the same bin edges in every detector"))
    ne = length(E)
    HIr = zeros(ne, D); HIi = zeros(ne, D); Cr = zeros(ne, D); Ci = zeros(ne, D); DE = zeros(ne, D); B0H = zeros(ne, D)
    for d in 1:D
        s = 1 / (4 * g.df[d])
        A0, A1, B0, B1, href = rb.A0[d], rb.A1[d], rb.B0[d], rb.B1[d], rb.href[d]
        for e in 1:ne
            hinv = abs2(href[e]) == 0 ? 0.0im : inv(href[e])
            HIr[e, d] = real(hinv); HIi[e, d] = imag(hinv)
            c = 0.0im; dd = 0.0
            if e > 1
                b = e - 1; db = E[b+1] - E[b]
                c += A0[b] / 2 + A1[b] / db
                dd += B0[b] / 4 + B1[b] / db
            end
            if e < ne
                b = e; db = E[b+1] - E[b]
                c += A0[b] / 2 - A1[b] / db
                dd += B0[b] / 4 - B1[b] / db
                B0H[e, d] = s * B0[b] / 2
            end
            Cr[e, d] = s * real(c); Ci[e, d] = s * imag(c); DE[e, d] = s * dd
        end
    end
    return (FE=Float64.(E), HIr=HIr, HIi=HIi, Cr=Cr, Ci=Ci, DE=DE, B0H=B0H, ne=ne)
end

# u_e for every (edge, walker): also stores A and Phi of H0(f_e) for the adjoint.
@kernel function _xasrb_eval_kernel!(AE, PH, UR, UI, @Const(K), @Const(TAU), @Const(FE), @Const(HIr), @Const(HIi),
                                     ::Val{D}) where {D}
    e, w = @index(Global, NTuple)
    k = ntuple(j -> @inbounds(K[j, w]), Val(_XASBP_NK))
    f = @inbounds FE[e]
    phi, A = _xas_phase_amp_flat(k, f)
    @inbounds AE[e, w] = A
    @inbounds PH[e, w] = phi
    for d in 1:D
        s, c = sincos(phi - 2 * pi * @inbounds(TAU[d, w]) * f)
        vr = A * c; vi = A * s
        hr = @inbounds HIr[e, d]; hi = @inbounds HIi[e, d]
        @inbounds UR[e, d, w] = vr * hr - vi * hi
        @inbounds UI[e, d, w] = vr * hi + vi * hr
    end
end

# (b), RB: thread (edge chunk, walker) -> partial (P, Re Q, Im Q) per detector.
@kernel function _xasrb_sums_kernel!(part, @Const(UR), @Const(UI), @Const(Cr), @Const(Ci), @Const(DE), @Const(B0H),
                                     ne, ch, ::Val{D}) where {D}
    c, w = @index(Global, NTuple)
    lo = (c - 1) * ch + 1
    hi = min(c * ch, ne)
    for d in 1:D
        P = 0.0; Qr = 0.0; Qi = 0.0
        for e in lo:hi
            ur = @inbounds UR[e, d, w]; ui = @inbounds UI[e, d, w]
            P += (ur * ur + ui * ui) * @inbounds(DE[e, d])
            if e < ne
                P += @inbounds(B0H[e, d]) * (ur * @inbounds(UR[e+1, d, w]) + ui * @inbounds(UI[e+1, d, w]))
            end
            cr = @inbounds Cr[e, d]; ci = @inbounds Ci[e, d]
            Qr += ur * cr + ui * ci
            Qi += ui * cr - ur * ci
        end
        @inbounds part[1, c, d, w] = P
        @inbounds part[2, c, d, w] = Qr
        @inbounds part[3, c, d, w] = Qi
    end
end

# (d), RB: thread (edge chunk, walker) -> partial bar_K (detector slot 1) and bar_tau_d, AD-free (hand VJP of H0).
@kernel function _xasrb_adjoint_kernel!(partk, @Const(K), @Const(TAU), @Const(BAR), @Const(AE), @Const(PH), @Const(UR),
                                        @Const(UI), @Const(FE), @Const(HIr), @Const(HIi), @Const(Cr), @Const(Ci),
                                        @Const(DE), @Const(B0H), ne, ch, ::Val{D}) where {D}
    c, w = @index(Global, NTuple)
    k = ntuple(jj -> @inbounds(K[jj, w]), Val(_XASBP_NK))
    tau = ntuple(d -> @inbounds(TAU[d, w]), Val(D))
    acc = ntuple(_ -> 0.0, Val(_XASBP_NK))
    btau = ntuple(_ -> 0.0, Val(D))
    lo = (c - 1) * ch + 1
    hi = min(c * ch, ne)
    for e in lo:hi
        f = @inbounds FE[e]
        A = @inbounds AE[e, w]
        phi = @inbounds PH[e, w]
        terms = ntuple(Val(D)) do d
            bP = @inbounds BAR[1, d, w]; bQr = @inbounds BAR[2, d, w]; bQi = @inbounds BAR[3, d, w]
            ur = @inbounds UR[e, d, w]; ui = @inbounds UI[e, d, w]
            cr = @inbounds Cr[e, d]; ci = @inbounds Ci[e, d]
            de2 = 2 * @inbounds(DE[e, d])
            gur = bQr * cr - bQi * ci + bP * de2 * ur
            gui = bQr * ci + bQi * cr + bP * de2 * ui
            if e > 1
                bl = bP * @inbounds(B0H[e-1, d])
                gur += bl * @inbounds(UR[e-1, d, w]); gui += bl * @inbounds(UI[e-1, d, w])
            end
            if e < ne
                br = bP * @inbounds(B0H[e, d])
                gur += br * @inbounds(UR[e+1, d, w]); gui += br * @inbounds(UI[e+1, d, w])
            end
            hr = @inbounds HIr[e, d]; hi_ = @inbounds HIi[e, d]
            gvr = hr * gur + hi_ * gui            # g_v = conj(1/h_ref) g_u
            gvi = hr * gui - hi_ * gur
            s, co = sincos(phi - 2 * pi * tau[d] * f)
            (gvr * co + gvi * s, gvr * s - gvi * co)   # Re, Im of conj(g_v) e^{i theta}
        end
        gA = 0.0; gPhi = 0.0
        for d in 1:D
            gA += terms[d][1]
            gPhi -= A * terms[d][2]
        end
        btau = _tadd(btau, ntuple(d -> 2 * pi * f * A * terms[d][2], Val(D)))   # no closure over the reassigned btau
        acc = _xas_phase_amp_flat_vjp(acc, k, f, gPhi, gA)
    end
    for jj in 1:_XASBP_NK
        @inbounds partk[jj, c, 1, w] = acc[jj]
    end
    for d in 1:D
        @inbounds partk[_XASBP_NK + 1, c, d, w] = btau[d]
    end
end

# ------------------------------------------------------------------------------------------------
# Forward-tangent single pass (`variant = :fwd`, resident, shared grid). K depends on 5 sampled rows only
# (mc, q, chi1, chi2, dL = `_XASBP_JROWS[1:5]`) and tau_d enters the sums only through
#     dQ_d/dtau_d = -2 pi i R_d,   R_d = sum f conj(d) w H0 e^{-2 pi i tau_d f},
# so ONE pass over the bins gives everything the gradient needs: per detector P, Q, R and, along the 5 K tangents
# t_m = dK/dx_m (from `_xasbp_coeffs_jac_kernel!`), dP_m = sum 2 w A dA_m and dQ_m = sum cw e^{i th}(dA_m + i A dPhi_m).
# The per-bin (dPhi_m, dA_m) reuse the hand-written VJP `_xas_phase_amp_flat_vjp` with a unit cotangent and an
# accumulator that contracts each coefficient partial with the tangents (`_XasTanAcc`): the phase reads coefficients
# 1, 5-38 and the amplitude 39-61, disjoint, so gPhi = gA = 1 separates them exactly. Coefficients are read through
# `_XasCol` at their use sites (no 61-tuple hoisted into registers). One work-group per walker reduces in shared
# memory, so no per-chunk partial arrays. Then the ll kernel as before and `_xasbp_fwd_grad_kernel!` (chain rule).
# Replaces the second primal pass, the in-kernel reverse VJP per bin, the adjoint partials and their reduction.
# ------------------------------------------------------------------------------------------------

const _XASBP_NT = 5                                  # K tangent directions: _XASBP_JROWS[1:5]
const _XASBP_NV = 5 + 3 * _XASBP_NT                  # values per detector: P, Qr, Qi, Rr, Ri, (dP, dQr, dQi) x NT
const _XASBP_FG = 64                                 # work-group size (<= 128; shared memory 8 NV D FG bytes)

# Column `w` of a (coefficient x walker) array as an indexable coefficient set.
struct _XasCol{A}
    a::A
    w::Int
end
@inline _XasCol(a, w::Integer) = _XasCol(a, Int(w))     # @index is Int32 on CUDA
@inline Base.getindex(c::_XasCol, j::Int) = @inbounds c.a[j, c.w]

# Accumulator for `_xas_phase_amp_flat_vjp`: `_acc(acc, j, v)` adds v * t_m[j] to the phase (j <= 38) or amplitude
# (j >= 39) tangent of every direction m. `j` is a literal at every call site, so the branch folds.
struct _XasTanAcc{M,TT}
    dp::NTuple{M,Float64}
    da::NTuple{M,Float64}
    t::TT
end
@inline function _acc(a::_XasTanAcc{M}, j::Int, v) where {M}
    if j <= 38
        return _XasTanAcc(ntuple(m -> a.dp[m] + v * a.t[m][j], Val(M)), a.da, a.t)
    else
        return _XasTanAcc(a.dp, ntuple(m -> a.da[m] + v * a.t[m][j], Val(M)), a.t)
    end
end

# (Phi, A) of H0 at f and their derivatives along the tangents `t` (NTuple of coefficient sets).
@inline function _xas_phase_amp_flat_jvp(k, f, t::NTuple{M}) where {M}
    z = ntuple(_ -> 0.0, Val(M))
    ta, phi, A = _xas_pa_core(_XasTanAcc(z, z, t), k, f)
    return phi, A, ta.dp, ta.da
end

# Primal (Phi, A) of `_xas_phase_amp_flat` together with `_xas_phase_amp_flat_vjp(acc, k, f, 1, 1)`, in one pass and
# with no `pow`: every fractional power from one cbrt (and a sqrt for the amplitude's 1/6 powers), the segment
# derivatives written out (GravitationalWaveforms' `_xas_*_deriv` forms, which use x^(p/q)). The library's `_val` forms call
# `x^(p/q)` ~25 times per bin -- a full FP64 pow each on the GPU. Equal to the `pow` forms to rounding.
@inline function _xas_pa_core(acc, k, f)
    fs = max(f, eps(typeof(f)))
    x = k[1] * fs
    phis = 0.0
    if x < k[4]
        gs = 1 / k[5]
        if x < k[2]
            lx = log(x); x13 = cbrt(x); x23 = x13 * x13; x43 = x * x13; x53 = x * x23; x2 = x * x
            x73 = x2 * x13; x83 = x2 * x23; x3 = x2 * x; x103 = x3 * x13; x113 = x3 * x23
            phiN = k[24]
            phi_Ins = k[8] + k[9] * x13 + k[10] * x23 + k[11] * x + k[12] * x43 + k[13] * x53 + k[14] * x53 * lx +
                      k[15] * x2 + k[16] * x2 * lx + k[17] * x73 + k[18] * x83 + k[19] * x83 * lx +
                      k[20] * x83 + k[21] * x3 + k[22] * x103 + k[23] * x113
            dphi_Ins = k[9] / (3 * x23) + k[10] * (2.0 / 3.0) / x13 + k[11] + k[12] * (4.0 / 3.0) * x13 +
                       k[13] * (5.0 / 3.0) * x23 + k[14] * ((5.0 / 3.0) * x23 * lx + x23) + k[15] * 2 * x +
                       k[16] * (2 * x * lx + x) + k[17] * (7.0 / 3.0) * x43 + k[18] * (8.0 / 3.0) * x53 +
                       k[19] * ((8.0 / 3.0) * x53 * lx + x53) + k[20] * (8.0 / 3.0) * x53 + k[21] * 3 * x2 +
                       k[22] * (10.0 / 3.0) * x73 + k[23] * (11.0 / 3.0) * x83
            xm53 = 1 / x53
            seg = phiN * xm53 * phi_Ins
            c = gs * phiN * xm53
            acc = _acc(acc, 8, c); acc = _acc(acc, 9, c * x13); acc = _acc(acc, 10, c * x23); acc = _acc(acc, 11, c * x)
            acc = _acc(acc, 12, c * x43); acc = _acc(acc, 13, c * x53); acc = _acc(acc, 14, c * x53 * lx)
            acc = _acc(acc, 15, c * x2); acc = _acc(acc, 16, c * x2 * lx); acc = _acc(acc, 17, c * x73)
            acc = _acc(acc, 18, c * x83); acc = _acc(acc, 19, c * x83 * lx); acc = _acc(acc, 20, c * x83)
            acc = _acc(acc, 21, c * x3); acc = _acc(acc, 22, c * x103); acc = _acc(acc, 23, c * x113)
            acc = _acc(acc, 24, gs * xm53 * phi_Ins)
            dsdx = phiN * xm53 * (dphi_Ins - (5.0 / 3.0) * phi_Ins / x)
        elseif x < k[3]
            fd = k[6]; fRD = k[7]; cL = k[36]
            u = x - fRD
            z = u / (2 * fd)
            at = atan(z)
            dz = 1 / (1 + z * z)
            ix = 1 / x
            seg = k[25] * x + k[26] * log(x) - k[27] * ix - k[28] * ix * ix / 2 - k[29] * ix * ix * ix / 3 +
                  (2 * cL / fd) * at + k[31] * x + k[30]
            acc = _acc(acc, 25, gs * x); acc = _acc(acc, 26, gs * log(x)); acc = _acc(acc, 27, -gs * ix)
            acc = _acc(acc, 28, -gs * ix * ix / 2); acc = _acc(acc, 29, -gs * ix * ix * ix / 3)
            acc = _acc(acc, 36, gs * (2 / fd) * at)
            acc = _acc(acc, 6, gs * (-(2 * cL / (fd * fd)) * at - (2 * cL / fd) * dz * u / (2 * fd * fd)))
            acc = _acc(acc, 7, -gs * (2 * cL / fd) * dz / (2 * fd))
            acc = _acc(acc, 30, gs); acc = _acc(acc, 31, gs * x)
            dsdx = k[25] + ix * (k[26] + ix * (k[27] + ix * (k[28] + ix * k[29]))) + 4 * cL / (4 * fd * fd + u * u) + k[31]
        else
            fd = k[6]; fRD = k[7]; cL = k[36]
            u = x - fRD
            z = u / fd
            at = atan(z)
            dz = 1 / (1 + z * z)
            x13 = cbrt(x); x23 = x13 * x13
            ix = 1 / x
            seg = k[32] * x + 1.5 * k[33] * x23 - k[34] * ix - (k[35] / 3) * ix * ix * ix + (cL / fd) * at + k[37] + k[38] * x
            acc = _acc(acc, 32, gs * x); acc = _acc(acc, 33, gs * 1.5 * x23); acc = _acc(acc, 34, -gs * ix)
            acc = _acc(acc, 35, -gs * ix * ix * ix / 3); acc = _acc(acc, 36, gs * at / fd)
            acc = _acc(acc, 6, gs * (-(cL / (fd * fd)) * at - (cL / fd) * dz * u / (fd * fd)))
            acc = _acc(acc, 7, -gs * (cL / fd) * dz / fd)
            acc = _acc(acc, 37, gs); acc = _acc(acc, 38, gs * x)
            dsdx = k[32] + k[33] / x13 + k[34] * ix * ix + k[35] * ix * ix * ix * ix + cL / (fd * fd + u * u) + k[38]
        end
        phis = seg * gs
        acc = _acc(acc, 5, -gs * seg * gs)
        acc = _acc(acc, 1, gs * dsdx * fs)
    end
    A = 0.0
    xa = k[39] * fs
    if xa < k[42]
        x13 = cbrt(xa)
        x76 = xa * sqrt(x13)                  # xa^(7/6)
        p = 1 / x76                           # xa^(-7/6)
        if xa < k[40]
            x23 = x13 * x13; x43 = xa * x13; x53 = xa * x23; x2 = xa * xa
            x73 = x2 * x13; x83 = x2 * x23; x3 = x2 * xa
            aimr = 1 + k[45] * x23 + k[46] * xa + k[47] * x43 + k[48] * x53 + k[49] * x2 + k[50] * x73 + k[51] * x83 + k[52] * x3
            ga = k[61] * p
            acc = _acc(acc, 45, ga * x23); acc = _acc(acc, 46, ga * xa); acc = _acc(acc, 47, ga * x43)
            acc = _acc(acc, 48, ga * x53); acc = _acc(acc, 49, ga * x2); acc = _acc(acc, 50, ga * x73)
            acc = _acc(acc, 51, ga * x83); acc = _acc(acc, 52, ga * x3)
            daimr = k[45] * (2.0 / 3.0) / x13 + k[46] + k[47] * (4.0 / 3.0) * x13 + k[48] * (5.0 / 3.0) * x23 +
                    k[49] * 2 * xa + k[50] * (7.0 / 3.0) * x43 + k[51] * (8.0 / 3.0) * x53 + k[52] * 3 * x2
        elseif xa < k[41]
            Pn = k[53] + xa * (k[54] + xa * (k[55] + xa * (k[56] + xa * k[57])))
            dPn = k[54] + xa * (2 * k[55] + xa * (3 * k[56] + xa * 4 * k[57]))
            aimr = x76 / Pn
            ga = k[61] * p
            c = -ga * aimr / Pn
            acc = _acc(acc, 53, c); acc = _acc(acc, 54, c * xa); acc = _acc(acc, 55, c * xa * xa)
            acc = _acc(acc, 56, c * xa * xa * xa); acc = _acc(acc, 57, c * xa * xa * xa * xa)
            daimr = (7.0 / 6.0) * aimr / xa - aimr * dPn / Pn
        else
            g1 = k[58]; g2 = k[59]; g3 = k[60]; fd = k[43]; fRD = k[44]
            u = xa - fRD
            gR = g2 / (fd * g3)
            Dn = u * u + (g3 * fd) * (g3 * fd)
            E = exp(-u * gR)
            aimr = E * (fd * g1 * g3) / Dn
            ga = k[61] * p
            acc = _acc(acc, 58, ga * E * fd * g3 / Dn)
            acc = _acc(acc, 59, ga * aimr * (-u / (fd * g3)))
            acc = _acc(acc, 60, ga * (aimr * (u * g2 / (fd * g3 * g3)) + E * fd * g1 / Dn - aimr * 2 * g3 * fd * fd / Dn))
            acc = _acc(acc, 43, ga * (aimr * (u * g2 / (fd * fd * g3)) + E * g1 * g3 / Dn - aimr * 2 * g3 * g3 * fd / Dn))
            acc = _acc(acc, 44, ga * aimr * (gR + 2 * u / Dn))
            daimr = aimr * (-gR - 2 * u / Dn)
        end
        A = k[61] * p * aimr
        acc = _acc(acc, 61, p * aimr)
        acc = _acc(acc, 39, k[61] * ((-7.0 / 6.0) * p / xa * aimr + p * daimr) * fs)
    end
    return acc, phis, A
end

# One bin's contribution for detector d (`_XASBP_NV` values, the layout of `_xasbp_fwd_kernel!`'s output).
@inline function _xasbp_fwd_terms(phi, A, dphi::NTuple{5}, dA::NTuple{5}, th, f, wt, cr, ci)
    s, c = sincos(th)
    ur = cr * c - ci * s                 # u = cw e^{i th}, cw = conj(d) w
    ui = cr * s + ci * c
    qr = A * ur; qi = A * ui
    w2 = 2 * wt * A
    a1 = A * dphi[1]; a2 = A * dphi[2]; a3 = A * dphi[3]; a4 = A * dphi[4]; a5 = A * dphi[5]
    return (wt * A * A, qr, qi, f * qr, f * qi,
            w2 * dA[1], ur * dA[1] - ui * a1, ui * dA[1] + ur * a1,
            w2 * dA[2], ur * dA[2] - ui * a2, ui * dA[2] + ur * a2,
            w2 * dA[3], ur * dA[3] - ui * a3, ui * dA[3] + ur * a3,
            w2 * dA[4], ur * dA[4] - ui * a4, ui * dA[4] + ur * a4,
            w2 * dA[5], ur * dA[5] - ui * a5, ui * dA[5] + ur * a5)
end

_inl(ex) = Expr(:block, Expr(:meta, :inline), ex)       # body of an inlined @generated function
# Per-detector helpers are @generated (literal d) rather than `ntuple(d -> ..., Val(D))`: on CUDA those closures were
# not inlined and every bin built the closure and its 20-value result on the stack (2 KB stack frame).
@generated _tadd2(a::NTuple{D}, b::NTuple{D}) where {D} = _inl(Expr(:tuple, (:(_tadd(a[$d], b[$d])) for d in 1:D)...))

# Closure-free accessors for the kernels (a closure over a kernel local that is reassigned, or that KA's @synchronize
# lowering touches, is boxed -> dynamic calls on the GPU).
@inline _xasbp_tancols(JT, w) = ntuple(m -> _XasCol(JT[m], w), Val(_XASBP_NT))
@inline _xasbp_taus(KT, w, ::Val{D}) where {D} = ntuple(d -> @inbounds(KT[_XASBP_NK + d, w]), Val(D))
@inline _xasbp_zacc(::Val{N}, ::Val{D}) where {N,D} = ntuple(_ -> ntuple(_ -> 0.0, Val(N)), Val(D))
@generated function _xasbp_fwd_bin(phi, A, dphi, dA, tau::NTuple{D}, f, W, CDr, CDi, i) where {D}
    _inl(Expr(:tuple, (:(_xasbp_fwd_terms(phi, A, dphi, dA, phi - 2 * pi * f * tau[$d], f, @inbounds(W[i, $d]),
                                          @inbounds(CDr[i, $d]), @inbounds(CDi[i, $d]))) for d in 1:D)...))
end

@generated function _xasbp_store_acc!(L, lid, acc::NTuple{D,Tuple}) where {D}
    N = fieldcount(fieldtype(acc, 1))                # (N from the type: NTuple{D,NTuple{N}} leaves N unbound at D = 0)
    Expr(:block, Expr(:meta, :inline), (:(@inbounds L[$((d - 1) * N + v), lid] = acc[$d][$v]) for d in 1:D for v in 1:N)...,
         :(return nothing))
end

@inline function _xasbp_red_step!(L, lid, s, nv)
    if 2 * s <= _XASBP_FG && lid <= s
        for v in 1:nv
            @inbounds L[v, lid] += L[v, lid + s]
        end
    end
    return nothing
end

# Work-group (`_XASBP_FG` threads) = walker. SUMS[(d-1) * NV + v, w] (see `_xasbp_fwd_terms`).
@kernel function _xasbp_fwd_kernel!(SUMS, @Const(KT), JT, @Const(F1), @Const(W), @Const(CDr), @Const(CDi),
                                    ::Val{D}) where {D}
    w = @index(Group, Linear)
    lid = @index(Local, Linear)
    L = @localmem Float64 (_XASBP_NV * D, _XASBP_FG)
    nb = length(F1)
    kc = _XasCol(KT, w)
    tc = _xasbp_tancols(JT, w)
    tau = _xasbp_taus(KT, w, Val(D))
    acc = _xasbp_zacc(Val(_XASBP_NV), Val(D))
    i = lid
    while i <= nb
        f = @inbounds F1[i]
        phi, A, dphi, dA = _xas_phase_amp_flat_jvp(kc, f, tc)
        acc = _tadd2(acc, _xasbp_fwd_bin(phi, A, dphi, dA, tau, f, W, CDr, CDi, i))
        i += _XASBP_FG
    end
    _xasbp_store_acc!(L, lid, acc)
    @synchronize
    _xasbp_red_step!(L, lid, 64, _XASBP_NV * D)
    @synchronize
    _xasbp_red_step!(L, lid, 32, _XASBP_NV * D)
    @synchronize
    _xasbp_red_step!(L, lid, 16, _XASBP_NV * D)
    @synchronize
    _xasbp_red_step!(L, lid, 8, _XASBP_NV * D)
    @synchronize
    _xasbp_red_step!(L, lid, 4, _XASBP_NV * D)
    @synchronize
    _xasbp_red_step!(L, lid, 2, _XASBP_NV * D)
    @synchronize
    _xasbp_red_step!(L, lid, 1, _XASBP_NV * D)
    @synchronize
    if lid <= _XASBP_NV * D
        @inbounds SUMS[lid, w] = L[lid, 1]
    end
end

# ---- Relative binning, forward-tangent (`variant = :fwd` with `relative_binning`) ----
# u_e = A e^{i th} / h_ref(f_e) per detector, du_m = (dA_m + i A dPhi_m) e^{i th} / h_ref. Per detector: P, Q = sum u conj(C),
# R = sum f u conj(C), the 5 (dP, dQ) tangents and Pt = dP/dtau (the neighbour term Re(u_e conj u_{e+1}) depends on tau).
# Thread = contiguous edge chunk; the pair (hi, hi + 1) uses the next thread's first edge, exchanged in shared memory.
const _XASRB_NU = 2 + 2 * _XASBP_NT                  # per detector and edge: ur, ui, (dur, dui) x NT
const _XASRB_NV = _XASBP_NV + 1                      # _XASBP_NV layout + Pt

@generated function _xasrb_fwd_vals(kc, tc, tau::NTuple{D}, f, HIr, HIi, e) where {D}
    quote
        $(Expr(:meta, :inline))
        phi, A, dphi, dA = _xas_phase_amp_flat_jvp(kc, f, tc)
        $(Expr(:tuple, (:(_xasrb_fwd_u(phi, A, dphi, dA, tau[$d], f, @inbounds(HIr[e, $d]), @inbounds(HIi[e, $d])))
                        for d in 1:D)...))
    end
end

@inline function _xasrb_fwd_u(phi, A, dphi, dA, tau_d, f, hr, hi)
    s, c = sincos(phi - 2 * pi * tau_d * f)
    ehr = c * hr - s * hi; ehi = c * hi + s * hr          # e^{i th} / h_ref
    b1 = A * dphi[1]; b2 = A * dphi[2]; b3 = A * dphi[3]; b4 = A * dphi[4]; b5 = A * dphi[5]
    (A * ehr, A * ehi,
     dA[1] * ehr - b1 * ehi, dA[1] * ehi + b1 * ehr, dA[2] * ehr - b2 * ehi, dA[2] * ehi + b2 * ehr,
     dA[3] * ehr - b3 * ehi, dA[3] * ehi + b3 * ehr, dA[4] * ehr - b4 * ehi, dA[4] * ehi + b4 * ehr,
     dA[5] * ehr - b5 * ehi, dA[5] * ehi + b5 * ehr)
end

@inline function _xasrb_edge_inc(u::NTuple{_XASRB_NU}, f, cr, ci, de)
    ur = u[1]; ui = u[2]
    qr = ur * cr + ui * ci; qi = ui * cr - ur * ci
    t(m) = (2 * (ur * u[1+2m] + ui * u[2+2m]) * de, u[1+2m] * cr + u[2+2m] * ci, u[2+2m] * cr - u[1+2m] * ci)
    return ((ur * ur + ui * ui) * de, qr, qi, f * qr, f * qi, t(1)..., t(2)..., t(3)..., t(4)..., t(5)..., 0.0)
end

@inline function _xasrb_cross_inc(a::NTuple{_XASRB_NU}, b::NTuple{_XASRB_NU}, B, fa, fb)
    t(m) = (B * (a[1+2m] * b[1] + a[2+2m] * b[2] + a[1] * b[1+2m] + a[2] * b[2+2m]), 0.0, 0.0)
    return (B * (a[1] * b[1] + a[2] * b[2]), 0.0, 0.0, 0.0, 0.0, t(1)..., t(2)..., t(3)..., t(4)..., t(5)...,
            B * 2 * pi * (fa - fb) * (a[2] * b[1] - a[1] * b[2]))
end

@generated _xasrb_edge_all(acc::NTuple{D}, u::NTuple{D}, f, Cr, Ci, DE, e) where {D} =
    _inl(Expr(:tuple, (:(_tadd(acc[$d], _xasrb_edge_inc(u[$d], f, @inbounds(Cr[e, $d]), @inbounds(Ci[e, $d]),
                                                        @inbounds(DE[e, $d])))) for d in 1:D)...))
@generated _xasrb_cross_all(acc::NTuple{D}, a::NTuple{D}, b::NTuple{D}, B0H, e, fa, fb) where {D} =
    _inl(Expr(:tuple, (:(_tadd(acc[$d], _xasrb_cross_inc(a[$d], b[$d], @inbounds(B0H[e, $d]), fa, fb))) for d in 1:D)...))

@generated function _xasrb_store_priv!(P, acc::NTuple{D,Tuple}) where {D}
    N = fieldcount(fieldtype(acc, 1))
    Expr(:block, Expr(:meta, :inline), (:(@inbounds P[$((d - 1) * N + v)] = acc[$d][$v]) for d in 1:D for v in 1:N)...,
         :(return nothing))
end

@generated function _xasrb_load_u(X, col, ::Val{D}) where {D}
    _inl(Expr(:tuple, (Expr(:tuple, (:(@inbounds X[$((d - 1) * _XASRB_NU + v), col]) for v in 1:_XASRB_NU)...) for d in 1:D)...))
end

@inline _xasrb_chunk(lid, ne) = (ch = cld(ne, _XASBP_FG); ((lid - 1) * ch + 1, min(lid * ch, ne)))

@kernel function _xasrb_fwd_kernel!(SUMS, @Const(KT), JT, @Const(FE), @Const(HIr), @Const(HIi), @Const(Cr), @Const(Ci),
                                    @Const(DE), @Const(B0H), ne, ::Val{D}) where {D}
    w = @index(Group, Linear)
    lid = @index(Local, Linear)
    # One buffer for the first-edge exchange (rows 1:NU*D) and, after a barrier, the reduction (rows 1:NV*D):
    # 21.5 KB instead of 33.8 KB per group -> twice the resident groups per SM on sm_70.
    L = @localmem Float64 (_XASRB_NV * D, _XASBP_FG)
    P = @private Float64 (_XASRB_NV * D,)            # the accumulator across the barrier below (KA CPU backend)
    lo, hi = _xasrb_chunk(lid, ne)
    if lo <= hi
        u = _xasrb_fwd_vals(_XasCol(KT, w), _xasbp_tancols(JT, w), _xasbp_taus(KT, w, Val(D)), @inbounds(FE[lo]),
                            HIr, HIi, lo)
        _xasbp_store_acc!(L, lid, u)
    end
    @synchronize
    lo2, hi2 = _xasrb_chunk(lid, ne)
    acc = _xasbp_zacc(Val(_XASRB_NV), Val(D))
    if lo2 <= hi2
        kc = _XasCol(KT, w)
        tc = _xasbp_tancols(JT, w)
        tau2 = _xasbp_taus(KT, w, Val(D))
        prev = _xasrb_load_u(L, lid, Val(D))
        fp = @inbounds FE[lo2]
        acc = _xasrb_edge_all(acc, prev, fp, Cr, Ci, DE, lo2)
        e = lo2 + 1
        while e <= hi2
            f = @inbounds FE[e]
            cur = _xasrb_fwd_vals(kc, tc, tau2, f, HIr, HIi, e)
            acc = _xasrb_edge_all(acc, cur, f, Cr, Ci, DE, e)
            acc = _xasrb_cross_all(acc, prev, cur, B0H, e - 1, fp, f)
            prev = cur; fp = f
            e += 1
        end
        if hi2 < ne
            acc = _xasrb_cross_all(acc, prev, _xasrb_load_u(L, lid + 1, Val(D)), B0H, hi2, fp, @inbounds(FE[hi2 + 1]))
        end
    end
    _xasrb_store_priv!(P, acc)
    @synchronize                                      # every thread has read its neighbour's edge
    for v in 1:(_XASRB_NV * D)
        @inbounds L[v, lid] = P[v]
    end
    @synchronize
    _xasbp_red_step!(L, lid, 64, _XASRB_NV * D)
    @synchronize
    _xasbp_red_step!(L, lid, 32, _XASRB_NV * D)
    @synchronize
    _xasbp_red_step!(L, lid, 16, _XASRB_NV * D)
    @synchronize
    _xasbp_red_step!(L, lid, 8, _XASRB_NV * D)
    @synchronize
    _xasbp_red_step!(L, lid, 4, _XASRB_NV * D)
    @synchronize
    _xasbp_red_step!(L, lid, 2, _XASRB_NV * D)
    @synchronize
    _xasbp_red_step!(L, lid, 1, _XASRB_NV * D)
    @synchronize
    if lid <= _XASRB_NV * D
        @inbounds SUMS[lid, w] = L[lid, 1]
    end
end

# G[:, w] = GX[:, w] + sum_d bar(P, Qr, Qi)_d . d(P, Qr, Qi)_d / dx, with d/dx from the K tangents (rows 1:5 of
# `_XASBP_JROWS`) and the tau tangents (all 8): dQ_d = -2 pi i R_d dtau_d (and, for relative binning, dP_d = Pt_d dtau_d).
# Row 12 of G = LL (one device-to-host copy for gradient and log-likelihood).
@kernel function _xasbp_fwd_grad_kernel!(G, @Const(GX), JT, @Const(BAR), @Const(S3), @Const(LL), ::Val{D}) where {D}
    w = @index(Global, Linear)
    @inbounds G[12, w] = LL[w]
    for k in 1:11
        @inbounds G[k, w] = GX[k, w]
    end
    for i in 1:_XASBP_NJ
        J = JT[i]
        s = 0.0
        for d in 1:D
            dtau = @inbounds J[_XASBP_NK + d, w]
            dP = size(S3, 1) > _XASBP_NV ? @inbounds(S3[_XASRB_NV, d, w]) * dtau : 0.0
            dQr = 2 * pi * @inbounds(S3[5, d, w]) * dtau
            dQi = -2 * pi * @inbounds(S3[4, d, w]) * dtau
            if i <= _XASBP_NT
                b = 5 + 3 * (i - 1)
                dP += @inbounds S3[b + 1, d, w]
                dQr += @inbounds S3[b + 2, d, w]
                dQi += @inbounds S3[b + 3, d, w]
            end
            s += @inbounds(BAR[1, d, w]) * dP + @inbounds(BAR[2, d, w]) * dQr + @inbounds(BAR[3, d, w]) * dQi
        end
        @inbounds G[_XASBP_JROWS[i], w] += s
    end
end

"""
    xas_binparallel_logp_and_grad(data, cfg, freqs; backend, nwalkers, chunk=4) -> logp_and_grad!

IMRPhenomXAS log-likelihood and gradient with the frequency-bin sums on `backend` (e.g. `CUDABackend()`,
or `KernelAbstractions.CPU()` for testing) and the per-walker work on the host (threads). Same likelihood
options, time frame and 11-row sampled parametrization as `gw_logp_and_grad(...; model=IMRPhenomXAS())`.
The returned closure takes HOST `grad_out`, `logp_out`, `positions` (like `gw_logp_and_grad_gpu_staged`)
and returns the log-likelihood plus the package's box prior, untempered (wrap with
`tempered_logp_and_grad` for beta and a `GWPrior`). All detectors must have the same number of bins.
`shared` (default: whether all detectors have identical frequency grids): one H0 evaluation and one
per-bin VJP for all detectors. `adjoint` (shared grid only): `:enzyme` (default, in-kernel Enzyme VJP per bin)
or `:hand` (`_xas_phase_amp_flat_vjp`, no AD in the kernels; same speed on the GTX 1650 Ti, where the adjoint is
FP64-bound, but ~4x longer first compile, ~135 s). `resident = true`: the per-walker stages (a), (c), (e) run as
device kernels too (forward-mode coefficient Jacobian, see `_xasbp_coeffs_jac_kernel!`); only positions and results
cross the bus. `relative_binning = rb` (from `build_xas_relative_binning`, same bin edges in every detector; implies
`resident`): the device evaluates H0 only at the bin edges and uses the summary data -- same numbers as the CPU
`gw_logp_and_grad(...; relative_binning=rb)` up to rounding. `timers`: optional `Vector{Float64}` of length 5, accumulating seconds spent in steps (a)-(e) (synchronizes).
`variant = :fwd` (implies `resident`; identical frequency grids, D <= 3, or D <= 2 with relative binning): forward-tangent
single pass over the bins (or edges) instead of sums + adjoint -- the fastest path, used by the driver (V100, H1+L1 all
options: 0.6-0.8 ms per call up to 256 walkers, 4.3 ms dense / 3.3 ms relative binning at 4096). `jac`
(resident): how the coefficient Jacobian is split across threads, `:split5` (default, fastest), `:split3`, `:split` or
`:width8` (one width-8 forward pass per walker). `llpar` (resident, distance marginalization): evaluate
the 64 quadrature nodes in parallel.
"""
function xas_binparallel_logp_and_grad(data, cfg::PEConfig{T}, freqs; backend, nwalkers::Int,
                                       chunk::Int=4, model=IMRPhenomXAS(), timers=nothing,
                                       shared::Union{Nothing,Bool}=nothing, adjoint::Symbol=:enzyme, resident::Bool=false,
                                       relative_binning=nothing, variant::Symbol=:adjoint,
                                       jac::Symbol=:split5, llpar::Bool=true, jacwg::Int=32) where {T}
    useRB = relative_binning !== nothing
    resident = resident | useRB | (variant === :fwd)
    variant in (:adjoint, :fwd) || throw(ArgumentError("variant must be :adjoint or :fwd"))
    adjoint in (:hand, :enzyme) || throw(ArgumentError("adjoint must be :hand or :enzyme"))
    dets = data isa AbstractVector ? collect(data) : [data]
    D = length(dets)
    nb = length(dets[1].freqs)
    all(d -> length(d.freqs) == nb, dets) || throw(ArgumentError("bin-parallel XAS needs equal bin counts per detector"))
    0 <= cfg.time_reference <= D || throw(ArgumentError("time_reference=$(cfg.time_reference) but only $D detector(s)"))
    g = _xasbp_geometry(dets, cfg)
    lik = KernelFused._lik_opts(cfg)
    dev(a) = (b = KernelAbstractions.allocate(backend, Float64, size(a)); copyto!(b, a); b)
    Fh = reduce(hcat, [Float64.(d.freqs) for d in dets])
    Wh = reduce(hcat, [[p == 0 ? 0.0 : 1 / p for p in d.psd] for d in dets])
    CD = reduce(hcat, [[d.psd[i] == 0 ? 0.0im : conj(d.strain[i]) / d.psd[i] for i in eachindex(d.psd)] for d in dets])
    F = dev(Fh); W = dev(Wh); CDr = dev(real.(CD)); CDi = dev(imag.(CD))
    nch = cld(nb, chunk)
    NK = _XASBP_NK
    Kh = zeros(NK, nwalkers); TAUh = zeros(D, nwalkers); BARh = zeros(3, D, nwalkers)
    Kd = dev(Kh); TAUd = dev(TAUh); BARd = dev(BARh)
    part = KernelAbstractions.allocate(backend, Float64, 3, nch, D, nwalkers)
    partk = KernelAbstractions.allocate(backend, Float64, NK + 1, nch, D, nwalkers)
    S = KernelAbstractions.zeros(backend, Float64, nch * D * nwalkers)
    dS = KernelAbstractions.zeros(backend, Float64, nch * D * nwalkers)
    same_grid = all(d -> d.freqs == dets[1].freqs, dets)
    shared = shared === nothing ? same_grid : shared
    (shared && !same_grid) && throw(ArgumentError("shared=true needs identical frequency grids"))
    F1 = dev(Fh[:, 1])
    shared && fill!(partk, 0.0)
    sums! = shared ? _xasbp_sums_shared_kernel!(backend) : _xasbp_sums_kernel!(backend)
    hand = shared && adjoint === :hand
    adj! = !shared ? _xasbp_adjoint_kernel!(backend) : hand ? _xasbp_adjoint_hand_kernel!(backend) :
           _xasbp_adjoint_shared_kernel!(backend)
    gx = zeros(11, nwalkers)
    gpu = Val(!(backend isa KernelAbstractions.CPU))
    if resident
        POSd = KernelAbstractions.allocate(backend, Float64, 11, nwalkers)
        KT = KernelAbstractions.zeros(backend, Float64, NK + D, nwalkers)
        JT = ntuple(_ -> KernelAbstractions.zeros(backend, Float64, NK + D, nwalkers), _XASBP_NJ)
        LLd = KernelAbstractions.zeros(backend, Float64, nwalkers)
        SCd = KernelAbstractions.zeros(backend, Float64, nwalkers)
        dSCd = KernelAbstractions.zeros(backend, Float64, nwalkers)
        GXd = KernelAbstractions.zeros(backend, Float64, 11, nwalkers)
        Gd = KernelAbstractions.zeros(backend, Float64, 11, nwalkers)
        REDd = KernelAbstractions.allocate(backend, Float64, 3, 1, D, nwalkers)
        REDKd = KernelAbstractions.allocate(backend, Float64, NK + 1, 1, D, nwalkers)
        posh = zeros(11, nwalkers); llh = zeros(nwalkers); gh = zeros(11, nwalkers); gh12 = zeros(12, nwalkers)
        Gd12 = KernelAbstractions.zeros(backend, Float64, 12, nwalkers)
        # page-locked host buffers: the per-call copies are then true async DMA (~0.1 ms per call at small walker counts)
        foreach(a -> KernelAbstractions.pagelock!(backend, a), (posh, llh, gh, gh12))
        coeffs_jac! = _xasbp_coeffs_jac_kernel!(backend, 32)
        # Jacobian kernels with a DYNAMIC workgroup size (given at launch): a static size emits PTX `.maxntid`, which
        # made ptxas cap the registers of these spill-heavy kernels (split: 168 regs / 32 KB spill stores with
        # .maxntid 32, 255 / 20 KB without).
        coeffs_jacs! = _xasbp_coeffs_jac_split_kernel!(backend)
        coeffs_jac3! = _xasbp_coeffs_jac_split3_kernel!(backend)
        coeffs_jac5! = _xasbp_coeffs_jac_split5_kernel!(backend)
        TSd = KernelAbstractions.zeros(backend, Float64, NK + D, nwalkers)
        ll1! = _xasbp_ll_kernel!(backend, 32)
        llnodes = lik.margdist && llpar
        if llnodes
            ENd = KernelAbstractions.zeros(backend, Float64, 4, _XASBP_NDIST, nwalkers)
            nodes! = _xasbp_llnodes_kernel!(backend, (_XASBP_NDIST, 1))
            ll2! = _xasbp_ll2_kernel!(backend, 32)
        end
        # (c) with the distance-marginalization nodes in parallel when that option is on
        ll! = function (LLa, SCa, dSCa, GXa, BARa, POSa, REDa, ga, lika, Dv, gpua; ndrange)
            if llnodes
                nodes!(ENd, POSa, REDa, ga, lika, Dv; ndrange=(_XASBP_NDIST, ndrange))
                ll2!(LLa, SCa, dSCa, GXa, BARa, POSa, REDa, ENd, ga, lika, Dv, gpua; ndrange=ndrange)
            else
                ll1!(LLa, SCa, dSCa, GXa, BARa, POSa, REDa, ga, lika, Dv, gpua; ndrange=ndrange)
            end
        end
        grad! = _xasbp_grad_kernel!(backend, 64)
        red3(a) = reshape(a, size(a, 1), D, nwalkers)
    end
    usefwd = variant === :fwd && !useRB
    rbfwd = variant === :fwd && useRB
    if rbfwd
        D <= 2 || throw(ArgumentError("variant = :fwd with relative binning supports up to 2 detectors"))
        SUMSd = KernelAbstractions.zeros(backend, Float64, _XASRB_NV * D, nwalkers)
        S3 = reshape(SUMSd, _XASRB_NV, D, nwalkers)
        rbfwd! = _xasrb_fwd_kernel!(backend, _XASBP_FG)
        fgrad! = _xasbp_fwd_grad_kernel!(backend, 64)
    end
    if usefwd
        shared || throw(ArgumentError("variant = :fwd needs identical frequency grids (shared)"))
        D <= 3 || throw(ArgumentError("variant = :fwd supports up to 3 detectors"))
        SUMSd = KernelAbstractions.zeros(backend, Float64, _XASBP_NV * D, nwalkers)
        S3 = reshape(SUMSd, _XASBP_NV, D, nwalkers)
        fwd! = _xasbp_fwd_kernel!(backend, _XASBP_FG)
        fgrad! = _xasbp_fwd_grad_kernel!(backend, 64)
    end
    if useRB
        rc = _xasrb_coeffs(relative_binning, g)
        ne = rc.ne
        FEd = dev(rc.FE); HIrd = dev(rc.HIr); HIid = dev(rc.HIi); Crd = dev(rc.Cr); Cid = dev(rc.Ci)
        DEd = dev(rc.DE); B0Hd = dev(rc.B0H)
        AEd = KernelAbstractions.zeros(backend, Float64, ne, nwalkers)
        PHd = KernelAbstractions.zeros(backend, Float64, ne, nwalkers)
        URd = KernelAbstractions.zeros(backend, Float64, ne, D, nwalkers)
        UId = KernelAbstractions.zeros(backend, Float64, ne, D, nwalkers)
        nchr = cld(ne, chunk)
        partr = KernelAbstractions.zeros(backend, Float64, 3, nchr, D, nwalkers)
        partkr = KernelAbstractions.zeros(backend, Float64, NK + 1, nchr, D, nwalkers)   # slots d > 1 of rows 1:NK stay 0
        rbeval! = _xasrb_eval_kernel!(backend, 64)
        rbsums! = _xasrb_sums_kernel!(backend, 64)
        rbadj! = _xasrb_adjoint_kernel!(backend, 32)
    end
    x_of(positions, w) = ntuple(k -> Float64(positions[k, w]), 11)

    return function (grad_out::AbstractMatrix{T}, logp_out::AbstractVector{T}, positions::AbstractMatrix{T}, active_mask)
        size(positions, 2) == nwalkers || throw(ArgumentError("built for $nwalkers walkers, got $(size(positions, 2))"))
        tick = time_ns()
        lap!(k) = timers === nothing || (KernelAbstractions.synchronize(backend); t = time_ns();
                                          timers[k] += (t - tick) / 1e9; tick = t)
        if resident
            posh .= positions
            copyto!(POSd, posh)
            if jac === :split5
                coeffs_jac5!(KT, JT, TSd, POSd, model, g, lik, gpu; ndrange=(nwalkers, 5), workgroupsize=(jacwg, 1))
            elseif jac === :split3
                coeffs_jac3!(KT, JT, TSd, POSd, model, g, lik, gpu; ndrange=(nwalkers, 3), workgroupsize=(jacwg, 1))
            elseif jac === :split
                coeffs_jacs!(KT, JT, TSd, POSd, model, g, lik, gpu; ndrange=nwalkers, workgroupsize=jacwg)
            else
                coeffs_jac!(KT, JT, POSd, model, g, lik, gpu; ndrange=nwalkers)
            end
            lap!(1)
            Kv = view(KT, 1:NK, :); TAUv = view(KT, NK+1:NK+D, :)
            if rbfwd
                rbfwd!(SUMSd, KT, JT, FEd, HIrd, HIid, Crd, Cid, DEd, B0Hd, ne, Val(D); ndrange=_XASBP_FG * nwalkers)
                lap!(2)
                ll!(LLd, SCd, dSCd, GXd, BARd, POSd, S3, g, lik, Val(D), gpu; ndrange=nwalkers)
                lap!(3)
                lap!(4)
                fgrad!(Gd12, GXd, JT, BARd, S3, LLd, Val(D); ndrange=nwalkers)
                copyto!(gh12, Gd12)
                lap!(5)
                grad_out .= view(gh12, 1:11, :)
                logp_out .= view(gh12, 12, :) .+ _call_log_prior_vec(positions)
                return nothing
            end
            if useRB
                rbeval!(AEd, PHd, URd, UId, Kv, TAUv, FEd, HIrd, HIid, Val(D); ndrange=(ne, nwalkers))
                rbsums!(partr, URd, UId, Crd, Cid, DEd, B0Hd, ne, chunk, Val(D); ndrange=(nchr, nwalkers))
                sum!(REDd, partr)
                lap!(2)
                ll!(LLd, SCd, dSCd, GXd, BARd, POSd, red3(REDd), g, lik, Val(D), gpu; ndrange=nwalkers)
                lap!(3)
                rbadj!(partkr, Kv, TAUv, BARd, AEd, PHd, URd, UId, FEd, HIrd, HIid, Crd, Cid, DEd, B0Hd, ne, chunk, Val(D);
                       ndrange=(nchr, nwalkers))
                sum!(REDKd, partkr)
                lap!(4)
                grad!(Gd, GXd, JT, red3(REDKd), Val(D); ndrange=nwalkers)
                copyto!(gh, Gd); copyto!(llh, LLd)
                lap!(5)
                grad_out .= gh
                logp_out .= llh .+ _call_log_prior_vec(positions)
                return nothing
            end
            if usefwd
                fwd!(SUMSd, KT, JT, F1, W, CDr, CDi, Val(D); ndrange=_XASBP_FG * nwalkers)
                lap!(2)
                ll!(LLd, SCd, dSCd, GXd, BARd, POSd, S3, g, lik, Val(D), gpu; ndrange=nwalkers)
                lap!(3)
                lap!(4)
                fgrad!(Gd12, GXd, JT, BARd, S3, LLd, Val(D); ndrange=nwalkers)
                copyto!(gh12, Gd12)
                lap!(5)
                grad_out .= view(gh12, 1:11, :)
                logp_out .= view(gh12, 12, :) .+ _call_log_prior_vec(positions)
                return nothing
            end
            if shared
                sums!(part, Kv, TAUv, F1, W, CDr, CDi, chunk, Val(D); ndrange=(nch, nwalkers))
            else
                sums!(part, Kv, TAUv, F, W, CDr, CDi, chunk; ndrange=(nch, D, nwalkers))
            end
            sum!(REDd, part)
            lap!(2)
            ll!(LLd, SCd, dSCd, GXd, BARd, POSd, red3(REDd), g, lik, Val(D), gpu; ndrange=nwalkers)
            lap!(3)
            if hand
                adj!(partk, Kv, TAUv, BARd, F1, W, CDr, CDi, chunk, Val(D); ndrange=(nch, nwalkers))
            elseif shared
                adj!(partk, S, dS, Kv, TAUv, BARd, F1, W, CDr, CDi, chunk, Val(D), gpu; ndrange=(nch, nwalkers))
            else
                adj!(partk, S, dS, Kv, TAUv, BARd, F, W, CDr, CDi, chunk, gpu; ndrange=(nch, D, nwalkers))
            end
            sum!(REDKd, partk)
            lap!(4)
            grad!(Gd, GXd, JT, red3(REDKd), Val(D); ndrange=nwalkers)
            copyto!(gh, Gd); copyto!(llh, LLd)
            lap!(5)
            grad_out .= gh
            logp_out .= llh .+ _call_log_prior_vec(positions)
            return nothing
        end
        # (a)
        Threads.@threads for w in 1:nwalkers
            K, tau = _xasbp_coeffs_tau(x_of(positions, w)..., model, g, lik)
            @inbounds for j in 1:NK; Kh[j, w] = K[j]; end
            @inbounds for d in 1:D; TAUh[d, w] = tau[d]; end
        end
        copyto!(Kd, Kh); copyto!(TAUd, TAUh)
        lap!(1)
        # (b)
        if shared
            sums!(part, Kd, TAUd, F1, W, CDr, CDi, chunk, Val(D); ndrange=(nch, nwalkers))
        else
            sums!(part, Kd, TAUd, F, W, CDr, CDi, chunk; ndrange=(nch, D, nwalkers))
        end
        red = Array(dropdims(sum(part; dims=2); dims=2))       # 3 x D x nwalkers
        lap!(2)
        # (c)
        Threads.@threads for w in 1:nwalkers
            x = x_of(positions, w)
            P = ntuple(d -> red[1, d, w], D); Qr = ntuple(d -> red[2, d, w], D); Qi = ntuple(d -> red[3, d, w], D)
            res = Enzyme.autodiff(Enzyme.ReverseWithPrimal, _xasbp_ll_from_sums, Enzyme.Active,
                map(Enzyme.Active, x)..., Enzyme.Active(P), Enzyme.Active(Qr), Enzyme.Active(Qi),
                Enzyme.Const(g), Enzyme.Const(lik))
            gr = res[1]
            logp_out[w] = T(res[2])
            @inbounds for k in 1:11
                gx[k, w] = gr[k] === nothing ? 0.0 : gr[k]
            end
            @inbounds for d in 1:D
                BARh[1, d, w] = gr[12][d]; BARh[2, d, w] = gr[13][d]; BARh[3, d, w] = gr[14][d]
            end
        end
        copyto!(BARd, BARh)
        lap!(3)
        # (d)
        if hand
            adj!(partk, Kd, TAUd, BARd, F1, W, CDr, CDi, chunk, Val(D); ndrange=(nch, nwalkers))
        elseif shared
            adj!(partk, S, dS, Kd, TAUd, BARd, F1, W, CDr, CDi, chunk, Val(D), gpu; ndrange=(nch, nwalkers))
        else
            adj!(partk, S, dS, Kd, TAUd, BARd, F, W, CDr, CDi, chunk, gpu; ndrange=(nch, D, nwalkers))
        end
        redk = Array(dropdims(sum(partk; dims=2); dims=2))     # (NK+1) x D x nwalkers
        lap!(4)
        # (e)
        Threads.@threads for w in 1:nwalkers
            barK = ntuple(j -> sum(redk[j, d, w] for d in 1:D), NK)
            bartau = ntuple(d -> redk[NK + 1, d, w], D)
            res = Enzyme.autodiff(Enzyme.Reverse, _xasbp_seed, Enzyme.Active,
                map(Enzyme.Active, x_of(positions, w))..., Enzyme.Const(model), Enzyme.Const(g), Enzyme.Const(lik),
                Enzyme.Const(barK), Enzyme.Const(bartau))
            gr = res[1]
            @inbounds for k in 1:11
                grad_out[k, w] = T(gx[k, w] + (gr[k] === nothing ? 0.0 : gr[k]))
            end
        end
        lap!(5)
        # box prior, as in gw_logp_and_grad (log-likelihood + 0 inside the box, -Inf outside)
        logp_out .+= _call_log_prior_vec(positions)
        return nothing
    end
end
