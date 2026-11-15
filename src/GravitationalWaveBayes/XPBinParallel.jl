# ------------------------------------------------------------------------------------------------
# IMRPhenomXP on the device: the mode-sum bin-parallel path of `XHMBinParallel.jl` with 5 "modes".
#
# `_xp_twist_up22` with alpha = Dalpha(f) + alpha0, epsilon = Depsilon(f) + epsilon0 (Dalpha(f) = alpha_NNLO(v) -
# alpha_NNLO(v_ref), likewise epsilon) and the symmetry d^2_{m,-2} = (-1)^m d^2_{-m,2} gives, after the zeta rotation
# and the antenna response,
#
#     h_d(f) = sum_{m=-2..2} c_md G_m(f) e^{-2 pi i f tau_d},
#     G_m(f) = (N / 2) A(f) e^{i (Phi(f) - 2 Depsilon(f) + m Dalpha(f))} d^2_{m,2}(beta(f)),
#     c_md = e^{-i phase} e^{-2 i epsilon0} e^{i m alpha0} [Fp' (Y_m + (-1)^m Y_-m) + i Fc' ((-1)^m Y_-m - Y_m)] / dL,
#
# with A e^{i Phi} the XAS (2,2) mode at dL = 1, t_c = phase = 0, N = `_XP_HAS_MODE_NORMALIZATION`, Y_m =
# -2Y_2m(thetaJN, 0), Fp' = Fp cos 2 zeta - Fc sin 2 zeta, Fc' = Fp sin 2 zeta + Fc cos 2 zeta. G_m depends on
# (M, eta, chi1z, chi2z) and chi_p only (beta, Dalpha and Depsilon through the NNLO coefficients, Sperp = chi_p m1^2;
# L(v) and v_ref through M and f_ref): theta = (mc, q, chi1, chi2, chi_p), 5 tangents per bin. Inclination, phase and
# the in-plane spin DIRECTIONS enter only through the constants thetaJN, alpha0, epsilon0, zeta in c_md, computed on the
# host by the same `_xp_precession_setup` as the CPU likelihood (`KernelFused._xprec_ll_given_physical`).
# ------------------------------------------------------------------------------------------------

struct _XPSpec end
_mbp_model(::_XPSpec) = IMRPhenomXP()
_mbp_nm(::_XPSpec) = 5                               # m = -2, -1, 0, 1, 2
_mbp_nt(::_XPSpec) = 5                               # mc, q, chi1, chi2, chi_p
_mbp_np(::_XPSpec) = 15

# chi_p of `_xp_precession_setup` (same operations) from the sampled rows.
@inline function _xpbp_chi_p(m1, m2, chi1x, chi1y, chi2x, chi2y)
    Mtot = m1 + m2
    m1n = m1 / Mtot
    m2n = m2 / Mtot
    m1_2 = m1n * m1n
    m2_2 = m2n * m2n
    A1 = 2.0 + (3.0 * m2n) / (2.0 * m1n)
    A2 = 2.0 + (3.0 * m1n) / (2.0 * m2n)
    S1_perp = m1_2 * sqrt(chi1x * chi1x + chi1y * chi1y)
    S2_perp = m2_2 * sqrt(chi2x * chi2x + chi2y * chi2y)
    return max(A1 * S1_perp, A2 * S2_perp) / (A1 * m1_2)
end
@inline function _mbp_theta(::_XPSpec, x)
    m1, m2, _, _ = _eobimr_physical(x[1], x[2], zero(x[1]), zero(x[1]))
    return (x[1], x[2], x[3], x[4], _xpbp_chi_p(m1, m2, x[12], x[13], x[14], x[15]))
end

# The precession coefficients `_xp_precession_setup` derives from (eta, chi1z, chi2z, chi_p), with alpha_NNLO /
# epsilon_NNLO at v_ref (shared with IMRPhenomXPHM).
@inline function _xpbp_prec(m1, m2, chi1, chi2, chip, fref)
    Mtot = m1 + m2
    m1n = m1 / Mtot
    m2n = m2 / Mtot
    etap = m1n * m2n
    delta = m1n - m2n           # (as `_xp_precession_setup_core`: smooth at q = 1)
    L = GravitationalWaveforms._xp_L3PN_coeffs(etap, chi1, chi2, delta)
    a1, a2, a3, a4L, a5, e1, e2, e3, e4L, e5 = GravitationalWaveforms._xp_alpha_epsilon_coeffs(etap, chi1, chi2, delta, m1n, chip)
    vr = cbrt(pi * (Mtot * KernelFused.GMsun_over_c3 * fref))
    om = vr * vr * vr
    lom = log(om)
    aref = a1 / om + a2 / (vr * vr) + a3 / vr + a4L * lom + a5 * vr
    eref = e1 / om + e2 / (vr * vr) + e3 / vr + e4L * lom + e5 * vr
    return (eta=etap, SL=chi1 * m1n * m1n + chi2 * m2n * m2n, Sperp=chip * m1n * m1n,
            L0=L[1], L1=L[2], L2=L[3], L3=L[4], L4=L[5], L5=L[6], L6=L[7], L7=L[8], L8=L[9], L8L=L[10],
            a1=a1, a2=a2, a3=a3, a4L=a4L, a5=a5, e1=e1, e2=e2, e3=e3, e4L=e4L, e5=e5, aref=aref, eref=eref)
end
# Intrinsic setup: the XAS flat coefficients at dL = 1 (`_xas_flat`, read by `_xas_pa_core`) and `_xpbp_prec`.
@inline function _xpbp_setup(mc, q, chi1, chi2, chip, fref)
    m1, m2, _, _ = _eobimr_physical(mc, q, zero(mc), zero(mc))
    mcc, eta = KernelFused._chirp_mass_eta(m1, m2)
    pc = KernelFused._xas_phi_coeffs(IMRPhenomXAS(), mcc, eta, chi1, chi2)
    ac = KernelFused._xas_ampl_coeffs(IMRPhenomXAS(), mcc, eta, chi1, chi2, one(mc))
    return (K=_xas_flat(pc, ac), P=_xpbp_prec(m1, m2, chi1, chi2, chip, fref)), pc
end
_mbp_setup_type(::_XPSpec) = typeof(_xpbp_setup(30.0, 0.8, 0.1, 0.1, 0.3, 20.0)[1])

@inline function _mbp_setup_store!(::_XPSpec, KT, w, mc, q, chi1, chi2, chip, kr, fref, lik)
    st, pc = _xpbp_setup(mc, q, chi1, chi2, chip, fref)
    v = _xhm_flatten(st)
    _xasbp_store_col!(KT, w, v, 0)
    @inbounds KT[length(v) + 1, w] = KernelFused._xas_time_shift(pc, lik)
    return nothing
end

# The NNLO angles at PN velocity v (as `_xp_cos_beta`, `_xp_pn_euler_alpha_nnlo`, `_xp_pn_euler_epsilon_nnlo`): the
# beta half-angles (cos, sin) and alpha, epsilon WITHOUT the f_ref constants alpha0, epsilon0. The half-angles in the
# cancellation-free form sin(beta/2) = |s| / sqrt(2 r (r + 1)), r = sqrt(1 + s^2): equal to rounding, and no sqrt(0) in
# the tangents at zero in-plane spin.
@inline function _xp_angles(v, P)
    v2 = v * v
    om = v2 * v
    lom = log(om)
    x2 = v2 * v2; x3 = x2 * v2; x4 = x3 * v2
    L = (P.eta / v) * (P.L0 + P.L1 * v + P.L2 * v2 + P.L3 * (v2 * v) + P.L4 * x2 + P.L5 * (x2 * v) + P.L6 * x3 +
                       P.L7 * (x3 * v) + P.L8 * x4 + P.L8L * x4 * log(v2))
    LS = L + P.SL
    s = P.Sperp / LS
    r = sqrt(1 + s * s)
    ch = sqrt((r + 1) / (2 * r))
    sh = ifelse(s < 0, -s, s) / sqrt(2 * r * (r + 1))   # |s| as a select: s = 0 at zero in-plane spin (device abs' NaN)
    cB = ifelse(LS >= 0, ch, sh)
    sB = ifelse(LS >= 0, sh, ch)
    al = P.a1 / om + P.a2 / v2 + P.a3 / v + P.a4L * lom + P.a5 * v - P.aref
    ep = P.e1 / om + P.e2 / v2 + P.e3 / v + P.e4L * lom + P.e5 * v - P.eref
    return cB, sB, al, ep
end
# G_m(f), m = -2..2, pow-free (`_xas_pa_core`, `_xp_angles`).
@inline function _xp_modes_fast(f, K, P)
    _, phis, A = _xas_pa_core(_NoAcc(), K, f)
    cB, sB, al, ep = _xp_angles(cbrt(pi * K[1] * f), P)
    c2 = cB * cB; s2 = sB * sB
    B = (KernelFused._XP_HAS_MODE_NORMALIZATION / 2) * A
    ph = phis - 2 * ep
    E0 = Complex(cos(ph), sin(ph))
    Ea = Complex(cos(al), sin(al))
    Ea2 = Ea * Ea
    return ((B * s2 * s2) * (E0 * conj(Ea2)), (B * 2 * cB * s2 * sB) * (E0 * conj(Ea)), (B * sqrt(6.0) * s2 * c2) * E0,
            (B * 2 * c2 * cB * sB) * (E0 * Ea), (B * c2 * c2) * (E0 * Ea2))
end
@inline function _mbp_modes(::_XPSpec, KT, w, f, stv::Val)
    K, P = _xhm_views(stv, KT, w)
    return _xp_modes_fast(f, K, P)
end

# Per detector c_md (m = -2..2) and tau_d + t_shift, from the sampled rows (generic element type: also the surrogate's).
@inline function _mbp_ext(::_XPSpec, x, g, ::Val{D}) where {D}
    mc, q, chi1, chi2, dL, t_c, phase, ra, sdec, cth, psi, c1x, c1y, c2x, c2y = x
    T = typeof(mc)
    m1, m2, dec, th = _eobimr_physical(mc, q, sdec, cth)
    st = KernelFused._xp_precession_setup(m1, m2, c1x, c1y, chi1, c2x, c2y, chi2, g.fref, th, phase;
                                          GMsun_over_c3 = KernelFused.GMsun_over_c3)
    ct = cos(st.thetaJN); sn = sin(st.thetaJN)
    Ym2 = sqrt(5.0 / (64.0 * pi)) * (1.0 - ct) * (1.0 - ct)
    Ym1 = sqrt(5.0 / (16.0 * pi)) * sn * (1.0 - ct)
    Y0 = sqrt(15.0 / (32.0 * pi)) * sn * sn
    Yp1 = sqrt(5.0 / (16.0 * pi)) * sn * (1.0 + ct)
    Yp2 = sqrt(5.0 / (64.0 * pi)) * (1.0 + ct) * (1.0 + ct)
    # (Y_m + (-1)^m Y_-m, (-1)^m Y_-m - Y_m) for m = -2..2
    yp = (Ym2 + Yp2, Ym1 - Yp1, 2 * Y0, Yp1 - Ym1, Yp2 + Ym2)
    yc = (Yp2 - Ym2, -Yp1 - Ym1, zero(T), -Ym1 - Yp1, Ym2 - Yp2)
    z2 = 2 * st.zeta_polarization
    cz = cos(z2); sz = sin(z2)
    pa = -phase - 2 * st.epsilon0
    pre = Complex(cos(pa), sin(pa)) / dL
    Ea = Complex(cos(st.alpha0), sin(st.alpha0))
    Ea2 = Ea * Ea
    em = (conj(Ea2), conj(Ea), one(Ea), Ea, Ea2)
    z0 = zero(T)                                     # (no DataType captured by the closure below: Julia 1.13.1 Enzyme)
    dt_ref = ifelse(g.geo & (g.tref > 0), time_delay_from_geocenter_gmst(g.dets[max(g.tref, 1)], ra, dec, g.gmst), z0)
    return ntuple(Val(D)) do d
        Fp, Fc = antenna_response_gmst(g.dets[d], ra, dec, psi, g.gmst)
        Fpp = Fp * cz - Fc * sz
        Fcc = Fp * sz + Fc * cz
        c = ntuple(k -> pre * em[k] * Complex(Fpp * yp[k], Fcc * yc[k]), Val(5))
        tau = t_c + ifelse(g.geo, time_delay_from_geocenter_gmst(g.dets[d], ra, dec, g.gmst) - dt_ref, z0)
        (c, tau)
    end
end

"""
    xp_binparallel_logp_and_grad(data, cfg, freqs; backend, nwalkers, setup=:host, timers=nothing) -> logp_and_grad!

IMRPhenomXP log-likelihood and gradient on `backend` (see the header of `XPBinParallel.jl`): the same closure as
[`xhm_binparallel_logp_and_grad`](@ref) for the 15-row sampled parametrization of `gw_logp_and_grad(...;
model=IMRPhenomXP())`, with the same likelihood options (distance marginalization, f_ref timing, time reference).
"""
xp_binparallel_logp_and_grad(data, cfg::PEConfig, freqs; kw...) = _mbp_logp_and_grad(_XPSpec(), data, cfg, freqs; kw...)
