# ------------------------------------------------------------------------------------------------
# IMRPhenomXPHM on the device: the mode-sum bin-parallel path of `XHMBinParallel.jl` with the two
# polarizations as the "modes",
#
#     h_d(f) = (Fp'_d hp(f) + Fc'_d hc(f)) e^{-2 pi i f tau_d} / dL,
#     Fp' = Fp cos 2 zeta - Fc sin 2 zeta,  Fc' = Fp sin 2 zeta + Fc cos 2 zeta,
#
# hp, hc = `_xphm_hphc_scalar` (before the zeta rotation) at dL = 1. Twisting each co-precessing mode (l, m') up with
# alpha = Dalpha(v_m') + alpha0, epsilon = Depsilon(v_m') + epsilon0 (`_xp_angles`, offsets as in `XPBinParallel.jl`)
# makes hp, hc functions of theta = (mc, q, chi1, chi2, chi_p) -- the co-precessing modes (XHM with the precessing
# final spin from chi_p) and the angles -- and of three walker constants kappa = (thetaJN, alpha0, epsilon0) only.
# theta gets Enzyme forward tangents (width 5, as XP); the kappa derivatives are written out in `_xphm_twist`
# (d/dalpha0: i m per term, d/depsilon0: -i m', d/dthetaJN: Y_lm -> dY_lm/dtheta), so the sums kernel accumulates
# 8 directions per detector at the cost of 5 tangents. Y_lm(thetaJN) and their derivatives are per-walker constants
# per-walker raws (`_mbp_kraw`) stored in the flat setup with zero tangents. 37 numbers per detector.
# The per-bin mode count is 33 (l, m', m) twisted components summed into hp, hc in registers.
# ------------------------------------------------------------------------------------------------

# PrecVersion V: 102 (NNLO angles, theta = (mc, q, chi1, chi2, chi_p)) or 223 (MSA angles; LAL's default).
# The MSA angles depend on the in-plane spins through |chi1_perp|, |chi2_perp| and the relative in-plane angle
# dphi = phi2 - phi1 (S1.S2 and sign((L x S1).S2)), invariant under a common rotation about L: theta = (mc, q, chi1,
# chi2, a1p, a2p, dphi), the MSA block built in the frame chi1 = (a1p, 0, chi1z), chi2 = a2p (cos dphi, sin dphi) +
# chi2z, plus the NNLO block for walkers where the MSA system fails (LAL's fallback to 102; the frame constants
# thetaJN, alpha0, epsilon0, zeta come from the version-aware `_xp_precession_setup`, which falls back the same way).
# 7 tangents need FG = 16 threads per walker to stay inside 48 KB of shared memory.
struct _XPHMSpec{V} end
_mbp_model(::_XPHMSpec{V}) where {V} = IMRPhenomXPHM{V}()
_mbp_nm(::_XPHMSpec) = 2                             # hp, hc
_mbp_nt(::_XPHMSpec{102}) = 5                        # mc, q, chi1, chi2, chi_p
_mbp_nt(::_XPHMSpec{223}) = 7                        # mc, q, chi1, chi2, a1p, a2p, dphi
_mbp_fg(::_XPHMSpec{223}) = 16
_mbp_bin_sums(::_XPHMSpec) = :gemm                   # (`_mbp_bin_sums`: -27% per gradient call)
# 223 angle table: per walker (cos beta, alpha~, epsilon~) at _XPHM_NG nodes uniform in x = log f'
# (f' = 2 f / m', the mode-scaled frequency: every mode reads the same table) from f_min / 2 to 4096 Hz, built in the
# (host) setup pass with its tangents, 4-point Lagrange interpolation per bin. The per-bin MSA evaluation with 7
# tangents cost 4.3 ms / walker at 1024 walkers (V100) vs 0.51 for 102; interpolation errors: median 6e-5 rad,
# 95% of configurations < 3e-3 rad in alpha / epsilon at 256 nodes (LAL multibands its angles too). Rows after the
# time shift: 3 per node, then x0 and the step, then the flat (NNLO block, MSA block, pi M G/c^3) the table is built
# from (the table was ~90% of the host setup, 1.5 of 1.6 ms / walker; now `_xphm_table_kernel!` builds it
# on the backend, one thread per (node, walker), after the setup pass).
const _XPHM_NG = 256
const _XPHM_FHI = 4096.0
_mbp_naux(::_XPHMSpec{223}) = 3 * _XPHM_NG + 2 + _XPHMBP_NPM      # (a constant: also evaluated in device code)
_mbp_nk(::_XPHMSpec) = 3                             # thetaJN, alpha0, epsilon0
_mbp_nkr(::_XPHMSpec) = 44                           # Y_lm (21), dY_lm/dtheta (21), alpha0, epsilon0
_mbp_np(::_XPHMSpec) = 15
@inline _mbp_theta(::_XPHMSpec{102}, x) = _mbp_theta(_XPSpec(), x)
# polar coordinates of an in-plane spin, with the origin nudged off (as `phiJ` in `_xp_precession_setup`): finite
# tangents at exactly zero in-plane spin
@inline function _xphmbp_polar(cx, cy)
    r2 = cx * cx + cy * cy
    small = r2 < 1.0e-12
    return sqrt(r2), atan(ifelse(small, zero(cy), cy), ifelse(small, oftype(cx, 1.0e-6), cx))
end
@inline function _mbp_theta(::_XPHMSpec{223}, x)
    a1, p1 = _xphmbp_polar(x[12], x[13])
    a2, p2 = _xphmbp_polar(x[14], x[15])
    return (x[1], x[2], x[3], x[4], a1, a2, p2 - p1)
end

# The CPU likelihood's precession setup (`KernelFused._xprec_state`) from the sampled rows, and dec.
@inline function _xphmbp_prec_setup(::_XPHMSpec{V}, x, g) where {V}
    mc, q, chi1, chi2, dL, t_c, phase, ra, sdec, cth, psi, c1x, c1y, c2x, c2y = x
    m1, m2, dec, th = _eobimr_physical(mc, q, sdec, cth)
    st = KernelFused._xp_precession_setup(m1, m2, c1x, c1y, chi1, c2x, c2y, chi2, g.fref, th, phase;
                                          GMsun_over_c3 = KernelFused.GMsun_over_c3, prec_version = Val(V))
    return st, dec
end
@inline function _mbp_kappa(spec::_XPHMSpec, x, g)
    st, _ = _xphmbp_prec_setup(spec, x, g)
    return (st.thetaJN, st.alpha0, st.epsilon0)
end
# -2Y_lm(theta, 0) for l = 2 (m = -2..2), 3 (-3..3), 4 (-4..4).
@inline _xphm_Y(th) = (ntuple(i -> GravitationalWaveforms._xhm_swsh_fac(2, i - 3, th), Val(5))...,
                       ntuple(i -> GravitationalWaveforms._xhm_swsh_fac(3, i - 4, th), Val(7))...,
                       ntuple(i -> GravitationalWaveforms._xhm_swsh_fac(4, i - 5, th), Val(9))...)
# dY/dtheta by complex step (every `_xhm_swsh_fac` branch is analytic in theta: exact to rounding, no subtraction;
# no nested AD, which did not infer inside the device's per-walker kernel)
@inline function _mbp_kraw(spec::_XPHMSpec, x, g)
    st, _ = _xphmbp_prec_setup(spec, x, g)
    th = st.thetaJN
    dY = map(y -> imag(y) * 1.0e30, _xphm_Y(Complex(th, 1.0e-30)))
    return (_xphm_Y(th)..., dY..., st.alpha0, st.epsilon0)
end

# Setup: XHM with the precessing final spin at dL = 1, phase 0 (as `_xprec_state`), `_xpbp_prec`, and the constants.
@inline function _xphmbp_sys102(mc, q, chi1, chi2, chip, kr, fref)
    m1, m2, _, _ = _eobimr_physical(mc, q, zero(mc), zero(mc))
    mcc, eta = KernelFused._chirp_mass_eta(m1, m2)
    fs = KernelFused._xas_precessing_final_spin(eta, chi1, chi2, chip)
    sys = KernelFused._xhm_system_setup(mcc, eta, chi1, chi2, one(mc), fref, zero(mc);
                                        GMsun_over_c3 = KernelFused.GMsun_over_c3,
                                        GMsun_over_c2_Gpc = KernelFused.GMsun_over_c2_Gpc, final_spin_override = fs)
    Y = (y2 = ntuple(i -> kr[i], Val(5)), y3 = ntuple(i -> kr[5 + i], Val(7)), y4 = ntuple(i -> kr[12 + i], Val(9)),
         d2 = ntuple(i -> kr[21 + i], Val(5)), d3 = ntuple(i -> kr[26 + i], Val(7)), d4 = ntuple(i -> kr[33 + i], Val(9)),
         a0 = kr[43], e0 = kr[44])
    return sys, _xpbp_prec(m1, m2, chi1, chi2, chip, fref), Y
end
@inline function _xphmbp_setup(mc, q, chi1, chi2, chip, kr, fref)
    sys, P, Y = _xphmbp_sys102(mc, q, chi1, chi2, chip, kr, fref)
    st = (sys, KernelFused._xhm_mode_setup(2, 1, sys), KernelFused._xhm_mode_setup(3, 3, sys),
          KernelFused._xhm_mode_setup(4, 4, sys), KernelFused._xhm_mode32_setup(sys), P, Y)
    return st, sys
end
_mbp_setup_type(::_XPHMSpec{102}) = typeof(_xphmbp_setup(30.0, 0.8, 0.1, 0.1, 0.3, ntuple(_ -> 0.1, Val(44)), 20.0)[1])
const _XPHMBP_ST102 = _mbp_setup_type(_XPHMSpec{102}())
const _XPHMBP_O102 = _xhm_flatn(_XPHMBP_ST102) + 1   # the time-shift row (a constant: device code)
# Setup stages: 1 = the system setup, P, Y, the time shift and the angle table's grid and pi M G / c^3;
# 2-5 = the mode setups (`_xhm_mode_stage!`).
_mbp_setup_nstage(::_XPHMSpec{102}) = 5
# 102 angle table: the NNLO angles on 223's grid (`_XPHM_NG` nodes uniform in log f', 4-point Lagrange),
# so 102 runs the same split mode / fused-twist GEMM sums (before: per-bin NNLO angles with 5 Enzyme tangents in one
# sums kernel, 1198 ms bin sums at 4096 walkers, V100). Rows after the time shift: 3 per node, x0, the step, pi M G/c^3.
_mbp_naux(::_XPHMSpec{102}) = 3 * _XPHM_NG + 3
# (ptxas sm_70: the (3,2) stage 53 KB at all 5 tangents; one stage for the whole setup was 99.6 KB)
_mbp_setup_width(::_XPHMSpec{102}) = 2
@inline function _mbp_setup_stage!(::_XPHMSpec{102}, ::Val{1}, KT, w, mc, q, chi1, chi2, chip, kr, fref, lik)
    sys, P, Y = _xphmbp_sys102(mc, q, chi1, chi2, chip, kr, fref)
    stv = Val(_XPHMBP_ST102)
    _xasbp_store_col!(KT, w, _xhm_flatten(sys), 0)
    _xasbp_store_col!(KT, w, _xhm_flatten(P), _xhm_part_off(stv, Val(6)))
    _xasbp_store_col!(KT, w, _xhm_flatten(Y), _xhm_part_off(stv, Val(7)))
    o = _XPHMBP_O102
    @inbounds KT[o, w] = KernelFused._xhm_time_shift(sys, lik)
    m1, m2, _, _ = _eobimr_physical(mc, q, zero(mc), zero(mc))
    x0 = log(fref / 2); hx = (log(_XPHM_FHI) - x0) / (_XPHM_NG - 1)
    @inbounds KT[o + 3_XPHM_NG + 1, w] = x0
    @inbounds KT[o + 3_XPHM_NG + 2, w] = hx
    @inbounds KT[o + 3_XPHM_NG + 3, w] = pi * (m1 + m2) * KernelFused.GMsun_over_c3
    return nothing
end
@inline _mbp_setup_stage!(::_XPHMSpec{102}, ::Val{s}, KT, w, args...) where {s} =
    _xhm_mode_stage!(KT, w, Val(_XPHMBP_ST102), Val(s))

# 223: XHM with the version-aware final spin (MSA FinalSpinMod 3, or chi_p on MSA failure) at dL = 1, phase 0, the
# NNLO block (fallback), Y, and the MSA block in the canonical in-plane frame with its angles at v_ref (the offsets:
# alpha = phi_z - phi_z(v_ref) + alpha0, epsilon = zeta - zeta(v_ref) + epsilon0) and `okf` (1.0 = MSA ok).
# In pieces (the setup stages below): the precession blocks (P, M) and the total mass ...
@inline function _xphmbp_prec223(mc, q, chi1, chi2, a1p, a2p, dphi, fref)
    m1, m2, _, _ = _eobimr_physical(mc, q, zero(mc), zero(mc))
    Mt = m1 + m2
    m1n = m1 / Mt; m2n = m2 / Mt
    c2x = a2p * cos(dphi); c2y = a2p * sin(dphi)
    chip = _xpbp_chi_p(m1, m2, a1p, zero(a1p), c2x, c2y)
    vref = cbrt(pi * (Mt * KernelFused.GMsun_over_c3 * fref))
    msa = GravitationalWaveforms._xp_msa_setup(m1n, m2n, a1p, zero(a1p), chi1, c2x, c2y, chi2, vref)
    phiz_ref, zeta_ref, _ = GravitationalWaveforms._xp_msa_angles(vref, msa)
    M = merge(Base.structdiff(msa, NamedTuple{(:ok,)}),
              (phiz_ref = phiz_ref, zeta_ref = zeta_ref, okf = ifelse(msa.ok, one(mc), zero(mc))))
    return _xpbp_prec(m1, m2, chi1, chi2, chip, fref), M, Mt
end
# ... and the XHM system setup with the final spin from M (and the NNLO block's chi_p), and Y.
@inline function _xphmbp_sys223(mc, q, chi1, chi2, a1p, a2p, dphi, kr, fref, M)
    m1, m2, _, _ = _eobimr_physical(mc, q, zero(mc), zero(mc))
    mcc, eta = KernelFused._chirp_mass_eta(m1, m2)
    Mt = m1 + m2
    m1n = m1 / Mt; m2n = m2 / Mt
    chip = _xpbp_chi_p(m1, m2, a1p, zero(a1p), a2p * cos(dphi), a2p * sin(dphi))
    eta_w = GravitationalWaveforms._xp_eta_w(typeof(mc), m1, m2)
    a_par = GravitationalWaveforms._xas_aligned_final_spin(eta_w, chi1, chi2)
    Lf = a_par - m1n * m1n * chi1 - m2n * m2n * chi2
    fs_msa = copysign(one(mc), a_par) * sqrt(abs(M.SAv2 + Lf * Lf + 2 * Lf * M.SL_pav))
    fs_102 = KernelFused._xas_precessing_final_spin(eta_w, chi1, chi2, chip)
    fs = ifelse(M.okf > 0.5, fs_msa, fs_102)
    sys = KernelFused._xhm_system_setup(mcc, eta, chi1, chi2, one(mc), fref, zero(mc);
                                        GMsun_over_c3 = KernelFused.GMsun_over_c3,
                                        GMsun_over_c2_Gpc = KernelFused.GMsun_over_c2_Gpc, final_spin_override = fs)
    Y = (y2 = ntuple(i -> kr[i], Val(5)), y3 = ntuple(i -> kr[5 + i], Val(7)), y4 = ntuple(i -> kr[12 + i], Val(9)),
         d2 = ntuple(i -> kr[21 + i], Val(5)), d3 = ntuple(i -> kr[26 + i], Val(7)), d4 = ntuple(i -> kr[33 + i], Val(9)),
         a0 = kr[43], e0 = kr[44])
    return sys, Y
end
@inline function _xphmbp_setup223(mc, q, chi1, chi2, a1p, a2p, dphi, kr, fref)
    P, M, Mt = _xphmbp_prec223(mc, q, chi1, chi2, a1p, a2p, dphi, fref)
    sys, Y = _xphmbp_sys223(mc, q, chi1, chi2, a1p, a2p, dphi, kr, fref, M)
    st = (sys, KernelFused._xhm_mode_setup(2, 1, sys), KernelFused._xhm_mode_setup(3, 3, sys),
          KernelFused._xhm_mode_setup(4, 4, sys), KernelFused._xhm_mode32_setup(sys), Y)
    return st, sys, P, M, Mt
end
_mbp_setup_type(::_XPHMSpec{223}) =
    typeof(_xphmbp_setup223(30.0, 0.8, 0.1, 0.1, 0.3, 0.2, 0.5, ntuple(_ -> 0.1, Val(44)), 20.0)[1])
const _XPHMBP_PMT = let r = _xphmbp_setup223(30.0, 0.8, 0.1, 0.1, 0.3, 0.2, 0.5, ntuple(_ -> 0.1, Val(44)), 20.0)
    typeof((r[3], r[4], 1.0))
end
const _XPHMBP_NPM = _xhm_flatn(_XPHMBP_PMT)
const _XPHMBP_ST223 = _mbp_setup_type(_XPHMSpec{223}())
const _XPHMBP_O223 = _xhm_flatn(_XPHMBP_ST223) + 1   # the time-shift row
# Stage 1: (P, M, pi M G/c^3) and the table grid x0, step (`_xphm_table_kernel!` fills the rows o + 1 : o + 3 _XPHM_NG
# in between); stage 2: the XHM system setup (M read back from stage 1's rows, with tangents), Y and the time shift;
# stages 3-6: the mode setups (`_xhm_mode_stage!`). ptxas sm_70, one tangent: one stage for all of it
# 44 KB stack frame, 110 KB spill stores; MSA alone 2.7 KB.
_mbp_setup_nstage(::_XPHMSpec{223}) = 6
# at most 2 tangents per stage pass (ptxas: the (3,2) stage 24 / 32 KB at widths 1 / 2; the others <= 16 KB at 4)
_mbp_setup_width(::_XPHMSpec{223}) = 2
@inline function _mbp_setup_stage!(::_XPHMSpec{223}, ::Val{1}, KT, w, mc, q, chi1, chi2, a1p, a2p, dphi, kr, fref, lik)
    P, M, Mt = _xphmbp_prec223(mc, q, chi1, chi2, a1p, a2p, dphi, fref)
    o = _XPHMBP_O223
    x0 = log(fref / 2); hx = (log(_XPHM_FHI) - x0) / (_XPHM_NG - 1)
    @inbounds KT[o + 3_XPHM_NG + 1, w] = x0
    @inbounds KT[o + 3_XPHM_NG + 2, w] = hx
    _xasbp_store_col!(KT, w, _xhm_flatten((P, M, pi * Mt * KernelFused.GMsun_over_c3)), o + 3_XPHM_NG + 2)
    return nothing
end
@inline function _mbp_setup_stage!(::_XPHMSpec{223}, ::Val{2}, KT, w, mc, q, chi1, chi2, a1p, a2p, dphi, kr, fref, lik)
    _, M, _ = _xhm_fv(_XPHMBP_PMT, KT, w, _XPHMBP_O223 + 3_XPHM_NG + 2)
    sys, Y = _xphmbp_sys223(mc, q, chi1, chi2, a1p, a2p, dphi, kr, fref, M)
    _xasbp_store_col!(KT, w, _xhm_flatten(sys), 0)
    _xasbp_store_col!(KT, w, _xhm_flatten(Y), _xhm_part_off(Val(_XPHMBP_ST223), Val(6)))
    @inbounds KT[_XPHMBP_O223, w] = KernelFused._xhm_time_shift(sys, lik)
    return nothing
end
@inline _mbp_setup_stage!(::_XPHMSpec{223}, ::Val{s}, KT, w, args...) where {s} =
    _xhm_mode_stage!(KT, w, Val(_XPHMBP_ST223), Val(s - 1))
# Table node j (0-based) of walker w from the stored (P, M, pi M G/c^3), read through lazy views; o = time-shift row.
@inline function _xphm_table_store!(::_XPHMSpec{223}, KT, j, w, o)
    P, M, piMG = _xhm_fv(_XPHMBP_PMT, KT, w, o + 3_XPHM_NG + 2)
    x0 = @inbounds KT[o + 3_XPHM_NG + 1, w]; hx = @inbounds KT[o + 3_XPHM_NG + 2, w]
    cb, al, ep = _xpbp_angles223_raw(cbrt(piMG * exp(x0 + j * hx)), P, M)
    @inbounds KT[o + 3j + 1, w] = cb; @inbounds KT[o + 3j + 2, w] = al; @inbounds KT[o + 3j + 3, w] = ep
    return nothing
end
# 102: the NNLO angles from the setup's P block
@inline function _xphm_table_store!(::_XPHMSpec{102}, KT, j, w, o)
    P = _xhm_views(Val(_XPHMBP_ST102), KT, w)[6]
    x0 = @inbounds KT[o + 3_XPHM_NG + 1, w]; hx = @inbounds KT[o + 3_XPHM_NG + 2, w]
    piMG = @inbounds KT[o + 3_XPHM_NG + 3, w]
    cBn, sBn, al, ep = _xp_angles(cbrt(piMG * exp(x0 + j * hx)), P)
    @inbounds KT[o + 3j + 1, w] = cBn * cBn - sBn * sBn
    @inbounds KT[o + 3j + 2, w] = al
    @inbounds KT[o + 3j + 3, w] = ep
    return nothing
end
# One thread per (node, walker); JS: the tangent directions of this launch (narrower Enzyme passes, as
# `_xphm_mode_kernel!`). ptxas sm_70: frame / spills 2.9 KB / 23 KB at width 7, 56 B / 16 B at width 1
# (with the MSA / NNLO branch and GravitationalWaveforms' branching `_jacobi_sn`; the first version, selects and all 16 AGM levels,
# 6.3 KB / 43 KB). V100, 4096 walkers, setup incl. host part 1002 (host table, loaded host) -> 204 / 155 / 169 ms
# at widths 7 / 4 / 1; logL+grad 320 -> 112 us/walker at width 4. GPU vs the CPU-built table (1024 random walkers):
# logp <= 3.9e-9, gradients <= 3.2e-7 rel.
@kernel function _xphm_table_kernel!(KT, JT, o, ::Val{JS}, gpu, spec) where {JS}
    j, w = @index(Global, NTuple)
    if length(JS) == 1                                # (a width-1 BatchDuplicated fails in the NVPTX back-end)
        _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(_xphm_table_store!), Enzyme.Const, Enzyme.Const(spec),
                  Enzyme.Duplicated(KT, JT[JS[1]]), Enzyme.Const(j - 1), Enzyme.Const(w), Enzyme.Const(o))
    else
        _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(_xphm_table_store!), Enzyme.Const, Enzyme.Const(spec),
                  Enzyme.BatchDuplicated(KT, map(k -> JT[k], JS), false), Enzyme.Const(j - 1), Enzyme.Const(w),
                  Enzyme.Const(o))
    end
end
@kernel function _xphm_table_val_kernel!(KT, o, spec)
    j, w = @index(Global, NTuple)
    _xphm_table_store!(spec, KT, j - 1, w, o)
end
# The nodes the bins read: x = log(2 f / m') <= log(2 f_max) (m' = 1), the stencil reaching node
# floor(u) + 2 (u = (x - x0) / hx); one node of margin. The node positions are unchanged (the grid still spans f_ref / 2
# to _XPHM_FHI), so the interpolated angles are bit-identical; only nodes no bin reads are skipped (f_max = 896 Hz,
# f_ref = 20: 224 of 256 nodes; 512 Hz: 200). `GRAVITATIONALWAVEBAYES_XPHM_TABLE_TRIM=false`: all nodes.
function _xphm_table_nodes(fx)
    GPU.tuning(:XPHM_TABLE_TRIM, true) || return _XPHM_NG
    x0 = log(fx.fref / 2); hx = (log(_XPHM_FHI) - x0) / (_XPHM_NG - 1)
    return clamp(floor(Int, (log(2 * fx.fmax) - x0) / hx) + 4, 4, _XPHM_NG)
end
_mbp_post_setup_value!(spec::_XPHMSpec, backend, KT, nflat, nwalkers, fx) =
    _xphm_table_val_kernel!(backend, 64)(KT, nflat + 1, spec; ndrange=(_xphm_table_nodes(fx), nwalkers))
function _mbp_post_setup!(spec::_XPHMSpec, backend, KT, JT, nflat, nwalkers, gpu, fx)
    # at most that many tangents per Enzyme pass (V100, 4096 walkers: setup (a) 33.7 ms at 4, 29.0 at 2)
    tw = GPU.tuning(:XPHM_TABLE_WIDTH, 2)
    tk! = _xphm_table_kernel!(backend, 64)
    ng = _xphm_table_nodes(fx)
    NT = _mbp_nt(spec)
    for js in Tuple(Tuple(j0:min(j0 + tw - 1, NT)) for j0 in 1:tw:NT)
        tk!(KT, JT, nflat + 1, Val(js), gpu, spec; ndrange=(ng, nwalkers))
    end
    return nothing
end
# (cos beta, alpha~, epsilon~) at v: MSA, or NNLO where the walker's MSA system failed
# (a branch, not a select: on the device a warp is 32 nodes of one walker, so only one of the two is evaluated)
@inline function _xpbp_angles223_raw(v, P, M)
    if M.okf > 0.5
        phiz, zeta, cb = GravitationalWaveforms._xp_msa_phiz_zeta_costhL(v, M.phiz_0, M.zeta_0, M)
        return cb, phiz - M.phiz_ref, zeta - M.zeta_ref
    end
    cBn, sBn, aln, epn = _xp_angles(v, P)
    return cBn * cBn - sBn * sBn, aln, epn           # cos beta from the NNLO half angles
end
# The table interpolation at x = log(2 f / m'): first row r of the 4-point Lagrange stencil and its weights.
@inline function _xpbp_tab_weights(KG, wg, o, x)
    x0 = @inbounds KG[o + 3_XPHM_NG + 1, wg]; hx = @inbounds KG[o + 3_XPHM_NG + 2, wg]
    u = (x - x0) / hx
    b = clamp(unsafe_trunc(Int, floor(u)) - 1, 0, _XPHM_NG - 4)
    t = u - b
    w0 = -(t - 1) * (t - 2) * (t - 3) / 6; w1 = t * (t - 2) * (t - 3) / 2
    w2 = -t * (t - 1) * (t - 3) / 2;      w3 = t * (t - 1) * (t - 2) / 6
    return o + 3b, (w0, w1, w2, w3)
end
# (cos beta, alpha~, epsilon~) interpolated from column wg of A (the table, or one of its tangent columns)
@inline function _xpbp_tab_interp(A, wg, r, (w0, w1, w2, w3))
    @inbounds begin
        cb = w0 * A[r + 1, wg] + w1 * A[r + 4, wg] + w2 * A[r + 7, wg] + w3 * A[r + 10, wg]
        al = w0 * A[r + 2, wg] + w1 * A[r + 5, wg] + w2 * A[r + 8, wg] + w3 * A[r + 11, wg]
        ep = w0 * A[r + 3, wg] + w1 * A[r + 6, wg] + w2 * A[r + 9, wg] + w3 * A[r + 12, wg]
    end
    return cb, al, ep
end
# The angles of mode m' at f from the table (4-point Lagrange in x = log(2 f / m')), as `_xp_angles` returns them
@inline function _xpbp_tab_angles(KG, wg, o, x)
    r, wt = _xpbp_tab_weights(KG, wg, o, x)
    cb, al, ep = _xpbp_tab_interp(KG, wg, r, wt)
    cb = clamp(cb, -1.0, 1.0)
    return sqrt(max((1 + cb) / 2, floatmin(cb))), sqrt(max((1 - cb) / 2, floatmin(cb))), al, ep
end
# ... and their derivatives along the setup tangent column JG (the table's rows of JT[j]), written out: the
# derivative of `_xpbp_tab_angles` (clamp and the floatmin guards as selects; the stencil does not depend on theta).
@inline function _xpbp_tab_angles_tan(KG, JG, wg, o, x)
    r, wt = _xpbp_tab_weights(KG, wg, o, x)
    cbr, al, ep = _xpbp_tab_interp(KG, wg, r, wt)
    dcb, dal, dep = _xpbp_tab_interp(JG, wg, r, wt)
    cb = clamp(cbr, -1.0, 1.0)
    dcb = ifelse((cbr > 1.0) | (cbr < -1.0), zero(dcb), dcb)
    c2 = (1 + cb) / 2; s2 = (1 - cb) / 2
    cB = sqrt(max(c2, floatmin(cb))); sB = sqrt(max(s2, floatmin(cb)))
    return (cB, sB, al, ep), (ifelse(c2 > floatmin(cb), dcb / (4 * cB), zero(dcb)),
                              ifelse(s2 > floatmin(cb), -dcb / (4 * sB), zero(dcb)), dal, dep)
end
# m' = 1, 2, 3, 4: f' = 2 f / m'
@inline _xpbp_tab_angles4(KG, wg, o, lf) =
    (_xpbp_tab_angles(KG, wg, o, lf + log(2.0)), _xpbp_tab_angles(KG, wg, o, lf),
     _xpbp_tab_angles(KG, wg, o, lf + log(2.0 / 3.0)), _xpbp_tab_angles(KG, wg, o, lf - log(2.0)))
@inline function _mbp_modes(::_XPHMSpec, KT, w, f, stv::Val, KG, wg)
    v = _xhm_views(stv, KT, w)
    sys, ms21, ms33, ms44, m32 = v
    Y = last(v)
    h = _xhm_modes_fast(f, sys, ms21, ms33, ms44, m32)
    o = _mbp_flatn(stv) + 1                          # the time-shift row; the table follows
    t = _xphm_twist_all(h, _xpbp_tab_angles4(KG, wg, o, log(f)), Y)
    return (t[1], t[2]), ((t[3], t[4]), (t[5], t[6]), (t[7], t[8]))
end

# d^l_{m,m'}(beta), m = -l..l, in the half-angle powers c_k = cos(beta/2)^k, s_k = sin(beta/2)^k
# (`_xphm_twist_up_mode` / `_xp_twist_up22`).
function _xphm_wigner(l, mp)
    c(k) = Symbol(:c, k)
    s(k) = Symbol(:s, k)
    if (l, mp) == (2, 2)
        return [:($(s(4))), :(2 * $(c(1)) * $(s(3))), :(sqrt(6.0) * $(s(2)) * $(c(2))), :(2 * $(c(3)) * $(s(1))), :($(c(4)))]
    elseif (l, mp) == (2, 1)
        return [:(2 * $(c(1)) * $(s(3))), :(3 * $(c(2)) * $(s(2)) - $(s(4))), :(sqrt(6.0) * ($(c(3)) * $(s(1)) - $(c(1)) * $(s(3)))),
                :($(c(2)) * ($(c(2)) - 3 * $(s(2)))), :(-2 * $(c(3)) * $(s(1)))]
    elseif (l, mp) == (3, 3)
        return [:($(s(6))), :(sqrt(6.0) * $(c(1)) * $(s(5))), :(sqrt(15.0) * $(c(2)) * $(s(4))), :(2 * sqrt(5.0) * $(c(3)) * $(s(3))),
                :(sqrt(15.0) * $(c(4)) * $(s(2))), :(sqrt(6.0) * $(c(5)) * $(s(1))), :($(c(6)))]
    elseif (l, mp) == (3, 2)
        return [:(sqrt(6.0) * $(c(1)) * $(s(5))), :($(s(4)) * (5 * $(c(2)) - $(s(2)))),
                :(sqrt(10.0) * $(s(3)) * (2 * $(c(3)) - $(c(1)) * $(s(2)))), :(sqrt(30.0) * $(c(2)) * ($(c(2)) - $(s(2))) * $(s(2))),
                :(sqrt(10.0) * $(c(3)) * ($(c(2)) * $(s(1)) - 2 * $(s(3)))), :($(c(4)) * ($(c(2)) - 5 * $(s(2)))),
                :(-sqrt(6.0) * $(c(5)) * $(s(1)))]
    elseif (l, mp) == (4, 4)
        return [:($(s(8))), :(2 * sqrt(2.0) * $(c(1)) * $(s(7))), :(2 * sqrt(7.0) * $(c(2)) * $(s(6))),
                :(2 * sqrt(14.0) * $(c(3)) * $(s(5))), :(sqrt(70.0) * $(c(4)) * $(s(4))), :(2 * sqrt(14.0) * $(c(5)) * $(s(3))),
                :(2 * sqrt(7.0) * $(c(6)) * $(s(2))), :(2 * sqrt(2.0) * $(c(7)) * $(s(1))), :($(c(8)))]
    end
    error("no Wigner table for ($l, $mp)")
end
# Mode (l, m') twisted up: (hp, hc) and their derivatives along thetaJN, alpha0, epsilon0 (8 complex numbers), from the
# co-precessing mode h, the angles at v_m' (`_xp_angles`), alpha0, epsilon0 and Y_lm, dY_lm/dtheta (m = -l..l):
#     hp = E sum_m Y_m (a_m + (-1)^l b_m),  hc = i E sum_m Y_m (a_m - (-1)^l b_m),
#     a_m = e^{-i m alpha} d^l_{m,-m'},  b_m = e^{i m alpha} d^l_{m,m'},  E = e^{-i m' epsilon} h / 2,
# with d^l_{m,-m'} = (-1)^{m+m'} d^l_{-m,m'}.
@generated function _xphm_twist(::Val{l}, ::Val{mp}, h, ang, a0, e0, Y, dY) where {l,mp}
    Expr(:block, Expr(:meta, :inline), :((cB, sB, al0, ep0) = ang), :(al = al0 + a0), :(ep = ep0 + e0),
         _xphm_twist_exprs(l, mp, false)...,
         :(Ee = Complex(cos($mp * ep), -sin($mp * ep)) * (h / 2)), :(hp = Ee * P), :(hc = im * (Ee * C)),
         :(return (hp, hc, Ee * Pth, im * (Ee * Cth), Ee * Pal, im * (Ee * Cal), (-im * $mp) * hp, (-im * $mp) * hc)))
end
# The sums of `_xphm_twist` at alpha = al (without E): P, C, their thetaJN derivatives Pth, Cth (through dY) and alpha
# derivatives Pal, Cal; with `parts`, also the derivatives along cos(beta/2), sin(beta/2): Pc, Cc, Ps, Cs.
function _xphm_twist_exprs(l, mp, parts::Bool)
    dp = _xphm_wigner(l, mp)
    sg = iseven(l) ? 1 : -1
    ex = Any[:(c1 = cB), :(s1 = sB)]
    for k in 2:2l
        push!(ex, :($(Symbol(:c, k)) = $(Symbol(:c, k - 1)) * cB), :($(Symbol(:s, k)) = $(Symbol(:s, k - 1)) * sB))
    end
    push!(ex, :(E1 = Complex(cos(al), sin(al))))
    for k in 2:l
        push!(ex, :($(Symbol(:E, k)) = $(Symbol(:E, k - 1)) * E1))
    end
    Pt = Any[]; Ct = Any[]; Pth = Any[]; Cth = Any[]; Pal = Any[]; Cal = Any[]
    D = Dict(v => (Any[], Any[]) for v in (:c, :s))
    for (i, m) in enumerate(-l:l)
        dpm = dp[i]                                   # d^l_{m,m'}
        dmm = isodd(m + mp) ? :(-$(dp[2l + 2 - i])) : dp[2l + 2 - i]   # d^l_{m,-m'} = (-1)^{m+m'} d^l_{-m,m'}
        emm = m == 0 ? :(one(E1)) : m > 0 ? :(conj($(Symbol(:E, m)))) : Symbol(:E, -m)   # e^{-i m alpha}
        epm = m == 0 ? :(one(E1)) : m > 0 ? Symbol(:E, m) : :(conj($(Symbol(:E, -m))))   # e^{+i m alpha}
        a = Symbol(:a, i); b = Symbol(:b, i)
        push!(ex, :($a = $emm * $dmm), :($b = $epm * $dpm))
        push!(Pt, :(Y[$i] * ($a + $sg * $b))); push!(Ct, :(Y[$i] * ($a - $sg * $b)))
        push!(Pth, :(dY[$i] * ($a + $sg * $b))); push!(Cth, :(dY[$i] * ($a - $sg * $b)))
        if m != 0
            push!(Pal, :(($m * Y[$i]) * (-$a + $sg * $b))); push!(Cal, :(($m * Y[$i]) * (-$a - $sg * $b)))
        end
        parts || continue
        for v in (:c, :s)
            av = Symbol(:a, v, i); bv = Symbol(:b, v, i)
            push!(ex, :($av = $emm * $(_xphm_dexpr(dmm, v))), :($bv = $epm * $(_xphm_dexpr(dpm, v))))
            push!(D[v][1], :(Y[$i] * ($av + $sg * $bv))); push!(D[v][2], :(Y[$i] * ($av - $sg * $bv)))
        end
    end
    sm(v) = foldl((x, y) -> :($x + $y), v)
    push!(ex, :(P = $(sm(Pt))), :(C = $(sm(Ct))), :(Pth = $(sm(Pth))), :(Cth = $(sm(Cth))),
          :(Pal = im * $(sm(Pal))), :(Cal = im * $(sm(Cal))))
    parts && push!(ex, :(Pc = $(sm(D[:c][1]))), :(Cc = $(sm(D[:c][2]))), :(Ps = $(sm(D[:s][1]))), :(Cs = $(sm(D[:s][2]))))
    return ex
end
# d/dv (v = :c or :s) of a `_xphm_wigner` expression: sums / products of numbers, sqrt(number) and the powers c_k, s_k.
function _xphm_dexpr(ex, v::Symbol)
    if ex isa Symbol
        m = match(r"^([cs])(\d+)$", String(ex))
        (m === nothing || Symbol(m[1]) !== v) && return 0
        k = parse(Int, m[2])
        return k == 1 ? 1 : :($k * $(Symbol(v, k - 1)))
    end
    ex isa Expr || return 0                          # a number
    ex.head === :call || error("unexpected Wigner expression $ex")
    op = ex.args[1]; a = ex.args[2:end]
    op === :sqrt && return 0                         # sqrt of a constant
    da = [_xphm_dexpr(x, v) for x in a]
    if op === :+ || (op === :- && length(a) == 1)
        t = [op === :- ? :(-$d) : d for d in da if d != 0]
    elseif op === :-
        t = Any[d == 0 ? nothing : (i == 1 ? d : :(-$d)) for (i, d) in enumerate(da)]
        t = filter(!isnothing, t)
    elseif op === :*
        t = Any[Expr(:call, :*, (j == i ? da[i] : a[j] for j in eachindex(a))...) for i in eachindex(a) if da[i] != 0]
    else
        error("unexpected operator $op in a Wigner expression")
    end
    return isempty(t) ? 0 : length(t) == 1 ? t[1] : Expr(:call, :+, t...)
end
@inline _xphm_add(a::NTuple{N}, b::NTuple{N}) where {N} = ntuple(i -> a[i] + b[i], Val(N))

# hp, hc and their thetaJN, alpha0, epsilon0 derivatives (`_xphm_twist`'s 8 numbers) summed over the 5 co-precessing
# modes h (`_xhm_modes_fast` order) with the angles an[m'] (m' = 1..4) and Y the setup's Y block.
@inline function _xphm_twist_all(h, an, Y)
    h22, h21, h33, h32, h44 = h
    an1, an2, an3, an4 = an
    a0 = Y.a0; e0 = Y.e0
    y2 = Y.y2; d2 = Y.d2; y3 = Y.y3; d3 = Y.d3
    t = _xphm_twist(Val(2), Val(2), h22, an2, a0, e0, y2, d2)
    t = _xphm_add(t, _xphm_twist(Val(2), Val(1), h21, an1, a0, e0, y2, d2))
    t = _xphm_add(t, _xphm_twist(Val(3), Val(3), h33, an3, a0, e0, y3, d3))
    t = _xphm_add(t, _xphm_twist(Val(3), Val(2), h32, an2, a0, e0, y3, d3))
    return _xphm_add(t, _xphm_twist(Val(4), Val(4), h44, an4, a0, e0, Y.y4, Y.d4))
end

# ---- XPHM 223 GEMM values in two stages ----
# One thread per (bin, walker) running a width-7 Enzyme forward pass over all 5 modes, the 4 table interpolations and
# the 33-term twist (`_mbp_gemm_modes_kernel!`) took 255 registers + an 11.9 KB stack frame with ~145 KB of spill
# traffic per thread (sm_70). Split: (1) `_xphm_mode_kernel!`, one launch per co-precessing mode k, thread per (bin,
# walker): the mode and its 7 Enzyme tangents into XM[bin, walker, k], XMT[j]; (2) `_xphm_twist_kernel!`, thread per
# (bin, walker) and output direction (0 = hp, hc and the kappa derivatives; j = d/dtheta_j): the twist with its tangent
# written out (the twist is linear in h; the angles enter through cos/sin(beta/2), alpha, epsilon), no Enzyme:
#     dhp = E [dh P + h (Pc dcB + Ps dsB + Pal dalpha)] - i m' depsilon hp,   E = e^{-i m' epsilon} / 2,
# and likewise hc with C. The GEMM and combine stages read the same blocks as before. ptxas (sm_70): twist per (mode,
# direction) 128-168 registers, no spills (all 5 modes in one thread still spilled); mode kernels 0.8-3.8 KB spills at
# width 7. V100 (GW150914 H1+L1, 20-896 Hz, GPU shared host): bin sums 2634 -> 296 ms per call at 4096 walkers, 657 ->
# 75 at 1024; logL+grad 817 -> 213 us/walker at 4096. Width 4 (283 ms) and `mode` grouping (331 ms) measured no better.
const _XPHM_LMP = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))   # (l, m') of the co-precessing modes, `_xhm_modes_fast` order
# The (3,2) mode as `_xhmf_32`, with the (2,2) phase (needed only in its ringdown branch) computed there.
@inline function _xphmf_32(f, Mf, c13, lMf, m76, amp0, sys, m32)
    if Mf < m32.fAmpMatchIM
        return _xhmf_32(Mf, c13, lMf, m76, amp0, zero(Mf), sys, m32)
    end
    _, phis, _ = _xas_pa_core(_NoAcc(), _XasFlatLazy(sys.coeffs22, sys.amplcoeffs22), f)
    return _xhmf_32(Mf, c13, lMf, m76, amp0, phis + sys.timeshift * Mf + sys.phiref22, sys, m32)
end
# Co-precessing mode k of `_xhm_modes_fast` alone.
@inline function _xphm_mode(::Val{k}, f, sys, ms21, ms33, ms44, m32) where {k}
    Mf = sys.coeffs22.M * sys.coeffs22.GMsun_over_c3 * f
    if k == 1
        _, phis, A = _xas_pa_core(_NoAcc(), _XasFlatLazy(sys.coeffs22, sys.amplcoeffs22), f)
        phi22 = phis + sys.timeshift * Mf + sys.phiref22
        return (KernelFused._XP_HAS_MODE_NORMALIZATION * A) * Complex(cos(phi22), sin(phi22))
    end
    c13 = cbrt(Mf)
    m76 = 1 / (Mf * sqrt(c13))
    lMf = log(Mf)
    amp0 = sys.amplcoeffs22.Overallamp * KernelFused._XP_HAS_MODE_NORMALIZATION
    k == 2 && return _xhmf_nonmixing(1.0, Mf, c13, lMf, m76, amp0, ms21)
    k == 3 && return _xhmf_nonmixing(-1.0, Mf, c13, lMf, m76, amp0, ms33)
    k == 4 && return _xphmf_32(f, Mf, c13, lMf, m76, amp0, sys, m32)
    return _xhmf_nonmixing(1.0, Mf, c13, lMf, m76, amp0, ms44)
end
@inline function _xphm_mode_store!(XM, i, w, w0, nw, kv::Val{k}, KT, F1, stv) where {k}
    wg = w0 + w
    wg > nw && return nothing                         # (not read: the twist zeroes walkers past nw)
    f = @inbounds F1[i]
    sys, ms21, ms33, ms44, m32, _ = _xhm_views(stv, KT, wg)
    @inbounds XM[i, w, k] = _xphm_mode(kv, f, sys, ms21, ms33, ms44, m32)
    return nothing
end
# JS: the tangent directions of this launch (a tuple of 1..NT: narrower Enzyme passes spill less, `_mbp_gemm_modes!`).
@kernel function _xphm_mode_kernel!(XM, XMT, KT, JT, @Const(F1), w0, nw, kv, stv, ::Val{JS}, gpu) where {JS}
    i, w = @index(Global, NTuple)
    if XMT === nothing
        _xphm_mode_store!(XM, i, w, w0, nw, kv, KT, F1, stv)
    elseif length(JS) == 1                            # (a width-1 BatchDuplicated fails in the NVPTX back-end)
        _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(_xphm_mode_store!), Enzyme.Const,
                  Enzyme.Duplicated(XM, XMT[JS[1]]), Enzyme.Const(i), Enzyme.Const(w), Enzyme.Const(w0),
                  Enzyme.Const(nw), Enzyme.Const(kv), Enzyme.Duplicated(KT, JT[JS[1]]), Enzyme.Const(F1),
                  Enzyme.Const(stv))
    else
        _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(_xphm_mode_store!), Enzyme.Const,
                  Enzyme.BatchDuplicated(XM, map(j -> XMT[j], JS), false), Enzyme.Const(i), Enzyme.Const(w),
                  Enzyme.Const(w0), Enzyme.Const(nw), Enzyme.Const(kv), Enzyme.BatchDuplicated(KT, map(j -> JT[j], JS), false),
                  Enzyme.Const(F1), Enzyme.Const(stv))
    end
end
# Mode (l, m')'s derivative of (hp, hc) along one direction, from the mode h and its derivative dh, the angles
# an = (cB, sB, alpha~, epsilon~) and their derivatives dan.
@inline function _xphm_twist_tan(::Val{l}, ::Val{mp}, h, dh, an, dan, a0, e0, Y, dY) where {l,mp}
    cB, sB, al0, ep0 = an
    dcB, dsB, dal, dep = dan
    P, C, _, _, Pal, Cal, Pc, Cc, Ps, Cs = _xphm_twist_parts(Val(l), Val(mp), cB, sB, al0 + a0, Y, dY)
    ep = ep0 + e0
    E = Complex(cos(mp * ep), -sin(mp * ep)) / 2
    Eh = E * h
    mde = (-im * mp) * dep
    return (E * (dh * P) + Eh * (Pc * dcB + Ps * dsB + Pal * dal) + mde * (Eh * P),
            im * (E * (dh * C) + Eh * (Cc * dcB + Cs * dsB + Cal * dal) + mde * (Eh * C)))
end
@generated function _xphm_twist_parts(::Val{l}, ::Val{mp}, cB, sB, al, Y, dY) where {l,mp}
    Expr(:block, Expr(:meta, :inline), _xphm_twist_exprs(l, mp, true)...,
         :(return (P, C, Pth, Cth, Pal, Cal, Pc, Cc, Ps, Cs)))
end
# (l, m') of mode k and its Y block
@inline _xphm_lmp(::Val{1}) = (Val(2), Val(2))
@inline _xphm_lmp(::Val{2}) = (Val(2), Val(1))
@inline _xphm_lmp(::Val{3}) = (Val(3), Val(3))
@inline _xphm_lmp(::Val{4}) = (Val(3), Val(2))
@inline _xphm_lmp(::Val{5}) = (Val(4), Val(4))
@inline _xphm_int(::Val{n}) where {n} = n
@inline _xphm_Yl(Y, ::Val{2}) = (Y.y2, Y.d2)
@inline _xphm_Yl(Y, ::Val{3}) = (Y.y3, Y.d3)
@inline _xphm_Yl(Y, ::Val{4}) = (Y.y4, Y.d4)
# Mode k's contribution to direction j of bin i, walker w (chunk-local; wg = w0 + w global): j = 0 -> `_xphm_twist`'s
# 8 numbers, j >= 1 -> d(hp, hc)/dtheta_j. The angles at x = log(2 f / m') from the table.
@inline function _xphm_twist_mode(kv::Val{k}, ::Val{j}, XM, XMT, KT, JT, i, w, wg, o, lf, Y) where {k,j}
    lv, mpv = _xphm_lmp(kv)
    x = lf + log(2.0 / _xphm_int(mpv))
    h = @inbounds XM[i, w, k]
    y, dy = _xphm_Yl(Y, lv)
    if j == 0
        return _xphm_twist(lv, mpv, h, _xpbp_tab_angles(KT, wg, o, x), Y.a0, Y.e0, y, dy)
    else
        an, dan = _xpbp_tab_angles_tan(KT, JT[j], wg, o, x)
        return _xphm_twist_tan(lv, mpv, h, @inbounds(XMT[j][i, w, k]), an, dan, Y.a0, Y.e0, y, dy)
    end
end
# Direction j of the modes KS into the blocks (j = 0: XH and, with G, the kappa blocks XK; j >= 1: XHT[j]), stored
# (ACC = false) or added to what earlier launches stored (ACC = true); walkers past nw (wg = 0) get zeros.
@generated function _xphm_twist_store!(::Val{KS}, ::Val{j}, XH, XHT, XK, XM, XMT, KT, JT, i, w, wg, f, stv, ::Val{G},
                                       ::Val{ACC}) where {KS,j,G,ACC}
    dst = j == 0 ? Any[:(XH[i, w, 1]), :(XH[i, w, 2])] : Any[:(XHT[$j][i, w, 1]), :(XHT[$j][i, w, 2])]
    if j == 0 && G
        append!(dst, [:(XK[$q][i, w, $c]) for q in 1:3 for c in 1:2])
    end
    n = length(dst)
    zs = ACC ? Any[] : Any[:(@inbounds $d = zero(ComplexF64)) for d in dst]
    sts = Any[ACC ? :(@inbounds $(dst[r]) += t[$r]) : :(@inbounds $(dst[r]) = t[$r]) for r in 1:n]
    sm = foldl((a, k) -> :(_xphm_add($a, _xphm_twist_mode(Val($k), Val($j), XM, XMT, KT, JT, i, w, wg, o, lf, Y))),
               KS[2:end]; init = :(_xphm_twist_mode(Val($(KS[1])), Val($j), XM, XMT, KT, JT, i, w, wg, o, lf, Y)))
    quote
        $(Expr(:meta, :inline))
        if wg === 0
            $(zs...)
            return nothing
        end
        Y = last(_xhm_views(stv, KT, wg))
        o = _mbp_flatn(stv) + 1                      # the time-shift row; the table follows
        lf = log(f)
        t = $sm
        $(sts...)
        return nothing
    end
end
@generated function _xphm_twist_dirs!(::Val{KS}, ::Val{JS}, XH, XHT, XK, XM, XMT, KT, JT, i, w, wg, f, stv, gv,
                                      av) where {KS,JS}
    Expr(:block, Expr(:meta, :inline),
         (:(_xphm_twist_store!(Val($KS), Val($j), XH, XHT, XK, XM, XMT, KT, JT, i, w, wg, f, stv, gv, av)) for j in JS)...,
         :(return nothing))
end
# One twist launch: the modes KS (a tuple of 1..5) along the directions JS (a tuple of 0..NT); G: also the kappa
# blocks; ACC: add to the blocks.
@kernel function _xphm_twist_kernel!(XH, XHT, XK, @Const(XM), XMT, @Const(KT), JT, @Const(F1), w0, nw, stv, ks, js, gv, av)
    i, w = @index(Global, NTuple)
    wg = w0 + w
    _xphm_twist_dirs!(ks, js, XH, XHT, XK, XM, XMT, KT, JT, i, w, ifelse(wg > nw, 0, wg), @inbounds(F1[i]), stv, gv, av)
end
# ---- fused twist: one thread per (bin, walker) and mode group, every direction at once ----
# The (mode, direction) launches above recompute the table stencil, the primal angles, the Wigner parts and E for
# each of the 8 directions. Here a mode's parts are computed once; direction j >= 1 then only reads dh and the angle
# tangents of JT[j] (same stencil) and does the few products of `_xphm_twist_tan`.
# JS: the directions of this thread (0 = hp, hc and, with G, their kappa derivatives; j >= 1 = d/dtheta_j).
@inline function _xphm_twist_mode_all(kv::Val{k}, jsv::Val{JS}, XM, XMT, KT, JT, i, w, wg, o, lf, Y,
                                      ::Val{G}) where {k,JS,G}
    lv, mpv = _xphm_lmp(kv)
    mp = _xphm_int(mpv)
    x = lf + log(2.0 / mp)
    h = @inbounds XM[i, w, k]
    y, dy = _xphm_Yl(Y, lv)
    r, wt = _xpbp_tab_weights(KT, wg, o, x)
    cbr, al0, ep0 = _xpbp_tab_interp(KT, wg, r, wt)
    cb = clamp(cbr, -1.0, 1.0)
    c2 = (1 + cb) / 2; s2 = (1 - cb) / 2
    cB = sqrt(max(c2, floatmin(cb))); sB = sqrt(max(s2, floatmin(cb)))
    P, C, Pth, Cth, Pal, Cal, Pc, Cc, Ps, Cs = _xphm_twist_parts(lv, mpv, cB, sB, al0 + Y.a0, y, dy)
    ep = ep0 + Y.e0
    E = Complex(cos(mp * ep), -sin(mp * ep)) / 2
    Eh = E * h
    hp = Eh * P; hc = im * (Eh * C)
    G || return (hp, hc)
    v0 = 0 in JS ? (hp, hc, Eh * Pth, im * (Eh * Cth), Eh * Pal, im * (Eh * Cal), (-im * mp) * hp, (-im * mp) * hc) : ()
    inb = !((cbr > 1.0) | (cbr < -1.0))
    okc = c2 > floatmin(cb); oks = s2 > floatmin(cb)
    vt = _xphm_twist_dirs_all(jsv, XMT, JT, i, w, k, wg, r, wt, inb, okc, oks, cB, sB, mp, E, Eh, hp, hc, P, C, Pal,
                              Cal, Pc, Cc, Ps, Cs)
    return (v0..., vt...)
end
@generated function _xphm_twist_dirs_all(::Val{JS}, XMT, JT, i, w, k, wg, r, wt, inb, okc, oks, cB, sB, mp, E, Eh,
                                         hp, hc, P, C, Pal, Cal, Pc, Cc, Ps, Cs) where {JS}
    ex = Any[Expr(:meta, :inline)]
    outs = Any[]
    for j in JS
        j == 0 && continue
        dh = Symbol(:dh, j); dcb = Symbol(:dcb, j); dal = Symbol(:dal, j); dep = Symbol(:dep, j)
        dcB = Symbol(:dcB, j); dsB = Symbol(:dsB, j); m = Symbol(:mde, j); a = Symbol(:A, j)
        append!(ex, (:($dh = @inbounds XMT[$j][i, w, k]),
                     :(($dcb, $dal, $dep) = _xpbp_tab_interp(JT[$j], wg, r, wt)),
                     :($dcB = ifelse(inb & okc, $dcb / (4 * cB), zero(cB))),
                     :($dsB = ifelse(inb & oks, -$dcb / (4 * sB), zero(sB))),
                     :($m = (-im * mp) * $dep), :($a = E * $dh)))
        push!(outs, :($a * P + Eh * (Pc * $dcB + Ps * $dsB + Pal * $dal) + $m * hp),
              :(im * ($a * C + Eh * (Cc * $dcB + Cs * $dsB + Cal * $dal)) + $m * hc))
    end
    push!(ex, :(return ($(outs...),)))
    return Expr(:block, ex...)
end
# The modes KS summed along the directions JS: stored into XH (+ XK with G) and XHT, or added (ACC).
@generated function _xphm_twist_fused_store!(::Val{KS}, ::Val{JS}, XH, XHT, XK, XM, XMT, KT, JT, i, w, wg, f, stv,
                                             ::Val{G}, ::Val{ACC}) where {KS,JS,G,ACC}
    dst = Any[]
    if 0 in JS
        append!(dst, [:(XH[i, w, 1]), :(XH[i, w, 2])])
        G && append!(dst, [:(XK[$q][i, w, $c]) for q in 1:3 for c in 1:2])
    end
    if G
        append!(dst, [:(XHT[$j][i, w, $c]) for j in JS if j > 0 for c in 1:2])
    end
    n = length(dst)
    zs = ACC ? Any[] : Any[:(@inbounds $d = zero(ComplexF64)) for d in dst]
    sts = Any[ACC ? :(@inbounds $(dst[r]) += t[$r]) : :(@inbounds $(dst[r]) = t[$r]) for r in 1:n]
    call(k) = :(_xphm_twist_mode_all(Val($k), Val($JS), XM, XMT, KT, JT, i, w, wg, o, lf, Y, Val($G)))
    sm = foldl((a, k) -> :(_xphm_add($a, $(call(k)))), KS[2:end]; init = call(KS[1]))
    quote
        $(Expr(:meta, :inline))
        if wg === 0
            $(zs...)
            return nothing
        end
        Y = last(_xhm_views(stv, KT, wg))
        o = _mbp_flatn(stv) + 1
        lf = log(f)
        t = $sm
        $(sts...)
        return nothing
    end
end
@kernel function _xphm_twist_fused_kernel!(XH, XHT, XK, @Const(XM), XMT, @Const(KT), JT, @Const(F1), w0, nw, stv, ks,
                                           js, gv, av)
    i, w = @index(Global, NTuple)
    wg = w0 + w
    _xphm_twist_fused_store!(ks, js, XH, XHT, XK, XM, XMT, KT, JT, i, w, ifelse(wg > nw, 0, wg), @inbounds(F1[i]), stv,
                             gv, av)
end
# fused (modes, directions) launch groups (`GRAVITATIONALWAVEBAYES_XPHM_TWIST`): `fused` = one launch per mode with every direction;
# `fusedall` = one launch; `fused2` = modes (1, 2), (3, 4, 5); `fdir3` = all modes, directions (0), (1-3), (4-7);
# `fdir4` = (0), (1, 2), (3, 4, 5), (6, 7); `fdir8` = one launch per direction; or a custom list in
# `GRAVITATIONALWAVEBAYES_XPHM_FUSED`, e.g. `12345:0,12345:123,12345:4567` (modes:directions per launch).
const _XPHM_JA = (0, 1, 2, 3, 4, 5, 6, 7)
_xphm_fused_groups(::Val{:fused}) = map(k -> ((k,), _XPHM_JA), (1, 2, 3, 4, 5))
_xphm_fused_groups(::Val{:fusedall}) = (((1, 2, 3, 4, 5), _XPHM_JA),)
_xphm_fused_groups(::Val{:fusedh}) = Tuple(((k,), js) for k in 1:5 for js in ((0, 1, 2, 3), (4, 5, 6, 7)))
# mode pairs with half the directions each (ptxas sm_70: 254-255 registers, <= 12 B spills; all directions spill ~1 KB).
# V100, GW150914 H1+L1 20-512 Hz, 4096 walkers, bin sums per call: modedir 141.4 ms, fused 110.2, fusedh
# 109.9, fusedh3 113.6, fpair 94.2 (the default), fpairb 95.5 (fused2 / fusedall spill: 198 / 533 in an earlier build); no
# accumulation at all (fused, wrong sums, timing only) 92.6.
_xphm_fused_groups(::Val{:fpair}) = (Tuple((ks, js) for ks in ((1, 2), (3, 4)) for js in ((0, 1, 2, 3), (4, 5, 6, 7)))...,
                                     ((5,), _XPHM_JA))
_xphm_fused_groups(::Val{:fpairb}) = (((1,), _XPHM_JA),
                                      Tuple((ks, js) for ks in ((2, 3), (4, 5)) for js in ((0, 1, 2, 3), (4, 5, 6, 7)))...)
_xphm_fused_groups(::Val{:fusedh3}) = Tuple(((k,), js) for k in 1:5 for js in ((0, 1, 2), (3, 4, 5), (6, 7)))
_xphm_fused_groups(::Val{:fused2}) = (((1, 2), _XPHM_JA), ((3, 4, 5), _XPHM_JA))
_xphm_fused_groups(::Val{:fdir3}) = map(js -> ((1, 2, 3, 4, 5), js), ((0,), (1, 2, 3), (4, 5, 6, 7)))
_xphm_fused_groups(::Val{:fdir4}) = map(js -> ((1, 2, 3, 4, 5), js), ((0,), (1, 2), (3, 4, 5), (6, 7)))
_xphm_fused_groups(::Val{:fdir8}) = map(j -> ((1, 2, 3, 4, 5), (j,)), _XPHM_JA)
_xphm_fused_groups(::Val{:custom}) =
    Tuple(Tuple(Tuple(parse.(Int, collect(p))) for p in split(g, ":")) for g in split(ENV["GRAVITATIONALWAVEBAYES_XPHM_FUSED"], ","))
_xphm_fused_groups(::Val) = nothing

# The twist launches as (modes, directions) groups (`GRAVITATIONALWAVEBAYES_XPHM_TWIST`: `all` = one launch; `dir` = one per
# direction; `mode` = one per mode; `modedir` = one per (mode, direction)).
const _XPHM_KS = (1, 2, 3, 4, 5)
const _XPHM_JS = (0, 1, 2, 3, 4, 5, 6, 7)
_xphm_twist_groups(::Val{:all}) = ((_XPHM_KS, _XPHM_JS),)
_xphm_twist_groups(::Val{:dir}) = map(j -> (_XPHM_KS, (j,)), _XPHM_JS)
_xphm_twist_groups(::Val{:mode}) = map(k -> ((k,), _XPHM_JS), _XPHM_KS)
_xphm_twist_groups(::Val{:modedir}) = Tuple(((k,), (j,)) for k in _XPHM_KS for j in _XPHM_JS)
_mbp_gemm_nmb(::_XPHMSpec) = 5
function _mbp_gemm_modes!(spec::_XPHMSpec, backend, gb, KT, JT, F1, w0, nw, stv, nmv, nkv, grad, gpu)
    _xphm_sums_launch!(Val(_xphm_sums_plan(grad, _mbp_nt(spec))), backend, gb, KT, JT, F1, w0, nw, stv, gpu)
    return nothing
end
# The (modes, directions) groups above are written for 223's 7 tangents; 102 (5) drops directions 6, 7.
_xphm_restrict(gs, NT) = Tuple((ks, Tuple(j for j in js if j <= NT)) for (ks, js) in gs if any(<=(NT), js))
# The launch plan of the (b') stage from the tuning knobs: ((k, js) mode passes), ((fused?, ks, js, acc) twist passes).
function _xphm_sums_plan(grad::Bool, NT::Int = 7)
    # tangent groups of the mode passes (`GRAVITATIONALWAVEBAYES_XPHM_MODE_WIDTH`: at most that many tangents per Enzyme pass)
    mw = GPU.tuning(:XPHM_MODE_WIDTH, 7)
    # per mode (`GRAVITATIONALWAVEBAYES_XPHM_MODE_WIDTHS`, e.g. `7,4,4,4,4` or `7/4/4/4/4`; default: `XPHM_MODE_WIDTH` for all five)
    mws = String(GPU.tuning(:XPHM_MODE_WIDTHS, Symbol("")))
    mwk = isempty(mws) ? ntuple(_ -> mw, 5) : Tuple(parse.(Int, split(mws, r"[,/]")))
    mp = Tuple((k, js) for k in 1:5
                       for js in (grad ? Tuple(Tuple(j0:min(j0 + mwk[k] - 1, NT)) for j0 in 1:mwk[k]:NT) : ((1,),)))
    tw = GPU.tuning(:XPHM_TWIST, :fpair)
    fg = _xphm_fused_groups(Val(tw))
    fused = fg !== nothing
    gs = !grad ? map(k -> ((k,), (0,)), _XPHM_KS) : _xphm_restrict(fused ? fg : _xphm_twist_groups(Val(tw)), NT)
    done = Set{Int}()                                 # directions already stored by an earlier launch
    tp = map(gs) do (ks, js)
        acc = all(in(done), js)
        any(in(done), js) && !acc && error("twist group $js mixes stored and new directions")
        union!(done, js)
        (fused, ks, js, acc)
    end
    return (grad, mp, Tuple(tp))
end
# Every launch of a plan as a literal call (no per-launch dynamic dispatch on the large setup-type arguments: the
# host side of this stage was ~0.5 ms per launch).
@generated function _xphm_sums_launch!(::Val{PL}, backend, gb, KT, JT, F1, w0, nw, stv, gpu) where {PL}
    grad, mp, tp = PL
    ex = Any[:(nd = (length(F1), gb.wc)), :(mk! = _xphm_mode_kernel!(backend)), :(tk! = _xphm_twist_kernel!(backend)),
             :(fk! = _xphm_twist_fused_kernel!(backend))]
    for (k, js) in mp
        push!(ex, :(mk!(gb.XM, $(grad ? :(gb.XMT) : nothing), KT, JT, F1, w0, nw, Val($k), stv, Val($js), gpu;
                        ndrange=nd)))
    end
    for (fused, ks, js, acc) in tp
        push!(ex, :($(fused ? :fk! : :tk!)(gb.XH, gb.XHT, gb.XK, gb.XM, gb.XMT, KT, JT, F1, w0, nw, stv, Val($ks),
                                             Val($js), Val($grad), Val($acc); ndrange=nd)))
    end
    return Expr(:block, ex..., :(return nothing))
end


# Per detector (Fp'/dL, Fc'/dL) and tau_d + t_shift, from the sampled rows (generic element type: also the surrogate's).
@inline function _mbp_ext(spec::_XPHMSpec, x, g, dv)
    st, dec = _xphmbp_prec_setup(spec, x, g)
    return _xphmbp_ext(st, dec, x, g, dv)
end
# kappa and the extrinsic factors from one precession setup (the MSA system is the costly part of both)
@inline function _mbp_coords_ext(spec::_XPHMSpec, x, g, dv)
    st, dec = _xphmbp_prec_setup(spec, x, g)
    return (_mbp_theta(spec, x)..., st.thetaJN, st.alpha0, st.epsilon0), _xphmbp_ext(st, dec, x, g, dv)
end
@inline function _xphmbp_ext(st, dec, x, g, ::Val{D}) where {D}
    dL = x[5]; t_c = x[6]; ra = x[8]; psi = x[11]
    z2 = 2 * st.zeta_polarization
    cz = cos(z2); sz = sin(z2)
    z0 = zero(dL)                                    # (no DataType captured by the closure below: Julia 1.13.1 Enzyme)
    dt_ref = ifelse(g.geo & (g.tref > 0), time_delay_from_geocenter_gmst(g.dets[max(g.tref, 1)], ra, dec, g.gmst), z0)
    return ntuple(Val(D)) do d
        Fp, Fc = antenna_response_gmst(g.dets[d], ra, dec, psi, g.gmst)
        c = (Complex((Fp * cz - Fc * sz) / dL, z0), Complex((Fp * sz + Fc * cz) / dL, z0))
        tau = t_c + ifelse(g.geo, time_delay_from_geocenter_gmst(g.dets[d], ra, dec, g.gmst) - dt_ref, z0)
        (c, tau)
    end
end

"""
    xphm_binparallel_logp_and_grad(data, cfg, freqs; backend, nwalkers, setup=:host, timers=nothing) -> logp_and_grad!

IMRPhenomXPHM log-likelihood and gradient on `backend` (see the header of `XPHMBinParallel.jl`): the same closure as
[`xhm_binparallel_logp_and_grad`](@ref) for the 15-row sampled parametrization of `gw_logp_and_grad(...;
model=IMRPhenomXPHM())`, with the same likelihood options (distance marginalization, f_ref timing, time reference).
"""
xphm_binparallel_logp_and_grad(data, cfg::PEConfig, freqs; model::IMRPhenomXPHM = IMRPhenomXPHM(), kw...) =
    _mbp_logp_and_grad(_XPHMSpec{typeof(model).parameters[1]}(), data, cfg, freqs; kw...)
