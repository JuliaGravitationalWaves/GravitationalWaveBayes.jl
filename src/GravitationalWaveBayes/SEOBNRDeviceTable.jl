# ------------------------------------------------------------------------------------------------
# SEOBNRv5 IMR frequency-domain table built on a KernelAbstractions device, walker x node parallel
# (primal). Included inside `BatchedNUTS`.
#
# Produces, per walker, exactly what `GravitationalWaveforms._eobimr_coeffs(SEOBNRv5IMRAlignedSpinF2(), ...)` returns
# (the `EOBIMRTable`), as plain (node, walker) device arrays plus a per-walker scalar NamedTuple. The
# per-node / per-sample physics is GravitationalWaveforms' own scalar code: the dense-output, NQC, merger-ringdown and
# table-evaluation functions are duck-typed, so they run unchanged on a "dyn" / "table" NamedTuple whose
# vector fields are column views of the device arrays. Stages (one kernel each; "node" kernels have one
# thread per (node, walker), "walker" kernels one thread per walker and a serial loop):
#   base + PA grid (walker) -> PA adiabatic j0 (node) -> 8 PA orders (node) -> dt/dr, dphi/dr (node)
#   -> PA t, phi integrals (walker) -> dynamics init from the PA nodes (node) -> RK4 plunge (walker)
#   -> node |h22| and phase correction (node) -> their time derivatives (node)
#   -> t_ISCO, t_attach, NQC, merger-ringdown, table layout (walker) -> SPA table (node) + padding (node)
#   -> Fourier segment samples (sample) -> two radix-2 FFTs (walker) -> Fourier-table bins (bin)
#   -> phase unwrap, SPA/Fourier phase alignment, table scalars (walker).
# SEOBNRv5 IMR only (the PA stage is SEOBNRv5's fixed post-adiabatic solve).
# ------------------------------------------------------------------------------------------------

const _SDT_W = GravitationalWaveforms
const _SDT_NPA = GravitationalWaveforms._SEOBV5_PA_NMAX
const _SDT_NORD = 8
const _SDT_ND = GravitationalWaveforms._EOBIMR_NDYN
const _SDT_NRK = GravitationalWaveforms._EOBIMR_NRK
const _SDT_NT = GravitationalWaveforms._EOBIMR_NT
const _SDT_NSPA_RK = GravitationalWaveforms._EOBIMR_NSPA_RK
const _SDT_NS = GravitationalWaveforms._EOBIMR_NSEG
const _SDT_NF = GravitationalWaveforms._EOBIMR_NF
const _SDT_TBLEND1 = GravitationalWaveforms._EOBIMR_TBLEND[1]
const _SDT_TBLEND2 = GravitationalWaveforms._EOBIMR_TBLEND[2]
const _SDT_TSEG_PRE = GravitationalWaveforms._EOBIMR_TSEG_PRE
const _SDT_TTAPER = GravitationalWaveforms._EOBIMR_TTAPER
const _SDT_NTAU = GravitationalWaveforms._EOBIMR_NTAU
const _SDT_FHI_QNM = GravitationalWaveforms._EOBIMR_FHI_QNM
const _SDT_R_START = GravitationalWaveforms._EOBIMR_R_START
# Bound here, not read as `GravitationalWaveforms.X[i]` inside kernels: a module-qualified global const tuple indexed at
# run time fails GPU compilation ("unsupported use of an undefined name").
const _SDT_D9 = GravitationalWaveforms._SEOBV5_FINDIFF9_COEFFS
const _SDT_I7 = GravitationalWaveforms._SEOBV5_CUMINT7_COEFFS

# ---------------------------------------------------------------------------------------------- per-node physics
@inline _sdt_pr_eqn(trial, r, pphi, D, b) = _SDT_W._seobv5_pr_eqn(trial, r, pphi, D, b.hp, b.fc)
@inline _sdt_pphi_eqn(trial, r, pr, D, b) = _SDT_W._seobv5_pphi_eqn_fixed(trial, r, pr, D, b.hp, b.fc)

# dH/dr at (r, 0, p) and its p-derivative: the adiabatic j0 Newton iteration of `_seobv5_adiabatic_j0_fixed`.
@inline _sdt_j0_gd(ri, p, b) = _SDT_W._seobv5_sym_j0(ri, p, b.hp)

# `GravitationalWaveforms._eobimr_turn_on` (C-infinity turn-on 1/(1 + e^z), z = 1/x - 1/(1 - x)) with the logistic written in its
# overflow-free form: same value to round-off, but forward-mode AD on CUDA of 1/(1 + e^z) at z ~ 400 (one segment sample
# just inside the taper) gave NaN (e^z squared overflows).
@inline function _sdt_turn_on(x)
    x <= 0.0 && return 0.0
    x >= 1.0 && return 1.0
    z = 1.0 / x - 1.0 / (1.0 - x)
    z > 700.0 && return 0.0
    z < -700.0 && return 1.0
    e = exp(-abs(z))
    return ifelse(z > 0.0, e / (1.0 + e), 1.0 / (1.0 + e))
end

@inline function _sdt_tphi(r, pr, pphi, b)
    xi, dHdpr, dHdpphi = _SDT_W._seobv5_sym_tphi(r, pr, pphi, b.hp)
    q = dHdpr * xi
    return 1.0 / q, dHdpphi / q
end

# Rows of the 9-point derivative stencil / 7th-order cumulative integral and their window starts, with the
# same branch precedence as `_seobv5_findiff_derivative9_fixed` / `_seobv5_cumulative_integral7_fixed`.
@inline function _sdt_d9_row(i, nreal)
    C = _SDT_D9
    i <= 4 && return C[i], 1
    i >= nreal - 3 && return C[i - nreal + 9], nreal - 8
    return C[5], i - 4
end
@inline function _sdt_i7_row(i, nreal)      # i >= 2
    R = _SDT_I7
    (i >= 2 && i <= 4) && return R[i - 1], 1
    i >= nreal - 2 && return R[i - nreal + 7], nreal - 7
    return R[4], i - 4
end

# PA grid metadata exactly as `seobnrv5alignedspin_postadiabatic_dynamics_fixed` (default `dr0 = 0.2`, NMAX).
@inline function _sdt_pa_meta(b)
    nu, X1, X2, chi1, chi2 = b.nu, b.X1, b.X2, b.chi1, b.chi2
    r0 = b.r0
    chi_eff = X1 * chi1 + X2 * chi2
    r_final_prefactor = 2.7 + chi_eff * (1.0 - 4.0 * nu)
    in_domain = (0.0 < nu <= 0.25) & (abs(chi1) <= 1.0) & (abs(chi2) <= 1.0)
    r_isco, _ = in_domain ? _SDT_W._seobv5_kerr_isco(chi1, chi2, X1, X2) : _SDT_W._seobv5_kerr_isco(0.0, 0.0, 0.5, 0.5)
    r_final = max(10.0, r_final_prefactor * r_isco)
    nreal_span = (r0 - r_final) / 0.2
    # `unsafe_trunc` of the clamped `ceil` (same value as `ceil(Int, ...)`): the throwing conversion's InexactError
    # path left an unresolved Julia global in the PTX of the forward-mode (Enzyme) CUDA kernel.
    nreal_raw = isfinite(nreal_span) ? unsafe_trunc(Int, ceil(clamp(nreal_span, -1.0e6, 1.0e6))) : 0
    success = in_domain & !(nreal_raw <= 4 || r0 <= 11.5)
    nreal_bumped = nreal_raw < 10 ? 10 : nreal_raw
    nreal = success ? min(nreal_bumped, _SDT_NPA) : 10
    r0_grid = success ? r0 : 50.0
    r_final_grid = success ? r_final : 40.0
    dr = (r0_grid - r_final_grid) / (nreal - 1)
    return (pa_success = success, nreal = nreal, r0g = r0_grid, dr = dr)
end

# The "dyn" NamedTuple `_eobimr_dyn_at` & co. read, as column views of the dynamics arrays.
@inline function _sdt_dyn(D, w, meta)
    return (t = view(D.t, :, w), r = view(D.r, :, w), phi = view(D.phi, :, w), pr = view(D.pr, :, w),
        pphi = view(D.pphi, :, w), dr = view(D.dr, :, w), dphi = view(D.dphi, :, w), dpr = view(D.dpr, :, w),
        dpphi = view(D.dpphi, :, w), amp = view(D.amp, :, w), damp = view(D.damp, :, w), phc = view(D.phc, :, w),
        dphc = view(D.dphc, :, w), nqc_scale = Inf, n = meta.n, npa = meta.npa, stopped = meta.stopped)
end

# The table NamedTuple `_eobimr_spa_eval` / `_eobimr_ft_eval` / `_eobimr_eval` read.
@inline function _sdt_table(TB, w, s)
    return (success = s.success, M_sec = s.M_sec, eta = s.eta, M = s.M, nreal = s.nreal,
        ts = view(TB.ts, :, w), phis = view(TB.phis, :, w), omegas = view(TB.oms, :, w), Qs = view(TB.Qs, :, w),
        dQs = view(TB.dQs, :, w), fmin = s.fmin, fmax = s.fmax, fspa_max = s.fspa_max, f_b1 = s.f_b1, f_b2 = s.f_b2,
        t_a = s.t_a, ft_shift = s.ft_shift, ft_f = view(TB.ftf, :, w), ft_amp = view(TB.fta, :, w),
        ft_damp = view(TB.ftda, :, w), ft_arg = view(TB.ftg, :, w), ft_darg = view(TB.ftdg, :, w))
end

# Scalar records (concrete field types, so device arrays of them can be allocated from a zero prototype).
@inline _sdt_dmeta(n, npa, stopped) = (n = Int(n), npa = Int(npa), stopped = Bool(stopped))
@inline function _sdt_bld(t_a, t_isco, success, nqc, mr, t_start, use_spa, npa_use, t_seg0, nspa, t_top, t_w, Lt, dts, norm)
    return (t_a = Float64(t_a), t_isco = Float64(t_isco), success = Bool(success),
        nqc = (a1 = Float64(nqc.a1), a2 = Float64(nqc.a2), a3 = Float64(nqc.a3), b1 = Float64(nqc.b1),
            b2 = Float64(nqc.b2), shrink = Float64(nqc.shrink)),
        mr = map(Float64, mr), t_start = Float64(t_start), use_spa = Bool(use_spa), npa_use = Int(npa_use),
        t_seg0 = Float64(t_seg0), nspa = Int(nspa), t_top = Float64(t_top), t_w = Float64(t_w), Lt = Float64(Lt),
        dts = Float64(dts), norm = Float64(norm))
end
@inline _sdt_ftm(fmin, fspa_max, df, f_b1, j0, fhi, covered) = (fmin = Float64(fmin), fspa_max = Float64(fspa_max),
    df = Float64(df), f_b1 = Float64(f_b1), j0 = Int(j0), fhi = Float64(fhi), covered = Bool(covered))
@inline _sdt_tabs(success, M_sec, eta, M, nreal, fmin, fmax, fspa_max, f_b1, f_b2, t_a, ft_shift) =
    (success = Bool(success), M_sec = Float64(M_sec), eta = Float64(eta), M = Float64(M), nreal = Int(nreal),
     fmin = Float64(fmin), fmax = Float64(fmax), fspa_max = Float64(fspa_max), f_b1 = Float64(f_b1),
     f_b2 = Float64(f_b2), t_a = Float64(t_a), ft_shift = Float64(ft_shift))

# The SEOBNRv5 base coefficients (`_imr_base` = `_seobv5_coeffs` minus `T`) with the two initial-condition Newton solves
# (`_seobv5_circular_ic`: 8 2-D steps; `_seobv5_dissipative_ic`: 6 1-D steps) replaced by ONE step of each from the
# converged roots `ic = (r0, pphi0, pr0)` (computed by a primal pass): differentiated in forward mode with `ic`
# constant, that step's tangent is the implicit-function derivative, as for the PA j0 root (`_sdt_b_j0_step!`).
@inline function _sdt_diss_residual(prtrial, r0, pphi0, b)
    H0, xi0, Heff0, dHdpr0, omega, omega_circ, d2Hdr2, d2HdrdL = _SDT_W._seobv5_sym_diss_terms(r0, prtrial, pphi0, b.hp)
    dLdr = -d2Hdr2 / d2HdrdL
    _, Fphi = _SDT_W._seobv5_rr_force(prtrial, pphi0, omega, omega_circ, Heff0, H0, b.nu, b.fc)
    rdot = (1.0 / xi0) * Fphi / dLdr
    return rdot - dHdpr0
end

@inline function _sdt_base_ift(mc, eta, chi1, chi2, f_start, GM3, ic)
    delta = sqrt(max(1.0 - 4.0 * eta, 0.0))
    X1 = 0.5 * (1.0 + delta); X2 = 0.5 * (1.0 - delta)
    ap = X1 * chi1 + X2 * chi2; am = X1 * chi1 - X2 * chi2
    chiS = 0.5 * (chi1 + chi2); chiA = 0.5 * (chi1 - chi2)
    atest = (1.0 - 2.0 * eta) * chiS + delta * chiA
    a6 = _SDT_W._seobv5_a6_NS(eta)
    dSO = _SDT_W._seobv5_dSO(eta, ap, am)
    absN33 = _SDT_W._seobv5_multipole_prefix_abs(eta, X1, X2, 3, 3)
    absN21 = _SDT_W._seobv5_multipole_prefix_abs(eta, X1, X2, 2, 1)
    absN44 = _SDT_W._seobv5_multipole_prefix_abs(eta, X1, X2, 4, 4)
    absN32 = _SDT_W._seobv5_multipole_prefix_abs(eta, X1, X2, 3, 2)
    absN55 = _SDT_W._seobv5_multipole_prefix_abs(eta, X1, X2, 5, 5)
    absN43 = _SDT_W._seobv5_multipole_prefix_abs(eta, X1, X2, 4, 3)
    fc = _SDT_W._seobv5_flux_coeffs(eta, delta, chiS, chiA, atest, absN33, absN21, absN44, absN32, absN55, absN43)
    M_sec0 = (mc / eta^0.6) * GM3
    v0 = cbrt(pi * M_sec0 * f_start)
    omega0 = v0^3
    # one step of `_seobv5_circular_ic` from the converged (r0, pphi0)
    rhat, pphi = ic[1], ic[2]
    hp = _SDT_W.SEOBNRv5HamParams(eta, X1, X2, chi1, chi2, a6, dSO)
    dHdr, dHdpphi, J21, J11, J12 = _SDT_W._seobv5_sym_circ_hessian(rhat, pphi, hp)
    J22 = J11                       # (J11 J12; J21 J22) = (H_r,pphi H_pphi,pphi; H_rr H_r,pphi)
    F1 = dHdpphi - omega0
    F2 = dHdr
    det = J11 * J22 - J12 * J21
    r0 = rhat + (-F1 * J22 + F2 * J12) / det
    pphi0 = pphi + (-J11 * F2 + J21 * F1) / det
    M = mc / (eta^(3.0 / 5.0))
    b = (M = M, M_sec = M * GM3, eta = eta, X1 = X1, X2 = X2, chi1 = chi1, chi2 = chi2, nu = eta, delta = delta,
        chiS = chiS, chiA = chiA, atest = atest, hp = hp, fc = fc, r0 = r0, pphi0 = pphi0, pr0 = 0.0)
    # one step of `_seobv5_dissipative_ic` from the converged pr0 (at the stepped r0, pphi0)
    p0 = ic[3]
    h = 1e-8
    R0 = _sdt_diss_residual(p0, r0, pphi0, b)
    dR = (_sdt_diss_residual(p0 + h, r0, pphi0, b) - _sdt_diss_residual(p0 - h, r0, pphi0, b)) / (2h)
    return merge(b, (pr0 = p0 - R0 / dR,))
end

# ---------------------------------------------------------------------------------------------- stage bodies
# Every stage is a body function `(i, w, args...)` (node stages: one thread per (node, walker)) or `(w, args...)`
# (walker stages: one thread per walker, serial loop), run by the generic kernels below either as is (primal)
# or under Enzyme forward mode with 4 tangent directions (`BatchDuplicated` shadows: the same buffers of four
# tangent copies of the `SEOBNRDeviceTables`). Forward mode needs no tape, so the loops inside the bodies (Newton,
# RK4 plunge, bisection, scans) are differentiated directly, as the CPU AD does.

@inline function _sdt_b_base!(w, B, PAM, INDOM, POS, m, f_min, GM3)
    @inbounds begin
        mc, eta, chi1, chi2 = POS[1, w], POS[2, w], POS[3, w], POS[4, w]
        Msec_est = mc / eta^0.6 * GM3
        f_start = min(f_min, _SDT_R_START^(-1.5) / (pi * Msec_est))
        base = _SDT_W._imr_base(m, mc, eta, chi1, chi2, f_start, GM3)
        in_domain = (0.0 < base.eta <= 0.25) & (abs(base.chi1) <= 1.0) & (abs(base.chi2) <= 1.0) & isfinite(base.r0)
        b = in_domain ? base : _SDT_W._imr_base(m, 20.0, 0.24, 0.0, 0.0, 20.0, GM3)
        B[w] = b
        INDOM[w] = in_domain
        PAM[w] = _sdt_pa_meta(b)
    end
    return nothing
end

# Forward mode: the converged initial-condition roots as plain values (primal pass, no AD) ...
@inline function _sdt_b_base_ic!(w, IC, POS, m, f_min, GM3)
    @inbounds begin
        mc, eta, chi1, chi2 = POS[1, w], POS[2, w], POS[3, w], POS[4, w]
        f_start = min(f_min, _SDT_R_START^(-1.5) / (pi * (mc / eta^0.6 * GM3)))
        b = _SDT_W._imr_base(m, mc, eta, chi1, chi2, f_start, GM3)
        IC[1, w] = b.r0; IC[2, w] = b.pphi0; IC[3, w] = b.pr0
    end
    return nothing
end
# ... then `_sdt_b_base!` with one Newton step of each solve from them (implicit-function tangents).
@inline function _sdt_b_base_step!(w, B, PAM, INDOM, POS, IC, m, f_min, GM3)
    @inbounds begin
        mc, eta, chi1, chi2 = POS[1, w], POS[2, w], POS[3, w], POS[4, w]
        Msec_est = mc / eta^0.6 * GM3
        f_start = min(f_min, _SDT_R_START^(-1.5) / (pi * Msec_est))
        base = _sdt_base_ift(mc, eta, chi1, chi2, f_start, GM3, (IC[1, w], IC[2, w], IC[3, w]))
        in_domain = (0.0 < base.eta <= 0.25) & (abs(base.chi1) <= 1.0) & (abs(base.chi2) <= 1.0) & isfinite(base.r0)
        b = in_domain ? base : _SDT_W._imr_base(m, 20.0, 0.24, 0.0, 0.0, 20.0, GM3)
        B[w] = b
        INDOM[w] = in_domain
        PAM[w] = _sdt_pa_meta(b)
    end
    return nothing
end

@inline function _sdt_b_j0!(i, w, PR, PP, B, PAM)
    @inbounds begin
        pm = PAM[w]
        if i <= pm.nreal
            b = B[w]
            ri = pm.r0g - (i - 1) * pm.dr
            p = sqrt(ri)
            for _ in 1:8          # as `_seobv5_adiabatic_j0_fixed` (exact derivative: round-off after 4)
                g, d = _sdt_j0_gd(ri, p, b)
                p -= g / d
            end
            PP[i, 1, w] = p
            PR[i, 1, w] = 0.0
        end
    end
    return nothing
end

# Forward mode: the converged j0 root as a plain value (no AD through the 8 Newton steps) ...
@inline function _sdt_b_j0_newton!(i, w, J0, B, PAM)
    @inbounds begin
        pm = PAM[w]
        if i <= pm.nreal
            b = B[w]
            ri = pm.r0g - (i - 1) * pm.dr
            p = sqrt(ri)
            for _ in 1:8          # as `_seobv5_adiabatic_j0_fixed` (exact derivative: round-off after 4)
                g, d = _sdt_j0_gd(ri, p, b)
                p -= g / d
            end
            J0[i, w] = p
        end
    end
    return nothing
end
# ... then one Newton step at it, differentiated: p0 is constant, so its tangent is the implicit-function derivative
# -(dg/dtheta)/(dg/dp) (the same as differentiating through the converged iterations, at ~1/20 of the cost).
@inline function _sdt_b_j0_step!(i, w, PR, PP, J0, B, PAM)
    @inbounds begin
        pm = PAM[w]
        if i <= pm.nreal
            p0 = J0[i, w]
            g, d = _sdt_j0_gd(pm.r0g - (i - 1) * pm.dr, p0, B[w])
            PP[i, 1, w] = p0 - g / d
            PR[i, 1, w] = 0.0
        end
    end
    return nothing
end

@inline function _sdt_b_order!(i, w, PR, PP, DD, B, PAM, n)
    @inbounds begin
        pm = PAM[w]
        nreal = pm.nreal
        if i <= nreal
            b = B[w]; dr = pm.dr
            ri = pm.r0g - (i - 1) * dr
            row, j0 = _sdt_d9_row(i, nreal)
            s = 0.0
            if isodd(n)
                for k in 1:9
                    s += row[k] * PP[j0 + k - 1, n, w]
                end
                D = -(s / dr)
                PR[i, n + 1, w] = _sdt_pr_eqn(PR[i, n, w], ri, PP[i, n, w], D, b)
                PP[i, n + 1, w] = PP[i, n, w]
            else
                for k in 1:9
                    s += row[k] * PR[j0 + k - 1, n, w]
                end
                D = -(s / dr)
                PP[i, n + 1, w] = _sdt_pphi_eqn(PP[i, n, w], ri, PR[i, n, w], D, b)
                PR[i, n + 1, w] = PR[i, n, w]
            end
            DD[i, n, w] = D
        end
    end
    return nothing
end

@inline function _sdt_b_tphi!(i, w, DTR, DPHR, PR, PP, B, PAM)
    @inbounds begin
        pm = PAM[w]
        if i <= pm.nreal
            ri = pm.r0g - (i - 1) * pm.dr
            a, c = _sdt_tphi(ri, PR[i, _SDT_NORD + 1, w], PP[i, _SDT_NORD + 1, w], B[w])
            DTR[i, w] = a; DPHR[i, w] = c
        end
    end
    return nothing
end

# Cumulative 7th-order integrals of dt/dr and dphi/dr (step -dr), as `_seobv5_cumulative_integral7_fixed`.
@inline function _sdt_b_pa_integrate!(w, PT, PPH, DTR, DPHR, PAM)
    @inbounds begin
        pm = PAM[w]
        nreal = pm.nreal; h = -pm.dr
        at = 0.0; ap = 0.0
        PT[1, w] = 0.0; PPH[1, w] = 0.0
        for i in 2:nreal
            row, j0 = _sdt_i7_row(i, nreal)
            st = 0.0; sp = 0.0
            for k in 1:8
                st += row[k] * DTR[j0 + k - 1, w]; sp += row[k] * DPHR[j0 + k - 1, w]
            end
            at += st * h; ap += sp * h
            PT[i, w] = at; PPH[i, w] = ap
        end
    end
    return nothing
end

# Dynamics nodes 1..npa from the PA table (or the short-inspiral start state), with the rhs at each node.
@inline function _sdt_b_dyn_init!(i, w, D, PT, PPH, PR, PP, B, PAM, m)
    @inbounds begin
        pm = PAM[w]
        use_pa = pm.pa_success
        npa = use_pa ? pm.nreal : 1
        if i <= npa
            b = B[w]
            r0s, pr0s, pp0s = _SDT_W._imr_short_start(m, b)
            ri = use_pa ? pm.r0g - (i - 1) * pm.dr : r0s
            pri = use_pa ? PR[i, _SDT_NORD + 1, w] : pr0s
            ppi = use_pa ? PP[i, _SDT_NORD + 1, w] : pp0s
            D.t[i, w] = use_pa ? PT[i, w] : 0.0
            D.phi[i, w] = use_pa ? PPH[i, w] : 0.0
            D.r[i, w] = ri; D.pr[i, w] = pri; D.pphi[i, w] = ppi
            a, bb, c, d = _SDT_W._imr_rhs4(m, ri, pri, ppi, b)
            D.dr[i, w] = a; D.dphi[i, w] = bb; D.dpr[i, w] = c; D.dpphi[i, w] = d
        end
    end
    return nothing
end

# The adaptive RK4 plunge of `_eobimr_dynamics` (its step rule `_imr_plunge_steps` / `_imr_plunge_h`, including the
# halving retries before a stop), appended after node npa (serial per walker).
@inline function _sdt_b_plunge!(w, DM, D, B, PAM, m)
    @inbounds begin
        pm = PAM[w]; b = B[w]
        npa = pm.pa_success ? pm.nreal : 1
        sr, sph, spr, spp, st = D.r[npa, w], D.phi[npa, w], D.pr[npa, w], D.pphi[npa, w], D.t[npa, w]
        om_prev = D.dphi[npa, w]
        # k1 of every step is the rhs at the current node: stored by `_sdt_b_dyn_init!` for node npa, then the
        # acceptance check's rhs (4 rhs evaluations per step instead of 5; equal to `_eobimr_dynamics` to round-off).
        k1r, k1f, k1p, k1j = D.dr[npa, w], om_prev, D.dpr[npa, w], D.dpphi[npa, w]
        sp = _SDT_W._imr_plunge_steps(m)
        hcap = sp.hmax
        stopped = false
        n = npa
        for _ in 1:_SDT_NRK
            if !stopped
                h = _SDT_W._imr_plunge_h(sp, k1f, sr, k1r, hcap)
                r2 = sr + 0.5h * k1r
                k2r, k2f, k2p, k2j = _SDT_W._imr_rhs4(m, _SDT_W._imr_stage_r(r2, sr), spr + 0.5h * k1p, spp + 0.5h * k1j, b)
                r3 = sr + 0.5h * k2r
                k3r, k3f, k3p, k3j = _SDT_W._imr_rhs4(m, _SDT_W._imr_stage_r(r3, sr), spr + 0.5h * k2p, spp + 0.5h * k2j, b)
                r4 = sr + h * k3r
                k4r, k4f, k4p, k4j = _SDT_W._imr_rhs4(m, _SDT_W._imr_stage_r(r4, sr), spr + h * k3p, spp + h * k3j, b)
                nr = sr + h / 6.0 * (k1r + 2k2r + 2k3r + k4r)
                nph = sph + h / 6.0 * (k1f + 2k2f + 2k3f + k4f)
                npr = spr + h / 6.0 * (k1p + 2k2p + 2k3p + k4p)
                npp = spp + h / 6.0 * (k1j + 2k2j + 2k3j + k4j)
                rs = _SDT_W._EOBIMR_R_STOP
                okr = isfinite(nr) & isfinite(npr) & isfinite(npp) & (nr > rs) & (npp > 0.0) & (min(r2, r3, r4) > rs)
                ok = okr
                if ok
                    a, bb, c, d = _SDT_W._imr_rhs4(m, nr, npr, npp, b)
                    ok = (bb > om_prev) & (a < 0.0)
                    if ok
                        n += 1
                        st += h
                        sr, sph, spr, spp = nr, nph, npr, npp
                        D.t[n, w] = st; D.r[n, w] = nr; D.phi[n, w] = nph; D.pr[n, w] = npr; D.pphi[n, w] = npp
                        D.dr[n, w] = a; D.dphi[n, w] = bb; D.dpr[n, w] = c; D.dpphi[n, w] = d
                        om_prev = bb
                        k1r, k1f, k1p, k1j = a, bb, c, d
                    end
                end
                # a step past the frequency peak / at dr/dt >= 0 is retried with half the step (`_imr_plunge_h`)
                retry = okr & !ok & (h > sp.hend)
                hcap = ifelse(retry, 0.5h, hcap)
                stopped = !ok & !retry
            end
        end
        DM[w] = _sdt_dmeta(n, npa, stopped)
    end
    return nothing
end

# The same plunge as host-driven stages (`plunge_split = true`, forward mode): one kernel per RK4 stage, each with ONE
# rhs evaluation, the loop state in a per-walker global array `PSF` (rows below) instead of registers. In the single
# kernel the loop state (state, k1, the RK sum, h, previous omega: ~15 values x 5 with the tangents) is live across
# every rhs call, on top of the rhs's own live values. Same operations in the same order as `_sdt_b_plunge!`.
const _PS_R, _PS_PHI, _PS_PR, _PS_PP, _PS_T, _PS_OM = 1, 2, 3, 4, 5, 6
const _PS_K1 = 7          # 7..10: k1 (r, phi, pr, pphi)
const _PS_K = 11          # 11..14: the latest stage's k
const _PS_ACC = 15        # 15..18: running RK sum k1 + 2 k2 + 2 k3 + k4
const _PS_H = 19
const _PS_C = 20          # 20..23: candidate state (r, phi, pr, pphi)
const _PS_HCAP = 24       # step cap of the halving retries
const _PS_BAD = 25        # 1.0 if a stage radius fell below the stopping radius
const _PS_NROW = 25

@inline function _sdt_ps_init!(w, PSF, PSI, PSB, D, PAM, m)
    @inbounds begin
        pm = PAM[w]
        npa = pm.pa_success ? pm.nreal : 1
        PSF[_PS_R, w] = D.r[npa, w]; PSF[_PS_PHI, w] = D.phi[npa, w]; PSF[_PS_PR, w] = D.pr[npa, w]
        PSF[_PS_PP, w] = D.pphi[npa, w]; PSF[_PS_T, w] = D.t[npa, w]; PSF[_PS_OM, w] = D.dphi[npa, w]
        PSF[_PS_K1, w] = D.dr[npa, w]; PSF[_PS_K1 + 1, w] = D.dphi[npa, w]
        PSF[_PS_K1 + 2, w] = D.dpr[npa, w]; PSF[_PS_K1 + 3, w] = D.dpphi[npa, w]
        PSF[_PS_HCAP, w] = _SDT_W._imr_plunge_steps(m).hmax
        PSI[w] = npa
        PSB[w] = false
    end
    return nothing
end

# k2 (and the step size): x = s + h/2 k1
@inline function _sdt_ps_k2!(w, PSF, PSB, B, m)
    @inbounds if !PSB[w]
        sr, spr, spp = PSF[_PS_R, w], PSF[_PS_PR, w], PSF[_PS_PP, w]
        k1r, k1f, k1p, k1j = PSF[_PS_K1, w], PSF[_PS_K1 + 1, w], PSF[_PS_K1 + 2, w], PSF[_PS_K1 + 3, w]
        h = _SDT_W._imr_plunge_h(_SDT_W._imr_plunge_steps(m), k1f, sr, k1r, PSF[_PS_HCAP, w])
        r2 = sr + 0.5h * k1r
        kr, kf, kp, kj = _SDT_W._imr_rhs4(m, _SDT_W._imr_stage_r(r2, sr), spr + 0.5h * k1p, spp + 0.5h * k1j, B[w])
        PSF[_PS_BAD, w] = ifelse(r2 > _SDT_W._EOBIMR_R_STOP, 0.0, 1.0)
        PSF[_PS_H, w] = h
        PSF[_PS_K, w] = kr; PSF[_PS_K + 1, w] = kf; PSF[_PS_K + 2, w] = kp; PSF[_PS_K + 3, w] = kj
        PSF[_PS_ACC, w] = k1r + 2kr; PSF[_PS_ACC + 1, w] = k1f + 2kf
        PSF[_PS_ACC + 2, w] = k1p + 2kp; PSF[_PS_ACC + 3, w] = k1j + 2kj
    end
    return nothing
end

# k3: x = s + h/2 k2
@inline function _sdt_ps_k3!(w, PSF, PSB, B, m)
    @inbounds if !PSB[w]
        sr, spr, spp, h = PSF[_PS_R, w], PSF[_PS_PR, w], PSF[_PS_PP, w], PSF[_PS_H, w]
        k2r, k2p, k2j = PSF[_PS_K, w], PSF[_PS_K + 2, w], PSF[_PS_K + 3, w]
        r3 = sr + 0.5h * k2r
        kr, kf, kp, kj = _SDT_W._imr_rhs4(m, _SDT_W._imr_stage_r(r3, sr), spr + 0.5h * k2p, spp + 0.5h * k2j, B[w])
        PSF[_PS_BAD, w] = ifelse(r3 > _SDT_W._EOBIMR_R_STOP, PSF[_PS_BAD, w], 1.0)
        PSF[_PS_K, w] = kr; PSF[_PS_K + 1, w] = kf; PSF[_PS_K + 2, w] = kp; PSF[_PS_K + 3, w] = kj
        PSF[_PS_ACC, w] += 2kr; PSF[_PS_ACC + 1, w] += 2kf; PSF[_PS_ACC + 2, w] += 2kp; PSF[_PS_ACC + 3, w] += 2kj
    end
    return nothing
end

# k4: x = s + h k3; candidate state
@inline function _sdt_ps_k4!(w, PSF, PSB, B, m)
    @inbounds if !PSB[w]
        sr, sph, spr, spp, h = PSF[_PS_R, w], PSF[_PS_PHI, w], PSF[_PS_PR, w], PSF[_PS_PP, w], PSF[_PS_H, w]
        k3r, k3p, k3j = PSF[_PS_K, w], PSF[_PS_K + 2, w], PSF[_PS_K + 3, w]
        r4 = sr + h * k3r
        kr, kf, kp, kj = _SDT_W._imr_rhs4(m, _SDT_W._imr_stage_r(r4, sr), spr + h * k3p, spp + h * k3j, B[w])
        PSF[_PS_BAD, w] = ifelse(r4 > _SDT_W._EOBIMR_R_STOP, PSF[_PS_BAD, w], 1.0)
        nr = sr + h / 6.0 * (PSF[_PS_ACC, w] + kr)
        nph = sph + h / 6.0 * (PSF[_PS_ACC + 1, w] + kf)
        npr = spr + h / 6.0 * (PSF[_PS_ACC + 2, w] + kp)
        npp = spp + h / 6.0 * (PSF[_PS_ACC + 3, w] + kj)
        PSF[_PS_C, w] = nr; PSF[_PS_C + 1, w] = nph; PSF[_PS_C + 2, w] = npr; PSF[_PS_C + 3, w] = npp
    end
    return nothing
end

# validity and rhs at the candidate: accept (store the node, advance, k1 = this rhs), retry with half the step, or stop
@inline function _sdt_ps_accept!(w, PSF, PSI, PSB, D, B, m)
    @inbounds if !PSB[w]
        nr, nph, npr, npp = PSF[_PS_C, w], PSF[_PS_C + 1, w], PSF[_PS_C + 2, w], PSF[_PS_C + 3, w]
        okr = isfinite(nr) & isfinite(npr) & isfinite(npp) & (nr > _SDT_W._EOBIMR_R_STOP) & (npp > 0.0) &
              (PSF[_PS_BAD, w] == 0.0)
        ok = okr
        a = bb = c = d = 0.0
        if ok
            a, bb, c, d = _SDT_W._imr_rhs4(m, nr, npr, npp, B[w])
            ok = (bb > PSF[_PS_OM, w]) & (a < 0.0)
        end
        if ok
            n = PSI[w] + 1
            st = PSF[_PS_T, w] + PSF[_PS_H, w]
            D.t[n, w] = st; D.r[n, w] = nr; D.phi[n, w] = nph; D.pr[n, w] = npr; D.pphi[n, w] = npp
            D.dr[n, w] = a; D.dphi[n, w] = bb; D.dpr[n, w] = c; D.dpphi[n, w] = d
            PSF[_PS_R, w] = nr; PSF[_PS_PHI, w] = nph; PSF[_PS_PR, w] = npr; PSF[_PS_PP, w] = npp; PSF[_PS_T, w] = st
            PSF[_PS_OM, w] = bb
            PSF[_PS_K1, w] = a; PSF[_PS_K1 + 1, w] = bb; PSF[_PS_K1 + 2, w] = c; PSF[_PS_K1 + 3, w] = d
            PSI[w] = n
        else
            h = PSF[_PS_H, w]
            retry = okr & (h > _SDT_W._imr_plunge_steps(m).hend)
            PSF[_PS_HCAP, w] = ifelse(retry, 0.5h, PSF[_PS_HCAP, w])
            PSB[w] = !retry
        end
    end
    return nothing
end

@inline function _sdt_ps_finish!(w, DM, PSI, PSB, PAM)
    @inbounds begin
        pm = PAM[w]
        DM[w] = _sdt_dmeta(PSI[w], pm.pa_success ? pm.nreal : 1, PSB[w])
    end
    return nothing
end

# Node |h22| and phase correction; padding nodes (> n) get strictly increasing times and zeros elsewhere.
@inline function _sdt_b_amp!(i, w, D, DM, B, m)
    @inbounds begin
        n = DM[w].n
        if i <= n
            b = B[w]
            D.amp[i, w] = _SDT_W._imr_amp(m, D.r[i, w], D.pr[i, w], D.pphi[i, w], D.dphi[i, w], b)
            D.phc[i, w] = _SDT_W._imr_phase_corr(m, D.r[i, w], D.pr[i, w], D.pphi[i, w], D.dphi[i, w], b)
        else
            D.t[i, w] = D.t[n, w] + Float64(i - n)
            D.r[i, w] = 0.0; D.phi[i, w] = 0.0; D.pr[i, w] = 0.0; D.pphi[i, w] = 0.0
            D.dr[i, w] = 0.0; D.dphi[i, w] = 0.0; D.dpr[i, w] = 0.0; D.dpphi[i, w] = 0.0
            D.amp[i, w] = 0.0; D.phc[i, w] = 0.0
        end
    end
    return nothing
end

# d|h22|/dt and d(phasecorr)/dt: 3-point non-uniform-grid derivative (one-sided at the ends).
@inline function _sdt_b_damp!(i, w, D, DM)
    @inbounds begin
        n = DM[w].n
        t = D.t; amp = D.amp; phc = D.phc
        if i <= n
            if i == 1
                D.damp[i, w] = (amp[2, w] - amp[1, w]) / (t[2, w] - t[1, w])
                D.dphc[i, w] = (phc[2, w] - phc[1, w]) / (t[2, w] - t[1, w])
            elseif i == n
                D.damp[i, w] = (amp[n, w] - amp[n - 1, w]) / (t[n, w] - t[n - 1, w])
                D.dphc[i, w] = (phc[n, w] - phc[n - 1, w]) / (t[n, w] - t[n - 1, w])
            else
                h1 = t[i, w] - t[i - 1, w]; h2 = t[i + 1, w] - t[i, w]
                c1 = -h2 / (h1 * (h1 + h2)); c2 = (h2 - h1) / (h1 * h2); c3 = h1 / (h2 * (h1 + h2))
                D.damp[i, w] = c1 * amp[i - 1, w] + c2 * amp[i, w] + c3 * amp[i + 1, w]
                D.dphc[i, w] = c1 * phc[i - 1, w] + c2 * phc[i, w] + c3 * phc[i + 1, w]
            end
        else
            D.damp[i, w] = 0.0; D.dphc[i, w] = 0.0
        end
    end
    return nothing
end

# t_ISCO, t_attach, NQC and merger-ringdown (`_eobimr_build`), then the table layout of `_eobimr_coeffs`.
@inline function _sdt_b_build!(w, BLD, DM, B, INDOM, D)
    @inbounds begin
        b = B[w]; meta = DM[w]
        dyn = _sdt_dyn(D, w, meta)
        n = meta.n
        nu, X1, X2, c1, c2 = b.eta, b.X1, b.X2, b.chi1, b.chi2
        t_start, t_end = dyn.t[1], dyn.t[n]
        rI, _ = _SDT_W._seobv5_kerr_isco(c1, c2, X1, X2)
        i_is = 1
        for i in 1:_SDT_ND
            if i <= n && dyn.r[i] >= rI
                i_is = i
            end
        end
        t_isco = t_end
        if i_is < n
            lo, hi = dyn.t[i_is], dyn.t[i_is + 1]
            for _ in 1:40
                mid = 0.5 * (lo + hi)
                if _SDT_W._eobimr_dyn_at(dyn, mid)[1] >= rI
                    lo = mid
                else
                    hi = mid
                end
            end
            tb = 0.5 * (lo + hi)
            x = _SDT_W._eobimr_dyn_at(dyn, tb)
            rb, rdot = x[1], x[7]
            t_isco = tb - (rb - rI) / rdot
        end
        ap = X1 * c1 + X2 * c2
        am = X1 * c1 - X2 * c2
        t_a = clamp(t_isco - _SDT_W._eobimr_nr_deltaT(nu, ap, am), t_start + 1.0, t_end - 1.0)
        chi = _SDT_W._eobimr_fit_chi(nu, X1, X2, c1, c2)
        nqc = _SDT_W._eobimr_nqc(dyn, t_a, nu, chi)
        mr = _SDT_W._eobimr_mr_params(dyn, b, nqc, t_a, nu, chi)
        success = INDOM[w] & (b.r0 >= 6.0) & meta.stopped & (n >= meta.npa + 8) &
                  (t_a > t_start + 20.0) & isfinite(nqc.a1 + nqc.a2 + nqc.a3 + nqc.b1 + nqc.b2) &
                  isfinite(mr.c1c + mr.c2c + mr.d1c) & (mr.w_a < mr.omR)
        t_spa_end = t_a - _SDT_TBLEND2
        use_spa = t_spa_end > t_start + 10.0
        npa = meta.npa
        npa_use = 1
        for i in 1:_SDT_NPA
            npa_use = (use_spa & (i <= npa) && (dyn.t[i] <= t_spa_end - 1.0)) ? i : npa_use
        end
        t_seg0 = dyn.t[npa_use]
        nspa = use_spa ? npa_use + _SDT_NSPA_RK : 2
        t_top = use_spa ? t_spa_end : t_start + 2.0
        Ms = b.M_sec
        t_w = max(t_start, t_a - _SDT_TSEG_PRE)
        Lt = min(_SDT_TTAPER, 0.4 * (t_a - t_w))
        t_e = t_a + _SDT_NTAU * mr.tau
        dts = (t_e - t_w) / Float64(_SDT_NS - 1)
        norm = Ms * dts / (4.0 * sqrt(pi / 5.0))
        BLD[w] = _sdt_bld(t_a, t_isco, success, nqc, mr, t_start, use_spa, npa_use, t_seg0, nspa, t_top, t_w, Lt, dts, norm)
    end
    return nothing
end

@inline function _sdt_b_spa!(i, w, TB, BLD, DM, B, D)
    @inbounds begin
        bl = BLD[w]
        if i <= bl.nspa
            dyn = _sdt_dyn(D, w, DM[w])
            nqc = bl.nqc; nu = B[w].eta
            npa_use = bl.npa_use; nspa = bl.nspa
            tq = i <= npa_use ? dyn.t[i] :
                 bl.t_seg0 + (bl.t_top - bl.t_seg0) * Float64(i - npa_use) / Float64(nspa - npa_use)
            TB.ts[i, w] = tq
            TB.phis[i, w] = 0.5 * _SDT_W._eobimr_mode_at(dyn, nqc, tq)[2]
            TB.oms[i, w] = 0.5 * _SDT_W._eobimr_freq_insp(dyn, nqc, tq)
            TB.Qs[i, w] = _SDT_W._eobimr_R(dyn, nqc, nu, tq)        # R and dR/dt; Q in `_sdt_b_spa_finish!`
            TB.dQs[i, w] = _SDT_W._eobimr_R(dyn, nqc, nu, tq + 0.5) - _SDT_W._eobimr_R(dyn, nqc, nu, tq - 0.5)
        end
    end
    return nothing
end

# Padding nodes (i > nspa), and R -> Q = R / sqrt(d omega/dt) on the real ones (`GravitationalWaveforms._spa_node_q`, as
# `_eobimr_coeffs`). Both read only real nodes' ts / oms, final after `_sdt_b_spa!`; node i writes only entry i.
@inline function _sdt_b_spa_finish!(i, w, TB, BLD)
    @inbounds begin
        nspa = BLD[w].nspa
        if i > nspa
            TB.ts[i, w] = TB.ts[nspa, w] + Float64(i - nspa)
            TB.oms[i, w] = TB.oms[nspa, w] + Float64(i - nspa)
            TB.phis[i, w] = 0.0; TB.Qs[i, w] = 0.0; TB.dQs[i, w] = 0.0
        else
            TB.Qs[i, w], TB.dQs[i, w] = _SDT_W._spa_node_q(view(TB.ts, :, w), view(TB.oms, :, w), TB.Qs[i, w],
                TB.dQs[i, w], i, nspa)
        end
    end
    return nothing
end

# Fourier segment: tapered mode samples s_k (e^{-i varphi}) and (t_k - t_a) s_k.
@inline function _sdt_b_segment!(k, w, SEG, BLD, DM, D)
    @inbounds begin
        bl = BLD[w]
        dyn = _sdt_dyn(D, w, DM[w])
        tk = bl.t_w + bl.dts * Float64(k - 1)
        A, vp = _SDT_W._eobimr_full_mode(dyn, bl.nqc, bl.mr, bl.t_a, tk)
        wgt = _sdt_turn_on((tk - bl.t_w) / bl.Lt) * bl.norm * A
        sr = wgt * cos(vp)
        si = -wgt * sin(vp)
        SEG.sre[k, w] = sr; SEG.sim[k, w] = si
        SEG.tre[k, w] = (tk - bl.t_a) * sr; SEG.tim[k, w] = (tk - bl.t_a) * si
    end
    return nothing
end

# Per-walker Fourier-table window (fmin, blend band start, first kept bin, table top).
@inline function _sdt_b_ftmeta!(w, FTM, TB, BLD, DM, B, D)
    @inbounds begin
        bl = BLD[w]; b = B[w]
        Ms = b.M_sec
        fmin = TB.oms[1, w] / (pi * Ms)
        fspa_max = TB.oms[bl.nspa, w] / (pi * Ms)
        df = 1.0 / (Float64(_SDT_NS) * bl.dts * Ms)
        dyn = _sdt_dyn(D, w, DM[w])
        f_b1_est = bl.use_spa ? _SDT_W._eobimr_freq_insp(dyn, bl.nqc, bl.t_a - _SDT_TBLEND1) / (2pi * Ms) : fmin
        # = clamp(floor(Int, x), 1, NS/2 - NF) for finite x, without the throwing conversion (see `_sdt_pa_meta`)
        j0 = clamp(unsafe_trunc(Int, floor(clamp(0.8 * f_b1_est / df, 1.0, Float64(_SDT_NS ÷ 2 - _SDT_NF)))), 1,
            _SDT_NS ÷ 2 - _SDT_NF)
        fhi = _SDT_FHI_QNM * bl.mr.omR / (2pi * Ms)
        covered = (Float64(j0 + _SDT_NF - 1) * df >= fhi) & (fhi < 0.45 / (bl.dts * Ms))
        FTM[w] = _sdt_ftm(fmin, fspa_max, df, f_b1_est, j0, fhi, covered)
    end
    return nothing
end

@inline function _sdt_b_ftbin!(j, w, TB, SEG, FTM, BLD, B)
    @inbounds begin
        fm = FTM[w]; bl = BLD[w]
        Ms = B[w].M_sec
        jj = fm.j0 + j - 1
        f = Float64(jj) * fm.df
        TB.ftf[j, w] = f
        th = 2pi * f * Ms * (bl.t_w - bl.t_a)
        cth, sth = cos(th), sin(th)
        Xr, Xi = SEG.sre[jj + 1, w], SEG.sim[jj + 1, w]
        Ur = Xr * cth - Xi * sth
        Ui = Xr * sth + Xi * cth
        Vr = SEG.tre[jj + 1, w] * cth - SEG.tim[jj + 1, w] * sth
        Vi = SEG.tre[jj + 1, w] * sth + SEG.tim[jj + 1, w] * cth
        dUr = -2pi * Ms * Vi
        dUi = 2pi * Ms * Vr
        a2 = Ur * Ur + Ui * Ui
        a = sqrt(a2)
        TB.fta[j, w] = a
        TB.ftda[j, w] = (Ur * dUr + Ui * dUi) / a
        TB.ftg[j, w] = atan(Ui, Ur)
        TB.ftdg[j, w] = (Ur * dUi - Ui * dUr) / a2
    end
    return nothing
end

# Phase unwrap (guided by the analytic phase derivative), SPA/Fourier phase alignment, table scalars.
@inline function _sdt_b_finalize!(w, TABS, TB, FTM, BLD, B)
    @inbounds begin
        for j in 2:_SDT_NF
            pred = TB.ftg[j - 1, w] + 0.5 * (TB.ftdg[j - 1, w] + TB.ftdg[j, w]) * (TB.ftf[j, w] - TB.ftf[j - 1, w])
            TB.ftg[j, w] += 2pi * round((pred - TB.ftg[j, w]) / (2pi))
        end
        fm = FTM[w]; bl = BLD[w]; b = B[w]
        f_b1 = fm.f_b1
        f_b2 = bl.use_spa ? fm.fspa_max : fm.fmin
        success = bl.success & fm.covered
        s0 = _sdt_tabs(success, b.M_sec, b.eta, b.M, bl.nspa, fm.fmin, fm.fhi, fm.fspa_max, f_b1, f_b2, bl.t_a, 0.0)
        tab0 = _sdt_table(TB, w, s0)
        fmid = 0.5 * (f_b1 + f_b2)
        shift = bl.use_spa ? 2pi * round((_SDT_W._eobimr_spa_eval(tab0, fmid)[2] - _SDT_W._eobimr_ft_eval(tab0, fmid)[2]) / (2pi)) : 0.0
        fmin_out = bl.use_spa ? fm.fmin : TB.ftf[1, w]
        TABS[w] = _sdt_tabs(success, b.M_sec, b.eta, b.M, bl.nspa, fmin_out, fm.fhi, fm.fspa_max, f_b1, f_b2, bl.t_a, shift)
    end
    return nothing
end

# ---------------------------------------------------------------------------------------------- generic kernels
# In-kernel AD: `autodiff_deferred` on a GPU, plain `autodiff` on the KernelAbstractions CPU backend.
@inline _sdt_ann(a, ::Nothing) = Enzyme.Const(a)
@inline _sdt_ann(a, s::NTuple{4,Any}) = Enzyme.BatchDuplicated(a, s, false)

@kernel function _sdt_node_kernel!(body, args)
    i, w = @index(Global, NTuple)
    body(i, w, args...)
end
@kernel function _sdt_walker_kernel!(body, args)
    w = @index(Global, Linear)
    body(w, args...)
end
@kernel function _sdt_node_fwd_kernel!(body, args, shadows, gpu)
    i, w = @index(Global, NTuple)
    _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(body), Enzyme.Const, Enzyme.Const(i), Enzyme.Const(w),
        map(_sdt_ann, args, shadows)...)
end
@kernel function _sdt_walker_fwd_kernel!(body, args, shadows, gpu)
    w = @index(Global, Linear)
    _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(body), Enzyme.Const, Enzyme.Const(w), map(_sdt_ann, args, shadows)...)
end

# Walker stages in forward mode with the 4 tangent directions on 4 threads per walker (width-1 `Duplicated`, the
# primal recomputed by each; identical primal stores): the serial stages (RK4 plunge, base Newton loops) are latency-
# bound, and a width-1 chain is much shorter than a width-4 one.
@inline _sdt_ann1(a, ::Nothing, k) = Enzyme.Const(a)
@inline _sdt_ann1(a, s::NTuple{4,Any}, k) = Enzyme.Duplicated(a, @inbounds s[k])
@kernel function _sdt_walker_dir_fwd_kernel!(body, args, shadows, gpu)
    k, w = @index(Global, NTuple)
    _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(body), Enzyme.Const, Enzyme.Const(w),
        map((a, s) -> _sdt_ann1(a, s, k), args, shadows)...)
end

# Two threads per walker, each with two of the four directions (width-2 `BatchDuplicated`).
@inline _sdt_ann2(a, ::Nothing, k) = Enzyme.Const(a)
@inline _sdt_ann2(a, s::NTuple{4,Any}, k) = @inbounds Enzyme.BatchDuplicated(a, (s[2k - 1], s[2k]), false)
@kernel function _sdt_walker_dir2_fwd_kernel!(body, args, shadows, gpu)
    k, w = @index(Global, NTuple)
    _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(body), Enzyme.Const, Enzyme.Const(w),
        map((a, s) -> _sdt_ann2(a, s, k), args, shadows)...)
end

@kernel function _sdt_node_dir_fwd_kernel!(body, args, shadows, gpu)
    k, i, w = @index(Global, NTuple)
    _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(body), Enzyme.Const, Enzyme.Const(i), Enzyme.Const(w),
        map((a, s) -> _sdt_ann1(a, s, k), args, shadows)...)
end

# Run `body` over ndrange `nd` (a tuple: node stage, an Int: walker stage); with shadows, forward mode
# (`dirpar = true`: one thread per (direction, walker) or (direction, node, walker), width-1 tangents; `dirpar = 2`,
# walker stages: two threads per walker with two directions each).
function _sdt_run!(be, body, nd, args, shadows; wg = nothing, dirpar = false)
    gpu = Val(!(be isa KernelAbstractions.CPU))
    # Kernels are always built with a DYNAMIC workgroup size and `wg` is given at launch: a static size becomes a PTX
    # `.maxntid` bound, with which ptxas capped the width-4 forward plunge kernel at 64 registers (68.8 KB stack frame,
    # 172 KB of spill stores) instead of 255 (offline `ptxas -v`).
    if shadows !== nothing && dirpar === 2 && !(nd isa Tuple)
        _sdt_walker_dir2_fwd_kernel!(be)(body, args, shadows, gpu; ndrange = (2, nd),
            workgroupsize = wg === nothing ? nothing : (2, max(wg ÷ 2, 1)))
        return nothing
    end
    if shadows !== nothing && dirpar === true
        if nd isa Tuple
            _sdt_node_dir_fwd_kernel!(be)(body, args, shadows, gpu; ndrange = (4, nd...))
        else
            _sdt_walker_dir_fwd_kernel!(be)(body, args, shadows, gpu; ndrange = (4, nd),
                workgroupsize = wg === nothing ? nothing : (4, max(wg ÷ 4, 1)))
        end
        return nothing
    end
    if shadows === nothing
        if nd isa Tuple
            _sdt_node_kernel!(be)(body, args; ndrange = nd)
        else
            _sdt_walker_kernel!(be)(body, args; ndrange = nd, workgroupsize = wg)
        end
    else
        if nd isa Tuple
            _sdt_node_fwd_kernel!(be)(body, args, shadows, gpu; ndrange = nd)
        else
            _sdt_walker_fwd_kernel!(be)(body, args, shadows, gpu; ndrange = nd, workgroupsize = wg)
        end
    end
    return nothing
end

# Thread-per-walker reference FFT (bitwise `_eobimr_fft!`).
@inline function _sdt_b_fft_serial!(w, SEG)
    _sdt_fft!(view(SEG.sre, :, w), view(SEG.sim, :, w), Val(_SDT_NS))
    _sdt_fft!(view(SEG.tre, :, w), view(SEG.tim, :, w), Val(_SDT_NS))
    return nothing
end

# In-place radix-2 FFT with the +i sign on vector views, operation for operation `_eobimr_fft!`.
@inline function _sdt_fft!(re, im, ::Val{N}) where {N}
    nbits = trailing_zeros(N)
    @inbounds for i in 0:N-1
        j = bitreverse(i) >>> (8 * sizeof(Int) - nbits)
        if j > i
            re[i + 1], re[j + 1] = re[j + 1], re[i + 1]
            im[i + 1], im[j + 1] = im[j + 1], im[i + 1]
        end
    end
    m = 2
    @inbounds while m <= N
        half = m >>> 1
        wr0, wi0 = cos(2pi / m), sin(2pi / m)
        wr, wi = 1.0, 0.0
        for k in 0:half-1
            for g in 0:m:N-1
                a = g + k + 1
                b = a + half
                tr = wr * re[b] - wi * im[b]
                ti = wr * im[b] + wi * re[b]
                re[b] = re[a] - tr; im[b] = im[a] - ti
                re[a] += tr; im[a] += ti
            end
            wr, wi = wr * wr0 - wi * wi0, wr * wi0 + wi * wr0
        end
        m <<= 1
    end
    return nothing
end

# The same transform with one workgroup of `_SDT_FFT_WG` threads per (walker, segment), in local (shared) memory:
# bit-reversed load, 11 radix-2 stages separated by barriers. Twiddles e^{+2 pi i k/m} are evaluated directly
# (the serial version uses a recurrence), so results agree with `_eobimr_fft!` to round-off (~1e-13), not bitwise.
const _SDT_FFT_WG = 256
const _SDT_NBITS = trailing_zeros(_SDT_NS)
# (KernelAbstractions' CPU backend splits a kernel at each `@synchronize`; nothing defined before a barrier is
# used after it, everything group-uniform is recomputed from the group index.)
@kernel function _sdt_fft_shared_kernel!(SRE, SIM, TRE, TIM)
    t = @index(Local, Linear)
    gi = @index(Group, Linear)
    re = @localmem Float64 (_SDT_NS,)
    im = @localmem Float64 (_SDT_NS,)
    @inbounds for i in (t - 1):_SDT_FFT_WG:(_SDT_NS - 1)
        j = bitreverse(i) >>> (8 * sizeof(Int) - _SDT_NBITS)
        w = (gi + 1) >> 1
        if isodd(gi)
            re[j + 1] = SRE[i + 1, w]; im[j + 1] = SIM[i + 1, w]
        else
            re[j + 1] = TRE[i + 1, w]; im[j + 1] = TIM[i + 1, w]
        end
    end
    @synchronize
    for st in 1:_SDT_NBITS
        @inbounds for q in (t - 1):_SDT_FFT_WG:((_SDT_NS >> 1) - 1)
            m = 1 << st
            half = m >> 1
            k = q % half
            a = (q ÷ half) * m + k + 1
            b = a + half
            ang = 2pi * Float64(k) / Float64(m)
            wr, wi = cos(ang), sin(ang)
            tr = wr * re[b] - wi * im[b]
            ti = wr * im[b] + wi * re[b]
            re[b] = re[a] - tr; im[b] = im[a] - ti
            re[a] += tr; im[a] += ti
        end
        @synchronize
    end
    @inbounds for i in (t - 1):_SDT_FFT_WG:(_SDT_NS - 1)
        w = (gi + 1) >> 1
        if isodd(gi)
            SRE[i + 1, w] = re[i + 1]; SIM[i + 1, w] = im[i + 1]
        else
            TRE[i + 1, w] = re[i + 1]; TIM[i + 1, w] = im[i + 1]
        end
    end
end

# ---------------------------------------------------------------------------------------------- buffers + driver
"""
    SEOBNRDeviceTables(backend, nwalkers)

Device buffers for [`seobnr_device_tables!`](@ref): the PA stage, the dynamics, the build scalars and the
frequency-domain table of `nwalkers` SEOBNRv5 IMR binaries on a KernelAbstractions `backend`. Four more of them
hold the tangents in forward mode (`seobnr_device_tables!(...; TT)`).
"""
struct SEOBNRDeviceTables{BK,A2,A3,VB,VP,VI,VD,VL,VF,VS,VN}
    backend::BK
    nw::Int
    B::VB; PAM::VP; INDOM::VI
    PR::A3; PP::A3; DD::A3; DTR::A2; DPHR::A2; PT::A2; PPH::A2; J0::A2; IC::A2
    D::NamedTuple{(:t, :r, :phi, :pr, :pphi, :dr, :dphi, :dpr, :dpphi, :amp, :damp, :phc, :dphc),NTuple{13,A2}}
    DM::VD; BLD::VL
    TB::NamedTuple{(:ts, :phis, :oms, :Qs, :dQs, :ftf, :fta, :ftda, :ftg, :ftdg),NTuple{10,A2}}
    SEG::NamedTuple{(:sre, :sim, :tre, :tim),NTuple{4,A2}}
    FTM::VF; TABS::VS
    POS::A2
    PSF::A2; PSI::VN; PSB::VI   # host-driven plunge state (`plunge_split`)
end

_sdt_zero(x::Number) = zero(x)
_sdt_zero(x::Union{Tuple,NamedTuple}) = map(_sdt_zero, x)
_sdt_zero(x::T) where {T} = T(ntuple(i -> _sdt_zero(getfield(x, i)), Val(fieldcount(T)))...)   # records' structs (`hp`)

function _sdt_alloc_records(backend, nw)
    m = GravitationalWaveforms.SEOBNRv5IMRAlignedSpinF2()
    b0 = _sdt_zero(GravitationalWaveforms._imr_base(m, 30.0, 0.24, 0.0, 0.0, 20.0, GravitationalWaveforms.GMsun_over_c3))
    nqc0 = (a1 = 0.0, a2 = 0.0, a3 = 0.0, b1 = 0.0, b2 = 0.0, shrink = 0.0)
    mr0 = (nu = 0.0, c1c = 0.0, c1f = 0.0, c2f = 0.0, c2c = 0.0, tau = 0.0, d1f = 0.0, d2f = 0.0, d1c = 0.0,
        omR = 0.0, phi_a = 0.0, w_a = 0.0, A_a = 0.0, Mf = 0.0, af = 0.0)
    arr(x) = (a = KernelAbstractions.allocate(backend, typeof(x), nw); fill!(a, x); a)
    return (B = arr(b0), PAM = arr((pa_success = false, nreal = 10, r0g = 0.0, dr = 0.0)), INDOM = arr(false),
        DM = arr(_sdt_dmeta(2, 1, false)),
        BLD = arr(_sdt_bld(0.0, 0.0, false, nqc0, mr0, 0.0, false, 1, 0.0, 2, 0.0, 0.0, 0.0, 0.0, 0.0)),
        FTM = arr(_sdt_ftm(0.0, 0.0, 0.0, 0.0, 1, 0.0, false)),
        TABS = arr(_sdt_tabs(false, 0.0, 0.0, 0.0, 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
end

function SEOBNRDeviceTables(backend, nw::Int)
    z(dims...) = KernelAbstractions.zeros(backend, Float64, dims...)
    R = _sdt_alloc_records(backend, nw)
    D = NamedTuple{(:t, :r, :phi, :pr, :pphi, :dr, :dphi, :dpr, :dpphi, :amp, :damp, :phc, :dphc)}(
        ntuple(_ -> z(_SDT_ND, nw), 13))
    TB = (ts = z(_SDT_NT, nw), phis = z(_SDT_NT, nw), oms = z(_SDT_NT, nw), Qs = z(_SDT_NT, nw), dQs = z(_SDT_NT, nw),
        ftf = z(_SDT_NF, nw), fta = z(_SDT_NF, nw), ftda = z(_SDT_NF, nw), ftg = z(_SDT_NF, nw), ftdg = z(_SDT_NF, nw))
    SEG = (sre = z(_SDT_NS, nw), sim = z(_SDT_NS, nw), tre = z(_SDT_NS, nw), tim = z(_SDT_NS, nw))
    return SEOBNRDeviceTables(backend, nw, R.B, R.PAM, R.INDOM,
        z(_SDT_NPA, _SDT_NORD + 1, nw), z(_SDT_NPA, _SDT_NORD + 1, nw), z(_SDT_NPA, _SDT_NORD, nw),
        z(_SDT_NPA, nw), z(_SDT_NPA, nw), z(_SDT_NPA, nw), z(_SDT_NPA, nw), z(_SDT_NPA, nw), z(3, nw),
        D, R.DM, R.BLD, TB, SEG, R.FTM, R.TABS, z(4, nw), z(_PS_NROW, nw), KernelAbstractions.zeros(backend, Int, nw),
        (b = KernelAbstractions.allocate(backend, Bool, nw); fill!(b, false); b))
end

"""
    seobnr_device_tables!(T::SEOBNRDeviceTables, pos; f_min = 20.0, timers = nothing, serial_fft = false,
        walker_wg = GPU.tuning(:SDT_WG, 32), TT = nothing, dirpar = (base = true, plunge = nothing, nodes = false), base_ift = true, plunge_split = false)

Build the SEOBNRv5 IMR frequency-domain tables of all walkers on the device. `pos` is a 4 x nwalkers array (host
or device) of (chirp mass, eta, chi1, chi2), or `nothing` if `T.POS` is already filled. Afterwards `T.TB`
(node, walker) arrays and `T.TABS` (per-walker scalars) hold, walker by walker, the fields of
`GravitationalWaveforms._eobimr_coeffs`'s `EOBIMRTable`.

`TT`: four more `SEOBNRDeviceTables` whose `POS` hold tangent directions of the inputs; every stage then runs in
Enzyme forward mode and `TT[k]` receives the tangents of every buffer (direction `k`).
`dirpar`: which stages run their forward mode with width-1 tangents on one thread per direction (`base`, `plunge`:
the serial walker stages; `nodes`: every node / sample / bin stage) instead of one width-4 thread. `plunge = 2`: two width-2 threads per walker. `plunge = nothing`:
direction-parallel below 2048 walkers only (`GRAVITATIONALWAVEBAYES_SDT_PLUNGE_DIRPAR_MAX`; V100: plunge forward 159 vs 221 ms at 1024 walkers, but 464 vs
342 ms at 4096, where 16k threads with 4x the primal work are throughput-bound).
`plunge_split`: in forward mode, the RK4 plunge as host-driven per-stage kernels (state in global memory) instead of
one serial kernel.
`base_ift`: in forward mode, the base coefficients by implicit-function steps at the converged initial-condition
roots (primal pass first) instead of forward mode through the Newton loops.
`build_only`: stop after the dynamics, node amplitudes and the attachment / NQC / merger-ringdown build (`T.D`, `T.DM`,
`T.BLD`), the input of the SEOBNRv5HM tables (`SEOBNRHMDeviceTable.jl`).
`timers`, if a `Dict{Symbol,Float64}`, accumulates per-stage seconds (synchronizing). `serial_fft = true` uses the
one-thread-per-walker FFT (bitwise `_eobimr_fft!`) instead of the workgroup FFT; `walker_wg` is the workgroup size
of the serial plunge stage.
"""
function seobnr_device_tables!(T::SEOBNRDeviceTables, pos; f_min = 20.0, timers = nothing, serial_fft = false,
        walker_wg = GPU.tuning(:SDT_WG, 32), TT = nothing, dirpar = (base = true, plunge = nothing, nodes = false), base_ift = true, plunge_split = false,
        build_only = false)
    be = T.backend; nw = T.nw
    m = GravitationalWaveforms.SEOBNRv5IMRAlignedSpinF2()
    pos === nothing || copyto!(T.POS, pos)
    tg(f) = ntuple(k -> f(TT[k]), 4)
    sh(xs...) = TT === nothing ? nothing : xs
    t0 = Ref(time_ns())
    lap!(k) = timers === nothing || (KernelAbstractions.synchronize(be); t = time_ns();
        timers[k] = get(timers, k, 0.0) + (t - t0[]) / 1e9; t0[] = t)
    fwd = TT !== nothing
    B = fwd ? tg(t -> t.B) : nothing
    PAM = fwd ? tg(t -> t.PAM) : nothing
    D = fwd ? tg(t -> t.D) : nothing
    BLD = fwd ? tg(t -> t.BLD) : nothing
    TB = fwd ? tg(t -> t.TB) : nothing
    SEG = fwd ? tg(t -> t.SEG) : nothing
    FTM = fwd ? tg(t -> t.FTM) : nothing
    GM3 = GravitationalWaveforms.GMsun_over_c3
    if fwd && base_ift
        _sdt_run!(be, _sdt_b_base_ic!, nw, (T.IC, T.POS, m, Float64(f_min), GM3), nothing)
        _sdt_run!(be, _sdt_b_base_step!, nw, (T.B, T.PAM, T.INDOM, T.POS, T.IC, m, Float64(f_min), GM3),
            sh(B, PAM, nothing, tg(t -> t.POS), nothing, nothing, nothing, nothing); dirpar = dirpar.base)
    else
        _sdt_run!(be, _sdt_b_base!, nw, (T.B, T.PAM, T.INDOM, T.POS, m, Float64(f_min), GM3),
            sh(B, PAM, nothing, fwd ? tg(t -> t.POS) : nothing, nothing, nothing, nothing); dirpar = dirpar.base)
    end
    # PA lanes: only the largest real grid matters (nreal <= NMAX); one small device -> host read.
    KernelAbstractions.synchronize(be)
    nl = maximum(p -> p.nreal, Array(T.PAM))
    lap!(:base)
    PR = fwd ? tg(t -> t.PR) : nothing; PP = fwd ? tg(t -> t.PP) : nothing
    if fwd
        _sdt_run!(be, _sdt_b_j0_newton!, (nl, nw), (T.J0, T.B, T.PAM), nothing)
        _sdt_run!(be, _sdt_b_j0_step!, (nl, nw), (T.PR, T.PP, T.J0, T.B, T.PAM), sh(PR, PP, nothing, B, PAM); dirpar = dirpar.nodes)
    else
        _sdt_run!(be, _sdt_b_j0!, (nl, nw), (T.PR, T.PP, T.B, T.PAM), nothing)
    end
    for n in 1:_SDT_NORD
        _sdt_run!(be, _sdt_b_order!, (nl, nw), (T.PR, T.PP, T.DD, T.B, T.PAM, n),
            sh(PR, PP, fwd ? tg(t -> t.DD) : nothing, B, PAM, nothing); dirpar = dirpar.nodes)
    end
    DTR = fwd ? tg(t -> t.DTR) : nothing; DPHR = fwd ? tg(t -> t.DPHR) : nothing
    _sdt_run!(be, _sdt_b_tphi!, (nl, nw), (T.DTR, T.DPHR, T.PR, T.PP, T.B, T.PAM), sh(DTR, DPHR, PR, PP, B, PAM); dirpar = dirpar.nodes)
    PT = fwd ? tg(t -> t.PT) : nothing; PPH = fwd ? tg(t -> t.PPH) : nothing
    _sdt_run!(be, _sdt_b_pa_integrate!, nw, (T.PT, T.PPH, T.DTR, T.DPHR, T.PAM), sh(PT, PPH, DTR, DPHR, PAM))
    lap!(:pa)
    _sdt_run!(be, _sdt_b_dyn_init!, (nl, nw), (T.D, T.PT, T.PPH, T.PR, T.PP, T.B, T.PAM, m),
        sh(D, PT, PPH, PR, PP, B, PAM, nothing); dirpar = dirpar.nodes)
    if fwd && plunge_split
        PSF = tg(t -> t.PSF)
        _sdt_run!(be, _sdt_ps_init!, nw, (T.PSF, T.PSI, T.PSB, T.D, T.PAM, m), sh(PSF, nothing, nothing, D, PAM, nothing); wg = walker_wg)
        for it in 1:_SDT_NRK
            _sdt_run!(be, _sdt_ps_k2!, nw, (T.PSF, T.PSB, T.B, m), sh(PSF, nothing, B, nothing); wg = walker_wg)
            _sdt_run!(be, _sdt_ps_k3!, nw, (T.PSF, T.PSB, T.B, m), sh(PSF, nothing, B, nothing); wg = walker_wg)
            _sdt_run!(be, _sdt_ps_k4!, nw, (T.PSF, T.PSB, T.B, m), sh(PSF, nothing, B, nothing); wg = walker_wg)
            _sdt_run!(be, _sdt_ps_accept!, nw, (T.PSF, T.PSI, T.PSB, T.D, T.B, m),
                sh(PSF, nothing, nothing, D, B, nothing); wg = walker_wg)
            # every 16 steps: stop once every walker has stopped (one small device -> host read)
            (it % 16 == 0 && all(Array(T.PSB))) && break
        end
        _sdt_run!(be, _sdt_ps_finish!, nw, (T.DM, T.PSI, T.PSB, T.PAM), nothing; wg = walker_wg)
    else
        _sdt_run!(be, _sdt_b_plunge!, nw, (T.DM, T.D, T.B, T.PAM, m), sh(nothing, D, B, PAM, nothing); wg = walker_wg,
            dirpar = something(dirpar.plunge, nw < GPU.tuning(:SDT_PLUNGE_DIRPAR_MAX, 2048)))
    end
    lap!(:plunge)
    _sdt_run!(be, _sdt_b_amp!, (_SDT_ND, nw), (T.D, T.DM, T.B, m), sh(D, nothing, B, nothing); dirpar = dirpar.nodes)
    _sdt_run!(be, _sdt_b_damp!, (_SDT_ND, nw), (T.D, T.DM), sh(D, nothing); dirpar = dirpar.nodes)
    lap!(:nodes)
    _sdt_run!(be, _sdt_b_build!, nw, (T.BLD, T.DM, T.B, T.INDOM, T.D), sh(BLD, nothing, B, nothing, D))
    lap!(:build)
    build_only && (KernelAbstractions.synchronize(be); return T)
    _sdt_run!(be, _sdt_b_spa!, (_SDT_NT, nw), (T.TB, T.BLD, T.DM, T.B, T.D), sh(TB, BLD, nothing, B, D); dirpar = dirpar.nodes)
    _sdt_run!(be, _sdt_b_spa_finish!, (_SDT_NT, nw), (T.TB, T.BLD), sh(TB, BLD); dirpar = dirpar.nodes)
    lap!(:spa)
    _sdt_run!(be, _sdt_b_segment!, (_SDT_NS, nw), (T.SEG, T.BLD, T.DM, T.D), sh(SEG, BLD, nothing, D); dirpar = dirpar.nodes)
    lap!(:segment)
    # The FFT is linear: tangents are transformed exactly like the primal.
    for S in (fwd ? (T.SEG, SEG...) : (T.SEG,))
        if serial_fft
            _sdt_run!(be, _sdt_b_fft_serial!, nw, (S,), nothing)
        else
            _sdt_fft_shared_kernel!(be, _SDT_FFT_WG)(S.sre, S.sim, S.tre, S.tim; ndrange = _SDT_FFT_WG * 2 * nw)
        end
    end
    lap!(:fft)
    _sdt_run!(be, _sdt_b_ftmeta!, nw, (T.FTM, T.TB, T.BLD, T.DM, T.B, T.D), sh(FTM, TB, BLD, nothing, B, D))
    _sdt_run!(be, _sdt_b_ftbin!, (_SDT_NF, nw), (T.TB, T.SEG, T.FTM, T.BLD, T.B), sh(TB, SEG, FTM, BLD, B); dirpar = dirpar.nodes)
    _sdt_run!(be, _sdt_b_finalize!, nw, (T.TABS, T.TB, T.FTM, T.BLD, T.B),
        sh(fwd ? tg(t -> t.TABS) : nothing, TB, FTM, BLD, B))
    lap!(:fourier)
    KernelAbstractions.synchronize(be)
    return T
end

"""
    seobnr_device_table_host(T::SEOBNRDeviceTables, w) -> GravitationalWaveforms.EOBIMRTable

Walker `w`'s device table copied back as the CPU `EOBIMRTable` (for validation).
"""
function seobnr_device_table_host(T::SEOBNRDeviceTables, w::Int)
    s = Array(T.TABS)[w]
    col(a) = SVector{size(a, 1),Float64}(Array(view(a, :, w)))
    TB = T.TB
    return GravitationalWaveforms.EOBIMRTable{_SDT_NT,_SDT_NF}(s.success, s.M_sec, s.eta, s.M, s.nreal, col(TB.ts), col(TB.phis),
        col(TB.oms), col(TB.Qs), col(TB.dQs), s.fmin, s.fmax, s.fspa_max, s.f_b1, s.f_b2, s.t_a, s.ft_shift,
        col(TB.ftf), col(TB.fta), col(TB.ftda), col(TB.ftg), col(TB.ftdg))
end
