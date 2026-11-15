# ------------------------------------------------------------------------------------------------
# SEOBNRv5PHM log-likelihood on a KernelAbstractions device (primal). The CPU
# likelihood's waveform (`GravitationalWaveforms.hphc(::SEOBNRv5PHM, ...)`, the FFT of the time-domain model) in stages:
#   (1) walker: physical rows -> model inputs (mass ordering, units, sampling step in M);
#   (2) PN spin table + dynamics (`_seobv5phm_pn_fill!`, the pieces of `_seobv5phm_dynamics!`) into global slices:
#       walker (table, start), node x walker (PA orders, with walker spline slopes in between), walker (plunge);
#   (3) walker: the build (`_seobv5phm_build_dyn!`: attachment, modes, NQC, remnant, frames, anti-symmetric fits)
#       into node tables + an isbits record (`_seobv5phm_rec`);
#   (4) sample x walker: h+ - i hx (with the turn-on) on a uniform time grid of N samples, N dt = 1 / df;
#   (5) batched in-place FFT along the samples (AbstractFFTs plan: FFTW on the CPU, CUFFT on CUDA);
#   (6) walker: per detector the sums of `_xprec_det_sums(::SEOBNRv5PHM, ...)` from the FFT bins and
#       `_marg_combine`, as `_xprec_ll_given_physical`.
# N is a power of two with dt <= 1 / (f_max + 0.3 / M_min) (`GravitationalWaveforms._seobv5phm_nfft`, the CPU model's rule at the
# batch's lightest mass): the signal (|f| <~ 0.25/M, the ringdown of the (5,5) mode) then aliases only beyond the
# data band, so the device and CPU values agree to the aliasing level (pass the CPU's `nfft` to compare exactly). Walkers whose waveform does not fit in 1/df or
# fails get `_SEOBV5F2_DEADZONE_LOGLIKELIHOOD`.
# ------------------------------------------------------------------------------------------------

const _PHM_NPN = GravitationalWaveforms._SEOBV5PHM_PN_N
const _PHM_NPA = GravitationalWaveforms._SEOBV5PHM_NPA
const _PHM_NRK = GravitationalWaveforms._SEOBV5PHM_NRK
const _PHM_ND = _PHM_NPA + _PHM_NRK
const _PHM_DM0 = (x1 = 0.0, h = 0.0, npn = 0, oe = 0.0, n = 0, npa = 0, pa_success = false, r_final = 0.0, r0 = 0.0,
    pphi0 = 0.0, pr0 = 0.0, stop = 0)
# model inputs per walker (rows of `P`): m1 / M, chi1 (3), chi2 (3), omega_ref, omega_start, inclination, model
# phiref, dt (M), M (s), K = M GMsun/c^2/dL M_s; `TP`: t_peak (s), ok, t*(f_ref) (s, `_phm_tref_body!`)
const _PHM_NP = 14

# The FFT length for the frequency grid `freqs` (spacing df) and the lightest total mass `mtot_min` (Msun).
_phm_nfft(freqs, mtot_min) = GravitationalWaveforms._seobv5phm_nfft(freqs[2] - freqs[1], maximum(freqs), mtot_min * GMsun_over_c3)

struct _PHMWork{BK,A2,A3,A4,VD,VR,VP,VQ,V1,X,PL}
    backend::BK
    nw::Int
    N::Int
    P::A2
    TP::A2
    Z::A3
    DZ::A3
    BUF::A3
    SCR::A3
    ZC::A3                            # PN carry steps' start states (9 x NPN)
    PAS::A4                           # PA states per order (NPA x 3 x 9)
    PAD::A3                           # PA spline slopes (NPA x 10)
    PAY::A3                           # dt/dr, dphi/dr at the PA nodes (NPA x 2)
    DM::VD
    MD::A4
    GT::A2
    REC::VR
    RPRE::VP                          # `_seobv5phm_build_pre` records
    QREC::VQ                          # `_seobv5phm_post_nqc` records
    TM::V1                            # t_peak search results (the Newton step's start)
    X::X
    plan::PL
end

_phm_rec_proto() = GravitationalWaveforms._seobv5phm_rec(GravitationalWaveforms._seobv5phm_build(0.6, 0.4, [0.3, 0.1, 0.2], [0.1, -0.2, 0.3],
    0.018, 0.017))
function _phm_pre_proto()
    d = GravitationalWaveforms.seobnrv5phm_dynamics(0.6, 0.4, [0.3, 0.1, 0.2], [0.1, -0.2, 0.3], 0.017, 0.017)
    R = GravitationalWaveforms._seobv5phm_build_pre(d, d.b, 0.2, 0.3)
    return R, GravitationalWaveforms._seobv5phm_post_nqc(d.b, R)
end

function _PHMWork(backend, nw::Int, N::Int)
    z(dims...) = (A = KernelAbstractions.allocate(backend, Float64, dims...); fill!(A, 0.0); A)
    DM = KernelAbstractions.allocate(backend, typeof(_PHM_DM0), nw); fill!(DM, _PHM_DM0)
    rec0 = _phm_rec_proto()
    REC = KernelAbstractions.allocate(backend, typeof(rec0), nw); fill!(REC, rec0)
    pre0, q0 = _phm_pre_proto()
    RPRE = KernelAbstractions.allocate(backend, typeof(pre0), nw); fill!(RPRE, pre0)
    QREC = KernelAbstractions.allocate(backend, typeof(q0), nw); fill!(QREC, q0)
    X = KernelAbstractions.allocate(backend, ComplexF64, N, nw); fill!(X, zero(ComplexF64))
    plan = FFTW.plan_fft!(X, 1)
    return _PHMWork(backend, nw, N, z(_PHM_NP, nw), z(3, nw), z(9, _PHM_NPN, nw), z(9, _PHM_NPN, nw), z(_PHM_ND, 11, nw),
        z(_PHM_NPA, 4, nw), z(9, _PHM_NPN, nw), z(_PHM_NPA, 3, 9, nw), z(_PHM_NPA, 10, nw), z(_PHM_NPA, 2, nw), DM,
        z(_PHM_ND, 6, 4, nw), z(_PHM_NPN, nw), REC, RPRE, QREC, z(nw), X, plan)
end
const _PHM_WORK = Ref{Any}(nothing)
function _phm_work(backend, nw::Int, N::Int)
    w = _PHM_WORK[]
    # never shrink N: reuse a larger FFT length (values change only at the aliasing level, as between
    # batches anyway); reallocate only when N grows -- each reallocation is a memory peak
    (w isa _PHMWork && typeof(w.backend) == typeof(backend) && w.nw == nw && w.N >= N) && return w
    # a full collection before reallocating: the previous buffers (several GB at 1024 walkers) must go back to the device
    # pool first -- with only a minor GC both generations coexisted and a 16 GB V100 ran out when N grew
    _PHM_WORK[] = nothing
    GC.gc(true); GPU.reclaim()              # the old buffers back to the driver before the new ones (see `GPU.reclaim`)
    w = _PHMWork(backend, nw, N)
    _PHM_WORK[] = w
    return w
end

@inline _phm_bufs(BUF, SCR, w) = (t = view(BUF, :, 1, w), r = view(BUF, :, 2, w), phi = view(BUF, :, 3, w),
    pr = view(BUF, :, 4, w), pphi = view(BUF, :, 5, w), dr = view(BUF, :, 6, w), dphi = view(BUF, :, 7, w),
    dpr = view(BUF, :, 8, w), dpphi = view(BUF, :, 9, w), omega_s = view(BUF, :, 10, w), omega_circ = view(BUF, :, 11, w),
    s1 = view(SCR, :, 1, w), s2 = view(SCR, :, 2, w), s3 = view(SCR, :, 3, w), s4 = view(SCR, :, 4, w))
@inline function _phm_base(P, Z, DZ, dm, w)
    m1 = P[1, w]
    tab = GravitationalWaveforms._seobv5phm_pn_view(view(Z, :, :, w), view(DZ, :, :, w), dm.x1, dm.h, dm.npn, dm.oe, m1, 1 - m1, 3.0)
    return GravitationalWaveforms._seobv5phm_base_tab(m1, 1 - m1, tab, P[8, w], P[9, w])
end
@inline _phm_dyn(BUF, SCR, dm, b, w) = merge(_phm_bufs(BUF, SCR, w), (n = dm.n, npa = dm.npa, pa_success = dm.pa_success,
    r_final = dm.r_final, r0 = dm.r0, pphi0 = dm.pphi0, pr0 = dm.pr0, stop = dm.stop, b = b))
@inline _phm_modes(MD, w) = (A = view(MD, :, :, 1, w), dA = view(MD, :, :, 2, w), th = view(MD, :, :, 3, w),
    dth = view(MD, :, :, 4, w))

# (1) physical rows (m1, m2, chi1z, chi2z, dL, t_c, phase, ra, dec, incl, psi, chi1x, chi1y, chi2x, chi2y) -> inputs;
# waveform from 0.9 f_ref, phase and frame at f_ref (as `_xprec_state(::SEOBNRv5PHM, ...)`), `hphc`'s conventions.
# The 14 model inputs from walker w's 15 physical rows (a pure function: its reverse gives the rows' cotangents).
# `fphi` > fref (`PEConfig.phase_ref_frequency`): phase, spins and frame at fphi instead (M omega capped at 0.06, inside
# most inspirals of the prior; a walker whose dynamics end below it is shifted at its last node, `_phm_phase_shift!`).
@inline function _phm_prep_vals(m1, m2, chi1z, chi2z, dL, phase, incl, c1x, c1y, c2x, c2y, fref, dt_s, fphi)
    c1 = (c1x, c1y, chi1z); c2 = (c2x, c2y, chi2z)
    if m1 < m2
        m1, m2, c1, c2 = m2, m1, c2, c1
        phase += pi
    end
    M = m1 + m2
    Ms = M * GMsun_over_c3
    om_ref = ifelse(fphi > fref, min(pi * fphi * Ms, 0.06), pi * fref * Ms)
    # start (as `_seobv5phm_build`): 0.9 f_ref, at most at r = 10.5 M, never above omega_ref (then no shift)
    om_s = pi * 0.9 * fref * Ms
    om_s = ifelse(om_s^(-2 / 3) < 10.5, 10.5^(-1.5), om_s)
    return (m1 / M, c1[1], c1[2], c1[3], c2[1], c2[2], c2[3], om_ref, min(om_s, om_ref), incl, pi / 2 - phase, dt_s / Ms, Ms,
        M * GMsun_over_c2_Gpc / dL * Ms)
end
@inline _phm_prep_args(params, w) = @inbounds (params[1, w], params[2, w], params[3, w], params[4, w], params[5, w],
    params[7, w], params[10, w], params[12, w], params[13, w], params[14, w], params[15, w])
@inline function _phm_prep_body!(w, P, params, fref, dt_s, fphi)
    v = _phm_prep_vals(_phm_prep_args(params, w)..., fref, dt_s, fphi)
    @inbounds for k in 1:_PHM_NP
        P[k, w] = v[k]
    end
    return nothing
end

# (2)
@inline function _phm_dyn_body!(w, P, Z, DZ, BUF, SCR, DM)
    @inbounds begin
        m1 = P[1, w]
        c1 = SVector(P[2, w], P[3, w], P[4, w]); c2 = SVector(P[5, w], P[6, w], P[7, w])
        x1, h, n, oe = GravitationalWaveforms._seobv5phm_pn_fill!(view(Z, :, :, w), view(DZ, :, :, w), m1, 1 - m1, c1, c2, P[8, w],
            P[9, w], _PHM_NPN, 3.0)
        dm0 = (x1 = x1, h = h, npn = n, oe = oe, n = 0, npa = 0, pa_success = false, r_final = 0.0, r0 = 0.0, pphi0 = 0.0,
            pr0 = 0.0, stop = 0)
        b = _phm_base(P, Z, DZ, dm0, w)
        sc = GravitationalWaveforms._seobv5phm_dynamics!(_phm_bufs(BUF, SCR, w), b, P[9, w]; NPA = _PHM_NPA, NRK = _PHM_NRK)
        DM[w] = (x1 = x1, h = h, npn = n, oe = oe, n = sc.n, npa = sc.npa, pa_success = sc.pa_success,
            r_final = sc.r_final, r0 = sc.r0, pphi0 = sc.pphi0, pr0 = sc.pr0, stop = sc.stop)
    end
    return nothing
end

# (2) staged (the device path; `_phm_dyn_body!` above is the same in one walker thread): the PN table and the start
# (walker), the PA solution order by order (walker: spline slopes; node x walker: the order's update, every order's
# state kept in PAS for the reverse), the PA nodes' t, phi integrals and rates, the plunge (walker).
# Arrays: ZC (9 x NPN) the PN carry steps' start states; PAS (NPA x (pr, pphi, om) x order 0..8); PAD (NPA x 10) the
# slopes of each order's spline (1:8), then of dt/dr and dphi/dr (9, 10); PAY (NPA x 2) dt/dr, dphi/dr.
@inline function _phm_dstart_body!(w, P, Z, DZ, ZC, BUF, DM)
    @inbounds begin
        m1 = P[1, w]
        c1 = SVector(P[2, w], P[3, w], P[4, w]); c2 = SVector(P[5, w], P[6, w], P[7, w])
        x1, h, n, oe = GravitationalWaveforms._seobv5phm_pn_fill!(view(Z, :, :, w), view(DZ, :, :, w), m1, 1 - m1, c1, c2, P[8, w],
            P[9, w], _PHM_NPN, 3.0; ZC = view(ZC, :, :, w))
        dm = (x1 = x1, h = h, npn = n, oe = oe, n = 0, npa = 0, pa_success = false, r_final = 0.0, r0 = 0.0, pphi0 = 0.0,
            pr0 = 0.0, stop = 0)
        b = _phm_base(P, Z, DZ, dm, w)
        for c in 1:11, i in 1:_PHM_ND
            BUF[i, c, w] = 0.0
        end
        r0, pphi0, pr0, succ, npa, r_final = GravitationalWaveforms._seobv5phm_dyn_start(b, P[9, w], _PHM_NPA)
        if succ
            hgrid = (r_final - r0) / (npa - 1)
            for i in 1:npa
                BUF[i, 2, w] = r0 + (i - 1) * hgrid
            end
        else
            npa = 1
            BUF[1, 2, w] = r0; BUF[1, 4, w] = pr0; BUF[1, 5, w] = pphi0; BUF[1, 10, w] = P[9, w]
            BUF[1, 6, w], BUF[1, 7, w], BUF[1, 8, w], BUF[1, 9, w], BUF[1, 11, w] =
                GravitationalWaveforms._seobv5phm_rhs(r0, pr0, pphi0, GravitationalWaveforms._seobv5phm_spins(b, P[9, w]), b)
        end
        DM[w] = merge(dm, (npa = npa, pa_success = succ, r_final = r_final, r0 = r0, pphi0 = pphi0, pr0 = pr0))
    end
    return nothing
end
@inline function _phm_padiab_body!(i, w, P, Z, DZ, BUF, DM, PAS)
    @inbounds begin
        dm = DM[w]
        dm.pa_success & (i <= dm.npa) || return nothing
        pphi, om = GravitationalWaveforms._seobv5phm_pa_adiabatic(BUF[i, 2, w], _phm_base(P, Z, DZ, dm, w))
        PAS[i, 1, 1, w] = 0.0; PAS[i, 2, 1, w] = pphi; PAS[i, 3, 1, w] = om
    end
    return nothing
end
@inline function _phm_paslope_body!(w, k, BUF, SCR, DM, PAS, PAD)
    @inbounds begin
        dm = DM[w]
        dm.pa_success || return nothing
        GravitationalWaveforms._seobv5phm_spline_slopes!(view(PAD, :, k, w), view(PAS, :, isodd(k) ? 2 : 1, k, w),
            BUF[2, 2, w] - BUF[1, 2, w], dm.npa, view(SCR, :, 1, w))
    end
    return nothing
end
@inline function _phm_paupd_body!(i, w, k, P, Z, DZ, BUF, DM, PAS, PAD)
    @inbounds begin
        dm = DM[w]
        dm.pa_success & (i <= dm.npa) || return nothing
        pr, pphi, om = GravitationalWaveforms._seobv5phm_pa_update(k, BUF[i, 2, w], PAS[i, 1, k, w], PAS[i, 2, k, w], PAS[i, 3, k, w],
            PAD[i, k, w], _phm_base(P, Z, DZ, dm, w))
        PAS[i, 1, k+1, w] = pr; PAS[i, 2, k+1, w] = pphi; PAS[i, 3, k+1, w] = om
    end
    return nothing
end
# the final PA state into the node buffer, dt/dr, dphi/dr and the node rates
@inline function _phm_panode_body!(i, w, P, Z, DZ, BUF, DM, PAS, PAY)
    @inbounds begin
        dm = DM[w]
        dm.pa_success & (i <= dm.npa) || return nothing
        b = _phm_base(P, Z, DZ, dm, w)
        r, pr, pp, om = BUF[i, 2, w], PAS[i, 1, 9, w], PAS[i, 2, 9, w], PAS[i, 3, 9, w]
        BUF[i, 4, w] = pr; BUF[i, 5, w] = pp; BUF[i, 10, w] = om
        PAY[i, 1, w], PAY[i, 2, w] = GravitationalWaveforms._seobv5phm_pa_dtdr(r, pr, pp, om, b)
        BUF[i, 6, w], BUF[i, 7, w], BUF[i, 8, w], BUF[i, 9, w], BUF[i, 11, w] =
            GravitationalWaveforms._seobv5phm_rhs(r, pr, pp, GravitationalWaveforms._seobv5phm_spins(b, om), b)
    end
    return nothing
end
@inline function _phm_paint_body!(w, BUF, SCR, DM, PAD, PAY)
    @inbounds begin
        dm = DM[w]
        dm.pa_success || return nothing
        h = BUF[2, 2, w] - BUF[1, 2, w]
        GravitationalWaveforms._seobv5phm_spline_slopes!(view(PAD, :, 9, w), view(PAY, :, 1, w), h, dm.npa, view(SCR, :, 1, w))
        GravitationalWaveforms._seobv5phm_spline_cumint!(view(BUF, :, 1, w), view(PAY, :, 1, w), view(PAD, :, 9, w), h, dm.npa)
        GravitationalWaveforms._seobv5phm_spline_slopes!(view(PAD, :, 10, w), view(PAY, :, 2, w), h, dm.npa, view(SCR, :, 1, w))
        GravitationalWaveforms._seobv5phm_spline_cumint!(view(BUF, :, 3, w), view(PAY, :, 2, w), view(PAD, :, 10, w), h, dm.npa)
    end
    return nothing
end
@inline function _phm_plunge_body!(w, P, Z, DZ, BUF, SCR, DM)
    @inbounds begin
        dm = DM[w]
        b = _phm_base(P, Z, DZ, dm, w)
        npa = dm.npa
        n, stop = GravitationalWaveforms._seobv5phm_plunge!(_phm_bufs(BUF, SCR, w), b, npa, dm.pa_success ? BUF[npa, 7, w] : P[9, w],
            _PHM_NRK, GravitationalWaveforms._SEOBV5PHM_C_OM, GravitationalWaveforms._SEOBV5PHM_C_R)
        for i in n+1:_PHM_ND
            BUF[i, 1, w] = BUF[n, 1, w] + (i - n)
        end
        DM[w] = merge(dm, (n = n, stop = stop))
    end
    return nothing
end

# (3) the build in the pieces of `_seobv5phm_build_dyn!`: walker (phase shift, attachment), node x walker (node modes,
# then their derivatives), walker (NQC .. anti-symmetric fits, t_peak).
@inline _phm_use_tref(P, w) = @inbounds abs(P[8, w] - P[9, w]) > 1e-12 * P[9, w]
# `GravitationalWaveforms._seobv5phm_phase_shift!` without its DomainError: with `phase_ref_frequency` a walker whose dynamics end
# before omega_ref (a failed / truncated evolution) is shifted at its last node (`_seobv5phm_t_at_omega` returns the last
# time), as the reverse (`_phm_refph_dot`, the same `_seobv5phm_ref_phase`) assumes; it threw inside the kernel (GW150914
# flow run, stage 4) and killed the run.
@inline function _phm_phase_shift!(d, omega_ref)
    ph_ref = GravitationalWaveforms._seobv5phm_ref_phase(d, omega_ref)
    @inbounds for i in 1:d.n
        d.phi[i] -= ph_ref
    end
    return nothing
end
@inline function _phm_bpre_body!(w, P, Z, DZ, BUF, SCR, DM, RPRE)
    @inbounds begin
        dm = DM[w]
        b = _phm_base(P, Z, DZ, dm, w)
        d = _phm_dyn(BUF, SCR, dm, b, w)
        _phm_use_tref(P, w) && _phm_phase_shift!(d, P[8, w])
        RPRE[w] = GravitationalWaveforms._seobv5phm_build_pre(d, b, P[4, w], P[7, w])
    end
    return nothing
end
@inline function _phm_bnode_body!(i, w, P, Z, DZ, BUF, SCR, DM, MD, RPRE)
    @inbounds begin
        dm = DM[w]
        i <= dm.n || return nothing
        b = _phm_base(P, Z, DZ, dm, w)
        GravitationalWaveforms._seobv5phm_node_modes!(_phm_modes(MD, w), _phm_dyn(BUF, SCR, dm, b, w), b, RPRE[w], i)
    end
    return nothing
end
@inline function _phm_bderiv_body!(i, w, P, Z, DZ, BUF, SCR, DM, MD)
    @inbounds begin
        dm = DM[w]
        i <= dm.n || return nothing
        b = _phm_base(P, Z, DZ, dm, w)
        GravitationalWaveforms._seobv5phm_node_derivs!(_phm_modes(MD, w), _phm_dyn(BUF, SCR, dm, b, w), i)
    end
    return nothing
end
@inline function _phm_bpost_body!(w, P, Z, DZ, BUF, SCR, DM, MD, GT, RPRE, QREC, REC, TP, TM, N)
    @inbounds begin
        dm = DM[w]
        b = _phm_base(P, Z, DZ, dm, w)
        d = _phm_dyn(BUF, SCR, dm, b, w)
        R = RPRE[w]
        # `_seobv5phm_build_post` with its NQC record kept (the reverse runs its pieces separately)
        Q = GravitationalWaveforms._seobv5phm_post_nqc(b, R)
        QREC[w] = Q
        rec0 = GravitationalWaveforms._seobv5phm_post_rest!(view(GT, :, w), d, b, R, Q, _phm_use_tref(P, w), P[8, w])
        S = merge((dyn = d, D = _phm_modes(MD, w), b = b, G = view(GT, :, w)), rec0)
        S = merge(S, (asym = GravitationalWaveforms._seobv5phm_asym_build(S, R.sts), use_asym = true))
        tm = GravitationalWaveforms._seobv5phm_tpeak_search(S)
        TM[w] = tm
        S = merge(S, (t_peak = GravitationalWaveforms._seobv5phm_tpeak_newton(S, tm),))
        REC[w] = GravitationalWaveforms._seobv5phm_rec(S)
        t_end = S.t_a + GravitationalWaveforms._EOBIMR_NTAU * S.mr[1].tau
        TP[1, w] = S.t_peak * P[13, w]
        TP[2, w] = ifelse(isfinite(S.t_peak) & isfinite(t_end) & (t_end <= (N - 1) * P[12, w]), 1.0, 0.0)
    end
    return nothing
end

# (4) sample j of walker w: h+ - i hx at t = (j - 1) dt (M) with the turn-on over the first four (2,2) cycles.
@inline function _phm_sample_body!(j, w, P, Z, DZ, BUF, SCR, DM, MD, GT, REC, TP, X)
    @inbounds begin
        h = zero(ComplexF64)
        if TP[2, w] > 0
            dm = DM[w]
            b = _phm_base(P, Z, DZ, dm, w)
            S = GravitationalWaveforms._seobv5phm_from_rec(REC[w], _phm_dyn(BUF, SCR, dm, b, w), _phm_modes(MD, w), b, view(GT, :, w))
            t = (j - 1) * P[12, w]
            t_end = S.t_a + GravitationalWaveforms._EOBIMR_NTAU * S.mr[1].tau
            # (the device always includes the anti-symmetric modes: the split evaluators the gradient differentiates)
            h = t > t_end ? zero(ComplexF64) :
                (t <= S.t_a ? GravitationalWaveforms._seobv5phm_hpc_insp(S, t, P[10, w], P[11, w]) :
                    GravitationalWaveforms._seobv5phm_hpc_mr(S, t, P[10, w], P[11, w])) * GravitationalWaveforms._eobimr_turn_on(t / (4 * pi / BUF[1, 7, w]))
        end
        X[j, w] = h
    end
    return nothing
end

# (5') `PEConfig.time_ref_frequency > 0`: the sampled time row is t_ref = t_c + t* + time_ref_offset, with
# t* = t(f_ref) - t_peak (s, negative) the time from where the orbital frequency of the dynamics passes pi f_ref M
# (the (2,2) frequency f_ref; linear between nodes, clamped to the dynamics) to the amplitude peak. An in-band arrival
# time is what the data measure; t_c (the peak) slides along a long curved ridge with masses and spins.
# Forward only: the gradient driver rejects it.
@inline function _phm_tref_body!(w, TP, BUF, DM, P, fref)
    @inbounds begin
        TP[3, w] = 0.0
        (fref > 0) & (TP[2, w] > 0) || return nothing
        n = DM[w].n
        Ms = P[13, w]
        omr = pi * fref * Ms
        i = 1
        while (i < n - 1) & (BUF[i+1, 7, w] < omr)
            i += 1
        end
        o0, o1 = BUF[i, 7, w], BUF[i+1, 7, w]
        s = clamp((omr - o0) / (o1 - o0), 0.0, 1.0)
        tf = BUF[i, 1, w] + s * (BUF[i+1, 1, w] - BUF[i, 1, w])
        TP[3, w] = tf * Ms - TP[1, w]
    end
    return nothing
end

# Detector calibration (matched-gw150914; bilby CubicSpline, `Calibration.jl`): the factor at bin i for walker
# w, C = (1 + dA)(2 + i dphi) / (2 - i dphi), dA = sum_k WC[i, k] CAL[o + k, w], dphi from rows o + NN + k. `calv`:
# `nothing` (no calibration) or (CAL, WC, o, Val(NN)).
@inline _cal_bin(::Nothing, i, w) = true
@inline function _cal_bin(calv::Tuple, i, w)
    CAL, WC, o, nnv = calv
    return _cal_factor_bin(WC, CAL, i, w, o, nnv)
end
@inline function _cal_factor_bin(WC, CAL, i, w, o, ::Val{NN}) where {NN}
    dA = 0.0; dP = 0.0
    for k in 1:NN
        wk = @inbounds WC[i, k]
        dA += wk * @inbounds(CAL[o + k, w])
        dP += wk * @inbounds(CAL[o + NN + k, w])
    end
    return (1 + dA) * Complex(2.0, dP) / Complex(2.0, -dP)
end
@inline _cal_det(::Nothing, d) = nothing
@inline _cal_det(cal::Tuple, d) = (cal[1], cal[2], (d - 1) * 2 * _valn(cal[3]), cal[3])
@inline _valn(::Val{N}) where {N} = N

# (6) the detector template at bin frequency f from walker w's FFT, and one detector's sums
# (`_xprec_det_sums(::SEOBNRv5PHM, ...)`); `seobnr_phm_device_strain` evaluates the same template for injections.
@inline function _phm_h_bin(X, w, N, dtM, Kd, tpk, Fplus, Fcross, tsec::T, f, df) where {T}
    @inbounds begin
        k = round(Int, f / df)
        sh = Complex{T}(cos(2 * T(pi) * f * tpk), sin(2 * T(pi) * f * tpk))
        a = dtM * X[k+1, w] * sh
        b = dtM * X[mod(-k, N)+1, w] * conj(sh)
        hp = -Kd * (a + conj(b)) / 2
        hc = Kd * (a - conj(b)) / (2im)
        tc_phase = -2 * T(pi) * f * tsec   # time sign: see _PHASE_CONVENTION
        return (Fplus * hp + Fcross * hc) * Complex{T}(cos(tc_phase), sin(tc_phase))
    end
end
@inline function _phm_det_sums(X, w, N, dtM, Kd, tpk, Fplus, Fcross, tsec::T, freqs, strain, psd, df, calv = nothing) where {T}
    acc = zero(T); hh = zero(T); zr = zero(T); zi = zero(T)
    @inbounds for i in eachindex(freqs)
        h = _phm_h_bin(X, w, N, dtM, Kd, tpk, Fplus, Fcross, tsec, freqs[i], df) * _cal_bin(calv, i, w)
        acc += _safe_div(abs2(h - strain[i]), psd[i])
        hh += _safe_div(abs2(h), psd[i])
        sh2 = conj(strain[i]) * h
        zr += _safe_div(real(sh2), psd[i])
        zi += _safe_div(imag(sh2), psd[i])
    end
    return acc, hh, zr, zi
end
@inline function _phm_sums_body!(w, ll, params, P, TP, X, N, df, freqs_tuple, strain_tuple, psd_tuple, det_tuple, gmst_val, lik_arr, tref::Int, geo::Bool, cal = nothing)
    @inbounds begin
        T = Float64
        lik = _lik_from_array(lik_arr)
        dL, t_r, ra, dec, psi = params[5, w], params[6, w], params[8, w], params[9, w], params[11, w]
        dt_ref = ifelse(geo & (tref > 0), time_delay_from_geocenter_gmst(det_tuple[max(tref, 1)], ra, dec, gmst_val), zero(T))
        t_c = t_r - ifelse(lik.fref > 0, TP[3, w] + lik.toff, zero(T))   # sampled row = t_ref (`_phm_tref_body!`)
        ll_plain = zero(T); hh = zero(T); zr = zero(T); zi = zero(T)
        for d in 1:length(freqs_tuple)
            dt_geo = ifelse(geo, time_delay_from_geocenter_gmst(det_tuple[d], ra, dec, gmst_val) - dt_ref, zero(T))
            Fplus, Fcross = antenna_response_gmst(det_tuple[d], ra, dec, psi, gmst_val)
            a, h2, r, i = _phm_det_sums(X, w, N, P[12, w], P[14, w], TP[1, w], Fplus, Fcross, t_c + dt_geo, freqs_tuple[d],
                strain_tuple[d], psd_tuple[d], df, _cal_det(cal, d))
            df_d = freqs_tuple[d][2] - freqs_tuple[d][1]
            ll_plain += T(-2) * df_d * a
            hh += T(4) * df_d * h2
            zr += T(4) * df_d * r
            zi += T(4) * df_d * i
        end
        v = _marg_combine(ll_plain, hh, zr, zi, dL, lik)
        ll[w] = ifelse((TP[2, w] > 0) & isfinite(v), v, T(_SEOBV5F2_DEADZONE_LOGLIKELIHOOD))
    end
    return nothing
end

# ---- Calibration marginalization (Payne et al. 2020, bilby's `calibration_marginalization`) ----
# The likelihood averaged over `nc` response curves C_k drawn from the calibration prior (`calibration_curves`):
#   logL = log( (1/nc) sum_k L_dist(hh_k, zr_k) ),
#   hh_k = 4 df sum_j |C_k(j)|^2 V(j),  z_k = 4 df sum_j C_k(j) U(j),  V = |h|^2 / S,  U = conj(d) h / S,
# j = (detector, bin) stacked, ll_plain_k = -dd/2 - hh_k/2 + zr_k. Per batch: a walker x bin kernel writes V, U (the
# uncalibrated template), two GEMMs give HH = A V, Z = B U (A = |C|^2, B = C), a walker kernel takes the log-sum-exp over
# curves. Reverse: with p_k the curves' posterior weights, the template's bin cotangent is (alpha h + beta d) / S with
#   alpha = -4 df sum_k p_k E_k[u^2] |C_k|^2,   beta = 4 df sum_k p_k E_k[u] conj(C_k)   (`_distance_marg_moments`),
# i.e. two more GEMMs (V, U are reused for alpha, beta); with C = 1 this is the uncalibrated reverse exactly. No phase
# marginalization. Buffers for `nw` walkers.
struct _PHMCalMarg{MR,MC,VR}
    A::MR; B::MC                # nc x J: |C|^2, C
    V::MR; U::MC                # J x nw: V, U (forward); alpha, beta (reverse)
    HH::MR; ZZ::MC              # nc x nw: A V, B U
    LK::MR; EU::MR; EU2::MR; ED::MR   # nc x nw: per curve value and moments
    G1::MR; G2::MC              # nc x nw: -4 df p E[u^2], 4 df p E[u]
    GDL::VR                     # nw: d logL / d dL (explicit)
    dd::Float64                 # 4 df sum |d|^2 / S over detectors
    nc::Int; nb::Int; nw::Int
end
"""
    phm_calmarg(C, data, nw; backend) -> _PHMCalMarg

Device buffers for calibration marginalization over the response curves `C` (`ncurves x ndet nbins`, from
`GravitationalWaveBayes.calibration_curves`) of the detectors `data` (host `LikelihoodData`s, one shared grid), for `nw` walkers.
"""
function phm_calmarg(C::AbstractMatrix{<:Complex}, data, nw::Int; backend)
    dets = data isa AbstractVector ? data : [data]
    nb = length(dets[1].freqs); J = length(dets) * nb
    size(C, 2) == J || throw(ArgumentError("calibration curves have $(size(C, 2)) columns, the data $J bins"))
    nc = size(C, 1)
    al(T, dims...) = KernelAbstractions.allocate(backend, T, dims...)
    A = al(Float64, nc, J); copyto!(A, abs2.(C))
    B = al(ComplexF64, nc, J); copyto!(B, ComplexF64.(C))
    dd = 0.0
    for d in dets
        fr = Array(d.freqs); s = Array(d.strain); p = Array(d.psd)
        dd += 4 * (fr[2] - fr[1]) * sum(_safe_div(abs2(s[i]), p[i]) for i in eachindex(s))
    end
    return _PHMCalMarg(A, B, al(Float64, J, nw), al(ComplexF64, J, nw), al(Float64, nc, nw), al(ComplexF64, nc, nw),
        al(Float64, nc, nw), al(Float64, nc, nw), al(Float64, nc, nw), al(Float64, nc, nw), al(Float64, nc, nw),
        al(ComplexF64, nc, nw), al(Float64, nw), dd, nc, nb, nw)
end
@inline function _phm_det_vu!(V, U, w, o, X, N, dtM, Kd, tpk, Fplus, Fcross, tsec::T, freqs, strain, psd, df) where {T}
    @inbounds for i in eachindex(freqs)
        h = _phm_h_bin(X, w, N, dtM, Kd, tpk, Fplus, Fcross, tsec, freqs[i], df)
        wgt = _safe_div(one(T), psd[i])
        V[o + i, w] = abs2(h) * wgt
        U[o + i, w] = conj(strain[i]) * h * wgt
    end
    return nothing
end
@inline function _phm_vu_body!(w, V, U, params, P, TP, X, N, df, freqs_tuple, strain_tuple, psd_tuple, det_tuple, gmst_val, lik_arr, tref::Int, geo::Bool, nb::Int)
    @inbounds begin
        T = Float64
        lik = _lik_from_array(lik_arr)
        t_r, ra, dec, psi = params[6, w], params[8, w], params[9, w], params[11, w]
        dt_ref = ifelse(geo & (tref > 0), time_delay_from_geocenter_gmst(det_tuple[max(tref, 1)], ra, dec, gmst_val), zero(T))
        t_c = t_r - ifelse(lik.fref > 0, TP[3, w] + lik.toff, zero(T))
        for d in 1:length(freqs_tuple)
            dt_geo = ifelse(geo, time_delay_from_geocenter_gmst(det_tuple[d], ra, dec, gmst_val) - dt_ref, zero(T))
            Fplus, Fcross = antenna_response_gmst(det_tuple[d], ra, dec, psi, gmst_val)
            _phm_det_vu!(V, U, w, (d - 1) * nb, X, N, P[12, w], P[14, w], TP[1, w], Fplus, Fcross, t_c + dt_geo,
                freqs_tuple[d], strain_tuple[d], psd_tuple[d], df)
        end
    end
    return nothing
end
# Walker w: logL = log-mean-exp over the curves; G1, G2, GDL for the reverse (zero for a dead / non-finite walker).
@inline function _phm_cm_body!(w, ll, G1, G2, GDL, LK, EU, EU2, ED, HH, ZZ, params, TP, dd, df4, nc::Int, lik_arr)
    @inbounds begin
        T = Float64
        lik = _lik_from_array(lik_arr)
        dL = params[5, w]
        m = T(-Inf)
        for k in 1:nc
            hh = df4 * HH[k, w]; zr = df4 * real(ZZ[k, w])
            lp = -dd / 2 - hh / 2 + zr
            # (without distance marginalization -- the posterior reconstruction -- l_k = ll_plain_k: E[u] = E[u^2] = 1)
            v, eu, eu2, ed = lik.margdist ? _distance_marg_moments(lp, hh, zr, dL, lik) : (lp, one(T), one(T), zero(T))
            LK[k, w] = v; EU[k, w] = eu; EU2[k, w] = eu2; ED[k, w] = ed
            m = max(m, v)
        end
        s = zero(T)
        for k in 1:nc
            s += exp(LK[k, w] - m)
        end
        lse = m + log(s)
        ok = (TP[2, w] > 0) & isfinite(lse)
        ll[w] = ifelse(ok, lse - log(T(nc)), T(_SEOBV5F2_DEADZONE_LOGLIKELIHOOD))
        gdl = zero(T)
        for k in 1:nc
            p = ifelse(ok, exp(LK[k, w] - lse), zero(T))
            G1[k, w] = -df4 * p * EU2[k, w]
            G2[k, w] = Complex(df4 * p * EU[k, w], zero(T))
            gdl += ifelse(ok, p * ED[k, w], zero(T))
        end
        GDL[w] = gdl
    end
    return nothing
end

# Workgroup size of the walker-serial kernels (one thread per walker). 32 measured fastest at 256, 1024 and 4096
# walkers (V100, scripts/bench_phm_wg.jl): smaller groups idle warp lanes and do not shorten the per-thread chain.
# `GRAVITATIONALWAVEBAYES_PHM_WG` overrides it (`GPU.tuning`).
_phm_wg(_) = GPU.tuning(:PHM_WG, 32)
@kernel function _phm_walker_k!(body, args)
    w = @index(Global, Linear)
    body(w, args...)
end
@kernel function _phm_sample_k!(body, args)
    j, w = @index(Global, NTuple)
    body(j, w, args...)
end

# The staged dynamics (2) of all walkers.
function _phm_dynamics!(W, backend, nw; stamp! = _ -> nothing)
    gpu = !(backend isa KernelAbstractions.CPU)
    wg = gpu ? _phm_wg(nw) : 1
    nwg = gpu ? (64, 1) : (_PHM_NPA, 1)
    _phm_walker_k!(backend)(_phm_dstart_body!, (W.P, W.Z, W.DZ, W.ZC, W.BUF, W.DM); ndrange = nw, workgroupsize = wg)
    stamp!(:dyn_start)
    _phm_sample_k!(backend)(_phm_padiab_body!, (W.P, W.Z, W.DZ, W.BUF, W.DM, W.PAS); ndrange = (_PHM_NPA, nw),
        workgroupsize = nwg)
    for k in 1:8
        _phm_walker_k!(backend)(_phm_paslope_body!, (k, W.BUF, W.SCR, W.DM, W.PAS, W.PAD); ndrange = nw, workgroupsize = wg)
        _phm_sample_k!(backend)(_phm_paupd_body!, (k, W.P, W.Z, W.DZ, W.BUF, W.DM, W.PAS, W.PAD); ndrange = (_PHM_NPA, nw),
            workgroupsize = nwg)
    end
    _phm_sample_k!(backend)(_phm_panode_body!, (W.P, W.Z, W.DZ, W.BUF, W.DM, W.PAS, W.PAY); ndrange = (_PHM_NPA, nw),
        workgroupsize = nwg)
    _phm_walker_k!(backend)(_phm_paint_body!, (W.BUF, W.SCR, W.DM, W.PAD, W.PAY); ndrange = nw, workgroupsize = wg)
    stamp!(:dyn_pa)
    _phm_walker_k!(backend)(_phm_plunge_body!, (W.P, W.Z, W.DZ, W.BUF, W.SCR, W.DM); ndrange = nw, workgroupsize = wg)
    return nothing
end

# Stages (1)-(5') + FFT for the walkers `params` (15 physical rows) on the data grid `freqs`: the FFT'd waveforms
# `W.X` and the peak / t*(f_ref) times `W.TP`. Returns the work buffers and the FFT length.
function _phm_waveforms!(params, cfg::PEConfig, freqs, nfft::Int, backend, stamp! = _ -> nothing)
    nw = size(params, 2)
    fh = Array(freqs[1:2])
    df = fh[2] - fh[1]
    N = nfft > 0 ? nfft : _phm_nfft(Array(freqs), minimum(Array(view(params, 1, :)) .+ Array(view(params, 2, :))))
    W = _phm_work(backend, nw, N)
    N = W.N                                  # (possibly larger: `_phm_work` never shrinks)
    gpu = !(backend isa KernelAbstractions.CPU)
    wg = gpu ? _phm_wg(nw) : 1
    _phm_walker_k!(backend)(_phm_prep_body!, (W.P, params, fh[1], 1 / (N * df), Float64(cfg.phase_ref_frequency));
        ndrange = nw, workgroupsize = wg)
    _phm_dynamics!(W, backend, nw; stamp! = stamp!)
    stamp!(:dynamics)
    _phm_walker_k!(backend)(_phm_bpre_body!, (W.P, W.Z, W.DZ, W.BUF, W.SCR, W.DM, W.RPRE); ndrange = nw, workgroupsize = wg)
    nwg = gpu ? (64, 1) : (_PHM_ND, 1)
    _phm_sample_k!(backend)(_phm_bnode_body!, (W.P, W.Z, W.DZ, W.BUF, W.SCR, W.DM, W.MD, W.RPRE); ndrange = (_PHM_ND, nw),
        workgroupsize = nwg)
    _phm_sample_k!(backend)(_phm_bderiv_body!, (W.P, W.Z, W.DZ, W.BUF, W.SCR, W.DM, W.MD); ndrange = (_PHM_ND, nw),
        workgroupsize = nwg)
    _phm_walker_k!(backend)(_phm_bpost_body!, (W.P, W.Z, W.DZ, W.BUF, W.SCR, W.DM, W.MD, W.GT, W.RPRE, W.QREC, W.REC, W.TP, W.TM,
        N); ndrange = nw, workgroupsize = wg)
    _phm_walker_k!(backend)(_phm_tref_body!, (W.TP, W.BUF, W.DM, W.P, Float64(cfg.time_ref_frequency)); ndrange = nw,
        workgroupsize = wg)
    stamp!(:build)
    _phm_sample_k!(backend)(_phm_sample_body!, (W.P, W.Z, W.DZ, W.BUF, W.SCR, W.DM, W.MD, W.GT, W.REC, W.TP, W.X);
        ndrange = (N, nw), workgroupsize = gpu ? (64, 1) : (1024, 1))
    stamp!(:samples)
    W.plan * W.X
    stamp!(:fft)
    return W, N
end

"""
    seobnr_phm_device_strain(params, cfg, data) -> (h::Vector{Vector{ComplexF64}}, ok::Bool)

The SEOBNRv5PHM detector templates the device likelihood compares with the data, for ONE parameter vector `params` (15
physical rows, the time row in the likelihood's frame: `cfg.time_reference` / `time_ref_frequency`, phase and spins at
`cfg.phase_ref_frequency`) on each detector of `data` (a vector of `LikelihoodData`, one shared grid; their strain is
not used). The same stages and per-bin template (`_phm_h_bin`) as `seobnr_phm_device_loglikelihood!`, on the CPU, so a
zero-noise injection of `h` has logL = 0 at `params` exactly. `ok = false`: the dynamics failed (dead zone).
"""
function seobnr_phm_device_strain(params::AbstractVector, cfg::PEConfig, data::AbstractVector)
    P = reshape(Float64.(collect(params)), 15, 1)
    freqs = Array(data[1].freqs)
    df = freqs[2] - freqs[1]
    W, N = _phm_waveforms!(P, cfg, freqs, 0, KernelAbstractions.CPU())
    X = Array(W.X); WP = Array(W.P); TP = Array(W.TP)
    gmst_val = gmst(data[1].gps_time)
    lik = _lik_opts(cfg)
    geo = length(data) > 1
    tref = cfg.time_reference
    ra, dec, psi = P[8, 1], P[9, 1], P[11, 1]
    dt_ref = geo && tref > 0 ? time_delay_from_geocenter_gmst(data[tref].detector, ra, dec, gmst_val) : 0.0
    t_c = P[6, 1] - (lik.fref > 0 ? TP[3, 1] + lik.toff : 0.0)
    h = map(data) do d
        dt_geo = geo ? time_delay_from_geocenter_gmst(d.detector, ra, dec, gmst_val) - dt_ref : 0.0
        Fplus, Fcross = antenna_response_gmst(d.detector, ra, dec, psi, gmst_val)
        [_phm_h_bin(X, 1, N, WP[12, 1], WP[14, 1], TP[1, 1], Fplus, Fcross, t_c + dt_geo, f, df) for f in Array(d.freqs)]
    end
    return collect(h), TP[2, 1] > 0
end

"""
    seobnr_phm_device_loglikelihood!(ll, params, cfg, data; backend, nfft = 0, timers = nothing)

SEOBNRv5PHM log-likelihood of the walkers `params` (15 physical rows, as `batched_log_likelihood_kernel!`) for the
detectors `data` (a `LikelihoodData` or a vector of them; one shared frequency grid) on `backend`, through the
stages of this file. `nfft` (0: `_phm_nfft` at the batch's lightest total mass) sets the FFT length. `timers`: a
`Dict{Symbol,Float64}` that receives the stage times (synchronizing after each stage).
"""
function seobnr_phm_device_loglikelihood!(ll, params, cfg::PEConfig, data; backend = KernelAbstractions.CPU(),
        nfft::Int = 0, timers = nothing, cal = nothing)
    geo = data isa AbstractVector
    dets = geo ? data : [data]
    ndet = length(dets)
    nw = size(params, 2)
    gmst_val = gmst(dets[1].gps_time)
    freqs_tuple = ntuple(i -> dets[i].freqs, ndet)
    strain_tuple = ntuple(i -> dets[i].strain, ndet)
    psd_tuple = ntuple(i -> dets[i].psd, ndet)
    det_tuple = ntuple(i -> geometry(dets[i].detector), ndet)
    fh = Array(freqs_tuple[1][1:2])
    df = fh[2] - fh[1]
    for d in 2:ndet
        length(freqs_tuple[d]) == length(freqs_tuple[1]) || throw(ArgumentError("SEOBNRv5PHM: all detectors must share one frequency grid"))
    end
    t0 = Ref(time())
    stamp!(k) = timers === nothing || (KernelAbstractions.synchronize(backend); t = time(); timers[k] = get(timers, k, 0.0) + t - t0[]; t0[] = t)
    W, N = _phm_waveforms!(params, cfg, freqs_tuple[1], nfft, backend, stamp!)
    gpu = !(backend isa KernelAbstractions.CPU)
    wg = gpu ? _phm_wg(nw) : 1
    if cal isa _PHMCalMarg
        cal.nw == nw || throw(ArgumentError("calibration-marginalization buffers for $(cal.nw) walkers, batch has $nw"))
        cfg.marginalize_phase && throw(ArgumentError("calibration marginalization: phase marginalization not supported"))
        la = _lik_array(cfg, backend)
        _phm_walker_k!(backend)(_phm_vu_body!, (cal.V, cal.U, params, W.P, W.TP, W.X, N, df, freqs_tuple, strain_tuple,
            psd_tuple, det_tuple, gmst_val, la, cfg.time_reference, geo, cal.nb);
            ndrange = nw, workgroupsize = wg)
        LinearAlgebra.mul!(cal.HH, cal.A, cal.V)
        LinearAlgebra.mul!(cal.ZZ, cal.B, cal.U)
        _phm_walker_k!(backend)(_phm_cm_body!, (ll, cal.G1, cal.G2, cal.GDL, cal.LK, cal.EU, cal.EU2, cal.ED, cal.HH, cal.ZZ,
            params, W.TP, cal.dd, 4 * df, cal.nc, la); ndrange = nw, workgroupsize = wg)
    else
        _phm_walker_k!(backend)(_phm_sums_body!, (ll, params, W.P, W.TP, W.X, N, df, freqs_tuple, strain_tuple, psd_tuple,
            det_tuple, gmst_val, _lik_array(cfg, backend), cfg.time_reference, geo, cal);
            ndrange = nw, workgroupsize = wg)
    end
    KernelAbstractions.synchronize(backend)
    stamp!(:sums)
    return ll
end
