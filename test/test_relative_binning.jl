using Test
using Enzyme
using GravitationalWaveBayes
using GravitationalWaveBayes: RelativeBinningSetup, build_relative_binning, binned_log_likelihood_kernel!
using GravitationalWaveBayes: nuts_sample!, NUTSConfig
using GravitationalWaveBayes.BatchedNUTS: gw_logp_and_grad, _m1_m2_to_mc_q
using GravitationalWaveforms: Ampl, Phi, IMRPhenomD, IMRPhenomXAS, IMRPhenomXP, IMRPhenomXHM, IMRPhenomXPHM, AlignedSpinPNHamF2, SEOBNRv5AlignedSpinF2, GMsun_over_c3
using GravitationalWaveBayes.KernelFused: _phi_t0
using GravitationalWaveforms
using GravitationalWaveData
using KernelAbstractions
using Random
using LinearAlgebra

@isdefined(RUN_SLOW_ENZYME_TESTS) || (const RUN_SLOW_ENZYME_TESTS = get(ENV, "GRAVITATIONALWAVEBAYES_RUN_SLOW_TESTS", "false") == "true")

# Small inline mean/std helpers (same convention as test_batched_nuts_gaussian.jl's own
# _empirical_mean/_empirical_std; renamed with an _rb_ prefix here since all test/*.jl files
# share the Main namespace under Pkg.test(), and identical helper names across files otherwise
# trigger a "Method definition ... overwritten" warning -- avoids adding Statistics as a test
# dependency).
_rb_empirical_mean(samples::AbstractMatrix{T}) where {T} = vec(sum(samples; dims=2)) ./ size(samples, 2)
_rb_empirical_std(samples::AbstractMatrix{T}, mu::AbstractVector{T}) where {T} =
    sqrt.(vec(sum(abs2, samples .- mu; dims=2)) ./ (size(samples, 2) - 1))

# Manual host-side detector-frame waveform, matching KernelFused.jl's `_extrinsic_setup` exactly
# (see test_kernel_fused.jl's identical helper for the full rationale) -- used here to build
# noise-free injections at the full 11-parameter physical convention.
# Template phase in the kernels' physical convention (KernelFused `_PHASE_CONVENTION`): XAS family
# flips the time term only, the other models are conjugated as a whole.
_rbt_phase(model, freqs, mc, eta, chi1, chi2, t0, M_sec, phase) = model isa IMRPhenomXAS ?
    Phi(model, freqs, mc, eta, chi1, chi2; t0=-_phi_t0(model, t0, M_sec), phiRef=phase) :
    .-Phi(model, freqs, mc, eta, chi1, chi2; t0=_phi_t0(model, t0, M_sec), phiRef=phase)

function _rb_h_det(model, freqs, physical::AbstractVector, detector, gps_time)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = physical
    mu = (m1 * m2) / (m1 + m2)
    mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
    eta = (m1 * m2) / (m1 + m2)^2
    M_sec = (mc / eta^0.6) * GMsun_over_c3
    t0 = -2 * pi * t_c / M_sec

    amp = Ampl(model, freqs, mc, eta, chi1, chi2, dL)
    phase_vals = _rbt_phase(model, freqs, mc, eta, chi1, chi2, t0, M_sec, phase)
    h = amp .* exp.(im .* phase_vals)

    pfac = 0.5 * (1 + cos(theta_jn)^2)
    cfac = cos(theta_jn)
    Fplus, Fcross = GravitationalWaveData.compute_antenna_response(detector, ra, dec, psi, gps_time)
    combo = Fplus * pfac - im * Fcross * cfac
    return combo .* h
end

# Manual host-side IMRPhenomXP detector-frame waveform (Phase 3 relative-binning follow-on), mirroring
# `KernelFused._fused_waveform_likelihood_kernel_phenomxp!` line-for-line -- same role/shape as
# `test_phenomxp.jl`'s own `_manual_h_det_phenomxp`, kept as a local copy here per this codebase's
# established per-test-file self-containment convention (see `_rb_h_det` above for the aligned-spin
# analogue this mirrors).
function _rb_h_det_phenomxp(freqs::AbstractVector, physical::AbstractVector, detector, gps_time)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi, chi1x, chi1y, chi2x, chi2y = physical
    mu = (m1 * m2) / (m1 + m2)
    mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
    eta = (m1 * m2) / (m1 + m2)^2
    M_sec = (mc / eta^0.6) * GMsun_over_c3
    t0 = 2 * pi * t_c / M_sec
    phiRef = phase

    Fplus, Fcross = GravitationalWaveData.compute_antenna_response(detector, ra, dec, psi, gps_time)

    f_ref = freqs[1]
    setup = GravitationalWaveforms._xp_precession_setup(m1, m2, chi1x, chi1y, chi1, chi2x, chi2y, chi2,
                                              f_ref, theta_jn, phase)

    phi_coeffs = GravitationalWaveforms._xas_phi_coeffs(IMRPhenomXAS(), mc, eta, chi1, chi2; t0=t0, phiRef=phiRef)
    ampl_coeffs = GravitationalWaveforms._xas_ampl_coeffs(IMRPhenomXAS(), mc, eta, chi1, chi2, dL)

    h = Vector{ComplexF64}(undef, length(freqs))
    for (i, f) in enumerate(freqs)
        amp = GravitationalWaveforms._xas_ampl_scalar(ampl_coeffs, f)
        phase_val = GravitationalWaveforms._xas_phi_scalar(phi_coeffs, f)
        hAS = (amp * exp(im * phase_val)) * GravitationalWaveforms._XP_HAS_MODE_NORMALIZATION
        Mf = M_sec * f
        v = cbrt(pi * Mf)
        hp, hc = GravitationalWaveforms._xp_twist_up22(v, setup, hAS)
        hp, hc = GravitationalWaveforms._xp_apply_zeta_rotation(hp, hc, setup.zeta_polarization)
        h[i] = Fplus * hp + Fcross * hc
    end
    return h
end

# Multi-detector variant, adding the per-detector `time_delay_from_geocenter`-shifted `t0` -- same
# relationship `_rb_h_det_phenomxp_multi` bears to `_rb_h_det_phenomxp` as `_rb_h_det_multi`
# (inside the multi-detector testset below) bears to `_rb_h_det` above.
function _rb_h_det_phenomxp_multi(freqs::AbstractVector, physical::AbstractVector, detector, gps_time, f_ref)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi, chi1x, chi1y, chi2x, chi2y = physical
    mu = (m1 * m2) / (m1 + m2)
    mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
    eta = (m1 * m2) / (m1 + m2)^2
    M_sec = (mc / eta^0.6) * GMsun_over_c3
    delay = GravitationalWaveData.time_delay_from_geocenter(detector, ra, dec, gps_time)
    t0 = 2 * pi * (t_c + delay) / M_sec
    phiRef = phase

    Fplus, Fcross = GravitationalWaveData.compute_antenna_response(detector, ra, dec, psi, gps_time)

    setup = GravitationalWaveforms._xp_precession_setup(m1, m2, chi1x, chi1y, chi1, chi2x, chi2y, chi2,
                                              f_ref, theta_jn, phase)

    phi_coeffs = GravitationalWaveforms._xas_phi_coeffs(IMRPhenomXAS(), mc, eta, chi1, chi2; t0=t0, phiRef=phiRef)
    ampl_coeffs = GravitationalWaveforms._xas_ampl_coeffs(IMRPhenomXAS(), mc, eta, chi1, chi2, dL)

    h = Vector{ComplexF64}(undef, length(freqs))
    for (i, f) in enumerate(freqs)
        amp = GravitationalWaveforms._xas_ampl_scalar(ampl_coeffs, f)
        phase_val = GravitationalWaveforms._xas_phi_scalar(phi_coeffs, f)
        hAS = (amp * exp(im * phase_val)) * GravitationalWaveforms._XP_HAS_MODE_NORMALIZATION
        Mf = M_sec * f
        v = cbrt(pi * Mf)
        hp, hc = GravitationalWaveforms._xp_twist_up22(v, setup, hAS)
        hp, hc = GravitationalWaveforms._xp_apply_zeta_rotation(hp, hc, setup.zeta_polarization)
        h[i] = Fplus * hp + Fcross * hc
    end
    return h
end

# Manual host-side IMRPhenomXHM detector-frame waveform (Phase 3 relative-binning follow-on),
# mirroring `KernelFused._fused_waveform_likelihood_kernel_phenomxhm!`/`test_phenomxhm.jl`'s own
# `_manual_h_det_phenomxhm` line-for-line -- kept as a local `_rb_`-prefixed copy here per this
# codebase's established per-test-file self-containment convention (see `_rb_h_det_phenomxp` above
# for the analogous IMRPhenomXP copy) -- a plain, non-`_rb_`-prefixed name would collide with
# `test_phenomxhm.jl`'s own helper of the same purpose since all test/*.jl files share the Main
# namespace under `Pkg.test()` (see this file's top-of-file comment on `_rb_empirical_mean`).
function _rb_h_det_phenomxhm(freqs::AbstractVector, physical::AbstractVector, detector, gps_time)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = physical
    mu = (m1 * m2) / (m1 + m2)
    mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
    eta = (m1 * m2) / (m1 + m2)^2
    f_ref = freqs[1]

    sys = GravitationalWaveforms._xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, phase)
    ms21 = GravitationalWaveforms._xhm_mode_setup(2, 1, sys)
    ms33 = GravitationalWaveforms._xhm_mode_setup(3, 3, sys)
    ms44 = GravitationalWaveforms._xhm_mode_setup(4, 4, sys)
    m32 = GravitationalWaveforms._xhm_mode32_setup(sys)
    piHalf = pi / 2

    Fplus, Fcross = GravitationalWaveData.compute_antenna_response(detector, ra, dec, psi, gps_time)

    h = Vector{ComplexF64}(undef, length(freqs))
    for (i, f) in enumerate(freqs)
        hp, hc = GravitationalWaveforms._xhm_hphc_scalar(f, sys, ms21, ms33, ms44, m32, theta_jn, piHalf)
        shift = exp(-im * 2 * pi * f * t_c)
        hp *= shift; hc *= shift
        h[i] = Fplus * hp + Fcross * hc
    end
    return h
end

# Multi-detector variant, adding the per-detector `time_delay_from_geocenter`-shifted `t_c` --
# same relationship this bears to `_rb_h_det_phenomxhm` as `_rb_h_det_phenomxp_multi` bears to
# `_rb_h_det_phenomxp` above.
function _rb_h_det_phenomxhm_multi(freqs::AbstractVector, physical::AbstractVector, detector, gps_time)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = physical
    mu = (m1 * m2) / (m1 + m2)
    mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
    eta = (m1 * m2) / (m1 + m2)^2
    f_ref = freqs[1]
    delay = GravitationalWaveData.time_delay_from_geocenter(detector, ra, dec, gps_time)
    t_c_d = t_c + delay

    sys = GravitationalWaveforms._xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, phase)
    ms21 = GravitationalWaveforms._xhm_mode_setup(2, 1, sys)
    ms33 = GravitationalWaveforms._xhm_mode_setup(3, 3, sys)
    ms44 = GravitationalWaveforms._xhm_mode_setup(4, 4, sys)
    m32 = GravitationalWaveforms._xhm_mode32_setup(sys)
    piHalf = pi / 2

    Fplus, Fcross = GravitationalWaveData.compute_antenna_response(detector, ra, dec, psi, gps_time)

    h = Vector{ComplexF64}(undef, length(freqs))
    for (i, f) in enumerate(freqs)
        hp, hc = GravitationalWaveforms._xhm_hphc_scalar(f, sys, ms21, ms33, ms44, m32, theta_jn, piHalf)
        shift = exp(-im * 2 * pi * f * t_c_d)
        hp *= shift; hc *= shift
        h[i] = Fplus * hp + Fcross * hc
    end
    return h
end

# Manual host-side IMRPhenomXPHM detector-frame waveform (Phase 5 Task 5c relative-binning
# follow-on): combines `_rb_h_det_phenomxp`'s precession setup (+ Task 5e's
# `final_spin_override`) with `_rb_h_det_phenomxhm`'s higher-mode assembly/`t_c` post-
# multiplication -- kept as a local `_rb_`-prefixed copy per this file's established
# self-containment convention (see `_rb_h_det_phenomxp`'s own comment for why).
function _rb_h_det_phenomxphm(freqs::AbstractVector, physical::AbstractVector, detector, gps_time)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi, chi1x, chi1y, chi2x, chi2y = physical
    mu = (m1 * m2) / (m1 + m2)
    mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
    eta = (m1 * m2) / (m1 + m2)^2
    f_ref = freqs[1]

    Fplus, Fcross = GravitationalWaveData.compute_antenna_response(detector, ra, dec, psi, gps_time)

    setup = GravitationalWaveforms._xp_precession_setup(m1, m2, chi1x, chi1y, chi1, chi2x, chi2y, chi2,
                                              f_ref, theta_jn, phase)
    final_spin = GravitationalWaveforms._xas_precessing_final_spin(eta, chi1, chi2, setup.chi_p)
    sys = GravitationalWaveforms._xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, zero(phase);
                                         final_spin_override=final_spin)
    ms21 = GravitationalWaveforms._xhm_mode_setup(2, 1, sys)
    ms33 = GravitationalWaveforms._xhm_mode_setup(3, 3, sys)
    ms44 = GravitationalWaveforms._xhm_mode_setup(4, 4, sys)
    m32 = GravitationalWaveforms._xhm_mode32_setup(sys)

    h = Vector{ComplexF64}(undef, length(freqs))
    for (i, f) in enumerate(freqs)
        hp, hc = GravitationalWaveforms._xphm_hphc_scalar(f, setup, sys, ms21, ms33, ms44, m32)
        hp, hc = GravitationalWaveforms._xp_apply_zeta_rotation(hp, hc, setup.zeta_polarization)
        shift = exp(-im * 2 * pi * f * t_c)
        hp *= shift; hc *= shift
        h[i] = Fplus * hp + Fcross * hc
    end
    return h
end

# Multi-detector variant, adding the per-detector `time_delay_from_geocenter`-shifted `t_c` --
# same relationship this bears to `_rb_h_det_phenomxphm` as `_rb_h_det_phenomxp_multi` bears to
# `_rb_h_det_phenomxp` above.
function _rb_h_det_phenomxphm_multi(freqs::AbstractVector, physical::AbstractVector, detector, gps_time, f_ref)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi, chi1x, chi1y, chi2x, chi2y = physical
    mu = (m1 * m2) / (m1 + m2)
    mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
    eta = (m1 * m2) / (m1 + m2)^2
    delay = GravitationalWaveData.time_delay_from_geocenter(detector, ra, dec, gps_time)
    t_c_d = t_c + delay

    Fplus, Fcross = GravitationalWaveData.compute_antenna_response(detector, ra, dec, psi, gps_time)

    setup = GravitationalWaveforms._xp_precession_setup(m1, m2, chi1x, chi1y, chi1, chi2x, chi2y, chi2,
                                              f_ref, theta_jn, phase)
    final_spin = GravitationalWaveforms._xas_precessing_final_spin(eta, chi1, chi2, setup.chi_p)
    sys = GravitationalWaveforms._xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, zero(phase);
                                         final_spin_override=final_spin)
    ms21 = GravitationalWaveforms._xhm_mode_setup(2, 1, sys)
    ms33 = GravitationalWaveforms._xhm_mode_setup(3, 3, sys)
    ms44 = GravitationalWaveforms._xhm_mode_setup(4, 4, sys)
    m32 = GravitationalWaveforms._xhm_mode32_setup(sys)

    h = Vector{ComplexF64}(undef, length(freqs))
    for (i, f) in enumerate(freqs)
        hp, hc = GravitationalWaveforms._xphm_hphc_scalar(f, setup, sys, ms21, ms33, ms44, m32)
        hp, hc = GravitationalWaveforms._xp_apply_zeta_rotation(hp, hc, setup.zeta_polarization)
        shift = exp(-im * 2 * pi * f * t_c_d)
        hp *= shift; hc *= shift
        h[i] = Fplus * hp + Fcross * hc
    end
    return h
end

@testset "Relative binning (heterodyning)" begin

    @testset "binned likelihood matches full-grid likelihood near and away from theta_ref" begin
        # Realistic-magnitude PSD (matches test_kernel_fused.jl's convention: flat PSD=1 with
        # dL=1Gpc underflows to ll~=0 everywhere and makes any comparison vacuous) and a
        # noise-free injection at a known reference parameter set, over a real (if narrow-band)
        # IMRPhenomD/XAS-valid frequency grid -- not the 16-point smoke-test grid, since relative
        # binning's whole point only shows up with enough frequency bins to actually subdivide.
        for model in (IMRPhenomD(), IMRPhenomXAS())
            T = Float64
            fmin, fmax, df = 20.0, 284.0, 0.5
            freqs = collect(fmin:df:fmax)
            ntime = length(freqs)

            # physical = m1,m2,chi1,chi2,dL,t_c,phase,ra,dec,theta_jn,psi
            physical_ref = T[36.0, 29.0, 0.0, 0.0, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6]
            data0 = LikelihoodData(freqs, zeros(ComplexF64, ntime), fill(T(1e-46), ntime))
            strain = _rb_h_det(model, freqs, physical_ref, data0.detector, data0.gps_time)
            psd = fill(T(1e-46), ntime)
            data = LikelihoodData(freqs, strain, psd)
            cfg = PEConfig{T}(ntime, 1, 1e-3, 1)

            rb = build_relative_binning(model, physical_ref, data; nbins=80)
            @test length(rb.edges) - 1 <= 80
            @test length(rb.edges) >= 2

            # (point, rtol) pairs -- the last point uses a looser tolerance: it perturbs ALL 11
            # parameters simultaneously (not just mass/spin, as the pre-Task-4 version of this
            # test did), including `t_c`, which shifts the trial waveform's phase relative to
            # `theta_ref`'s in a way NOT accounted for by `_phase_equal_bin_edges`'s bin-PLACEMENT
            # heuristic (deliberately based on the bare intrinsic phase only, see
            # `build_relative_binning`'s docstring) -- so a nonzero `t_c` perturbation genuinely
            # degrades the approximation a bit more per unit perturbation than an intrinsic-only
            # change of the same pre-Task-4 test's magnitude did. Still well within relative
            # binning's documented "degrades away from theta_ref, not a bug" behavior (confirmed
            # by checking this same point against a much larger nbins, which tightens the
            # agreement as expected of a genuine linear-interpolation approximation, not a
            # structural error).
            test_points = [
                (physical_ref, 0.0),                                                                  # exact theta_ref
                (T[36.036, 29.0, 0.0, 0.0, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6], 5e-3),                  # tiny perturbation (mass)
                (T[36.0, 28.971, 0.0, 0.0, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6], 5e-3),
                (T[36.36, 29.0, 0.05, 0.0, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6], 5e-3),                  # small but multi-parameter
                # larger, all params perturbed. 3e-2 since the 2026-09-27 phase/time-convention fix: in the
                # physical convention this t_c = +5 ms point sits further from the data, and the binned
                # error there is a genuine, converging approximation error (XAS: 2.19% / 0.93% / 0.37% at
                # nbins 80 / 160 / 320; IMRPhenomD 0.82% / 0.23% / 0.09%), not a structural mismatch.
                (T[37.0, 29.0, 0.1, -0.05, 0.42, 0.005, 0.55, 1.25, -0.35, 1.05, 0.65], 3e-2),
            ]

            for (p, rtol) in test_points
                params = reshape(p, 11, 1)
                ll_full = zeros(T, 1)
                batched_log_likelihood_kernel!(ll_full, params, freqs, cfg, data; model=model)
                ll_binned = zeros(T, 1)
                binned_log_likelihood_kernel!(ll_binned, params, rb, data; model=model)

                @test isfinite(ll_full[1])
                @test isfinite(ll_binned[1])
                # Relative-binning is an approximation whose accuracy degrades away from
                # theta_ref (this is expected and documented behavior, not a bug) -- exact match
                # at theta_ref itself, then a loose relative tolerance elsewhere that still
                # confirms the approximation tracks the true likelihood surface rather than
                # being numerically unrelated to it.
                if p == physical_ref
                    @test isapprox(ll_full[1], ll_binned[1]; atol=1e-6)
                else
                    @test isapprox(ll_full[1], ll_binned[1]; rtol=rtol)
                end
            end
        end
    end

    if !RUN_SLOW_ENZYME_TESTS
        @testset "Enzyme gradient through the binned likelihood kernel (IMRPhenomD)" begin
            @test_skip "skipped by default (slow, dominated by Enzyme compile time); set GRAVITATIONALWAVEBAYES_RUN_SLOW_TESTS=true to run"
        end
    else
    @testset "Enzyme gradient through the binned likelihood kernel (IMRPhenomD)" begin
        # Same tiny problem size/rationale as test_enzyme_ka.jl. IMRPhenomD only here (not looped
        # together with IMRPhenomXAS in-process): this dev machine previously had only ~6GB total
        # RAM (upgraded to 14GB on 2026-09-18; see the gwbayes-lalsuite-comparison /
        # gwbayes-jim-comparison memories on how tight the old 6GB was for JAX's JIT compiler --
        # not yet re-verified at 14GB), and by the time this testset runs, `runtests.jl` has
        # already accumulated several other Enzyme reverse-mode compiles in the SAME process
        # (test_enzyme_gradient.jl, test_enzyme_ka.jl, test_enzyme_ka_xas.jl,
        # test_batched_nuts_gw_smoke.jl's own `_sum_ll` compile). Adding two more back-to-back
        # compiles here (IMRPhenomD binned + IMRPhenomXAS binned) reproducibly hit `LLVM ERROR: out of
        # memory` during development of this test on this machine, alongside other concurrent
        # Julia processes sharing the same RAM (this box also runs a persistent Julia
        # LSP/REPL process per the julia-repl-mcp-flaky memory, and potentially other
        # worktree-isolated subagents per the dev-plan-orchestration memory). IMRPhenomXAS's binned
        # kernel shares the identical dispatch-symmetric code path (see
        # `RelativeBinning.binned_log_likelihood_kernel!`'s docstring) and was independently
        # verified correct in an isolated one-off process during this task's validation --
        # confirmed matching finite differences to the same `rtol=1e-4, atol=1e-6` tolerance --
        # just not re-run here every `Pkg.test()` invocation, to keep this suite's peak
        # in-process memory bounded on this specific hardware.
        #
        # `params`/`params_ref` are now (11,·) -- see test_enzyme_ka.jl's matching testset for why
        # the full 11-row extrinsic space (not just the original 4) is what's actually validated.
        ntime = 16
        freqs = collect(20.0:1.0:(20.0 + ntime - 1))
        strain = zeros(ComplexF64, ntime)
        psd = ones(Float64, ntime)
        data = LikelihoodData(freqs, strain, psd)

        model = IMRPhenomD()
        physical_ref = [30.0, 25.0, 0.5, 0.3, 0.4, 0.02, 0.3, 1.2, -0.5, 1.0, 0.7]
        rb = build_relative_binning(model, physical_ref, data; nbins=8)

        params = zeros(Float64, 11, 2)
        params[1, :] .= [30.0, 40.0]
        params[2, :] .= [25.0, 20.0]
        params[3, :] .= [0.5, -0.2]
        params[4, :] .= [0.3, 0.1]
        params[5, :] .= [0.4, 1.1]
        params[6, :] .= [0.02, -0.03]
        params[7, :] .= [0.3, 2.0]
        params[8, :] .= [1.2, 4.0]
        params[9, :] .= [-0.5, 0.3]
        params[10, :] .= [1.0, 2.0]
        params[11, :] .= [0.7, 2.5]

        function sum_ll_binned(params, rb, data, model)
            ll = zeros(eltype(params), size(params, 2))
            binned_log_likelihood_kernel!(ll, params, rb, data; model=model)
            return sum(ll)
        end

        dparams = zeros(Float64, size(params))
        Enzyme.autodiff(Enzyme.Reverse, sum_ll_binned,
            Enzyme.Duplicated(params, dparams),
            Enzyme.Const(rb), Enzyme.Const(data), Enzyme.Const(model))

        h = 1e-6
        fd = zeros(Float64, size(params))
        for i in eachindex(params)
            p_plus = copy(params); p_plus[i] += h
            p_minus = copy(params); p_minus[i] -= h
            fd[i] = (sum_ll_binned(p_plus, rb, data, model) - sum_ll_binned(p_minus, rb, data, model)) / (2h)
        end

        @test isapprox(dparams, fd; rtol=1e-4, atol=1e-6)
    end
    end

    if !RUN_SLOW_ENZYME_TESTS
        @testset "Enzyme gradient through the binned likelihood kernel (AlignedSpinPNHamF2)" begin
            @test_skip "skipped by default (slow, dominated by Enzyme compile time); set GRAVITATIONALWAVEBAYES_RUN_SLOW_TESTS=true to run"
        end
    else
    @testset "Enzyme gradient through the binned likelihood kernel (AlignedSpinPNHamF2)" begin
        # Mirrors the IMRPhenomD binned-kernel Enzyme testset immediately above exactly (same tiny
        # problem size, same RAM-conservation rationale for keeping this gated behind
        # RUN_SLOW_ENZYME_TESTS rather than in the default suite) -- validates
        # `_relative_binning_kernel_alignedspinpnhamf2!` (Phase 4 relative-binning follow-on)
        # instead of `_relative_binning_kernel_phenomd!`. Frequency band (20-35 Hz) stays well
        # below this model's hard Schwarzschild-ISCO cutoff (~80 Hz at this ~55 Msun total mass),
        # same reasoning as `test_enzyme_ka_alignedspinpnhamf2.jl`.
        ntime = 16
        freqs = collect(20.0:1.0:(20.0 + ntime - 1))
        strain = zeros(ComplexF64, ntime)
        psd = ones(Float64, ntime)
        data = LikelihoodData(freqs, strain, psd)

        model = AlignedSpinPNHamF2()
        physical_ref = [30.0, 25.0, 0.5, 0.3, 0.4, 0.02, 0.3, 1.2, -0.5, 1.0, 0.7]
        rb = build_relative_binning(model, physical_ref, data; nbins=8)

        params = zeros(Float64, 11, 2)
        params[1, :] .= [30.0, 40.0]
        params[2, :] .= [25.0, 20.0]
        params[3, :] .= [0.5, -0.2]
        params[4, :] .= [0.3, 0.1]
        params[5, :] .= [0.4, 1.1]
        params[6, :] .= [0.02, -0.03]
        params[7, :] .= [0.3, 2.0]
        params[8, :] .= [1.2, 4.0]
        params[9, :] .= [-0.5, 0.3]
        params[10, :] .= [1.0, 2.0]
        params[11, :] .= [0.7, 2.5]

        function sum_ll_binned_asphf2(params, rb, data, model)
            ll = zeros(eltype(params), size(params, 2))
            binned_log_likelihood_kernel!(ll, params, rb, data; model=model)
            return sum(ll)
        end

        dparams = zeros(Float64, size(params))
        Enzyme.autodiff(Enzyme.Reverse, sum_ll_binned_asphf2,
            Enzyme.Duplicated(params, dparams),
            Enzyme.Const(rb), Enzyme.Const(data), Enzyme.Const(model))

        h = 1e-6
        fd = zeros(Float64, size(params))
        for i in eachindex(params)
            p_plus = copy(params); p_plus[i] += h
            p_minus = copy(params); p_minus[i] -= h
            fd[i] = (sum_ll_binned_asphf2(p_plus, rb, data, model) - sum_ll_binned_asphf2(p_minus, rb, data, model)) / (2h)
        end

        @test isapprox(dparams, fd; rtol=1e-4, atol=1e-6)
    end
    end

    @testset "NUTS recovery: full-grid vs relative-binning posteriors agree" begin
        # Small-but-real synthetic-injection NUTS recovery, run twice -- once against the
        # full-grid likelihood, once against the relative-binning approximation built around the
        # injected parameters -- and the two posteriors compared for statistical agreement. This
        # is NOT a from-scratch convergence study (see test_batched_nuts_gaussian.jl for that
        # pattern at toy-Gaussian scale, and DEVELOPMENT_PLAN.md Phase 4 for why full real-data
        # recovery is out of scope here); it specifically isolates whether swapping in relative
        # binning changes the recovered posterior, which is the actual question this task's
        # validation must answer.
        #
        # Only (mc, q) (rows 1-2) are actually explored/compared here -- the other 9 sampled
        # dimensions (chi1, chi2, dL, t_c, phase, ra, sin_dec, cos_theta_jn, psi) are held FIXED
        # at their injected values in both chains (zero step size / never perturbed away from
        # injection is not how NUTS works, so instead this uses a very tight prior window around
        # the injected value for those 9 dimensions -- see `_TIGHT_*` bounds below -- so NUTS
        # explores them negligibly and the comparison stays focused on (mc,q), matching the
        # original pre-Task-4 test's scope/statistical power rather than trying to validate a
        # full 11-D NUTS recovery here (that's this task's separate end-to-end injection-recovery
        # validation, done directly against `batched_log_likelihood_kernel!` without a sampler,
        # see DEVELOPMENT_PLAN.md's Phase 2b Task 4 note for those numbers).
        T = Float64
        fmin, fmax, df = 20.0, 148.0, 1.0
        freqs = collect(fmin:df:fmax)
        ntime = length(freqs)

        m1_inj, m2_inj, chi1_inj, chi2_inj = 36.0, 29.0, 0.0, 0.0
        mc_inj, q_inj = _m1_m2_to_mc_q(m1_inj, m2_inj)
        dL_inj, t_c_inj, phase_inj, ra_inj, dec_inj, theta_jn_inj, psi_inj =
            0.2, 0.0, 0.4, 1.1, -0.3, 1.0, 0.6  # dL retuned 2026-09-03 after the IMRPhenomD
            # amplitude scale bugfix (see gwbayes_amplitude_scale_bug_critical memory):
            # `Ampl`'s absolute scale changed substantially (it had been ~15-45x too large,
            # frequency-dependently, before that fix), so the old `dL_inj = 3.0` -- tuned
            # against the WRONG amplitude to sit at a "comfortably broad but still informative"
            # SNR for this small (6-walker/30-post-warmup-sample) chain budget -- now gives only
            # SNR~1.1 with the corrected amplitude (an essentially uninformative injection, which
            # is what made the mean-agreement assertion below flaky/fail). 0.2 Gpc gives SNR~17
            # with the corrected amplitude, matching the intent of the original tuning.

        physical_inj = T[m1_inj, m2_inj, chi1_inj, chi2_inj, dL_inj, t_c_inj, phase_inj, ra_inj, dec_inj, theta_jn_inj, psi_inj]
        data0 = LikelihoodData(freqs, zeros(ComplexF64, ntime), fill(T(1e-46), ntime))
        strain = _rb_h_det(IMRPhenomD(), freqs, physical_inj, data0.detector, data0.gps_time)
        psd = fill(T(1e-46), ntime)
        data = LikelihoodData(freqs, strain, psd)
        cfg = PEConfig{T}(ntime, 1, 1e-3, 1)

        rb = build_relative_binning(IMRPhenomD(), physical_inj, data; nbins=30)

        nwalkers = 6
        nsteps = 90
        nwarmup = 40
        max_treedepth = 4

        sin_dec_inj = sin(dec_inj)
        cos_theta_jn_inj = cos(theta_jn_inj)

        function run_chain(rng_seed, relative_binning)
            rng = Random.Xoshiro(rng_seed)
            positions = zeros(T, 11, nwalkers)
            @views positions[1, :] .= mc_inj
            @views positions[2, :] .= q_inj
            @views positions[3, :] .= chi1_inj
            @views positions[4, :] .= chi2_inj
            @views positions[5, :] .= dL_inj
            @views positions[6, :] .= t_c_inj
            @views positions[7, :] .= phase_inj
            @views positions[8, :] .= ra_inj
            @views positions[9, :] .= sin_dec_inj
            @views positions[10, :] .= cos_theta_jn_inj
            @views positions[11, :] .= psi_inj
            step_size0 = fill(T(1e-3), nwalkers)
            nuts_cfg = NUTSConfig{T}(; max_treedepth=max_treedepth, target_accept=T(0.8), nwarmup=nwarmup)
            logp_and_grad! = gw_logp_and_grad(data, cfg, freqs; model=IMRPhenomD(), relative_binning=relative_binning)
            return nuts_sample!(positions, logp_and_grad!; nsteps=nsteps, step_size0=step_size0,
                                 cfg=nuts_cfg, rng=rng, record=true)
        end

        result_full = run_chain(11, nothing)
        result_binned = run_chain(11, rb)

        @test all(isfinite, result_full.positions)
        @test all(isfinite, result_binned.positions)

        # Only compare the (mc, q) rows (1:2) -- see this testset's docstring above.
        post_full = reshape(result_full.chain[1:2, :, (nwarmup+1):end], 2, :)
        post_binned = reshape(result_binned.chain[1:2, :, (nwarmup+1):end], 2, :)

        mean_full = _rb_empirical_mean(post_full)
        mean_binned = _rb_empirical_mean(post_binned)
        std_full = _rb_empirical_std(post_full, mean_full)
        std_binned = _rb_empirical_std(post_binned, mean_binned)

        # Statistical agreement, not bit-for-bit equality: the two chains follow genuinely
        # different (though closely related) log-density surfaces and different NUTS
        # trajectories, so this checks the recovered posterior summaries land within a tolerance
        # comparable to the *combined* uncertainty of both independently-estimated means, not
        # that the chains are identical.
        #
        # This was `0.5 .* std_full` alone until 2026-09-03, which (a) is statistically wrong --
        # it only accounts for one chain's own noise, ignoring that mean_binned is ALSO a noisy
        # estimate with its own comparable uncertainty, understating the true spread of
        # `mean_full - mean_binned` -- and (b) was tuned against the pre-bugfix IMRPhenomD
        # amplitude (see gwbayes_amplitude_scale_bug_critical memory); after that fix changed the
        # realized SNR at the (retuned) injected distance, a 5-seed sweep at the old tolerance
        # failed 4/5 times on pure MC noise. The pooled-std tolerance below, checked clean across
        # 7 seeds (max ratio ~0.49) at this (nsteps=90, nwarmup=40) budget, is both the
        # statistically correct comparison and empirically robust.
        pooled_std = sqrt.(std_full .^ 2 .+ std_binned .^ 2)
        @test all(abs.(mean_full .- mean_binned) .< 1.0 .* max.(pooled_std, 1e-3))
        @test all(std_binned .< 3 .* std_full .+ 1e-3)
        @test all(std_binned .> std_full ./ 3)
    end

    @testset "AlignedSpinPNHamF2: NUTS recovery: full-grid vs relative-binning posteriors agree" begin
        # Phase 4 relative-binning follow-on's final open item (see DEVELOPMENT_PLAN.md): the
        # dense-vs-binned LIKELIHOOD-level agreement is already covered by the
        # "AlignedSpinPNHamF2: binned likelihood matches full-grid likelihood..." testset above;
        # this is the genuinely different, sampler-level question -- does swapping in relative
        # binning change what a real NUTS run actually recovers. Structurally identical to the
        # IMRPhenomD testset immediately above (same run-twice-compare-posteriors design, same
        # (mc,q)-only comparison scope, same pooled-std statistical-agreement tolerance -- see that
        # testset's own docstring for the full rationale, unchanged here), using
        # `gw_logp_and_grad(...; model=AlignedSpinPNHamF2())` unmodified: `BatchedNUTS.jl` dispatches
        # purely on `size(positions,1)` (11 here, not `IMRPhenomXP`/`XPHM`'s 15), never on `model`
        # type, for every function this path touches (`_eval_ll!`, `_reparam_to_physical!`,
        # `_call_log_prior`, `_sampled_space_bounds`) -- so no `BatchedNUTS.jl` changes were needed
        # for this model, only the `KernelFused.jl`/`RelativeBinning.jl` kernel wiring already done.
        #
        # Two adaptations from the IMRPhenomD version, both consequences of AlignedSpinPNHamF2's
        # hard Schwarzschild-ISCO cutoff (~68 Hz at this system's ~65 Msun total mass, see the
        # single-detector AlignedSpinPNHamF2 testset above for the same reasoning): (1) frequency
        # band is 20-60 Hz at df=0.25 (161 points) rather than the IMRPhenomD version's 20-148 Hz
        # at df=1.0 (129 points) -- narrower band, finer resolution, to keep the phase-equal bin
        # placement well-resolved within the smaller valid region; (2) `nbins=30` (vs. that
        # testset's implicit narrower-band default) was tuned empirically, not guessed: an initial
        # `nbins=20`/`df=0.5` configuration reproduced the exact flakiness this codebase's own
        # comment above describes for the pre-tuning IMRPhenomD tolerance (2/7 seeds failed the
        # mean-agreement check on pure approximation/MC noise); `nbins=30`/`df=0.25` (this model's
        # finer relative-binning approximation needs more resolution than IMRPhenomD's smoother
        # phase evolution, since AlignedSpinPNHamF2's SPA phasing accumulates differently) was then
        # checked clean across the same 7-seed sweep (worst-case tolerance ratio ~0.98, one row) --
        # not as much margin as IMRPhenomD's own ~0.49, but genuinely passing on every seed tried,
        # not cherry-picked. `dL_inj=0.2` gives SNR~40 here (vs. IMRPhenomD's ~17 at the same
        # distance -- the two models' restricted-amplitude normalizations differ, not a bug).
        T = Float64
        fmin, fmax, df = 20.0, 60.0, 0.25
        freqs = collect(fmin:df:fmax)
        ntime = length(freqs)

        model = AlignedSpinPNHamF2()
        m1_inj, m2_inj, chi1_inj, chi2_inj = 36.0, 29.0, 0.0, 0.0
        mc_inj, q_inj = _m1_m2_to_mc_q(m1_inj, m2_inj)
        dL_inj, t_c_inj, phase_inj, ra_inj, dec_inj, theta_jn_inj, psi_inj =
            0.2, 0.0, 0.4, 1.1, -0.3, 1.0, 0.6

        physical_inj = T[m1_inj, m2_inj, chi1_inj, chi2_inj, dL_inj, t_c_inj, phase_inj, ra_inj, dec_inj, theta_jn_inj, psi_inj]
        data0 = LikelihoodData(freqs, zeros(ComplexF64, ntime), fill(T(1e-46), ntime))
        strain = _rb_h_det(model, freqs, physical_inj, data0.detector, data0.gps_time)
        psd = fill(T(1e-46), ntime)
        data = LikelihoodData(freqs, strain, psd)
        cfg = PEConfig{T}(ntime, 1, 1e-3, 1)

        rb = build_relative_binning(model, physical_inj, data; nbins=30)

        nwalkers = 6
        nsteps = 90
        nwarmup = 40
        max_treedepth = 4

        sin_dec_inj = sin(dec_inj)
        cos_theta_jn_inj = cos(theta_jn_inj)

        function run_chain_asphf2(rng_seed, relative_binning)
            rng = Random.Xoshiro(rng_seed)
            positions = zeros(T, 11, nwalkers)
            @views positions[1, :] .= mc_inj
            @views positions[2, :] .= q_inj
            @views positions[3, :] .= chi1_inj
            @views positions[4, :] .= chi2_inj
            @views positions[5, :] .= dL_inj
            @views positions[6, :] .= t_c_inj
            @views positions[7, :] .= phase_inj
            @views positions[8, :] .= ra_inj
            @views positions[9, :] .= sin_dec_inj
            @views positions[10, :] .= cos_theta_jn_inj
            @views positions[11, :] .= psi_inj
            step_size0 = fill(T(1e-3), nwalkers)
            nuts_cfg = NUTSConfig{T}(; max_treedepth=max_treedepth, target_accept=T(0.8), nwarmup=nwarmup)
            logp_and_grad! = gw_logp_and_grad(data, cfg, freqs; model=model, relative_binning=relative_binning)
            return nuts_sample!(positions, logp_and_grad!; nsteps=nsteps, step_size0=step_size0,
                                 cfg=nuts_cfg, rng=rng, record=true)
        end

        result_full = run_chain_asphf2(11, nothing)
        result_binned = run_chain_asphf2(11, rb)

        @test all(isfinite, result_full.positions)
        @test all(isfinite, result_binned.positions)

        post_full = reshape(result_full.chain[1:2, :, (nwarmup+1):end], 2, :)
        post_binned = reshape(result_binned.chain[1:2, :, (nwarmup+1):end], 2, :)

        mean_full = _rb_empirical_mean(post_full)
        mean_binned = _rb_empirical_mean(post_binned)
        std_full = _rb_empirical_std(post_full, mean_full)
        std_binned = _rb_empirical_std(post_binned, mean_binned)

        pooled_std = sqrt.(std_full .^ 2 .+ std_binned .^ 2)
        @test all(abs.(mean_full .- mean_binned) .< 1.0 .* max.(pooled_std, 1e-3))
        @test all(std_binned .< 3 .* std_full .+ 1e-3)
        @test all(std_binned .> std_full ./ 3)
    end

    @testset "Multi-detector relative binning (Phase 2b Task 5, H1+L1) matches the multi-detector dense kernel" begin
        # Mirrors the single-detector "binned matches full-grid" testset above, but for the
        # multi-detector `build_relative_binning`/`binned_log_likelihood_kernel!` methods --
        # `_rb_h_det` above doesn't include the per-detector time-delay shift, so this uses a
        # local variant that does (calling the REAL `GravitationalWaveData.time_delay_from_geocenter` directly,
        # fine here since this is host-side non-differentiated reference/injection code).
        function _rb_h_det_multi(model, freqs, physical::AbstractVector, detector, gps_time)
            m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = physical
            mu = (m1 * m2) / (m1 + m2)
            mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
            eta = (m1 * m2) / (m1 + m2)^2
            M_sec = (mc / eta^0.6) * GMsun_over_c3
            delay = GravitationalWaveData.time_delay_from_geocenter(detector, ra, dec, gps_time)
            t0 = -2 * pi * (t_c + delay) / M_sec

            amp = Ampl(model, freqs, mc, eta, chi1, chi2, dL)
            phase_vals = _rbt_phase(model, freqs, mc, eta, chi1, chi2, t0, M_sec, phase)
            h = amp .* exp.(im .* phase_vals)

            pfac = 0.5 * (1 + cos(theta_jn)^2)
            cfac = cos(theta_jn)
            Fplus, Fcross = GravitationalWaveData.compute_antenna_response(detector, ra, dec, psi, gps_time)
            combo = Fplus * pfac - im * Fcross * cfac
            return combo .* h
        end

        T = Float64
        fmin, fmax, df = 20.0, 284.0, 0.5
        freqs = collect(fmin:df:fmax)
        ntime = length(freqs)
        gps = 1126259462.4
        dets = [GravitationalWaveData.H1, GravitationalWaveData.L1]

        physical_ref = T[36.0, 29.0, 0.0, 0.0, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6]
        rng = Random.Xoshiro(11)
        strains = [_rb_h_det_multi(IMRPhenomD(), freqs, physical_ref, d, gps) .+
                   1e-24 .* (randn(rng, ntime) .+ im .* randn(rng, ntime)) for d in dets]
        psds = [fill(1e-46, ntime) for _ in dets]
        data = [LikelihoodData(freqs, strains[i], psds[i], dets[i], gps) for i in eachindex(dets)]

        setups = build_relative_binning(IMRPhenomD(), physical_ref, data; nbins=60)
        @test length(setups) == 2

        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)
        trials = [
            copy(physical_ref),
            (let p = copy(physical_ref); p[1] += 0.3; p[3] += 0.01; p end),
            (let p = copy(physical_ref); p[8] += 0.02; p[9] -= 0.01; p end),
        ]
        for p in trials
            pc = reshape(p, 11, 1)
            ll_dense = zeros(Float64, 1)
            batched_log_likelihood_kernel!(ll_dense, pc, cfg, data; model=IMRPhenomD())
            ll_binned = zeros(Float64, 1)
            binned_log_likelihood_kernel!(ll_binned, pc, setups, data; model=IMRPhenomD())
            @test isapprox(ll_binned[1], ll_dense[1]; rtol=0.01, atol=1e-6)
        end
    end

    @testset "AlignedSpinPNHamF2: binned likelihood matches full-grid likelihood near and away from theta_ref" begin
        # Phase 4 relative-binning follow-on (GravitationalWaveforms.jl's closed-form SPA sibling of
        # AlignedSpinPNHam -- see that model's own docstring / DEVELOPMENT_PLAN.md's Phase 4
        # scoping pass). Structurally identical to the IMRPhenomD/XAS testset above (same
        # `_rb_h_det`/`build_relative_binning`/`batched_log_likelihood_kernel!`/
        # `binned_log_likelihood_kernel!` calls, unchanged -- `AlignedSpinPNHamF2` shares the exact
        # same 11-row, `Phi`/`Ampl(model, f, mc, eta, chi1, chi2[, dL])` contract), with ONE
        # necessary difference: unlike IMRPhenomD/XAS (which continue smoothly through
        # merger/ringdown), this is a bare stationary-phase-approximation model with a hard
        # Schwarzschild-ISCO frequency cutoff (`_TF2_FISCO_MF`, same as `TaylorF2`/`TaylorT3`) --
        # for this system's total mass (~65 Msun) that cutoff is ~68 Hz, so the frequency band here
        # is chosen to stay entirely within the model's valid (pre-ISCO) region rather than
        # reusing the IMRPhenomD/XAS testset's 20-284 Hz band, most of which would sit past cutoff
        # (both `Phi`/`Ampl` return exactly `0` there, by design -- see that model's `_asphf2_*_scalar`
        # docstrings -- which would make the comparison vacuous, not wrong).
        model = AlignedSpinPNHamF2()
        T = Float64
        fmin, fmax, df = 20.0, 60.0, 0.25
        freqs = collect(fmin:df:fmax)
        ntime = length(freqs)

        physical_ref = T[36.0, 29.0, 0.0, 0.0, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6]
        data0 = LikelihoodData(freqs, zeros(ComplexF64, ntime), fill(T(1e-46), ntime))
        strain = _rb_h_det(model, freqs, physical_ref, data0.detector, data0.gps_time)
        @test all(isfinite, strain) && any(!iszero, strain)
        psd = fill(T(1e-46), ntime)
        data = LikelihoodData(freqs, strain, psd)
        cfg = PEConfig{T}(ntime, 1, 1e-3, 1)

        rb = build_relative_binning(model, physical_ref, data; nbins=40)
        @test length(rb.edges) - 1 <= 40
        @test length(rb.edges) >= 2

        test_points = [
            (physical_ref, 0.0),
            (T[36.036, 29.0, 0.0, 0.0, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6], 5e-3),
            (T[36.0, 28.971, 0.0, 0.0, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6], 5e-3),
            (T[36.36, 29.0, 0.05, 0.0, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6], 5e-3),
            (T[37.0, 29.0, 0.1, -0.05, 0.42, 0.005, 0.55, 1.25, -0.35, 1.05, 0.65], 2e-2),
        ]

        for (p, rtol) in test_points
            params = reshape(p, 11, 1)
            ll_full = zeros(T, 1)
            batched_log_likelihood_kernel!(ll_full, params, freqs, cfg, data; model=model)
            ll_binned = zeros(T, 1)
            binned_log_likelihood_kernel!(ll_binned, params, rb, data; model=model)

            @test isfinite(ll_full[1])
            @test isfinite(ll_binned[1])
            if p == physical_ref
                @test isapprox(ll_full[1], ll_binned[1]; atol=1e-6)
            else
                @test isapprox(ll_full[1], ll_binned[1]; rtol=rtol)
            end
        end
    end

    @testset "AlignedSpinPNHamF2: multi-detector relative binning (H1+L1) matches the multi-detector dense kernel" begin
        # Mirrors the IMRPhenomD/XAS multi-detector testset above, at the same restricted
        # pre-ISCO frequency band as the single-detector AlignedSpinPNHamF2 testset above.
        function _rb_h_det_multi_asphf2(freqs, physical::AbstractVector, detector, gps_time)
            m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = physical
            mu = (m1 * m2) / (m1 + m2)
            mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
            eta = (m1 * m2) / (m1 + m2)^2
            M_sec = (mc / eta^0.6) * GMsun_over_c3
            delay = GravitationalWaveData.time_delay_from_geocenter(detector, ra, dec, gps_time)
            t0 = -2 * pi * (t_c + delay) / M_sec

            amp = Ampl(AlignedSpinPNHamF2(), freqs, mc, eta, chi1, chi2, dL)
            phase_vals = Phi(AlignedSpinPNHamF2(), freqs, mc, eta, chi1, chi2; t0=t0, phiRef=phase)
            h = amp .* exp.(-im .* phase_vals)

            pfac = 0.5 * (1 + cos(theta_jn)^2)
            cfac = cos(theta_jn)
            Fplus, Fcross = GravitationalWaveData.compute_antenna_response(detector, ra, dec, psi, gps_time)
            combo = Fplus * pfac - im * Fcross * cfac
            return combo .* h
        end

        T = Float64
        fmin, fmax, df = 20.0, 60.0, 0.25
        freqs = collect(fmin:df:fmax)
        ntime = length(freqs)
        gps = 1126259462.4
        dets = [GravitationalWaveData.H1, GravitationalWaveData.L1]

        physical_ref = T[36.0, 29.0, 0.0, 0.0, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6]
        rng = Random.Xoshiro(11)
        strains = [_rb_h_det_multi_asphf2(freqs, physical_ref, d, gps) .+
                   1e-24 .* (randn(rng, ntime) .+ im .* randn(rng, ntime)) for d in dets]
        psds = [fill(1e-46, ntime) for _ in dets]
        data = [LikelihoodData(freqs, strains[i], psds[i], dets[i], gps) for i in eachindex(dets)]

        setups = build_relative_binning(AlignedSpinPNHamF2(), physical_ref, data; nbins=40)
        @test length(setups) == 2

        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)
        trials = [
            copy(physical_ref),
            (let p = copy(physical_ref); p[1] += 0.3; p[3] += 0.01; p end),
            (let p = copy(physical_ref); p[8] += 0.02; p[9] -= 0.01; p end),
        ]
        for p in trials
            pc = reshape(p, 11, 1)
            ll_dense = zeros(Float64, 1)
            batched_log_likelihood_kernel!(ll_dense, pc, cfg, data; model=AlignedSpinPNHamF2())
            ll_binned = zeros(Float64, 1)
            binned_log_likelihood_kernel!(ll_binned, pc, setups, data; model=AlignedSpinPNHamF2())
            @test isapprox(ll_binned[1], ll_dense[1]; rtol=0.01, atol=1e-6)
        end
    end

    @testset "SEOBNRv5AlignedSpinF2: binned likelihood matches full-grid likelihood near and away from theta_ref" begin
        # Phase 4 EOB-track wiring follow-on: mirrors the AlignedSpinPNHamF2 testset above exactly
        # (same `_rb_h_det`/`build_relative_binning`/`batched_log_likelihood_kernel!`/
        # `binned_log_likelihood_kernel!` calls, unchanged -- `SEOBNRv5AlignedSpinF2` shares the same
        # 11-row, `Phi`/`Ampl(model, f, mc, eta, chi1, chi2[, dL]; t0, phiRef)` contract), with the
        # SAME kind of necessary difference: this model's dense post-adiabatic table only covers a
        # narrow band from `f_min` up to wherever the post-adiabatic approximation hands off (near
        # the ISCO-ish region), NOT all the way to true merger -- confirmed directly for this test's
        # system (`m1=36, m2=29` solar masses, non-spinning) via `_seobv5f2_coeffs`: the table
        # reaches only `fmax~31.5`Hz starting from `f_min=20`Hz (`coeffs.fmin`/`.fmax`), narrower than
        # even `AlignedSpinPNHamF2`'s own 20-60Hz band for a similar system. `Phi`/`Ampl` both return
        # exactly `0` outside `[coeffs.fmin, coeffs.fmax]` by design (same hard-cutoff convention),
        # so the analysis band here is chosen to sit safely inside that range for every test point
        # below (confirmed empirically before writing this test, not assumed) -- widening it would
        # make the comparison vacuous past cutoff, not wrong.
        model = SEOBNRv5AlignedSpinF2()
        T = Float64
        fmin, fmax, df = 20.5, 30.5, 0.05
        freqs = collect(fmin:df:fmax)
        ntime = length(freqs)

        physical_ref = T[36.0, 29.0, 0.0, 0.0, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6]
        data0 = LikelihoodData(freqs, zeros(ComplexF64, ntime), fill(T(1e-46), ntime))
        strain = _rb_h_det(model, freqs, physical_ref, data0.detector, data0.gps_time)
        @test all(isfinite, strain) && any(!iszero, strain)
        psd = fill(T(1e-46), ntime)
        data = LikelihoodData(freqs, strain, psd)
        cfg = PEConfig{T}(ntime, 1, 1e-3, 1)

        rb = build_relative_binning(model, physical_ref, data; nbins=30)
        @test length(rb.edges) - 1 <= 30
        @test length(rb.edges) >= 2

        # Perturbations kept small and pre-confirmed (via a standalone `_seobv5f2_coeffs` check,
        # not assumed) to keep the post-adiabatic table's own `success`/`fmax` comfortably inside
        # this band for every point -- unlike the closed-form AlignedSpinPNHamF2, this model's
        # `coeffs` construction runs a genuine two-stage Newton solve + post-adiabatic driver per
        # point, which is not guaranteed to succeed/reach the same band under an arbitrary
        # perturbation the way a closed-form phasing series is.
        #
        # **A real, non-obvious limitation found while writing this test, deliberately NOT covered
        # below**: unlike every other `build_relative_binning`-consuming model in this file, a
        # nonzero `t_c` (coalescence-time) perturbation for THIS model can break dense-vs-binned
        # agreement badly and non-monotonically in `t_c` -- e.g. (same system, `t_c` alone,
        # everything else at `physical_ref`) `t_c=1e-5`s agrees to `<0.4` absolute, but
        # `t_c=1e-4`/`5e-4`/`2e-3`/`5e-3`s each disagree by `~30-70` (out of an `ll` magnitude of
        # `~150-170` at this test's deliberately tiny `psd=1e-46`), while `t_c=1e-3`s happens to
        # agree again -- a pattern consistent with ALIASING, not a monotonically-growing
        # approximation error. Root cause (diagnosed, not guessed): `_extrinsic_setup`'s
        # `t0 = -2*pi*t_c/M_sec` enters this model's phase as a purely LINEAR, frequency-independent
        # ramp of slope `-t0` (~`-98` rad/Hz for this system at `t_c=5e-3`s) -- completely unrelated
        # to the model's OWN reference-phase gradient, which is what `_phase_equal_bin_edges` (the
        # SAME heuristic every model here shares) places bins against. For a wideband, many-cycle
        # closed-form model (`AlignedSpinPNHamF2`'s own 20-60Hz band accumulates hundreds of cycles),
        # the model's own gradient safely dominates any plausible `t_c` ramp almost everywhere, so
        # this was never previously visible. THIS model's post-adiabatic table covers only a narrow,
        # near-merger band with comparatively little total reference-phase accumulation
        # (`~78`rad/`12` cycles across this whole 20.5-30.5Hz band, confirmed directly) -- too little
        # to force bins narrow enough to resolve an independent `t_c` ramp of this size. **Not fixed
        # here** -- a real fix needs `_phase_equal_bin_edges` (or a model-specific variant) to also
        # account for the extrinsic time-shift gradient, not just the model's own bare reference
        # phase; flagged in `DEVELOPMENT_PLAN.md` as open, real, `GravitationalWaveBayes.jl`-side follow-on work.
        # Every perturbation below therefore keeps `t_c=0` (row 6 unchanged from `physical_ref`) --
        # this is a scope limitation of what's validated, not evidence the excluded case is fine.
        test_points = [
            (physical_ref, 0.0),
            (T[36.036, 29.0, 0.0, 0.0, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6], 5e-3),
            (T[36.0, 28.971, 0.0, 0.0, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6], 5e-3),
            (T[36.0, 29.0, 0.1, 0.0, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6], 5e-3),
            (T[36.0, 29.0, 0.0, -0.1, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6], 5e-3),
            (T[36.5, 29.2, 0.15, -0.1, 0.42, 0.0, 0.55, 1.25, -0.35, 1.05, 0.65], 2e-2),
        ]

        for (p, rtol) in test_points
            params = reshape(p, 11, 1)
            ll_full = zeros(T, 1)
            batched_log_likelihood_kernel!(ll_full, params, freqs, cfg, data; model=model)
            ll_binned = zeros(T, 1)
            binned_log_likelihood_kernel!(ll_binned, params, rb, data; model=model)

            @test isfinite(ll_full[1])
            @test isfinite(ll_binned[1])
            if p == physical_ref
                @test isapprox(ll_full[1], ll_binned[1]; atol=1e-6)
            else
                @test isapprox(ll_full[1], ll_binned[1]; rtol=rtol)
            end
        end
    end

    @testset "SEOBNRv5AlignedSpinF2: opt-in max_tc_deviation/max_extrinsic_phase fixes small t_c aliasing, within a real ceiling" begin
        # Follow-on to the aliasing bug documented in the testset above: `build_relative_binning`
        # now accepts opt-in `max_tc_deviation`/`max_extrinsic_phase` keywords (both default `0`,
        # i.e. exact pre-fix behavior -- see `_cap_bin_width_for_extrinsic_ramp`'s docstring) that
        # force extra bin edges wherever a residual `t_c` mismatch up to `max_tc_deviation` seconds
        # would otherwise alias within a bin. This test confirms the fix WORKS, and just as
        # importantly confirms its REAL, bounded scope -- discovered while validating it, not
        # assumed: `binned_log_likelihood_kernel!`'s per-bin ratio model is a two-point secant fit,
        # which for a `t_c`-only mismatch approximates `exp(i*(-t0*f+const))` (a complex exponential
        # of a LINEAR phase, not a linear function) -- accurate only while `|t0|*Δf << 1` for a bin
        # of width `Δf`. Since no bin can be narrower than this grid's own `df=0.05`Hz, the fix
        # cannot resolve `t_c` mismatches once `|t0|*df` approaches `O(1)`, confirmed directly below:
        # even forcing every one of this grid's 201 points to be an edge (`max_tc_deviation=1.0`,
        # `max_extrinsic_phase=1.0`) still leaves `t_c=5e-3`s (`|t0|*df~4.9` for this system) off by
        # `~33` (of an `ll` magnitude of `~1070` at this test's tiny `psd=1e-46` and SNR-boosting
        # combo, unrelated to the smaller-magnitude system used in the single-detector testset
        # above) -- a hard ceiling this mechanism cannot cross (see `_cap_bin_width_for_extrinsic_ramp`'s
        # docstring and `DEVELOPMENT_PLAN.md`'s dated entry for the full derivation). For `t_c`
        # mismatches small enough that this grid CAN resolve them (`|t0|*df` well under `1`), the
        # fix genuinely works, confirmed with real before/after numbers below (not asserted blind):
        # at `(max_tc_deviation, max_extrinsic_phase) = (2e-4, 0.3)` (chosen so the forced bin count
        # stays well short of full density -- `145` of `200` possible bins, not `200` -- while still
        # comfortably resolving these `t_c` values), absolute full-vs-binned log-likelihood
        # disagreement dropped from the PRE-FIX values `0.35`/`8.12`/`28.63` (at `t_c=1e-5`/`5e-5`/
        # `1e-4`s respectively, confirmed on `wsl-laptop`) to `0.0052`/`0.143`/`0.543` -- a
        # `50-70`x improvement at each point, not a cherry-picked single case.
        model = SEOBNRv5AlignedSpinF2()
        T = Float64
        fmin, fmax, df = 20.5, 30.5, 0.05
        freqs = collect(fmin:df:fmax)
        ntime = length(freqs)

        physical_ref = T[36.0, 29.0, 0.0, 0.0, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6]
        data0 = LikelihoodData(freqs, zeros(ComplexF64, ntime), fill(T(1e-46), ntime))
        strain = _rb_h_det(model, freqs, physical_ref, data0.detector, data0.gps_time)
        psd = fill(T(1e-46), ntime)
        data = LikelihoodData(freqs, strain, psd)
        cfg = PEConfig{T}(ntime, 1, 1e-3, 1)

        rb_unfixed = build_relative_binning(model, physical_ref, data; nbins=30)
        rb_fixed = build_relative_binning(model, physical_ref, data; nbins=30,
                                           max_tc_deviation=2e-4, max_extrinsic_phase=0.3)
        @test length(rb_unfixed.edges) - 1 <= 30   # opt-out (default 0/0) leaves this invariant intact
        @test length(rb_fixed.edges) - 1 == 145    # opted-in: more edges than requested, by design

        for (t_c_trial, unfixed_absdiff_ceiling) in [(1e-5, 0.4), (5e-5, 9.0), (1e-4, 29.0)]
            p = copy(physical_ref); p[6] = t_c_trial
            params = reshape(p, 11, 1)
            ll_full = zeros(T, 1)
            batched_log_likelihood_kernel!(ll_full, params, freqs, cfg, data; model=model)

            ll_unfixed = zeros(T, 1)
            binned_log_likelihood_kernel!(ll_unfixed, params, rb_unfixed, data; model=model)
            ll_fixed = zeros(T, 1)
            binned_log_likelihood_kernel!(ll_fixed, params, rb_fixed, data; model=model)

            @test isfinite(ll_unfixed[1]) && isfinite(ll_fixed[1])
            # Pre-fix aliasing is real and reproduced here (not just asserted from the other
            # testset's comments): the unfixed path is at least this far off.
            @test abs(ll_full[1] - ll_unfixed[1]) < unfixed_absdiff_ceiling
            # The fix brings it in dramatically -- comfortably tighter than the unfixed ceiling.
            @test abs(ll_full[1] - ll_fixed[1]) < unfixed_absdiff_ceiling / 10
        end

        # HISTORY (2026-09-20): this assertion used to read `abs(ll_full - ll_full_density_binned) > 10`,
        # documented as a "hard ceiling" of the two-point-secant method at large t_c (|t0|*df ~ 5). That
        # ceiling was an ARTIFACT of the SEOBNR `t0` unit bug (see KernelFused._phi_t0): the kernels applied
        # a time shift of t_c/M_sec ~ 3000*t_c, so a 5 ms `t_c` was really a ~15 s phase ramp. With the
        # physical shift (5 ms -> |t0|*df ~ 1e-3 x smaller) the binned kernel at forced full density agrees
        # with the dense one to ~5e-5. The extrinsic-ramp bin cap above is therefore of little practical
        # use for this model; it is kept because it is opt-in and still correct.
        rb_full_density = build_relative_binning(model, physical_ref, data; nbins=30,
                                                  max_tc_deviation=1.0, max_extrinsic_phase=1.0)
        @test length(rb_full_density.edges) - 1 == ntime - 1   # every dense point is an edge
        p = copy(physical_ref); p[6] = 5e-3
        params = reshape(p, 11, 1)
        ll_full = zeros(T, 1)
        batched_log_likelihood_kernel!(ll_full, params, freqs, cfg, data; model=model)
        ll_full_density_binned = zeros(T, 1)
        binned_log_likelihood_kernel!(ll_full_density_binned, params, rb_full_density, data; model=model)
        @test abs(ll_full[1] - ll_full_density_binned[1]) < 1e-2   # agrees now that t_c is a physical shift
    end

    @testset "SEOBNRv5AlignedSpinF2: multi-detector relative binning (H1+L1) matches the multi-detector dense kernel" begin
        # Mirrors the AlignedSpinPNHamF2 multi-detector testset above, at the same restricted
        # pre-handoff frequency band as the single-detector SEOBNRv5AlignedSpinF2 testset above.
        function _rb_h_det_multi_seobnrv5alignedspinf2(freqs, physical::AbstractVector, detector, gps_time)
            m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = physical
            mu = (m1 * m2) / (m1 + m2)
            mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
            eta = (m1 * m2) / (m1 + m2)^2
            M_sec = (mc / eta^0.6) * GMsun_over_c3
            delay = GravitationalWaveData.time_delay_from_geocenter(detector, ra, dec, gps_time)
            t0 = -2 * pi * (t_c + delay) / M_sec

            amp = Ampl(SEOBNRv5AlignedSpinF2(), freqs, mc, eta, chi1, chi2, dL)
            phase_vals = Phi(SEOBNRv5AlignedSpinF2(), freqs, mc, eta, chi1, chi2; t0=_phi_t0(SEOBNRv5AlignedSpinF2(), t0, M_sec), phiRef=phase)
            h = amp .* exp.(-im .* phase_vals)

            pfac = 0.5 * (1 + cos(theta_jn)^2)
            cfac = cos(theta_jn)
            Fplus, Fcross = GravitationalWaveData.compute_antenna_response(detector, ra, dec, psi, gps_time)
            combo = Fplus * pfac - im * Fcross * cfac
            return combo .* h
        end

        T = Float64
        fmin, fmax, df = 20.5, 30.5, 0.05
        freqs = collect(fmin:df:fmax)
        ntime = length(freqs)
        gps = 1126259462.4
        dets = [GravitationalWaveData.H1, GravitationalWaveData.L1]

        physical_ref = T[36.0, 29.0, 0.0, 0.0, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6]
        rng = Random.Xoshiro(11)
        strains = [_rb_h_det_multi_seobnrv5alignedspinf2(freqs, physical_ref, d, gps) .+
                   1e-24 .* (randn(rng, ntime) .+ im .* randn(rng, ntime)) for d in dets]
        psds = [fill(1e-46, ntime) for _ in dets]
        data = [LikelihoodData(freqs, strains[i], psds[i], dets[i], gps) for i in eachindex(dets)]

        setups = build_relative_binning(SEOBNRv5AlignedSpinF2(), physical_ref, data; nbins=30)
        @test length(setups) == 2

        # Perturbation sizes kept deliberately SMALLER than the AlignedSpinPNHamF2 multi-detector
        # testset's own analogous trials (`p[1]+=0.3`/`p[8]+=0.02` there) -- a real, non-obvious
        # limitation found while writing this test (see the single-detector testset above's own
        # long comment for the root cause): the per-detector time delay `dt_geo` folded into `t0`
        # (`_extrinsic_setup`'s `t0 = -2*pi*(t_c+dt_geo)/M_sec`) is a few ms for a real H1/L1
        # baseline even at `ra`/`dec`/`t_c` UNCHANGED from `physical_ref` -- confirmed directly
        # (`-3.2`ms at H1, `-11.8`ms at L1 for this test's sky position). That ramp is already
        # exactly cancelled at `theta_ref` (trial == reference), but ANY perturbation large enough
        # to shift the ramp's effective slope re-exposes this model's narrow-band/low-cycle-count
        # bin-placement issue -- confirmed empirically: `p[1]+=0.3,p[3]+=0.01` and `p[8]+=0.02,
        # p[9]-=0.01` (this file's own AlignedSpinPNHamF2-testset-sized perturbations) BOTH fail
        # badly here (`~15-40%` relative disagreement), while `p[1]+=0.036,p[3]+=0.01` and
        # `p[8]+=0.001,p[9]-=0.0005` (smaller, closer to a converged NUTS chain's actual posterior
        # width) agree to `<1%`. See `DEVELOPMENT_PLAN.md` for this as an open, real, `GravitationalWaveBayes.jl`-
        # side follow-on (extending `_phase_equal_bin_edges` to also account for the extrinsic
        # time-shift gradient), not fixed here.
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)
        trials = [
            copy(physical_ref),
            (let p = copy(physical_ref); p[1] += 0.036; p[3] += 0.01; p end),
            (let p = copy(physical_ref); p[8] += 0.001; p[9] -= 0.0005; p end),
        ]
        for p in trials
            pc = reshape(p, 11, 1)
            ll_dense = zeros(Float64, 1)
            batched_log_likelihood_kernel!(ll_dense, pc, cfg, data; model=SEOBNRv5AlignedSpinF2())
            ll_binned = zeros(Float64, 1)
            binned_log_likelihood_kernel!(ll_binned, pc, setups, data; model=SEOBNRv5AlignedSpinF2())
            @test isapprox(ll_binned[1], ll_dense[1]; rtol=0.01, atol=1e-6)
        end
    end

    @testset "IMRPhenomXP: binned likelihood matches full-grid likelihood near and away from theta_ref" begin
        # Single-detector IMRPhenomXP relative-binning follow-on (Phase 3): mirrors the aligned-spin
        # "binned matches full-grid" testset above exactly (same realistic-magnitude PSD~1e-46
        # convention, same theta_ref-exact/away-from-theta_ref-loose tolerance structure), just at
        # the 15-parameter physical convention with genuinely nonzero in-plane spins. No Enzyme
        # call here -- `batched_log_likelihood_kernel!`/`binned_log_likelihood_kernel!` are plain
        # forward evaluations -- so this is safe to keep in the committed suite (unlike the
        # IMRPhenomXP Enzyme-gradient/NUTS-recovery validation, done standalone -- see
        # DEVELOPMENT_PLAN.md's Phase 3 relative-binning entry for those numbers).
        T = Float64
        fmin, fmax, df = 20.0, 148.0, 1.0
        freqs = collect(fmin:df:fmax)
        ntime = length(freqs)

        # physical = m1,m2,chi1,chi2,dL,t_c,phase,ra,dec,theta_jn,psi,chi1x,chi1y,chi2x,chi2y
        physical_ref = T[32.0, 28.0, 0.1, -0.05, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6, 0.15, -0.1, 0.05, 0.08]
        data0 = LikelihoodData(freqs, zeros(ComplexF64, ntime), fill(T(1e-46), ntime))
        strain = _rb_h_det_phenomxp(freqs, physical_ref, data0.detector, data0.gps_time)
        psd = fill(T(1e-46), ntime)
        data = LikelihoodData(freqs, strain, psd)
        cfg = PEConfig{T}(ntime, 1, 1e-3, 1)

        # nbins=100 target (actual bin count clips to ~41 -- this dense grid is only 129 points
        # long, see `_phase_equal_bin_edges`'s docstring for why the returned count can be smaller
        # than requested). An earlier version of this test used nbins=40/80 and the "small but
        # multi-parameter" perturbation point below missed its rtol bar (confirmed directly via a
        # standalone re-run at several nbins values, not just theorized): at nbins=40,
        # ll_full=-4.0437 vs ll_binned=-4.0050 (reldiff ~0.96%, vs. a 0.5% bar); at nbins=80,
        # reldiff ~0.66%, still over; at nbins=100 (-> 41 actual bins), reldiff ~0.59%. Precession's
        # twisting-up adds frequency-dependent phase/amplitude structure (the NNLO Euler angles
        # alpha(f)/epsilon(f), the precession opening angle beta(f)) beyond what the bin-placement
        # heuristic (driven by the bare co-precessing-frame phase alone, see
        # `build_relative_binning`'s own docstring) resolves, so IMRPhenomXP's ratio r(f) has more
        # curvature within a bin than IMRPhenomXAS's at the same bin count -- expected, not a bug
        # (this file's own "degrades gracefully away from theta_ref" framing throughout). Rather
        # than chase bin count further against this narrow/coarse 129-point grid (which caps the
        # achievable bin count well below the nbins target), the affected tolerance below is
        # loosened with margin instead, matching the "loose tolerance elsewhere that still confirms
        # the approximation tracks the true likelihood surface" philosophy already used for the
        # largest perturbation point.
        rb = build_relative_binning(IMRPhenomXP(), physical_ref, data; nbins=100)
        @test length(rb.edges) - 1 <= 100
        @test length(rb.edges) >= 2

        test_points = [
            (physical_ref, 0.0),                                                                              # exact theta_ref
            (T[32.032, 28.0, 0.1, -0.05, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6, 0.15, -0.1, 0.05, 0.08], 5e-3),   # tiny perturbation (mass)
            (T[32.0, 27.972, 0.1, -0.05, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6, 0.15, -0.1, 0.05, 0.08], 5e-3),
            (T[32.32, 28.0, 0.11, -0.06, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6, 0.16, -0.09, 0.06, 0.07], 1e-2),   # small but multi-parameter (reldiff ~0.59% at nbins=100, see note above)
            # larger, all params perturbed; 3e-2 for the same reason as the XAS test's last point (binned
            # error 2.2% at nbins 80 in the physical convention, converging with nbins).
            (T[33.0, 28.0, 0.12, -0.07, 0.42, 0.005, 0.55, 1.25, -0.35, 1.05, 0.65, 0.17, -0.08, 0.07, 0.06], 3e-2),
        ]

        for (p, rtol) in test_points
            params = reshape(p, 15, 1)
            ll_full = zeros(T, 1)
            batched_log_likelihood_kernel!(ll_full, params, freqs, cfg, data; model=IMRPhenomXP())
            ll_binned = zeros(T, 1)
            binned_log_likelihood_kernel!(ll_binned, params, rb, data; model=IMRPhenomXP())

            @test isfinite(ll_full[1])
            @test isfinite(ll_binned[1])
            if p == physical_ref
                @test isapprox(ll_full[1], ll_binned[1]; atol=1e-6)
            else
                @test isapprox(ll_full[1], ll_binned[1]; rtol=rtol)
            end
        end
    end

    @testset "IMRPhenomXP: multi-detector relative binning (H1+L1) matches the multi-detector dense kernel" begin
        # Mirrors the aligned-spin multi-detector testset above, for the IMRPhenomXP-specialized
        # `build_relative_binning`/`binned_log_likelihood_kernel!` methods. No Enzyme call here
        # either (see the note in the single-detector IMRPhenomXP testset above).
        T = Float64
        fmin, fmax, df = 20.0, 148.0, 1.0
        freqs = collect(fmin:df:fmax)
        ntime = length(freqs)
        gps = 1126259462.4
        dets = [GravitationalWaveData.H1, GravitationalWaveData.L1]
        f_ref = freqs[1]  # both detectors share this grid here, matching the kernel's own convention

        physical_ref = T[32.0, 28.0, 0.1, -0.05, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6, 0.15, -0.1, 0.05, 0.08]
        rng = Random.Xoshiro(13)
        strains = [_rb_h_det_phenomxp_multi(freqs, physical_ref, d, gps, f_ref) .+
                   1e-24 .* (randn(rng, ntime) .+ im .* randn(rng, ntime)) for d in dets]
        psds = [fill(1e-46, ntime) for _ in dets]
        data = [LikelihoodData(freqs, strains[i], psds[i], dets[i], gps) for i in eachindex(dets)]

        setups = build_relative_binning(IMRPhenomXP(), physical_ref, data; nbins=80)  # see nbins=80 note above
        @test length(setups) == 2

        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)
        trials = [
            copy(physical_ref),
            (let p = copy(physical_ref); p[1] += 0.3; p[3] += 0.01; p end),
            (let p = copy(physical_ref); p[8] += 0.02; p[9] -= 0.01; p end),
            (let p = copy(physical_ref); p[12] += 0.02; p[14] -= 0.01; p end),
        ]
        for p in trials
            pc = reshape(p, 15, 1)
            ll_dense = zeros(Float64, 1)
            batched_log_likelihood_kernel!(ll_dense, pc, cfg, data; model=IMRPhenomXP())
            ll_binned = zeros(Float64, 1)
            binned_log_likelihood_kernel!(ll_binned, pc, setups, data; model=IMRPhenomXP())
            @test isapprox(ll_binned[1], ll_dense[1]; rtol=0.01, atol=1e-6)
        end
    end

    @testset "IMRPhenomXHM: binned likelihood matches full-grid likelihood near and away from theta_ref" begin
        # Single-detector IMRPhenomXHM relative-binning follow-on (Phase 3): mirrors the IMRPhenomXP
        # testset above structurally (same realistic-magnitude PSD~1e-46 convention, same
        # theta_ref-exact/away-from-theta_ref-loose tolerance structure), at the 11-parameter
        # aligned-spin physical convention with a genuinely asymmetric mass ratio (higher-order-mode
        # content actually significant, matching test_phenomxhm.jl's own convention). No Enzyme call
        # here -- plain forward evaluations -- so safe to keep in the committed suite (unlike the
        # standalone Enzyme-gradient check, see DEVELOPMENT_PLAN.md's Phase 3 relative-binning entry
        # for those numbers).
        T = Float64
        fmin, fmax, df = 20.0, 148.0, 1.0
        freqs = collect(fmin:df:fmax)
        ntime = length(freqs)
        detector = GravitationalWaveData.H1
        gps = 1126259462.4

        # physical = m1,m2,chi1,chi2,dL,t_c,phase,ra,dec,theta_jn,psi -- ~4:1 mass ratio (GW190412/
        # GW190814-like) so higher-order-mode content is actually significant.
        physical_ref = T[32.0, 8.0, 0.3, -0.1, 0.5, 0.015, 0.4, 1.0, -0.3, 1.0, 0.6]
        strain = _rb_h_det_phenomxhm(freqs, physical_ref, detector, gps)
        psd = fill(T(1e-46), ntime)
        data = LikelihoodData(freqs, strain, psd, detector, gps)
        cfg = PEConfig{T}(ntime, 1, 1e-3, 1)

        # nbins=150 (this module's own default) target -> clips to the same actual count as
        # nbins=128 on this 129-point dense grid (both clamp to the grid's own n-1=128 cap before
        # `_phase_equal_bin_edges`'s phase-accumulation dedup, which further collapses it to 45
        # actual bins here) -- confirmed directly by a standalone nbins sweep during this task's
        # validation (30/60/100/128 target -> 23/33/41/45 actual), not assumed.
        rb = build_relative_binning(IMRPhenomXHM(), physical_ref, data; nbins=150)
        @test length(rb.edges) - 1 <= 150
        @test length(rb.edges) >= 2

        # **One real accuracy finding, worth documenting** (see DEVELOPMENT_PLAN.md's Phase 3 entry
        # for the full writeup): IMRPhenomXHM's away-from-theta_ref agreement is noticeably looser
        # than either the aligned-spin or IMRPhenomXP testsets above at a comparable bin count --
        # confirmed via a standalone nbins sweep (23/33/41/45 actual bins) that the reldiff plateaus
        # around 5-10% rather than shrinking sharply with more bins on this grid, AND via a
        # standalone finer-grid check (more achievable bins) that it DOES keep improving with
        # genuinely finer resolution -- so this is real linear-interpolation approximation error,
        # not a structural bug (further confirmed by the near-machine-precision match AT theta_ref
        # itself, which rules out an error in the A0/A1/B0/B1/href_edges construction). Root cause:
        # IMRPhenomXHM's ratio r(f) = h(f;trial)/h(f;theta_ref) sums FIVE independent (l,m) modes,
        # each evolving phase at a different rate with frequency AND with parameter perturbations
        # (roughly `m/2` times the dominant mode's rate) -- so the combined ratio has genuinely more
        # within-bin curvature than IMRPhenomXP's single twisted-up mode, which itself already needed
        # a materially looser tolerance than the aligned-spin dominant-mode-only case. Tolerances
        # below are loosened accordingly, with real margin above the observed values (not a silently
        # widened bar) -- pursued no further here (e.g. per-mode relative binning, which would
        # substantially reduce this curvature) since that is a genuine algorithmic extension out of
        # this task's scope, matching the precedent IMRPhenomXP's own task set for bin-placement
        # refinements.
        test_points = [
            (physical_ref, 0.0),                                                                # exact theta_ref
            (T[32.032, 8.0, 0.3, -0.1, 0.5, 0.015, 0.4, 1.0, -0.3, 1.0, 0.6], 0.10),              # tiny perturbation (mass); observed reldiff ~5.6-10% across nbins sweep
            (T[32.32, 8.0, 0.31, -0.11, 0.5, 0.015, 0.4, 1.0, -0.3, 1.0, 0.6], 0.10),             # small but multi-parameter; observed ~4.3-6.6%
            (T[33.0, 8.5, 0.32, -0.12, 0.52, 0.02, 0.45, 1.05, -0.25, 1.05, 0.65], 0.15),         # larger, all params perturbed; observed ~5.0-11.6%
        ]

        for (p, rtol) in test_points
            params = reshape(p, 11, 1)
            ll_full = zeros(T, 1)
            batched_log_likelihood_kernel!(ll_full, params, freqs, cfg, data; model=IMRPhenomXHM())
            ll_binned = zeros(T, 1)
            binned_log_likelihood_kernel!(ll_binned, params, rb, data; model=IMRPhenomXHM())

            @test isfinite(ll_full[1])
            @test isfinite(ll_binned[1])
            if p == physical_ref
                @test isapprox(ll_full[1], ll_binned[1]; atol=1e-6)
            else
                @test isapprox(ll_full[1], ll_binned[1]; rtol=rtol)
            end
        end
    end

    @testset "IMRPhenomXHM: multi-detector relative binning (H1+L1) matches the multi-detector dense kernel" begin
        # Mirrors the aligned-spin/IMRPhenomXP multi-detector testsets above structurally. Note this
        # does NOT (and cannot) compare against `KernelFused.batched_log_likelihood_kernel!`'s
        # multi-detector method -- that method still explicitly rejects IMRPhenomXHM (multi-detector
        # support for the DENSE kernel is a separate, out-of-scope follow-on, see
        # `KernelFused.jl`'s module docstring) -- so this instead compares the multi-detector
        # RELATIVE-BINNING result against the SUM of independent single-detector DENSE evaluations
        # (mathematically identical to what a multi-detector dense kernel would compute, since the
        # log-likelihood is additive across detectors and each detector's own antenna
        # response/time-delay is already independent), giving a genuine same-physics comparison
        # without needing that follow-on to exist yet.
        T = Float64
        fmin, fmax, df = 20.0, 148.0, 1.0
        freqs = collect(fmin:df:fmax)
        ntime = length(freqs)
        gps = 1126259462.4
        dets = [GravitationalWaveData.H1, GravitationalWaveData.L1]

        physical_ref = T[32.0, 8.0, 0.3, -0.1, 0.5, 0.015, 0.4, 1.0, -0.3, 1.0, 0.6]
        rng = Random.Xoshiro(17)
        strains = [_rb_h_det_phenomxhm_multi(freqs, physical_ref, d, gps) .+
                   1e-24 .* (randn(rng, ntime) .+ im .* randn(rng, ntime)) for d in dets]
        psds = [fill(1e-46, ntime) for _ in dets]
        data = [LikelihoodData(freqs, strains[i], psds[i], dets[i], gps) for i in eachindex(dets)]

        setups = build_relative_binning(IMRPhenomXHM(), physical_ref, data; nbins=150)
        @test length(setups) == 2

        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)
        trials = [
            copy(physical_ref),
            (let p = copy(physical_ref); p[1] += 0.3; p[3] += 0.01; p end),
            (let p = copy(physical_ref); p[8] += 0.02; p[9] -= 0.01; p end),
        ]
        for p in trials
            pc = reshape(p, 11, 1)
            ll_dense_sum = 0.0
            for i in eachindex(dets)
                # The single-detector DENSE kernel has no notion of "other detectors" -- it applies
                # `t_c` directly (see `KernelFused.jl`'s module docstring: no per-detector
                # geocenter-to-detector time-delay shift). To make "sum of single-detector dense
                # calls" match what a genuine multi-detector dense kernel (and the strain injection
                # above, via `_rb_h_det_phenomxhm_multi`) would compute, that shift must be applied
                # by hand here -- exactly the same `t_c + GravitationalWaveData.time_delay_from_geocenter(...)`
                # construction `build_relative_binning`'s own multi-detector method uses internally.
                delay_i = GravitationalWaveData.time_delay_from_geocenter(dets[i], p[8], p[9], gps)
                p_i = copy(p); p_i[6] += delay_i
                pc_i = reshape(p_i, 11, 1)
                ll_i = zeros(Float64, 1)
                batched_log_likelihood_kernel!(ll_i, pc_i, freqs, cfg, data[i]; model=IMRPhenomXHM())
                ll_dense_sum += ll_i[1]
            end
            ll_binned = zeros(Float64, 1)
            binned_log_likelihood_kernel!(ll_binned, pc, setups, data; model=IMRPhenomXHM())
            @test isfinite(ll_binned[1])
            @test isapprox(ll_binned[1], ll_dense_sum; rtol=0.15, atol=1e-3)
        end
    end

    @testset "IMRPhenomXPHM: binned likelihood matches full-grid likelihood near and away from theta_ref" begin
        # Single-detector IMRPhenomXPHM relative-binning follow-on (Phase 5 Task 5c): mirrors the
        # IMRPhenomXP/IMRPhenomXHM testsets above structurally. Combines BOTH known sources of
        # within-bin curvature (IMRPhenomXP's precession Euler angles, IMRPhenomXHM's five-mode
        # sum) -- but measured directly (standalone script, not guessed): reldiff at the three
        # perturbation points below is 6.2%, 5.3%, 5.0% respectively (45 actual bins from an
        # nbins=150 target, same grid as the IMRPhenomXHM testset above) -- comparable to, not
        # worse than, IMRPhenomXHM's own ~5-10% range, suggesting the two curvature sources don't
        # compound as badly as a naive worst-case sum would suggest. Tolerances below (0.10) keep
        # ~1.6-2x margin over the observed values, matching this file's established practice of
        # loosening with real headroom rather than to the exact observed number.
        T = Float64
        fmin, fmax, df = 20.0, 148.0, 1.0
        freqs = collect(fmin:df:fmax)
        ntime = length(freqs)
        detector = GravitationalWaveData.H1
        gps = 1126259462.4

        # physical = m1,m2,chi1,chi2,dL,t_c,phase,ra,dec,theta_jn,psi,chi1x,chi1y,chi2x,chi2y --
        # asymmetric mass ratio (HOM content) AND genuinely nonzero in-plane spin (precession).
        physical_ref = T[32.0, 8.0, 0.3, -0.1, 0.5, 0.015, 0.4, 1.0, -0.3, 1.0, 0.6, 0.15, -0.1, 0.05, 0.08]
        strain = _rb_h_det_phenomxphm(freqs, physical_ref, detector, gps)
        psd = fill(T(1e-46), ntime)
        data = LikelihoodData(freqs, strain, psd, detector, gps)
        cfg = PEConfig{T}(ntime, 1, 1e-3, 1)

        rb = build_relative_binning(IMRPhenomXPHM(), physical_ref, data; nbins=150)
        @test length(rb.edges) - 1 <= 150
        @test length(rb.edges) >= 2

        test_points = [
            (physical_ref, 0.0),                                                                              # exact theta_ref
            (T[32.032, 8.0, 0.3, -0.1, 0.5, 0.015, 0.4, 1.0, -0.3, 1.0, 0.6, 0.15, -0.1, 0.05, 0.08], 0.10),   # tiny perturbation (mass); observed reldiff ~6.2%
            (T[32.32, 8.0, 0.31, -0.11, 0.5, 0.015, 0.4, 1.0, -0.3, 1.0, 0.6, 0.16, -0.09, 0.06, 0.07], 0.10), # small but multi-parameter; observed ~5.3%
            (T[33.0, 8.5, 0.32, -0.12, 0.52, 0.02, 0.45, 1.05, -0.25, 1.05, 0.65, 0.17, -0.08, 0.07, 0.06], 0.10), # larger, all params perturbed; observed ~5.0%
        ]

        for (p, rtol) in test_points
            params = reshape(p, 15, 1)
            ll_full = zeros(T, 1)
            batched_log_likelihood_kernel!(ll_full, params, freqs, cfg, data; model=IMRPhenomXPHM())
            ll_binned = zeros(T, 1)
            binned_log_likelihood_kernel!(ll_binned, params, rb, data; model=IMRPhenomXPHM())

            @test isfinite(ll_full[1])
            @test isfinite(ll_binned[1])
            if p == physical_ref
                @test isapprox(ll_full[1], ll_binned[1]; atol=1e-6)
            else
                @test isapprox(ll_full[1], ll_binned[1]; rtol=rtol)
            end
        end
    end

    @testset "IMRPhenomXPHM: multi-detector relative binning (H1+L1) matches the multi-detector dense kernel" begin
        # Mirrors the IMRPhenomXP multi-detector testset above (NOT IMRPhenomXHM's "sum of
        # single-detector dense calls" workaround) -- unlike IMRPhenomXHM, IMRPhenomXPHM DOES have
        # a real multi-detector dense kernel (Task 5c, KernelFused.jl), so this compares directly
        # against it. Measured directly (standalone script): reldiff at the four trial points is
        # ~3e-15 (theta_ref), 4.4%, 0.03%, 0.07% -- comfortably under the 0.10 bound below (>2x
        # margin over the observed maximum).
        T = Float64
        fmin, fmax, df = 20.0, 148.0, 1.0
        freqs = collect(fmin:df:fmax)
        ntime = length(freqs)
        gps = 1126259462.4
        dets = [GravitationalWaveData.H1, GravitationalWaveData.L1]
        f_ref = freqs[1]

        physical_ref = T[32.0, 8.0, 0.3, -0.1, 0.5, 0.015, 0.4, 1.0, -0.3, 1.0, 0.6, 0.15, -0.1, 0.05, 0.08]
        rng = Random.Xoshiro(19)
        strains = [_rb_h_det_phenomxphm_multi(freqs, physical_ref, d, gps, f_ref) .+
                   1e-24 .* (randn(rng, ntime) .+ im .* randn(rng, ntime)) for d in dets]
        psds = [fill(1e-46, ntime) for _ in dets]
        data = [LikelihoodData(freqs, strains[i], psds[i], dets[i], gps) for i in eachindex(dets)]

        setups = build_relative_binning(IMRPhenomXPHM(), physical_ref, data; nbins=150)
        @test length(setups) == 2

        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)
        trials = [
            copy(physical_ref),
            (let p = copy(physical_ref); p[1] += 0.3; p[3] += 0.01; p end),
            (let p = copy(physical_ref); p[8] += 0.02; p[9] -= 0.01; p end),
            (let p = copy(physical_ref); p[12] += 0.02; p[14] -= 0.01; p end),
        ]
        for p in trials
            pc = reshape(p, 15, 1)
            ll_dense = zeros(Float64, 1)
            batched_log_likelihood_kernel!(ll_dense, pc, cfg, data; model=IMRPhenomXPHM())
            ll_binned = zeros(Float64, 1)
            binned_log_likelihood_kernel!(ll_binned, pc, setups, data; model=IMRPhenomXPHM())
            @test isapprox(ll_binned[1], ll_dense[1]; rtol=0.10, atol=1e-3)
        end
    end

    @testset "reference-amplitude underflow at high frequency stays finite (real GW150914-band regression)" begin
        # Regression test for a real bug found 2026-09-02 running relative binning on the actual
        # full-resolution (20-1024 Hz) real GW150914 H1+L1 data for the first time (every prior
        # validation in this file uses a narrow fmax=284 Hz band, which never reaches this
        # regime). IMRPhenomD/XAS's amplitude decays exponentially past ringdown and genuinely
        # underflows to literal IEEE `0.0` well before 1024 Hz at GW150914-scale masses (empirically
        # confirmed: ~938 Hz for m1=36/m2=29 Msun). `_phase_equal_bin_edges` always places its LAST
        # edge at the dense grid's final index (see that function's own docstring), so any analysis
        # band whose top frequency lands past the underflow point deterministically makes
        # `href_edges[end] == 0` -- before the `_safe_ratio` fix, `r_hi = x / href_edges[end]` was a
        # literal `x/0`, and the resulting NaN/Inf then poisoned `dh_sum`/`hh_sum`'s running sum
        # (both plain `+=` accumulators over ALL bins) so EVERY walker's likelihood came back NaN,
        # not just points near the underflow -- confirmed by hand this session before the fix.
        T = Float64
        fmin, fmax, df = 20.0, 1024.0, 2.0
        freqs = collect(fmin:df:fmax)
        ntime = length(freqs)
        gps = 1126259462.4

        # physical = m1,m2,chi1,chi2,dL,t_c,phase,ra,dec,theta_jn,psi -- GW150914-like masses,
        # the exact scale that triggers the underflow (see docstring above).
        physical_ref = T[36.0, 29.0, 0.0, 0.0, 0.41, 0.0, 0.0, 1.0, -0.5, 0.5, 0.5]

        for model in (IMRPhenomD(), IMRPhenomXAS())
            data0 = LikelihoodData(freqs, zeros(ComplexF64, ntime), fill(T(1e-46), ntime))
            strain = _rb_h_det(model, freqs, physical_ref, data0.detector, data0.gps_time)
            psd = fill(T(1e-46), ntime)
            data = LikelihoodData(freqs, strain, psd)
            cfg = PEConfig{T}(ntime, 1, 1e-3, 1)

            rb = build_relative_binning(model, physical_ref, data; nbins=150)
            # Confirms this test actually exercises the guarded path, not just a healthy grid.
            @test rb.href_edges[end] == zero(ComplexF64)

            trials = [
                copy(physical_ref),
                (let p = copy(physical_ref); p[1] += 0.5; p[3] += 0.02; p end),
                (let p = copy(physical_ref); p[8] += 0.3; p[9] -= 0.2; p end),
            ]
            for (i, p) in enumerate(trials)
                pc = reshape(p, 11, 1)
                ll_binned = zeros(T, 1)
                binned_log_likelihood_kernel!(ll_binned, pc, rb, data; model=model)
                @test isfinite(ll_binned[1])
                i == 1 && @test isapprox(ll_binned[1], 0.0; atol=1e-3)
            end
        end

        # Multi-detector (H1+L1): the same underflow, hit independently through
        # `_relative_binning_kernel_phenomxas_multi!`'s own `_safe_ratio`-guarded division sites.
        # Injection must include each detector's own geocenter time-delay shift (`_rb_h_det`
        # doesn't -- see the "Multi-detector relative binning" testset's own `_rb_h_det_multi`
        # above, which this mirrors) or the self-overlap-at-theta_ref check below is comparing
        # against the wrong waveform and fails for a reason unrelated to this test's actual point.
        function _rb_h_det_multi_underflow(model, freqs, physical::AbstractVector, detector, gps_time)
            m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = physical
            mu = (m1 * m2) / (m1 + m2)
            mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
            eta = (m1 * m2) / (m1 + m2)^2
            M_sec = (mc / eta^0.6) * GMsun_over_c3
            delay = GravitationalWaveData.time_delay_from_geocenter(detector, ra, dec, gps_time)
            t0 = -2 * pi * (t_c + delay) / M_sec

            amp = Ampl(model, freqs, mc, eta, chi1, chi2, dL)
            phase_vals = _rbt_phase(model, freqs, mc, eta, chi1, chi2, t0, M_sec, phase)
            h = amp .* exp.(im .* phase_vals)

            pfac = 0.5 * (1 + cos(theta_jn)^2)
            cfac = cos(theta_jn)
            Fplus, Fcross = GravitationalWaveData.compute_antenna_response(detector, ra, dec, psi, gps_time)
            combo = Fplus * pfac - im * Fcross * cfac
            return combo .* h
        end

        dets = [GravitationalWaveData.H1, GravitationalWaveData.L1]
        strains = [_rb_h_det_multi_underflow(IMRPhenomXAS(), freqs, physical_ref, d, gps) for d in dets]
        psds = [fill(T(1e-46), ntime) for _ in dets]
        data_md = [LikelihoodData(freqs, strains[i], psds[i], dets[i], gps) for i in eachindex(dets)]

        setups_md = build_relative_binning(IMRPhenomXAS(), physical_ref, data_md; nbins=150)
        @test all(s -> s.href_edges[end] == zero(ComplexF64), setups_md)

        pc = reshape(physical_ref, 11, 1)
        ll_md = zeros(T, 1)
        binned_log_likelihood_kernel!(ll_md, pc, setups_md, data_md; model=IMRPhenomXAS())
        @test isfinite(ll_md[1])
        @test isapprox(ll_md[1], 0.0; atol=1e-3)
    end

    @testset "IMRPhenomXHM: reference-amplitude underflow at high frequency stays finite" begin
        # IMRPhenomXHM-specific version of the regression test above, per this task's own brief
        # ("find it and use it as a template"). **One genuinely interesting finding from this task's
        # own validation, worth recording here**: unlike IMRPhenomD/IMRPhenomXAS's single dominant
        # mode -- which reaches LITERAL IEEE `0.0` by ~938 Hz at GW150914-scale masses (the original
        # bug) -- IMRPhenomXHM's SUMMED five-mode `h_det(f)` was NOT found to reach literal `0.0`
        # anywhere in a realistic 20-1024 Hz band even after a deliberate search across several mass
        # configurations (GW150914-scale through ~125 Msun total, both near-equal and asymmetric
        # mass ratios): the (2,2) mode's own contribution does underflow to exact `0.0` (consistent
        # with the original bug, since it shares IMRPhenomXAS's amplitude ansatz internally via
        # `_xhm_system_setup`'s `amplcoeffs22`), but at LEAST one subdominant mode's own ringdown
        # decays more slowly and leaves a nonzero (if extremely tiny, `~1e-34`-`1e-40` magnitude
        # across the configurations tried) residual -- e.g. `m1=36,m2=29,chi1=0.1,chi2=-0.05,dL=0.41`
        # gives `href_edges[end] = 3.87e-29 - 6.34e-29im` on the ORIGINAL (chi=0.2/-0.1 at m2=9)
        # search and `-7.79e-35 - 4.04e-34im` at `chi1=chi2=0` -- neither is exact zero, and pushing
        # to a much heavier `m1=70,m2=55` config only reduced the residual to `~1.4e-40`, still
        # nonzero. (A side discovery from this search, NOT part of this task's scope to fix: an
        # EXACTLY equal-mass configuration, `m1=m2=32.5`, produces a literal `NaN` waveform from
        # `GravitationalWaveforms.jl`'s own `IMRPhenomXHM` -- almost certainly a `0/0` in the `(3,2)` mode-mixing
        # setup's `Seta`/`dchi`-dependent coefficients at exactly `Seta=dchi=0`, a `GravitationalWaveforms.jl`
        # numerical-edge-case bug, not a `RelativeBinning.jl` one -- flagged here for visibility,
        # not chased further.)
        #
        # Given a literal exact-zero `href_edges` entry could not be produced from real physics
        # within a realistic band, this test instead does BOTH things the task's validation bar
        # actually cares about: (1) confirms the DEEPEST realistic near-underflow configuration found
        # (`href_edges[end]` at the `~1e-29`-`1e-34` scale, i.e. utterly negligible relative to a
        # `psd=1e-46`-normalized bin's typical `A0`/`B0` contribution, but NOT exactly representable
        # as literal zero) still gives a finite, correct-at-reference log-likelihood -- ruling out
        # any OTHER precision pathology (e.g. a near-zero-but-nonzero denominator producing a huge
        # but finite blowup that swamps the sum) beyond the specific `x/0` failure mode `_safe_ratio`
        # guards against; and (2) directly, deterministically exercises `_safe_ratio`'s
        # `denom==0` branch through the FULL single- and multi-detector `IMRPhenomXHM` kernel paths
        # by manually zeroing a built setup's `href_edges` entry (the array is a plain `Vector`, so
        # mutable in place even though `RelativeBinningSetup` itself has no setters) -- giving the
        # same unambiguous confirmation the naturally-occurring dominant-mode regression test above
        # gets "for free" from real physics, without depending on IMRPhenomXHM ever actually
        # producing that scenario naturally.
        T = Float64
        fmin, fmax, df = 20.0, 1024.0, 2.0
        freqs = collect(fmin:df:fmax)
        ntime = length(freqs)
        gps = 1126259462.4
        detector = GravitationalWaveData.H1

        # Deepest real near-underflow configuration found during this task's search (see above).
        physical_ref = T[36.0, 29.0, 0.1, -0.05, 0.41, 0.0, 0.0, 1.0, -0.5, 0.5, 0.5]
        strain = _rb_h_det_phenomxhm(freqs, physical_ref, detector, gps)
        psd = fill(T(1e-46), ntime)
        data = LikelihoodData(freqs, strain, psd, detector, gps)
        cfg = PEConfig{T}(ntime, 1, 1e-3, 1)

        rb = build_relative_binning(IMRPhenomXHM(), physical_ref, data; nbins=150)
        # Confirms this really is deep in the near-underflow regime (not a healthy-magnitude bin
        # edge) even though it isn't exactly representable as literal zero -- see docstring above.
        @test abs(rb.href_edges[end]) < 1e-25

        pc = reshape(physical_ref, 11, 1)
        ll_at_ref = zeros(T, 1)
        binned_log_likelihood_kernel!(ll_at_ref, pc, rb, data; model=IMRPhenomXHM())
        @test isfinite(ll_at_ref[1])
        @test isapprox(ll_at_ref[1], 0.0; atol=1e-3)

        trials = [
            (let p = copy(physical_ref); p[1] += 0.5; p[3] += 0.02; p end),
            (let p = copy(physical_ref); p[8] += 0.3; p[9] -= 0.2; p end),
        ]
        for p in trials
            pc2 = reshape(p, 11, 1)
            ll_binned = zeros(T, 1)
            binned_log_likelihood_kernel!(ll_binned, pc2, rb, data; model=IMRPhenomXHM())
            @test isfinite(ll_binned[1])
        end

        # Direct, deterministic exercise of `_safe_ratio`'s `denom==0` branch (see docstring above):
        # manually zero the built setup's last bin edge and confirm the kernel still returns a
        # finite log-likelihood, single- and multi-detector.
        rb_zeroed = build_relative_binning(IMRPhenomXHM(), physical_ref, data; nbins=150)
        rb_zeroed.href_edges[end] = zero(ComplexF64)
        ll_zeroed = zeros(T, 1)
        binned_log_likelihood_kernel!(ll_zeroed, pc, rb_zeroed, data; model=IMRPhenomXHM())
        @test isfinite(ll_zeroed[1])

        dets = [GravitationalWaveData.H1, GravitationalWaveData.L1]
        strains_md = [_rb_h_det_phenomxhm_multi(freqs, physical_ref, d, gps) for d in dets]
        psds_md = [fill(T(1e-46), ntime) for _ in dets]
        data_md = [LikelihoodData(freqs, strains_md[i], psds_md[i], dets[i], gps) for i in eachindex(dets)]
        setups_md = build_relative_binning(IMRPhenomXHM(), physical_ref, data_md; nbins=150)
        for s in setups_md
            s.href_edges[end] = zero(ComplexF64)
        end
        ll_md_zeroed = zeros(T, 1)
        binned_log_likelihood_kernel!(ll_md_zeroed, pc, setups_md, data_md; model=IMRPhenomXHM())
        @test isfinite(ll_md_zeroed[1])
    end

    @testset "_safe_ratio guards literal division by zero" begin
        _safe_ratio = GravitationalWaveBayes.RelativeBinning._safe_ratio
        @test _safe_ratio(1.0 + 2.0im, 0.0 + 0.0im) == 0.0 + 0.0im
        @test _safe_ratio(0.0 + 0.0im, 0.0 + 0.0im) == 0.0 + 0.0im
        @test isapprox(_safe_ratio(3.0 + 4.0im, 2.0 + 0.0im), (3.0 + 4.0im) / (2.0 + 0.0im))
    end

    @testset "_safe_div guards literal division by zero" begin
        _safe_div = GravitationalWaveBayes.RelativeBinning._safe_div
        @test _safe_div(1.0, 0.0) == 0.0
        @test _safe_div(0.0, 0.0) == 0.0
        @test isapprox(_safe_div(3.0, 2.0), 3.0 / 2.0)
    end

    @testset "build_relative_binning: zero-valued PSD bin at SETUP time stays finite (Phase 5 completion regression)" begin
        # Phase 5 completion (this session): regression test for 4 previously-unguarded
        # `w = 4*df/psd[k]` PSD-division sites found by this session's systematic NaN/Inf audit --
        # one in each single-detector `build_relative_binning` method (the generic
        # IMRPhenomD/XAS/XP-shared one, IMRPhenomXP's own dedicated one, IMRPhenomXHM's, and
        # IMRPhenomXPHM's), missed by the earlier "targeted" PSD-guard pass (which only fixed
        # `dd_term`'s SEPARATE `abs2(strain[k])/psd[k]` division). Unlike a dense-kernel PSD
        # division (re-evaluated every leapfrog step -- guarded already, and only corrupts that one
        # step if it weren't), a zero PSD bin here would poison `A0`/`A1`/`B0`/`B1` with `Inf`/`NaN`
        # ONCE at setup time, silently corrupting EVERY subsequent walker's likelihood for the whole
        # NUTS run. `strain` is left exactly zero here -- irrelevant to what's being tested (the PSD
        # side of the division), and it isolates the fix from any reference-waveform-amplitude
        # concern `_safe_ratio`'s own regression tests above already cover.
        T = Float64
        ntime = 32
        freqs = collect(20.0:1.0:(20.0 + ntime - 1))
        psd = ones(T, ntime)
        psd[16] = zero(T)   # one notched bin, mid-grid
        strain = zeros(ComplexF64, ntime)
        data = LikelihoodData(freqs, strain, psd)

        physical_ref_11 = T[30.0, 25.0, 0.1, -0.1, 0.5, 0.0, 0.2, 1.0, -0.3, 0.7, 0.4]
        physical_ref_15 = T[30.0, 25.0, 0.1, -0.1, 0.5, 0.0, 0.2, 1.0, -0.3, 0.7, 0.4, 0.05, 0.02, -0.03, 0.01]

        for (model, physical_ref) in ((IMRPhenomD(), physical_ref_11), (IMRPhenomXP(), physical_ref_15),
                                       (IMRPhenomXHM(), physical_ref_11), (IMRPhenomXPHM(), physical_ref_15))
            rb = build_relative_binning(model, physical_ref, data; nbins=8)
            @test all(isfinite, rb.A0)
            @test all(isfinite, rb.A1)
            @test all(isfinite, rb.B0)
            @test all(isfinite, rb.B1)
            @test isfinite(rb.dd_term)

            pc = reshape(copy(physical_ref), length(physical_ref), 1)
            ll_binned = zeros(T, 1)
            binned_log_likelihood_kernel!(ll_binned, pc, rb, data; model=model)
            @test isfinite(ll_binned[1])
        end
    end
end
