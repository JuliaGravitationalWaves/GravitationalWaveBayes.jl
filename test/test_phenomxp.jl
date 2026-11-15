using Test
using GravitationalWaveBayes
using GravitationalWaveforms: IMRPhenomXAS, IMRPhenomXP
using GravitationalWaveforms
using GravitationalWaveData

# Manual, host-side reference implementation of the 15-parameter IMRPhenomXP physical forward model,
# mirroring `KernelFused._fused_waveform_likelihood_kernel_phenomxp!` line-for-line (t0/phiRef
# construction, antenna response, precession setup, per-frequency twisting-up + zeta rotation) --
# same role as `test_kernel_fused.jl`'s `_manual_h_det` for IMRPhenomD/IMRPhenomXAS, kept self-contained
# in this file per this codebase's existing per-test-file duplication style.
function _manual_h_det_phenomxp(freqs::AbstractVector, physical::AbstractVector, detector, gps_time)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi, chi1x, chi1y, chi2x, chi2y = physical
    mu = (m1 * m2) / (m1 + m2)
    mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
    eta = (m1 * m2) / (m1 + m2)^2
    M_sec = (mc / eta^0.6) * GravitationalWaveforms.GMsun_over_c3
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

function _manual_ll_phenomxp(freqs, physical::AbstractVector, data::LikelihoodData)
    h_det = _manual_h_det_phenomxp(freqs, physical, data.detector, data.gps_time)
    df = freqs[2] - freqs[1]
    return -2 * df * sum(abs2.(h_det .- data.strain) ./ data.psd)
end

# Same-shape aligned-spin (IMRPhenomXAS) reference as `test_kernel_fused.jl`'s `_manual_h_det`, kept
# local here (not `include`d from that file, to keep this file self-contained) -- used only for the
# phase-independent |h_det| aligned-spin-limit sanity check below.
function _manual_h_det_phenomxas(freqs::AbstractVector, physical11::AbstractVector, detector, gps_time)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = physical11
    mu = (m1 * m2) / (m1 + m2)
    mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
    eta = (m1 * m2) / (m1 + m2)^2
    M_sec = (mc / eta^0.6) * GravitationalWaveforms.GMsun_over_c3
    t0 = 2 * pi * t_c / M_sec

    amp = GravitationalWaveforms.Ampl(IMRPhenomXAS(), freqs, mc, eta, chi1, chi2, dL)
    phase_vals = GravitationalWaveforms.Phi(IMRPhenomXAS(), freqs, mc, eta, chi1, chi2; t0=t0, phiRef=phase)
    h = amp .* exp.(im .* phase_vals)

    pfac = 0.5 * (1 + cos(theta_jn)^2)
    cfac = cos(theta_jn)
    Fplus, Fcross = GravitationalWaveData.compute_antenna_response(detector, ra, dec, psi, gps_time)
    combo = Fplus * pfac - im * Fcross * cfac
    return combo .* h
end

@testset "IMRPhenomXP (Phase 3 precessing-spin) single-detector kernel" begin
    ntime = 48
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    gps = 1126259462.4
    detector = GravitationalWaveData.H1

    # Genuinely nonzero in-plane spins, comfortably inside the unit-magnitude constraint:
    # |chi1|^2 = 0.1^2+0.15^2+(-0.1)^2 = 0.0425, |chi2|^2 = (-0.05)^2+0.05^2+0.08^2 = 0.0114.
    ref_physical = [32.0, 28.0, 0.1, -0.05, 0.5, 0.01, 0.4, 1.0, -0.3, 1.2, 0.6, 0.15, -0.1, 0.05, 0.08]

    @testset "kernel agrees with a manual host-side reference" begin
        data0 = LikelihoodData(freqs, zeros(ComplexF64, ntime), ones(Float64, ntime), detector, gps)
        strain = _manual_h_det_phenomxp(freqs, ref_physical, detector, gps) .+
                 1e-24 .* (randn(ntime) .+ im .* randn(ntime))
        psd = fill(1e-46, ntime)
        data = LikelihoodData(freqs, strain, psd, detector, gps)
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)

        ll_manual = _manual_ll_phenomxp(freqs, ref_physical, data)
        ll_kernel = zeros(Float64, 1)
        batched_log_likelihood_kernel!(ll_kernel, reshape(ref_physical, 15, 1), freqs, cfg, data; model=IMRPhenomXP())
        @test isapprox(ll_kernel[1], ll_manual; rtol=1e-8, atol=1e-8)
    end

    @testset "noise-free injection recovery: peak at injection, single-parameter perturbations strictly decrease ll" begin
        # Self-consistent injection-recovery check (see this file's own module-level context /
        # DEVELOPMENT_PLAN.md Phase 3 for why this, not a cross-kernel comparison, is the primary
        # IMRPhenomXP correctness check: IMRPhenomXP's twisted-up h_det differs from IMRPhenomXAS's by a
        # constant exp(-2i*zeta) phase even at zero in-plane spin, so injecting and recovering
        # with the SAME IMRPhenomXP kernel sidesteps that entirely).
        strain = _manual_h_det_phenomxp(freqs, ref_physical, detector, gps)
        psd = fill(1e-46, ntime)
        data = LikelihoodData(freqs, strain, psd, detector, gps)
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)

        ll0 = zeros(Float64, 1)
        batched_log_likelihood_kernel!(ll0, reshape(ref_physical, 15, 1), freqs, cfg, data; model=IMRPhenomXP())
        @test isapprox(ll0[1], 0.0; atol=1e-6)

        row_names = ["m1", "m2", "chi1", "chi2", "dL", "t_c", "phase", "ra", "dec", "theta_jn", "psi",
                     "chi1x", "chi1y", "chi2x", "chi2y"]
        perturbations = [0.5, 0.5, 0.02, 0.02, 0.05, 0.002, 0.05, 0.05, 0.05, 0.05, 0.05,
                          0.02, 0.02, 0.02, 0.02]
        for i in 1:15
            p2 = copy(ref_physical)
            p2[i] += perturbations[i]
            ll2 = zeros(Float64, 1)
            batched_log_likelihood_kernel!(ll2, reshape(p2, 15, 1), freqs, cfg, data; model=IMRPhenomXP())
            @test ll2[1] < ll0[1]
        end
    end

    @testset "aligned-spin limit: |h_det| matches IMRPhenomXAS's kernel path (phase-independent check)" begin
        # chi1x=chi1y=chi2x=chi2y=0: IMRPhenomXP's twisted-up h_det and IMRPhenomXAS's aligned-spin
        # h_det differ by a constant exp(-2i*zeta_polarization) phase factor (a real,
        # LALSuite-confirmed property of the IMRPhenomXP polarization convention, NOT a bug -- see
        # this file's own header / DEVELOPMENT_PLAN.md Phase 3), so only |h_det| is directly
        # comparable between the two kernel paths, not the raw complex value.
        physical_xp = [32.0, 28.0, 0.1, -0.05, 0.5, 0.01, 0.4, 1.0, -0.3, 1.2, 0.6, 0.0, 0.0, 0.0, 0.0]
        physical_xas = physical_xp[1:11]

        h_xp = _manual_h_det_phenomxp(freqs, physical_xp, detector, gps)
        h_xas = _manual_h_det_phenomxas(freqs, physical_xas, detector, gps)

        @test isapprox(abs.(h_xp), abs.(h_xas); rtol=1e-6, atol=1e-30)
    end

    @testset "row-count assertion: IMRPhenomXP requires 15-row params" begin
        data = LikelihoodData(freqs, zeros(ComplexF64, ntime), ones(Float64, ntime), detector, gps)
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)
        ll = zeros(Float64, 1)
        params11 = reshape(ref_physical[1:11], 11, 1)
        @test_throws AssertionError batched_log_likelihood_kernel!(ll, params11, freqs, cfg, data; model=IMRPhenomXP())
    end
end

@testset "IMRPhenomXP (Phase 3 follow-on) multi-detector kernel" begin
    # Manual, host-side multi-detector IMRPhenomXP reference: mirrors `_manual_h_det_phenomxp` above,
    # adding the per-detector `GravitationalWaveData.time_delay_from_geocenter`-shifted `t0` -- same relationship
    # `test_kernel_fused.jl`'s `_manual_h_det_multi` bears to its own single-detector
    # `_manual_h_det` (host-side, non-differentiated reference code, so calling the real `GravitationalWaveData`
    # functions directly is fine here, unlike inside the kernel body itself -- see
    # KernelFused.jl's module docstring for why).
    function _manual_h_det_phenomxp_multi(freqs::AbstractVector, physical::AbstractVector, detector, gps_time, f_ref)
        m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi, chi1x, chi1y, chi2x, chi2y = physical
        mu = (m1 * m2) / (m1 + m2)
        mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
        eta = (m1 * m2) / (m1 + m2)^2
        M_sec = (mc / eta^0.6) * GravitationalWaveforms.GMsun_over_c3
        delay = GravitationalWaveData.time_delay_from_geocenter(detector, ra, dec, gps_time)
        t0 = 2 * pi * (t_c + delay) / M_sec
        phiRef = phase

        Fplus, Fcross = GravitationalWaveData.compute_antenna_response(detector, ra, dec, psi, gps_time)

        # f_ref is taken from the FIRST detector's first frequency bin, matching the kernel's own
        # `freqs_tuple[1][1]` convention (setup is hoisted outside the per-detector loop, so it
        # can only anchor to one representative grid) -- see KernelFused.jl's
        # `_fused_waveform_likelihood_kernel_phenomxp_multi!` comment block.
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

    ntime = 40
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    gps = 1126259462.4
    dets = [GravitationalWaveData.H1, GravitationalWaveData.L1]
    f_ref = freqs[1]  # both detectors share this grid here, matching the kernel's own convention
    ref_physical = [32.0, 28.0, 0.1, -0.05, 0.5, 0.01, 0.4, 1.0, -0.3, 1.2, 0.6, 0.15, -0.1, 0.05, 0.08]

    @testset "agrees with a manual host-side multi-detector reference" begin
        strains = [_manual_h_det_phenomxp_multi(freqs, ref_physical, d, gps, f_ref) .+
                   1e-24 .* (randn(ntime) .+ im .* randn(ntime)) for d in dets]
        psds = [fill(1e-46, ntime) for _ in dets]
        data = [LikelihoodData(freqs, strains[i], psds[i], dets[i], gps) for i in eachindex(dets)]
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)

        ll_manual = sum(begin
            h = _manual_h_det_phenomxp_multi(freqs, ref_physical, dets[i], gps, f_ref)
            df = freqs[2] - freqs[1]
            -2 * df * sum(abs2.(h .- data[i].strain) ./ data[i].psd)
        end for i in eachindex(dets))

        ll_kernel = zeros(Float64, 1)
        batched_log_likelihood_kernel!(ll_kernel, reshape(ref_physical, 15, 1), cfg, data; model=IMRPhenomXP())
        @test isapprox(ll_kernel[1], ll_manual; rtol=1e-8, atol=1e-8)
    end

    @testset "noise-free injection recovery: peak at injection, single-parameter perturbations strictly decrease ll" begin
        strains = [_manual_h_det_phenomxp_multi(freqs, ref_physical, d, gps, f_ref) for d in dets]
        psds = [fill(1e-46, ntime) for _ in dets]
        data = [LikelihoodData(freqs, strains[i], psds[i], dets[i], gps) for i in eachindex(dets)]
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)

        ll0 = zeros(Float64, 1)
        batched_log_likelihood_kernel!(ll0, reshape(ref_physical, 15, 1), cfg, data; model=IMRPhenomXP())
        @test isapprox(ll0[1], 0.0; atol=1e-6)

        perturbations = [0.5, 0.5, 0.02, 0.02, 0.05, 0.002, 0.05, 0.05, 0.05, 0.05, 0.05,
                          0.02, 0.02, 0.02, 0.02]
        for i in 1:15
            p2 = copy(ref_physical)
            p2[i] += perturbations[i]
            ll2 = zeros(Float64, 1)
            batched_log_likelihood_kernel!(ll2, reshape(p2, 15, 1), cfg, data; model=IMRPhenomXP())
            @test ll2[1] < ll0[1]
        end
    end

    @testset "mismatched gps_time across detectors is rejected" begin
        ntime_small = 8
        freqs_small = collect(20.0:1.0:(20.0 + ntime_small - 1))
        strain_small = zeros(ComplexF64, ntime_small)
        psd_small = ones(Float64, ntime_small)
        data_bad = [LikelihoodData(freqs_small, strain_small, psd_small, GravitationalWaveData.H1, gps),
                    LikelihoodData(freqs_small, strain_small, psd_small, GravitationalWaveData.L1, gps + 1.0)]
        cfg_small = PEConfig{Float64}(ntime_small, 1, 0.01, 1)
        ll = zeros(Float64, 1)
        params = reshape(ref_physical, 15, 1)
        @test_throws ArgumentError batched_log_likelihood_kernel!(ll, params, cfg_small, data_bad; model=IMRPhenomXP())
    end

    @testset "row-count assertion: IMRPhenomXP multi-detector requires 15-row params" begin
        ntime_small = 8
        freqs_small = collect(20.0:1.0:(20.0 + ntime_small - 1))
        strain_small = zeros(ComplexF64, ntime_small)
        psd_small = ones(Float64, ntime_small)
        data = [LikelihoodData(freqs_small, strain_small, psd_small, GravitationalWaveData.H1, gps),
                LikelihoodData(freqs_small, strain_small, psd_small, GravitationalWaveData.L1, gps)]
        cfg_small = PEConfig{Float64}(ntime_small, 1, 0.01, 1)
        ll = zeros(Float64, 1)
        params11 = reshape(ref_physical[1:11], 11, 1)
        @test_throws AssertionError batched_log_likelihood_kernel!(ll, params11, cfg_small, data; model=IMRPhenomXP())
    end
end

@testset "IMRPhenomXP sampled-space prior (_log_prior 15-argument method)" begin
    # `_log_prior` is internal to BatchedNUTS -- accessed via the module's qualified name, matching
    # this codebase's existing style for testing internal-but-important helpers (see
    # test_batched_nuts_gw_smoke.jl for precedent).
    _log_prior = GravitationalWaveBayes.BatchedNUTS._log_prior

    # Comfortably in-bounds 15-argument point (mc, q, chi1, chi2, dL, t_c, phase, ra, sin_dec,
    # cos_theta_jn, psi, chi1x, chi1y, chi2x, chi2y).
    good = (30.0, 0.8, 0.1, -0.05, 0.5, 0.0, 0.4, 1.0, -0.3, 0.2, 0.6, 0.15, -0.1, 0.05, 0.08)
    @test _log_prior(good...) == 0.0

    # Out-of-box in-plane spin component.
    bad_box = (30.0, 0.8, 0.1, -0.05, 0.5, 0.0, 0.4, 1.0, -0.3, 0.2, 0.6, 1.5, -0.1, 0.05, 0.08)
    @test _log_prior(bad_box...) == -Inf

    # In-box per-component but violates |chi1|^2 <= 1 (chi1=0.1 aligned + chi1x=chi1y=0.9 each).
    bad_norm = (30.0, 0.8, 0.1, -0.05, 0.5, 0.0, 0.4, 1.0, -0.3, 0.2, 0.6, 0.9, 0.9, 0.05, 0.08)
    @test _log_prior(bad_norm...) == -Inf

    # The 11-argument method is untouched by this addition.
    @test _log_prior(30.0, 0.8, 0.1, -0.05, 0.5, 0.0, 0.4, 1.0, -0.3, 0.2, 0.6) == 0.0
end
