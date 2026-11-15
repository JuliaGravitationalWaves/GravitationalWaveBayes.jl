using Test
using GravitationalWaveBayes
using GravitationalWaveforms: IMRPhenomXHM, IMRPhenomXP, IMRPhenomXPHM
using GravitationalWaveforms
using GravitationalWaveData

# Manual, host-side reference implementation of the 15-parameter IMRPhenomXPHM physical forward
# model, mirroring `KernelFused._fused_waveform_likelihood_kernel_phenomxphm!` line-for-line
# (precession setup, final_spin_override, per-mode setups, per-frequency twisting-up + zeta
# rotation, t_c post-multiplication) -- same role `test_phenomxp.jl`'s `_manual_h_det_phenomxp`
# plays for `IMRPhenomXP`. Does NOT reuse that function despite the near-identical precession setup
# (this file stays self-contained, matching this codebase's existing per-test-file duplication
# style) -- the two diverge after `setup` is built (this one needs `_xhm_system_setup`'s
# `final_spin_override` and the five-mode `_xphm_hphc_scalar` sum, not `IMRPhenomXAS`'s
# single-mode `Ampl`/`Phi` pair).
function _manual_h_det_phenomxphm(freqs::AbstractVector, physical::AbstractVector, detector, gps_time; version = 223)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi, chi1x, chi1y, chi2x, chi2y = physical
    mu = (m1 * m2) / (m1 + m2)
    mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
    eta = (m1 * m2) / (m1 + m2)^2

    Fplus, Fcross = GravitationalWaveData.compute_antenna_response(detector, ra, dec, psi, gps_time)

    f_ref = freqs[1]
    setup = GravitationalWaveforms._xp_precession_setup(m1, m2, chi1x, chi1y, chi1, chi2x, chi2y, chi2,
                                              f_ref, theta_jn, phase; prec_version = Val(version))

    # Precessing final spin, as the kernel uses it -- omitting this would silently compare against
    # the WRONG (aligned-only) physics. Version-dependent: 102 chi_p / 223 MSA FinalSpinMod 3
    # (`setup.final_spin`).
    final_spin = setup.final_spin
    sys = GravitationalWaveforms._xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, zero(phase);
                                         final_spin_override = final_spin)
    ms21 = GravitationalWaveforms._xhm_mode_setup(2, 1, sys)
    ms33 = GravitationalWaveforms._xhm_mode_setup(3, 3, sys)
    ms44 = GravitationalWaveforms._xhm_mode_setup(4, 4, sys)
    m32 = GravitationalWaveforms._xhm_mode32_setup(sys)

    h = Vector{ComplexF64}(undef, length(freqs))
    for (i, f) in enumerate(freqs)
        hp, hc = GravitationalWaveforms._xphm_hphc_scalar(f, setup, sys, ms21, ms33, ms44, m32)
        hp, hc = GravitationalWaveforms._xp_apply_zeta_rotation(hp, hc, setup.zeta_polarization)

        tc_phase = -2 * pi * f * t_c
        shift = cis(tc_phase)
        hp *= shift
        hc *= shift

        h[i] = Fplus * hp + Fcross * hc
    end
    return h
end

function _manual_ll_phenomxphm(freqs, physical::AbstractVector, data::LikelihoodData)
    h_det = _manual_h_det_phenomxphm(freqs, physical, data.detector, data.gps_time)
    df = freqs[2] - freqs[1]
    return -2 * df * sum(abs2.(h_det .- data.strain) ./ data.psd)
end

# Aligned-spin (`IMRPhenomXHM`) reference, for the aligned-spin-limit sanity check below --
# mirrors `KernelFused._fused_waveform_likelihood_kernel_phenomxhm!`'s body directly (11-row
# convention, own `t_c` post-multiplication), kept local per this file's self-contained style.
function _manual_h_det_phenomxhm(freqs::AbstractVector, physical11::AbstractVector, detector, gps_time)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = physical11
    mu = (m1 * m2) / (m1 + m2)
    mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
    eta = (m1 * m2) / (m1 + m2)^2

    Fplus, Fcross = GravitationalWaveData.compute_antenna_response(detector, ra, dec, psi, gps_time)

    f_ref = freqs[1]
    sys = GravitationalWaveforms._xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, phase)
    ms21 = GravitationalWaveforms._xhm_mode_setup(2, 1, sys)
    ms33 = GravitationalWaveforms._xhm_mode_setup(3, 3, sys)
    ms44 = GravitationalWaveforms._xhm_mode_setup(4, 4, sys)
    m32 = GravitationalWaveforms._xhm_mode32_setup(sys)
    piHalf = pi / 2

    h = Vector{ComplexF64}(undef, length(freqs))
    for (i, f) in enumerate(freqs)
        hp, hc = GravitationalWaveforms._xhm_hphc_scalar(f, sys, ms21, ms33, ms44, m32, theta_jn, piHalf)
        shift = cis(-2 * pi * f * t_c)
        h[i] = Fplus * (hp * shift) + Fcross * (hc * shift)
    end
    return h
end

@testset "IMRPhenomXPHM single-detector kernel" begin
    ntime = 48
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    gps = 1126259462.4
    detector = GravitationalWaveData.H1

    # Genuinely precessing, asymmetric-mass-ratio configuration (HOM content matters, mirroring
    # test_phenomxhm.jl's own injection-recovery config) -- same in-plane-spin magnitudes as
    # test_phenomxp.jl's own `ref_physical` (comfortably inside the unit-norm constraint).
    ref_physical = [32.0, 8.0, 0.1, -0.05, 0.5, 0.01, 0.4, 1.0, -0.3, 1.2, 0.6, 0.15, -0.1, 0.05, 0.08]

    @testset "kernel agrees with a manual host-side reference" begin
        data0 = LikelihoodData(freqs, zeros(ComplexF64, ntime), ones(Float64, ntime), detector, gps)
        strain = _manual_h_det_phenomxphm(freqs, ref_physical, detector, gps) .+
                 1e-24 .* (randn(ntime) .+ im .* randn(ntime))
        psd = fill(1e-46, ntime)
        data = LikelihoodData(freqs, strain, psd, detector, gps)
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)

        ll_manual = _manual_ll_phenomxphm(freqs, ref_physical, data)
        ll_kernel = zeros(Float64, 1)
        batched_log_likelihood_kernel!(ll_kernel, reshape(ref_physical, 15, 1), freqs, cfg, data; model=IMRPhenomXPHM())
        @test isapprox(ll_kernel[1], ll_manual; rtol=1e-8, atol=1e-8)
    end

    @testset "noise-free injection recovery: peak at injection, single-parameter perturbations strictly decrease ll" begin
        # Same rationale as test_phenomxp.jl's own analogous testset: injecting and recovering
        # with the SAME IMRPhenomXPHM kernel sidesteps the constant exp(-2i*zeta) phase / mode-
        # normalization conventions entirely, so this is the primary correctness check here.
        strain = _manual_h_det_phenomxphm(freqs, ref_physical, detector, gps)
        psd = fill(1e-46, ntime)
        data = LikelihoodData(freqs, strain, psd, detector, gps)
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)

        ll0 = zeros(Float64, 1)
        batched_log_likelihood_kernel!(ll0, reshape(ref_physical, 15, 1), freqs, cfg, data; model=IMRPhenomXPHM())
        @test isapprox(ll0[1], 0.0; atol=1e-6)

        row_names = ["m1", "m2", "chi1", "chi2", "dL", "t_c", "phase", "ra", "dec", "theta_jn", "psi",
                     "chi1x", "chi1y", "chi2x", "chi2y"]
        perturbations = [0.5, 0.5, 0.02, 0.02, 0.05, 0.002, 0.05, 0.05, 0.05, 0.05, 0.05,
                          0.02, 0.02, 0.02, 0.02]
        for i in 1:15
            p2 = copy(ref_physical)
            p2[i] += perturbations[i]
            ll2 = zeros(Float64, 1)
            batched_log_likelihood_kernel!(ll2, reshape(p2, 15, 1), freqs, cfg, data; model=IMRPhenomXPHM())
            @test ll2[1] < ll0[1]
        end
    end

    @testset "aligned-spin limit: |h_det| matches IMRPhenomXHM's kernel path (phase-independent check)" begin
        # chi1x=chi1y=chi2x=chi2y=0: like IMRPhenomXP vs IMRPhenomXAS, IMRPhenomXPHM's twisted-up
        # h_det and IMRPhenomXHM's aligned-spin h_det differ by a constant exp(-2i*zeta_polarization)
        # phase factor even at zero in-plane spin, so only |h_det| is directly comparable.
        physical_xphm = [32.0, 8.0, 0.1, -0.05, 0.5, 0.01, 0.4, 1.0, -0.3, 1.2, 0.6, 0.0, 0.0, 0.0, 0.0]
        physical_xhm = physical_xphm[1:11]

        # NNLO (102) property: LAL's 223 (MSA) does not reduce to XHM for aligned spins
        h_xphm = _manual_h_det_phenomxphm(freqs, physical_xphm, detector, gps; version = 102)
        h_xhm = _manual_h_det_phenomxhm(freqs, physical_xhm, detector, gps)

        @test isapprox(abs.(h_xphm), abs.(h_xhm); rtol=1e-6, atol=1e-30)
    end

    @testset "row-count assertion: IMRPhenomXPHM requires 15-row params" begin
        data = LikelihoodData(freqs, zeros(ComplexF64, ntime), ones(Float64, ntime), detector, gps)
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)
        ll = zeros(Float64, 1)
        params11 = reshape(ref_physical[1:11], 11, 1)
        @test_throws AssertionError batched_log_likelihood_kernel!(ll, params11, freqs, cfg, data; model=IMRPhenomXPHM())
    end
end

@testset "IMRPhenomXPHM multi-detector kernel" begin
    # Manual, host-side multi-detector IMRPhenomXPHM reference: mirrors `_manual_h_det_phenomxphm`
    # above, adding the per-detector `GravitationalWaveData.time_delay_from_geocenter`-shifted `t_c` -- same
    # relationship `test_phenomxp.jl`'s own `_manual_h_det_phenomxp_multi` bears to its
    # single-detector `_manual_h_det_phenomxp` (host-side, non-differentiated reference code, so
    # calling the real `GravitationalWaveData` functions directly is fine here, unlike inside the kernel body).
    function _manual_h_det_phenomxphm_multi(freqs::AbstractVector, physical::AbstractVector, detector, gps_time, f_ref)
        m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi, chi1x, chi1y, chi2x, chi2y = physical
        mu = (m1 * m2) / (m1 + m2)
        mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
        eta = (m1 * m2) / (m1 + m2)^2
        delay = GravitationalWaveData.time_delay_from_geocenter(detector, ra, dec, gps_time)

        Fplus, Fcross = GravitationalWaveData.compute_antenna_response(detector, ra, dec, psi, gps_time)

        setup = GravitationalWaveforms._xp_precession_setup(m1, m2, chi1x, chi1y, chi1, chi2x, chi2y, chi2,
                                                  f_ref, theta_jn, phase; prec_version = Val(223))
        final_spin = setup.final_spin   # IMRPhenomXPHM() = PrecVersion 223
        sys = GravitationalWaveforms._xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, zero(phase);
                                             final_spin_override = final_spin)
        ms21 = GravitationalWaveforms._xhm_mode_setup(2, 1, sys)
        ms33 = GravitationalWaveforms._xhm_mode_setup(3, 3, sys)
        ms44 = GravitationalWaveforms._xhm_mode_setup(4, 4, sys)
        m32 = GravitationalWaveforms._xhm_mode32_setup(sys)

        h = Vector{ComplexF64}(undef, length(freqs))
        for (i, f) in enumerate(freqs)
            hp, hc = GravitationalWaveforms._xphm_hphc_scalar(f, setup, sys, ms21, ms33, ms44, m32)
            hp, hc = GravitationalWaveforms._xp_apply_zeta_rotation(hp, hc, setup.zeta_polarization)

            shift = cis(-2 * pi * f * (t_c + delay))
            h[i] = Fplus * (hp * shift) + Fcross * (hc * shift)
        end
        return h
    end

    ntime = 40
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    gps = 1126259462.4
    dets = [GravitationalWaveData.H1, GravitationalWaveData.L1]
    f_ref = freqs[1]  # both detectors share this grid here, matching the kernel's own convention
    ref_physical = [32.0, 8.0, 0.1, -0.05, 0.5, 0.01, 0.4, 1.0, -0.3, 1.2, 0.6, 0.15, -0.1, 0.05, 0.08]

    @testset "agrees with a manual host-side multi-detector reference" begin
        strains = [_manual_h_det_phenomxphm_multi(freqs, ref_physical, d, gps, f_ref) .+
                   1e-24 .* (randn(ntime) .+ im .* randn(ntime)) for d in dets]
        psds = [fill(1e-46, ntime) for _ in dets]
        data = [LikelihoodData(freqs, strains[i], psds[i], dets[i], gps) for i in eachindex(dets)]
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)

        ll_manual = sum(begin
            h = _manual_h_det_phenomxphm_multi(freqs, ref_physical, dets[i], gps, f_ref)
            df = freqs[2] - freqs[1]
            -2 * df * sum(abs2.(h .- data[i].strain) ./ data[i].psd)
        end for i in eachindex(dets))

        ll_kernel = zeros(Float64, 1)
        batched_log_likelihood_kernel!(ll_kernel, reshape(ref_physical, 15, 1), cfg, data; model=IMRPhenomXPHM())
        @test isapprox(ll_kernel[1], ll_manual; rtol=1e-8, atol=1e-8)
    end

    @testset "noise-free injection recovery: peak at injection, single-parameter perturbations strictly decrease ll" begin
        strains = [_manual_h_det_phenomxphm_multi(freqs, ref_physical, d, gps, f_ref) for d in dets]
        psds = [fill(1e-46, ntime) for _ in dets]
        data = [LikelihoodData(freqs, strains[i], psds[i], dets[i], gps) for i in eachindex(dets)]
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)

        ll0 = zeros(Float64, 1)
        batched_log_likelihood_kernel!(ll0, reshape(ref_physical, 15, 1), cfg, data; model=IMRPhenomXPHM())
        @test isapprox(ll0[1], 0.0; atol=1e-6)

        perturbations = [0.5, 0.5, 0.02, 0.02, 0.05, 0.002, 0.05, 0.05, 0.05, 0.05, 0.05,
                          0.02, 0.02, 0.02, 0.02]
        for i in 1:15
            p2 = copy(ref_physical)
            p2[i] += perturbations[i]
            ll2 = zeros(Float64, 1)
            batched_log_likelihood_kernel!(ll2, reshape(p2, 15, 1), cfg, data; model=IMRPhenomXPHM())
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
        @test_throws ArgumentError batched_log_likelihood_kernel!(ll, params, cfg_small, data_bad; model=IMRPhenomXPHM())
    end

    @testset "row-count assertion: IMRPhenomXPHM multi-detector requires 15-row params" begin
        ntime_small = 8
        freqs_small = collect(20.0:1.0:(20.0 + ntime_small - 1))
        strain_small = zeros(ComplexF64, ntime_small)
        psd_small = ones(Float64, ntime_small)
        data = [LikelihoodData(freqs_small, strain_small, psd_small, GravitationalWaveData.H1, gps),
                LikelihoodData(freqs_small, strain_small, psd_small, GravitationalWaveData.L1, gps)]
        cfg_small = PEConfig{Float64}(ntime_small, 1, 0.01, 1)
        ll = zeros(Float64, 1)
        params11 = reshape(ref_physical[1:11], 11, 1)
        @test_throws AssertionError batched_log_likelihood_kernel!(ll, params11, cfg_small, data; model=IMRPhenomXPHM())
    end
end
