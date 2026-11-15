using Test
using GravitationalWaveBayes
using KernelAbstractions
using GravitationalWaveforms: Ampl, Phi, IMRPhenomD, IMRPhenomXAS, GMsun_over_c3
using GravitationalWaveBayes.KernelFused: _phi_t0
using GravitationalWaveData

# Manual, host-side reference implementation of the full 11-parameter physical forward model
# (distance scaling, t_c/phase shift via Phi's t0/phiRef, LALSuite-ported plus/cross by
# theta_jn, single-detector antenna response), independent of KernelFused.jl's kernel bodies --
# used throughout this file to validate `batched_log_likelihood_kernel!` against a
# straightforward, non-fused computation.
# Reference template phase in the kernels' physical convention (KernelFused `_PHASE_CONVENTION`): the
# XAS family's Phi has the data's sign and only the time term flips; the others are conjugated.
_ref_phase(model, freqs, mc, eta, chi1, chi2, t0, M_sec, phase) = model isa IMRPhenomXAS ?
    Phi(model, freqs, mc, eta, chi1, chi2; t0=-_phi_t0(model, t0, M_sec), phiRef=phase) :
    .-Phi(model, freqs, mc, eta, chi1, chi2; t0=_phi_t0(model, t0, M_sec), phiRef=phase)

function _manual_h_det(model, freqs, physical::AbstractVector, detector, gps_time)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = physical
    mu = (m1 * m2) / (m1 + m2)
    mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
    eta = (m1 * m2) / (m1 + m2)^2
    M_sec = (mc / eta^0.6) * GMsun_over_c3
    t0 = -2 * pi * t_c / M_sec

    amp = Ampl(model, freqs, mc, eta, chi1, chi2, dL)
    phase_vals = _ref_phase(model, freqs, mc, eta, chi1, chi2, t0, M_sec, phase)
    h = amp .* exp.(im .* phase_vals)

    pfac = 0.5 * (1 + cos(theta_jn)^2)
    cfac = cos(theta_jn)
    Fplus, Fcross = GravitationalWaveData.compute_antenna_response(detector, ra, dec, psi, gps_time)
    combo = Fplus * pfac - im * Fcross * cfac
    return combo .* h
end

function _manual_ll(model, freqs, physical::AbstractVector, data::LikelihoodData)
    h_det = _manual_h_det(model, freqs, physical, data.detector, data.gps_time)
    df = freqs[2] - freqs[1]
    return -2 * df * sum(abs2.(h_det .- data.strain) ./ data.psd)
end

@testset "KernelAbstractions fused waveform+likelihood kernel" begin
    ntime = 96
    nwalkers = 4
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))

    # Non-trivial (non-zero) strain/PSD so the residual isn't a degenerate near-zero number and
    # isapprox tolerances are meaningful. Built from the manual 11-parameter reference above
    # (rather than the old 4-param batched_log_likelihood!) at one reference physical point.
    ref_physical = [32.0, 28.0, 0.1, -0.05, 0.5, 0.01, 0.4, 1.0, -0.3, 0.9, 0.6]
    data0 = LikelihoodData(freqs, zeros(ComplexF64, ntime), ones(Float64, ntime))
    strain = _manual_h_det(IMRPhenomD(), freqs, ref_physical, data0.detector, data0.gps_time) .+
             1e-24 .* (randn(ntime) .+ im .* randn(ntime))
    psd = fill(1e-46, ntime)
    data = LikelihoodData(freqs, strain, psd)
    cfg = PEConfig{Float64}(ntime, nwalkers, 0.01, 1)

    @testset "agrees with a manual host-side reference across several parameter sets" begin
        param_sets = [
            [30.0 40.0 55.0 20.0;
             25.0 20.0 15.0 18.0;
             0.5 -0.2 0.1 0.9;
             0.3 0.1 -0.4 -0.8;
             0.4 1.0 0.6 2.0;
             0.01 -0.02 0.0 0.05;
             0.3 1.5 0.0 4.0;
             1.0 3.0 5.0 0.5;
             -0.5 0.2 0.9 -1.2;
             0.5 1.5 2.5 3.0;
             0.2 1.0 2.0 2.9],
            [10.0 100.0 60.0 45.0;
             9.0 80.0 59.0 44.0;
             0.0 0.99 -0.99 0.5;
             0.0 -0.99 0.99 -0.5;
             0.1 3.5 1.5 0.8;
             -0.05 0.0 0.03 -0.01;
             0.0 6.0 3.0 1.0;
             0.0 6.2 2.0 4.5;
             -1.5 1.5 0.0 0.7;
             0.0 3.1 1.0 2.5;
             0.0 3.1 0.5 1.5],
        ]

        for params in param_sets
            ll_manual = [_manual_ll(IMRPhenomD(), freqs, params[:, j], data) for j in 1:nwalkers]

            ll_new = zeros(Float64, nwalkers)
            batched_log_likelihood_kernel!(ll_new, params, freqs, cfg, data)

            @test isapprox(ll_manual, ll_new; rtol = 1e-8, atol = 1e-8)
        end
    end

    @testset "explicit KernelAbstractions.CPU() backend" begin
        params = zeros(Float64, 11, nwalkers)
        params[1, :] .= [30.0, 40.0, 55.0, 20.0]
        params[2, :] .= [25.0, 20.0, 15.0, 18.0]
        params[3, :] .= [0.5, -0.2, 0.1, 0.9]
        params[4, :] .= [0.3, 0.1, -0.4, -0.8]
        params[5, :] .= [0.4, 1.0, 0.6, 2.0]
        params[6, :] .= [0.01, -0.02, 0.0, 0.05]
        params[7, :] .= [0.3, 1.5, 0.0, 4.0]
        params[8, :] .= [1.0, 3.0, 5.0, 0.5]
        params[9, :] .= [-0.5, 0.2, 0.9, -1.2]
        params[10, :] .= [0.5, 1.5, 2.5, 3.0]
        params[11, :] .= [0.2, 1.0, 2.0, 2.9]

        ll_manual = [_manual_ll(IMRPhenomD(), freqs, params[:, j], data) for j in 1:nwalkers]

        ll_new = zeros(Float64, nwalkers)
        batched_log_likelihood_kernel!(ll_new, params, freqs, cfg, data; backend = KernelAbstractions.CPU())

        @test isapprox(ll_manual, ll_new; rtol = 1e-8, atol = 1e-8)
        @test all(isfinite, ll_new)
    end

    @testset "generic over model (IMRPhenomXAS, not just IMRPhenomD)" begin
        params = zeros(Float64, 11, 2)
        params[1, :] .= [32.0, 50.0]
        params[2, :] .= [28.0, 45.0]
        params[3, :] .= [0.2, -0.3]
        params[4, :] .= [0.1, 0.25]
        params[5, :] .= [0.5, 1.2]
        params[6, :] .= [0.02, -0.01]
        params[7, :] .= [1.0, 2.5]
        params[8, :] .= [2.0, 4.5]
        params[9, :] .= [0.4, -0.6]
        params[10, :] .= [1.2, 2.1]
        params[11, :] .= [0.3, 1.4]

        ll_kernel = zeros(Float64, 2)
        batched_log_likelihood_kernel!(ll_kernel, params, freqs, cfg, data; model = IMRPhenomXAS())

        ll_manual = [_manual_ll(IMRPhenomXAS(), freqs, params[:, j], data) for j in 1:2]

        @test isapprox(ll_kernel, ll_manual; rtol = 1e-8, atol = 1e-8)
    end

    @testset "theta_jn = 0 (face-on) reduces to the bare dominant-mode strain" begin
        # Direct check of the specific claim in KernelFused.jl's module docstring: at
        # theta_jn=0, pfac=cfac=1, so hplus == h exactly (the pre-Task-4 no-inclination
        # behavior). hcross = -im*h is nonzero, but at theta_jn=0 both plus and cross project
        # through the antenna response identically to `combo = Fplus - im*Fcross`, which this
        # test checks reproduces the SAME complex scalar as face-on in the manual reference
        # above (i.e. this isn't a redundant re-test of the manual reference itself, it's a
        # targeted check of the theta_jn=0 special case documented in the module docstring).
        params = reshape([32.0, 28.0, 0.1, -0.05, 0.5, 0.0, 0.0, 1.0, -0.3, 0.0, 0.6], 11, 1)
        ll_kernel = zeros(Float64, 1)
        batched_log_likelihood_kernel!(ll_kernel, params, freqs, cfg, data)

        m1, m2, chi1, chi2, dL = 32.0, 28.0, 0.1, -0.05, 0.5
        mu = (m1 * m2) / (m1 + m2)
        mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
        eta = (m1 * m2) / (m1 + m2)^2
        h = Ampl(IMRPhenomD(), freqs, mc, eta, chi1, chi2, dL) .* exp.(-im .* Phi(IMRPhenomD(), freqs, mc, eta, chi1, chi2))   # PhenomD enters conjugated
        Fplus, Fcross = GravitationalWaveData.compute_antenna_response(data.detector, 1.0, -0.3, 0.6, data.gps_time)
        h_det = (Fplus - im * Fcross) .* h
        df = freqs[2] - freqs[1]
        ll_expected = -2 * df * sum(abs2.(h_det .- data.strain) ./ data.psd)

        @test isapprox(ll_kernel[1], ll_expected; rtol = 1e-10)
    end
end

@testset "Multi-detector fused kernel (Phase 2b Task 5, H1+L1)" begin
    # Manual, host-side multi-detector reference: mirrors `_manual_h_det`/`_manual_ll` above, but
    # adds the per-detector `GravitationalWaveData.time_delay_from_geocenter`-shifted `t0` (calling the REAL
    # `GravitationalWaveData` functions directly, not the local scalar ports -- this is host-side non-
    # differentiated reference code, so the Enzyme-vs-`Detector`-struct concern documented in
    # KernelFused.jl's module docstring doesn't apply here).
    function _manual_h_det_multi(model, freqs, physical::AbstractVector, detector, gps_time)
        m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = physical
        mu = (m1 * m2) / (m1 + m2)
        mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
        eta = (m1 * m2) / (m1 + m2)^2
        M_sec = (mc / eta^0.6) * GMsun_over_c3
        delay = GravitationalWaveData.time_delay_from_geocenter(detector, ra, dec, gps_time)
        t0 = -2 * pi * (t_c + delay) / M_sec

        amp = Ampl(model, freqs, mc, eta, chi1, chi2, dL)
        phase_vals = _ref_phase(model, freqs, mc, eta, chi1, chi2, t0, M_sec, phase)
        h = amp .* exp.(im .* phase_vals)

        pfac = 0.5 * (1 + cos(theta_jn)^2)
        cfac = cos(theta_jn)
        Fplus, Fcross = GravitationalWaveData.compute_antenna_response(detector, ra, dec, psi, gps_time)
        combo = Fplus * pfac - im * Fcross * cfac
        return combo .* h
    end

    ntime = 64
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    gps = 1126259462.4
    dets = [GravitationalWaveData.H1, GravitationalWaveData.L1]
    ref_physical = [32.0, 28.0, 0.1, -0.05, 0.5, 0.01, 0.4, 1.0, -0.3, 0.9, 0.6]

    @testset "agrees with a manual host-side multi-detector reference" begin
        strains = [_manual_h_det_multi(IMRPhenomD(), freqs, ref_physical, d, gps) .+
                   1e-24 .* (randn(ntime) .+ im .* randn(ntime)) for d in dets]
        psds = [fill(1e-46, ntime) for _ in dets]
        data = [LikelihoodData(freqs, strains[i], psds[i], dets[i], gps) for i in eachindex(dets)]
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)

        for model in (IMRPhenomD(), IMRPhenomXAS())
            params = reshape(ref_physical, 11, 1)
            ll_manual = sum(begin
                h = _manual_h_det_multi(model, freqs, ref_physical, dets[i], gps)
                df = freqs[2] - freqs[1]
                -2 * df * sum(abs2.(h .- data[i].strain) ./ data[i].psd)
            end for i in eachindex(dets))

            ll_kernel = zeros(Float64, 1)
            batched_log_likelihood_kernel!(ll_kernel, params, cfg, data; model=model)
            @test isapprox(ll_kernel[1], ll_manual; rtol=1e-8, atol=1e-8)
        end
    end

    @testset "noise-free injection recovery: peak at injection, single-parameter perturbations strictly decrease ll" begin
        # Direct likelihood-surface check (not a NUTS run), same style as Task 4's single-detector
        # injection-recovery validation, extended to confirm the time-delay/per-detector antenna
        # response combination doesn't break the basic "peak at truth" property.
        strains = [_manual_h_det_multi(IMRPhenomD(), freqs, ref_physical, d, gps) for d in dets]
        psds = [fill(1e-46, ntime) for _ in dets]
        data = [LikelihoodData(freqs, strains[i], psds[i], dets[i], gps) for i in eachindex(dets)]
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)

        ll0 = zeros(Float64, 1)
        batched_log_likelihood_kernel!(ll0, reshape(ref_physical, 11, 1), cfg, data)
        @test isapprox(ll0[1], 0.0; atol=1e-8)

        row_names = ["m1", "m2", "chi1", "chi2", "dL", "t_c", "phase", "ra", "dec", "theta_jn", "psi"]
        perturbations = [0.5, 0.5, 0.02, 0.02, 0.05, 0.002, 0.05, 0.05, 0.05, 0.05, 0.05]
        for i in 1:11
            p2 = copy(ref_physical)
            p2[i] += perturbations[i]
            ll2 = zeros(Float64, 1)
            batched_log_likelihood_kernel!(ll2, reshape(p2, 11, 1), cfg, data)
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
        params = reshape(ref_physical, 11, 1)
        @test_throws ArgumentError batched_log_likelihood_kernel!(ll, params, cfg_small, data_bad)
    end
end

@testset "PEState device-residency (logp/energy generalized off host Vector)" begin
    positions = zeros(Float64, 4, 5)
    momenta = zeros(Float64, 4, 5)
    logp = zeros(Float64, 5)
    energy = zeros(Float64, 5)

    state = PEState(positions, momenta, logp, energy)
    @test state isa PEState{Float64, Matrix{Float64}, Vector{Float64}}
    @test size(state.positions) == (4, 5)
    @test size(state.logp) == (5,)
    @test size(state.energy) == (5,)

    # `compatible_array` is the device-dispatch point (Array on CPU-only environments, CuArray
    # under the CUDA extension when functional): building logp/energy through it, as
    # BatchedNUTS.run_gw150914_estimation now does, keeps them the same array family as
    # positions/momenta rather than hardcoding host Vector.
    T = Float64
    nwalkers = 6
    pos2 = compatible_array(T, (4, nwalkers))
    mom2 = compatible_array(T, (4, nwalkers))
    logp2 = compatible_array(T, (nwalkers,))
    energy2 = compatible_array(T, (nwalkers,))
    fill!(pos2, 0.0)
    fill!(mom2, 0.0)
    fill!(logp2, 0.0)
    fill!(energy2, 0.0)

    state2 = PEState(pos2, mom2, logp2, energy2)
    @test typeof(state2.positions) == typeof(pos2)
    @test typeof(state2.logp) == typeof(logp2)
    # On this CPU-only test environment compatible_array yields Array; the point of the
    # PEState{T,A,V} generalization is that this equality is exactly what lets A and V agree on
    # "device family" once a GPU backend is functional (A = CuMatrix{T}, V = CuVector{T}).
    @test typeof(state2.positions) <: AbstractArray{T,2}
    @test typeof(state2.logp) <: AbstractVector{T}

    # A 1-D array type mismatched with A's element type should still be rejected (V<:AbstractVector{T} is tied to the same T as A).
    @test_throws MethodError PEState(pos2, mom2, Float32.(logp2), Float32.(energy2))
end
