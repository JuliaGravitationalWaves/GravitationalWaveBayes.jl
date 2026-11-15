using Test
using GravitationalWaveBayes
using GravitationalWaveBayes: analytic_reference_metric, informed_initial_positions, analytic_snr2, analytic_beta_min
using GravitationalWaveBayes: DenseMetric, LikelihoodData
using GravitationalWaveforms: IMRPhenomXAS, IMRPhenomXHM
using GravitationalWaveBayes.AnalyticInit: _wrap_and_clamp!
using GravitationalWaveBayes.BatchedNUTS: _sampled_space_bounds, _fill_gw150914_base_point!
using LinearAlgebra
using Random, Statistics

@isdefined(RUN_SLOW_ENZYME_TESTS) || (const RUN_SLOW_ENZYME_TESTS = get(ENV, "GRAVITATIONALWAVEBAYES_RUN_SLOW_TESTS", "false") == "true")

# ---------------------------------------------------------------------------------------------
# Analytic pre-computation from a reference waveform. `informed_initial_
# positions`/`analytic_beta_min` are pure linear algebra -- fast, always run. `analytic_reference_
# metric`/`analytic_snr2` both go through `gw_logp_and_grad`, which always computes an Enzyme
# gradient even when only `logp` is used -- "slow" by this codebase's own existing convention
# (every other Enzyme-gradient test here is gated the same way), so gated behind
# GRAVITATIONALWAVEBAYES_RUN_SLOW_TESTS.
# ---------------------------------------------------------------------------------------------

# Minimal synthetic single-detector fixture. `analytic_snr2`/`analytic_reference_metric` never
# read `strain` for anything other than the noise (dd/dh) terms `gw_logp_and_grad` computes
# internally -- `analytic_snr2` zeroes it out itself -- so a flat PSD and placeholder strain are
# enough; no real injection or real GW150914 data needed for these checks.
function _toy_data(; psd_scale::Float64=1.0)
    freqs = collect(20.0:1.0:200.0)
    n = length(freqs)
    psd = fill(psd_scale, n)
    strain = zeros(ComplexF64, n)
    return LikelihoodData(freqs, strain, psd), freqs
end

@testset "AnalyticInit" begin

    @testset "informed_initial_positions: dispersion matches the metric's covariance" begin
        # A deliberately anisotropic, non-diagonal metric (same rotated-covariance construction
        # style as test_nuts_core_dense_mass.jl) so no axis-aligned bug could pass by accident.
        Q = Matrix(qr(randn(MersenneTwister(1), 11, 11)).Q)
        sd = [1e-3, 1e-2, 0.05, 0.05, 0.02, 1e-4, 0.01, 0.01, 0.01, 0.01, 0.01]
        Minv = Q * Diagonal(sd .^ 2) * Q'
        Minv = (Minv .+ Minv') ./ 2
        metric = DenseMetric(Minv)

        # Deliberately NOT `_fill_gw150914_base_point!`'s real point: it sets `phase=0`, which
        # sits exactly ON that row's periodic wrap boundary (`_PHASE_PRIOR_BOUNDS` starts at 0),
        # so even a tiny perturbation wraps ~half the time and corrupts this row's empirical
        # covariance -- a real property of wrapping at a boundary-sitting reference point, not a
        # bug, but irrelevant to what THIS test isolates (the linear algebra of the dispersion
        # itself; boundary handling is covered separately below). Use each row's box MIDPOINT
        # instead, interior to every row (periodic or not) regardless of `scale`.
        bounds0 = _sampled_space_bounds(Float64, 11)
        ref = (bounds0.lo .+ bounds0.hi) ./ 2
        nwalkers = 500_000
        scale = 0.01   # tiny relative to the prior box, so wrap/clamp never fires -- isolates the
                        # covariance-matching property from the (separately tested) boundary handling
        positions = informed_initial_positions(Float64, nwalkers, metric, ref;
                                                rng=MersenneTwister(2), scale=scale)
        @test size(positions) == (11, nwalkers)
        deviations = positions .- ref
        empirical_cov = cov(deviations; dims=2)
        # Monte Carlo comparison (500k draws) -- rtol=0.1 gives comfortable margin over the
        # ~0.2% per-entry sampling noise at this n while still being a real, discriminating check
        # (a wrong scale factor, e.g. Minv vs Minv^-1 or a stray factor of `scale` vs `scale^2`,
        # would miss by 100%+, not a few percent).
        @test isapprox(empirical_cov, scale^2 .* Minv; rtol=0.1)
    end

    @testset "informed_initial_positions: periodic wrap / box clamp still applied" begin
        # A huge metric scale forces draws far outside the prior box on every row; confirm every
        # returned coordinate still lands inside [lo, hi] (wrapped for periodic rows, clamped
        # otherwise) exactly like `_gw150914_initial_positions`'s own post-processing.
        metric = DenseMetric(Matrix(100.0 * I, 11, 11))
        ref = zeros(11); GravitationalWaveBayes.BatchedNUTS._fill_gw150914_base_point!(reshape(ref, 11, 1))
        positions = informed_initial_positions(Float64, 500, metric, ref; rng=MersenneTwister(3), scale=1.0)
        bounds = _sampled_space_bounds(Float64, 11)
        for i in 1:11, w in axes(positions, 2)
            @test bounds.lo[i] - 1e-9 <= positions[i, w] <= bounds.hi[i] + 1e-9
        end
    end

    @testset "informed_initial_positions: metric=nothing falls back to independent dispersion" begin
        ref = zeros(11); GravitationalWaveBayes.BatchedNUTS._fill_gw150914_base_point!(reshape(ref, 11, 1))
        positions = informed_initial_positions(Float64, 1000, nothing, ref; rng=MersenneTwister(4))
        @test size(positions) == (11, 1000)
        @test all(isfinite, positions)
        # Walkers are dispersed, not identical (the whole point of dispersion in the first place).
        @test std(positions[1, :]) > 0
        bounds = _sampled_space_bounds(Float64, 11)
        for i in 1:11, w in axes(positions, 2)
            @test bounds.lo[i] - 1e-9 <= positions[i, w] <= bounds.hi[i] + 1e-9
        end
    end

    @testset "analytic_beta_min" begin
        @test analytic_beta_min(1000.0, 11) ≈ 11 / 1000.0
        @test analytic_beta_min(1.0, 11) == 1.0            # clamped to the ceiling
        @test analytic_beta_min(1e9, 11) == 1e-3            # clamped to the default floor
        @test analytic_beta_min(1e9, 11; floor=1e-10) ≈ 11 / 1e9
        @test_throws ArgumentError analytic_beta_min(0.0, 11)
        @test_throws ArgumentError analytic_beta_min(-5.0, 11)
    end

    if RUN_SLOW_ENZYME_TESTS
        @testset "analytic_snr2: scales as 1/distance^2 (exact physical property)" begin
            data, freqs = _toy_data()
            ref = zeros(11); GravitationalWaveBayes.BatchedNUTS._fill_gw150914_base_point!(reshape(ref, 11, 1))
            ref = vec(ref)
            dL_row = 5
            ref2x = copy(ref); ref2x[dL_row] *= 2
            snr2_a = analytic_snr2(data, freqs, ref; model=IMRPhenomXAS())
            snr2_b = analytic_snr2(data, freqs, ref2x; model=IMRPhenomXAS())
            @test snr2_a > 0 && isfinite(snr2_a)
            @test isapprox(snr2_b, snr2_a / 4; rtol=1e-6)
        end

        @testset "analytic_snr2: scales as 1/psd (exact physical property)" begin
            data1, freqs = _toy_data(; psd_scale=1.0)
            data3, _ = _toy_data(; psd_scale=3.0)
            ref = vec(zeros(11)); GravitationalWaveBayes.BatchedNUTS._fill_gw150914_base_point!(reshape(ref, 11, 1))
            snr2_1 = analytic_snr2(data1, freqs, ref; model=IMRPhenomXAS())
            snr2_3 = analytic_snr2(data3, freqs, ref; model=IMRPhenomXAS())
            @test isapprox(snr2_3, snr2_1 / 3; rtol=1e-6)
        end

        @testset "analytic_reference_metric: cheap model (IMRPhenomXAS) end-to-end" begin
            data, freqs = _toy_data()
            metric, ref_point = analytic_reference_metric(data, freqs; model=IMRPhenomXAS())
            @test length(ref_point) == 11
            if metric !== nothing
                @test size(metric.Minv) == (11, 11)
                @test metric.Minv ≈ metric.Minv'
                @test isposdef(Symmetric(metric.Minv))
            end
        end

        @testset "analytic_reference_metric: default model (IMRPhenomXHM) does not error" begin
            data, freqs = _toy_data()
            metric, ref_point = analytic_reference_metric(data, freqs)   # model=IMRPhenomXHM() default
            @test length(ref_point) == 11
            @test metric === nothing || size(metric.Minv) == (11, 11)
        end
    end
end
