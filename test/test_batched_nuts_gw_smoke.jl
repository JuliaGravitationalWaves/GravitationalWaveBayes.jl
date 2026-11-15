using Test
using GravitationalWaveBayes
using GravitationalWaveBayes.BatchedNUTS: _reparam_to_physical!, gw_logp_and_grad
using Random

@testset "run_gw150914_estimation end-to-end smoke test" begin
    # Tiny synthetic problem (same scale as test_enzyme_ka.jl) so the tree-building loop's
    # repeated Enzyme.autodiff calls stay fast: IMRPhenomD's Enzyme derivative rule compiles once
    # (first call) and every subsequent call within this test just runs the compiled code. Small
    # nsteps/nwarmup/max_treedepth keep the total number of leapfrog (== Enzyme.autodiff) calls
    # bounded -- this is a smoke test for "runs end-to-end and produces finite output", not a
    # convergence/recovery check (that's the toy-Gaussian test's job).
    #
    # `state.positions` is now (11, nwalkers) -- the full sampled convention (mc, q, chi1, chi2,
    # dL, t_c, phase, ra, sin_dec, cos_theta_jn, psi).
    ntime = 16
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    strain = zeros(ComplexF64, ntime)
    psd = ones(Float64, ntime)
    data = LikelihoodData(freqs, strain, psd)

    rng = Random.Xoshiro(1)
    state = run_gw150914_estimation(data; nwalkers=2, nsteps=3, step_size=1e-4,
                                     nwarmup=2, max_treedepth=3, rng=rng)

    @test size(state.positions) == (11, 2)
    @test all(isfinite, state.positions)
    @test all(isfinite, state.logp)
end

@testset "sin_dec/cos_theta_jn near the reparametrization boundary" begin
    # Task-flagged risk: dec = +-pi/2 (sin_dec = +-1) and theta_jn = 0/pi (cos_theta_jn = +-1)
    # are coordinate singularities of the asin/acos reparametrization in
    # BatchedNUTS._reparam_to_physical! -- asin/acos throw a DomainError outside [-1,1], and
    # (found empirically while validating the Enzyme gradients)
    # their DERIVATIVE is genuinely unbounded (mathematically, not a bug) exactly AT +-1, which is
    # itself a valid point of the closed [-1,1] prior support (not just a transient leapfrog
    # overshoot) -- an earlier version of `_clamp_unit` that clamped only to the literal [-1,1]
    # interval let `Enzyme.autodiff` return an `Inf` gradient entry at exactly sin_dec=1.0, caught
    # by this very testset. `_clamp_unit` now clamps to a `sqrt(eps(T))`-margin strictly inside
    # [-1,1] (see its docstring for the reasoning), so gradients stay large-but-FINITE everywhere,
    # including exactly at the nominal +-1 boundary. This test exercises exactly that: values at,
    # just inside, and just outside [-1,1].
    ntime = 16
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    data = LikelihoodData(freqs, zeros(ComplexF64, ntime), ones(Float64, ntime))
    cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)

    logp_and_grad! = gw_logp_and_grad(data, cfg, freqs)

    for (sin_dec, cos_theta_jn) in [(1.0, -1.0), (-1.0, 1.0), (1.0 + 1e-7, -1.0 - 1e-7), (0.999999, -0.999999)]
        positions = reshape([30.0, 0.8, 0.1, 0.1, 0.4, 0.0, 0.3, 1.2, sin_dec, cos_theta_jn, 0.7], 11, 1)
        grad_out = zeros(Float64, 11, 1)
        logp_out = zeros(Float64, 1)
        # Must not throw (DomainError from an un-clamped asin/acos would previously crash here).
        logp_and_grad!(grad_out, logp_out, positions, trues(1))
        @test all(isfinite, grad_out)
        @test isfinite(logp_out[1]) || logp_out[1] == -Inf  # -Inf is the CORRECT prior-rejection outcome when out of [-1,1]
    end

    # Physical conversion itself: clamped values land very near (not exactly at, per
    # `_clamp_unit`'s margin) the poles, not NaN/complex -- `atol` here is generous enough to
    # cover both the margin itself and the input's own overshoot.
    physical = zeros(Float64, 11, 1)
    _reparam_to_physical!(physical, reshape([30.0, 0.8, 0.1, 0.1, 0.4, 0.0, 0.3, 1.2, 1.0000003, -1.0000003, 0.7], 11, 1))
    @test physical[9, 1] ≈ pi / 2 atol=1e-3      # dec, from clamped sin_dec ~ 1
    @test physical[10, 1] ≈ pi atol=1e-3         # theta_jn, from clamped cos_theta_jn ~ -1
    @test all(isfinite, physical)
end

@testset "run_gw150914_estimation with adapt_mass=true (diagonal mass-matrix adaptation)" begin
    # Smoke test only: the mass-matrix machinery through the REAL 11-row sampled space with bounds
    # reflection/wrapping and the Enzyme gradient. Statistical correctness of the adaptation itself
    # is checked on analytic targets in test_nuts_core_mass_matrix.jl. nwarmup=30 >= 20 so at least
    # one metric update actually fires (Stan schedule falls back to 15%/10% buffers here).
    ntime = 16
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    data = LikelihoodData(freqs, zeros(ComplexF64, ntime), ones(Float64, ntime))

    state = run_gw150914_estimation(data; nwalkers=3, nsteps=36, step_size=1e-4,
                                     nwarmup=30, max_treedepth=3, rng=Random.Xoshiro(2),
                                     adapt_mass=true)
    @test size(state.positions) == (11, 3)
    @test all(isfinite, state.positions)
    @test all(isfinite, state.logp)
end
