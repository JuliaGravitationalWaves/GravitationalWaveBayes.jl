using Test
using GravitationalWaveBayes
using GravitationalWaveBayes.BatchedNUTS: gw_logp_and_grad
using Random

# Parallel-tempering GW-integration smoke tests. Mirrors test_batched_nuts_gw_smoke.jl's
# own scope exactly ("runs end-to-end and produces finite output" for IMRPhenomD reverse-mode
# Enzyme, tiny synthetic data, NOT a convergence/recovery check) and its budget (unconditional,
# not gated behind GRAVITATIONALWAVEBAYES_RUN_SLOW_TESTS -- a single small IMRPhenomD reverse-mode Enzyme compile
# is the same order of cost that file already pays unconditionally).

@testset "gw_logp_and_grad `beta` tempers the likelihood term only (exact scaling check)" begin
    # Two lanes at the IDENTICAL position with different beta: logp_out[w] must equal
    # beta[w]*logL(theta) + logprior(theta) exactly (not approximately -- this is a plain scalar
    # multiply, not a numerically-approximated quantity), so with the SAME position in both
    # columns, logp_out[2]/logp_out[1] == beta[2]/beta[1] to full floating-point precision
    # whenever the (shared) log-prior term is exactly zero -- true here since this starting point
    # (same one `run_gw150914_estimation`'s `_gw150914_initial_positions` uses, GW150914-scale) is
    # comfortably interior to every prior bound. Same reasoning for the gradient (Enzyme.Reverse
    # path here, IMRPhenomD): d(beta*logL)/dtheta is an exact scalar multiple of d(logL)/dtheta
    # wherever the prior's own gradient is zero (see `_sum_ll`'s weighted-method docstring).
    ntime = 16
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    strain = zeros(ComplexF64, ntime)
    psd = ones(Float64, ntime)
    data = LikelihoodData(freqs, strain, psd)
    cfg = PEConfig{Float64}(ntime, 2, 0.01, 1)

    pos0 = [30.0, 0.8, 0.1, 0.1, 0.4, 0.0, 0.3, 1.2, -0.2, 0.1, 0.7]
    positions = repeat(pos0, 1, 2)
    beta = [1.0, 0.5]
    logp_and_grad! = gw_logp_and_grad(data, cfg, freqs; model=IMRPhenomD(), beta=beta)

    grad_out = zeros(11, 2)
    logp_out = zeros(2)
    logp_and_grad!(grad_out, logp_out, positions, trues(2))

    @test all(isfinite, grad_out)
    @test all(isfinite, logp_out)
    @test logp_out[2] / logp_out[1] ≈ 0.5 rtol=1e-10
    @test all(isapprox.(grad_out[:, 2] ./ grad_out[:, 1], 0.5; rtol=1e-8))

    # beta=nothing (default) must reproduce the exact pre-PT untempered closure bit-for-bit --
    # this feature must be a strict opt-in with zero effect when unused.
    logp_and_grad_untempered! = gw_logp_and_grad(data, cfg, freqs; model=IMRPhenomD())
    grad_untempered = zeros(11, 2)
    logp_untempered = zeros(2)
    logp_and_grad_untempered!(grad_untempered, logp_untempered, positions, trues(2))
    @test logp_out[1] == logp_untempered[1]     # beta=1.0 lane exactly matches untempered
    @test grad_out[:, 1] == grad_untempered[:, 1]
end

@testset "tempered_logp_and_grad (host-side beta + GWPrior) == gw_logp_and_grad(...; beta, prior)" begin
    # The wrapper that lets SMC/PT temper a closure that has no beta/prior options (the GPU-staged one)
    # must reproduce the CPU path exactly, including a non-flat prior (volume distance, component masses)
    # and a beta = 0 lane.
    ntime = 16
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    rng = Random.Xoshiro(3)
    data = LikelihoodData(freqs, 1e-22 .* complex.(randn(rng, ntime), randn(rng, ntime)), fill(1e-44, ntime))
    cfg = PEConfig{Float64}(ntime, 4, 0.01, 1)
    pr = GWPrior{Float64}(; bounds=Dict("mc" => (20.0, 40.0), "q" => (0.4, 1.0), "dL" => (0.1, 1.0)),
                          distance=:volume, masses=:components)
    positions = sample_prior(rng, pr, 4)
    betas = [1.0, 0.37, 0.0, 0.8]
    ref! = gw_logp_and_grad(data, cfg, freqs; model=IMRPhenomD(), beta=betas, prior=pr)
    wrapped! = GravitationalWaveBayes.BatchedNUTS.tempered_logp_and_grad(gw_logp_and_grad(data, cfg, freqs; model=IMRPhenomD()), betas, pr)
    g1 = zeros(11, 4); l1 = zeros(4); g2 = zeros(11, 4); l2 = zeros(4)
    ref!(g1, l1, positions, trues(4))
    wrapped!(g2, l2, positions, trues(4))
    @test all(isfinite, l1) && all(isfinite, g1)
    @test l2 ≈ l1 rtol=1e-12
    @test g2 ≈ g1 rtol=1e-10
    # Outside the prior support: -Inf with zero gradient, never NaN (also at beta = 0).
    out = copy(positions); out[1, :] .= 45.0
    wrapped!(g2, l2, out, trues(4))
    @test all(==(-Inf), l2) && all(iszero, g2)
end

@testset "batched_pt_nuts!/run_gw150914_pt_estimation end-to-end smoke test" begin
    ntime = 16
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    strain = zeros(ComplexF64, ntime)
    psd = ones(Float64, ntime)
    data = LikelihoodData(freqs, strain, psd)

    rng = Random.Xoshiro(7)
    ntemps = 4
    state, pt, cold_chain = run_gw150914_pt_estimation(data; ntemps=ntemps, nsteps=6, step_size=1e-4,
                                                         nwarmup=3, max_treedepth=3, swap_interval=2,
                                                         rng=rng)

    @test size(state.positions) == (11, ntemps)
    @test all(isfinite, state.positions)
    @test all(isfinite, state.logp)
    @test size(cold_chain) == (11, 6)
    @test all(isfinite, cold_chain)
    @test sort(Array(pt.betas)) ≈ sort(pt.beta_ladder)  # betas always a permutation of the ladder
    @test all(>=(0), pt.nattempt)

    @test_throws ArgumentError batched_pt_nuts!(state, data,
        PEConfig{Float64}(ntime, ntemps, 1e-4, 2); ntemps=ntemps + 1)
end

@testset "run_gw150914_pt_estimation with adapt_mass=true (per-temperature-rank metric)" begin
    # Smoke only (finite output through the real 11-row sampled space, bounds and Enzyme gradient);
    # statistical correctness of the per-rank metric is checked on analytic targets with the real
    # PTLadderState/pt_swap! in test_nuts_core_mass_matrix.jl. nwarmup=30 >= 20 so a metric update fires.
    ntime = 16
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    data = LikelihoodData(freqs, zeros(ComplexF64, ntime), ones(Float64, ntime))
    state, pt, cold_chain = run_gw150914_pt_estimation(data; ntemps=3, nsteps=36, step_size=1e-4,
                                                         nwarmup=30, max_treedepth=3, swap_interval=2,
                                                         rng=Random.Xoshiro(8), adapt_mass=true)
    @test size(state.positions) == (11, 3)
    @test all(isfinite, state.positions)
    @test all(isfinite, state.logp)
    @test all(isfinite, cold_chain)
    @test sort(Array(pt.betas)) ≈ sort(pt.beta_ladder)
end
