using Test
using GravitationalWaveBayes
using GravitationalWaveBayes: PTLadderState, pt_swap!, nuts_sample!, NUTSConfig
using Random

# Pure NUTSCore-level tests (no Enzyme, no GW likelihood, no KernelAbstractions GPU backend --
# `PTLadderState`'s default `backend=KernelAbstractions.CPU()`) for the GPU-native
# parallel tempering primitives added to `NUTSCore.jl`: `PTLadderState` (temperature-ladder/lane
# bookkeeping) and `pt_swap!` (the swap-the-beta-LABEL kernel). Mirrors
# `test_batched_nuts_gaussian.jl`'s own "validate the generic engine on a toy target before
# pointing it at a GW likelihood" scoping -- runs unconditionally
# (fast, no Enzyme compile cost).

@testset "PTLadderState construction" begin
    pt = PTLadderState(6; T=Float64, beta_min=0.02)
    @test length(pt.beta_ladder) == 6
    @test pt.beta_ladder[1] == 1.0
    @test pt.beta_ladder[end] ≈ 0.02
    @test issorted(pt.beta_ladder; rev=true)  # descending: rank 1 = beta=1 ("cold")
    @test Array(pt.betas) == pt.beta_ladder   # every lane starts at its own rank's ladder value
    @test Array(pt.lane_of_rank) == 1:6       # identity permutation initially
    @test Array(pt.rank_of_lane) == 1:6
    @test pt.parity == 0
    @test length(pt.naccept) == length(pt.nattempt) == 5  # ntemps-1 adjacent pairs

    # ntemps=1: no adjacent pairs, but construction itself must not throw.
    pt1 = PTLadderState(1)
    @test pt1.beta_ladder == [1.0]
    @test isempty(pt1.naccept)

    @test_throws ArgumentError PTLadderState(4; beta_min=0.0)
    @test_throws ArgumentError PTLadderState(4; beta_min=1.5)
    @test_throws ArgumentError PTLadderState(0)
end

@testset "pt_swap! acceptance formula and label-not-state swapping" begin
    # Lane 4 has a MUCH better raw log-likelihood than every other lane: the standard PT
    # acceptance formula (see NUTSCore.jl's `_pt_swap_kernel!` comment) should pull it toward
    # beta=1 (the coldest/target rung) within a handful of rounds, and -- since only the LABEL
    # moves -- `pt.betas` (not any position array) is what changes.
    rng = Random.Xoshiro(1)
    pt = PTLadderState(4; T=Float64, beta_min=0.1)
    ll = [0.0, 0.0, 0.0, 100.0]
    for _ in 1:6
        pt_swap!(pt, ll, rng)
    end
    @test Array(pt.betas)[4] == 1.0
    @test Array(pt.lane_of_rank)[1] == 4
    @test Array(pt.rank_of_lane)[4] == 1
    # betas is always a permutation of beta_ladder.
    @test sort(Array(pt.betas)) ≈ sort(pt.beta_ladder)
    @test all(pt.nattempt .> 0)

    # Degenerate ntemps=1 is a documented no-op, not an error.
    pt1 = PTLadderState(1)
    @test pt_swap!(pt1, [0.0], rng) === pt1
    @test Array(pt1.betas) == [1.0]

    # Identical log-likelihoods across every lane: `log_ratio == 0` for every pair, so the
    # Metropolis acceptance probability `min(1, exp(0))` is exactly 1 -- every proposed swap must
    # be accepted (there is nothing for the target to discriminate between), not merely likely.
    rng2 = Random.Xoshiro(99)
    pt_flat = PTLadderState(8; T=Float64, beta_min=0.05)
    ll_flat = zeros(8)
    for _ in 1:20
        pt_swap!(pt_flat, ll_flat, rng2)
    end
    @test sum(pt_flat.naccept) == sum(pt_flat.nattempt)

    # Genuinely stochastic case: resample random (fixed-scale) per-lane log-likelihoods EVERY
    # round (mimicking that in real use `ll` changes between rounds as NUTS moves each lane) --
    # the resulting acceptance rate should be neither degenerately 0 nor 1, confirming `u`
    # actually gates the accept/reject decision rather than the kernel unconditionally
    # accepting/rejecting.
    rng3 = Random.Xoshiro(123)
    pt_rand = PTLadderState(8; T=Float64, beta_min=0.05)
    for _ in 1:400
        pt_swap!(pt_rand, randn(rng3, 8) .* 0.5, rng3)
    end
    rate_rand = sum(pt_rand.naccept) / sum(pt_rand.nattempt)
    @test 0.02 < rate_rand < 0.98
end

# Small inline log-sum-exp helpers for a toy 1-D bimodal (two well-separated Gaussians) target --
# used to test the swap machinery wired into a REAL NUTS run via `NUTSCore.nuts_sample!`'s new
# `callback` hook, exactly the mechanism `BatchedNUTS.batched_pt_nuts!` uses for the GW likelihood.
# `log_L_tempered = beta * log_L` is applied here
# directly on the toy target's raw log-likelihood, with no prior term (flat/improper, matching
# `make_gaussian_logp_and_grad`'s own toy-target convention in test_batched_nuts_gaussian.jl).
function _bimodal_raw_logL(x::T, mu1::T, mu2::T, sigma::T) where {T}
    d1 = x - mu1
    d2 = x - mu2
    l1 = -T(0.5) * (d1 / sigma)^2
    l2 = -T(0.5) * (d2 / sigma)^2
    m = max(l1, l2)
    return m + log(exp(l1 - m) + exp(l2 - m)), l1, l2, m
end

function _make_bimodal_logp_and_grad(mu1::T, mu2::T, sigma::T;
                                      beta::Union{Nothing,AbstractVector{T}}=nothing) where {T}
    return function (grad_out::AbstractMatrix{T}, logp_out::AbstractVector{T},
                      positions::AbstractMatrix{T}, active_mask)
        nwalkers = size(positions, 2)
        @inbounds for w in 1:nwalkers
            x = positions[1, w]
            logL, l1, l2, m = _bimodal_raw_logL(x, mu1, mu2, sigma)
            b = beta === nothing ? one(T) : beta[w]
            logp_out[w] = b * logL
            w1 = exp(l1 - m); w2 = exp(l2 - m)
            dlogL = (w1 * (-(x - mu1) / sigma^2) + w2 * (-(x - mu2) / sigma^2)) / (w1 + w2)
            grad_out[1, w] = b * dlogL
        end
        return nothing
    end
end

@testset "Parallel-tempered NUTS escapes a bimodal target that traps plain NUTS" begin
    # 12-sigma-separated modes: a gradient-following HMC/NUTS trajectory started in one mode has
    # essentially zero chance of ever crossing to the other (confirmed below by the plain-NUTS
    # baseline getting 100% stuck) -- the multimodal-posterior scenario parallel tempering is for
    # (sky-position mirrors / distance-inclination degeneracies for real GW posteriors; this toy
    # target is the generic-engine analogue).
    T = Float64
    mu1, mu2, sigma = T(-6.0), T(6.0), T(0.5)
    ntemps = 6
    nsteps = 3000
    nwarmup = 1500
    swap_interval = 5
    step_size0 = fill(T(0.3), ntemps)
    cfg = NUTSConfig{T}(; max_treedepth=8, target_accept=T(0.8), nwarmup=nwarmup)

    rng = Random.Xoshiro(42)
    pt = PTLadderState(ntemps; T=T, beta_min=T(0.01))
    logp_and_grad! = _make_bimodal_logp_and_grad(mu1, mu2, sigma; beta=pt.betas)

    positions = fill(mu1, 1, ntemps)  # every lane starts stuck in mode 1
    cold_chain = Vector{T}(undef, nsteps)
    ll_scratch = Vector{T}(undef, ntemps)
    callback = function (it, pos, logp, step_size, stats)
        if it % swap_interval == 0
            @inbounds for w in 1:ntemps
                ll_scratch[w] = _bimodal_raw_logL(pos[1, w], mu1, mu2, sigma)[1]
            end
            pt_swap!(pt, ll_scratch, rng)
        end
        cold_chain[it] = pos[1, Array(pt.lane_of_rank)[1]]
        return nothing
    end

    nuts_sample!(positions, logp_and_grad!; nsteps=nsteps, step_size0=step_size0, cfg=cfg,
                 rng=rng, callback=callback)

    @test all(isfinite, cold_chain)
    post = cold_chain[(nwarmup+1):end]
    frac_mode1 = count(x -> x < 0, post) / length(post)
    # Both modes visited with non-trivial weight -- a wide tolerance band (the toy target is
    # exactly symmetric, so ~0.5 is expected, but this is a stochastic MCMC run at a fixed seed,
    # not a claim of exact balance).
    @test 0.15 < frac_mode1 < 0.85
    @test all(>(0), pt.nattempt)

    # Baseline: the SAME target/seed/step-size/tree-depth WITHOUT tempering (beta=nothing, i.e.
    # ordinary NUTS) stays completely stuck in mode 1 -- demonstrates PT is doing real work here,
    # not that this toy target happens to be easy for plain NUTS too.
    rng_plain = Random.Xoshiro(42)
    logp_and_grad_plain! = _make_bimodal_logp_and_grad(mu1, mu2, sigma; beta=nothing)
    positions_plain = fill(mu1, 1, ntemps)
    result_plain = nuts_sample!(positions_plain, logp_and_grad_plain!; nsteps=nsteps,
                                 step_size0=step_size0, cfg=cfg, rng=rng_plain, record=true)
    post_plain = reshape(result_plain.chain[:, :, (nwarmup+1):end], :)
    @test all(x -> x < 0, post_plain)  # never escapes mode 1
end
