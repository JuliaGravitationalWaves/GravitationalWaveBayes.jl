using Test
using GravitationalWaveBayes: nuts_sample!, NUTSConfig
using GravitationalWaveBayes.NUTSCore: nuts_draw!
using Random

# ---------------------------------------------------------------------------------------------
# Quorum-based tree-doubling termination (NUTSCore). Motivated by production runs:
# `while any(active) && depth <= cfg.max_treedepth` forces the WHOLE walker batch
# to keep deepening as long as a single straggler is still active, and 100% of post-warmup draws hit `max_treedepth` for exactly this reason. `quorum_frac` lets the
# round stop once too few walkers remain active, cutting the remaining stragglers off early.
# ---------------------------------------------------------------------------------------------

# Standard normal target, one parameter: logp = -0.5*x^2, grad = -x. Simple enough that how many
# doublings a given step size needs before a natural U-turn is easy to reason about directly.
function _standard_normal!(grad_out, logp_out, positions, active_mask)
    for w in axes(positions, 2)
        active_mask[w] || continue
        x = positions[1, w]
        grad_out[1, w] = -x
        logp_out[w] = -0.5 * x^2
    end
    return nothing
end

@testset "NUTSCore quorum-based tree-doubling termination" begin

    @testset "config validation and default reproduces any(active) exactly" begin
        @test NUTSConfig{Float64}().quorum_frac == 0.0
        @test_throws ArgumentError NUTSConfig{Float64}(; quorum_frac=-0.1)
        @test_throws ArgumentError NUTSConfig{Float64}(; quorum_frac=1.1)
        @test NUTSConfig{Float64}(; quorum_frac=0.0).quorum_frac == 0.0
        @test NUTSConfig{Float64}(; quorum_frac=1.0).quorum_frac == 1.0
    end

    @testset "quorum_frac=0 (default) is bit-for-bit identical to no quorum at all" begin
        nwalkers = 8
        cfg_default = NUTSConfig{Float64}(; max_treedepth=6)
        cfg_explicit_zero = NUTSConfig{Float64}(; max_treedepth=6, quorum_frac=0.0)
        for seed in 1:3
            pos_a = randn(MersenneTwister(seed), 1, nwalkers) .* 3
            pos_b = copy(pos_a)
            step = fill(0.3, nwalkers)
            logp_a = zeros(nwalkers); logp_b = zeros(nwalkers)
            stats_a = nuts_draw!(pos_a, logp_a, step, _standard_normal!, cfg_default, MersenneTwister(100 + seed))
            stats_b = nuts_draw!(pos_b, logp_b, step, _standard_normal!, cfg_explicit_zero, MersenneTwister(100 + seed))
            @test pos_a == pos_b
            @test logp_a == logp_b
            @test stats_a.depth == stats_b.depth
            @test stats_a.n_leapfrog == stats_b.n_leapfrog
            @test !any(stats_a.quorum_forced)
            @test !any(stats_b.quorum_forced)
        end
    end

    @testset "a straggler walker (tiny step size) is cut off early under quorum, not naturally" begin
        nwalkers = 8
        # Walkers 1:7 well-tuned (U-turn within a handful of doublings); walker 8's step size is
        # 500x smaller, so it needs many more doublings to cover the same physical trajectory
        # length and is still growing long after the others have naturally terminated.
        step = vcat(fill(0.5, nwalkers - 1), [0.001])
        max_treedepth = 12  # generous, so walker 8 would keep going past where quorum should cut it

        rng_seed = 42
        pos0 = randn(MersenneTwister(1), 1, nwalkers) .* 2

        # No quorum: walker 8 runs unchecked (either naturally terminates late or exhausts the cap).
        cfg_noquorum = NUTSConfig{Float64}(; max_treedepth=max_treedepth, quorum_frac=0.0)
        pos_nq = copy(pos0); logp_nq = zeros(nwalkers)
        stats_nq = nuts_draw!(pos_nq, logp_nq, copy(step), _standard_normal!, cfg_noquorum, MersenneTwister(rng_seed))

        # Quorum: stop the round once fewer than 6 of 8 walkers remain active.
        cfg_quorum = NUTSConfig{Float64}(; max_treedepth=max_treedepth, quorum_frac=0.75)
        pos_q = copy(pos0); logp_q = zeros(nwalkers)
        stats_q = nuts_draw!(pos_q, logp_q, copy(step), _standard_normal!, cfg_quorum, MersenneTwister(rng_seed))

        # Quorum can only cut a trajectory SHORTER (or leave it unchanged), never longer, for every
        # walker -- multiple well-tuned walkers can legitimately finish together within the same
        # doubling round (not just walker 8), so this is the robust per-walker invariant, not an
        # assumption that walkers 1:7 are entirely unaffected.
        @test all(stats_q.depth .<= stats_nq.depth)
        @test !any(stats_nq.quorum_forced)   # no quorum configured on this run at all

        # Walker 8 (the deliberate straggler, 500x smaller step size) is cut off strictly earlier
        # under quorum, and flagged as such -- not via the max_treedepth cap and not via its own
        # divergence/U-turn.
        @test stats_q.depth[8] < stats_nq.depth[8]
        @test stats_q.quorum_forced[8]
        @test !stats_q.diverged[8]
        @test stats_q.depth[8] < max_treedepth

        # Real computational saving: strictly fewer total leapfrog evaluations under quorum.
        @test sum(stats_q.n_leapfrog) < sum(stats_nq.n_leapfrog)
    end

    @testset "quorum_frac=1.0: the round stops the instant any single walker finishes" begin
        nwalkers = 6
        step = fill(0.5, nwalkers)
        cfg = NUTSConfig{Float64}(; max_treedepth=10, quorum_frac=1.0)
        pos = randn(MersenneTwister(7), 1, nwalkers) .* 2
        logp = zeros(nwalkers)
        stats = nuts_draw!(pos, logp, step, _standard_normal!, cfg, MersenneTwister(9))
        # quorum_count = ceil(1.0*6) = 6, so the round stops as soon as ANY walker(s) go inactive
        # (count(active) drops below 6), forcing every other still-active walker to stop at the
        # same depth right after. Walkers can finish together within one doubling round, so more
        # than one may have stopped "naturally" in that same final round -- but at least one walker
        # must have been force-cut (otherwise every walker happened to finish in the exact same
        # round with none left active to force, which `quorum_forced` correctly reports as empty;
        # assert the mechanism actually engaged for this seed/setup).
        @test count(stats.quorum_forced) >= 1
        # Every quorum-forced walker was cut off at the SAME depth (the round-boundary the first
        # natural termination(s) triggered).
        forced_depths = stats.depth[stats.quorum_forced]
        @test all(==(forced_depths[1]), forced_depths)
    end

    @testset "end-to-end: nuts_sample! runs cleanly with quorum_frac set (multi-draw sanity check)" begin
        nwalkers = 8
        cfg = NUTSConfig{Float64}(; max_treedepth=8, nwarmup=5, quorum_frac=0.5)
        positions = randn(MersenneTwister(3), 1, nwalkers) .* 2
        result = nuts_sample!(positions, _standard_normal!; nsteps=10, step_size0=fill(0.3, nwalkers),
                               cfg=cfg, rng=MersenneTwister(11))
        @test all(isfinite, result.logp)
        @test all(isfinite, result.positions)
    end
end
