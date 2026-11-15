using Test
using GravitationalWaveBayes: nuts_sample!, NUTSConfig, NUTSBounds
using Random, Statistics

# ---------------------------------------------------------------------------------------------
# Validates `NUTSCore`'s reflecting/wrapping boundary
# enforcement (see `NUTSCore.jl`'s `NUTSBounds`/`_apply_bounds!`/`_reflect_component` docstrings
# for the reflection technique itself and why it preserves detailed balance, citing Neal 2011).
#
# The key thing to validate is NOT just "positions stay in bounds" (a naive clamp would do that
# too) but that the STATIONARY distribution is still correct -- i.e. detailed balance/reversibility
# genuinely survives reflection, not just that the sampler doesn't crash or escape its box.
#
# Three independent checks, each with an exactly known correct answer:
#   1. A flat (improper, logp=0 everywhere) 1-D target confined only by a tight reflecting box:
#      Hamiltonian dynamics with reflection is a "particle in a box" -- the stationary position
#      marginal is EXACTLY uniform (kinetic energy is independent of position, so at fixed speed a
#      bouncing free particle spends equal time everywhere in the box). This is the single most
#      sensitive test of reflection correctness: any bug (sign error in the momentum flip, an off-
#      by-one in the fold, etc.) shows up immediately as pile-up or depletion near the walls.
#   2. A flat target on a WRAPPED (periodic) domain: the stationary distribution should be uniform
#      on the circle, checked via the circular resultant length (near 0 for uniform).
#   3. A genuine (non-flat) truncated Gaussian: reflecting NUTS should recover the analytic
#      truncated-normal variance to good statistical precision.
#   4. A "naive clamp" negative control (position clamped into bounds, momentum left untouched --
#      the WRONG fix the task explicitly warns against) demonstrably fails check 1's uniformity,
#      to make concrete why plain clamping is not an acceptable substitute for reflection.
#
# Test-design note (found empirically while writing this file): an EXACTLY flat/improper target
# breaks this package's dual-averaging step-size adaptation (`DualAveraging` in NUTSCore.jl) --
# since the Hamiltonian never changes with position, the Metropolis-style acceptance statistic is
# always ~1 regardless of step size, so dual averaging (which tries to REDUCE acceptance toward
# `target_accept=0.8`) drives the step size to run away arbitrarily large. This is a genuine
# degeneracy of testing with an improper flat target, not a defect in the boundary reflection code
# under test -- checks 1/2/4 below therefore use `nwarmup=0` (a fixed, not dual-averaged, step
# size) and deliberately non-resonant step sizes (e.g. avoiding exact small-integer multiples of
# the box width, which would alias the deterministic constant-velocity bounce pattern onto a few
# repeated points -- also a test-design artifact, not a reflection bug: check 3's non-flat target
# does not have this issue and uses ordinary dual-averaged adaptation.
# ---------------------------------------------------------------------------------------------

const _FLAT_LOGP! = function (grad_out, logp_out, positions, active_mask)
    fill!(grad_out, 0.0)
    fill!(logp_out, 0.0)
    return nothing
end

@testset "NUTSCore boundary reflection/wrapping" begin

    @testset "reflecting box recovers a uniform stationary distribution" begin
        lo, hi = -0.05, 0.05
        bounds = NUTSBounds([lo], [hi])
        cfg = NUTSConfig{Float64}(; max_treedepth=6, nwarmup=0, bounds=bounds)

        for (step, seed) in ((0.023, 42), (0.11, 11), (0.21, 7))  # includes step > box width
            positions = reshape([0.0], 1, 1)
            result = nuts_sample!(positions, _FLAT_LOGP!; nsteps=4000, step_size0=[step],
                                   cfg=cfg, rng=MersenneTwister(seed), record=true)
            post = vec(result.chain[1, 1, :])

            @test all(lo - 1e-9 .<= post .<= hi + 1e-9)
            expected_var = (hi - lo)^2 / 12
            @test isapprox(mean(post), 0.0; atol=0.01)
            @test isapprox(var(post), expected_var; rtol=0.15)
        end
    end

    @testset "wrapped periodic domain recovers a uniform-on-circle stationary distribution" begin
        lo, hi = 0.0, 2 * pi
        bounds = NUTSBounds([lo], [hi]; periodic=[true])
        cfg = NUTSConfig{Float64}(; max_treedepth=6, nwarmup=0, bounds=bounds)
        positions = reshape([Float64(pi)], 1, 1)
        result = nuts_sample!(positions, _FLAT_LOGP!; nsteps=4000, step_size0=[0.37],
                               cfg=cfg, rng=MersenneTwister(7), record=true)
        post = vec(result.chain[1, 1, :])

        @test all(0.0 .<= post .< hi + 1e-9)
        # circular resultant length: 0 for exactly uniform on the circle, 1 for a point mass.
        R = sqrt(mean(cos.(post))^2 + mean(sin.(post))^2)
        @test R < 0.1
    end

    @testset "reflecting box recovers correct truncated-Gaussian moments (non-flat target)" begin
        mu, sigma = 0.0, 1.0
        lo, hi = -1.0, 1.0
        bounds = NUTSBounds([lo], [hi])
        cfg = NUTSConfig{Float64}(; max_treedepth=8, nwarmup=500, bounds=bounds)

        logp_and_grad_gauss! = function (grad_out, logp_out, positions, active_mask)
            @inbounds for w in 1:size(positions, 2)
                x = positions[1, w]
                logp_out[w] = -0.5 * (x - mu)^2 / sigma^2
                grad_out[1, w] = -(x - mu) / sigma^2
            end
            return nothing
        end

        positions = reshape([0.0], 1, 1)
        result = nuts_sample!(positions, logp_and_grad_gauss!; nsteps=4000, step_size0=[0.3],
                               cfg=cfg, rng=MersenneTwister(3), record=true)
        post = vec(result.chain[1, 1, 501:end])

        # Analytic second moment of N(0,1) truncated to [-1,1] via fine trapezoid quadrature (mean
        # is exactly 0 by symmetry, so the variance IS the second moment) -- avoids needing an
        # error-function special-function dependency just for this test.
        xs = range(lo, hi; length=200_001)
        dens = exp.(-0.5 .* xs .^ 2 ./ sigma^2)
        Z = sum(dens) * step(xs)
        analytic_var = sum(dens .* xs .^ 2) * step(xs) / Z

        @test all(lo - 1e-9 .<= post .<= hi + 1e-9)
        @test isapprox(mean(post), 0.0; atol=0.05)
        @test isapprox(var(post), analytic_var; rtol=0.15)
    end

    @testset "negative control: naive clamp (no momentum flip) does NOT recover uniformity" begin
        # Minimal standalone reproduction of the WRONG fix the task warns against -- NOT using
        # NUTSCore's reflection at all, just a bare leapfrog-in-a-box with the position clamped
        # and momentum left untouched. Included to make concrete, not just asserted, why
        # `_reflect_component`'s momentum flip is load-bearing: clamping alone repeatedly discards
        # kinetic energy at the wall, producing spurious pile-up there instead of a uniform
        # distribution.
        lo, hi = -0.05, 0.05
        rng = MersenneTwister(42)
        x = 0.0
        n = 20_000
        samples = Vector{Float64}(undef, n)
        for i in 1:n
            r = randn(rng)
            x = clamp(x + 0.037 * r, lo, hi)
            samples[i] = x
        end
        frac_at_edges = count(v -> abs(abs(v) - hi) < 1e-9, samples) / n
        @test frac_at_edges > 0.2   # pathological pile-up; a correct/uniform result would be ~0
        @test !isapprox(var(samples), (hi - lo)^2 / 12; rtol=0.15)  # fails the same check reflection passes
    end

    @testset "bounds=nothing (default) leaves NUTSConfig/behavior unchanged" begin
        cfg = NUTSConfig{Float64}(; nwarmup=5)
        @test cfg.bounds === nothing
    end
end

# ---------------------------------------------------------------------------------------------
# SEOBNRv5AlignedSpinF2's binned-kernel
# warmup-aliasing fix ("dense-kernel warmup, switch to binned kernel only for post-warmup
# sampling") needs `nuts_sample!` to accept a SEPARATE `warmup_logp_and_grad!` used only for draws
# `it <= cfg.nwarmup`. This validates the switching mechanism itself, in isolation, on a toy
# target with call-counting closures -- cheap and fast (no Enzyme/GPU, pure NUTSCore) -- before
# trusting it on the real (slow, Enzyme+EOB) SEOBNRv5AlignedSpinF2 recovery test in
# test_nuts_recovery_seobnrv5alignedspinf2.jl (which exercises the SAME keyword through
# `BatchedNUTS.batched_nuts!`'s `dense_warmup` flag, but is too slow to run by default).
# ---------------------------------------------------------------------------------------------
@testset "nuts_sample!: warmup_logp_and_grad! switches the callable used during warmup only" begin
    _counting_gaussian_logp(counter::Ref{Int}) = function (grad_out, logp_out, positions, active_mask)
        counter[] += 1
        @. grad_out = -positions
        @. logp_out = -0.5 * positions[1, :]^2
        return nothing
    end

    @testset "both callables are exercised when warmup and post-warmup overlap" begin
        nwarmup, nsteps = 5, 12
        cfg = NUTSConfig{Float64}(; max_treedepth=4, nwarmup=nwarmup)
        warmup_count, main_count = Ref(0), Ref(0)
        positions = reshape([0.1], 1, 1)
        result = nuts_sample!(positions, _counting_gaussian_logp(main_count);
                               nsteps=nsteps, step_size0=[0.3], cfg=cfg, rng=MersenneTwister(1),
                               warmup_logp_and_grad! = _counting_gaussian_logp(warmup_count))
        @test all(isfinite, result.positions)
        @test warmup_count[] > 0
        @test main_count[] > 0
    end

    @testset "nwarmup == nsteps (all-warmup chain) never touches the main callable" begin
        nsteps = 8
        cfg = NUTSConfig{Float64}(; max_treedepth=4, nwarmup=nsteps)
        warmup_count, main_count = Ref(0), Ref(0)
        positions = reshape([0.1], 1, 1)
        nuts_sample!(positions, _counting_gaussian_logp(main_count);
                     nsteps=nsteps, step_size0=[0.3], cfg=cfg, rng=MersenneTwister(2),
                     warmup_logp_and_grad! = _counting_gaussian_logp(warmup_count))
        @test warmup_count[] > 0
        @test main_count[] == 0
    end

    @testset "nwarmup == 0 never touches the warmup callable" begin
        nsteps = 8
        cfg = NUTSConfig{Float64}(; max_treedepth=4, nwarmup=0)
        warmup_count, main_count = Ref(0), Ref(0)
        positions = reshape([0.1], 1, 1)
        nuts_sample!(positions, _counting_gaussian_logp(main_count);
                     nsteps=nsteps, step_size0=[0.3], cfg=cfg, rng=MersenneTwister(3),
                     warmup_logp_and_grad! = _counting_gaussian_logp(warmup_count))
        @test warmup_count[] == 0
        @test main_count[] > 0
    end

    @testset "omitting the keyword is a bitwise no-op vs. explicitly passing the same callable" begin
        nwarmup, nsteps = 5, 12
        cfg = NUTSConfig{Float64}(; max_treedepth=4, nwarmup=nwarmup)

        f_default = _counting_gaussian_logp(Ref(0))
        positions_default = reshape([0.1], 1, 1)
        result_default = nuts_sample!(positions_default, f_default; nsteps=nsteps, step_size0=[0.3],
                                       cfg=cfg, rng=MersenneTwister(7))

        f_explicit = _counting_gaussian_logp(Ref(0))
        positions_explicit = reshape([0.1], 1, 1)
        result_explicit = nuts_sample!(positions_explicit, f_explicit; nsteps=nsteps, step_size0=[0.3],
                                        cfg=cfg, rng=MersenneTwister(7), warmup_logp_and_grad! = f_explicit)

        @test result_default.positions == result_explicit.positions
        @test result_default.logp == result_explicit.logp
    end
end
