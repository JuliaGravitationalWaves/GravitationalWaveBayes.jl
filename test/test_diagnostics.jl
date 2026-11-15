using Test
using GravitationalWaveBayes: split_rhat, ess_geyer, ess_multichain, ParamDiagnostic, chain_diagnostics, worst_rhat, min_ess, print_diagnostics
using Random, Statistics

# ---------------------------------------------------------------------------------------------
# Tests for the Rhat/ESS diagnostics in `src/GravitationalWaveBayes/Diagnostics.jl`.
#
# They use SYNTHETIC chains with known convergence and mixing properties, so both directions
# are checked: non-converged chains are flagged and converged chains pass.
# ---------------------------------------------------------------------------------------------

@testset "Diagnostics" begin

    @testset "split_rhat: converged chains (same stationary AR(1) process, independent seeds)" begin
        # nwalkers independent AR(1) chains sharing the same (phi, stationary variance) -- i.e.
        # genuinely drawn from the same target, so they should agree once "converged" (here, exact
        # by construction from step 1, since AR(1) with these parameters IS already stationary --
        # x[1] ~ N(0, sigma^2/(1-phi^2)) is the stationary marginal itself).
        rng = MersenneTwister(1)
        phi = 0.3
        sigma = 1.0
        nsteps, nwalkers = 400, 8
        stat_var = sigma^2 / (1 - phi^2)
        draws = Matrix{Float64}(undef, nsteps, nwalkers)
        for w in 1:nwalkers
            draws[1, w] = randn(rng) * sqrt(stat_var)
            for t in 2:nsteps
                draws[t, w] = phi * draws[t - 1, w] + randn(rng) * sigma
            end
        end
        rhat = split_rhat(draws)
        @test rhat < 1.05
    end

    @testset "split_rhat: non-converged chains (distinct per-walker means, no overlap)" begin
        # Directly modeling the real GW190412 finding: some walkers stuck in a distinctly
        # different location than others. Each walker is individually a well-behaved, low-variance
        # AR(1) chain (so within-chain variance W is small) but walkers are centered at very
        # different means (so between-chain variance B is large) -- the textbook Rhat failure
        # mode, and exactly what caught the real run's non-convergence.
        rng = MersenneTwister(2)
        nsteps, nwalkers = 400, 8
        draws = Matrix{Float64}(undef, nsteps, nwalkers)
        walker_means = collect(range(-10.0, 10.0; length=nwalkers))  # widely separated modes
        for w in 1:nwalkers
            draws[:, w] = walker_means[w] .+ 0.1 .* randn(rng, nsteps)
        end
        rhat = split_rhat(draws)
        @test rhat > 2.0   # should be flagged as badly non-converged, matching the real run's 2.5-5.5
    end

    @testset "split_rhat: degenerate/edge cases" begin
        # Perfectly identical constant draws everywhere: W=B=0, defined as Rhat=1 (nothing to
        # detect non-convergence FROM -- a constant chain trivially "agrees" with itself).
        draws_const = ones(8, 4)
        @test split_rhat(draws_const) == 1.0

        # Zero within-chain variance (each chain is individually constant) but walkers disagree
        # with each other: the worst-case signature (W==0, B>0) -> Inf, not a domain error/NaN.
        draws_disagree = hcat(fill(1.0, 8), fill(2.0, 8), fill(3.0, 8))
        @test isinf(split_rhat(draws_disagree))

        @test_throws ArgumentError split_rhat(ones(3, 4))  # < 4 draws
    end

    @testset "ess_geyer: i.i.d. draws vs. strongly autocorrelated draws" begin
        rng = MersenneTwister(3)
        n = 5000

        iid = randn(rng, n)
        ess_iid = ess_geyer(iid)
        @test ess_iid > 0.5 * n   # i.i.d.: ESS should be close to n (allow statistical slack)
        @test ess_iid <= n + 1e-6

        # Strong AR(1) (phi close to 1): highly autocorrelated, ESS should be MUCH smaller than n.
        phi = 0.98
        ar1 = Vector{Float64}(undef, n)
        ar1[1] = randn(rng)
        for t in 2:n
            ar1[t] = phi * ar1[t - 1] + randn(rng) * sqrt(1 - phi^2)  # unit marginal variance
        end
        ess_ar1 = ess_geyer(ar1)
        @test ess_ar1 < 0.1 * n
        @test ess_ar1 < ess_iid

        # Constant sequence: autocorrelation undefined (zero variance) -> NaN, not a crash/Inf.
        @test isnan(ess_geyer(ones(100)))
    end

    @testset "ess_multichain: many short chains (an SMC final phase)" begin
        rng = MersenneTwister(5)
        n, m = 10, 1024
        @test abs(ess_multichain(randn(rng, n, m)) / (n * m) - 1) < 0.2
        # stationary AR(1) chains, phi 0.36 (the lag-1 correlation of the 2026-10-15 GW150914 best2 final phase):
        # ESS ~ n m (1 - phi) / (1 + phi)
        phi = 0.36
        x = Matrix{Float64}(undef, n, m)
        x[1, :] .= randn(rng, m)
        for t in 2:n
            x[t, :] .= phi .* x[t - 1, :] .+ sqrt(1 - phi^2) .* randn(rng, m)
        end
        @test abs(ess_multichain(x) / (n * m * (1 - phi) / (1 + phi)) - 1) < 0.2
        # each chain at its own offset: about one sample per chain, however many draws
        @test ess_multichain(x .+ 10 .* randn(rng, 1, m)) < 1.5m
    end

    @testset "chain_diagnostics: end-to-end on a synthetic (nparams, nwalkers, nsteps) array" begin
        rng = MersenneTwister(4)
        nparams, nwalkers, nsteps, warmup = 3, 6, 300, 100
        chain = Array{Float64}(undef, nparams, nwalkers, nsteps)
        # param 1: converged (shared stationary AR(1) across walkers)
        # param 2: non-converged (distinct per-walker means)
        # param 3: converged but with strong within-chain autocorrelation (low ESS, still low Rhat)
        for w in 1:nwalkers
            x1 = 0.0
            x3 = 0.0
            for t in 1:nsteps
                x1 = 0.4 * x1 + randn(rng)
                chain[1, w, t] = x1
                chain[2, w, t] = 5.0 * w + 0.1 * randn(rng)
                x3 = 0.9 * x3 + randn(rng) * sqrt(1 - 0.9^2)
                chain[3, w, t] = x3
            end
        end

        names = ["converged", "stuck_modes", "high_autocorr"]
        diags = chain_diagnostics(chain; warmup=warmup, names=names)
        @test length(diags) == nparams
        @test diags[1].name == "converged"
        @test diags[1].rhat < 1.1
        @test diags[2].rhat > 2.0
        @test diags[3].rhat < 1.2
        @test diags[3].ess < diags[1].ess   # strong autocorrelation costs effective samples

        # walkers stuck in two modes: the pooled ESS sees the between-walker variance (about one sample per walker
        # at most), where a sum of per-walker ESS counted every draw
        @test diags[2].ess < 2 * nwalkers
        @test diags[2].ess < 0.05 * sum(ess_geyer(chain[2, w, (warmup + 1):end]) for w in 1:nwalkers)

        wr = worst_rhat(diags)
        @test wr.name == "stuck_modes"
        me = min_ess(diags)
        @test me.name in ("high_autocorr", "converged", "stuck_modes")  # sanity: returns *some* valid entry

        # walker_mean_range should show the huge spread for the stuck-modes parameter, and a small
        # spread for the converged one -- exactly the "per-walker mean/std comparison" the
        # investigation's throwaway script printed, now structured/queryable.
        @test diags[2].walker_mean_range[2] - diags[2].walker_mean_range[1] > 10.0
        @test diags[1].walker_mean_range[2] - diags[1].walker_mean_range[1] < 5.0

        io = IOBuffer()
        print_diagnostics(io, diags)
        out = String(take!(io))
        @test occursin("converged", out)
        @test occursin("stuck_modes", out)
        @test occursin("Rhat=", out)
        @test occursin("ESS~", out)
    end

    @testset "chain_diagnostics: input validation" begin
        chain = randn(2, 4, 20)
        @test_throws ArgumentError chain_diagnostics(chain; warmup=20)   # no post-warmup draws left
        @test_throws ArgumentError chain_diagnostics(chain; warmup=17)   # < 4 post-warmup draws
        @test_throws ArgumentError chain_diagnostics(chain; names=["a"]) # wrong length
    end
end
