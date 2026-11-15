using Test
using GravitationalWaveBayes: nuts_sample!, NUTSConfig, NUTSBounds
using GravitationalWaveBayes.NUTSCore: PTLadderState, pt_swap!
using GravitationalWaveBayes.NUTSCore: _mass_adaptation_windows
using Random, Statistics

# ---------------------------------------------------------------------------------------------
# Diagonal mass-matrix adaptation (NUTSCore). Every check here has an exactly known
# answer (a diagonal Gaussian with a known per-parameter variance), so a wrong momentum draw,
# kinetic energy, drift or U-turn criterion shows up as a biased variance rather than passing
# silently:
#   1. the Stan window schedule itself (pure arithmetic);
#   2. a FIXED, deliberately mismatched non-unit metric (`inv_mass0`, no adaptation) must still
#      leave the target exactly stationary -- this isolates the correctness of the M-weighted
#      leapfrog/U-turn code from the adaptation logic, including through reflecting bounds;
#   3. adaptation on a badly-scaled target (4 orders of magnitude between parameters) must learn
#      the scales, cut the treedepth, and recover the variances that an unadapted run cannot.
# ---------------------------------------------------------------------------------------------

function _diag_gaussian(mu::Vector{Float64}, var::Vector{Float64})
    return function (grad_out, logp_out, positions, active_mask)
        for w in 1:size(positions, 2)
            s = 0.0
            for i in eachindex(mu)
                d = positions[i, w] - mu[i]
                grad_out[i, w] = -d / var[i]
                s += d * d / var[i]
            end
            logp_out[w] = -0.5 * s
        end
        return nothing
    end
end

@testset "NUTSCore diagonal mass-matrix adaptation" begin

    @testset "Stan window schedule" begin
        @test _mass_adaptation_windows(19) == UnitRange{Int}[]
        # nwarmup=1000 with the default 75/25/50 buffers: windows of 25, 50, 100, 200 then the
        # last one stretched to the terminal buffer (Stan's documented schedule).
        @test _mass_adaptation_windows(1000) == [76:100, 101:150, 151:250, 251:450, 451:950]
        # Too short for the default buffers -> 15%/10% proportional split; windows must tile
        # [init+1, nwarmup-term] contiguously with no gaps or overlap.
        w = _mass_adaptation_windows(100)
        @test first(first(w)) == 16
        @test last(last(w)) == 90
        @test all(first(w[k + 1]) == last(w[k]) + 1 for k in 1:length(w) - 1)
    end

    @testset "fixed mismatched metric leaves the target stationary" begin
        T = Float64
        mu = [1.0, -2.0, 0.5]
        var = [1.0, 9.0, 0.04]
        f = _diag_gaussian(mu, var)
        # Deliberately WRONG metric (off by up to 10x either way): must cost efficiency, never
        # correctness.
        inv_mass0 = [4.0, 0.3, 1.0]
        nwalkers = 8
        pos = randn(MersenneTwister(3), T, 3, nwalkers)
        cfg = NUTSConfig{T}(; max_treedepth=8, nwarmup=300)
        res = nuts_sample!(pos, f; nsteps=1500, step_size0=fill(0.3, nwalkers), cfg=cfg,
                            rng=MersenneTwister(4), record=true, inv_mass0=inv_mass0)
        @test res.inv_mass == inv_mass0            # nothing adapts without cfg.adapt_mass
        post = res.chain[:, :, 301:end]
        for i in 1:3
            x = vec(post[i, :, :])
            @test isapprox(mean(x), mu[i]; atol=4 * sqrt(var[i]) / sqrt(nwalkers * 900 / 8))
            @test isapprox(Statistics.var(x), var[i]; rtol=0.12)
        end
    end

    @testset "fixed non-unit metric through reflecting bounds (truncated Gaussian)" begin
        T = Float64
        f = _diag_gaussian([0.0, 0.0], [1.0, 1.0])
        bounds = NUTSBounds([-1.0, -10.0], [1.0, 10.0])   # row 1 truncated at +-1 sigma
        cfg = NUTSConfig{T}(; max_treedepth=8, nwarmup=200, bounds=bounds)
        nwalkers = 8
        pos = 0.1 .* randn(MersenneTwister(5), T, 2, nwalkers)
        res = nuts_sample!(pos, f; nsteps=1400, step_size0=fill(0.3, nwalkers), cfg=cfg,
                            rng=MersenneTwister(6), record=true, inv_mass0=[0.2, 3.0])
        x1 = vec(res.chain[1, :, 201:end])
        @test all(-1 - 1e-9 .<= x1 .<= 1 + 1e-9)
        # variance of N(0,1) truncated to [-1,1]: 1 - 2*phi(1)/(2*Phi(1)-1)
        phi1 = exp(-0.5) / sqrt(2pi)
        Phi1 = 0.5 * (1 + 0.6826894921370859)  # Phi(1) = (1 + erf(1/sqrt2))/2
        expected = 1 - 2 * phi1 / (2 * Phi1 - 1)
        @test isapprox(Statistics.var(x1), expected; rtol=0.1)
        @test isapprox(Statistics.var(vec(res.chain[2, :, 201:end])), 1.0; rtol=0.12)
    end

    @testset "adaptation learns badly-scaled variances and fixes mixing" begin
        T = Float64
        sd = [1e-3, 1e-1, 1.0, 30.0]
        var = sd .^ 2
        mu = [0.5, -0.3, 2.0, 100.0]
        f = _diag_gaussian(mu, var)
        nwalkers = 8
        nwarmup, nsamp = 600, 600
        start() = mu .+ sd .* randn(MersenneTwister(7), T, 4, nwalkers)

        function run(adapt)
            cfg = NUTSConfig{T}(; max_treedepth=6, nwarmup=nwarmup, adapt_mass=adapt)
            depths = Int[]
            cb = (it, pos, lp, ss, stats) -> it > nwarmup && append!(depths, stats.depth)
            res = nuts_sample!(start(), f; nsteps=nwarmup + nsamp, step_size0=fill(0.05, nwalkers),
                                cfg=cfg, rng=MersenneTwister(8), record=true, callback=cb)
            return res, mean(depths)
        end

        res_off, depth_off = run(false)
        res_on, depth_on = run(true)

        @test res_off.inv_mass == ones(4)
        # Learned scales: within a factor 2 of the truth for every parameter (4 orders of magnitude
        # apart -- a shared/unit metric cannot be anywhere near this).
        @test all(0.5 .< res_on.inv_mass ./ var .< 2.0)
        # The unadapted run is pinned at the treedepth cap; the adapted one is not.
        @test depth_off >= 5.5
        @test depth_on < depth_off - 1
        # And it actually samples the right distribution, per parameter.
        post = res_on.chain[:, :, nwarmup + 1:end]
        for i in 1:4
            x = vec(post[i, :, :])
            @test isapprox(mean(x), mu[i]; atol=0.3 * sd[i])
            @test isapprox(Statistics.var(x), var[i]; rtol=0.3)
        end
    end

    @testset "adapt_mass with nwarmup < 20 is a no-op" begin
        T = Float64
        f = _diag_gaussian([0.0], [4.0])
        cfg = NUTSConfig{T}(; max_treedepth=5, nwarmup=10, adapt_mass=true)
        res = nuts_sample!(reshape([0.1, -0.2], 1, 2), f; nsteps=30, step_size0=fill(0.3, 2),
                            cfg=cfg, rng=MersenneTwister(9))
        @test res.inv_mass == [1.0]
    end
end

@testset "per-temperature-rank mass matrix (parallel tempering)" begin
    T = Float64
    # Untempered likelihood: independent Gaussians with variances `var` (2 decades apart); flat
    # prior. Tempered lane target at beta is N(0, var/beta), so rank r's true inverse-mass diagonal
    # is var/beta_r -- known exactly. Uses the REAL PTLadderState/pt_swap! so labels really move.
    var = [1e-2, 25.0]
    ntemps = 6
    pt = PTLadderState(ntemps; T=T, beta_min=0.1)
    ll_of(pos) = [-0.5 * sum(pos[i, w]^2 / var[i] for i in 1:2) for w in 1:size(pos, 2)]
    f = function (grad_out, logp_out, positions, active_mask)
        for w in 1:size(positions, 2)
            b = pt.betas[w]
            s = 0.0
            for i in 1:2
                grad_out[i, w] = -b * positions[i, w] / var[i]
                s += positions[i, w]^2 / var[i]
            end
            logp_out[w] = -0.5 * b * s
        end
        return nothing
    end
    rng = MersenneTwister(21)
    pos = sqrt.(var) .* randn(rng, T, 2, ntemps)
    cold = Vector{Vector{Float64}}()
    nwarmup, nsteps = 700, 1500
    cb = function (it, positions, logp, step_size, stats)
        pt_swap!(pt, ll_of(positions), rng)
        it > nwarmup && push!(cold, positions[:, pt.lane_of_rank[1]])
        return nothing
    end
    cfg = NUTSConfig{T}(; max_treedepth=7, nwarmup=nwarmup, adapt_mass=true)
    res = nuts_sample!(pos, f; nsteps=nsteps, step_size0=fill(0.3, ntemps), cfg=cfg, rng=rng,
                        callback=cb, metric_rank=pt.rank_of_lane)

    @test size(res.inv_mass) == (2, ntemps)
    truth = var ./ reshape(pt.beta_ladder, 1, :)
    ratio = res.inv_mass ./ truth
    # Each rank learned ITS OWN scale (up to a factor of 10 broader for the hottest): a metric
    # shared across lanes cannot satisfy this. Generous tolerance: one stream per rank, short windows.
    @test all(0.4 .< ratio .< 2.5)
    @test all(res.inv_mass[:, end] .> 3 .* res.inv_mass[:, 1])  # hot rank clearly broader than cold
    # The cold chain still samples the untempered target (the PT + per-rank-metric combination is
    # statistically valid, not merely self-consistent).
    x = reduce(hcat, cold)
    for i in 1:2
        @test isapprox(Statistics.var(x[i, :]), var[i]; rtol=0.35)
    end

    # Invalid `metric_rank` is rejected rather than silently mis-indexing.
    @test_throws ArgumentError nuts_sample!(copy(pos), f; nsteps=2, step_size0=fill(0.3, ntemps),
        cfg=NUTSConfig{T}(; nwarmup=1), metric_rank=fill(1, ntemps))
    @test_throws ArgumentError nuts_sample!(copy(pos), f; nsteps=2, step_size0=fill(0.3, ntemps),
        cfg=NUTSConfig{T}(; nwarmup=1), inv_mass0=ones(2, ntemps))
end
