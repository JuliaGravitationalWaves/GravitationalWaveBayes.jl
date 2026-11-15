using Test
using GravitationalWaveBayes: nuts_sample!, NUTSConfig, DenseMetric
using GravitationalWaveBayes.NUTSCore: _block_mask, _hessian_metric
using LinearAlgebra
using Random, Statistics

# ---------------------------------------------------------------------------------------------
# Block-diagonal mass matrix (NUTSCore). A full dense metric over 11 GW parameters from only 16
# walkers is data-starved as well as curvature-limited (a GW150914 run had Rhat up to 228 for t_c). `mass_blocks` restricts the adapted metric to a block-diagonal
# structure grouped around the correlations that actually matter (e.g. mc/q/t_c/phase), which
# needs no new sampling/kinetic/drift/reflection code -- see `_block_mask`'s own comment for why
# masking a symmetric positive definite matrix to its block-diagonal restriction is always still
# positive definite, so `DenseMetric`'s existing Cholesky-based machinery already decouples
# correctly with zero changes.
# ---------------------------------------------------------------------------------------------

function _corr_gaussian(mu::Vector{Float64}, Sigma::Matrix{Float64})
    P = inv(Symmetric(Sigma))
    return function (grad_out, logp_out, positions, active_mask)
        for w in 1:size(positions, 2)
            d = positions[:, w] .- mu
            g = P * d
            grad_out[:, w] .= -g
            logp_out[w] = -0.5 * dot(d, g)
        end
        return nothing
    end
end

_rotation(n::Int, seed::Int) = Matrix(qr(randn(MersenneTwister(seed), n, n)).Q)
function _rotated_cov(sd::Vector{Float64}, seed::Int)
    Q = _rotation(length(sd), seed)
    return Q * Diagonal(sd .^ 2) * Q'
end

# A target whose TRUE covariance is exactly block-diagonal in {1,2,3,4} / {5,6,7,8}: each block
# internally correlated (its own rotation), zero correlation ACROSS blocks by construction.
function _block_structured_cov(sd_a::Vector{Float64}, sd_b::Vector{Float64}, seed::Int)
    Sigma = zeros(8, 8)
    Sigma[1:4, 1:4] .= _rotated_cov(sd_a, seed)
    Sigma[5:8, 5:8] .= _rotated_cov(sd_b, seed + 1)
    return Sigma
end

@testset "NUTSCore block-diagonal mass matrix" begin

    @testset "_block_mask: structural correctness and PD preservation" begin
        A = collect(reshape(1.0:25.0, 5, 5)); A = A + A' + 20I   # arbitrary symmetric PD-ish 5x5
        @test _block_mask(A, nothing) === A   # no-op, identity

        masked = _block_mask(A, [[1, 2], [3, 4]])   # index 5: own singleton block
        for i in 1:5, j in 1:5
            same = (i in (1,2) && j in (1,2)) || (i in (3,4) && j in (3,4)) || (i == j == 5)
            @test (masked[i, j] == 0) == !same
        end
        @test masked[1,1] == A[1,1] && masked[1,2] == A[1,2] && masked[3,4] == A[3,4]
        @test masked[1,3] == 0 && masked[5,1] == 0 && masked[5,4] == 0

        # PD preservation: mask a genuine random SPD matrix, confirm the result stays SPD.
        S = _rotated_cov(rand(MersenneTwister(11), 6) .+ 0.1, 12)
        masked2 = _block_mask(S, [[1,3,5],[2,4],[6]])
        @test isposdef(Symmetric(masked2))
        @test masked2 ≈ masked2'
    end

    @testset "NUTSConfig: mass_blocks validation" begin
        @test_throws ArgumentError NUTSConfig{Float64}(; mass_blocks=[[1,2]])                       # needs dense_mass
        @test_throws ArgumentError NUTSConfig{Float64}(; adapt_mass=true, dense_mass=true, mass_blocks=[[1,2],[2,3]])   # index 2 in two blocks
        @test_throws ArgumentError NUTSConfig{Float64}(; adapt_mass=true, dense_mass=true, mass_blocks=[Int[]])         # empty block
        cfg = NUTSConfig{Float64}(; adapt_mass=true, dense_mass=true, mass_blocks=[[1,2],[3,4,5]])
        @test cfg.mass_blocks == [[1,2],[3,4,5]]
        cfg_default = NUTSConfig{Float64}(; adapt_mass=true, dense_mass=true)
        @test cfg_default.mass_blocks === nothing
    end

    @testset "_hessian_metric with mass_blocks: recovers a block-diagonal target and drops cross-block terms" begin
        Sigma = _block_structured_cov([1.0, 0.3, 0.05, 0.02], [2.0, 0.5, 0.1, 0.05], 1)
        mu = zeros(8)
        f = _corr_gaussian(mu, Sigma)
        cfg = NUTSConfig{Float64}(; adapt_mass=true, dense_mass=true, hessian_init=true, nwarmup=30,
                                   mass_blocks=[[1,2,3,4],[5,6,7,8]])
        positions = reshape(mu, 8, 1) .+ 0.01 .* randn(MersenneTwister(2), 8, 1)
        metric, nvalid = _hessian_metric(f, positions, cfg)
        @test metric !== nothing
        @test nvalid == 1
        # Cross-block entries are EXACTLY zero by construction (masked), regardless of estimation noise.
        @test all(metric.Minv[i, j] == 0 for i in 1:4, j in 5:8)
        @test all(metric.Minv[i, j] == 0 for i in 5:8, j in 1:4)
        # Within-block recovery is still accurate (the true target has no cross-block correlation
        # to lose in the first place here, so masking costs nothing on this target).
        @test isapprox(metric.Minv[1:4, 1:4], Sigma[1:4, 1:4]; rtol=0.3)
        @test isapprox(metric.Minv[5:8, 5:8], Sigma[5:8, 5:8]; rtol=0.3)
    end

    @testset "block-diagonal metric samples a block-structured target correctly and efficiently" begin
        Sigma = _block_structured_cov([1.0, 0.3, 0.03, 0.01], [1.5, 0.4, 0.08, 0.03], 3)
        mu = zeros(8)
        f = _corr_gaussian(mu, Sigma)
        nwalkers = 8
        nsteps = 80; nwarmup = 40
        cfg = NUTSConfig{Float64}(; max_treedepth=8, adapt_mass=true, dense_mass=true, hessian_init=true,
                                   nwarmup=nwarmup, mass_blocks=[[1,2,3,4],[5,6,7,8]])
        positions = 3.0 .* randn(MersenneTwister(4), 8, nwalkers)
        result = nuts_sample!(positions, f; nsteps=nsteps, step_size0=fill(1e-3, nwalkers), cfg=cfg,
                               rng=MersenneTwister(5), record=true)
        @test all(isfinite, result.positions)
        post = result.chain[:, :, (nwarmup+1):end]
        flat = reshape(post, 8, :)
        empirical_mean = vec(sum(flat; dims=2)) ./ size(flat, 2)
        empirical_cov = cov(flat'; dims=1)
        @test isapprox(empirical_mean, mu; atol=0.3)
        @test isapprox(diag(empirical_cov), diag(Sigma); rtol=0.5)
    end

    @testset "negative control: a WRONG block split still samples correctly but less efficiently than the right one" begin
        # Ground truth correlates {1,2} strongly and {3,4} strongly, with 1-2 and 3-4 uncorrelated
        # -- the "right" split. A deliberately WRONG split groups {1,3} and {2,4} instead, which
        # cannot capture either true correlation. NUTS remains asymptotically correct under any
        # valid PD metric (this is a discriminating EFFICIENCY test, not a correctness one, unlike
        # `test_nuts_core_dense_mass.jl`'s "ignoring the metric" negative control).
        rho = 0.97
        s1, s2 = 1.0, 0.02
        Sigma = zeros(4, 4)
        Sigma[1:2, 1:2] .= [s1^2 rho*s1*s1; rho*s1*s1 s1^2]
        Sigma[3:4, 3:4] .= [s2^2 rho*s2*s2; rho*s2*s2 s2^2]
        mu = zeros(4)
        f = _corr_gaussian(mu, Sigma)

        # `step_size_max_change` set large (effectively uncapped): this test's short `nwarmup=30`
        # leaves few restarts to recover the right scale, and its point is the BLOCK CHOICE
        # (right vs. wrong split), not the mid-warmup restart-overshoot guard (see
        # `test_nuts_core_dense_mass.jl`'s "step_size_max_change caps a restart overshoot" for that
        # guard's own dedicated test) -- the default cap is orthogonal to what's being checked here.
        cfg_right = NUTSConfig{Float64}(; max_treedepth=6, adapt_mass=true, dense_mass=true, hessian_init=true,
                                         nwarmup=30, mass_blocks=[[1,2],[3,4]], step_size_max_change=1e6)
        cfg_wrong = NUTSConfig{Float64}(; max_treedepth=6, adapt_mass=true, dense_mass=true, hessian_init=true,
                                         nwarmup=30, mass_blocks=[[1,3],[2,4]], step_size_max_change=1e6)
        nwalkers = 8
        start = 2.0 .* randn(MersenneTwister(6), 4, nwalkers)
        result_right = nuts_sample!(copy(start), f; nsteps=40, step_size0=fill(1e-3, nwalkers),
                                     cfg=cfg_right, rng=MersenneTwister(7), record=true)
        result_wrong = nuts_sample!(copy(start), f; nsteps=40, step_size0=fill(1e-3, nwalkers),
                                     cfg=cfg_wrong, rng=MersenneTwister(7), record=true)
        @test all(isfinite, result_right.positions)
        @test all(isfinite, result_wrong.positions)

        # Efficiency proxy: whitened displacement per draw during SAMPLING (post-warmup), median
        # over walkers and draws -- the wrong split should move less per draw for the same
        # step-size-adaptation budget, since it can't whiten either true correlated direction.
        disp_right = Float64[]
        disp_wrong = Float64[]
        cb_right = (it, pos, logp, step, stats) -> (it > 30 && append!(disp_right, stats.disp))
        cb_wrong = (it, pos, logp, step, stats) -> (it > 30 && append!(disp_wrong, stats.disp))
        nuts_sample!(copy(start), f; nsteps=40, step_size0=fill(1e-3, nwalkers), cfg=cfg_right,
                     rng=MersenneTwister(7), callback=cb_right)
        nuts_sample!(copy(start), f; nsteps=40, step_size0=fill(1e-3, nwalkers), cfg=cfg_wrong,
                     rng=MersenneTwister(7), callback=cb_wrong)
        @test median(disp_right) > median(disp_wrong)
    end
end
