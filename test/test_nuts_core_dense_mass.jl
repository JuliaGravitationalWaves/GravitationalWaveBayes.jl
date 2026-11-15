using Test
import GravitationalWaveBayes
using GravitationalWaveBayes: nuts_sample!, NUTSConfig, NUTSBounds, DenseMetric
using GravitationalWaveBayes.NUTSCore: _hessian_metric, _find_reasonable_step_size!, _drift_and_bound!, _whitened_norm
using LinearAlgebra
using Random, Statistics

# ---------------------------------------------------------------------------------------------
# Dense mass matrix + Hessian bootstrap (NUTSCore, 2026-09-20). Motivated by the 2026-09-20
# GW150914 H1 run: 300 draws, every draw at `max_treedepth`, every walker frozen (Rhat up to 24000),
# because (a) a diagonal metric cannot whiten a correlated posterior, and (b) sample-based
# adaptation from a tiny initial step size learns the scale of its own frozen jitter. All targets
# here are Gaussians with a known covariance, so every check has an exact answer.
# ---------------------------------------------------------------------------------------------

# Gaussian with covariance Q diag(sd^2) Q' (a fixed rotation Q mixes ALL axes, so no axis
# rescaling can isolate the narrow direction).
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

@testset "NUTSCore dense mass matrix" begin

    @testset "config validation" begin
        @test_throws ArgumentError NUTSConfig{Float64}(; dense_mass=true)                       # needs adapt_mass
        @test_throws ArgumentError NUTSConfig{Float64}(; adapt_mass=true, hessian_init=true)    # needs dense_mass
        f = _corr_gaussian([0.0, 0.0], [1.0 0.0; 0.0 1.0])
        cfg = NUTSConfig{Float64}(; adapt_mass=true, dense_mass=true, nwarmup=30)
        @test_throws ArgumentError nuts_sample!(zeros(2, 2), f; nsteps=2, step_size0=fill(0.3, 2), cfg=cfg,
                                                 metric_rank=[1, 2])
        @test_throws ArgumentError nuts_sample!(zeros(2, 2), f; nsteps=2, step_size0=fill(0.3, 2),
                                                 inv_mass0=DenseMetric(Matrix(1.0I, 3, 3)))
    end

    @testset "DenseMetric algebra" begin
        S = _rotated_cov([1.0, 0.1, 3.0], 1)
        dm = DenseMetric(S)
        @test dm.L * dm.L' ≈ S
        @test_throws PosDefException DenseMetric([1.0 2.0; 2.0 1.0])
        # momentum r ~ N(0, M = Minv^-1): sample covariance of r must be inv(S)
        r = GravitationalWaveBayes.NUTSCore._draw_momentum(MersenneTwister(2), Float64, 3, 200_000, dm)
        @test isapprox(cov(r; dims=2), inv(S); rtol=0.03)
        # whitened norm of a draw from N(0, S) has mean-square = dimension
        x = dm.L * randn(MersenneTwister(3), 3, 50_000)
        @test isapprox(mean(abs2(_whitened_norm(x[:, k], dm)) for k in 1:size(x, 2)), 3.0; rtol=0.03)
    end

    @testset "FIXED mismatched dense metric leaves a correlated target stationary" begin
        T = Float64
        mu = [1.0, -2.0, 0.5]
        Sigma = _rotated_cov([1.0, 3.0, 0.2], 4)
        f = _corr_gaussian(mu, Sigma)
        # Deliberately WRONG, non-diagonal metric: must cost efficiency, never correctness.
        wrong = DenseMetric(_rotated_cov([2.0, 0.5, 1.0], 5))
        nwalkers = 8
        pos = randn(MersenneTwister(6), T, 3, nwalkers)
        cfg = NUTSConfig{T}(; max_treedepth=8, nwarmup=300)
        res = nuts_sample!(pos, f; nsteps=2000, step_size0=fill(0.3, nwalkers), cfg=cfg,
                            rng=MersenneTwister(7), record=true, inv_mass0=wrong)
        @test res.inv_mass === wrong               # nothing adapts without cfg.adapt_mass
        x = reshape(res.chain[:, :, 301:end], 3, :)
        @test all(abs.(mean(x; dims=2)[:, 1] .- mu) .< 0.15 .* sqrt.(diag(Sigma)))
        @test isapprox(cov(x; dims=2), Sigma; rtol=0.15, atol=0.03 * maximum(Sigma))
    end

    @testset "dense metric through reflecting bounds matches exact truncated moments" begin
        T = Float64
        Sigma = [1.0 0.8; 0.8 1.0]
        f = _corr_gaussian([0.0, 0.0], Sigma)
        lo, hi = [-0.7, -4.0], [1.2, 4.0]
        bounds = NUTSBounds(lo, hi)
        # exact moments of the truncated Gaussian by 2-D quadrature
        xs = range(lo[1], hi[1]; length=1200); ys = range(lo[2], hi[2]; length=2400)
        P = inv(Sigma)
        wgt = [exp(-0.5 * ([x, y]' * P * [x, y])) for x in xs, y in ys]
        wgt ./= sum(wgt)
        m1 = sum(wgt .* xs); m2 = sum(wgt .* reshape(ys, 1, :))
        v1 = sum(wgt .* (xs .- m1) .^ 2); v2 = sum(wgt .* (reshape(ys, 1, :) .- m2) .^ 2)
        c12 = sum(wgt .* (xs .- m1) .* (reshape(ys, 1, :) .- m2))
        # metric with a strong off-diagonal so the naive r_i -> -r_i flip would visibly bias this
        dm = DenseMetric([0.5 0.4; 0.4 0.6])
        nwalkers = 8
        pos = [0.1, 0.1] .+ 0.05 .* randn(MersenneTwister(8), T, 2, nwalkers)
        cfg = NUTSConfig{T}(; max_treedepth=8, nwarmup=300, bounds=bounds)
        res = nuts_sample!(pos, f; nsteps=3000, step_size0=fill(0.3, nwalkers), cfg=cfg,
                            rng=MersenneTwister(9), record=true, inv_mass0=dm)
        x = reshape(res.chain[:, :, 301:end], 2, :)
        @test all(lo[1] - 1e-9 .<= x[1, :] .<= hi[1] + 1e-9)
        @test isapprox(mean(x[1, :]), m1; atol=0.03)
        @test isapprox(mean(x[2, :]), m2; atol=0.06)
        @test isapprox(var(x[1, :]), v1; rtol=0.08)
        @test isapprox(var(x[2, :]), v2; rtol=0.08)
        @test isapprox(cov(x[1, :], x[2, :]), c12; atol=0.03)
    end

    @testset "dense billiard: stays in the box, conserves K, and is reversible" begin
        dm = DenseMetric([0.5 0.4 0.1; 0.4 0.6 -0.2; 0.1 -0.2 0.9])
        b = NUTSBounds([-1.0, -1.0, -0.5], [1.0, 1.0, 2.0])
        bp = NUTSBounds([-1.0, -1.0, 0.0], [1.0, 1.0, 2pi]; periodic=[false, false, true])
        rng = MersenneTwister(10)
        for bb in (b, bp), _ in 1:200
            theta = [0.6 * (2rand(rng) - 1), 0.6 * (2rand(rng) - 1), 0.3 + rand(rng)]; r = 3 .* randn(rng, 3)
            th0, r0 = copy(theta), copy(r)
            eps = 0.05 + 2rand(rng)                      # long enough for several wall hits
            K0 = GravitationalWaveBayes.NUTSCore._kinetic(r, dm)
            _drift_and_bound!(theta, r, eps, dm, bb)
            @test all(bb.lo[i] - 1e-12 <= theta[i] <= bb.hi[i] + 1e-12 for i in 1:3)
            @test GravitationalWaveBayes.NUTSCore._kinetic(r, dm) ≈ K0
            # reversibility: flip momentum, take the same step, flip back -> the start state
            theta2, r2 = copy(theta), -r
            _drift_and_bound!(theta2, r2, eps, dm, bb)
            @test isapprox(theta2, th0; atol=1e-9)
            @test isapprox(-r2, r0; atol=1e-9)
        end
    end

    @testset "Hessian metric recovers the exact covariance of a Gaussian" begin
        T = Float64
        sd = [1e-4, 1e-2, 0.3, 1.0]
        Sigma = _rotated_cov(sd, 11)
        f = _corr_gaussian(zeros(4), Sigma)
        bounds = NUTSBounds(fill(-50.0, 4), fill(50.0, 4))
        pos = 5.0 .* randn(MersenneTwister(12), T, 4, 16)            # far from the mode, all over the place
        cfg = NUTSConfig{T}(; adapt_mass=true, dense_mass=true, hessian_init=true, bounds=bounds, nwarmup=40)
        dm, nvalid = _hessian_metric(f, pos, cfg)
        @test nvalid == 16
        @test isapprox(dm.Minv, Sigma; rtol=1e-3, atol=1e-3 * minimum(sd)^2)
    end

    @testset "step-size heuristic lands near the right scale from either side" begin
        T = Float64
        Sigma = _rotated_cov([0.5, 2.0, 0.1], 13)
        f = _corr_gaussian(zeros(3), Sigma)
        dm = DenseMetric(Sigma)                       # exact metric -> right eps is O(1)
        pos = dm.L * randn(MersenneTwister(14), T, 3, 8)
        cfg = NUTSConfig{T}(; adapt_mass=true, dense_mass=true, nwarmup=30)
        for eps0 in (1e-8, 1e4)
            ss = fill(eps0, 8)
            _find_reasonable_step_size!(ss, pos, f, cfg, MersenneTwister(15), dm)
            @test all(0.05 .< ss .< 20)
        end
    end

    @testset "step_size_max_change caps a restart overshoot (2026-09-23/24 termbuffer-smoke)" begin
        # Reproduces, on a synthetic Gaussian (no GW likelihood, CPU-only, seconds not GPU-hours),
        # the mechanism the production handoff diagnosed: right after a metric change, the search
        # starts from a step size badly mismatched to the new metric's scale (here: a metric-naive
        # identity `inv_mass` against a target with a 5000:1 condition number), and a single test
        # leapfrog's accept/reject can read "good" for several doublings past the true crossing
        # before flipping -- overshooting the starting value by far more than any one restart
        # should be allowed to move. `max_change` (the kwarg `nuts_sample!`'s mid-warmup restart
        # call site passes as `cfg.step_size_max_change`, this fix) bounds that; the DEFAULT (`Inf`,
        # what the INITIAL bootstrap call site still uses) is deliberately left uncapped -- see
        # "step-size heuristic lands near the right scale from either side" above, which relies on
        # exactly that call site being able to climb many orders of magnitude in one call.
        T = Float64
        sd = [1e-3, 1.0, 5.0]
        Sigma = _rotated_cov(sd, 21)
        f = _corr_gaussian(zeros(3), Sigma)
        nwalkers = 8
        pos = zeros(3, nwalkers)
        ss0 = 1e-4
        cfg = NUTSConfig{T}(; adapt_mass=true, dense_mass=true, nwarmup=30)
        @test cfg.step_size_max_change == 8

        # Uncapped (the function's own default, matching the initial-bootstrap call site): reproduces
        # the unbounded-search overshoot directly.
        ss_uncapped = fill(ss0, nwalkers)
        _find_reasonable_step_size!(ss_uncapped, pos, f, cfg, MersenneTwister(1), nothing)
        @test maximum(ss_uncapped ./ ss0) > 8      # the instability is real: overshoots past 8x

        # Same target, same seed, `max_change=cfg.step_size_max_change` (matching the restart call
        # site): every walker's move is bounded both ways.
        ss_capped = fill(ss0, nwalkers)
        _find_reasonable_step_size!(ss_capped, pos, f, cfg, MersenneTwister(1), nothing;
                                     max_change=cfg.step_size_max_change)
        @test all(ss0 / 8 - eps() .<= ss_capped .<= ss0 * 8 + eps())
        # The cap actually bound something real here, not a vacuous check against a search that
        # would have stayed in range anyway.
        @test any(ss_capped .== ss0 * 8)

        @test_throws ArgumentError NUTSConfig{T}(; step_size_max_change=0.5)   # must be >= 1
        @test NUTSConfig{T}(; step_size_max_change=1.0).step_size_max_change == 1.0
    end

    @testset "warmup culling rescues walkers trapped in a much lower secondary mode" begin
        T = Float64
        # 1-D bimodal: main mode at 0 (logp 0), secondary at 40 (logp -100, i.e. posterior weight e^-100).
        # Half the walkers start in each basin; HMC cannot cross the ~ -800 barrier, so without culling
        # the trapped half stays at 40 forever. Both modes have unit sd.
        f = function (grad_out, logp_out, positions, active_mask)
            for w in 1:size(positions, 2)
                x = positions[1, w]
                la = -0.5 * x^2
                lb = -100 - 0.5 * (x - 40)^2
                m = max(la, lb)
                wa = exp(la - m); wb = exp(lb - m)
                logp_out[w] = m + log(wa + wb)
                grad_out[1, w] = (wa * (-x) + wb * (-(x - 40))) / (wa + wb)
            end
            return nothing
        end
        nwalkers = 16
        start() = reshape([iseven(w) ? 0.3 * randn(MersenneTwister(w)) : 40 + 0.3 * randn(MersenneTwister(w)) for w in 1:nwalkers], 1, :)
        function run(gap)
            cfg = NUTSConfig{T}(; max_treedepth=6, nwarmup=100, adapt_mass=true, cull_logp_gap=gap,
                                 mass_init_buffer=10, mass_base_window=20, mass_term_buffer=20)
            nuts_sample!(start(), f; nsteps=400, step_size0=fill(0.3, nwalkers), cfg=cfg,
                          rng=MersenneTwister(31), record=true)
        end
        off = run(0.0)
        @test count(x -> x > 20, off.chain[1, :, end]) >= 5                       # most of the trapped half stays trapped
        on = @test_logs (:info, r"culled 8 walker") match_mode = :any run(30.0)
        post = on.chain[1, :, 101:end]
        @test all(abs.(post) .< 8)                                                # nobody left in the wrong mode
        @test isapprox(mean(post), 0.0; atol=0.15)
        @test isapprox(var(vec(post)), 1.0; rtol=0.2)
        @test_throws ArgumentError NUTSConfig{T}(; cull_logp_gap=1.0)               # needs adapt_mass
        @test_throws ArgumentError nuts_sample!(zeros(1, 2), f; nsteps=2, step_size0=fill(0.3, 2),
            cfg=NUTSConfig{T}(; adapt_mass=true, nwarmup=30, cull_logp_gap=5.0), metric_rank=[1, 2])
    end

    @testset "ILL-CONDITIONED correlated target from a frozen-style start (the 2026-09-20 failure)" begin
        T = Float64
        sd = [1e-4, 1e-2, 0.3, 1.0]
        Sigma = _rotated_cov(sd, 16)
        mu = [0.3, -0.2, 0.1, 5.0]
        f = _corr_gaussian(mu, Sigma)
        nwalkers = 8
        Q = _rotation(4, 16)                                                   # same rotation as Sigma
# 8 posterior sd off along EVERY principal axis (|logp| ~ 1e2): far enough that a step size
# of 1e-6 is hopelessly wrong, near enough that NUTS' divergence threshold is not tripped.
start() = mu .+ Q * (8 .* sd .* randn(MersenneTwister(17), T, 4, nwalkers))
        nwarmup, nsamp = 150, 400
        depths = Int[]; nleap = Int[]
        cb = (it, pos, lp, ss, st) -> if it > nwarmup
            append!(depths, st.depth); append!(nleap, st.n_leapfrog)
        end
        # Distance moved per gradient evaluation, in units of the TRUE posterior sd (whitened by the
        # exact Sigma, independent of whichever metric a run adapted): the quantity that decides how
        # long a real (minutes-per-gradient) run needs.
        Lt = LowerTriangular(cholesky(Symmetric(Sigma)).L)
        function moved_per_grad(chain)
            post = chain[:, :, nwarmup:end]
            tot = 0.0
            for k in 2:size(post, 3), w in 1:nwalkers
                tot += norm(Lt \ (post[:, w, k] .- post[:, w, k - 1]))
            end
            return tot / sum(nleap)
        end
        # --- the OLD path: diagonal adaptation from the same tiny step ----------------------------
        cfg_d = NUTSConfig{T}(; max_treedepth=4, nwarmup=nwarmup, adapt_mass=true)
        res_d = nuts_sample!(start(), f; nsteps=nwarmup + nsamp, step_size0=fill(1e-6, nwalkers), cfg=cfg_d,
                              rng=MersenneTwister(18), record=true, callback=cb)
        old_depth = mean(depths); old_eff = moved_per_grad(res_d.chain)
        empty!(depths); empty!(nleap)
        # --- the new path: dense metric bootstrapped from the Hessian ------------------------------
        cfg_n = NUTSConfig{T}(; max_treedepth=4, nwarmup=nwarmup, adapt_mass=true, dense_mass=true,
                               hessian_init=true, mass_init_buffer=15, mass_base_window=25, mass_term_buffer=30)
        res_n = nuts_sample!(start(), f; nsteps=nwarmup + nsamp, step_size0=fill(1e-6, nwalkers), cfg=cfg_n,
                              rng=MersenneTwister(18), record=true, callback=cb)
        @test res_n.inv_mass isa DenseMetric
        new_depth = mean(depths); new_eff = moved_per_grad(res_n.chain)
        @info "ill-conditioned target" old_depth new_depth old_eff new_eff ratio = new_eff / old_eff
        # The old path is pinned at the depth cap (trajectories truncated, never reaching a natural
        # U-turn); the new one terminates on its own and moves much further per gradient evaluation.
        @test old_depth >= 3.5
        @test new_depth < old_depth - 1
        @test new_eff > 3 * old_eff
        # And the learned metric IS the posterior covariance (correlations included), and the chain
        # recovers it.
        S = res_n.inv_mass.Minv
        @test isapprox(S, Sigma; rtol=0.4, atol=0.4 * sd[1]^2)
        post = reshape(res_n.chain[:, :, nwarmup + 1:end], 4, :)
        @test all(abs.(mean(post; dims=2)[:, 1] .- mu) .< 0.3 .* sqrt.(diag(Sigma)))
        @test isapprox(cov(post; dims=2), Sigma; rtol=0.4, atol=0.4 * sd[1]^2)
    end
end
