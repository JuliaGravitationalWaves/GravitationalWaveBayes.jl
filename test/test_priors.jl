using Test
using Random
using GravitationalWaveBayes

# `GWPrior` (src/GravitationalWaveBayes/Priors.jl): default reproduces the package's flat box, closed-form gradients
# match finite differences, and `sample_prior` draws follow the stated densities.

const BNp = GravitationalWaveBayes.BatchedNUTS

@testset "GWPrior" begin
    rng = Random.Xoshiro(7)
    default = GWPrior{Float64}()
    pos = sample_prior(rng, default, 200)
    lp = zeros(200)
    prior_logpdf!(lp, default, pos)
    @test all(iszero, lp)
    @test all(BNp._call_log_prior(pos, w) == 0 for w in 1:200)
    bad = copy(pos[:, 1:1]); bad[1, 1] = 5.0
    @test prior_logpdf!(zeros(1), default, bad)[1] == -Inf

    pr = GWPrior{Float64}(; bounds=Dict("mc" => (5.0, 30.0), "dL" => (0.1, 2.0)), distance=:volume, masses=:components)
    x = sample_prior(rng, pr, 5)
    g = zeros(11, 5); l = zeros(5)
    prior_logpdf_and_grad!(l, g, pr, x)
    h = 1e-6
    for w in 1:5, r in 1:11
        xp = copy(x); xp[r, w] += h
        xm = copy(x); xm[r, w] -= h
        fd = (prior_logpdf!(zeros(5), pr, xp)[w] - prior_logpdf!(zeros(5), pr, xm)[w]) / (2h)
        @test isapprox(g[r, w], fd; atol=1e-5)
    end

    # distributions: dL ~ dL^2 on [0.1, 2] -> E[dL] = (3/4)(hi^4 - lo^4)/(hi^3 - lo^3)
    big = sample_prior(rng, pr, 200_000)
    lo, hi = 0.1, 2.0
    @test isapprox(sum(big[5, :]) / 200_000, 0.75 * (hi^4 - lo^4) / (hi^3 - lo^3); rtol=5e-3)
    # uniform in m1, m2 -> check the (mc, q) marginal of q against numerical integration of m1^2/mc
    qs = range(0.125, 1.0; length=401); mcs = range(5.0, 30.0; length=401)
    dens = [mc * (1 + q)^0.4 * q^-1.2 for mc in mcs, q in qs]
    pq = vec(sum(dens; dims=1)); pq ./= sum(pq)
    @test isapprox(sum(big[2, :]) / 200_000, sum(pq .* qs); rtol=5e-3)

    lo_, hi_, per = prior_bounds(pr)
    @test lo_[1] == 5.0 && hi_[5] == 2.0 && per == (1:11 .∈ Ref((7, 8, 11)))
    @test_throws ArgumentError GWPrior{Float64}(; bounds=Dict("ra" => (0.0, 1.0)))
    @test_throws ArgumentError GWPrior{Float64}(; distance=:comoving)
end

@testset "GWPrior with in-plane spins (nrows = 15)" begin
    # Precessing models (2026-10-05): each spin vector uniform in the ball |chi_i| <= max|chi_i bounds|, cut to the z box.
    rng = Random.Xoshiro(11)
    pr = GWPrior{Float64}(; bounds = Dict("chi1" => (-0.9, 0.9), "chi2" => (-0.5, 0.8)), nrows = 15)
    @test pr.spin_max == (0.9, 0.8)
    pos = sample_prior(rng, pr, 20_000)
    @test size(pos) == (15, 20_000)
    lp = zeros(20_000)
    prior_logpdf!(lp, pr, pos)
    @test all(iszero, lp)
    @test all(pos[3, :] .^ 2 .+ pos[12, :] .^ 2 .+ pos[13, :] .^ 2 .<= 0.81)
    @test all(-0.5 .<= pos[4, :] .<= 0.8) && all(pos[4, :] .^ 2 .+ pos[14, :] .^ 2 .+ pos[15, :] .^ 2 .<= 0.64)
    # uniform in the ball: |chi1|^3 / a^3 is uniform on [0, 1]
    u = sort(sqrt.(pos[3, :] .^ 2 .+ pos[12, :] .^ 2 .+ pos[13, :] .^ 2) .^ 3 ./ 0.9^3)
    @test maximum(abs.(u .- (1:20_000) ./ 20_000)) < 0.015
    out = copy(pos[:, 1:1]); out[3, 1] = 0.7; out[12, 1] = 0.7; out[13, 1] = 0.0
    @test prior_logpdf!(zeros(1), pr, out)[1] == -Inf
    # normalizer: the in-box fraction of uniform draws from the box equals vol / box volume
    lo, hi, _ = prior_bounds(pr)
    @test length(lo) == 15 && lo[12] == -0.9 && hi[15] == 0.8
    box = sum(log.(hi .- lo))
    n = 400_000; inside = 0
    for _ in 1:n
        x = lo .+ (hi .- lo) .* rand(rng, 15)
        inside += (x[3]^2 + x[12]^2 + x[13]^2 <= 0.81) && (x[4]^2 + x[14]^2 + x[15]^2 <= 0.64)
    end
    @test isapprox(prior_log_normalizer(pr), box + log(inside / n); atol = 0.01)
    @test_throws ArgumentError prior_logpdf!(zeros(1), pr, zeros(11, 1))
end

@testset "LVK spin map: per-body caps, aligned-spin z-projection" begin
    rng = Random.Xoshiro(5)
    # 15 rows, per-body caps: |chi_i| = a_i |y_i|^3, pullback = J' g (finite differences of a linear functional)
    Y = sample_prior(rng, GWPrior{Float64}(; nrows = 15), 4)
    a = (0.99, 0.05)
    X = isotropic_spins!(similar(Y), Y, a)
    @test all(isapprox.(sqrt.(sum(abs2, X[[12, 13, 3], :]; dims = 1)), 0.99 .* sqrt.(sum(abs2, Y[[12, 13, 3], :]; dims = 1)) .^ 3))
    @test all(isapprox.(sqrt.(sum(abs2, X[[14, 15, 4], :]; dims = 1)), 0.05 .* sqrt.(sum(abs2, Y[[14, 15, 4], :]; dims = 1)) .^ 3))
    @test isotropic_spins!(similar(Y), Y, 0.7) == isotropic_spins!(similar(Y), Y, (0.7, 0.7))
    c = randn(rng, 15, 4)
    g = isotropic_spins_pullback!(copy(c), Y, a)
    h = 1e-6
    for w in 1:4, r in (3, 4, 12, 13, 14, 15)
        Yp = copy(Y); Yp[r, w] += h; Ym = copy(Y); Ym[r, w] -= h
        fd = sum(c[:, w] .* (isotropic_spins!(similar(Yp), Yp, a)[:, w] .- isotropic_spins!(similar(Ym), Ym, a)[:, w])) / (2h)
        @test isapprox(g[r, w], fd; rtol = 1e-6, atol = 1e-9)
    end

    # 11 rows: y uniform on [-1, 1] -> p(chi) = log(a / |chi|) / (2a), CDF of |chi| / a: x (1 - log x)
    n = 200_000
    Y = zeros(11, n); Y[3, :] .= 2 .* rand(rng, n) .- 1; Y[4, :] .= 2 .* rand(rng, n) .- 1
    X = isotropic_spins!(similar(Y), Y, (0.99, 0.05))
    @test maximum(abs, X[3, :]) <= 0.99 && maximum(abs, X[4, :]) <= 0.05
    @test all(sign.(X[3, :]) .== sign.(Y[3, :]))
    x = sort(abs.(X[3, :]) ./ 0.99)
    @test maximum(abs.(x .* (1 .- log.(x)) .- (1:n) ./ n)) < 0.01
    # the inverse map is exact: x (1 - log x) = |y|
    xs = abs.(X[4, 1:1000]) ./ 0.05
    @test maximum(abs.(xs .* (1 .- log.(xs)) .- abs.(Y[4, 1:1000]))) < 1e-13
    @test isotropic_spins!(zeros(11, 1), zeros(11, 1), 0.9)[3] == 0.0
    yy = reshape([0.0, 0.0, 1.0, -1.0, 0, 0, 0, 0, 0, 0, 0], 11, 1)
    @test isotropic_spins!(similar(yy), yy, 0.9)[3:4] ≈ [0.9, -0.9]
    # pullback d chi / d y vs finite differences, interior points incl. small |y|
    Y = zeros(11, 6); Y[3, :] .= [1e-6, 0.01, 0.3, -0.5, 0.9, -0.999]; Y[4, :] .= [0.2, -0.7, 0.05, 1e-3, -0.95, 0.6]
    g = isotropic_spins_pullback!(ones(11, 6), Y, (0.99, 0.05))
    for w in 1:6, r in (3, 4)
        hh = 1e-7 * min(1.0, 1 - abs(Y[r, w]), abs(Y[r, w]) + 1e-12)
        Yp = copy(Y); Yp[r, w] += hh; Ym = copy(Y); Ym[r, w] -= hh
        fd = (isotropic_spins!(similar(Yp), Yp, (0.99, 0.05))[r, w] - isotropic_spins!(similar(Ym), Ym, (0.99, 0.05))[r, w]) / (2hh)
        @test isapprox(g[r, w], fd; rtol = 1e-5)
    end
    @test all(isfinite, isotropic_spins_pullback!(ones(11, 1), yy, 0.9))
end

@testset "GWPrior component-mass constraints" begin
    rng = Random.Xoshiro(3)
    m1f(mc, q) = mc * (1 + q)^(1 / 5) * q^(-3 / 5)
    for masses in (:components, :uniform_mc_q)
        pr = GWPrior{Float64}(; bounds = Dict("mc" => (10.2, 19.1), "q" => (0.05, 1.0)), masses, m1 = (5.0, 60.0),
                              m2 = (5.0, 60.0))
        X = sample_prior(rng, pr, 50_000)
        m1 = m1f.(X[1, :], X[2, :]); m2 = X[2, :] .* m1
        @test all(5 .<= m1 .<= 60) && all(5 .<= m2 .<= 60)
        @test all(isfinite, prior_logpdf!(zeros(50_000), pr, X))
        # q >= 0.05 at mc ~ 19 gives m1 > 60: excluded
        bad = copy(X[:, 1:1]); bad[1] = 19.0; bad[2] = 0.06
        @test prior_logpdf!(zeros(1), pr, bad)[1] == -Inf
        @test prior_support_vec(pr, bad)[1] == -Inf
        # draws follow the cut density: the uncut prior's draws, rejected, have the same q distribution
        free = GWPrior{Float64}(; bounds = Dict("mc" => (10.2, 19.1), "q" => (0.05, 1.0)), masses)
        Z = sample_prior(rng, free, 400_000)
        keep = [5 <= m1f(Z[1, w], Z[2, w]) <= 60 && 5 <= Z[2, w] * m1f(Z[1, w], Z[2, w]) <= 60 for w in axes(Z, 2)]
        @test isapprox(sum(X[2, :]) / 50_000, sum(Z[2, keep]) / count(keep); rtol = 0.01)
        # normalizer: the cut removes the uncut prior's rejected mass fraction
        @test isapprox(prior_log_normalizer(pr) - prior_log_normalizer(free), log(count(keep) / 400_000); atol = 0.01)
    end
    @test_throws ArgumentError GWPrior{Float64}(; m1 = (60.0, 5.0))
    @test_throws ArgumentError GWPrior{Float64}(; bounds = Dict("mc" => (10.0, 20.0)), m1 = (100.0, 200.0))
end

@testset "UniformSourceFrame log weights" begin
    # normalized: the mean weight over d_L^2 prior draws is 1; ratios match source_frame_prior_weights
    rng = Random.Xoshiro(9)
    lo, hi = 10.0, 10_000.0
    D = cbrt.(lo^3 .+ rand(rng, 400_000) .* (hi^3 - lo^3))
    lw = source_frame_log_weights(D, lo, hi)
    @test isapprox(sum(exp.(lw)) / length(lw), 1.0; rtol = 0.01)
    ref = source_frame_prior_weights(D[1:100])
    @test exp.(lw[1:100] .- lw[1]) ≈ ref ./ ref[1]
    # nearby sources (z << 1): both priors ~ d_L^2, weight -> the ratio of normalizers
    @test isapprox(exp(source_frame_log_weights([20.0], 10.0, 30.0)[1]), 1.0; rtol = 0.02)
end
