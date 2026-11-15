using Test
using GravitationalWaveBayes
using Random
using Statistics
using LinearAlgebra: I

# The SMC reference density (`Priors.GaussianPriorMixture`): annealing from it is evidence-exact only
# if it is NORMALIZED on the prior's support, so check the normalizers, its gradient, and its sampler.

@testset "GaussianPriorMixture: normalization, gradient, sampling" begin
    rng = Random.Xoshiro(7)
    pr = GWPrior{Float64}(; bounds=Dict("mc" => (15.0, 27.0), "q" => (0.3, 1.0), "chi1" => (-0.8, 0.8),
                                        "chi2" => (-0.8, 0.8), "dL" => (0.05, 2.0), "t_c" => (-0.05, 0.05)),
                          distance=:volume, masses=:components)
    lo, hi = pr.lo, pr.hi

    # prior_log_normalizer against Monte Carlo over the box.
    n = 400_000
    u = lo .+ (hi .- lo) .* rand(rng, 11, n)
    lp = prior_logpdf!(zeros(n), pr, u)
    mc_logC = sum(log, hi .- lo) + log(mean(exp.(lp)))
    @test isapprox(prior_log_normalizer(pr), mc_logC; atol=0.01)

    rows = [1, 2, 3, 4, 6]
    mu = [20.0, 0.75, 0.2, 0.0, 0.01]
    sd = [1.5, 0.12, 0.25, 0.3, 0.01]
    C = [0.6 0.1 0.0 0.0 0.0; 0.1 1.0 0.3 0.0 0.0; 0.0 0.3 1.0 0.0 0.0; 0.0 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 0.0 1.0]
    cov = (sd * sd') .* C + 1e-12 * I(5)
    ref = GaussianPriorMixture(pr, rows, mu, cov; alpha=0.2, rng=rng)

    # ∫ q0 = E_prior[q0 / p_normalized] = 1.
    x = sample_prior(rng, pr, 200_000)
    lq = zeros(size(x, 2)); g = zeros(size(x))
    reference_logpdf_and_grad!(lq, g, ref, x)
    lpx = prior_logpdf!(zeros(size(x, 2)), pr, x) .- prior_log_normalizer(pr)
    @test isapprox(mean(exp.(lq .- lpx)), 1.0; atol=0.03)

    # Gradient against central differences.
    xs = sample_reference(rng, ref, 6)
    lq6 = zeros(6); g6 = zeros(11, 6)
    reference_logpdf_and_grad!(lq6, g6, ref, xs)
    for w in 1:6, r in 1:11
        h = 1e-6 * max(1.0, abs(xs[r, w]))
        xp = copy(xs); xp[r, w] += h
        xm = copy(xs); xm[r, w] -= h
        lp_ = zeros(6); lm_ = zeros(6)
        reference_logpdf_and_grad!(lp_, zeros(11, 6), ref, xp)
        reference_logpdf_and_grad!(lm_, zeros(11, 6), ref, xm)
        @test isapprox(g6[r, w], (lp_[w] - lm_[w]) / (2h); atol=1e-4, rtol=1e-4)
    end

    # Sampler matches the density: E_q[f] by sampling vs by importance weighting prior draws.
    s = sample_reference(rng, ref, 200_000)
    wts = exp.(lq .- lpx)
    for r in (1, 2, 3, 6, 9)
        @test isapprox(mean(s[r, :]), sum(wts .* x[r, :]) / sum(wts); atol=0.02 * (hi[r] - lo[r]))
    end
    @test all(lo .<= minimum(s; dims=2)[:] ) && all(maximum(s; dims=2)[:] .<= hi)
end

@testset "KDEPriorMixture: normalization across a periodic wrap, gradient, sampling" begin
    rng = Random.Xoshiro(9)
    pr = GWPrior{Float64}(; bounds=Dict("mc" => (15.0, 27.0), "q" => (0.3, 1.0)), distance=:volume, masses=:components)
    lo, hi = pr.lo, pr.hi
    rows = [1, 2, 8, 9]                                   # mc, q, ra (periodic), sin_dec
    K = 12
    # Kernels straddling the ra = 0 / 2pi wrap and the q = 1 edge.
    centers = vcat(20 .+ randn(rng, 1, K), min.(0.95, 0.8 .+ 0.1 .* randn(rng, 1, K)),
                   mod.(0.1 .* randn(rng, 1, K), 2pi), -0.5 .+ 0.2 .* randn(rng, 1, K))
    bw = [1.5, 0.08, 0.3, 0.15]
    ref = KDEPriorMixture(pr, rows, centers, (bw * bw') .* I(4); alpha=0.2, rng=rng, nmc=20_000)

    x = sample_prior(rng, pr, 300_000)
    lq = zeros(size(x, 2))
    reference_logpdf_and_grad!(lq, zeros(size(x)), ref, x)
    lpx = prior_logpdf!(zeros(size(x, 2)), pr, x) .- prior_log_normalizer(pr)
    wts = exp.(lq .- lpx)
    @test isapprox(mean(wts), 1.0; atol=0.03)

    xs = sample_reference(rng, ref, 6)
    lq6 = zeros(6); g6 = zeros(11, 6)
    reference_logpdf_and_grad!(lq6, g6, ref, xs)
    for w in 1:6, r in 1:11
        h = 1e-6 * max(1.0, abs(xs[r, w]))
        xp = copy(xs); xp[r, w] += h
        xm = copy(xs); xm[r, w] -= h
        lp_ = zeros(6); lm_ = zeros(6)
        reference_logpdf_and_grad!(lp_, zeros(11, 6), ref, xp)
        reference_logpdf_and_grad!(lm_, zeros(11, 6), ref, xm)
        @test isapprox(g6[r, w], (lp_[w] - lm_[w]) / (2h); atol=1e-4, rtol=1e-4)
    end

    s = sample_reference(rng, ref, 300_000)
    @test all(lo .<= minimum(s; dims=2)[:]) && all(maximum(s; dims=2)[:] .<= hi)
    for r in (1, 2, 9)
        @test isapprox(mean(s[r, :]), sum(wts .* x[r, :]) / sum(wts); atol=0.02 * (hi[r] - lo[r]))
    end
    # ra is periodic: compare the circular mean
    cm(v, ww) = angle(sum(ww .* cis.(v)))
    @test abs(cm(s[8, :], ones(size(s, 2))) - cm(x[8, :], wts)) < 0.05
end

@testset "KDEPriorMixture on a 15-row prior: spin-ball normalizer and support" begin
    rng = Random.Xoshiro(11)
    pr = GWPrior{Float64}(; bounds=Dict("mc" => (15.0, 27.0), "q" => (0.3, 1.0), "chi1" => (-0.9, 0.9),
                                        "chi2" => (-0.8, 0.8)), distance=:volume, masses=:components, nrows=15)
    # rows: masses, both spin balls complete, t_c; kernels near the edge of ball 1 so its truncation matters
    rows = [1, 2, 3, 4, 6, 12, 13, 14, 15]
    K = 10
    centers = vcat(20 .+ randn(rng, 1, K), 0.8 .+ 0.05 .* randn(rng, 1, K), 0.6 .+ 0.05 .* randn(rng, 1, K),
                   0.1 .* randn(rng, 1, K), 0.01 .* randn(rng, 1, K), 0.6 .+ 0.05 .* randn(rng, 1, K),
                   0.1 .* randn(rng, 1, K), 0.2 .* randn(rng, 1, K), 0.2 .* randn(rng, 1, K))
    bw = [1.5, 0.06, 0.15, 0.2, 0.01, 0.15, 0.15, 0.2, 0.2]
    ref = KDEPriorMixture(pr, rows, centers, (bw * bw') .* I(9); alpha=0.2, rng=rng, nmc=20_000)
    # normalized, and its sampler draws from it: E_q0[p / q0] = 1 for the normalized prior p (bounded by 1 / alpha,
    # unlike E_p[q0 / p] with narrow kernels in 9 dimensions); a box-only in-support mass would leave it above 1
    s = sample_reference(rng, ref, 300_000)
    @test all(isfinite, prior_logpdf!(zeros(size(s, 2)), pr, s))      # draws stay inside the spin balls
    lq = zeros(size(s, 2))
    reference_logpdf_and_grad!(lq, zeros(size(s)), ref, s)
    lps = prior_logpdf!(zeros(size(s, 2)), pr, s) .- prior_log_normalizer(pr)
    @test isapprox(mean(exp.(lps .- lq)), 1.0; atol=0.01)
    # the same with the box-only kernel masses would fail: the kernels near the ball edge lose mass to it
    @test mean(s[3, :] .^ 2 .+ s[12, :] .^ 2 .+ s[13, :] .^ 2 .> 0.9^2) == 0
    # a marginal needs all or none of a ball's rows
    @test_throws ArgumentError KDEPriorMixture(pr, [1, 2, 3, 12], centers[[1, 2, 3, 6], :], (bw[[1, 2, 3, 6]] * bw[[1, 2, 3, 6]]') .* I(4);
                                               alpha=0.2, rng=rng, nmc=100)
    # the full-row marginal is the full normalizer
    @test GravitationalWaveBayes.Priors._log_marginal_normalizer(pr, 1:15) == prior_log_normalizer(pr)
    # a kernel far outside the box (a pass-1 time row beyond this prior's window) is an error, not an endless rejection
    out = copy(centers); out[5, :] .= pr.hi[6] + 0.2
    bad = KDEPriorMixture(pr, rows, out, (bw * bw') .* I(9); alpha=0.2, rng=rng, nmc=100)
    @test_throws ArgumentError sample_reference(rng, bad, 50)
end

@testset "local KDE follows a thin ring (per-kernel covariances)" begin
    rng = Random.Xoshiro(13)
    pr = GWPrior{Float64}()
    rows = [9, 10]                                          # sin_dec, cos_theta_jn: box [-1, 1]^2
    N = 3000
    r0, w0 = 0.5, 0.01
    th = 2pi .* rand(rng, N)
    rr = r0 .+ w0 .* randn(rng, N)
    X = vcat(reshape(rr .* cos.(th), 1, :), reshape(rr .* sin.(th), 1, :))
    ring_logpdf(x) = (r = hypot(x[1], x[2]); -0.5 * ((r - r0) / w0)^2 - log(sqrt(2pi) * w0) - log(2pi * r))
    idx = round.(Int, range(1, N; length=300))
    G = cov(X; dims=2)
    hs = 300^(-1 / 6)
    glob = KDEPriorMixture(pr, rows, X[:, idx], hs^2 .* G; alpha=0.05, rng=rng, nmc=2000)
    loc = KDEPriorMixture(pr, rows, X[:, idx], local_kde_covariances(X, idx, [0.0, 0.0]); alpha=0.05, rng=rng, nmc=2000)
    # Spread of log(target / q) under q (sampled rows only; the other rows are the prior in both).
    function ratio_sd(ref)
        s = sample_reference(rng, ref, 4000)
        lq = zeros(4000); reference_logpdf_and_grad!(lq, zeros(size(s)), ref, s)
        lp = prior_logpdf!(zeros(4000), pr, s) .- prior_log_normalizer(pr)
        lrows = -2 * log(2.0)                                # prior marginal on the two rows
        lt = [ring_logpdf(s[rows, w]) for w in 1:4000] .+ (lp .- lrows)
        good = isfinite.(lt) .& (lq .> log(0.5) .+ lp)      # the KDE part, not the 5 % prior part
        return std((lt .- lq)[good])
    end
    sd_glob, sd_loc = ratio_sd(glob), ratio_sd(loc)
    @test sd_loc < sd_glob / 3
    # Normalization and gradient of the local mixture.
    x = sample_prior(rng, pr, 200_000)
    lq = zeros(size(x, 2)); reference_logpdf_and_grad!(lq, zeros(size(x)), loc, x)
    lpx = prior_logpdf!(zeros(size(x, 2)), pr, x) .- prior_log_normalizer(pr)
    @test isapprox(mean(exp.(lq .- lpx)), 1.0; atol=0.05)
    xs = sample_reference(rng, loc, 4)
    g4 = zeros(11, 4); l4 = zeros(4); reference_logpdf_and_grad!(l4, g4, loc, xs)
    for w in 1:4, r in rows
        h = 1e-7
        xp = copy(xs); xp[r, w] += h; xm = copy(xs); xm[r, w] -= h
        lp_ = zeros(4); lm_ = zeros(4)
        reference_logpdf_and_grad!(lp_, zeros(11, 4), loc, xp); reference_logpdf_and_grad!(lm_, zeros(11, 4), loc, xm)
        @test isapprox(g4[r, w], (lp_[w] - lm_[w]) / (2h); rtol=1e-4, atol=1e-3)
    end
end
