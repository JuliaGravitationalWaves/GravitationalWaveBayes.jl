using Test
using GravitationalWaveBayes
using GravitationalWaveBayes: smc_sample, NUTSConfig, NUTSBounds
using Random
using Statistics
import LinearAlgebra

# NUTSCore-level tests of the adaptive tempered SMC sampler (`SMCCore.jl`) on toy targets with a
# known evidence: a uniform box prior times (a) a narrow Gaussian likelihood and (b) a bimodal
# likelihood whose modes NUTS cannot cross at beta = 1, so the final mode fractions are set entirely
# by the importance weights accumulated on the way up the ladder.

# Tempered toy density `betas[w] * logL + 0` (flat prior inside the reflecting box).
function _toy_tempered(logL_and_grad, betas)
    return function (grad, logp, pos, mask)
        for w in axes(pos, 2)
            l, g = logL_and_grad(view(pos, :, w))
            logp[w] = betas[w] * l
            grad[:, w] .= betas[w] .* g
        end
        return nothing
    end
end
_toy_loglik(logL_and_grad) = (ll, pos) -> (for w in axes(pos, 2); ll[w] = logL_and_grad(view(pos, :, w))[1]; end; ll)

@testset "SMC: Gaussian likelihood, evidence and moments" begin
    d, sigma, box = 4, 0.1, 10.0
    m = [1.0, -2.0, 0.5, 3.0]
    lg(x) = (-0.5 * sum(abs2, (x .- m) ./ sigma), -(x .- m) ./ sigma^2)
    n = 256
    rng = Random.Xoshiro(11)
    pos0 = box .* (2 .* rand(rng, d, n) .- 1)
    betas = zeros(n)
    cfg = NUTSConfig{Float64}(; max_treedepth=8, nwarmup=0, bounds=NUTSBounds(fill(-box, d), fill(box, d)))
    res = smc_sample(pos0, _toy_loglik(lg), _toy_tempered(lg, betas), betas; cfg=cfg, rng=rng, nfinal=10)
    exact = (d / 2) * log(2pi * sigma^2) - d * log(2box)
    @test res.beta_schedule[end] == 1.0
    @test abs(res.logZ - exact) < 0.3
    samples = reshape(res.chain, d, :)
    @test all(abs.(vec(mean(samples; dims=2)) .- m) .< 0.02)
    @test all(abs.(vec(std(samples; dims=2)) ./ sigma .- 1) .< 0.1)
    # Stage count follows the Gaussian thermodynamic length sqrt(d/2) log(1/beta_1) / O(1).
    @test 5 <= res.nstages <= 40
end

@testset "SMC: final_max_treedepth caps the recorded phase only" begin
    d, sigma, box = 4, 0.1, 10.0
    m = [1.0, -2.0, 0.5, 3.0]
    lg(x) = (-0.5 * sum(abs2, (x .- m) ./ sigma), -(x .- m) ./ sigma^2)
    n, nfinal = 256, 10
    cfg = NUTSConfig{Float64}(; max_treedepth=8, nwarmup=0, bounds=NUTSBounds(fill(-box, d), fill(box, d)))
    run(nf, fd) = (r = Random.Xoshiro(11); betas = zeros(n);
                   smc_sample(box .* (2 .* rand(r, d, n) .- 1), _toy_loglik(lg), _toy_tempered(lg, betas), betas;
                              cfg=cfg, rng=r, nfinal=nf, final_max_treedepth=fd))
    ladder = run(0, nothing).n_leapfrog                 # same seed: the ladder is identical in every run
    full, capped = run(nfinal, nothing), run(nfinal, 2)
    @test capped.beta_schedule == full.beta_schedule
    @test 0 < capped.n_leapfrog - ladder <= n * nfinal * 2^2
    @test capped.n_leapfrog - ladder < full.n_leapfrog - ladder
    samples = reshape(capped.chain, d, :)
    @test all(abs.(vec(mean(samples; dims=2)) .- m) .< 0.03)
    @test all(abs.(vec(std(samples; dims=2)) ./ sigma .- 1) .< 0.15)
end

@testset "SMC: separated modes get the right weights" begin
    d, sigma, box = 2, 0.2, 10.0
    a, b = [-4.0, 0.0], [4.0, 0.0]
    wa, wb = 0.3, 0.7
    function lg(x)
        la = log(wa) - 0.5 * sum(abs2, (x .- a) ./ sigma)
        lb = log(wb) - 0.5 * sum(abs2, (x .- b) ./ sigma)
        mx = max(la, lb)
        pa, pb = exp(la - mx), exp(lb - mx)
        l = mx + log(pa + pb) - log(2pi * sigma^2)
        g = -(pa .* (x .- a) .+ pb .* (x .- b)) ./ (sigma^2 * (pa + pb))
        return l, g
    end
    n = 512
    rng = Random.Xoshiro(5)
    pos0 = box .* (2 .* rand(rng, d, n) .- 1)
    betas = zeros(n)
    cfg = NUTSConfig{Float64}(; max_treedepth=8, nwarmup=0, bounds=NUTSBounds(fill(-box, d), fill(box, d)))
    res = smc_sample(pos0, _toy_loglik(lg), _toy_tempered(lg, betas), betas; cfg=cfg, rng=rng,
                     dense=false, nfinal=5)
    @test abs(res.logZ - (-log((2box)^2))) < 0.3
    frac_b = mean(res.chain[1, :, :] .> 0)
    # One run's fraction scatters by ~0.06 (16 seeds: 0.698 +- 0.016, sd 0.063): a 2 sd band.
    @test abs(frac_b - wb) < 0.12
    @test res.ancestors_unique > 10
end

@testset "SMC: annealing from a reference (defensive mixture) is exact and shorter" begin
    d, sigma, box = 4, 0.1, 10.0
    m = [1.0, -2.0, 0.5, 3.0]
    lg(x) = (-0.5 * sum(abs2, (x .- m) ./ sigma), -(x .- m) ./ sigma^2)
    # Reference: a deliberately biased, too-wide Gaussian (a Laplace fit around an imperfect
    # reference point) mixed with the prior, alpha = 0.1, so it covers everything the prior does.
    mu, s, alpha = m .+ 0.1, 0.3, 0.1
    logprior = -d * log(2box)
    function logq_and_grad!(lq, g, pos)
        for w in axes(pos, 2)
            x = view(pos, :, w)
            lgauss = -0.5 * sum(abs2, (x .- mu) ./ s) - d * log(sqrt(2pi) * s)
            a, b = log(alpha) + logprior, log(1 - alpha) + lgauss
            mx = max(a, b)
            lq[w] = mx + log(exp(a - mx) + exp(b - mx))
            g[:, w] .= exp(b - lq[w]) .* (-(x .- mu) ./ s^2)
        end
        return nothing
    end
    logprior_and_grad!(lp, g, pos) = (fill!(lp, logprior); fill!(g, 0.0); nothing)
    n = 256
    rng = Random.Xoshiro(3)
    pos0 = similar(zeros(d, n))
    for w in 1:n
        pos0[:, w] .= rand(rng) < alpha ? box .* (2 .* rand(rng, d) .- 1) : mu .+ s .* randn(rng, d)
    end
    betas = zeros(n)
    cfg = NUTSConfig{Float64}(; max_treedepth=8, nwarmup=0, bounds=NUTSBounds(fill(-box, d), fill(box, d)))
    res = smc_sample(pos0, _toy_loglik(lg), _toy_tempered(lg, betas), betas; cfg=cfg, rng=rng, nfinal=10,
                     reference=(; logq_and_grad!, logprior_and_grad!))
    exact = (d / 2) * log(2pi * sigma^2) - d * log(2box)
    @test abs(res.logZ - exact) < 0.3
    samples = reshape(res.chain, d, :)
    @test all(abs.(vec(mean(samples; dims=2)) .- m) .< 0.02)
    @test all(abs.(vec(std(samples; dims=2)) ./ sigma .- 1) .< 0.1)
    @test res.nstages <= 8
end

@testset "SMC: independence moves from a reference cross between separated modes" begin
    d, sigma, box = 2, 0.2, 10.0
    a, b = [-4.0, 0.0], [4.0, 0.0]
    wa, wb = 0.3, 0.7
    function lg(x)
        la = log(wa) - 0.5 * sum(abs2, (x .- a) ./ sigma)
        lb = log(wb) - 0.5 * sum(abs2, (x .- b) ./ sigma)
        mx = max(la, lb)
        pa, pb = exp(la - mx), exp(lb - mx)
        return mx + log(pa + pb) - log(2pi * sigma^2), -(pa .* (x .- a) .+ pb .* (x .- b)) ./ (sigma^2 * (pa + pb))
    end
    # Reference: equal-weight wide Gaussians on both modes (deliberately wrong 0.5/0.5 weights) + prior.
    s_, alpha = 0.5, 0.1
    logprior = -d * log(2box)
    function logq_and_grad!(lq, g, pos)
        for w in axes(pos, 2)
            x = view(pos, :, w)
            ca = log(0.5) - 0.5 * sum(abs2, (x .- a) ./ s_) - d * log(sqrt(2pi) * s_)
            cb = log(0.5) - 0.5 * sum(abs2, (x .- b) ./ s_) - d * log(sqrt(2pi) * s_)
            cp = log(alpha) + logprior
            ca += log(1 - alpha); cb += log(1 - alpha)
            m = max(ca, cb, cp)
            lq[w] = m + log(exp(ca - m) + exp(cb - m) + exp(cp - m))
            g[:, w] .= exp(ca - lq[w]) .* (-(x .- a) ./ s_^2) .+ exp(cb - lq[w]) .* (-(x .- b) ./ s_^2)
        end
        return nothing
    end
    function sample_q(rng, n)
        out = zeros(d, n)
        for w in 1:n
            u = rand(rng)
            out[:, w] .= u < alpha ? box .* (2 .* rand(rng, d) .- 1) :
                         (u < alpha + (1 - alpha) / 2 ? a : b) .+ s_ .* randn(rng, d)
        end
        return out
    end
    logprior_and_grad!(lp, g, pos) = (fill!(lp, logprior); fill!(g, 0.0); nothing)
    n = 256
    rng = Random.Xoshiro(21)
    betas = zeros(n)
    cfg = NUTSConfig{Float64}(; max_treedepth=8, nwarmup=0, bounds=NUTSBounds(fill(-box, d), fill(box, d)))
    res = smc_sample(sample_q(rng, n), _toy_loglik(lg), _toy_tempered(lg, betas), betas; cfg=cfg, rng=rng,
                     dense=false, nfinal=10, independence=true,
                     reference=(; logq_and_grad!, logprior_and_grad!, sample=sample_q))
    @test abs(res.logZ - (-log((2box)^2))) < 0.3
    @test abs(mean(res.chain[1, :, :] .> 0) - wb) < 0.06
    # Particles switch modes during the recorded phase (NUTS alone never does at sigma = 0.2).
    switches = sum((res.chain[1, :, 2:end] .> 0) .!= (res.chain[1, :, 1:end-1] .> 0))
    @test switches > 0.25 * res.independence_accept * n * (size(res.chain, 3) - 1)
    @test 0.1 < res.independence_accept < 0.95
end

@testset "SMC: online KDE proposal (no reference) mixes separated modes, stays exact" begin
    d, sigma, box = 2, 0.2, 10.0
    a, b = [-4.0, 0.0], [4.0, 0.0]
    wa, wb = 0.3, 0.7
    function lg(x)
        la = log(wa) - 0.5 * sum(abs2, (x .- a) ./ sigma)
        lb = log(wb) - 0.5 * sum(abs2, (x .- b) ./ sigma)
        mx = max(la, lb)
        pa, pb = exp(la - mx), exp(lb - mx)
        return mx + log(pa + pb) - log(2pi * sigma^2), -(pa .* (x .- a) .+ pb .* (x .- b)) ./ (sigma^2 * (pa + pb))
    end
    # Pool -> KDE (isotropic bandwidth from nearest-neighbour distances) + 10% uniform box, normalized
    # (kernel mass outside the box is negligible here).
    function build(pool, W)
        K = size(pool, 2)
        dmin = [minimum(sqrt(sum(abs2, pool[:, i] .- pool[:, j])) for j in 1:K if j != i) for i in 1:K]
        h = max(sort(dmin)[cld(K, 2)], 1e-3)
        alpha, lbox = 0.1, -d * log(2box)
        function logq_and_grad!(lq, g, pos)
            for w in axes(pos, 2)
                x = view(pos, :, w)
                lk = [-0.5 * sum(abs2, (x .- pool[:, j]) ./ h) - d * log(sqrt(2pi) * h) for j in 1:K]
                m = maximum(lk); s_ = sum(exp.(lk .- m))
                lkde = m + log(s_) - log(K)
                A, B = log(alpha) + lbox, log(1 - alpha) + lkde
                mm = max(A, B); lq[w] = mm + log(exp(A - mm) + exp(B - mm))
                gk = sum(exp(lk[j] - m) / s_ .* (-(x .- pool[:, j]) ./ h^2) for j in 1:K)
                g[:, w] .= exp(B - lq[w]) .* gk
            end
            return nothing
        end
        sample(rng, n) = reduce(hcat, [rand(rng) < alpha ? box .* (2 .* rand(rng, d) .- 1) :
                                        pool[:, rand(rng, 1:K)] .+ h .* randn(rng, d) for _ in 1:n])
        return (; sample, logq_and_grad!)
    end
    n = 256
    rng = Random.Xoshiro(31)
    betas = zeros(n)
    cfg = NUTSConfig{Float64}(; max_treedepth=8, nwarmup=0, bounds=NUTSBounds(fill(-box, d), fill(box, d)))
    res = smc_sample(box .* (2 .* rand(rng, d, n) .- 1), _toy_loglik(lg), _toy_tempered(lg, betas), betas;
                     cfg=cfg, rng=rng, dense=false, nfinal=10, online_proposal=build, online_pool_max=300)
    @test abs(res.logZ - (-log((2box)^2))) < 0.3
    @test abs(mean(res.chain[1, :, :] .> 0) - wb) < 0.06
    switches = sum((res.chain[1, :, 2:end] .> 0) .!= (res.chain[1, :, 1:end-1] .> 0))
    @test switches > 0.25 * res.independence_accept * n * (size(res.chain, 3) - 1)
    @test res.independence_accept > 0.05
    # with `logprior!`, the independence moves evaluate beta * loglik + log prior (no gradient call), three per draw:
    # still exact
    betas2 = zeros(n)
    lprior! = (lp, pos) -> (for w in axes(pos, 2); lp[w] = all(abs.(pos[:, w]) .<= box) ? 0.0 : -Inf; end; lp)
    res2 = smc_sample(box .* (2 .* rand(rng, d, n) .- 1), _toy_loglik(lg), _toy_tempered(lg, betas2), betas2;
                      cfg=cfg, rng=rng, dense=false, nfinal=10, online_proposal=build, online_pool_max=300,
                      logprior! = lprior!, independence_tries=3)
    @test abs(res2.logZ - (-log((2box)^2))) < 0.3
    @test abs(mean(res2.chain[1, :, :] .> 0) - wb) < 0.06
    @test res2.independence_accept > 0.05
    # plus differential-evolution Metropolis steps (2 per draw): still exact, and they accept
    betas3 = zeros(n)
    res3 = smc_sample(box .* (2 .* rand(rng, d, n) .- 1), _toy_loglik(lg), _toy_tempered(lg, betas3), betas3;
                      cfg=cfg, rng=rng, dense=false, nfinal=10, online_proposal=build, online_pool_max=300,
                      logprior! = lprior!, independence_tries=3, de_tries=2)
    @test abs(res3.logZ - (-log((2box)^2))) < 0.3
    @test abs(mean(res3.chain[1, :, :] .> 0) - wb) < 0.06
    @test res3.de_accept > 0.05
    # gradient-free: random-walk + DE moves with online-KDE independence tries (likelihood only)
    betas4 = zeros(n)
    res4 = smc_sample(box .* (2 .* rand(rng, d, n) .- 1), _toy_loglik(lg), nothing, betas4;
                      cfg=cfg, rng=rng, moves=:rw, nfinal=10, online_proposal=build, online_pool_max=300,
                      logprior! = lprior!, independence_tries=2)
    @test abs(res4.logZ - (-log((2box)^2))) < 0.3
    @test abs(mean(res4.chain[1, :, :] .> 0) - wb) < 0.06
    @test res4.independence_accept > 0.05
end

@testset "SMC: persistent sampling keeps evidence, moments and mode weights exact" begin
    # Gaussian
    d, sigma, box = 4, 0.1, 10.0
    m = [1.0, -2.0, 0.5, 3.0]
    lg(x) = (-0.5 * sum(abs2, (x .- m) ./ sigma), -(x .- m) ./ sigma^2)
    # log Z scatters by ~0.24 per run at 512 particles (8 seeds; ~0.55-0.65 at 128): a 2.5 sd band.
    n = 512
    rng = Random.Xoshiro(41)
    betas = zeros(n)
    cfg = NUTSConfig{Float64}(; max_treedepth=8, nwarmup=0, bounds=NUTSBounds(fill(-box, d), fill(box, d)))
    res = smc_sample(box .* (2 .* rand(rng, d, n) .- 1), _toy_loglik(lg), _toy_tempered(lg, betas), betas;
                     cfg=cfg, rng=rng, nfinal=10, persistent=true)
    exact = (d / 2) * log(2pi * sigma^2) - d * log(2box)
    @test abs(res.logZ - exact) < 0.6
    samples = reshape(res.chain, d, :)
    @test all(abs.(vec(mean(samples; dims=2)) .- m) .< 0.02)
    @test all(abs.(vec(std(samples; dims=2)) ./ sigma .- 1) .< 0.1)

    # Separated modes, small population: weights exact on average, scatter over seeds reported.
    a, b = [-4.0, 0.0], [4.0, 0.0]
    function lg2(x)
        la = log(0.3) - 0.5 * sum(abs2, (x .- a) ./ 0.2)
        lb = log(0.7) - 0.5 * sum(abs2, (x .- b) ./ 0.2)
        mx = max(la, lb)
        pa, pb = exp(la - mx), exp(lb - mx)
        return mx + log(pa + pb) - log(2pi * 0.04), -(pa .* (x .- a) .+ pb .* (x .- b)) ./ (0.04 * (pa + pb))
    end
    cfg2 = NUTSConfig{Float64}(; max_treedepth=8, nwarmup=0, bounds=NUTSBounds(fill(-box, 2), fill(box, 2)))
    fracs = Dict(false => Float64[], true => Float64[])
    zs = Dict(false => Float64[], true => Float64[])
    for pers in (false, true), seed in 1:6
        r = Random.Xoshiro(100 + seed)
        bt = zeros(32)
        out = smc_sample(box .* (2 .* rand(r, 2, 32) .- 1), _toy_loglik(lg2), _toy_tempered(lg2, bt), bt;
                         cfg=cfg2, rng=r, dense=false, nfinal=5, persistent=pers)
        push!(fracs[pers], mean(out.chain[1, :, :] .> 0)); push!(zs[pers], out.logZ)
    end
    println("  mode-b fraction over 6 seeds (32 particles, truth 0.7): plain ", round(mean(fracs[false]); digits=3),
            " +- ", round(std(fracs[false]); digits=3), ", persistent ", round(mean(fracs[true]); digits=3), " +- ",
            round(std(fracs[true]); digits=3), "; log Z sd ", round(std(zs[false]); digits=3), " vs ", round(std(zs[true]); digits=3))
    @test abs(mean(fracs[true]) - 0.7) < 0.1
    @test abs(mean(zs[true]) - (-log((2box)^2))) < 0.3
end

@testset "SMC: robust stage-length statistic (median move / median pair distance)" begin
    _med = GravitationalWaveBayes.NUTSCore._slowest_row_move_median
    _rms = GravitationalWaveBayes.NUTSCore._slowest_row_move
    DM = GravitationalWaveBayes.NUTSCore.DenseMetric
    rng = Random.Xoshiro(3)
    n = 20_000
    # Gaussian rows: same calibration as the RMS / sd rule (independent -> sqrt(2), lag-one correlation 0.5 -> 1).
    x = randn(rng, 2, n)
    y = similar(x)
    y[1, :] .= randn(rng, n)
    y[2, :] .= 0.5 .* x[2, :] .+ sqrt(0.75) .* randn(rng, n)
    metric = DM([1.0 0.0; 0.0 1.0])
    v, r = _med(x, y, metric)
    @test r == 2 && abs(v - 1) < 0.04
    @test abs(_rms(x, y, metric)[1] - 1) < 0.03
    # Flat row (e.g. a uniform angle): independent draws still give sqrt(2) (a MAD scale would saturate at 0.79).
    @test abs(_med(rand(rng, 1, n), rand(rng, 1, n), DM(fill(1 / 12, 1, 1)))[1] - sqrt(2)) < 0.05
    # Bimodal row (10% minor mode 20 sd away), particles decorrelate within their mode, 1% jump between modes:
    # the robust statistic sees the within-mode move (~1.1), the RMS one is dominated by the mode spread / jumps.
    mode = rand(rng, n) .< 0.1
    xb = reshape(randn(rng, n) .+ 20 .* mode, 1, n)
    jump = rand(rng, n) .< 0.01
    yb = reshape(randn(rng, n) .+ 20 .* xor.(mode, jump), 1, n)
    mb = DM(fill(var(xb), 1, 1))
    @test 1.0 < _med(xb, yb, mb)[1] < 1.25
    @test _rms(xb, yb, mb)[1] < 0.5
    # Mostly duplicates (pair distance 0): falls back to the population sd instead of dividing by zero.
    xd = zeros(1, 100); yd = fill(0.5, 1, 100)
    @test _med(xd, yd, DM(fill(1.0, 1, 1)))[1] ≈ sqrt(2) * 0.5 / 0.9539
    @test_throws ArgumentError GravitationalWaveBayes.NUTSCore._slowest_move(:foo, x, y, metric)

    # Both rules end to end.
    d, sigma, box = 2, 0.1, 10.0
    lg(z) = (-0.5 * sum(abs2, z ./ sigma), -z ./ sigma^2)
    np = 256
    pos0 = box .* (2 .* rand(rng, d, np) .- 1)
    betas = zeros(np)
    cfg = NUTSConfig{Float64}(; max_treedepth=8, nwarmup=0, bounds=NUTSBounds(fill(-box, d), fill(box, d)))
    for stat in (:median, :rms)
        res = smc_sample(copy(pos0), _toy_loglik(lg), _toy_tempered(lg, betas), betas; cfg=cfg, rng=Random.Xoshiro(8),
                         nfinal=5, move_stat=stat)
        @test abs(res.logZ - ((d / 2) * log(2pi * sigma^2) - d * log(2box))) < 0.3
    end
end

@testset "SMC: a capturing callback works (Julia 1.13 dropped boxed-capture stores in mutate!)" begin
    lg(z) = (-0.5 * sum(abs2, z ./ 0.1), -z ./ 0.01)
    function run_with_callback()
        cfg = NUTSConfig{Float64}(; max_treedepth=8, nwarmup=0, bounds=NUTSBounds(fill(-10.0, 4), fill(10.0, 4)))
        ndraws = Int[]
        for seed in 1:2
            r = Random.Xoshiro(40 + seed); bt = zeros(64)
            smc_sample(10 .* (2 .* rand(r, 4, 64) .- 1), _toy_loglik(lg), _toy_tempered(lg, bt), bt; cfg=cfg, rng=r,
                       nfinal=2, callback=(s, i) -> push!(ndraws, i.ndraws))
        end
        return ndraws
    end
    @test all(>=(2), run_with_callback())
end

# Gradient-free random-walk mutations (`moves = :rw`), each compared with the NUTS SMC on the same target.
using LinearAlgebra: dot, logdet
# I0(k) = (1/pi) int_0^pi exp(k cos t) dt and I1/I0, by the midpoint rule (spectrally accurate for periodic integrands).
_bessel_mid(k, f) = (t = ((1:4000) .- 0.5) .* (pi / 4000); sum(f.(t) .* exp.(k .* cos.(t))) / 4000)
besseli0(k) = _bessel_mid(k, t -> 1.0)
besseli1_over_i0(k) = _bessel_mid(k, cos) / besseli0(k)
_box_logprior(box) = (lp, pos) -> (for w in axes(pos, 2); lp[w] = all(abs.(view(pos, :, w)) .<= box) ? 0.0 : -Inf; end; lp)

@testset "SMC random-walk moves: correlated Gaussian vs NUTS SMC" begin
    d, box = 6, 10.0
    m = [1.0, -2.0, 0.5, 3.0, -1.0, 0.0]
    # Strongly correlated, anisotropic posterior covariance (sds 0.02..0.2, correlations up to 0.95).
    A = [0.9^abs(i - j) for i in 1:d, j in 1:d]
    sd = [0.2, 0.1, 0.05, 0.02, 0.1, 0.15]
    S = sd .* A .* sd'
    Si = inv(S)
    lg(x) = (-0.5 * dot(x .- m, Si * (x .- m)), -(Si * (x .- m)))
    exact = 0.5 * logdet(2pi * S) - d * log(2box)
    n = 512
    cfg = NUTSConfig{Float64}(; max_treedepth=8, nwarmup=0, bounds=NUTSBounds(fill(-box, d), fill(box, d)))
    # One run's log Z scatters by ~0.25 for both kernels (8-16 seeds): compare means over 4 seeds.
    out = Dict(:nuts => [], :rw => [])
    for mv in (:nuts, :rw), seed in 1:4
        rng = Random.Xoshiro(16 + seed)
        betas = zeros(n)
        push!(out[mv], smc_sample(box .* (2 .* rand(rng, d, n) .- 1), _toy_loglik(lg),
                                  mv === :nuts ? _toy_tempered(lg, betas) : nothing, betas; cfg=cfg, rng=rng, nfinal=10,
                                  moves=mv, logprior! = _box_logprior(box)))
    end
    zr = mean(r.logZ for r in out[:rw]); zn = mean(r.logZ for r in out[:nuts])
    rw, nu = out[:rw][1], out[:nuts][1]
    println("  correlated Gaussian d=6 (mean of 4 seeds): logZ exact ", round(exact; digits=3), ", NUTS ", round(zn; digits=3),
            " (", nu.nstages, " stages, ", nu.n_leapfrog, " gradients), RW ", round(zr; digits=3), " (", rw.nstages,
            " stages, ", rw.n_loglik, " likelihoods, final scale ", round(rw.step_size; digits=3), ")")
    @test all(r -> r.n_leapfrog == 0, out[:rw])
    @test abs(zr - exact) < 0.3
    @test abs(zr - zn) < 0.6         # a difference of two 4-seed means: sd ~0.2
    for res in (rw, nu)
        X = reshape(res.chain, d, :)
        @test all(abs.(vec(mean(X; dims=2)) .- m) .< 0.15 .* sd)
        @test all(abs.(vec(std(X; dims=2)) ./ sd .- 1) .< 0.08)
        @test abs(cor(X[1, :], X[2, :]) - 0.9) < 0.03
    end
    # Acceptance tuned toward 0.234 on every ladder stage after the first few.
    @test all(a -> 0.12 < a < 0.4, rw.accept[3:end])
    @test rw.beta_schedule[end] == 1.0
end

@testset "SMC random-walk moves: separated modes get the right weights (vs NUTS SMC)" begin
    d, sigma, box = 2, 0.2, 10.0
    a, b = [-4.0, 0.0], [4.0, 0.0]
    wa, wb = 0.3, 0.7
    function lg(x)
        la = log(wa) - 0.5 * sum(abs2, (x .- a) ./ sigma)
        lb = log(wb) - 0.5 * sum(abs2, (x .- b) ./ sigma)
        mx = max(la, lb)
        pa, pb = exp(la - mx), exp(lb - mx)
        return mx + log(pa + pb) - log(2pi * sigma^2), -(pa .* (x .- a) .+ pb .* (x .- b)) ./ (sigma^2 * (pa + pb))
    end
    cfg = NUTSConfig{Float64}(; max_treedepth=8, nwarmup=0, bounds=NUTSBounds(fill(-box, d), fill(box, d)))
    # Mean over seeds (one run's mode fraction scatters by ~0.06 at 512 particles, see the NUTS test above).
    fr = Dict(:nuts => Float64[], :rw => Float64[]); zs = Dict(:nuts => Float64[], :rw => Float64[])
    for mv in (:nuts, :rw), seed in 1:4
        rng = Random.Xoshiro(60 + seed)
        betas = zeros(512)
        res = smc_sample(box .* (2 .* rand(rng, d, 512) .- 1), _toy_loglik(lg),
                         mv === :nuts ? _toy_tempered(lg, betas) : nothing, betas; cfg=cfg, rng=rng, dense=false,
                         nfinal=5, moves=mv, logprior! = _box_logprior(box))
        push!(fr[mv], mean(res.chain[1, :, :] .> 0)); push!(zs[mv], res.logZ)
    end
    println("  bimodal: mode-b fraction NUTS ", round(mean(fr[:nuts]); digits=3), " RW ", round(mean(fr[:rw]); digits=3),
            " (truth 0.7); logZ NUTS ", round(mean(zs[:nuts]); digits=3), " RW ", round(mean(zs[:rw]); digits=3),
            " (exact ", round(-log((2box)^2); digits=3), ")")
    @test abs(mean(fr[:rw]) - wb) < 0.07
    @test abs(mean(fr[:rw]) - mean(fr[:nuts])) < 0.1
    @test abs(mean(zs[:rw]) - (-log((2box)^2))) < 0.2
    # Differential-evolution jumps (default rw_de = 0.5) move particles between the modes during the recorded phase
    # (16 seeds: 0.34 switches per particle and step; mode-weight sd 0.010 vs 0.026 with Gaussian steps only).
    rng = Random.Xoshiro(70); betas = zeros(512)
    res = smc_sample(box .* (2 .* rand(rng, d, 512) .- 1), _toy_loglik(lg), nothing, betas; cfg=cfg, rng=rng, dense=false,
                     nfinal=5, moves=:rw, logprior! = _box_logprior(box))
    @test mean((res.chain[1, :, 2:end] .> 0) .!= (res.chain[1, :, 1:end-1] .> 0)) > 0.15
    res0 = smc_sample(box .* (2 .* rand(rng, d, 512) .- 1), _toy_loglik(lg), nothing, betas; cfg=cfg, rng=rng, dense=false,
                      nfinal=5, moves=:rw, logprior! = _box_logprior(box), rw_de=0.0)
    @test all((res0.chain[1, :, 2:end] .> 0) .== (res0.chain[1, :, 1:end-1] .> 0))
    # `rw_moves`: a caller-supplied symmetric proposal (the reflection x1 -> -x1, an involution preserving Lebesgue
    # measure) moves particles between the modes without DE jumps, and the weights stay exact; at beta = 1 the
    # acceptance is 0.3 * 1 + 0.7 * 0.3/0.7 = 0.6
    flip! = (xp, pos, rng) -> (xp .= pos; xp[1, :] .*= -1; xp)
    frm = Float64[]; accs = Float64[]
    for seed in 1:4
        rng = Random.Xoshiro(80 + seed); betas = zeros(512)
        resm = smc_sample(box .* (2 .* rand(rng, d, 512) .- 1), _toy_loglik(lg), nothing, betas; cfg=cfg, rng=rng,
                          dense=false, nfinal=5, moves=:rw, logprior! = _box_logprior(box), rw_de=0.0, rw_moves=(flip!,))
        push!(frm, mean(resm.chain[1, :, :] .> 0)); push!(accs, resm.move_accept)
        @test mean((resm.chain[1, :, 2:end] .> 0) .!= (resm.chain[1, :, 1:end-1] .> 0)) > 0.15
    end
    @test abs(mean(frm) - wb) < 0.05
    @test all(a -> 0.45 < a < 0.95, accs)       # (summed over all stages; low beta accepts more)
    @test_throws ArgumentError smc_sample(box .* (2 .* rand(rng, d, 512) .- 1), _toy_loglik(lg), _toy_tempered(lg, betas),
                                          betas; cfg=cfg, rng=rng, nfinal=5, rw_moves=(flip!,))
end

@testset "SMC hybrid: random walk below nuts_beta, NUTS above (+ rw_moves after NUTS draws)" begin
    # correlated Gaussian of the RW test: evidence and moments exact, NUTS used only from beta >= nuts_beta
    d, box = 6, 10.0
    m = [1.0, -2.0, 0.5, 3.0, -1.0, 0.0]
    A = [0.9^abs(i - j) for i in 1:d, j in 1:d]
    sd = [0.2, 0.1, 0.05, 0.02, 0.1, 0.15]
    S = sd .* A .* sd'; Si = inv(S)
    lg(x) = (-0.5 * dot(x .- m, Si * (x .- m)), -(Si * (x .- m)))
    exact = 0.5 * logdet(2pi * S) - d * log(2box)
    n = 512
    cfg = NUTSConfig{Float64}(; max_treedepth=8, nwarmup=0, bounds=NUTSBounds(fill(-box, d), fill(box, d)))
    zs = Float64[]; res = nothing; seen = Tuple{Float64,Bool}[]
    for seed in 1:4
        rng = Random.Xoshiro(30 + seed); betas = zeros(n)
        cb = (st, info) -> push!(seen, (info.beta, info.nuts))
        res = smc_sample(box .* (2 .* rand(rng, d, n) .- 1), _toy_loglik(lg), _toy_tempered(lg, betas), betas; cfg=cfg,
                         rng=rng, nfinal=10, moves=:rw, logprior! = _box_logprior(box), nuts_beta=0.01, callback=cb)
        push!(zs, res.logZ)
    end
    println("  hybrid correlated Gaussian: logZ ", round(mean(zs); digits=3), " (exact ", round(exact; digits=3), "), ",
            res.nstages, " stages, ", res.n_leapfrog, " gradients, ", res.n_loglik, " likelihoods")
    @test abs(mean(zs) - exact) < 0.3
    @test res.n_leapfrog > 0
    @test all(((b, nu),) -> nu == (b >= 0.01), seen)         # NUTS exactly from the first stage at beta >= nuts_beta
    @test any(((b, nu),) -> !nu, seen)                         # and random walk before it
    X = reshape(res.chain, d, :)
    @test all(abs.(vec(mean(X; dims=2)) .- m) .< 0.15 .* sd)
    @test all(abs.(vec(std(X; dims=2)) ./ sd .- 1) .< 0.08)
    # bimodal target with the reflection move after the NUTS draws: exact mode weights (NUTS alone cannot cross)
    sigma = 0.2; a, b = [-4.0, 0.0], [4.0, 0.0]; wa, wb = 0.3, 0.7
    function lg2(x)
        la = log(wa) - 0.5 * sum(abs2, (x .- a) ./ sigma); lb = log(wb) - 0.5 * sum(abs2, (x .- b) ./ sigma)
        mx = max(la, lb); pa, pb = exp(la - mx), exp(lb - mx)
        return mx + log(pa + pb) - log(2pi * sigma^2), -(pa .* (x .- a) .+ pb .* (x .- b)) ./ (sigma^2 * (pa + pb))
    end
    cfg2 = NUTSConfig{Float64}(; max_treedepth=8, nwarmup=0, bounds=NUTSBounds(fill(-box, 2), fill(box, 2)))
    flip! = (xp, pos, rng) -> (xp .= pos; xp[1, :] .*= -1; xp)
    fr = Float64[]
    for seed in 1:4
        rng = Random.Xoshiro(90 + seed); betas = zeros(512)
        r2 = smc_sample(box .* (2 .* rand(rng, 2, 512) .- 1), _toy_loglik(lg2), _toy_tempered(lg2, betas), betas;
                        cfg=cfg2, rng=rng, dense=false, nfinal=5, moves=:rw, logprior! = _box_logprior(box), rw_de=0.0,
                        rw_moves=(flip!,), nuts_beta=0.05)
        push!(fr, mean(r2.chain[1, :, :] .> 0))
        @test mean((r2.chain[1, :, 2:end] .> 0) .!= (r2.chain[1, :, 1:end-1] .> 0)) > 0.15   # flips in the NUTS phase
    end
    @test abs(mean(fr) - wb) < 0.05
    # metric from the top-half-logL core (`metric_core`): still exact
    zc = Float64[]
    for seed in 1:4
        rng = Random.Xoshiro(40 + seed); betas = zeros(n)
        push!(zc, smc_sample(box .* (2 .* rand(rng, d, n) .- 1), _toy_loglik(lg), _toy_tempered(lg, betas), betas;
                             cfg=cfg, rng=rng, nfinal=10, moves=:rw, logprior! = _box_logprior(box), nuts_beta=0.01,
                             metric_core=0.5).logZ)
    end
    println("  hybrid + metric_core 0.5: logZ ", round(mean(zc); digits=3), " (exact ", round(exact; digits=3), ")")
    @test abs(mean(zc) - exact) < 0.3
    Wm = GravitationalWaveBayes.NUTSCore._metric_weights(fill(0.25, 4), [-1.0, -5.0, 0.0, -3.0], 0.5)
    @test Wm == [0.5, 0.0, 0.5, 0.0]
    rng = Random.Xoshiro(1); betas = zeros(n)
    @test_throws ArgumentError smc_sample(box .* (2 .* rand(rng, d, n) .- 1), _toy_loglik(lg), nothing, betas; cfg=cfg,
                                          rng=rng, moves=:rw, logprior! = _box_logprior(box), nuts_beta=0.5)
end

@testset "SMC Hessian-augmented metric (hessian_metric)" begin
    # unit: a Gaussian whose population metric already has the target's shape (S at beta = 1) keeps it; a population
    # metric 100x too wide in one direction gets that direction's local width
    d = 4
    S = [0.9^abs(i - j) * 0.1 * i for i in 1:d, j in 1:d]; S = (S .+ S') ./ 2 .+ 0.05 .* LinearAlgebra.I(d)
    Si = inv(S); mu = [1.0, -1.0, 0.5, 2.0]
    lg(x) = (-0.5 * dot(x .- mu, Si * (x .- mu)), -(Si * (x .- mu)))
    n = 256; betas = ones(n)
    rng = Random.Xoshiro(3)
    X = mu .+ LinearAlgebra.cholesky(LinearAlgebra.Symmetric(S)).L * randn(rng, d, n)
    ll = [lg(view(X, :, w))[1] for w in 1:n]
    M, nh = GravitationalWaveBayes.NUTSCore._hessian_metric(GravitationalWaveBayes.NUTSCore.DenseMetric(S), X, ll, _toy_tempered(lg, betas), 8, true)
    @test nh == n
    @test maximum(abs.(M.Minv .- S)) < 1e-6 * maximum(abs.(S))
    Sw = copy(S); Sw[1, 1] *= 1e4                   # row 1 far too wide in the population metric
    M2, _ = GravitationalWaveBayes.NUTSCore._hessian_metric(GravitationalWaveBayes.NUTSCore.DenseMetric(Sw), X, ll, _toy_tempered(lg, betas), 8, true)
    @test M2.Minv[1, 1] < 2 * S[1, 1]               # back to (about) the target's width
    # end to end: a curved ridge (banana) x2 = a x1^2 + noise of sd 0.01, x1 ~ N(0, 1), 4 more N(0, 0.3^2) rows: the
    # population is wide across the ridge (its curvature), the target locally thin; the hybrid with the Hessian metric
    # stays exact and moves along the ridge with far fewer gradients
    d, box, a, s2 = 6, 10.0, 1.0, 0.01
    function lgb(x)
        r = x[2] - a * x[1]^2
        l = -0.5 * x[1]^2 - 0.5 * (r / s2)^2 - 0.5 * sum(abs2, view(x, 3:d)) / 0.09
        g = zeros(d)
        g[1] = -x[1] + (r / s2^2) * 2a * x[1]; g[2] = -r / s2^2; g[3:d] .= -view(x, 3:d) ./ 0.09
        return l, g
    end
    exact = log(2pi * s2^2) / 2 + log(2pi) / 2 + (d - 2) * log(2pi * 0.09) / 2 - d * log(2box)
    cfg = NUTSConfig{Float64}(; max_treedepth=8, nwarmup=0, bounds=NUTSBounds(fill(-box, d), fill(box, d)))
    out = Dict(0 => Float64[], 8 => Float64[]); grads = Dict(0 => 0, 8 => 0); x1sd = Dict(0 => Float64[], 8 => Float64[])
    for hm in (0, 8), seed in 1:3
        rng = Random.Xoshiro(50 + seed); betas = zeros(512)
        r = smc_sample(box .* (2 .* rand(rng, d, 512) .- 1), _toy_loglik(lgb), _toy_tempered(lgb, betas), betas;
                       cfg=cfg, rng=rng, nfinal=5, moves=:rw, logprior! = _box_logprior(box), nuts_beta=1e-3,
                       hessian_metric=hm)
        push!(out[hm], r.logZ); grads[hm] += r.n_leapfrog
        push!(x1sd[hm], std(vec(r.chain[1, :, :])))
    end
    println("  banana: logZ exact ", round(exact; digits=3), "; plain metric ", round(mean(out[0]); digits=3), " (",
            grads[0], " gradients, sd x1 ", round(mean(x1sd[0]); digits=3), "), Hessian metric ",
            round(mean(out[8]); digits=3), " (", grads[8], " gradients, sd x1 ", round(mean(x1sd[8]); digits=3), "; exact 1)")
    @test abs(mean(out[8]) - exact) < 0.3
    @test abs(mean(x1sd[8]) - 1) < 0.1
end

@testset "SMC random-walk moves: periodic row wrapped, hard boundary rejected" begin
    # Row 1: periodic on [0, 2pi) with a von Mises likelihood centred on the identification point 0 == 2pi.
    # Row 2: Gaussian truncated by the prior box at 0 (half-normal) -- RW proposals past it are rejected.
    kappa, s2 = 20.0, 0.5
    lg(x) = (kappa * cos(x[1]) - 0.5 * (x[2] / s2)^2, [-kappa * sin(x[1]), -x[2] / s2^2])
    lo, hi = [0.0, 0.0], [2pi, 5.0]
    lp!(lp, pos) = (for w in axes(pos, 2); lp[w] = (lo[2] <= pos[2, w] <= hi[2] && lo[1] <= pos[1, w] < hi[1]) ? 0.0 : -Inf; end; lp)
    cfg = NUTSConfig{Float64}(; max_treedepth=8, nwarmup=0, bounds=NUTSBounds(lo, hi; periodic=[true, false]))
    n = 512
    rng = Random.Xoshiro(9)
    betas = zeros(n)
    pos0 = vcat(2pi .* rand(rng, 1, n), 5 .* rand(rng, 1, n))
    res = smc_sample(pos0, _toy_loglik(lg), nothing, betas; cfg=cfg, rng=rng, nfinal=10, moves=:rw, logprior! = lp!)
    X = reshape(res.chain, 2, :)
    @test all(0 .<= X[1, :] .< 2pi) && all(X[2, :] .>= 0)
    # circular mean 0 (both sides of the seam populated), concentration of von Mises(kappa)
    @test abs(atan(mean(sin.(X[1, :])), mean(cos.(X[1, :])))) < 0.03
    @test mean(X[1, :] .< pi) > 0.4 && mean(X[1, :] .> pi) > 0.4
    @test abs(mean(cos.(X[1, :])) - besseli1_over_i0(kappa)) < 0.01
    # half-normal mean s2 sqrt(2/pi)
    @test abs(mean(X[2, :]) - s2 * sqrt(2 / pi)) < 0.03
    # exact logZ: log(2pi I0(kappa) e^0 ...) relative to the flat prior: log(I0(kappa)) + log(sqrt(pi/2) s2 / 5)
    @test abs(res.logZ - (log(besseli0(kappa)) + log(sqrt(pi / 2) * s2 / 5))) < 0.25
    @test_throws ArgumentError smc_sample(pos0, _toy_loglik(lg), nothing, zeros(n); cfg=cfg, moves=:rw)
end

@testset "SMC random-walk moves on a reference path (multi-fidelity)" begin
    # As the NUTS reference test: a biased, too-wide Gaussian reference mixed with the prior; RW moves only
    # (8 seeds: log Z - exact -0.02 +- 0.10, 3-4 stages).
    d, sigma, box = 4, 0.1, 10.0
    m = [1.0, -2.0, 0.5, 3.0]
    lg(x) = (-0.5 * sum(abs2, (x .- m) ./ sigma), -(x .- m) ./ sigma^2)
    mu, s, alpha = m .+ 0.1, 0.3, 0.1
    logprior = -d * log(2box)
    function logq_and_grad!(lq, g, pos)
        for w in axes(pos, 2)
            x = view(pos, :, w)
            lgauss = -0.5 * sum(abs2, (x .- mu) ./ s) - d * log(sqrt(2pi) * s)
            a, b = log(alpha) + logprior, log(1 - alpha) + lgauss
            mx = max(a, b); lq[w] = mx + log(exp(a - mx) + exp(b - mx)); g[:, w] .= 0
        end
        return nothing
    end
    logprior_and_grad!(lp, g, pos) = (fill!(lp, logprior); fill!(g, 0.0); nothing)
    n = 256
    rng = Random.Xoshiro(3)
    pos0 = zeros(d, n)
    for w in 1:n
        pos0[:, w] .= rand(rng) < alpha ? box .* (2 .* rand(rng, d) .- 1) : mu .+ s .* randn(rng, d)
    end
    cfg = NUTSConfig{Float64}(; max_treedepth=8, nwarmup=0, bounds=NUTSBounds(fill(-box, d), fill(box, d)))
    res = smc_sample(pos0, _toy_loglik(lg), nothing, zeros(n); cfg=cfg, rng=rng, nfinal=10, moves=:rw,
                     logprior! = _box_logprior(box), reference=(; logq_and_grad!, logprior_and_grad!))
    @test abs(res.logZ - ((d / 2) * log(2pi * sigma^2) - d * log(2box))) < 0.3
    X = reshape(res.chain, d, :)
    @test all(abs.(vec(mean(X; dims=2)) .- m) .< 0.02)
    @test all(abs.(vec(std(X; dims=2)) ./ sigma .- 1) .< 0.1)
    @test res.nstages <= 6
    # with online-proposal independence tries (a weighted Gaussian fit of the pool, inflated): still exact
    function gbuild(pool, W)
        mu_ = pool * W
        C = (pool .- mu_) * ((pool .- mu_) .* W')' .* 4.0 + 1e-6 * LinearAlgebra.I(d)
        L = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(C)).L
        ld = sum(log, LinearAlgebra.diag(L)) + d * log(sqrt(2pi))
        lq!(lq, g, pos) = (for w in axes(pos, 2); u = L \ (pos[:, w] .- mu_); lq[w] = -0.5 * sum(abs2, u) - ld;
                           g[:, w] .= 0; end; nothing)
        return (; sample = (r, nn) -> mu_ .+ L * randn(r, d, nn), logq_and_grad! = lq!)
    end
    res2 = smc_sample(pos0, _toy_loglik(lg), nothing, zeros(n); cfg=cfg, rng=rng, nfinal=10, moves=:rw,
                      logprior! = _box_logprior(box), reference=(; logq_and_grad!, logprior_and_grad!),
                      online_proposal=gbuild, independence_tries=2)
    @test abs(res2.logZ - ((d / 2) * log(2pi * sigma^2) - d * log(2box))) < 0.3
    X2 = reshape(res2.chain, d, :)
    @test all(abs.(vec(mean(X2; dims=2)) .- m) .< 0.02)
    @test all(abs.(vec(std(X2; dims=2)) ./ sigma .- 1) .< 0.1)
    @test res2.independence_accept > 0.05
end

@testset "SMC hybrid (RW -> NUTS, rw_moves) on a reference path (multi-fidelity)" begin
    # correlated Gaussian, a biased too-wide Gaussian reference mixed with the prior: evidence and moments exact with
    # NUTS from nuts_beta on (the NUTS target and the rw_moves' base carry log q0, 2026-10-11)
    d, box = 4, 10.0
    m = [1.0, -2.0, 0.5, 3.0]
    A = [0.8^abs(i - j) for i in 1:d, j in 1:d]
    sd = [0.2, 0.1, 0.05, 0.1]
    S = sd .* A .* sd'; Si = inv(S)
    lg(x) = (-0.5 * dot(x .- m, Si * (x .- m)), -(Si * (x .- m)))
    exact = 0.5 * logdet(2pi * S) - d * log(2box)
    function mkref(mu, s, alpha)
        logprior = -length(mu) * log(2box)
        function logq_and_grad!(lq, g, pos)
            for w in axes(pos, 2)
                x = view(pos, :, w)
                lgauss = -0.5 * sum(abs2, (x .- mu) ./ s) - sum(log.(sqrt(2pi) .* s))
                la_, lb_ = log(alpha) + logprior, log(1 - alpha) + lgauss   # (not a, b: the outer modes below)
                mx = max(la_, lb_); pa, pb = exp(la_ - mx), exp(lb_ - mx)
                lq[w] = mx + log(pa + pb); g[:, w] .= -(pb / (pa + pb)) .* (x .- mu) ./ s .^ 2
            end
            return nothing
        end
        logprior_and_grad!(lp, g, pos) = (fill!(lp, logprior); fill!(g, 0.0); nothing)
        sample(rng, nn) = reduce(hcat, [rand(rng) < alpha ? box .* (2 .* rand(rng, length(mu)) .- 1) :
                                        mu .+ s .* randn(rng, length(mu)) for _ in 1:nn])
        return (; logq_and_grad!, logprior_and_grad!, sample)
    end
    ref = mkref(m .+ 0.5 .* sd, 2 .* sd, 0.1)
    cfg = NUTSConfig{Float64}(; max_treedepth=8, nwarmup=0, bounds=NUTSBounds(fill(-box, d), fill(box, d)))
    n = 512; zs = Float64[]; seen = Tuple{Float64,Bool}[]; res = nothing
    for seed in 1:4
        rng = Random.Xoshiro(140 + seed); betas = zeros(n)
        cb = (st, info) -> push!(seen, (info.beta, info.nuts))
        res = smc_sample(ref.sample(rng, n), _toy_loglik(lg), _toy_tempered(lg, betas), betas; cfg=cfg, rng=rng,
                         nfinal=10, moves=:rw, logprior! = _box_logprior(box), reference=ref, nuts_beta=0.2, callback=cb)
        push!(zs, res.logZ)
    end
    println("  hybrid on a reference path: logZ ", round(mean(zs); digits=3), " (exact ", round(exact; digits=3), ")")
    @test abs(mean(zs) - exact) < 0.3
    @test any(((b, nu),) -> nu, seen) && any(((b, nu),) -> !nu, seen)
    X = reshape(res.chain, d, :)
    @test all(abs.(vec(mean(X; dims=2)) .- m) .< 0.15 .* sd)
    @test all(abs.(vec(std(X; dims=2)) ./ sd .- 1) .< 0.1)
    # bimodal target, a reference covering only the heavier mode (+ the prior), the reflection move after the NUTS
    # draws: the mode weights stay exact (a wrong base for the reflection would bias them toward q0's mode)
    sigma = 0.2; a, b = [-4.0, 0.0], [4.0, 0.0]; wa, wb = 0.3, 0.7
    function lg2(x)
        la = log(wa) - 0.5 * sum(abs2, (x .- a) ./ sigma); lb = log(wb) - 0.5 * sum(abs2, (x .- b) ./ sigma)
        mx = max(la, lb); pa, pb = exp(la - mx), exp(lb - mx)
        return mx + log(pa + pb) - log(2pi * sigma^2), -(pa .* (x .- a) .+ pb .* (x .- b)) ./ (sigma^2 * (pa + pb))
    end
    ref2 = mkref(b, [0.6, 0.6], 0.2)
    cfg2 = NUTSConfig{Float64}(; max_treedepth=8, nwarmup=0, bounds=NUTSBounds(fill(-box, 2), fill(box, 2)))
    flip! = (xp, pos, rng) -> (xp .= pos; xp[1, :] .*= -1; xp)
    fr = Float64[]; z2 = Float64[]
    for seed in 1:4
        rng = Random.Xoshiro(150 + seed); betas = zeros(512)
        r2 = smc_sample(ref2.sample(rng, 512), _toy_loglik(lg2), _toy_tempered(lg2, betas), betas; cfg=cfg2, rng=rng,
                        dense=false, nfinal=5, moves=:rw, logprior! = _box_logprior(box), rw_de=0.0, rw_moves=(flip!,),
                        nuts_beta=0.05, reference=ref2)
        push!(fr, mean(r2.chain[1, :, :] .> 0)); push!(z2, r2.logZ)
    end
    println("  hybrid + reflection on a reference path: weight ", round(mean(fr); digits=3), " (exact ", wb, "), logZ ",
            round(mean(z2); digits=3), " (exact ", round(-log((2box)^2); digits=3), ")")
    @test abs(mean(fr) - wb) < 0.05
    @test abs(mean(z2) - (-log((2box)^2))) < 0.3
end

@testset "SMC independence proposals outside the prior support are not evaluated" begin
    # prior uniform on the unit disk inside the box [-1, 1]^2; the likelihood THROWS outside the disk (like a waveform
    # at spin magnitudes > 1); a global proposal uniform on the box lands outside ~21% of the time
    d, n = 2, 256
    mu, sigma = [0.3, 0.0], 0.2
    inside(x) = sum(abs2, x) <= 1
    function lg(x)
        inside(x) || throw(DomainError(x, "outside the disk"))
        return -0.5 * sum(abs2, (x .- mu) ./ sigma) - log(2pi * sigma^2)
    end
    loglik! = (ll, pos) -> (for w in axes(pos, 2); ll[w] = lg(view(pos, :, w)); end; ll)
    logprior! = (lp, pos) -> (for w in axes(pos, 2); lp[w] = inside(view(pos, :, w)) ? 0.0 : -Inf; end; lp)
    betas = zeros(n)
    lpg! = function (g, lp, pos, mask)
        for w in axes(pos, 2)
            x = view(pos, :, w)
            if inside(x)
                lp[w] = betas[w] * lg(x); g[:, w] .= -betas[w] .* (x .- mu) ./ sigma^2
            else
                lp[w] = -Inf; g[:, w] .= 0
            end
        end
    end
    build = (pool, W) -> (; sample = (r, nn) -> 2 .* rand(r, d, nn) .- 1,
                            logq_and_grad! = (lq, g, pos) -> (lq .= -log(4.0); g .= 0; nothing))
    rng = Random.Xoshiro(4)
    pos0 = zeros(d, n)
    for w in 1:n
        x = 2 .* rand(rng, d) .- 1
        while !inside(x); x = 2 .* rand(rng, d) .- 1; end
        pos0[:, w] .= x
    end
    cfg = NUTSConfig{Float64}(; max_treedepth=6, nwarmup=0, bounds=NUTSBounds(fill(-1.0, d), fill(1.0, d)))
    res = smc_sample(pos0, loglik!, lpg!, betas; cfg=cfg, rng=rng, nfinal=10, logprior! = logprior!,
                     online_proposal=build, independence_tries=3)
    @test abs(res.logZ - (-log(pi))) < 0.15
    X = reshape(res.chain, d, :)
    @test all(abs.(vec(mean(X; dims=2)) .- mu) .< 0.03)
    @test res.independence_accept > 0.01
    # all of a draw's tries in one batched likelihood call (`ind_llk`, 3n columns): NUTS and random-walk stages
    nb = Ref(0)
    ind = (ll, pos) -> (size(pos, 2) == 3n || error("batch width"); nb[] += 1; loglik!(ll, pos))
    for mv in (:nuts, :rw)
        nb[] = 0
        rb = smc_sample(pos0, loglik!, lpg!, betas; cfg=cfg, rng=Random.Xoshiro(5), nfinal=10, logprior! = logprior!,
                        online_proposal=build, independence_tries=3, ind_llk=ind, moves=mv)
        @test nb[] > 0
        @test abs(rb.logZ - (-log(pi))) < 0.15
        Xb = reshape(rb.chain, d, :)
        @test all(abs.(vec(mean(Xb; dims=2)) .- mu) .< 0.03)
        @test all(abs.(vec(std(Xb; dims=2)) ./ sigma .- 1) .< 0.1)
        @test rb.independence_accept > 0.01
    end
    # the same on a REFERENCE path (2026-10-15): q0 uniform on the disk of radius 0.95 (normalized, unlike the prior's
    # density, so logq0 - logprior_ref is not constant); the batched tries then also carry q0 and the reference prior
    rq = 0.95
    inq(x) = sum(abs2, x) <= rq^2
    ref = (; logq_and_grad! = (lq, g, pos) -> (for w in axes(pos, 2); lq[w] = inq(view(pos, :, w)) ? -log(pi * rq^2) : -Inf; end; g .= 0; nothing),
             logprior_and_grad! = (lp, g, pos) -> (for w in axes(pos, 2); lp[w] = inside(view(pos, :, w)) ? -log(pi) : -Inf; end; g .= 0; nothing))
    rngq = Random.Xoshiro(6)
    posq = zeros(d, n)
    for w in 1:n
        x = 2rq .* rand(rngq, d) .- rq
        while !inq(x); x = 2rq .* rand(rngq, d) .- rq; end
        posq[:, w] .= x
    end
    for mv in (:nuts, :rw)
        nb[] = 0
        rr = smc_sample(posq, loglik!, lpg!, betas; cfg=cfg, rng=Random.Xoshiro(7), nfinal=10, logprior! = logprior!,
                        reference=ref, online_proposal=build, independence_tries=3, ind_llk=ind, moves=mv)
        @test nb[] > 0
        @test abs(rr.logZ - (-log(pi))) < 0.15
        Xr = reshape(rr.chain, d, :)
        @test all(abs.(vec(mean(Xr; dims=2)) .- mu) .< 0.03)
        @test all(abs.(vec(std(Xr; dims=2)) ./ sigma .- 1) .< 0.1)
        @test rr.independence_accept > 0.01
    end
end
