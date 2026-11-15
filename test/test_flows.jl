using Test
using GravitationalWaveBayes
using GravitationalWaveBayes: fit_flow, flow_logpdf!, flow_sample, flow_to_latent, flow_from_latent, flow_latent_pull
using Random
using Statistics

# The normalizing flow behind the SMC flow proposals (`Flows.jl`).
const FL = GravitationalWaveBayes.Flows

@testset "flow: hand-written backprop matches finite differences" begin
    rng = Random.Xoshiro(1)
    d = 5
    X = rand(rng, d, 300)
    F = fit_flow(X, ones(300); lo=zeros(d), hi=ones(d), nlayers=3, hidden=8, steps=0, rng=rng)
    for L in F.layers                       # non-trivial layers (the fit starts from the identity)
        L.W3 .= 0.3 .* randn(rng, size(L.W3)); L.b3 .= 0.1 .* randn(rng, length(L.b3))
    end
    Y, _ = FL._box_to_y(F, X[:, 1:40])
    loss, g = FL._loss_and_grad(F, Y)
    maxrel = 0.0
    for (k, L) in enumerate(F.layers), f in FL._PFIELDS
        P = getfield(L, f); G = getfield(g[k], f)
        for i in rand(rng, 1:length(P), 4)
            h = 1e-6; old = P[i]
            P[i] = old + h; lp, _ = FL._loss_and_grad(F, Y)
            P[i] = old - h; lm, _ = FL._loss_and_grad(F, Y)
            P[i] = old
            fd = (lp - lm) / 2h
            maxrel = max(maxrel, abs(fd - G[i]) / max(abs(fd), 1e-3))
        end
    end
    @test maxrel < 1e-5
    # the training loss is the mean negative log density in y space
    lq = FL._logq_y(F, Y)
    @test loss ≈ -mean(lq)
    # sample -> logpdf consistency: inverse(forward(y)) == y
    Z = Y
    for L in F.layers
        Z, _, _ = FL._cforward(L, Z, F.smax)
    end
    for L in Iterators.reverse(F.layers)
        Z = FL._cinverse(L, Z, F.smax)
    end
    @test maximum(abs.(Z .- Y)) < 1e-10
end

@testset "flow: normalized on the box, fits a banana and two modes" begin
    rng = Random.Xoshiro(2)
    # banana in [-4, 4] x [-2, 10]: x1 ~ N(0, 1), x2 = x1^2 + N(0, 0.3^2)
    n = 6000
    x1 = randn(rng, n); x2 = x1 .^ 2 .+ 0.3 .* randn(rng, n)
    keep = (abs.(x1) .< 4) .& (-2 .< x2 .< 10)
    X = vcat(x1[keep]', x2[keep]')
    lo, hi = [-4.0, -2.0], [4.0, 10.0]
    F = fit_flow(X, ones(size(X, 2)); lo=lo, hi=hi, steps=2000, rng=rng)
    # normalization: E_uniform[q] * volume = 1
    U = lo .+ (hi .- lo) .* rand(rng, 2, 400_000)
    lqU = flow_logpdf!(zeros(size(U, 2)), F, U)
    @test isapprox(mean(exp.(lqU)) * prod(hi .- lo), 1.0; atol=0.03)
    # importance efficiency of the flow for the banana: ESS / n of weights p / q on flow samples
    S = flow_sample(rng, F, 20_000)
    @test all(lo[1] .<= S[1, :] .<= hi[1]) && all(lo[2] .<= S[2, :] .<= hi[2])
    lq = flow_logpdf!(zeros(size(S, 2)), F, S)
    lp = -0.5 .* S[1, :] .^ 2 .- 0.5 .* ((S[2, :] .- S[1, :] .^ 2) ./ 0.3) .^ 2
    lw = lp .- lq; w = exp.(lw .- maximum(lw))
    ess = sum(w)^2 / sum(abs2, w) / length(w)
    println("  banana flow: importance ESS fraction ", round(ess; digits=3))
    @test ess > 0.5
    # two separated modes, weights 0.3 / 0.7
    m = rand(rng, n) .< 0.3
    Y2 = vcat((ifelse.(m, -2.0, 2.0) .+ 0.2 .* randn(rng, n))', (0.2 .* randn(rng, n))')
    F2 = fit_flow(Y2, ones(n); lo=[-4.0, -2.0], hi=[4.0, 2.0], steps=2000, rng=rng)
    S2 = flow_sample(rng, F2, 20_000)
    @test abs(mean(S2[1, :] .< 0) - 0.3) < 0.03
    # weighted fit: equal-count samples, weights making it 0.3 / 0.7
    mw = vcat(fill(true, 3000), fill(false, 3000))
    Y3 = vcat((ifelse.(mw, -2.0, 2.0) .+ 0.2 .* randn(rng, 6000))', (0.2 .* randn(rng, 6000))')
    w3 = ifelse.(mw, 0.3, 0.7)
    F3 = fit_flow(Y3, w3; lo=[-4.0, -2.0], hi=[4.0, 2.0], steps=2000, rng=rng)
    @test abs(mean(flow_sample(rng, F3, 20_000)[1, :] .< 0) - 0.3) < 0.03
end

@testset "flow: warm start from a previous fit" begin
    rng = Random.Xoshiro(4)
    n = 3000
    x = randn(rng, n); y = 0.5 .* x .^ 2 .+ 0.3 .* randn(rng, n) .- 1.0
    X = vcat(clamp.(x, -3.9, 3.9)', clamp.(y, -1.9, 7.9)'); lo = [-4.0, -2.0]; hi = [4.0, 8.0]
    F1 = fit_flow(X, ones(n); lo, hi, steps=1500, nlayers=4, hidden=16, rng=Random.Xoshiro(1))
    # steps = 0: the parameters (and masks) of `init`, copied (training the copy leaves `init` alone)
    F0 = fit_flow(X, ones(n); lo, hi, steps=0, nlayers=4, hidden=16, init=F1, rng=Random.Xoshiro(2))
    @test all(L0.mask == L1.mask && L0.W1 == L1.W1 && L0.W3 == L1.W3 && L0.W1 !== L1.W1 for (L0, L1) in zip(F0.layers, F1.layers))
    # a short warm fit on slightly shifted samples beats a short cold one (held-out-free comparison on fresh samples)
    Xs = X .+ [0.1, 0.05]
    Fw = fit_flow(Xs, ones(n); lo, hi, steps=200, nlayers=4, hidden=16, init=F1, rng=Random.Xoshiro(3))
    Fc = fit_flow(Xs, ones(n); lo, hi, steps=200, nlayers=4, hidden=16, rng=Random.Xoshiro(3))
    xt = randn(rng, 2000); Xt = vcat(clamp.(xt, -3.9, 3.9)', clamp.(0.5 .* xt .^ 2 .+ 0.3 .* randn(rng, 2000) .- 1.0, -1.9, 7.9)') .+ [0.1, 0.05]
    lw = zeros(2000); lc = zeros(2000)
    flow_logpdf!(lw, Fw, Xt); flow_logpdf!(lc, Fc, Xt)
    @test mean(lw) > mean(lc)
    # a shape mismatch falls back to the cold start
    Fm = fit_flow(X, ones(n); lo, hi, steps=0, nlayers=3, hidden=16, init=F1, rng=Random.Xoshiro(2))
    @test length(Fm.layers) == 3
end

@testset "flow: periodic row across the seam" begin
    rng = Random.Xoshiro(3)
    n = 5000
    th = mod.(0.3 .* randn(rng, n), 2pi)            # von-Mises-like around 0 == 2 pi
    y = 0.5 .* randn(rng, n)
    X = vcat(th', y')
    F = fit_flow(X, ones(n); lo=[0.0, -3.0], hi=[2pi, 3.0], periodic=[true, false], steps=1500, rng=rng)
    S = flow_sample(rng, F, 20_000)
    @test all(0 .<= S[1, :] .< 2pi)
    @test abs(angle(mean(cis.(S[1, :])))) < 0.03
    @test isapprox(abs(mean(cis.(S[1, :]))), abs(mean(cis.(th))); atol=0.02)
    # a representative outside [lo, hi) evaluates the same
    l1 = flow_logpdf!(zeros(1), F, reshape([0.1, 0.2], 2, 1))
    l2 = flow_logpdf!(zeros(1), F, reshape([0.1 + 2pi, 0.2], 2, 1))
    @test l1 ≈ l2
end

@testset "flow online proposal in SMC: exact weights, mode switching" begin
    using GravitationalWaveBayes: smc_sample, NUTSConfig, NUTSBounds
    d, sigma, box = 3, 0.15, 5.0
    a, b = [-2.5, 0.0, 0.0], [2.5, 0.0, 0.0]
    wa, wb = 0.3, 0.7
    function lg(x)
        la = log(wa) - 0.5 * sum(abs2, (x .- a) ./ sigma); lb = log(wb) - 0.5 * sum(abs2, (x .- b) ./ sigma)
        mx = max(la, lb)
        return mx + log(exp(la - mx) + exp(lb - mx)) - 1.5 * log(2pi * sigma^2)
    end
    loglik! = (ll, pos) -> (for w in axes(pos, 2); ll[w] = lg(view(pos, :, w)); end; ll)
    logprior! = (lp, pos) -> (for w in axes(pos, 2); lp[w] = all(abs.(view(pos, :, w)) .<= box) ? 0.0 : -Inf; end; lp)
    lo, hi = fill(-box, d), fill(box, d)
    alpha = 0.1; lpn = d * log(2box)
    build = function (pool, W)
        F = fit_flow(pool, W; lo=lo, hi=hi, steps=600, hidden=32, nlayers=6, rng=Random.Xoshiro(size(pool, 2)))
        (; sample = function (r, n)
               X = flow_sample(r, F, n); U = lo .+ (hi .- lo) .* rand(r, d, n)
               for w in 1:n; rand(r) < alpha && (X[:, w] .= U[:, w]); end
               X
           end,
           logq_and_grad! = function (lq, g, pos)
               lf = flow_logpdf!(zeros(length(lq)), F, pos)
               lq .= log.((1 - alpha) .* exp.(lf) .+ alpha .* exp(-lpn)); fill!(g, 0.0); nothing
           end)
    end
    cfg = NUTSConfig{Float64}(; max_treedepth=6, nwarmup=0, bounds=NUTSBounds(lo, hi))
    fr = Float64[]; zs = Float64[]; sw = Float64[]; acc = Float64[]
    for seed in 1:3
        rng = Random.Xoshiro(10 + seed)
        res = smc_sample(box .* (2 .* rand(rng, d, 512) .- 1), loglik!, nothing, zeros(512); cfg=cfg, rng=rng,
                         moves=:rw, logprior! = logprior!, rw_de=0.0, nfinal=6, rw_thin=5, online_proposal=build,
                         online_pool_max=4000)
        push!(fr, mean(res.chain[1, :, :] .> 0)); push!(zs, res.logZ); push!(acc, res.independence_accept)
        push!(sw, mean((res.chain[1, :, 2:end] .> 0) .!= (res.chain[1, :, 1:end-1] .> 0)))
    end
    println("  flow-proposal SMC: mode-b fraction ", round(mean(fr); digits=3), " (0.7), logZ ", round(mean(zs); digits=3),
            " (exact ", round(-d * log(2box); digits=3), "), switches per recorded step ", round(mean(sw); digits=3),
            ", independence acceptance ", round(mean(acc); digits=3))
    @test abs(mean(fr) - wb) < 0.04
    @test abs(mean(zs) - (-d * log(2box))) < 0.25
    @test mean(sw) > 0.1                    # only the flow jumps can switch modes (no DE, sigma << separation)
end

@testset "flow latent map (NeuTra): round trip, density, pulled-back gradient" begin
    rng = Random.Xoshiro(5)
    d = 4
    lo, hi = [0.0, -1.0, 0.0, 2.0], [1.0, 3.0, 2pi, 5.0]
    per = [false, false, true, false]
    X = lo .+ (hi .- lo) .* rand(rng, d, 400)
    F = fit_flow(X, ones(400); lo, hi, periodic=per, nlayers=4, hidden=8, steps=0, rng=rng)
    for L in F.layers                       # non-trivial layers
        L.W3 .= 0.3 .* randn(rng, size(L.W3)); L.b3 .= 0.1 .* randn(rng, length(L.b3))
    end
    Z = randn(rng, d, 30)
    Xz, ld, cache = flow_from_latent(F, Z)
    @test all(lo .<= Xz .< hi)
    # x -> z inverts z -> x (periodic rows: any representative)
    @test maximum(abs.(flow_to_latent(F, Xz) .- Z)) < 1e-8
    Xs = copy(Xz); Xs[3, :] .+= 2pi
    @test maximum(abs.(flow_to_latent(F, Xs) .- Z)) < 1e-8
    # the flow's own density pulled back is N(0, I): log q(T(z)) + ldj = log N(z)
    lq = flow_logpdf!(zeros(30), F, Xz)
    @test maximum(abs.(lq .+ ld .- vec(-0.5 .* sum(abs2, Z; dims=1) .- 0.5d * log(2pi)))) < 1e-8
    # gradient of h(z) = c' T(z) + ldj(z) (c per column) against central differences
    C = randn(rng, d, 30)
    G = flow_latent_pull(F, cache, C)
    h(Zp) = (Xp = flow_from_latent(F, Zp); vec(sum(C .* Xp[1]; dims=1)) .+ Xp[2])
    maxrel = 0.0
    for r in 1:d
        e = 1e-5
        Zp = copy(Z); Zp[r, :] .+= e; Zm = copy(Z); Zm[r, :] .-= e
        fd = (h(Zp) .- h(Zm)) ./ 2e
        # (a periodic row may wrap between Zp and Zm: skip those columns)
        ok = abs.(fd) .< 1e3
        maxrel = max(maxrel, maximum(abs.(fd[ok] .- G[r, ok]) ./ max.(abs.(fd[ok]), 1e-2)))
    end
    @test maxrel < 1e-6
    # a non-finite outer gradient zeroes the column
    C2 = copy(C); C2[1, 1] = NaN
    @test all(iszero, flow_latent_pull(F, cache, C2)[:, 1])
end

@testset "latent-space kernels (NeuTra) and MALA in SMC: thin banana" begin
    using GravitationalWaveBayes: smc_sample, NUTSConfig, NUTSBounds
    # x1 ~ N(0, 1), x2 = x1^2 + N(0, s^2), x3 ~ N(0, s^2) in a box (truncation ~1e-4): a thin curved ridge, where
    # x-space NUTS with one population metric needs steps ~s, and the flow straightens it
    d, s = 3, 0.05
    lo, hi = [-4.5, -2.0, -1.0], [4.5, 22.0, 1.0]
    lg(x) = -0.5 * x[1]^2 - 0.5 * ((x[2] - x[1]^2) / s)^2 - 0.5 * (x[3] / s)^2 - 1.5 * log(2pi) - 2 * log(s)
    function lgg(x)
        r2 = (x[2] - x[1]^2) / s^2
        return [-x[1] + 2x[1] * r2, -r2, -x[3] / s^2]
    end
    inbox(x) = all(lo .<= x .<= hi)
    loglik! = (ll, pos) -> (for w in axes(pos, 2); ll[w] = lg(view(pos, :, w)); end; ll)
    logprior! = (lp, pos) -> (for w in axes(pos, 2); lp[w] = inbox(view(pos, :, w)) ? 0.0 : -Inf; end; lp)
    lpn = sum(log.(hi .- lo))
    alpha = 0.1
    build = function (pool, W)
        F = fit_flow(pool, W; lo=lo, hi=hi, steps=1500, hidden=32, nlayers=6, rng=Random.Xoshiro(size(pool, 2)))
        (; sample = function (r, n)
               X = flow_sample(r, F, n); U = lo .+ (hi .- lo) .* rand(r, d, n)
               for w in 1:n; rand(r) < alpha && (X[:, w] .= U[:, w]); end
               X
           end,
           logq_and_grad! = function (lq, g, pos)
               lf = flow_logpdf!(zeros(length(lq)), F, pos)
               lq .= log.((1 - alpha) .* exp.(lf) .+ alpha .* exp(-lpn)); fill!(g, 0.0); nothing
           end,
           to_latent = X -> flow_to_latent(F, X),
           from_latent = function (Z)
               X, ld, c = flow_from_latent(F, Z)
               X, ld, GX -> flow_latent_pull(F, c, GX)
           end)
    end
    cfg = NUTSConfig{Float64}(; max_treedepth=6, nwarmup=0, bounds=NUTSBounds(lo, hi))
    n = 512
    function run(seed; kw...)
        rng = Random.Xoshiro(seed)
        betas = zeros(n)
        lpg! = function (g, lp, pos, mask)
            for w in axes(pos, 2)
                x = view(pos, :, w)
                if inbox(x)
                    lp[w] = betas[w] * lg(x); g[:, w] .= betas[w] .* lgg(x)
                else
                    lp[w] = -Inf; g[:, w] .= 0
                end
            end
        end
        X0 = lo .+ (hi .- lo) .* rand(rng, d, n)
        steps = Float64[]
        res = smc_sample(X0, loglik!, lpg!, betas; cfg=cfg, rng=rng, moves=:rw, logprior! = logprior!, nuts_beta=0.02,
                         nuts_max_mutate=30, nfinal=8, online_proposal=build, online_pool_max=4000,
                         callback=(k, info) -> info.nuts && push!(steps, info.step_size), kw...)
        C = reshape(res.chain, d, :)
        return (; logZ=res.logZ, m1=mean(C[1, :]), v1=var(C[1, :]), m2=mean(C[2, :]), sd3=std(C[3, :]),
                step=isempty(steps) ? NaN : median(steps), nleap=res.n_leapfrog)
    end
    exact = -lpn
    for (name, kw) in (("x-space NUTS", (;)), ("latent NUTS", (; latent_nuts=true)),
                       ("latent MALA", (; latent_nuts=true, local_kernel=:mala, mala_steps=5)))
        r = [run(20 + k; kw...) for k in 1:2]
        println("  $name: logZ ", round.(getfield.(r, :logZ); digits=3), " (exact ", round(exact; digits=3), "), E x1 ",
                round.(getfield.(r, :m1); digits=3), ", V x1 ", round.(getfield.(r, :v1); digits=3), ", E x2 ",
                round.(getfield.(r, :m2); digits=3), " (1), sd x3 ", round.(getfield.(r, :sd3); digits=4), " (", s,
                "), median step ", round.(getfield.(r, :step); digits=3), ", gradient evals ", getfield.(r, :nleap))
        @test all(abs.(getfield.(r, :logZ) .- exact) .< 0.3)
        @test all(abs.(getfield.(r, :m1)) .< 0.15)
        @test all(abs.(getfield.(r, :v1) .- 1) .< 0.2)
        @test all(abs.(getfield.(r, :m2) .- 1) .< 0.2)
        @test all(abs.(getfield.(r, :sd3) .- s) .< 0.1s)
        name == "latent NUTS" && @test all(getfield.(r, :step) .> 0.2)
    end
end
