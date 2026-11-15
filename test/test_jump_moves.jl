using Test
using Random
using GravitationalWaveBayes
using GravitationalWaveBayes.NUTSCore: nuts_sample!, NUTSConfig, pt_position_swaps!
using GravitationalWaveData: H1, L1, gmst, time_delay_from_geocenter_gmst

const BNj = GravitationalWaveBayes.BatchedNUTS
const KFj = GravitationalWaveBayes.KernelFused

# NUTSCore jump moves: exactness on a target with two well-separated modes of unequal weight (plain
# NUTS started in one mode never visits the other), and the geometry of the GW moves.

@testset "jump moves: exact mode weights on a bimodal target" begin
    # p(x) ∝ 0.7 N(x; +m, 1) + 0.3 N(x; -m, 1) in the first coordinate, N(0,1) in the second.
    m = 8.0
    function lg!(grad, lp, pos, _)
        for w in axes(pos, 2)
            x, y = pos[1, w], pos[2, w]
            a = log(0.7) - (x - m)^2 / 2
            b = log(0.3) - (x + m)^2 / 2
            mx = max(a, b)
            lp[w] = mx + log(exp(a - mx) + exp(b - mx)) - y^2 / 2
            wa = exp(a - lp[w] - y^2 / 2); wb = 1 - wa
            grad[1, w] = -wa * (x - m) - wb * (x + m)
            grad[2, w] = -y
        end
    end
    flip! = x -> (x[1] = -x[1]; nothing)     # involution, |J| = 1
    rng = Random.Xoshiro(5)
    pos = fill(0.0, 2, 32); pos[1, :] .= m
    cfg = NUTSConfig{Float64}(; nwarmup=100, max_treedepth=6)
    r = nuts_sample!(pos, lg!; nsteps=1100, step_size0=fill(0.5, 32), cfg=cfg, rng=rng, record=true, jumps=(flip!,))
    post = r.chain[1, :, 101:end]
    frac = count(>(0), post) / length(post)
    @test isapprox(frac, 0.7; atol=0.03)
    @test 0.2 < r.jump_acceptance[1] < 0.8
    # without jumps the walkers never leave the start mode
    pos2 = fill(0.0, 2, 32); pos2[1, :] .= m
    r2 = nuts_sample!(pos2, lg!; nsteps=300, step_size0=fill(0.5, 32), cfg=cfg, rng=Random.Xoshiro(6), record=true)
    @test count(>(0), r2.chain[1, :, 101:end]) / length(r2.chain[1, :, 101:end]) > 0.99
end

@testset "gw_jump_moves geometry" begin
    freqs = collect(20.0:0.25:30.0)
    mk(det) = LikelihoodData(freqs, zeros(ComplexF64, length(freqs)), ones(length(freqs)), det, 1126259462.4)
    data = [mk(H1), mk(L1)]
    moves = BNj.gw_jump_moves(data)
    @test length(moves) == 3
    g = gmst(1126259462.4)
    rng = Random.Xoshiro(2)
    for _ in 1:200
        x = [12.0, 0.7, 0.1, 0.2, 0.3, 0.01, 1.0, 2pi * rand(rng), 2rand(rng) - 1, 2rand(rng) - 1, pi * rand(rng)]
        for f! in moves
            y = copy(x); f!(y)
            z = copy(y); f!(z)
            @test isapprox(z, x; atol=1e-9)                     # involution
            @test y[1:7] == x[1:7] && y[11] == x[11]           # only sky / inclination rows change
            # arrival-time delays at both detectors are preserved
            for det in (H1, L1)
                @test isapprox(time_delay_from_geocenter_gmst(det, y[8], asin(y[9]), g),
                               time_delay_from_geocenter_gmst(det, x[8], asin(x[9]), g); atol=1e-12)
            end
            @test 0 <= y[8] < 2pi && -1 <= y[9] <= 1
        end
    end
    @test length(BNj.gw_jump_moves(data[1])) == 1                # single detector: inclination flip only
end

@testset "DE jumps: exact mode weights from an evenly split start" begin
    m = 8.0
    function lg2!(grad, lp, pos, _)
        for w in axes(pos, 2)
            x, y = pos[1, w], pos[2, w]
            a = log(0.7) - (x - m)^2 / 2
            b = log(0.3) - (x + m)^2 / 2
            mx = max(a, b)
            lp[w] = mx + log(exp(a - mx) + exp(b - mx)) - y^2 / 2
            wa = exp(a - lp[w] - y^2 / 2); wb = 1 - wa
            grad[1, w] = -wa * (x - m) - wb * (x + m)
            grad[2, w] = -y
        end
    end
    cfg = NUTSConfig{Float64}(; nwarmup=100, max_treedepth=6)
    start() = (p = zeros(2, 32); p[1, 1:16] .= m; p[1, 17:32] .= -m; p)
    r = nuts_sample!(start(), lg2!; nsteps=1600, step_size0=fill(0.5, 32), cfg=cfg, rng=Random.Xoshiro(9),
                     record=true, de_jumps=true)
    post = r.chain[1, :, 101:end]
    @test isapprox(count(>(0), post) / length(post), 0.7; atol=0.03)
    @test r.de_acceptance > 0.05
    r0 = nuts_sample!(start(), lg2!; nsteps=400, step_size0=fill(0.5, 32), cfg=cfg, rng=Random.Xoshiro(9), record=true)
    @test isapprox(count(>(0), r0.chain[1, :, 101:end]) / length(r0.chain[1, :, 101:end]), 0.5; atol=0.02)

    # unimodal sanity: DE + NUTS leaves N(0, diag(1, 4)) intact
    function lgg!(grad, lp, pos, _)
        for w in axes(pos, 2)
            lp[w] = -pos[1, w]^2 / 2 - pos[2, w]^2 / 8
            grad[1, w] = -pos[1, w]; grad[2, w] = -pos[2, w] / 4
        end
    end
    rg = nuts_sample!(randn(Random.Xoshiro(1), 2, 16), lgg!; nsteps=2100, step_size0=fill(0.5, 16), cfg=cfg,
                      rng=Random.Xoshiro(2), record=true, de_jumps=true)
    xs = reshape(rg.chain[:, :, 101:end], 2, :)
    @test abs(sum(xs[1, :]) / size(xs, 2)) < 0.05
    @test isapprox(sum(abs2, xs[1, :]) / size(xs, 2), 1.0; rtol=0.06)
    @test isapprox(sum(abs2, xs[2, :]) / size(xs, 2), 4.0; rtol=0.06)
end

@testset "PT with several walkers per rung (pt_position_swaps!): finds an unseen mode, exact weights" begin
    # Same 0.7/0.3 bimodal target, but EVERY walker starts in the +m mode, so neither NUTS nor DE jumps
    # (which need a walker already in the other mode) can find -m; hot rungs flatten the barrier.
    m = 5.0
    K, ntemps = 16, 4
    betas = [0.03^((r - 1) / (ntemps - 1)) for r in 1:ntemps for _ in 1:K]
    rungs = [collect((r - 1) * K + 1 : r * K) for r in 1:ntemps]
    function ll_bimodal(x)
        a = log(0.7) - (x - m)^2 / 2
        b = log(0.3) - (x + m)^2 / 2
        mx = max(a, b)
        return mx + log(exp(a - mx) + exp(b - mx))
    end
    function lgt!(grad, lp, pos, _)
        for w in axes(pos, 2)
            x, y = pos[1, w], pos[2, w]
            l = ll_bimodal(x)
            wa = exp(log(0.7) - (x - m)^2 / 2 - l)
            dl = -wa * (x - m) - (1 - wa) * (x + m)
            lp[w] = betas[w] * l - y^2 / 2          # tempered likelihood, untempered N(0,1) "prior" on y
            grad[1, w] = betas[w] * dl
            grad[2, w] = -y
        end
    end
    ll = zeros(K * ntemps); acc = zeros(Int, ntemps - 1)
    rng = Random.Xoshiro(21)
    swap_cb = function (it, pos, logp, ss, stats)
        for w in axes(pos, 2)
            ll[w] = ll_bimodal(pos[1, w])
        end
        pt_position_swaps!(pos, ll, betas, rungs, acc, rng)
    end
    pos = zeros(2, K * ntemps); pos[1, :] .= m
    cfg = NUTSConfig{Float64}(; nwarmup=150, max_treedepth=6)
    r = nuts_sample!(pos, lgt!; nsteps=2150, step_size0=fill(0.5, K * ntemps), cfg=cfg, rng=rng,
                     record=true, callback=swap_cb, metric_walkers=rungs[1])
    cold = r.chain[1, rungs[1], 151:end]
    @test isapprox(count(>(0), cold) / length(cold), 0.7; atol=0.04)
    @test all(>(0.05), acc ./ (2150 * K))
    # the swap move itself: always accepted when it raises the tempered density of both
    p2 = [1.0 2.0]; l2 = [0.0, 10.0]; a2 = zeros(Int, 1)
    pt_position_swaps!(p2, l2, [1.0, 0.5], [[1], [2]], a2, Random.Xoshiro(1))
    @test p2 == [2.0 1.0] && l2 == [10.0, 0.0] && a2 == [1]
end

@testset "adaptive PT ladder (pt_adapt_ladder) + DEO swaps" begin
    NC = GravitationalWaveBayes.NUTSCore
    # Fixed point: rejection already equal on every pair -> ladder unchanged.
    b = [0.1^((r - 1) / 4) for r in 1:5]
    nb, lam = NC.pt_adapt_ladder(b, fill(0.3, 4))
    @test nb ≈ b && lam ≈ 1.2
    # A barrier concentrated on the cold pair pulls the rungs toward beta = 1; ends are kept.
    nb, _ = NC.pt_adapt_ladder(b, [0.9, 0.1, 0.1, 0.1])
    @test nb[1] == 1.0 && nb[end] ≈ 0.1 && issorted(nb; rev=true)
    @test nb[2] > b[2] && nb[3] > b[3]
    # Gaussian-barrier estimates.
    @test NC.pt_gaussian_barrier(2, exp(-1)) ≈ sqrt(1 / pi)
    @test NC.pt_recommended_ntemps(3.6) == 9 && NC.pt_recommended_ntemps(0.1) == 2 && NC.pt_recommended_ntemps(100) == 16

    # Tempered d-dim Gaussian (ll = -|x|^2/2 ... scaled by 1/sigma^2): the adapted ladder equalizes the
    # measured swap rejection and the measured barrier matches the Gaussian formula.
    d, K, ntemps, sig = 2, 8, 10, 0.05
    ladder = [1e-3^((r - 1) / (ntemps - 1)) for r in 1:ntemps]     # deliberately poor (geometric) start
    betas = [ladder[r] for r in 1:ntemps for _ in 1:K]
    rungs = [collect((r - 1) * K + 1 : r * K) for r in 1:ntemps]
    llg(x) = -sum(abs2, x) / (2 * sig^2)
    function lgg!(grad, lp, pos, _)
        for w in axes(pos, 2)
            x = view(pos, :, w)
            lp[w] = betas[w] * llg(x) - sum(abs2, x) / 2           # N(0, I) prior, untempered
            grad[:, w] .= -betas[w] .* x ./ sig^2 .- x
        end
    end
    rng = Random.Xoshiro(3)
    ll = zeros(K * ntemps); acc = zeros(Int, ntemps - 1)
    rej = zeros(ntemps - 1); att = zeros(Int, ntemps - 1)
    rounds = Ref(0); nwarm = 400
    adapt_at = (50, 100, 200, 300)
    lam_est = Ref(NaN)
    cb = function (it, pos, logp, ss, stats)
        for w in axes(pos, 2)
            ll[w] = llg(view(pos, :, w))
        end
        pt_position_swaps!(pos, ll, betas, rungs, acc, rng; parity=rounds[] % 2, rej_sum=rej, attempts=att)
        rounds[] += 1
        if it in adapt_at || it == nwarm + 600
            r = rej ./ att
            if it in adapt_at
                newl, lam_est[] = NC.pt_adapt_ladder(ladder, r)
                ladder .= newl
                for k in 1:ntemps, w in rungs[k]
                    betas[w] = ladder[k]
                end
            else
                lam_est[] = sum(r)
                @test maximum(r) - minimum(r) < 0.12               # equalized
            end
            rej .= 0; att .= 0
        end
    end
    pos = randn(Random.Xoshiro(4), d, K * ntemps) .* 0.05
    cfg = NUTSConfig{Float64}(; nwarmup=nwarm, max_treedepth=6)
    nuts_sample!(pos, lgg!; nsteps=nwarm + 600, step_size0=fill(0.02, K * ntemps), cfg=cfg, rng=rng,
                 callback=cb, metric_walkers=rungs[1])
    # Exact barrier with the N(0, I) prior: sd(-ll) = sqrt(d/2)/(beta + sig^2), so
    # Lambda = sqrt(d/2pi) log((1 + sig^2)/(beta_min + sig^2)); summed rejections are biased low by the
    # finite rung spacing, hence the tolerance.
    @test isapprox(lam_est[], sqrt(d / 2pi) * log((1 + sig^2) / (1e-3 + sig^2)); rtol=0.25)
end

@testset "PT exactness with DEO swaps and warmup ladder adaptation" begin
    m = 5.0
    K, ntemps = 16, 5
    ladder = [0.03^((r - 1) / (ntemps - 1)) for r in 1:ntemps]
    betas = [ladder[r] for r in 1:ntemps for _ in 1:K]
    rungs = [collect((r - 1) * K + 1 : r * K) for r in 1:ntemps]
    function llb(x)
        a = log(0.7) - (x - m)^2 / 2
        b = log(0.3) - (x + m)^2 / 2
        mx = max(a, b)
        return mx + log(exp(a - mx) + exp(b - mx))
    end
    function lgt!(grad, lp, pos, _)
        for w in axes(pos, 2)
            x, y = pos[1, w], pos[2, w]
            l = llb(x)
            wa = exp(log(0.7) - (x - m)^2 / 2 - l)
            lp[w] = betas[w] * l - y^2 / 2
            grad[1, w] = betas[w] * (-wa * (x - m) - (1 - wa) * (x + m))
            grad[2, w] = -y
        end
    end
    ll = zeros(K * ntemps); acc = zeros(Int, ntemps - 1)
    rej = zeros(ntemps - 1); att = zeros(Int, ntemps - 1)
    rng = Random.Xoshiro(22); rounds = Ref(0)
    cb = function (it, pos, logp, ss, stats)
        for w in axes(pos, 2)
            ll[w] = llb(pos[1, w])
        end
        pt_position_swaps!(pos, ll, betas, rungs, acc, rng; parity=rounds[] % 2, rej_sum=rej, attempts=att)
        rounds[] += 1
        if it in (20, 40, 80, 120)
            ladder .= GravitationalWaveBayes.NUTSCore.pt_adapt_ladder(ladder, rej ./ att)[1]
            for k in 1:ntemps, w in rungs[k]
                betas[w] = ladder[k]
            end
            rej .= 0; att .= 0
        end
    end
    pos = zeros(2, K * ntemps); pos[1, :] .= m
    cfg = NUTSConfig{Float64}(; nwarmup=160, max_treedepth=6)
    r = nuts_sample!(pos, lgt!; nsteps=2160, step_size0=fill(0.5, K * ntemps), cfg=cfg, rng=rng,
                     record=true, callback=cb, metric_walkers=rungs[1])
    cold = r.chain[1, rungs[1], 161:end]
    @test isapprox(count(>(0), cold) / length(cold), 0.7; atol=0.04)
    @test ladder[1] == 1.0 && ladder[end] ≈ 0.03 && issorted(ladder; rev=true)
end
