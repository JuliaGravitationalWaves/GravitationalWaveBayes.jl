using Test
using GravitationalWaveBayes
using GravitationalWaveBayes: smc_sample, NUTSConfig, NUTSBounds
using Random
using Statistics

# Learned mode jumps in SMC (`smc_sample(...; mode_jumps=true)`, `SMCCore._mode_jump_learn!` / `_mode_jump_tries!`).
# The toy is experiments/widthcheck/toy_jump.jl's "pedestal" (Papers repo): a narrow high lobe (sd 0.03, 5% of the mass)
# on a broad low pedestal next to a bulk Gaussian, in the box [-1, 1]^6. The lobe holds little weight until beta ~ 1, so
# tempering resampling leaves it nearly empty and NUTS / RW cannot refill it at beta = 1 (the GW150914 XPHM corner).

const _MJ = GravitationalWaveBayes.NUTSCore
const _PD = 6; const _PSB = 0.1; const _PMU = [0.45, 0.4, 0.35, -0.3, 0.2, 0.1]; const _PSL = 0.03; const _PSP = 0.15
const _PPP = -8.0; const _PLL = log(0.05 / 0.95) + _PD * log(_PSB / _PSL)
_plse(v...) = (m = maximum(v); m + log(sum(exp(x - m) for x in v)))
function _ped_lg(x)
    lb = -0.5 * sum(abs2, x ./ _PSB); la = -0.5 * sum(abs2, (x .- _PMU) ./ _PSL) + _PLL
    lp = -0.5 * sum(abs2, (x .- _PMU) ./ _PSP) + _PPP
    l = _plse(lb, la, lp)
    g = exp(lb - l) .* (-x ./ _PSB^2) .+ exp(la - l) .* (-(x .- _PMU) ./ _PSL^2) .+ exp(lp - l) .* (-(x .- _PMU) ./ _PSP^2)
    return l, g
end
_ped_in(x) = sum(abs2, (x .- _PMU) ./ _PSL) < 9.0              # within 3 sd of the lobe
_ped_frac(X) = mean(_ped_in(view(X, :, w)) for w in axes(X, 2))
_ped_box(lp, pos) = (for w in axes(pos, 2); lp[w] = all(x -> abs(x) <= 1, view(pos, :, w)) ? 0.0 : -Inf; end; lp)
_ped_llk(ll, pos) = (for w in axes(pos, 2); ll[w] = all(x -> abs(x) <= 1, view(pos, :, w)) ? _ped_lg(pos[:, w])[1] : -Inf; end; ll)
function _ped_tempered(betas)
    return function (g, lp, pos, mask)
        for w in axes(pos, 2)
            l, gg = _ped_lg(pos[:, w]); lp[w] = betas[w] * l; g[:, w] .= betas[w] .* gg
        end
    end
end
# exact draws (the box truncation is negligible for all three components)
function _ped_exact(rng, n)
    p = [1.0, exp(_PLL) * (_PSL / _PSB)^_PD, exp(_PPP) * (_PSP / _PSB)^_PD]; p ./= sum(p)
    X = zeros(_PD, n); k = 0
    while k < n
        u = rand(rng)
        x = u < p[1] ? _PSB .* randn(rng, _PD) : u < p[1] + p[2] ? _PMU .+ _PSL .* randn(rng, _PD) : _PMU .+ _PSP .* randn(rng, _PD)
        all(v -> abs(v) <= 1, x) && (k += 1; X[:, k] .= x)
    end
    return X
end

@testset "mode jumps: the move keeps exact draws exact (fixed beta), sequential and speculative rounds" begin
    rng = Random.Xoshiro(3); n = 20000
    X0 = _ped_exact(rng, n); f0 = _ped_frac(X0)
    @test abs(f0 - 0.05 * 0.826) < 0.006                     # lobe mass x P(chi2_6 < 9)
    deltas = [copy(_PMU), _PMU .* [1, -1, 1, 1, -1, 1] .* 0.5, 0.3 .* randn(rng, _PD)]
    for batch in (1, 3)
        X = copy(X0); ll = _ped_llk(zeros(n), X); lb = zeros(n); cur = copy(ll)
        buf = (zeros(_PD, batch * n), zeros(batch * n), zeros(batch * n), falses(batch * n))
        acc = Ref(0); cnt = Ref(0); ncols = 0; calls = Ref(0)
        llk_c = (l, p) -> (calls[] += 1; _ped_llk(l, p))
        for _ in 1:20
            ncols += _MJ._mode_jump_tries!(X, cur, ll, lb, falses(n), 1.0, deltas, 1:_PD, nothing, 12, 0.2, 0.7, batch,
                                           llk_c, _ped_box, buf, acc, cnt, rng)
        end
        @test cnt[] == 20 * 12 * n                            # every try counted once
        @test 0.1 < acc[] / cnt[] < 0.5                       # the vectors do move particles
        @test abs(_ped_frac(X) - f0) < 4 * sqrt(f0 * (1 - f0) / n)
        @test all(abs.(vec(mean(X; dims=2)) .- vec(mean(X0; dims=2))) .< 0.004)
        @test cur ≈ _ped_llk(zeros(n), X)                     # tracked densities stay those of the states
        @test ll == cur
        # one call per round, `batch n` columns each: sequential = 12 calls per sweep; speculative >= 4, fewer than 12
        @test ncols == calls[] * batch * n
        batch == 1 ? (@test calls[] == 20 * 12) : (@test 20 * 4 <= calls[] < 20 * 12)
    end
    # an empty set is a no-op without likelihood calls
    calls = Ref(0); cnt = Ref(0)
    @test _MJ._mode_jump_tries!(copy(X0), zeros(n), zeros(n), zeros(n), falses(n), 1.0, Vector{Float64}[], 1:_PD, nothing,
                                12, 0.2, 0.7, 1, (l, p) -> (calls[] += 1; l), _ped_box,
                                (zeros(_PD, n), zeros(n), zeros(n), falses(n)), Ref(0), cnt, rng) == 0
    @test calls[] == 0 && cnt[] == 0
end

@testset "mode jumps: learned vectors are cell-mean differences (circular on periodic rows), jump rows only" begin
    rng = Random.Xoshiro(5); n = 600
    # two clusters in (x, phase, y): x 0 vs 3, phase near 0.1 vs near 2pi - 0.1 (adjacent across the wrap), y unused
    X = zeros(3, n)
    for w in 1:n
        a = w <= n ÷ 3
        X[1, w] = (a ? 0.0 : 3.0) + 0.05 * randn(rng)
        X[2, w] = mod((a ? 0.1 : -0.1) + 0.02 * randn(rng), 2pi)
        X[3, w] = randn(rng)
    end
    b = NUTSBounds([-10.0, 0.0, -10.0], [10.0, 2pi, 10.0]; periodic=[false, true, false])
    D = Vector{Vector{Float64}}()
    added, cells = _MJ._mode_jump_learn!(D, X, [1, 2], b, 2, 0.01, rng)
    @test (added, cells) == (1, 2)
    d = D[1] .* sign(D[1][1])                                 # oriented big-x minus small-x
    @test isapprox(d[1], 3.0; atol=0.02)
    @test isapprox(d[2], -0.2; atol=0.01)                     # the short way round, not 2pi - 0.2
    @test d[3] == 0                                           # not a jump row
    # a cell with fewer than max(10, min_cell n) members gives no vector; the set is never pruned. (On x alone: the far
    # points hold nearly all of the standardized variance, so k-means++ seeds a centre there. A row that is nearly
    # constant across the population -- here sin(phase) -- is stretched to unit variance by the standardization and can
    # make k-means split a tight cluster along its noise instead.)
    Y = hcat(X[:, 1:n÷3], [8.0, 3.0, 0.0] .+ 0.01 .* randn(rng, 3, 9))
    @test _MJ._mode_jump_learn!(D, Y, [1], b, 2, 0.01, rng) == (0, 1)
    @test length(D) == 1
end

# 2026-11-01: 5 seeds per variant, checks on the 5-seed means (with 3 seeds and +-0.015 the k-means check was
# seed-fragile). Observed on the box CPU (seeds 101-105): plain 0.0014 0.026 0 0.035 0 (mean 0.012); k-means jumps
# 0.034 0.050 0.018 0.014 0.009 (mean 0.025); tail jumps 0.031 0.028 0.029 0.027 0.019 (mean 0.027). Single seeds
# scatter by +-0.015 either way; the jumps refill a good part of the lobe on average.
const _PED_N = 512; const _PED_NFINAL = 150; const _PED_SEEDS = 101:105
_ped_cfg() = NUTSConfig{Float64}(; max_treedepth=7, nwarmup=0, bounds=NUTSBounds(fill(-1.0, _PD), fill(1.0, _PD)))
function _ped_smc(seed, jumps, learn)
    rng = Random.Xoshiro(seed); betas = zeros(_PED_N)
    seen = Int[]; added = Int[]; evals = Int[]
    r = smc_sample(2 .* rand(rng, _PD, _PED_N) .- 1, _ped_llk, _ped_tempered(betas), betas; cfg=_ped_cfg(), rng, moves=:rw,
                   logprior! = _ped_box, nuts_beta=0.05, nuts_max_mutate=30, max_mutate=100, nfinal=_PED_NFINAL,
                   mode_jumps=jumps, mode_jump_learn=learn,
                   callback=(st, info) -> (push!(seen, info.mode_jumps); push!(added, info.mode_jumps_added);
                                           push!(evals, info.mode_jump_evals)))
    return r, _ped_frac(reshape(r.chain[:, :, _PED_NFINAL÷2+1:end], _PD, :)), seen, added, evals
end
const _PED_TRUTH = 0.05 * 0.826
_ped_plain = Float64[]
@testset "mode jumps in SMC: tempering alone drains the lobe" begin
    for seed in _PED_SEEDS
        r, f, = _ped_smc(seed, false, :kmeans)
        @test r.mode_jumps == 0 && isnan(r.mode_jump_accept) && r.mode_jump_evals == 0
        push!(_ped_plain, f)
    end
    println("  pedestal lobe (truth ", round(_PED_TRUTH; digits=4), "): plain ", round.(_ped_plain; digits=4))
    @test mean(_ped_plain) < 0.02
end
@testset "mode jumps in SMC: refill a narrow lobe that tempering drains (hybrid RW -> NUTS), $learn" for learn in (:kmeans, :tails)
    jumped = Float64[]
    for seed in _PED_SEEDS
        r, f, seen, added, evals = _ped_smc(seed, true, learn)
        push!(jumped, f)
        @test r.mode_jumps > 0 && 0 < r.mode_jump_accept < 1 && r.mode_jump_evals > 0
        @test issorted(seen)                                  # the set only grows
        @test seen == cumsum(added)                           # ... by what each stage learned, after the stage
        # frozen per stage: a stage makes jump tries only if the set was non-empty at its start
        @test evals[1] == 0
        @test all(k -> evals[k] == evals[k-1] || seen[k-1] > 0, 2:length(evals))
    end
    println("  pedestal lobe ($learn): mode jumps ", round.(jumped; digits=4))
    @test mean(jumped) > mean(_ped_plain) + 0.008
    @test abs(mean(jumped) - _PED_TRUTH) < 0.02
end

@testset "mode jumps: tail vectors = mean(tail) - mean(rest) (circular on periodic rows), jump rows only" begin
    rng = Random.Xoshiro(11); n = 1000; nt = 20
    # rows (x, phase, y): the 20 lowest-x particles sit at x ~ -3, phase ~ 2pi - 0.1 (across the wrap from the rest's
    # 0.1); y is not a jump row
    X = zeros(3, n)
    for w in 1:n
        t = w <= nt
        X[1, w] = (t ? -3.0 : 0.0) + 0.1 * randn(rng)
        X[2, w] = mod((t ? -0.1 : 0.1) + 0.01 * randn(rng), 2pi)
        X[3, w] = randn(rng) + (t ? 5.0 : 0.0)
    end
    b = NUTSBounds([-10.0, 0.0, -10.0], [10.0, 2pi, 10.0]; periodic=[false, true, false])
    D = Vector{Vector{Float64}}()
    q = [Xp -> Xp[1, :]]
    @test _MJ._mode_jump_tails!(D, X, [1, 2], b, q, (0.02, 0.05), 8, 0) == (4, 1)   # lower + upper tail per fraction
    lo2 = D[1]
    @test isapprox(lo2[1], -3.0; atol=0.05)
    @test isapprox(lo2[2], -0.2; atol=0.01)                  # the short way round
    @test lo2[3] == 0
    tl = 1:nt; rest = nt+1:n                                  # exact: the tail is the first 20 columns
    @test lo2[1] ≈ mean(X[1, tl]) - mean(X[1, rest])
    # the 5% lower tail mixes the shifted 20 with 30 bulk particles; the upper tails point (slightly) the other way
    @test isapprox(D[3][1], -1.33; atol=0.1) && D[2][1] > 0 && D[4][1] > 0
    # fewer than `min_count` particles in a tail: no vector for that fraction
    @test _MJ._mode_jump_tails!(D, X, [1, 2], b, q, (0.02, 0.05), 30, 0) == (2, 1)
    @test length(D) == 6
    # non-finite quantity values are left out of both tail and rest
    q2 = [Xp -> [w <= 5 ? NaN : Xp[1, w] for w in axes(Xp, 2)]]
    D2 = Vector{Vector{Float64}}()
    _MJ._mode_jump_tails!(D2, X, [1], b, q2, (0.02,), 8, 0)
    fin = 6:n; o = fin[sortperm(X[1, fin])][1:20]
    @test D2[1][1] ≈ mean(X[1, o]) - mean(X[1, setdiff(fin, o)])
    # principal components: the leading PC of (x, y) standardized is the shifted direction; both PC tails are found
    D3 = Vector{Vector{Float64}}()
    @test _MJ._mode_jump_tails!(D3, X, [1, 3], nothing, [], (0.02,), 8, 1) == (2, 1)
    @test any(d -> d[1] < -2.5 && d[3] > 4, (D3[1], .-D3[1], D3[2], .-D3[2]))
    @test_throws ArgumentError _MJ._mode_jump_tails!(D3, X, [1], b, [Xp -> zeros(3)], (0.02,), 8, 0)
end

@testset "mode jumps: tail vectors keep exact draws exact (fixed beta)" begin
    rng = Random.Xoshiro(21); n = 20000
    X1 = _ped_exact(rng, 4000)                               # the vectors learned from an independent exact sample
    D = Vector{Vector{Float64}}()
    added, nq = _MJ._mode_jump_tails!(D, X1, 1:_PD, nothing, [Xp -> vec(sum(Xp; dims=1))], (0.02, 0.05), 8, 2)
    @test added == 12 && nq == 3
    X = _ped_exact(rng, n); f0 = _ped_frac(X); m0 = vec(mean(X; dims=2))
    ll = _ped_llk(zeros(n), X); lb = zeros(n); cur = copy(ll)
    buf = (zeros(_PD, n), zeros(n), zeros(n), falses(n)); acc = Ref(0); cnt = Ref(0)
    for _ in 1:20
        _MJ._mode_jump_tries!(X, cur, ll, lb, falses(n), 1.0, D, 1:_PD, nothing, 12, 0.2, 0.7, 1, _ped_llk, _ped_box,
                              buf, acc, cnt, rng)
    end
    @test acc[] / cnt[] > 0.05
    @test abs(_ped_frac(X) - f0) < 4 * sqrt(f0 * (1 - f0) / n)
    @test all(abs.(vec(mean(X; dims=2)) .- m0) .< 0.004)
    @test cur ≈ _ped_llk(zeros(n), X)
end

@testset "rw_moves: a no-op proposal makes no likelihood call" begin
    d = 2; n = 64; box = 5.0
    calls = Ref(0)
    llk = (ll, pos) -> (calls[] += 1; for w in axes(pos, 2); ll[w] = -0.5 * sum(abs2, view(pos, :, w)); end; ll)
    lp = (l, pos) -> (for w in axes(pos, 2); l[w] = all(x -> abs(x) <= box, view(pos, :, w)) ? 0.0 : -Inf; end; l)
    cfg = NUTSConfig{Float64}(; max_treedepth=5, nwarmup=0, bounds=NUTSBounds(fill(-box, d), fill(box, d)))
    counts = Int[]
    for moves in ((), ((xp, pos, rng) -> (xp .= pos; xp),))
        calls[] = 0; rng = Random.Xoshiro(2)
        smc_sample(box .* (2 .* rand(rng, d, n) .- 1), llk, nothing, zeros(n); cfg, rng, moves=:rw, logprior! = lp,
                   nfinal=3, rw_moves=moves)
        push!(counts, calls[])
    end
    @test counts[1] == counts[2]
end

@testset "mode jumps: argument checks" begin
    d = 2; n = 16
    cfg = NUTSConfig{Float64}(; max_treedepth=4, nwarmup=0, bounds=NUTSBounds(fill(-1.0, d), fill(1.0, d)))
    llk = (ll, pos) -> (ll .= 0; ll)
    lp = (l, pos) -> (l .= 0; l)
    x0 = rand(Random.Xoshiro(1), d, n)
    @test_throws ArgumentError smc_sample(x0, llk, nothing, zeros(n); cfg, moves=:rw, logprior! = lp, mode_jumps=true,
                                          mode_jump_rows=[3])
    @test_throws ArgumentError smc_sample(x0, llk, nothing, zeros(n); cfg, moves=:rw, logprior! = lp, mode_jumps=true,
                                          mode_jump_scale=(0.7, 0.2))
    @test_throws ArgumentError smc_sample(x0, llk, nothing, zeros(n); cfg, moves=:rw, logprior! = lp, mode_jumps=true,
                                          mode_jump_k=1)
    @test_throws ArgumentError smc_sample(x0, llk, (g, l, p, m) -> nothing, zeros(n); cfg, mode_jumps=true)  # no logprior!
    @test_throws ArgumentError smc_sample(x0, llk, nothing, zeros(n); cfg, moves=:rw, logprior! = lp, mode_jumps=true,
                                          mode_jump_learn=:pca)
    @test_throws ArgumentError smc_sample(x0, llk, nothing, zeros(n); cfg, moves=:rw, logprior! = lp, mode_jumps=true,
                                          mode_jump_tail_frac=(0.0, 0.05))
end
