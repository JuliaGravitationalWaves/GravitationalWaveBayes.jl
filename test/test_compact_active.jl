using Test, GravitationalWaveBayes, Random

# `compact_active` (2026-10-15): NUTS gradient calls with few active walkers evaluated on a narrower closure give the
# same values / gradients on the active columns; NUTS draws with it are identical to the full-width draws.
@testset "compact_active" begin
    BN = GravitationalWaveBayes.BatchedNUTS
    widths_seen = Int[]
    mk(W) = function (g, lp, X, mask; value_only = false)
        size(X, 2) == W || error("built for $W walkers, got $(size(X, 2))")
        push!(widths_seen, W)
        for w in axes(X, 2)
            lp[w] = -sum(abs2, view(X, :, w)) / 2 - X[1, w]^4 / 10
            g[:, w] .= -view(X, :, w); g[1, w] -= 0.4 * X[1, w]^3
        end
        return nothing
    end
    n = 64; d = 3
    f, st = BN.compact_active(mk(n), n; narrow = [(16, mk(16)), (32, mk(32))])
    X = randn(Xoshiro(1), d, n)
    for k in (0, 1, 5, 16, 17, 32, 33, 64)
        mask = falses(n); mask[randperm(Xoshiro(k), n)[1:k]] .= true
        g1 = fill(NaN, d, n); l1 = fill(NaN, n); g2 = zeros(d, n); l2 = zeros(n)
        empty!(widths_seen)
        f(g1, l1, X, mask)
        seen = copy(widths_seen)
        mk(n)(g2, l2, X, trues(n))
        @test all(l1[mask] .== l2[mask]) && all(g1[:, mask] .== g2[:, mask])
        @test seen == [k == 0 ? n : k <= 16 ? 16 : k <= 32 ? 32 : n]
    end
    @test st.active[] == 0 + 1 + 5 + 16 + 17 + 32 + 33 + 64
    # NUTS: same draws with and without compaction (the closure is deterministic)
    cfg = GravitationalWaveBayes.NUTSConfig{Float64}(; max_treedepth = 6, nwarmup = 0)
    run(lpg) = (P = copy(X); lp = zeros(n); lpg(zeros(d, n), lp, P, trues(n));
                s = GravitationalWaveBayes.NUTSCore.nuts_draw!(P, lp, fill(0.3, n), lpg, cfg, Xoshiro(5)); (P, s.n_leapfrog))
    Pa, na = run(mk(n)); Pb, nb = run(f)
    @test Pa == Pb && na == nb
end
