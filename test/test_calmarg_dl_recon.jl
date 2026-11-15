# Distance reconstruction under calibration marginalization (`reconstruct_marginalized!(...; curve_loglik)`): synthetic
# response curves with per-curve amplitude factors A_k give exactly quadratic per-curve log-likelihoods in u = 1/D,
#   l_k(D) = -hh A_k^2 / (2 D^2) + z A_k / D,
# and the joint (curve, D) draws must follow p(k, D) ~ D^2 exp(l_k(D)); the curve-averaged single-quadratic rule does not.
using Test, Random, Statistics
using GravitationalWaveBayes
const BNc = GravitationalWaveBayes.BatchedNUTS

@testset "calibration-marginalized d_L reconstruction" begin
    lo, hi, kpow = 0.01, 10.0, 2
    cfg = PEConfig{Float64}(16, 1, 0.01, 1; marginalize_distance=true, distance_lo=lo, distance_hi=hi,
                            distance_power=kpow)
    # GW150914-like: optimal SNR^2 ~ 600 at D ~ 0.45 Gpc; 10% curve-to-curve amplitude scatter
    nc, nw = 200, 6000
    rng = Random.Xoshiro(1)
    A = 1 .+ 0.1 .* randn(rng, nc)
    D0 = 0.45; hh = 600 * D0^2; z = 600 * D0
    lk(k, D) = -hh * A[k]^2 / (2D^2) + z * A[k] / D
    Dg = collect(range(lo, hi; length=200_000))
    # per-curve distance-marginalized values (any k-independent constant is irrelevant)
    vm = [(l = [kpow * log(D) + lk(k, D) for D in Dg]; m = maximum(l); m + log(sum(exp.(l .- m)))) for k in 1:nc]
    curve_loglik = (; ncurves=nc,
        marg=(LK, pos) -> (for w in axes(pos, 2), k in 1:nc; LK[k, w] = vm[k]; end; LK),
        plain=(LK, pos) -> (for w in axes(pos, 2), k in 1:nc; LK[k, w] = lk(k, pos[5, w]); end; LK))
    pos = zeros(15, nw); pos[5, :] .= 1.0
    kk = zeros(Int, nw)
    BNc._calmarg_reconstruct_distance!(pos, cfg, curve_loglik, kk; rng=Random.Xoshiro(2), ngrid=4000)
    # brute-force mixture: p(D) ~ D^2 sum_k exp(l_k(D))
    lw = [(l = [lk(k, D) for k in 1:nc]; m = maximum(l); kpow * log(D) + m + log(sum(exp.(l .- m)))) for D in Dg]
    w = exp.(lw .- maximum(lw)); w ./= sum(w)
    cdf = cumsum(w)
    q(p) = Dg[searchsortedfirst(cdf, p)]
    Ds = pos[5, :]
    for p in (0.05, 0.25, 0.5, 0.75, 0.95)
        # the empirical quantile within ~4 binomial sd of the brute-force one
        @test abs(mean(Ds .<= q(p)) - p) < 4 * sqrt(p * (1 - p) / nw)
    end
    # the curve given D: E[A_k] over draws vs the brute-force joint expectation
    EA = sum(w[i] * (l = [lk(k, Dg[i]) for k in 1:nc]; e = exp.(l .- maximum(l)); sum(e .* A) / sum(e)) for i in 1:500:length(Dg)) /
         sum(w[1:500:end])
    @test abs(mean(A[kk]) - EA) < 4 * std(A) / sqrt(nw)
    # and the old rule (one quadratic through the curve-AVERAGED likelihood at hi, hi/2, hi/4) is measurably off
    mix(D) = (l = [lk(k, D) for k in 1:nc]; m = maximum(l); m + log(mean(exp.(l .- m))))
    l1, l2, l4 = mix(hi), mix(hi / 2), mix(hi / 4)
    d21 = l2 - l1; d42 = l4 - l2; h1 = (2d21 - d42) / 3; z1 = d21 + 3h1 / 2
    lo_ = [kpow * log(D) - h1 * (hi / D)^2 / 2 + z1 * (hi / D) for D in Dg]
    wo = exp.(lo_ .- maximum(lo_)); wo ./= sum(wo)
    @test abs(sum(wo .* Dg) - sum(w .* Dg)) > 5 * sqrt(sum(w .* (Dg .- sum(w .* Dg)) .^ 2)) / sqrt(nw)
end
