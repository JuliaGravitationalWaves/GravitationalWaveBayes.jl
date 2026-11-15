module Diagnostics

export split_rhat, ess_geyer, ess_multichain, ParamDiagnostic, chain_diagnostics, worst_rhat, min_ess, print_diagnostics

using Statistics

# ---------------------------------------------------------------------------------------------
# Convergence/mixing diagnostics for `NUTSCore`'s batched, per-walker-independent chains: split-Rhat,
# per-parameter walker-mean/std comparison and autocorrelation-based ESS.
#
# Motivation: a 300-step, 16-walker NUTS run on the real event GW190412 had not converged (split-Rhat
# 2.5-5.5 across all 11 sampled parameters, two walkers stuck in a distinctly better mode than the
# other 14), which only diagnostics of this kind reveal.
#
# This package's own convention (see `NUTSCore.jl`'s module docstring): each walker in a
# `batched_nuts!` run is an INDEPENDENT chain -- there is no inter-walker coupling (unlike e.g. an
# ensemble/affine-invariant sampler), so treating each walker as its own MCMC chain for
# Gelman-Rubin-style between/within-chain diagnostics is exactly the right model, not an
# approximation.
# ---------------------------------------------------------------------------------------------

"""
    split_rhat(draws::AbstractMatrix{T}) where {T<:Real} -> T

Classic split-`Rhat` (potential scale reduction factor) for one parameter, `draws[t, w]` = the
`t`-th post-warmup draw of walker `w` (size `(ndraws, nwalkers)`).

Method (Gelman & Rubin 1992; Gelman, Carlin, Stern, Dunson, Vehtari & Rubin, *Bayesian Data
Analysis*, 3rd ed., Section 11.4): each walker's chain is split in half, giving `2*nwalkers`
half-length "chains" of length `n = ndraws ÷ 2`. Between-chain variance `B` and within-chain
variance `W` are combined into `var_hat = ((n-1)/n)*W + B/n`, an estimate of the marginal posterior
variance that is unbiased if the chains have converged and an overestimate otherwise (each
individual chain, having not yet explored the full posterior, underestimates `W` relative to the
true variance while `B` picks up any persistent inter-chain offset); `Rhat = sqrt(var_hat / W)` is
therefore >= 1 in expectation, converging to 1 as the chains mix. Splitting each chain in half
(rather than using whole chains directly) additionally makes `Rhat` sensitive to non-stationarity
*within* a single chain (a chain that drifts over its own length looks like two disagreeing
half-chains), which is why this is the version used here and in Stan, not the original 1992
whole-chain-only statistic.

This is the classical statistic, not the rank-normalized/folded `Rhat` of Vehtari et al. (2021,
*Bayesian Analysis*, "Rank-normalization, folding, and localization: An improved Rhat for assessing
convergence of MCMC") -- that improvement (more robust to heavy tails / multimodality in the
per-chain marginal, and adds a separate check for scale, not just location) is a reasonable future
upgrade but out of scope here: the classical statistic is exactly what already caught the real
GW190412 non-convergence (values of 2.5-5.5, an order of magnitude past the ~1.01-1.1 target) this
module exists to make routine, so there is no evidence it is too weak for this codebase's actual
failure modes yet.

Returns `one(T)` for a degenerate all-identical input (`W == B == 0`); returns `T(Inf)` if the
within-chain variance is exactly zero while the between-chain variance is not (chains agree
internally but disagree with each other -- the worst-case non-convergence signature). Throws
`ArgumentError` if `ndraws < 4` (need at least 2 draws per half-chain per walker).
"""
function split_rhat(draws::AbstractMatrix{T}) where {T<:Real}
    ndraws, nw = size(draws)
    ndraws >= 4 || throw(ArgumentError("split_rhat needs at least 4 draws per walker (got $ndraws)"))
    nw >= 1 || throw(ArgumentError("split_rhat needs at least 1 walker"))

    half = ndraws ÷ 2
    nchains = 2 * nw
    splits = Matrix{T}(undef, half, nchains)
    @inbounds for w in 1:nw
        splits[:, 2w - 1] = @view draws[1:half, w]
        splits[:, 2w] = @view draws[(half + 1):(2half), w]
    end

    chain_means = vec(mean(splits, dims=1))
    chain_vars = vec(var(splits, dims=1))
    W = mean(chain_vars)
    B = T(half) * var(chain_means)
    var_hat = ((half - 1) / T(half)) * W + B / T(half)

    if W == zero(T)
        return B == zero(T) ? one(T) : T(Inf)
    end
    return sqrt(var_hat / W)
end

"""
    ess_geyer(x::AbstractVector{T}) where {T<:Real} -> T

Effective sample size of one (single-chain) draw sequence `x`, via Geyer's initial monotone
sequence estimator (Geyer, C.J. 1992, "Practical Markov Chain Monte Carlo", *Statistical Science*
7(4):473-483, Section 3.3; the same construction underlies Stan's/`bayesplot`'s ESS). Consecutive
lag-autocorrelation pairs `rho[2m] + rho[2m+1]` are summed while the running pair-sum stays
positive AND the sequence of pair-sums stays non-increasing (each pair-sum is clipped down to the
previous one if it would otherwise rise, per Geyer's monotonicity correction -- this rejects noisy
upward blips in the empirical autocorrelation at large lag rather than including them, which is
what makes this more robust than a naive single-lag cutoff like `|rho| < threshold`). The
integrated autocorrelation time is `tau = -1 + 2*sum(clipped pair-sums)`, and `ess = n / tau`.

Returns `T(NaN)` for a constant sequence (zero sample variance -- autocorrelation is undefined).
Returns `T(length(x))` (i.e. `tau` floored at `1`) if the very first pair-sum is already
non-positive, matching the standard convention that a sequence with no detectable positive
autocorrelation is treated as (at best) as informative as i.i.d. draws, not MORE informative.
"""
function ess_geyer(x::AbstractVector{T}) where {T<:Real}
    n = length(x)
    n >= 2 || return T(n)
    xm = x .- mean(x)
    c0 = sum(abs2, xm) / n
    c0 == zero(T) && return T(NaN)

    # Autocorrelation at lag k, biased (divide by n, not n-k) estimator -- standard for this
    # purpose (Geyer 1992; also what Stan uses), since the biased estimator is guaranteed
    # non-negative-definite (a valid autocorrelation sequence), which the pairing/monotonicity
    # argument below relies on.
    rho(k) = (sum(@view(xm[1:(n - k)]) .* @view(xm[(k + 1):n])) / n) / c0

    tau = -one(T)
    prev_pair = T(Inf)
    m = 0
    while 2m + 1 <= n - 1
        pair = rho(2m) + rho(2m + 1)
        pair > zero(T) || break
        pair = min(pair, prev_pair)   # Geyer's monotonicity clip
        tau += 2 * pair
        prev_pair = pair
        m += 1
    end
    tau = max(tau, one(T))
    return T(n) / tau
end

"""
    ess_multichain(draws::AbstractMatrix{T}) where {T<:Real} -> T

Effective sample size of `draws` (`(ndraws, nchains)`) pooled over all chains: `n m / tau` with the lag-t
autocorrelation measured about the POOLED mean and variance, `rho(t) = mean_{i,j} (x[i,j] - mu)(x[i+t,j] - mu) / s^2`,
truncated by Geyer's initial monotone sequence as in [`ess_geyer`](@ref). The between-chain variance therefore
counts: walkers that each sit near their own value (particles stuck in one of several modes, too few draws to leave a
resampled duplicate's neighbourhood) count as about one sample each, where a sum of per-chain ESS counts every draw.
It assumes the chains start in the target (the walkers of a stationary run, an SMC population at beta = 1); a drifting
start shows up as extra correlation, i.e. a conservative ESS. Unlike the per-chain-mean form (Vehtari et al. 2021,
eq. 10) it is unbiased for a few draws per chain: centring each 10-draw chain on its own mean makes the lag-1
autocovariance negative (-s^2 / n) and puts i.i.d. draws at ~1.6 n m.
"""
function ess_multichain(draws::AbstractMatrix{T}) where {T<:Real}
    n, m = size(draws)
    n >= 2 || return T(n * m)
    mu = mean(draws)
    s2 = sum(abs2, draws .- mu) / (n * m)
    s2 == zero(T) && return T(NaN)
    function rho(k)
        s = zero(T)
        @inbounds for j in 1:m, i in 1:(n - k)
            s += (draws[i, j] - mu) * (draws[i + k, j] - mu)
        end
        return s / ((n - k) * m) / s2
    end
    tau = -one(T)
    prev_pair = T(Inf)
    k = 0
    while 2k + 1 <= n - 1
        pair = rho(2k) + rho(2k + 1)
        pair > zero(T) || break
        pair = min(pair, prev_pair)
        tau += 2 * pair
        prev_pair = pair
        k += 1
    end
    # (an odd n leaves lag n - 1 unpaired; it is dropped, as in `ess_geyer`)
    return T(n * m) / max(tau, one(T))
end

"""
    ParamDiagnostic{T}

Per-parameter convergence/mixing summary produced by [`chain_diagnostics`](@ref).

- `name`: parameter label (from the caller's `names` argument, or `"param\$i"`).
- `rhat`: split-Rhat ([`split_rhat`](@ref)); target `<~ 1.01-1.1` for a trustworthy run.
- `ess`: effective sample size pooled over all walkers ([`ess_multichain`](@ref): the between-walker variance counts,
  so walkers stuck near their own values -- e.g. a few draws from a resampled SMC population -- are not counted as
  independent samples, which a sum of per-walker [`ess_geyer`](@ref) would do).
- `mean`/`std`: pooled posterior mean/std across all post-warmup draws of all walkers.
- `walker_mean_range`/`walker_std_range`: `(min, max)` of each walker's OWN mean/std -- a large
  spread here (as opposed to `rhat` alone) is a direct, easy-to-read symptom of walkers stuck in
  different modes (this is exactly what caught the GW190412 two-good/fourteen-stuck split).
"""
struct ParamDiagnostic{T}
    name::String
    rhat::T
    ess::T
    mean::T
    std::T
    walker_mean_range::Tuple{T,T}
    walker_std_range::Tuple{T,T}
end

"""
    chain_diagnostics(chain::AbstractArray{T,3}; warmup::Int=0, names=nothing) -> Vector{ParamDiagnostic{T}}

Compute per-parameter [`ParamDiagnostic`](@ref)s from a recorded NUTS chain, `chain[p, w, t]` =
parameter `p`'s value for walker `w` at recorded step `t` -- exactly the layout `NUTSCore.nuts_sample!`
returns as its `chain` field when called with `record=true`, and the layout the real GW190412
result file (`gw190412_xhm_result.jls`) stores.

`warmup`: number of leading steps (dual-averaging adaptation window) to discard before computing
diagnostics -- pass `cfg.nwarmup`/`nuts_cfg.nwarmup` from the run that produced `chain`. Must leave
at least 4 post-warmup draws (see [`split_rhat`](@ref)).

`names`: optional `Vector{String}`/`Vector{Symbol}` of length `size(chain, 1)` for readable output
(e.g. `["mc", "q", "chi1", ...]` for this codebase's 11/15-row sampled convention documented in
`BatchedNUTS.jl`); defaults to `"param1"`, `"param2"`, ... .
"""
function chain_diagnostics(chain::AbstractArray{T,3}; warmup::Int=0,
                            names::Union{Nothing,AbstractVector}=nothing) where {T<:Real}
    nparams, nwalkers, nsteps = size(chain)
    warmup < nsteps || throw(ArgumentError("warmup ($warmup) must leave at least 1 post-warmup step (nsteps=$nsteps)"))
    post = (warmup + 1):nsteps
    npost = length(post)
    npost >= 4 || throw(ArgumentError("need at least 4 post-warmup draws, got $npost"))

    if names !== nothing
        length(names) == nparams || throw(ArgumentError("names has length $(length(names)), expected $nparams"))
    end

    diags = Vector{ParamDiagnostic{T}}(undef, nparams)
    for p in 1:nparams
        draws = permutedims(@view(chain[p, :, post]))  # (npost, nwalkers)
        rhat = split_rhat(draws)
        ess = ess_multichain(draws)
        walker_means = vec(mean(draws, dims=1))
        walker_stds = vec(std(draws, dims=1))
        nm = names === nothing ? "param$p" : string(names[p])
        diags[p] = ParamDiagnostic{T}(nm, rhat, ess, mean(draws), std(draws),
                                       (minimum(walker_means), maximum(walker_means)),
                                       (minimum(walker_stds), maximum(walker_stds)))
    end
    return diags
end

"""
    worst_rhat(diags::AbstractVector{<:ParamDiagnostic}) -> ParamDiagnostic

Returns the `ParamDiagnostic` with the largest `rhat` (the single worst-converged parameter) --
the natural one-line "is this run trustworthy" check: e.g.
`worst_rhat(diags).rhat < 1.1 || @warn "not converged"`.
"""
worst_rhat(diags::AbstractVector{<:ParamDiagnostic}) = diags[argmax(d.rhat for d in diags)]

"""
    min_ess(diags::AbstractVector{<:ParamDiagnostic}) -> ParamDiagnostic

Returns the `ParamDiagnostic` with the smallest total `ess` (the parameter with the fewest
effectively-independent draws, i.e. the slowest-mixing direction).
"""
min_ess(diags::AbstractVector{<:ParamDiagnostic}) = diags[argmin(d.ess for d in diags)]

"""
    print_diagnostics([io::IO=stdout,] diags::AbstractVector{<:ParamDiagnostic})

Human-readable table, one line per parameter, of `Rhat`/total `ESS`/pooled mean & std/walker
mean & std ranges, for any recorded chain.
"""
function print_diagnostics(io::IO, diags::AbstractVector{<:ParamDiagnostic})
    for d in diags
        println(io, rpad(d.name, 14),
                " Rhat=", round(d.rhat, digits=4),
                "  ESS~", round(d.ess, digits=1),
                "  mean=", round(d.mean, digits=5),
                " std=", round(d.std, digits=5))
        println(io, "    walker means range: [", round(d.walker_mean_range[1], digits=5), ", ",
                round(d.walker_mean_range[2], digits=5), "]",
                "   walker stds range: [", round(d.walker_std_range[1], digits=5), ", ",
                round(d.walker_std_range[2], digits=5), "]")
    end
end
print_diagnostics(diags::AbstractVector{<:ParamDiagnostic}) = print_diagnostics(stdout, diags)

end
