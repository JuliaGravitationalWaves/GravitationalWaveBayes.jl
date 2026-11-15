# ---------------------------------------------------------------------------------------------
# Adaptive tempered Sequential Monte Carlo with batched NUTS mutations (included inside NUTSCore).
#
# The population alternative to parallel tempering. Both anneal along the same path
#   pi_beta(theta) ∝ prior(theta) * L(theta)^beta,  beta: 0 -> 1,
# but where PT keeps one chain population per rung FOR THE WHOLE RUN (and only the beta = 1 rung
# produces samples), SMC moves ONE population of N particles up the ladder once:
#
#   1. draw N particles from the prior (beta = 0);
#   2. pick the next beta so the conditional ESS of the incremental weights L^(beta' - beta) is
#      `target_cess * N` (Zhou, Johansen & Aston 2016) -- the schedule adapts itself, with the same
#      thermodynamic-length geometry as the PT communication barrier (stages ~ sqrt(d/2) log(1/beta_0)
#      for a Gaussian);
#   3. accumulate log Z (the evidence comes for free), resample (systematic);
#   4. mutate every particle with NUTS (or, `moves = :rw`, a gradient-free random-walk Metropolis) at
#      the new beta, using a metric estimated from the weighted particle cloud (population adaptation
#      is exact: the kernel is fixed before the stage's draws, cf. Buchholz, Chopin & Jacob 2021),
#      until the particles have decorrelated;
#   5. at beta = 1, run `nfinal` further draws and keep them ALL (every particle is a chain started
#      in equilibrium, so there is no burn-in to discard).
#
# Everything is batched across particles exactly as `nuts_draw!` batches walkers, so the GPU/quorum
# machinery applies unchanged. `logp_and_grad!` must be the TEMPERED log-density
# `betas[w] * logL + logprior` reading the caller's `betas` vector by reference (e.g.
# `BatchedNUTS.gw_logp_and_grad(...; beta=betas)`), and `loglik!(ll, positions)` the untempered
# log-likelihood. `positions0` must be exact draws from the SAME prior the tempered density uses.
# ---------------------------------------------------------------------------------------------

function _logsumexp(x::AbstractVector{T}) where {T}
    m = maximum(x)
    isfinite(m) || return m
    s = zero(T)
    @inbounds for v in x
        s += exp(v - m)
    end
    return m + log(s)
end

# Normalized weights and ESS/N of log-weights `logw`.
function _normalized_weights(logw::AbstractVector{T}) where {T}
    W = exp.(logw .- _logsumexp(logw))
    return W, one(T) / (sum(abs2, W) * length(W))
end

# Largest beta' in (beta, 1] with ESS(exp((beta' - beta) * ll)) / N >= rho, by bisection.
function _next_beta(ll::AbstractVector{T}, beta::T, rho::T) where {T}
    ess(db) = _normalized_weights(db .* ll)[2]
    ess(one(T) - beta) >= rho && return one(T)
    lo, hi = zero(T), one(T) - beta
    for _ in 1:60
        mid = (lo + hi) / 2
        ess(mid) >= rho ? (lo = mid) : (hi = mid)
    end
    return beta + max(lo, eps(T) * 16)
end

# Systematic resampling: N indices drawn from the normalized weights W.
function _systematic_resample(rng::AbstractRNG, W::AbstractVector{T}) where {T}
    n = length(W)
    idx = Vector{Int}(undef, n)
    u = rand(rng, T) / n
    c = W[1]
    j = 1
    @inbounds for i in 1:n
        while u > c && j < n
            j += 1
            c += W[j]
        end
        idx[i] = j
        u += one(T) / n
    end
    return idx
end

# Systematic resampling of `m` indices from normalized weights W over a pool of any size.
function _systematic_resample_n(rng::AbstractRNG, W::AbstractVector{T}, m::Int) where {T}
    idx = Vector{Int}(undef, m)
    u = rand(rng, T) / m
    c = W[1]
    j = 1
    @inbounds for i in 1:m
        while u > c && j < length(W)
            j += 1
            c += W[j]
        end
        idx[i] = j
        u += one(T) / m
    end
    return idx
end

# Weighted covariance of the particle cloud -> metric for the next mutation. `mass_blocks` masks
# cross-block entries (see `NUTSConfig.mass_blocks`); `dense=false` keeps the diagonal only. Falls
# back to the diagonal, then to `fallback`, if the estimate is not positive definite.
function _smc_metric(positions::AbstractMatrix{T}, W::AbstractVector{T}, dense::Bool,
                     mass_blocks, fallback) where {T}
    nparams = size(positions, 1)
    mu = positions * W
    X = positions .- mu
    S = (X .* reshape(W, 1, :)) * X'
    S ./= max(one(T) - sum(abs2, W), eps(T))            # unbiased weighted covariance
    d = diag_of(S)
    ok = all(v -> isfinite(v) && v > 0, d)
    ok || return fallback
    if dense
        A = Matrix{T}(Symmetric(_block_mask(S, mass_blocks)))
        for i in 1:nparams
            A[i, i] *= one(T) + T(1e-6)
        end
        isposdef(Symmetric(A)) && return DenseMetric(A)
    end
    return DenseMetric(Matrix{T}(Diagonal(d)))
end

diag_of(S::AbstractMatrix) = [S[i, i] for i in 1:size(S, 1)]

# `hessian_metric = k`: the stage metric with the local curvature folded in. Hbar = the negative Hessian of the tempered
# log target averaged over up to k distinct top-logL particles (central differences of `lpg!`, steps 1e-3 population
# sds, all columns in ONE batched call). In coordinates whitened by the population metric Sigma = L L', A = L' Hbar L =
# V diag(lambda) V'; Minv = L V diag(1 / max(lambda, 1)) V' L': directions the target pins far more tightly than the
# population spread (lambda >> 1) get the local width, all others keep Sigma exactly (a Gaussian, whose tempered
# population already has the target's shape, gets lambda ~ 1: Sigma unchanged). For SEOBNRv5PHM the inspiral
# phase-coherence direction (mc, chi1z, q, phase) has lambda ~ 1e4 (next 1e2) and caps every step without it. A fixed
# metric per stage: moves stay exact. Returns (metric, gradient columns used).
function _hessian_metric(metric::DenseMetric{T}, positions::AbstractMatrix{T}, ll::AbstractVector{T}, lpg!, k::Int,
                         dense::Bool) where {T}
    nparams, n = size(positions)
    k = min(k, n ÷ (2nparams))
    k >= 1 || return metric, 0
    order = sortperm(ll; rev=true)
    tops = Int[]
    for w in order                                  # distinct positions (resampling duplicates particles)
        isfinite(ll[w]) || break
        any(j -> view(positions, :, j) == view(positions, :, w), tops) || push!(tops, w)
        length(tops) == k && break
    end
    isempty(tops) && return metric, 0
    h = T(1e-3) .* sqrt.(diag_of(metric.Minv))
    P = repeat(positions[:, tops[1]], 1, n)
    for (j, w) in enumerate(tops), r in 1:nparams
        c = (j - 1) * 2nparams + 2r - 1
        @views P[:, c] .= positions[:, w]; P[r, c] += h[r]
        @views P[:, c+1] .= positions[:, w]; P[r, c+1] -= h[r]
    end
    G = similar(P); lp = Vector{T}(undef, n)
    lpg!(G, lp, P, trues(n))
    H = zeros(T, nparams, nparams); m = 0
    for j in eachindex(tops)
        cols = (j - 1) * 2nparams .+ (1:2nparams)
        (all(isfinite, view(lp, cols)) && all(isfinite, view(G, :, cols))) || continue
        for r in 1:nparams
            c = (j - 1) * 2nparams + 2r - 1
            @views H[:, r] .-= (G[:, c] .- G[:, c+1]) ./ (2h[r])
        end
        m += 1
    end
    m == 0 && return metric, n
    H ./= m
    L = metric.L
    E = eigen(Symmetric(L' * ((H .+ H') ./ 2) * L))
    B = L * E.vectors
    A = B * Diagonal(one(T) ./ max.(E.values, one(T))) * B'
    A = Matrix{T}(Symmetric((A .+ A') ./ 2))
    dense || (A = Matrix{T}(Diagonal(diag_of(A))))
    isposdef(Symmetric(A)) || return metric, n
    return DenseMetric(A), n
end

# `metric_core` (a fraction c): the weights the metric is estimated with -- W restricted to the ceil(c n) particles with
# the highest log-likelihood, renormalized. While the population is still handing over to a narrow signal mode, a few
# prior-like stragglers set the population covariance (e.g. a time row ~100x wider than the mode), which forces tiny
# whitened steps on every particle; any fixed positive-definite metric keeps the moves exact. `nothing`: W itself.
function _metric_weights(W::AbstractVector{T}, ll::AbstractVector{T}, core) where {T}
    core === nothing && return W
    n = length(W)
    keep = partialsortperm(ll, 1:clamp(ceil(Int, core * n), 2, n); rev=true)
    Wm = zeros(T, n)
    Wm[keep] .= W[keep]
    s = sum(Wm)
    return s > 0 ? Wm ./ s : W
end

# (min over rows of RMS_particles(after - before) / population sd, argmin row). A fully decorrelated
# move of a stationary chain has RMS sqrt(2) sd; 1 sd means a lag-one correlation of about 0.5.
function _slowest_row_move(before::AbstractMatrix{T}, after::AbstractMatrix{T}, metric::DenseMetric{T}) where {T}
    nparams, n = size(before)
    best, arg = T(Inf), 0
    for r in 1:nparams
        m = zero(T)
        for w in 1:n
            m += abs2(after[r, w] - before[r, w])
        end
        v = sqrt(m / n) / sqrt(metric.Minv[r, r])
        v < best && ((best, arg) = (v, r))
    end
    return best, arg
end

# Robust variant: (min over rows of sqrt(2) median_w |after - before| / median_w |before_w - before_pi(w)|, argmin row),
# pi a fixed random pairing of the particles. The denominator is the typical displacement of a FULLY decorrelated move
# for any row shape (flat, skewed, bimodal), so the statistic is sqrt(2) for independent draws and 1 at lag-one
# correlation 0.5 for a Gaussian -- the RMS rule's calibration -- but neither a few particles jumping between modes
# (which dominate an RMS) nor a minor mode (which inflates the sd) decides the stage length: both made the RMS rule
# stop after 5 or after 20 draws on the same problem. (A MAD denominator would saturate below 1 for flat rows.)
function _slowest_row_move_median(before::AbstractMatrix{T}, after::AbstractMatrix{T}, metric::DenseMetric{T}) where {T}
    nparams, n = size(before)
    perm = randperm(Xoshiro(0x5eed), n)          # fixed pairing, independent of the sampler's RNG stream
    best, arg = T(Inf), 0
    d = Vector{T}(undef, n)
    for r in 1:nparams
        for w in 1:n
            d[w] = abs(after[r, w] - before[r, w])
        end
        md = median!(d)
        for w in 1:n
            d[w] = abs(before[r, w] - before[r, perm[w]])
        end
        ref = median!(d)
        # More than half the pairs identical (heavy duplication): the Gaussian value from the population sd.
        ref > 0 || (ref = T(0.9539) * sqrt(metric.Minv[r, r]))
        v = sqrt(T(2)) * md / ref
        v < best && ((best, arg) = (v, r))
    end
    return best, arg
end

_slowest_move(stat::Symbol, before, after, metric) =
    stat === :median ? _slowest_row_move_median(before, after, metric) :
    stat === :rms ? _slowest_row_move(before, after, metric) :
    throw(ArgumentError("move_stat must be :median or :rms, got :$stat"))

# The geometric path from a normalized reference q0 instead of the prior: pi_beta ∝ q0^(1-beta) (prior L)^beta.
# The weights use the log-ratio r = logL + logprior - logq0 in place of logL (`llk!`), and the mutation target adds
# (1-beta)(logq0 - logprior) to the caller's `beta logL + logprior` (`lpg!`, beta read from `beta_ref`).
function _smc_reference_path(reference, base_lpg!, base_ll!, beta_ref::Ref{T}, nparams, n) where {T}
    lp_buf = Vector{T}(undef, n); lq_buf = Vector{T}(undef, n)
    gp_buf = Matrix{T}(undef, nparams, n); gq_buf = Matrix{T}(undef, nparams, n)
    lpg! = function (grad, logp, pos, mask)
        # The reference terms do not depend on the target: computed on another task while `base_lpg!` runs (a
        # device likelihood leaves the host idle; with thousands of particles they cost ~0.1 s per call).
        ref_task = Threads.@spawn begin
            reference.logprior_and_grad!(lp_buf, gp_buf, pos)
            reference.logq_and_grad!(lq_buf, gq_buf, pos)
        end
        base_lpg!(grad, logp, pos, mask)
        fetch(ref_task)
        c = one(T) - beta_ref[]
        logp .+= c .* (lq_buf .- lp_buf)
        grad .+= c .* (gq_buf .- gp_buf)
        return nothing
    end
    llk! = function (ll, pos)
        base_ll!(ll, pos)
        reference.logprior_and_grad!(lp_buf, gp_buf, pos)
        reference.logq_and_grad!(lq_buf, gq_buf, pos)
        ll .+= lp_buf .- lq_buf
        return ll
    end
    return lpg!, llk!
end

# `de_tries` differential-evolution Metropolis steps (see `smc_sample`'s `de_tries`) on all particles; `logp` holds the
# tempered log density beta * loglik + logprior of the current states and stays exact. Proposals outside the prior
# support are not evaluated (their column is the current state in the one batched `llk!` call, a fixed batch shape).
function _de_steps!(positions, logp, start, metric, betas, xp, z, llp, lbp, llk!, logprior!, bounds, ntries, de_jump,
                    acc, cnt, rng)
    T = eltype(positions)
    nparams, n = size(positions)
    g0 = T(2.38) / sqrt(T(2 * nparams))
    for _ in 1:ntries
        randn!(rng, z)
        mul!(xp, LowerTriangular(metric.L), z)
        @inbounds for w in 1:n
            a = rand(rng, 1:n); b = rand(rng, 1:n-1); b += b >= a
            g = rand(rng, T) < de_jump ? one(T) : g0 * (T(0.8) + T(0.4) * rand(rng, T))
            for r in 1:nparams
                dd = start[r, a] - start[r, b]
                per = bounds !== nothing && bounds.periodic[r]
                per && (dd -= (bounds.hi[r] - bounds.lo[r]) * round(dd / (bounds.hi[r] - bounds.lo[r])))
                x = positions[r, w] + g * dd + T(0.01) * g * xp[r, w]
                per && (x = bounds.lo[r] + mod(x - bounds.lo[r], bounds.hi[r] - bounds.lo[r]))
                xp[r, w] = x
            end
        end
        logprior!(lbp, xp)
        @inbounds for w in 1:n
            isfinite(lbp[w]) || (@views xp[:, w] .= positions[:, w])
        end
        llk!(llp, xp)
        @inbounds for w in 1:n
            (isfinite(lbp[w]) && isfinite(llp[w])) || continue
            lpn = betas[w] * llp[w] + lbp[w]
            if !isfinite(logp[w]) || log(rand(rng, T)) < lpn - logp[w]
                @views positions[:, w] .= xp[:, w]
                logp[w] = lpn
                acc[] += 1
            end
        end
        cnt[] += n
    end
    return nothing
end

# The population on the jump `rows` as standardized features (one column per particle): a periodic row of `bounds` as
# (cos, sin) of its period, every feature centred and scaled to unit sd.
function _mode_jump_features(X::AbstractMatrix{T}, rows, bounds) where {T}
    n = size(X, 2)
    per(r) = bounds !== nothing && bounds.periodic[r]
    nf = sum(r -> per(r) ? 2 : 1, rows)
    F = Matrix{T}(undef, nf, n)
    i = 0
    for r in rows
        if per(r)
            P = bounds.hi[r] - bounds.lo[r]
            @inbounds for w in 1:n
                F[i+1, w], F[i+2, w] = cos(2pi * X[r, w] / P), sin(2pi * X[r, w] / P)
            end
            i += 2
        else
            @views F[i+1, :] .= X[r, :]
            i += 1
        end
    end
    for f in 1:nf
        m = sum(view(F, f, :)) / n
        s = sqrt(sum(x -> abs2(x - m), view(F, f, :)) / (n - 1))
        @views F[f, :] .= (F[f, :] .- m) ./ (s + T(1e-12))
    end
    return F
end

# mean(X[:, a]) - mean(X[:, b]) on the jump `rows` (circular means, shortest signed difference, on periodic rows of
# `bounds`), zero off `rows`
function _mode_jump_diff(X::AbstractMatrix{T}, a, b, rows, bounds) where {T}
    d = zeros(T, size(X, 1))
    for r in rows
        if bounds !== nothing && bounds.periodic[r]
            P = bounds.hi[r] - bounds.lo[r]
            ang(ix) = (c = zero(T); s = zero(T); @inbounds(for w in ix; c += cos(2pi * X[r, w] / P); s += sin(2pi * X[r, w] / P); end);
                       atan(s, c) * P / 2pi)
            d[r] = mod(ang(a) - ang(b) + P / 2, P) - P / 2
        else
            sa = zero(T); sb = zero(T)
            @inbounds for w in a; sa += X[r, w]; end
            @inbounds for w in b; sb += X[r, w]; end
            d[r] = sa / length(a) - sb / length(b)
        end
    end
    return d
end

# Tail-targeted jump vectors (`mode_jump_learn = :tails`). A minor mode that hangs off the bulk along a
# physical direction (the GW150914 XPHM low-Mc / anti-aligned corner: a TAIL of the chi_eff / Mc distribution, not a
# k-means cell) is reached by the vector from the rest of the population to its tail along that quantity: for every
# tail quantity q (a function `q(X) -> Vector` of the population matrix, one value per particle) and every fraction p
# in `fracs`, the lower and the upper p-tail of q's finite values (>= `min_count` particles, rest >= `min_count`) give
# mean(tail) - mean(rest) on the jump `rows` (`_mode_jump_diff`), appended to `deltas`. `npcs` adds the leading
# principal components of the standardized jump-row features (`_mode_jump_features`) as quantities. On the s62 stage
# dumps the lowest-2% chi_eff vector had cos 0.91-0.97 with the release corner-minus-bulk oracle from beta 0.07 on.
# Returns (vectors added, quantities used).
function _mode_jump_tails!(deltas::Vector{Vector{T}}, X::AbstractMatrix{T}, rows, bounds, quantities, fracs,
                           min_count::Int, npcs::Int) where {T}
    n = size(X, 2)
    vals = Vector{Vector{T}}()
    for q in quantities
        v = q(X)
        length(v) == n || throw(ArgumentError("mode-jump tail quantity returned $(length(v)) values for $n particles"))
        push!(vals, Vector{T}(v))
    end
    if npcs > 0
        F = _mode_jump_features(X, rows, bounds)
        E = eigen(Symmetric(F * F' ./ (n - 1)))
        for j in 1:min(npcs, size(F, 1))
            push!(vals, vec(E.vectors[:, end+1-j]' * F))
        end
    end
    added = 0
    for v in vals
        fin = findall(isfinite, v)
        nf = length(fin)
        o = fin[sortperm(v[fin])]
        for p in fracs
            m = round(Int, p * nf)
            (m >= min_count && nf - m >= min_count) || continue
            lo = o[1:m]; hi = o[nf-m+1:nf]
            push!(deltas, _mode_jump_diff(X, lo, setdiff(fin, lo), rows, bounds))
            push!(deltas, _mode_jump_diff(X, hi, setdiff(fin, hi), rows, bounds))
            added += 2
        end
    end
    return added, length(vals)
end

# Mode jumps (`smc_sample`'s `mode_jumps`; a mode-jumping proposal, Tjelmeland & Hegstad 2001, with the jump
# vectors learned from the population like differential evolution's mode hops). After a stage, k-means (k-means++, 20
# Lloyd iterations) clusters the population on the jump `rows`, standardized, periodic rows embedded as (cos, sin) of
# their period; the mean difference (circular on periodic rows, zero off `rows`) of every pair of cells holding
# >= max(10, min_cell n) particles is appended to `deltas`. Vectors are never dropped: a minor mode that is a populated
# cell early on (the GW150914 XPHM low-Mc / anti-aligned corner holds 5-14% of the particles at beta ~0.1) keeps its
# vector after tempering has drained it. Returns (vectors added, populated cells).
function _mode_jump_learn!(deltas::Vector{Vector{T}}, X::AbstractMatrix{T}, rows, bounds, k::Int, min_cell,
                           rng::AbstractRNG) where {T}
    nparams, n = size(X)
    per(r) = bounds !== nothing && bounds.periodic[r]
    F = _mode_jump_features(X, rows, bounds)
    nf = size(F, 1)
    k = min(k, n)
    d2(w, C, c) = (s = zero(T); @inbounds(for f in 1:nf; s += abs2(F[f, w] - C[f, c]); end); s)
    C = Matrix{T}(undef, nf, k)
    C[:, 1] .= view(F, :, rand(rng, 1:n))
    dmin = fill(T(Inf), n)
    for c in 2:k                                  # k-means++ seeding
        for w in 1:n
            dmin[w] = min(dmin[w], d2(w, C, c - 1))
        end
        u = rand(rng, T) * sum(dmin); acc = zero(T); j = n
        for w in 1:n
            acc += dmin[w]
            acc >= u && (j = w; break)
        end
        C[:, c] .= view(F, :, j)
    end
    a = ones(Int, n); cnt = zeros(Int, k)
    for _ in 1:20                                 # Lloyd
        for w in 1:n
            best = T(Inf)
            for c in 1:k
                v = d2(w, C, c)
                v < best && (best = v; a[w] = c)
            end
        end
        fill!(cnt, 0)
        S = zeros(T, nf, k)
        for w in 1:n
            cnt[a[w]] += 1
            @views S[:, a[w]] .+= F[:, w]
        end
        for c in 1:k
            cnt[c] > 0 && (@views C[:, c] .= S[:, c] ./ cnt[c])
        end
    end
    big = [c for c in 1:k if cnt[c] >= max(10, min_cell * n)]
    # per-cell means on the original rows (circular means on periodic rows)
    mu = zeros(T, nparams, k); ms = zeros(T, nparams, k)
    for w in 1:n, r in rows
        if per(r)
            P = bounds.hi[r] - bounds.lo[r]
            mu[r, a[w]] += cos(2pi * X[r, w] / P); ms[r, a[w]] += sin(2pi * X[r, w] / P)
        else
            mu[r, a[w]] += X[r, w]
        end
    end
    added = 0
    for i in big, j in big
        i < j || continue
        d = zeros(T, nparams)
        for r in rows
            if per(r)
                P = bounds.hi[r] - bounds.lo[r]
                ci, cj = atan(ms[r, i], mu[r, i]) * P / 2pi, atan(ms[r, j], mu[r, j]) * P / 2pi
                d[r] = mod(ci - cj + P / 2, P) - P / 2
            else
                d[r] = mu[r, i] / cnt[i] - mu[r, j] / cnt[j]
            end
        end
        push!(deltas, d)
        added += 1
    end
    return added, length(big)
end

# `ntries` mode-jump MH steps for every particle: x' = x + s delta on the jump rows (periodic rows wrapped), delta uniform
# over `deltas` (fixed during the stage), |s| ~ U(smin, smax) with a random sign -- symmetric with a unit Jacobian (the
# set is used with both signs), so plain MH on base + beta llk. `cur`, `ll`, `lb` hold the current states' target
# density, log-likelihood and base density and are updated on acceptance (`moved[w]` set). The tries are evaluated
# speculatively in rounds of up to `batch` per particle from its current state, column (t - 1) n + w = its t-th try of
# the round, in one `llk!` call of `batch n` columns; they are applied in order, and after an acceptance the particle's
# remaining tries of the round are discarded (their proposals assumed the old state; the next round redraws them from
# the new one). That is the same chain as `ntries` sequential steps; `batch = 1` IS the sequential evaluation. Columns
# without a try (or outside the base density's support) hold the current state; a round with nothing to evaluate makes
# no `llk!` call. Returns the number of evaluated columns.
function _mode_jump_tries!(positions::AbstractMatrix{T}, cur, ll, lb, moved, beta, deltas, rows, bounds, ntries::Int,
                           smin, smax, batch::Int, llk!, base!, buf, acc, cnt, rng::AbstractRNG) where {T}
    isempty(deltas) && return 0
    n = size(positions, 2)
    Xb, lbb, llb, live = buf
    left = fill(ntries, n)
    ncols = 0
    while any(>(0), left)
        @inbounds for w in 1:n, t in 1:batch
            j = (t - 1) * n + w
            @views Xb[:, j] .= positions[:, w]
            live[j] = t <= left[w]
            live[j] || continue
            d = deltas[rand(rng, 1:length(deltas))]
            s = (smin + (smax - smin) * rand(rng, T)) * (rand(rng, Bool) ? one(T) : -one(T))
            for r in rows
                x = positions[r, w] + s * d[r]
                if bounds !== nothing && bounds.periodic[r]
                    x = bounds.lo[r] + mod(x - bounds.lo[r], bounds.hi[r] - bounds.lo[r])
                end
                Xb[r, j] = x
            end
        end
        base!(lbb, Xb)
        any_live = false
        @inbounds for j in eachindex(live)
            if live[j] && !isfinite(lbb[j])
                live[j] = false                   # outside the support: a rejected try, not evaluated
                @views Xb[:, j] .= positions[:, mod1(j, n)]
            end
            any_live |= live[j]
        end
        if any_live
            llk!(llb, Xb)
            ncols += length(llb)
        end
        @inbounds for w in 1:n
            for t in 1:min(batch, left[w])
                j = (t - 1) * n + w
                left[w] -= 1; cnt[] += 1
                (live[j] && isfinite(llb[j])) || continue
                lpn = lbb[j] + beta * llb[j]
                if !isfinite(cur[w]) || log(rand(rng, T)) < lpn - cur[w]
                    @views positions[:, w] .= Xb[:, j]
                    cur[w] = lpn; ll[w] = llb[j]; lb[w] = lbb[j]; moved[w] = true
                    acc[] += 1
                    break
                end
            end
        end
    end
    return ncols
end

# Does any in-support column of the proposal `xp` differ from the current state? (No: an `rw_moves` proposal is a no-op
# and its likelihood call is skipped.)
function _any_moved(xp::AbstractMatrix, positions::AbstractMatrix, lb::AbstractVector)
    @inbounds for w in axes(xp, 2)
        isfinite(lb[w]) && any(r -> xp[r, w] != positions[r, w], axes(xp, 1)) && return true
    end
    return false
end

# The online proposal for a stage at beta from the previous stage's mutation states (`pool`, sampled at beta - dbeta):
# thinned at random to `pool_max` (diverse, unlike the freshly resampled particles), with normalized importance
# weights exp(dbeta * ll) toward the new tempered target (uniform if a log-likelihood is unknown), passed as
# `online_proposal(states, weights)`. Returns the new proposal or `nothing` (keep the previous one).
function _online_build(online_proposal, pool, pool_ll, dbeta, pool_max, rng)
    T = eltype(pool_ll)
    sel = length(pool) <= pool_max ? collect(1:length(pool)) : randperm(rng, length(pool))[1:pool_max]
    lw = [dbeta * pool_ll[i] for i in sel]
    if any(isnan, lw) || !any(isfinite, lw)
        W = ones(T, length(sel))
    else
        m = maximum(filter(isfinite, lw))
        W = [isfinite(x) ? exp(x - m) : zero(T) for x in lw]
    end
    W ./= sum(W)
    return online_proposal(reduce(hcat, pool[sel]), W)
end

# `latent_nuts` (NeuTra, Hoffman et al. 2019): the stage target pulled back through the stage's flow proposal x = T(z)
# (`proposal.from_latent(Z) -> (X, ldj, pull)`, `pull(GX)` = J' GX + d ldj/dz), log pi~(z) = log pi(T(z)) + ldj(z).
# T is a fixed bijection during the stage, so an exact kernel on z is an exact kernel on x. Where the flow has learned
# the posterior, pi~ is close to N(0, I): the thin, curved directions that cap the x-space step size are straightened.
function _latent_lpg(lpg!, proposal, nparams::Int, n::Int, ::Type{T}) where {T}
    gx = Matrix{T}(undef, nparams, n); lpx = Vector{T}(undef, n)
    return function (gz, lpz, Z, mask)
        X, ld, pull = proposal.from_latent(Z)
        lpg!(gx, lpx, Matrix{T}(X), mask)
        gz .= pull(gx)
        @inbounds for w in eachindex(lpz)
            lpz[w] = isfinite(lpx[w]) && isfinite(ld[w]) ? lpx[w] + ld[w] : T(-Inf)
        end
        return nothing
    end
end

# `cfg` without support bounds (the latent space is unbounded; the flow maps it into the prior box)
_unbounded(cfg::NUTSConfig{T}) where {T} =
    NUTSConfig{T}((f === :bounds ? nothing : getfield(cfg, f) for f in fieldnames(NUTSConfig))...)
# `cfg` with another tree-depth cap (`final_max_treedepth`)
_with_max_treedepth(cfg::NUTSConfig{T}, d::Int) where {T} =
    NUTSConfig{T}((f === :max_treedepth ? d : getfield(cfg, f) for f in fieldnames(NUTSConfig))...)

# One MALA draw (`local_kernel = :mala`, flowMC's local sampler): `nsteps` preconditioned Metropolis-adjusted Langevin
# steps x' = x + (h^2/2) Minv g(x) + h L xi (Minv = L L', h = `step[w]`, drawn independently of the state), each an exact
# MH step with the proposal-density ratio. One gradient call per step plus one for the starting state; the support
# is enforced by rejection (`lpg!` returns -Inf outside it). Returns (; accept_stat, n_leapfrog) like `nuts_draw!`.
function mala_draw!(positions::AbstractMatrix{T}, logp::AbstractVector{T}, step::AbstractVector{T}, lpg!,
                    metric::DenseMetric{T}, nsteps::Int, rng::AbstractRNG) where {T}
    d, n = size(positions)
    g = similar(positions); gp = similar(positions); lpp = similar(logp)
    lpg!(g, logp, positions, trues(n))
    Lt = LowerTriangular(metric.L)
    acc = zeros(T, n)
    xp = similar(positions)
    for _ in 1:nsteps
        xi = randn(rng, T, d, n)
        mg = metric.Minv * g
        xp .= positions .+ (step' .^ 2 ./ 2) .* mg .+ (Lt * xi) .* step'
        lpg!(gp, lpp, xp, trues(n))
        # reverse-move residual x - x' - (h^2/2) Minv g(x'), whitened: L^-1 r / h
        Rw = Lt \ (positions .- xp .- (step' .^ 2 ./ 2) .* (metric.Minv * gp))
        @inbounds for w in 1:n
            if !isfinite(lpp[w])
                continue
            elseif !isfinite(logp[w])
                a = one(T)
            else
                la = lpp[w] - logp[w] + T(0.5) * sum(abs2, view(xi, :, w)) - T(0.5) * sum(abs2, view(Rw, :, w)) / step[w]^2
                a = isfinite(la) ? min(one(T), exp(la)) : zero(T)
            end
            acc[w] += a
            if rand(rng, T) < a
                @views positions[:, w] .= xp[:, w]
                @views g[:, w] .= gp[:, w]
                logp[w] = lpp[w]
            end
        end
    end
    return (; accept_stat=acc ./ nsteps, n_leapfrog=fill(nsteps + 1, n))
end

"""
    smc_sample(positions0, loglik!, logp_and_grad!, betas; cfg, rng, kwargs...) -> NamedTuple

Adaptive tempered SMC sampler with batched NUTS mutations (see the comment block at the top of
`SMCCore.jl`). `positions0` is `(nparams, N)`: N exact prior draws. `betas` is the length-N vector
the tempered `logp_and_grad!` closure reads; this function overwrites it with the current stage's
beta before every mutation.

Keywords:
- `cfg::NUTSConfig`: tree depth, bounds, quorum; its `nwarmup`/`adapt_mass` are ignored (the metric
  and step size are set by the population), `mass_blocks` restricts the population metric.
- `moves` (`:nuts`): the mutation kernel. `:rw` = gradient-free adaptive random-walk Metropolis with the
  population covariance as the proposal shape (for models whose gradient costs many likelihoods); it only calls
  `loglik!` (one batched call per draw over all particles) and `logprior!`, and `logp_and_grad!` may be
  `nothing`. Needs `logprior!(lp, positions)`: the log prior per column, `-Inf` outside its support (no
  likelihood is evaluated there). Periodic rows of `cfg.bounds` are wrapped. The scale starts at `rw_scale0`
  (2.38/sqrt(d) whitened) and is tuned after every ladder draw toward `rw_target_accept` (0.234); `max_mutate`
  defaults to 1000 draws and `disp_target` to 1.2 (at 1.0 a correlated d = 6 Gaussian's log Z came out 0.25 low
  on average over 8 seeds, at 1.2 and 1.35 unbiased within 0.06 / 0.07; independent draws give sqrt(2)). The recorded beta = 1 phase keeps one state every `rw_thin` draws (default: the last
  ladder stage's draw count). A fraction `rw_de` (0.5) of the proposals are differential-evolution steps
  x + s/sqrt(2) (x_a - x_b) (+ 1% jitter) along differences of the stage's starting population, the full difference
  (a mode jump) with probability `rw_de_jump` (0.1); `rw_de = 0` = Gaussian steps only. Not combinable with
  `independence`, `online_proposal` or `block_move`.
- `target_cess` (0.5): conditional-ESS fraction that sets each beta increment.
- `final_ess` (`nothing` = off): a stage that would stop short of beta = 1 (in a hybrid only from the first NUTS
  stage on) goes to beta = 1 instead when the incremental weights' ESS / N there is >= `final_ess`. Only useful below
  `target_cess` (at or above it the target alone already decides).
- `dense` (true): block-dense population metric; `false` = diagonal. `mass_blocks` (default
  `cfg.mass_blocks`) masks cross-block entries, as in `NUTSConfig`.
- `min_mutate`/`max_mutate` (2/20): NUTS draws per stage; a stage stops once EVERY row's typical
  displacement from the post-resampling positions reaches `disp_target` (1.0) population sds, so
  resampled duplicates have separated in the slowest direction, not just in unconstrained ones.
  `move_stat` (`:median`) measures it as sqrt(2) median |displacement| / median distance between
  randomly paired particles (robust to particles jumping between modes, to minor modes and to flat
  rows); `:rms` = RMS displacement / weighted population sd (the earlier rule).
- `nfinal` (20): draws at beta = 1 after the ladder, all recorded (`chain`, `(nparams, N, nfinal)`).
- `final_max_treedepth` (`nothing` = `cfg.max_treedepth`): the NUTS tree-depth cap of the recorded beta = 1 phase only
  (every move stays exact; a shallower cap trades autocorrelation between recorded draws for gradients per draw).
- `step_size0` (0.5): initial step size in whitened units (refined once by the step-size heuristic),
  then tuned after every ladder draw toward `cfg.target_accept` from the population's mean
  acceptance statistic; frozen during the recorded beta = 1 phase.
- `reference` (default `nothing` = anneal from the prior): a NamedTuple
  `(; logq_and_grad!, logprior_and_grad!)`, each `(lp, grad, positions) -> nothing` filling a
  log-density and its gradient. `positions0` are then draws from the NORMALIZED reference q0 and the
  path is `q0^(1-beta) (prior L)^beta`; `logZ` is still the evidence relative to the prior (exact
  as long as q0 is normalized on the sampled support and covers the posterior -- use a defensive
  mixture with the prior). The returned `ll` is then `logL + logprior - logq0`. An optional
  `sample(rng, n)` field enables `independence`.
- `independence` (false): after every NUTS draw, also propose each particle afresh from q0 and
  accept by independence Metropolis-Hastings (exact for pi_beta; a global move across modes/ridges).
- `online_proposal` (default `nothing`): a function `(pool, weights) -> (; sample, logq_and_grad!)` (or `nothing`
  to keep the current proposal). After every ladder stage it is called with that stage's mutation
  states (`(nparams, m)`, thinned to `online_pool_max`) and the returned normalized density is used for
  independence-MH moves throughout the NEXT stage (and the final phase). The annealing path is unchanged
  (prior, or `reference`); only the global proposal adapts, and each stage's proposal is fixed before its
  moves, so every move is exact for its pi_beta. This is the training-free analogue of flowMC's
  on-the-fly flow (e.g. a local KDE of the pool plus a prior component), removing the need for a
  reference from an earlier run. `online_nuts = false`: a hybrid builds and uses it in its RW stages only (a
  hybrid's NUTS stages then make no independence moves and fit no proposal).
- `persistent` (false): persistent sampling (Karamanis et al. 2025). Every stage's post-mutation
  population is kept; each new stage resamples from the WHOLE pool with multiple-importance weights
  against the equal mixture of the earlier targets, and log Z uses the same weights. Particles lost to an
  earlier resampling can return, which stabilizes minor-mode weights at no extra likelihood cost (the
  pool's log-ratios are stored). The beta schedule still uses the current population's CESS.
- `independence_tries` (1): independence / online-proposal MH steps per NUTS draw (each one exact; cheap with
  `logprior!`, so several per draw multiply the global moves at little cost).
- `ind_llk` (`nothing`): a log-likelihood `ind_llk(ll, X)` for `independence_tries * N` columns at once (prior or reference path
  with `logprior!` only). All of a draw's independence proposals are then drawn and evaluated in one batch (one wide
  device call instead of `independence_tries` narrow ones) and applied as the same sequence of MH steps: the proposals
  do not depend on the current state, so this is the same chain.
- `de_tries` (0), `de_jump` (0.1): differential-evolution Metropolis steps per NUTS draw (needs `logprior!`, prior
  path): x' = x + g (x_a - x_b) + 1% jitter with a random pair of the stage's starting population, g = 2.38/sqrt(2d)
  (x U(0.8, 1.2)) or 1 with probability `de_jump`; one likelihood call each. They follow curved ridges and jump
  between modes, where NUTS with one population metric needs tiny steps.
- `logprior!` with `moves = :nuts`: optional; on the prior path it makes independence / online-proposal moves
  evaluate `beta * loglik! + logprior!` (one likelihood call) instead of `logp_and_grad!` (must be the same density).
- `rw_moves` (`()`, `moves = :rw` only): extra proposals after every random-walk draw, each a function
  `move!(xp, positions, rng)` filling `xp` (`(nparams, N)`) with one proposal per particle; each must be symmetric
  with respect to Lebesgue measure on the sampled coordinates (e.g. a measure-preserving map with a symmetrically
  drawn parameter: a rotation of the sky about a detector baseline by a random angle), so plain MH applies. One
  likelihood call each; acceptance reported as `move_accept`.
- `mode_jumps` (false): learned mode-jumping moves (`_mode_jump_learn!`, `_mode_jump_tries!`). After every ladder stage
  at beta >= `mode_jump_learn_beta` (0) the population is clustered by k-means into `mode_jump_k` (10) cells on the
  `mode_jump_rows` (`nothing` = all rows; standardized, periodic rows of `cfg.bounds` as (cos, sin)), and the mean
  difference of every pair of cells holding >= max(10, `mode_jump_min_cell` (0.01) N) particles joins a growing set of
  jump vectors (never dropped). After every draw of every kernel (RW, NUTS, the recorded phase) each particle then makes
  `mode_jump_tries` (12) MH steps x -> x + s delta on those rows, delta uniform over the set as it stood at the stage's
  start, |s| ~ U(`mode_jump_scale`...) ((0.2, 0.7)) with a random sign: symmetric, unit Jacobian, plain MH on the stage
  density. Carries weight between a bulk and a minor mode that NUTS / RW cannot cross, and refills a minor mode that
  tempering resampling drained. Needs `logprior!`. The tries are evaluated in rounds of up to `mode_jump_batch` (1) per
  particle, one `mode_jump_llk` call (default `loglik!`; it must accept `mode_jump_batch * N` columns; prior path only
  when > 1) per round (the same chain as sequential tries; a wide device call amortizes latency). Acceptance
  `mode_jump_accept`, set size `mode_jumps`, evaluated columns `mode_jump_evals` (not in `n_loglik`).
  `mode_jump_learn` (`:kmeans`) picks the vector source: `:tails` (`_mode_jump_tails!`) instead appends, for every tail
  quantity in `mode_jump_tails` (functions `q(X) -> values`, one per particle; `nothing` = every non-periodic jump row)
  plus `mode_jump_tail_pcs` leading principal components (default 3 without `mode_jump_tails`, else 0), and every
  fraction p in `mode_jump_tail_frac` ((0.02, 0.05)), mean(tail) - mean(rest) on the jump rows for the lower and the
  upper p-tail (>= `mode_jump_tail_min` (8) particles); `:both` appends both kinds. The callback's `mode_jump_cells`
  then counts the tail quantities used. Applying the jumps is the same either way.
- `nuts_beta` (`nothing`; `moves = :rw` only): a hybrid -- from the first stage at beta >= `nuts_beta` (and in the
  recorded beta = 1 phase) mutate with NUTS (`logp_and_grad!` required, with `logprior!` the same density), whose
  whitened step size is set once at the switch (Stan's heuristic) and tuned as in `moves = :nuts`; at most
  `nuts_max_mutate` (20) draws per stage. Random walk finds the signal cheaply while the population is prior-like; NUTS
  follows the curved posterior manifolds (e.g. mass-spin) once it has. `rw_moves` keep running after every NUTS draw
  (prior path).
- `metric_core` (`nothing`): estimate each stage's metric (NUTS mass matrix / RW proposal shape) from only the
  fraction `metric_core` of particles with the highest log-likelihood (`_metric_weights`): robust to prior-like
  stragglers while the population hands over to a narrow mode. Any fixed metric keeps the moves exact.
- `hessian_metric` (0): fold the local curvature into each stage's metric, Minv = (Sigma^-1 + Hbar)^-1 with Hbar the
  negative Hessian of the tempered target averaged over up to `hessian_metric` distinct top-logL particles (one
  batched `logp_and_grad!` call per stage; needs `logp_and_grad!`, also shapes the RW proposals of a hybrid). For
  targets far thinner locally than the population in a few directions (`_hessian_metric`).
- `local_kernel` (`:nuts`): the gradient kernel of `moves = :nuts` and of a hybrid's NUTS stages. `:mala` = `mala_steps`
  (10) preconditioned MALA steps per draw (flowMC's local sampler; one gradient each), step size tuned toward
  `mala_target_accept` (0.574).
- `latent_nuts` (false): run the gradient kernel in the latent space of the stage's online proposal (NeuTra): the
  proposal must provide `to_latent(X) -> Z` and `from_latent(Z) -> (X, ldj, pull)` (as the flow proposals do; stages
  whose proposal does not fall back to x space). The kernel targets pi(T(z)) |det dT/dz| with the population metric
  of the particles' latent coordinates and its own step size (`_latent_lpg`).
- `block_move` (default `nothing`): a function `(positions, betas, metric, rng) -> n_evals` called after
  every NUTS draw; it must leave each particle's pi_beta = prior L^beta invariant (e.g. NUTS on a
  parameter block at fixed other rows -- Metropolis-within-Gibbs) and returns the number of (cheap)
  evaluations it made, reported as `block_evals`. Prior path only (no `reference`).
- `callback(stage, info)`: optional per-stage hook for progress output (`info.positions`, `info.ll`: the current
  population, not copies).

Returns `(; positions, ll, logp, logZ, beta_schedule, chain, nstages, n_leapfrog, n_loglik, step_size,
metric, accept, ancestors_unique, mode_jump_accept, mode_jumps, mode_jump_evals, ...)`: `logZ` is the log-evidence estimate (normalized to the prior),
`n_leapfrog` the total gradient evaluations summed over particles, `n_loglik` the untempered
log-likelihood evaluations (`step_size` is the RW scale for `moves = :rw`), `ancestors_unique` how many
distinct initial prior draws survive in the final population (a mode-collapse diagnostic).
"""
function smc_sample(positions0::AbstractMatrix{T}, loglik!::L, logp_and_grad!::F,
                    betas::AbstractVector{T}; cfg::NUTSConfig{T}, rng::AbstractRNG=Random.default_rng(),
                    moves::Symbol=:nuts, logprior! = nothing,
                    target_cess=T(0.5), dense::Bool=true, min_mutate::Int=2,
                    max_mutate::Int=(moves === :rw ? 1000 : 20),
                    disp_target=T(moves === :rw ? 1.2 : 1.0), nfinal::Int=20, step_size0=T(0.5), max_stages::Int=1000,
                    rw_scale0=T(2.38) / sqrt(T(size(positions0, 1))), rw_target_accept=T(0.234),
                    rw_thin::Union{Nothing,Int}=nothing, rw_de=T(0.5), rw_de_jump=T(0.1),
                    mass_blocks=cfg.mass_blocks, reference=nothing, independence::Bool=false,
                    online_proposal=nothing, online_pool_max::Int=512, independence_tries::Int=1, online_nuts::Bool=true,
                    ind_llk=nothing, de_tries::Int=0, de_jump=T(0.1), persistent::Bool=false,
                    block_move=nothing, rw_moves=(), nuts_beta=nothing, nuts_max_mutate::Int=20, metric_core=nothing,
                    hessian_metric::Int=0, local_kernel::Symbol=:nuts, mala_steps::Int=10,
                    mala_target_accept=T(0.574), latent_nuts::Bool=false,
                    move_stat::Symbol=:median, final_ess=nothing, final_max_treedepth=nothing,
                    mode_jumps::Bool=false, mode_jump_rows=nothing, mode_jump_tries::Int=12, mode_jump_k::Int=10,
                    mode_jump_learn_beta=T(0), mode_jump_scale=(T(0.2), T(0.7)), mode_jump_min_cell=T(0.01),
                    mode_jump_batch::Int=1, mode_jump_llk=nothing, mode_jump_learn::Symbol=:kmeans,
                    mode_jump_tails=nothing, mode_jump_tail_frac=(T(0.02), T(0.05)), mode_jump_tail_min::Int=8,
                    mode_jump_tail_pcs=nothing,
                    callback=nothing) where {T,L,F}
    nparams, n = size(positions0)
    length(betas) == n || throw(ArgumentError("`betas` must have one entry per particle"))
    n >= 4 || throw(ArgumentError("smc_sample needs at least 4 particles"))
    moves in (:nuts, :rw) || throw(ArgumentError("moves must be :nuts or :rw, got :$moves"))
    local_kernel in (:nuts, :mala) || throw(ArgumentError("local_kernel must be :nuts or :mala, got :$local_kernel"))
    mala_steps >= 1 || throw(ArgumentError("mala_steps must be >= 1"))
    rw = moves === :rw
    rw && logprior! === nothing && throw(ArgumentError("moves = :rw needs `logprior!`"))
    # (online proposals on a reference path are exact too: the RW state's base is log q0 and `llk!` the reference-path
    # log-likelihood, so base + beta llk is the tempered target)
    rw && (independence || block_move !== nothing) &&
        throw(ArgumentError("moves = :rw does not support independence or block_move"))
    (!rw && logp_and_grad! === nothing) && throw(ArgumentError("moves = :nuts needs `logp_and_grad!`"))
    (!rw && !isempty(rw_moves)) && throw(ArgumentError("rw_moves need moves = :rw"))
    hybrid = nuts_beta !== nothing
    hybrid && (rw && logp_and_grad! !== nothing) || !hybrid ||
        throw(ArgumentError("nuts_beta needs moves = :rw and `logp_and_grad!`"))
    (independence && (reference === nothing || !hasproperty(reference, :sample))) && throw(ArgumentError(
        "independence moves need a `reference` with a `sample(rng, n)` field"))
    (block_move !== nothing && reference !== nothing) && throw(ArgumentError(
        "block_move targets prior * L^beta; it cannot be combined with a `reference` path"))
    jrows = mode_jump_rows === nothing ? collect(1:nparams) : collect(Int, mode_jump_rows)
    if mode_jumps
        logprior! === nothing && throw(ArgumentError("mode_jumps needs `logprior!`"))
        (!isempty(jrows) && all(r -> 1 <= r <= nparams, jrows) && allunique(jrows)) ||
            throw(ArgumentError("mode_jump_rows must be distinct rows in 1:$nparams, got $jrows"))
        (mode_jump_tries >= 1 && mode_jump_k >= 2 && mode_jump_batch >= 1) ||
            throw(ArgumentError("mode_jump_tries, mode_jump_batch must be >= 1 and mode_jump_k >= 2"))
        0 <= mode_jump_scale[1] <= mode_jump_scale[2] ||
            throw(ArgumentError("mode_jump_scale must be (smin, smax) with 0 <= smin <= smax"))
        (mode_jump_batch > 1 && reference !== nothing) &&
            throw(ArgumentError("mode_jump_batch > 1 needs the prior path (no reference)"))
        mode_jump_learn in (:kmeans, :tails, :both) ||
            throw(ArgumentError("mode_jump_learn must be :kmeans, :tails or :both, got :$mode_jump_learn"))
        (all(p -> 0 < p < 1, mode_jump_tail_frac) && mode_jump_tail_min >= 1) ||
            throw(ArgumentError("mode_jump_tail_frac must lie in (0, 1) and mode_jump_tail_min be >= 1"))
    end
    # tail quantities (`mode_jump_learn = :tails`): the caller's, else every non-periodic jump row plus the leading
    # principal components (`mode_jump_tail_pcs`: default 3 without caller quantities, 0 with)
    mj_tailq = mode_jump_tails !== nothing ? collect(Any, mode_jump_tails) :
               Any[let r = r; X -> X[r, :]; end for r in jrows if !(cfg.bounds !== nothing && cfg.bounds.periodic[r])]
    mj_npcs = Int(something(mode_jump_tail_pcs, mode_jump_tails === nothing ? 3 : 0))
    beta = zero(T)
    # No closure below captures a variable that is reassigned (no Core.Box): the current beta reaches the reference path
    # through `beta_ref`, and `mutate!` takes the evolving state as arguments. Julia 1.13.0/1.13.1 deleted every store
    # into some boxed captures of `mutate!` (`beta`, `total_leap`, `block_evals`) for some argument types (a capturing
    # `callback`), which surfaced as an UndefVarError inside `mutate!`.
    beta_ref = Ref(beta)
    lpg!, llk! = reference === nothing ? (logp_and_grad!, loglik!) :
                 _smc_reference_path(reference, logp_and_grad!, loglik!, beta_ref, nparams, n)
    total_ll = n                      # untempered log-likelihood evaluations (particles x calls)
    positions = Matrix{T}(positions0)
    ll = Vector{T}(undef, n)
    llk!(ll, positions)
    ll[.!isfinite.(ll)] .= T(-Inf)
    ancestor = collect(1:n)
    logZ = zero(T)
    schedule = T[beta]
    metric = DenseMetric(Matrix{T}(I, nparams, nparams))
    eps_ = rw ? T(rw_scale0) : T(step_size0)      # NUTS step size, or the RW proposal scale (whitened units)
    nuts_eps = T(step_size0); nuts_started = false # the NUTS step size (moves = :nuts, or a hybrid's NUTS stages)
    lat_eps = T(step_size0); x_set = false    # latent / x-space kernel step sizes (x: carried once set)
    stage_latent = false; hsteps = (T(NaN), T(NaN))   # this stage's kernel space; heuristic steps (latent, x)
    logp = Vector{T}(undef, n)
    total_leap = 0
    block_evals = 0
    accept_hist = T[]
    last_move = Ref((T(NaN), 0))
    stage = 0
    last_k = 1
    g_ind = Matrix{T}(undef, nparams, n); lp_ind = Vector{T}(undef, n); ll_ind = Vector{T}(undef, n)
    lb_ind = Vector{T}(undef, n)
    # Independence proposals need only the tempered log density: on the prior path with `logprior!` it is
    # beta * loglik + log prior (what `logp_and_grad!` computes, the NUTS draw recomputes its own gradient), one
    # likelihood call instead of a gradient call.
    cheap_ind = reference === nothing && logprior! !== nothing
    lqp_ind = Vector{T}(undef, n); lqc_ind = Vector{T}(undef, n); gq_ind = Matrix{T}(undef, nparams, n)
    ind_acc = Ref(0); ind_tries = Ref(0)
    # per-stage wall-time split (seconds, reset every stage; `callback`'s `time_split`): `llk` likelihood calls (device +
    # the closure's host side), `base` prior / reference density, `flow_sample` / `flow_logq` online-proposal draws and
    # densities, `moves` the caller's rw_moves proposals, `flow_fit` the online proposal's build, `nuts` NUTS mutations
    prof = Dict{Symbol,Float64}()
    tacc!(k, t0) = (prof[k] = get(prof, k, 0.0) + (time_ns() - t0) / 1e9; nothing)
    (ind_llk === nothing || logprior! !== nothing) || throw(ArgumentError("ind_llk needs `logprior!`"))
    nb_ind = independence_tries * n
    lb_b = ind_llk === nothing ? T[] : Vector{T}(undef, nb_ind); ll_b = similar(lb_b); lq_b = similar(lb_b)
    gq_b = ind_llk === nothing ? Matrix{T}(undef, 0, 0) : Matrix{T}(undef, nparams, nb_ind)
    # reference path: the tempered density `logp` holds there is beta logL + logprior + (1 - beta)(logq0 -
    # logprior_ref) (`_smc_reference_path`'s `lpg!`), so the batch also needs q0's and the reference prior's densities
    rq_b = (ind_llk === nothing || reference === nothing) ? T[] : similar(lb_b); rp_b = similar(rq_b)
    # All `independence_tries` proposals of a draw in one batch (`ind_llk`): log prior, log-likelihood and proposal
    # density of every column, invalid columns replaced by the particle's current state (rejected anyway). Returns the
    # proposals; column (t - 1) N + w is try t of particle w.
    function ind_batch!(proposal, positions)
        t1 = time_ns(); xb = proposal.sample(rng, nb_ind); tacc!(:flow_sample, t1)
        t1 = time_ns(); logprior!(lb_b, xb); tacc!(:base, t1)
        @inbounds for j in 1:nb_ind
            isfinite(lb_b[j]) || (@views xb[:, j] .= positions[:, mod1(j, n)])
        end
        t1 = time_ns(); ind_llk(ll_b, xb); tacc!(:llk, t1)
        if reference !== nothing
            t1 = time_ns()
            reference.logq_and_grad!(rq_b, gq_b, xb); reference.logprior_and_grad!(rp_b, gq_b, xb)
            tacc!(:base, t1)
        end
        t1 = time_ns()
        proposal.logq_and_grad!(lq_b, gq_b, xb)
        proposal.logq_and_grad!(lqc_ind, gq_ind, positions)
        tacc!(:flow_logq, t1)
        return xb
    end
    # Differential-evolution Metropolis steps after each NUTS draw (`de_tries`, prior path with `logprior!`):
    # x' = x + g (x_a - x_b) + jitter, (a, b) a random pair of the stage's starting population (fixed during the stage:
    # the proposal is symmetric), g = 2.38 / sqrt(2 d) jittered, or 1 (a jump between modes) with probability `de_jump`.
    (de_tries > 0 && !cheap_ind) && throw(ArgumentError("de_tries needs `logprior!` and the prior path (no reference)"))
    de_xp = Matrix{T}(undef, nparams, n); de_z = Matrix{T}(undef, nparams, n)
    de_acc = Ref(0); de_n = Ref(0)
    mv_acc = Ref(0); mv_n = Ref(0)                # `rw_moves`
    mv_xp = Matrix{T}(undef, nparams, n); mv_lp = Vector{T}(undef, n); mv_ll = Vector{T}(undef, n)
    mv_lq = Vector{T}(undef, n); mv_gq = Matrix{T}(undef, nparams, n)
    # The base density of the tempered target base + beta * llk: the log prior, or on a reference path log q0 (inside
    # the prior's support; `llk!` is then logL + logprior - logq0), so base + beta llk is what the NUTS `logp` holds.
    function base_density!(lb, pos)
        logprior!(lb, pos)
        reference === nothing && return lb
        reference.logq_and_grad!(mv_lq, mv_gq, pos)
        @inbounds for w in eachindex(lb)
            lb[w] = isfinite(lb[w]) ? mv_lq[w] : T(-Inf)
        end
        return lb
    end
    # Mode jumps (docstring): the learned vectors (appended between stages only), counters, round buffers.
    mj_deltas = Vector{Vector{T}}()
    mj_acc = Ref(0); mj_n = Ref(0); mj_cols = Ref(0)
    mj_buf = mode_jumps ? (Matrix{T}(undef, nparams, mode_jump_batch * n), Vector{T}(undef, mode_jump_batch * n),
                           Vector{T}(undef, mode_jump_batch * n), falses(mode_jump_batch * n)) : nothing
    mj_llk! = something(mode_jump_llk, llk!)
    mj_base! = mode_jump_batch > 1 ? logprior! : base_density!
    mj_ll = Vector{T}(undef, n); mj_lb = Vector{T}(undef, n); mj_old = Vector{T}(undef, n); mj_moved = falses(n)
    function mode_jump!(positions, cur, ll, lb, beta)
        fill!(mj_moved, false)
        (mode_jumps && !isempty(mj_deltas)) || return nothing
        t1 = time_ns()
        mj_cols[] += _mode_jump_tries!(positions, cur, ll, lb, mj_moved, beta, mj_deltas, jrows, cfg.bounds,
                                       mode_jump_tries, T(mode_jump_scale[1]), T(mode_jump_scale[2]), mode_jump_batch,
                                       mj_llk!, mj_base!, mj_buf, mj_acc, mj_n, rng)
        tacc!(:jumps, t1)
        return nothing
    end
    # `rw_moves` after a NUTS draw (hybrid), then the mode jumps (any NUTS stage): plain MH on base + beta * llk. On the prior path that is the density
    # `logp` holds. On a reference path `logp` (from `lpg!`) carries the caller's unnormalized prior where llk carries
    # the reference's normalized one -- they differ by a constant -- so the current states' base + beta llk is evaluated
    # (one `llk!` call) and `logp` is shifted by the accepted difference, keeping its own normalization.
    mv_cur = Vector{T}(undef, n)
    function nuts_rw_moves!(positions, beta)
        (isempty(rw_moves) && (!mode_jumps || isempty(mj_deltas))) && return nothing
        if reference === nothing
            mv_cur .= logp
        else
            llk!(mv_ll, positions)
            base_density!(mv_lp, positions)
            @inbounds for w in 1:n
                mv_cur[w] = beta * mv_ll[w] + mv_lp[w]
            end
        end
        for move! in rw_moves
            move!(mv_xp, positions, rng)
            base_density!(mv_lp, mv_xp)
            @inbounds for w in 1:n
                isfinite(mv_lp[w]) || (@views mv_xp[:, w] .= positions[:, w])
            end
            mv_n[] += n
            _any_moved(mv_xp, positions, mv_lp) || continue           # a no-op proposal: no likelihood call
            llk!(mv_ll, mv_xp)
            @inbounds for w in 1:n
                (isfinite(mv_lp[w]) && isfinite(mv_ll[w])) || continue
                lpn = beta * mv_ll[w] + mv_lp[w]
                if !isfinite(logp[w]) || !isfinite(mv_cur[w]) || log(rand(rng, T)) < lpn - mv_cur[w]
                    @views positions[:, w] .= mv_xp[:, w]
                    logp[w] = isfinite(logp[w]) && isfinite(mv_cur[w]) ? logp[w] + (lpn - mv_cur[w]) : lpn
                    mv_cur[w] = lpn
                    mv_acc[] += 1
                end
            end
        end
        if mode_jumps
            # (`mv_cur` is base + beta llk of the current states; `logp` shifted by the accepted difference, as above)
            mj_old .= mv_cur
            mode_jump!(positions, mv_cur, mj_ll, mj_lb, beta)
            @inbounds for w in 1:n
                mj_moved[w] || continue
                logp[w] = isfinite(logp[w]) && isfinite(mj_old[w]) ? logp[w] + (mv_cur[w] - mj_old[w]) : mv_cur[w]
            end
        end
        return nothing
    end
    # Online proposal (see docstring): built after every ladder stage from that stage's mutation states,
    # fixed during the next stage. `pool` holds the states of the current stage.
    proposal = independence ? reference : nothing
    pool = Vector{Vector{T}}()
    pool_ll = T[]                     # the pool states' untempered log-likelihoods (NaN: unknown)
    pool_beta = Ref(zero(T))          # the beta the pool was sampled at
    # Persistent sampling: every stage's post-mutation population with its log-ratios, beta and log Z.
    ps_pos = Matrix{T}[]; ps_r = Vector{T}[]; ps_beta = T[]; ps_logZ = T[]; ps_anc = Vector{Int}[]

    # Stan's step-size heuristic for a kernel at the current particles: the median step with a single-leapfrog
    # acceptance of 0.5, in that kernel's whitened units (one batched gradient call per doubling / halving from `start`).
    function heuristic_step(kern, positions, beta, start)
        fill!(betas, beta)
        ss = fill(T(start), n)
        _find_reasonable_step_size!(ss, Matrix{T}(kern.latent ? kern.to(positions) : positions), kern.lpg, kern.cfg, rng,
                                    kern.metric)
        return sort(ss)[cld(n, 2)]
    end
    # The gradient kernel for a stage and its starting step size: x space (log density `lpg!`, the population metric,
    # the tuned step carried between stages in `x_known`, or the heuristic's / 2 the first time), or -- `latent_nuts`
    # with a proposal providing the maps -- the latent space of the stage's proposal, with the population metric of the
    # particles' latent coordinates. The latent space is used only where it is better conditioned: both heuristic steps
    # are in population-whitened units, so the larger one marks the space the kernel moves further in per gradient (a
    # flow fitted to too few states can make the latent target far stiffer than x). Decided before the stage's moves
    # from the population, like the metric: every move stays exact. (State passed in: no boxed captures.)
    function stage_kernel(proposal, positions, metric, beta, x_known)
        xk = (; latent=false, lpg=lpg!, metric, cfg, to=identity, from=identity)
        if latent_nuts && proposal !== nothing && hasproperty(proposal, :from_latent)
            Z0 = Matrix{T}(proposal.to_latent(positions))
            km = _smc_metric(Z0, fill(one(T) / n, n), dense, nothing, DenseMetric(Matrix{T}(I, nparams, nparams)))
            lk = (; latent=true, lpg=_latent_lpg(lpg!, proposal, nparams, n, T), metric=km, cfg=_unbounded(cfg),
                  to=proposal.to_latent, from=proposal.from_latent)
            hz = heuristic_step(lk, positions, beta, step_size0)
            hx = heuristic_step(xk, positions, beta, x_known === nothing ? step_size0 : 2x_known)
            hz > hx && return lk, hz / 2, (hz, hx)
            return xk, something(x_known, hx / 2), (hz, hx)
        end
        return xk, x_known === nothing ? heuristic_step(xk, positions, beta, step_size0) / 2 : x_known, (T(NaN), T(NaN))
    end
    tgt_accept = local_kernel === :mala ? T(mala_target_accept) : cfg.target_accept
    # One mutation stage at `beta`: NUTS draws until the particles have moved `disp_target` whitened
    # sds (bounded by min/max_mutate), or exactly `ndraws` draws when given (the final beta = 1
    # sampling phase, which records every draw). Updates `positions`, `ll` (and `logp`) in place;
    # returns (draws, step size, gradient evaluations, block evaluations). `kern`: `stage_kernel`.
    function mutate!(ndraws, chain, positions, ll, metric, eps_, beta, proposal, kern)
        fill!(betas, beta)
        start = copy(positions)
        empty!(pool); empty!(pool_ll)
        acc_sum = zero(T); nacc = 0
        nleap = 0; nblock = 0
        k = 0
        while true
            k += 1
            step = eps_ .* (T(0.8) .+ T(0.4) .* rand(rng, T, n))          # jittered, shared scale
            P = kern.latent ? Matrix{T}(kern.to(positions)) : positions
            stats = local_kernel === :mala ? mala_draw!(P, logp, step, kern.lpg, kern.metric, mala_steps, rng) :
                    nuts_draw!(P, logp, step, kern.lpg, kern.cfg, rng, kern.metric)
            if kern.latent                         # back to x; logp from the latent density: log pi(x) = log pi~(z) - ldj
                X, ld, _ = kern.from(P)
                positions .= X
                @inbounds for w in 1:n
                    logp[w] = isfinite(logp[w]) ? logp[w] - ld[w] : T(-Inf)
                end
            end
            nleap += sum(stats.n_leapfrog)
            acc_sum += sum(stats.accept_stat); nacc += n
            if block_move !== nothing
                # Caller-supplied pi_beta-invariant block update (e.g. extrinsic parameters with a cached
                # waveform); its evaluations are counted separately (they are much cheaper).
                nblock += block_move(positions, betas, metric, rng)
                if proposal !== nothing
                    lpg!(g_ind, logp, positions, trues(n))
                    nleap += n
                end
            end
            if proposal !== nothing
                # Independence Metropolis-Hastings from a proposal that is fixed during this stage (the
                # reference q0, or the online proposal built from the previous stage): exact for pi_beta,
                # and a global move -- it jumps between modes/ridges NUTS can only diffuse across. One
                # log-density evaluation per particle and try (`independence_tries` per draw; each try is an exact
                # MH step, so any number is valid).
                if ind_llk !== nothing
                    xb = ind_batch!(proposal, positions)
                    @inbounds for t in 1:independence_tries, w in 1:n
                        j = (t - 1) * n + w
                        lpj = isfinite(lb_b[j]) && isfinite(ll_b[j]) ? betas[w] * ll_b[j] + lb_b[j] : T(-Inf)
                        reference === nothing || (lpj += (one(T) - betas[w]) * (rq_b[j] - rp_b[j]))
                        la = (lpj - lq_b[j]) - (logp[w] - lqc_ind[w])
                        if isfinite(la) && log(rand(rng, T)) < la
                            @views positions[:, w] .= xb[:, j]
                            logp[w] = lpj; lqc_ind[w] = lq_b[j]
                            ind_acc[] += 1
                        end
                    end
                    ind_tries[] += nb_ind
                end
                for _ in 1:(ind_llk === nothing ? independence_tries : 0)
                    xp = proposal.sample(rng, n)
                    if cheap_ind
                        # proposals outside the prior's support are not evaluated (a global proposal can land
                        # anywhere in the box, e.g. at spin magnitudes > 1 where a waveform throws): their column is
                        # replaced by the current state in the one batched `llk!` call, and they are rejected below
                        logprior!(lb_ind, xp)
                        xe = copy(xp)
                        @inbounds for w in 1:n
                            isfinite(lb_ind[w]) || (@views xe[:, w] .= positions[:, w])
                        end
                        llk!(ll_ind, xe)
                        @inbounds for w in 1:n
                            lp_ind[w] = isfinite(lb_ind[w]) && isfinite(ll_ind[w]) ? betas[w] * ll_ind[w] + lb_ind[w] : T(-Inf)
                        end
                    else
                        lpg!(g_ind, lp_ind, xp, trues(n))
                    end
                    proposal.logq_and_grad!(lqp_ind, gq_ind, xp)
                    proposal.logq_and_grad!(lqc_ind, gq_ind, positions)
                    cheap_ind || (nleap += n)                 # (likelihood-only tries are counted in `ind_tries`)
                    for w in 1:n
                        la = (lp_ind[w] - lqp_ind[w]) - (logp[w] - lqc_ind[w])
                        if isfinite(la) && log(rand(rng, T)) < la
                            @views positions[:, w] .= xp[:, w]
                            logp[w] = lp_ind[w]
                            ind_acc[] += 1
                        end
                    end
                    ind_tries[] += n
                end
            end
            if de_tries > 0
                _de_steps!(positions, logp, start, metric, betas, de_xp, de_z, ll_ind, lb_ind, llk!, logprior!, cfg.bounds,
                           de_tries, de_jump, de_acc, de_n, rng)
            end
            (hybrid || mode_jumps) && nuts_rw_moves!(positions, beta)
            chain === nothing || (chain[:, :, k] .= positions)
            logprior! === nothing || base_density!(lb_ind, positions)
            for w in 1:n
                push!(pool, positions[:, w])
                push!(pool_ll, logprior! === nothing ? T(NaN) : (logp[w] - lb_ind[w]) / beta)
            end
            pool_beta[] = beta
            if ndraws === nothing
                # Ladder stages: population step-size update after every draw (Robbins-Monro on log eps
                # toward `cfg.target_accept`; it couples particles only through the population mean,
                # an O(1/N) effect). The recorded beta = 1 phase keeps the step frozen (exact MCMC).
                a = sum(stats.accept_stat) / n
                eps_ *= exp(clamp(T(2) * (a - tgt_accept), T(-1), T(0.5)))
                # Slowest row's RMS displacement in units of the population sd. The MIN over rows
                # matters: rows the likelihood does not constrain (e.g. marginalized phase/distance)
                # move freely and would satisfy an averaged criterion on their own.
                slowest = _slowest_move(move_stat, start, positions, metric)
                (k >= (hybrid ? nuts_max_mutate : max_mutate) || (k >= min_mutate && slowest[1] >= disp_target)) && break
            else
                k >= ndraws && break
            end
        end
        llk!(ll, positions)
        ll[.!isfinite.(ll)] .= T(-Inf)
        acc = acc_sum / nacc
        push!(accept_hist, acc)
        last_move[] = _slowest_move(move_stat, start, positions, metric)
        return k, eps_, nleap, nblock
    end

    # Gradient-free mutation (`moves = :rw`): Gaussian random-walk Metropolis with the population covariance
    # (`metric.Minv = L L'`) as the proposal shape, x' = x + s L z, targeting base(x) + beta * llk(x), where base is the
    # log prior (or log q0 on a reference path; then llk = logL + logprior - logq0, see `_smc_reference_path`). The
    # scale s is shared, jittered by U(0.8, 1.2) per particle and draw (state-independent, so the proposal stays
    # symmetric), and tuned after every ladder draw toward `rw_target_accept`; frozen in the recorded phase.
    # With probability `rw_de` a particle's proposal is instead a differential-evolution step along the difference of
    # two particles of the stage's starting population (fixed during the stage, like the covariance; the pairs (a, b)
    # and (b, a) are equally likely, so the proposal is symmetric): it follows curved ridges and, with the full
    # difference (`rw_de_jump`), jumps between modes, which one global Gaussian cannot.
    # Periodic rows of `cfg.bounds` are wrapped (symmetric: a sum over translates); every other support boundary is a
    # rejection -- reflecting a CORRELATED proposal per coordinate would not be symmetric. Proposals outside the base
    # density's support are not evaluated (their column is replaced by the current state in the one batched `llk!`
    # call per draw, so device closures see a fixed batch shape). `ll` stays exact for the current states throughout,
    # so no re-evaluation at the end. `ndraws`/`chain`: the recorded phase, one stored state every `thin` draws.
    # Returns (draws, scale, log-likelihood evaluations).
    rw_lb = Vector{T}(undef, n); rw_lbp = Vector{T}(undef, n); rw_llp = Vector{T}(undef, n)
    rw_xp = Matrix{T}(undef, nparams, n); rw_z = Matrix{T}(undef, nparams, n)
    rw_lpr = Vector{T}(undef, n); rw_g = Matrix{T}(undef, nparams, n)
    function rw_base!(lb, pos)
        logprior!(lb, pos)
        reference === nothing && return lb
        reference.logq_and_grad!(rw_lpr, rw_g, pos)
        @inbounds for w in eachindex(lb)
            lb[w] = isfinite(lb[w]) ? rw_lpr[w] : T(-Inf)
        end
        return lb
    end
    function rw_mutate!(ndraws, thin, chain, positions, ll, logp, metric, scale, beta, proposal)
        start = copy(positions)
        rw_base!(rw_lb, positions)
        bounds = cfg.bounds
        acc_sum = zero(T)
        nevals = 0
        k = 0
        while true
            k += 1
            randn!(rng, rw_z)
            mul!(rw_xp, LowerTriangular(metric.L), rw_z)
            @inbounds for w in 1:n
                s = scale * (T(0.8) + T(0.4) * rand(rng, T))
                if rw_de > 0 && rand(rng, T) < rw_de
                    # differential evolution: s/sqrt(2) (x_a - x_b) (the same size as s L z), or the whole difference
                    # (a jump between modes) with probability `rw_de_jump`, plus a 1% Gaussian jitter
                    a = rand(rng, 1:n); b = rand(rng, 1:n-1); b += b >= a
                    g = rand(rng, T) < rw_de_jump ? one(T) : s / sqrt(T(2))
                    for r in 1:nparams
                        dd = start[r, a] - start[r, b]
                        if bounds !== nothing && bounds.periodic[r]
                            P = bounds.hi[r] - bounds.lo[r]
                            dd -= P * round(dd / P)
                        end
                        x = positions[r, w] + g * dd + T(0.01) * s * rw_xp[r, w]
                        if bounds !== nothing && bounds.periodic[r]
                            x = bounds.lo[r] + mod(x - bounds.lo[r], bounds.hi[r] - bounds.lo[r])
                        end
                        rw_xp[r, w] = x
                    end
                    continue
                end
                for r in 1:nparams
                    x = positions[r, w] + s * rw_xp[r, w]
                    if bounds !== nothing && bounds.periodic[r]
                        x = bounds.lo[r] + mod(x - bounds.lo[r], bounds.hi[r] - bounds.lo[r])
                    end
                    rw_xp[r, w] = x
                end
            end
            t1 = time_ns(); rw_base!(rw_lbp, rw_xp); tacc!(:base, t1)
            @inbounds for w in 1:n
                isfinite(rw_lbp[w]) || (@views rw_xp[:, w] .= positions[:, w])
            end
            t1 = time_ns(); llk!(rw_llp, rw_xp); tacc!(:llk, t1)
            nevals += n
            nacc = 0
            @inbounds for w in 1:n
                (isfinite(rw_lbp[w]) && isfinite(rw_llp[w])) || continue
                lpn = rw_lbp[w] + beta * rw_llp[w]
                lpc = isfinite(ll[w]) ? rw_lb[w] + beta * ll[w] : T(-Inf)
                if !isfinite(lpc) || log(rand(rng, T)) < lpn - lpc
                    @views positions[:, w] .= rw_xp[:, w]
                    ll[w] = rw_llp[w]; rw_lb[w] = rw_lbp[w]
                    nacc += 1
                end
            end
            acc_sum += nacc
            for move! in rw_moves
                # caller-supplied symmetric proposals (docstring), plain MH against base + beta llk
                t1 = time_ns(); move!(rw_xp, positions, rng); tacc!(:moves, t1)
                t1 = time_ns(); rw_base!(rw_lbp, rw_xp); tacc!(:base, t1)
                @inbounds for w in 1:n
                    isfinite(rw_lbp[w]) || (@views rw_xp[:, w] .= positions[:, w])
                end
                mv_n[] += n
                _any_moved(rw_xp, positions, rw_lbp) || continue      # a no-op proposal: no likelihood call
                t1 = time_ns(); llk!(rw_llp, rw_xp); tacc!(:llk, t1)
                nevals += n
                @inbounds for w in 1:n
                    (isfinite(rw_lbp[w]) && isfinite(rw_llp[w])) || continue
                    lpn = rw_lbp[w] + beta * rw_llp[w]
                    lpc = isfinite(ll[w]) ? rw_lb[w] + beta * ll[w] : T(-Inf)
                    if !isfinite(lpc) || log(rand(rng, T)) < lpn - lpc
                        @views positions[:, w] .= rw_xp[:, w]
                        ll[w] = rw_llp[w]; rw_lb[w] = rw_lbp[w]
                        mv_acc[] += 1
                    end
                end
            end
            if mode_jumps && !isempty(mj_deltas)
                # learned mode jumps on base + beta llk; `ll` / `rw_lb` updated in place
                @inbounds for w in 1:n
                    mv_cur[w] = isfinite(ll[w]) ? rw_lb[w] + beta * ll[w] : T(-Inf)
                end
                mode_jump!(positions, mv_cur, ll, rw_lb, beta)
            end
            if proposal !== nothing
                # online-proposal independence MH (`independence_tries` per draw), as in `mutate!`
                if ind_llk !== nothing
                    xb = ind_batch!(proposal, positions)
                    nevals += nb_ind
                    @inbounds for t in 1:independence_tries, w in 1:n
                        j = (t - 1) * n + w
                        (isfinite(lb_b[j]) && isfinite(ll_b[j])) || continue
                        # (reference path: base log q0, llk logL + logprior_ref - logq0, as `rw_base!` / `llk!`)
                        lbj = reference === nothing ? lb_b[j] : rq_b[j]
                        llj = reference === nothing ? ll_b[j] : ll_b[j] + rp_b[j] - rq_b[j]
                        lpn = lbj + beta * llj
                        lpc = isfinite(ll[w]) ? rw_lb[w] + beta * ll[w] : T(-Inf)
                        if !isfinite(lpc) || log(rand(rng, T)) < (lpn - lq_b[j]) - (lpc - lqc_ind[w])
                            @views positions[:, w] .= xb[:, j]
                            ll[w] = llj; rw_lb[w] = lbj; lqc_ind[w] = lq_b[j]
                            ind_acc[] += 1
                        end
                    end
                    ind_tries[] += nb_ind
                end
                for _ in 1:(ind_llk === nothing ? independence_tries : 0)
                    t1 = time_ns(); xp = proposal.sample(rng, n); tacc!(:flow_sample, t1)
                    t1 = time_ns(); rw_base!(lb_ind, xp); tacc!(:base, t1)
                    @inbounds for w in 1:n
                        isfinite(lb_ind[w]) || (@views xp[:, w] .= positions[:, w])
                    end
                    t1 = time_ns(); llk!(ll_ind, xp); tacc!(:llk, t1)
                    nevals += n
                    t1 = time_ns()
                    proposal.logq_and_grad!(lqp_ind, gq_ind, xp)
                    proposal.logq_and_grad!(lqc_ind, gq_ind, positions)
                    tacc!(:flow_logq, t1)
                    @inbounds for w in 1:n
                        (isfinite(lb_ind[w]) && isfinite(ll_ind[w])) || continue
                        lpn = lb_ind[w] + beta * ll_ind[w]
                        lpc = isfinite(ll[w]) ? rw_lb[w] + beta * ll[w] : T(-Inf)
                        if !isfinite(lpc) || log(rand(rng, T)) < (lpn - lqp_ind[w]) - (lpc - lqc_ind[w])
                            @views positions[:, w] .= xp[:, w]
                            ll[w] = ll_ind[w]; rw_lb[w] = lb_ind[w]
                            ind_acc[] += 1
                        end
                    end
                    ind_tries[] += n
                end
            end
            if online_proposal !== nothing && ndraws === nothing
                for w in 1:n
                    push!(pool, positions[:, w])
                    push!(pool_ll, ll[w])
                end
                pool_beta[] = beta
            end
            if ndraws === nothing
                scale *= exp(clamp(T(nacc) / n - T(rw_target_accept), T(-1), T(0.5)))
                slowest = _slowest_move(move_stat, start, positions, metric)
                (k >= max_mutate || (k >= min_mutate && slowest[1] >= disp_target)) && break
            else
                k % thin == 0 && (chain[:, :, k ÷ thin] .= positions)
                k >= ndraws * thin && break
            end
        end
        @inbounds for w in 1:n
            logp[w] = isfinite(ll[w]) ? rw_lb[w] + beta * ll[w] : T(-Inf)
        end
        push!(accept_hist, acc_sum / (k * n))
        last_move[] = _slowest_move(move_stat, start, positions, metric)
        return k, scale, nevals
    end

    while beta < one(T)
        stage += 1
        empty!(prof); t_stage = time_ns()
        stage > max_stages && throw(ErrorException("smc_sample: exceeded max_stages=$max_stages (beta=$beta)"))
        any(isfinite, ll) || throw(ErrorException("smc_sample: no particle has a finite log-likelihood"))
        beta_new = _next_beta(ll, beta, T(target_cess))
        # `final_ess`: a stage that would stop short of beta = 1 -- in a hybrid, only once the NUTS stages
        # begin -- goes straight to 1 when the incremental weights' ESS / N there is still >= final_ess: no short last
        # stage (a NUTS stage costs about the same number of draws whatever its beta step)
        if final_ess !== nothing && beta_new < one(T) && (!hybrid || beta_new >= nuts_beta) &&
           _normalized_weights((one(T) - beta) .* ll)[2] >= T(final_ess)
            beta_new = one(T)
        end
        if persistent
            # Persistent sampling (Karamanis et al. 2025): resample from ALL stored populations, weighted by
            # multiple importance sampling against the equal mixture of their targets pi_s ∝ p0 L^beta_s / Z_s
            # (p0 = prior or reference; it cancels): w ∝ exp(beta_new r) / mean_s exp(beta_s r - log Z_s).
            push!(ps_pos, copy(positions)); push!(ps_r, copy(ll)); push!(ps_beta, beta); push!(ps_logZ, logZ)
            push!(ps_anc, copy(ancestor))
            Xp = reduce(hcat, ps_pos); Rp = reduce(vcat, ps_r); Ap = reduce(vcat, ps_anc)
            S = length(ps_beta)
            logw = similar(Rp)
            terms = Vector{T}(undef, S)
            for i in eachindex(Rp)
                if isfinite(Rp[i])
                    for j in 1:S
                        terms[j] = ps_beta[j] * Rp[i] - ps_logZ[j]
                    end
                    logw[i] = beta_new * Rp[i] - (_logsumexp(terms) - log(T(S)))
                else
                    logw[i] = T(-Inf)
                end
            end
            logZ = _logsumexp(logw) - log(T(length(logw)))
            W, _ = _normalized_weights(logw)
            metric = _smc_metric(Xp, _metric_weights(W, Rp, metric_core), dense, mass_blocks, metric)
            idx = _systematic_resample_n(rng, W, n)
            positions = Xp[:, idx]
            ll = Rp[idx]
            ancestor = Ap[idx]
        else
            logw = (beta_new - beta) .* ll
            logZ += _logsumexp(logw) - log(T(n))
            W, _ = _normalized_weights(logw)
            metric = _smc_metric(positions, _metric_weights(W, ll, metric_core), dense, mass_blocks, metric)
            idx = _systematic_resample(rng, W)
            positions = positions[:, idx]
            ll = ll[idx]
            ancestor = ancestor[idx]
        end
        beta = beta_new
        beta_ref[] = beta
        if hessian_metric > 0 && lpg! !== nothing
            fill!(betas, beta)
            metric, nh = _hessian_metric(metric, positions, ll, lpg!, hessian_metric, dense)
            total_leap += nh
        end
        if online_proposal !== nothing && !online_nuts && hybrid && beta_new >= nuts_beta
            # (`online_nuts = false`: no online proposal -- and no fit -- in a hybrid's NUTS stages)
            proposal = nothing
        elseif online_proposal !== nothing && !isempty(pool)
            # the previous stage's mutation states, weighted to this stage's beta (see `_online_build`)
            t1 = time_ns()
            proposal = something(_online_build(online_proposal, pool, pool_ll, beta - pool_beta[], online_pool_max, rng),
                                 Some(proposal))
            tacc!(:flow_fit, t1)
        end
        push!(schedule, beta)
        if rw && (!hybrid || beta < nuts_beta)
            empty!(pool); empty!(pool_ll)
            k, eps_, nl = rw_mutate!(nothing, 1, nothing, positions, ll, logp, metric, eps_, beta, proposal)
            total_ll += nl; last_k = k
        else
            t1 = time_ns()
            kern, keps, hsteps = stage_kernel(proposal, positions, metric, beta, x_set ? nuts_eps : nothing)
            nuts_started = true
            k, keps, nl, nb = mutate!(nothing, nothing, positions, ll, metric, keps, beta, proposal, kern)
            tacc!(:nuts, t1)
            kern.latent ? (lat_eps = keps) : (nuts_eps = keps; x_set = true)
            stage_latent = kern.latent
            total_leap += nl; block_evals += nb; total_ll += n
        end
        mj_added = (0, 0)
        if mode_jumps && beta >= mode_jump_learn_beta
            # learn from this stage's population; the next stage (and the recorded phase) uses the grown set
            t1 = time_ns()
            ka, kc = mode_jump_learn === :tails ? (0, 0) :
                     _mode_jump_learn!(mj_deltas, positions, jrows, cfg.bounds, mode_jump_k, T(mode_jump_min_cell), rng)
            ta, tq = mode_jump_learn === :kmeans ? (0, 0) :
                     _mode_jump_tails!(mj_deltas, positions, jrows, cfg.bounds, mj_tailq, T.(mode_jump_tail_frac),
                                       mode_jump_tail_min, mj_npcs)
            mj_added = (ka + ta, kc + tq)
            tacc!(:jump_learn, t1)
        end
        callback === nothing || callback(stage, (; beta, logZ, ndraws=k,
                                                 step_size=nuts_started ? (stage_latent ? lat_eps : nuts_eps) : eps_,
                                                 nuts=nuts_started, latent=stage_latent, heuristic_steps=hsteps,
                                                 accept=accept_hist[end], total_leap, total_ll,
                                                 slowest_move=last_move[][1], slowest_row=last_move[][2], block_evals,
                                                 independence_accept=ind_tries[] == 0 ? T(NaN) : ind_acc[] / ind_tries[],
                                                 de_accept=de_n[] == 0 ? T(NaN) : de_acc[] / de_n[],
                                                 move_accept=mv_n[] == 0 ? T(NaN) : mv_acc[] / mv_n[],
                                                 mode_jump_accept=mj_n[] == 0 ? T(NaN) : mj_acc[] / mj_n[],
                                                 mode_jumps=length(mj_deltas), mode_jumps_added=mj_added[1],
                                                 mode_jump_cells=mj_added[2], mode_jump_evals=mj_cols[],
                                                 unique=length(unique(ancestor)), positions, ll,
                                                 time_split=merge(copy(prof), Dict(:total => (time_ns() - t_stage) / 1e9))))
    end
    if online_proposal !== nothing && !isempty(pool) && (online_nuts || !hybrid)
        proposal = something(_online_build(online_proposal, pool, pool_ll, beta - pool_beta[], online_pool_max, rng),
                             Some(proposal))
    end
    # Final metric from the equilibrated beta = 1 cloud, then the recorded sampling phase.
    metric = _smc_metric(positions, _metric_weights(fill(one(T) / n, n), ll, metric_core), dense, mass_blocks, metric)
    if hessian_metric > 0 && lpg! !== nothing
        fill!(betas, beta)
        metric, nh = _hessian_metric(metric, positions, ll, lpg!, hessian_metric, dense)
        total_leap += nh
    end
    chain = Array{T}(undef, nparams, n, max(nfinal, 1))
    if nfinal > 0 && rw && !hybrid
        # One recorded state every `rw_thin` draws (default: the last ladder stage's draw count, i.e. about
        # `disp_target` population sds of movement between recorded states).
        _, _, nl = rw_mutate!(nfinal, max(something(rw_thin, last_k), 1), chain, positions, ll, logp, metric, eps_, beta,
                              proposal)
        total_ll += nl
    elseif nfinal > 0
        # (also a hybrid whose ladder ended below nuts_beta)
        kern, keps, _ = stage_kernel(proposal, positions, metric, beta, x_set ? nuts_eps : nothing)
        final_max_treedepth === nothing ||
            (kern = merge(kern, (; cfg=_with_max_treedepth(kern.cfg, Int(final_max_treedepth)))))
        nuts_started = true
        _, _, nl, nb = mutate!(nfinal, chain, positions, ll, metric, keps, beta, proposal, kern)
        kern.latent ? (lat_eps = keps) : (nuts_eps = keps)
        total_leap += nl; block_evals += nb; total_ll += n
    end
    nfinal == 0 && (chain[:, :, 1] .= positions)
    return (; positions, ll, logp, logZ, beta_schedule=schedule, chain, nstages=stage, block_evals,
            independence_accept=ind_tries[] == 0 ? T(NaN) : ind_acc[] / ind_tries[],
            de_accept=de_n[] == 0 ? T(NaN) : de_acc[] / de_n[],
            move_accept=mv_n[] == 0 ? T(NaN) : mv_acc[] / mv_n[],
            mode_jump_accept=mj_n[] == 0 ? T(NaN) : mj_acc[] / mj_n[], mode_jumps=length(mj_deltas),
            mode_jump_evals=mj_cols[],
            n_leapfrog=total_leap, n_loglik=total_ll, step_size=(rw && !hybrid) ? eps_ : nuts_eps, metric, accept=accept_hist,
            ancestors_unique=length(unique(ancestor)))
end
