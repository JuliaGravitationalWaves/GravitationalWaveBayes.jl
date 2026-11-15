module NUTSCore

export smc_sample, pt_position_swaps!, pt_adapt_ladder, pt_gaussian_barrier, pt_recommended_ntemps, nuts_sample!, NUTSConfig, DualAveraging, NUTSBounds, PTLadderState, pt_swap!, DenseMetric

using Random
using LinearAlgebra: I, dot, cholesky, Symmetric, UpperTriangular, LowerTriangular, Diagonal, eigen, isposdef, mul!
using KernelAbstractions
using Statistics: median!

# ---------------------------------------------------------------------------------------------
# Generic, GW-agnostic vectorized/masked NUTS core.
#
# This module knows nothing about gravitational waves, waveforms, or KernelAbstractions -- it
# operates purely on a caller-supplied `logp_and_grad!` callable with the interface:
#
#     logp_and_grad!(grad_out::AbstractMatrix{T}, logp_out::AbstractVector{T},
#                     positions::AbstractMatrix{T}, active_mask::AbstractVector{Bool}) -> nothing
#
# `positions` is (nparams, nwalkers); `grad_out`/`logp_out` are filled in place with the
# gradient/log-density for every walker column. Implementations are free to (and, for a batched
# GPU kernel, SHOULD) compute all columns unconditionally every call regardless of `active_mask`
# -- the mask is a hint for implementations that can exploit it, not a contract that inactive
# columns won't be touched. The tree-building loop below always calls `logp_and_grad!` with the
# full-width (nparams, nwalkers) arrays at every leapfrog step, and only *uses*/*updates* results
# for walkers currently marked active in its own bookkeeping -- this keeps the batch shape (and
# therefore e.g. a KernelAbstractions kernel launch's ndrange) constant across the whole tree
# build, which is what makes a single fused kernel call per step a real batching win instead of
# being defeated by shrinking/resizing the batch as walkers terminate at different times.
#
# Algorithm shape:
#   - Iterative (not recursive) doubling: at doubling depth d we extend the trajectory by 2^d
#     leapfrog steps in one randomly chosen direction, one leapfrog step at a time.
#   - Per-walker independent trajectories: each walker in the batch picks its own random
#     direction, its own U(0,1) draws for multinomial selection, and terminates independently.
#     A `subtree_active`/`active` boolean mask per walker tracks this; a terminated walker's
#     columns are simply not updated in the (still full-width) leapfrog/gradient calls.
#   - Generalized/multiplicative (Betancourt-style) U-turn criterion: rather than the 2011 paper's
#     recursive per-subtree check, we track a running sum of momenta ("rho") against the
#     trajectory's own endpoint momenta, both locally (within the subtree currently being built,
#     checked against its own first/latest leaf) and globally (the whole trajectory so far,
#     checked against theta_minus/theta_plus's momenta after each doubling merges in). This is
#     what makes iterative doubling tractable without an explicit recursive call stack.
#   - Multinomial (not slice) sampling: each new leaf is folded into the running candidate sample
#     via proportional progressive sampling (accept the new leaf with probability
#     weight_new/(weight_running_total)), and merging two subtrees uses the same rule with combined
#     log-weights (via log-sum-exp). This is exact proportional-to-exp(-H) multinomial sampling
#     over the whole trajectory regardless of how the merges are nested (each point ends up
#     selected with probability exactly weight_i/total_weight -- straightforward to verify by
#     induction on the merge tree), and IS the acceptance mechanism in modern NUTS -- there is no
#     separate binary MH accept/reject step layered on top; multinomial resampling over the
#     trajectory subsumes it. (Betancourt 2017 additionally describes a "biased" progressive
#     variant, `min(1, weight_new/weight_old)`, needed for a slice-restricted trajectory; since
#     this module uses plain multinomial weights throughout, not a slice, the simpler proportional
#     rule is the correct one here.)
# ---------------------------------------------------------------------------------------------

@inline function _logaddexp(a::T, b::T) where {T}
    (isinf(a) && a < 0) && return b
    (isinf(b) && b < 0) && return a
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end

"""
    NUTSBounds{T}(lo::AbstractVector{T}, hi::AbstractVector{T}; periodic=falses(length(lo)))

Per-parameter box constraints enforced DURING leapfrog integration (see the docstring
of `NUTSConfig` for why a hard `-Inf` wall alone is correctness-safe but wasteful).

`lo[i]`/`hi[i]` are parameter `i`'s bounds; use `T(-Inf)`/`T(Inf)` (either or both) to leave a row
unconstrained by this mechanism (it falls back to whatever the caller's `logp_and_grad!` itself
does -- e.g. this package's own `-Inf`-prior fallback in `BatchedNUTS.jl`, which stays in place
regardless, see that file's `_log_prior`).

`periodic[i]`: `true` for an angular coordinate whose `lo`/`hi` are identified (e.g. a phase or
right-ascension on `[0, 2*pi)`) -- handled by WRAPPING the position modulo `hi-lo` with the
momentum left untouched, not by reflecting. This is the standard treatment of a periodic/angular
HMC coordinate: `lo` and `hi` are the SAME point on the underlying circle, so there is no physical
wall there for the trajectory to bounce off -- wrapping is just using the coordinate's own
periodicity (working on its universal cover), and is trivially measure-preserving and reversible
since it does not touch the momentum at all. Reflecting a periodic coordinate instead would be
wrong: it would treat two identified points as a hard wall, folding the sampled distribution over
on itself. `false` (default) uses REFLECTION (see [`nuts_draw!`](@ref)'s leapfrog step): the
standard technique for a genuine hard box constraint (Neal, R.M. 2011, "MCMC Using Hamiltonian
Dynamics", in the Handbook of Markov Chain Monte Carlo (Brooks, Gelman, Jones & Meng, eds.),
Chapman & Hall/CRC -- the "bounce" construction for a constrained parameter: when a leapfrog step
would carry a coordinate past a boundary, reflect the coordinate back into the domain and negate
the momentum component normal to that boundary, i.e. treat the boundary as an idealized elastic
wall). See this module's comment above `_reflect_component` for why this exactly preserves the
Hamiltonian and the leapfrog map's reversibility/volume-preservation (the two properties NUTS's
correctness proof needs), and `test/test_nuts_core_bounds.jl` for an empirical check (a flat-target
"particle in a box" recovers a uniform stationary distribution, and a truncated-Gaussian target
recovers the correct truncated-normal moments) rather than relying on the argument alone.
"""
struct NUTSBounds{T}
    lo::Vector{T}
    hi::Vector{T}
    periodic::BitVector
end

function NUTSBounds(lo::AbstractVector{T}, hi::AbstractVector{T};
                     periodic::AbstractVector{Bool}=falses(length(lo))) where {T}
    length(lo) == length(hi) == length(periodic) ||
        throw(ArgumentError("lo, hi, periodic must all have the same length"))
    all(lo .<= hi) || throw(ArgumentError("NUTSBounds requires lo[i] <= hi[i] for every i"))
    # A PERIODIC row with
    # lo[i]==hi[i] (allowed by the `<=` check above) would make `_wrap_component`'s `mod(x-lo,
    # span)` divide by `span = hi-lo == 0`: Julia's
    # `mod(x, 0.0)` returns `NaN` for every `x`, so every leapfrog step touching that row would
    # silently NaN the whole walker's position from then on. Not reachable via this package's own
    # `BatchedNUTS._sampled_space_bounds` today (every periodic row -- phase/ra span `2*pi`, psi
    # spans `pi` -- has genuine positive width), so this is a latent constructor-level footgun for
    # a future direct `NUTSBounds` caller, not a live bug; guarded here as a cheap, non-algorithm-
    # changing safety check rather than left for `_wrap_component` to discover via NaN propagation.
    all(i -> !periodic[i] || lo[i] < hi[i], eachindex(lo)) ||
        throw(ArgumentError("NUTSBounds requires lo[i] < hi[i] (strict) for periodic rows"))
    return NUTSBounds{T}(Vector{T}(lo), Vector{T}(hi), BitVector(periodic))
end

"""
    NUTSConfig{T}(; max_treedepth=10, target_accept=T(0.8), max_delta_energy=T(1000), nwarmup=500,
                 bounds=nothing, adapt_mass=false)

Tuning parameters for [`nuts_sample!`](@ref).

- `max_treedepth`: cap on the doubling depth; a trajectory grows to at most `2^(max_treedepth+1)-1`
  leapfrog steps.
- `target_accept`: dual-averaging target acceptance statistic (`delta` in Hoffman & Gelman 2014
  Algorithm 6).
- `max_delta_energy`: divergence threshold on `|H - H0|`.
- `nwarmup`: number of draws (out of `nsteps` passed to `nuts_sample!`) during which the
  per-walker step size is adapted via dual averaging; the remaining draws use the frozen,
  averaged step size.
- `bounds`: `nothing` (default, no change from pre-Phase-5 behavior) or a [`NUTSBounds`](@ref)
  enforcing per-parameter box constraints by reflection/wrapping during leapfrog integration
  (see `NUTSBounds`'s docstring). A caller whose `logp_and_grad!` already returns `-Inf` outside
  some region (e.g. this package's `BatchedNUTS._log_prior`) does not NEED `bounds` for
  correctness -- NUTS already handles a `-Inf`/non-finite Hamiltonian as a divergence and discards
  the offending subtree (see `nuts_draw!`'s `subtree_valid` handling) -- but relying on that alone
  means every trajectory that would cross a tight bound gets truncated there instead of continuing
  to explore, which is wasteful when the bound is tight relative to the typical step size. Passing `bounds` lets the trajectory keep growing past what would
  otherwise be a dead end.
- `adapt_mass` (default `false`, exact pre-existing unit-mass behavior): adapt a DIAGONAL inverse
  mass matrix (one variance-like scale per parameter, shared by all walkers) during warmup, using
  Stan's windowed schedule -- see [`nuts_sample!`](@ref) and `_mass_adaptation_windows`. Needs
  `nwarmup >= 20`; with fewer warmup draws it is a silent no-op (Stan does the same).
- `dense_mass` (default `false`; needs `adapt_mass=true`): adapt a DENSE (full-covariance) inverse
  mass matrix instead of a diagonal one -- see [`DenseMetric`](@ref). Needed whenever the target has
  strong parameter correlations (every GW posterior does: chirp mass / mass ratio / spin / `t_c` /
  `phase`), because a diagonal metric can only rescale the axes and the step size is then still
  limited by the narrowest correlated direction. Not combinable with `metric_rank` (parallel
  tempering).
- `hessian_init` (default `false`; needs `dense_mass=true`): before warmup, bootstrap the dense metric
  from a finite-difference Hessian of the supplied gradient (see `_hessian_metric`), evaluated at every
  walker and pooled. Fixes the circular failure of purely sample-based adaptation from a badly-scaled
  start: if the initial step size is tiny relative to the posterior, the walkers barely move, the
  window variance measures that jitter, and the metric learns the scale of the failure rather than of
  the posterior (seen on GW150914: 300 draws, no mixing). Costs `2*nparams` batched gradient
  evaluations. `hessian_rel_step` (default `1e-6`) is the central-difference step as a fraction of
  each parameter's box width (`bounds`), or of `max(1, |x_i|)` for an unbounded row.
- `mass_init_buffer`, `mass_base_window`, `mass_term_buffer` (defaults `75`, `25`, `50`, Stan's
  schedule): the step-size-only initial buffer, the first slow window's length (later ones double)
  and the step-size-only terminal buffer, in warmup draws. Only used when `adapt_mass=true`; if they
  do not fit in `nwarmup` the schedule falls back to a 15%/75%/10% split. With `hessian_init` the
  initial buffer can be short (the metric is already sensible), so set it small to spend warmup on
  slow windows instead.
- `cull_logp_gap` (default `0` = off; needs `adapt_mass=true`, not for `metric_rank`): at the end of every
  slow window during WARMUP, any walker whose logp is more than this far below the best walker's is
  treated as stuck in a secondary local mode (HMC cannot hop between modes): it is left out of that
  window's metric estimate and re-seeded from a randomly chosen good walker (position, logp, step
  size). Never runs after warmup, so the sampled chain is untouched. On real
  IMRPhenomXAS/H1 data half the walkers, started at 2% of the box span around the base point, settled
  in modes ~165 logp units below the true one, for the diagonal and dense metric alike.
- `quorum_frac` (default `0` = off, exact pre-existing behavior): each tree-doubling round normally
  runs for EVERY walker until `any(active)` becomes false, i.e. until the LAST walker's trajectory
  stops growing -- one pathological walker (e.g. sitting on a stiff, poorly-whitened ridge) forces
  every other walker's tree to keep doubling too, since the batch advances one depth level at a
  time together (on real data, 100% of post-warmup draws hit `max_treedepth`). With
  `quorum_frac > 0`, once fewer than `ceil(quorum_frac * nwalkers)` walkers remain active (with a
  floor of `1`, so `quorum_frac=0` reproduces `any(active)` exactly), the round stops early and
  every still-active walker is administratively truncated at its current depth -- same handling as
  exhausting `max_treedepth`, just triggered by straggler count instead of depth. Surfaced per-walker
  as `stats.quorum_forced` (distinct from `diverged`: this is a scheduling cutoff, not a numerical
  failure). E.g. `quorum_frac=0.75` with 16 walkers stops once fewer than 12 remain active.
- `step_size_max_change` (default `8`, needs `>= 1`): bounds how far a single MID-WARMUP RESTART of
  `_find_reasonable_step_size!` (Stan's doubling/halving search from a single test leapfrog, re-run
  every time a metric-adaptation window ends and installs a new metric) may move `step_size` from
  its value going in, in EITHER direction. A metric-driven restart's single
  test leapfrog can give an unlucky accept/reject read for several doublings in a row, letting the
  search overshoot 30-45x before its own crossing criterion fires -- dual averaging then spends many
  expensive draws (each a full waveform-gradient batch) correcting back down, with a spike in
  divergence rate along the way. `8` bounds any one restart to at most 3 doublings/halvings from
  where it started; sacrifices some of the search's ability to jump straight to a very different
  step size in one restart, but that's the whole point -- multiple smaller corrections (each still
  gated by dual averaging afterward) are safer than one large jump from a single noisy leapfrog.
  Deliberately does NOT apply to the INITIAL `hessian_init` bootstrap call (before any draws): that
  call's whole point is to climb from whatever `step_size0` the caller supplied, which may
  legitimately be many orders of magnitude off (see "step-size heuristic lands near the right scale
  from either side" in `test_nuts_core_dense_mass.jl`) -- capping it there would defeat it, not
  hardened it. Only the mid-warmup restart's incoming step size is dual-averaging-tuned already, so
  a large jump there is the anomaly this option guards against, not the intended behavior.
- `mass_blocks` (default `nothing` = full dense; needs `dense_mass=true`): restrict the adapted
  DENSE metric to a block-diagonal structure, e.g. `[[1,2,6,7], [3,4]]` for a GW posterior's
  `mc`/`q`/`t_c`/`phase` ridge kept separate from `chi1`/`chi2` -- see `DenseMetric`'s own comment
  for why this needs no new sampling/kinetic/drift/reflection code: a block-diagonal SPD matrix's
  Cholesky factor is ITSELF block-diagonal, so `DenseMetric`'s existing machinery already decouples
  correctly once the matrix it's built from has no cross-block entries. `mass_blocks` only changes
  WHAT gets estimated -- every `Minv` this module estimates (Hessian bootstrap, window sample
  covariance) is masked to zero cross-block entries (`_block_mask!`) before becoming a
  `DenseMetric`, right before the one point that matters: masking a symmetric positive definite
  matrix down to its block-diagonal restriction is ALWAYS still positive definite (each diagonal
  block is a principal submatrix of a PD matrix, hence PD itself), so this never introduces a new
  `PosDefException` the unmasked estimate wouldn't already have risked. Indices not mentioned in
  any block are treated as their own independent singleton block (decoupled from everything,
  including each other) -- the same as a diagonal metric for that one parameter. Motivated by a
  real-data finding: full dense over 11 params from only 16
  walkers is data-starved (Rhat up to 228 for `t_c`) as well as curvature-limited; restricting to
  the correlations that actually matter needs fewer entries to estimate from the same data.
"""
struct NUTSConfig{T}
    max_treedepth::Int
    target_accept::T
    max_delta_energy::T
    nwarmup::Int
    bounds::Union{Nothing,NUTSBounds{T}}
    adapt_mass::Bool
    dense_mass::Bool
    hessian_init::Bool
    hessian_rel_step::T
    mass_init_buffer::Int
    mass_base_window::Int
    mass_term_buffer::Int
    cull_logp_gap::T
    quorum_frac::T
    mass_blocks::Union{Nothing,Vector{Vector{Int}}}
    step_size_max_change::T
end

function NUTSConfig{T}(; max_treedepth::Int=10, target_accept=T(0.8), max_delta_energy=T(1000), nwarmup::Int=500,
                        bounds::Union{Nothing,NUTSBounds{T}}=nothing, adapt_mass::Bool=false,
                        dense_mass::Bool=false, hessian_init::Bool=false, hessian_rel_step=T(1e-6),
                        mass_init_buffer::Int=75, mass_base_window::Int=25, mass_term_buffer::Int=50,
                        cull_logp_gap=T(0), quorum_frac=T(0),
                        mass_blocks::Union{Nothing,Vector{Vector{Int}}}=nothing,
                        step_size_max_change=T(8)) where {T}
    cull_logp_gap >= 0 || throw(ArgumentError("cull_logp_gap must be >= 0"))
    cull_logp_gap > 0 && !adapt_mass && throw(ArgumentError("cull_logp_gap > 0 requires adapt_mass=true (culling runs at metric-window ends)"))
    dense_mass && !adapt_mass && throw(ArgumentError("dense_mass=true requires adapt_mass=true"))
    hessian_init && !dense_mass && throw(ArgumentError("hessian_init=true requires dense_mass=true"))
    (quorum_frac >= 0 && quorum_frac <= 1) || throw(ArgumentError("quorum_frac must lie in [0, 1]"))
    step_size_max_change >= 1 || throw(ArgumentError("step_size_max_change must be >= 1"))
    if mass_blocks !== nothing
        dense_mass || throw(ArgumentError("mass_blocks requires dense_mass=true"))
        all(b -> !isempty(b) && all(>(0), b), mass_blocks) ||
            throw(ArgumentError("mass_blocks entries must be non-empty vectors of positive indices"))
        seen = Int[]
        for b in mass_blocks, i in b
            i in seen && throw(ArgumentError("mass_blocks index $i appears in more than one block"))
            push!(seen, i)
        end
    end
    NUTSConfig{T}(max_treedepth, T(target_accept), T(max_delta_energy), nwarmup, bounds, adapt_mass,
                  dense_mass, hessian_init, T(hessian_rel_step), mass_init_buffer, mass_base_window,
                  mass_term_buffer, T(cull_logp_gap), T(quorum_frac), mass_blocks, T(step_size_max_change))
end

# ---------------------------------------------------------------------------------------------
# Dual-averaging step-size adaptation (Hoffman & Gelman 2014, Algorithm 6).
#
# Design decision: step size is tracked PER WALKER (a length-nwalkers vector), not shared across
# the batch. This matches this codebase's convention of everything being batched per-walker
# (PEState's positions/momenta/logp/energy are all per-walker), and lets walkers that end up
# exploring differently-scaled regions (e.g. during early warmup, before they've all converged
# to the same typical set) adapt independently rather than being coupled to the average behavior
# of the batch. The tradeoff: nwalkers times the dual-averaging bookkeeping/state instead of one
# shared scalar, and no cross-walker sharing of information that could in principle speed up
# adaptation of a badly-initialized walker. Either choice is statistically valid here since all
# walkers target the same distribution; per-walker was chosen for consistency with the rest of
# the codebase, not because it's demonstrably better.
# ---------------------------------------------------------------------------------------------
mutable struct DualAveraging{T}
    mu::Vector{T}
    log_step::Vector{T}
    log_step_bar::Vector{T}
    h_bar::Vector{T}
    m::Int
    gamma::T
    t0::T
    kappa::T
end

function DualAveraging(init_step::AbstractVector{T}; gamma::T=T(0.05), t0::T=T(10), kappa::T=T(0.75)) where {T}
    mu = log.(10 .* init_step)
    DualAveraging{T}(mu, log.(init_step), zeros(T, length(init_step)), zeros(T, length(init_step)), 0, gamma, t0, kappa)
end

function update!(da::DualAveraging{T}, accept_stat::AbstractVector{T}, target::T) where {T}
    da.m += 1
    m = T(da.m)
    @. da.h_bar = (1 - 1 / (m + da.t0)) * da.h_bar + (1 / (m + da.t0)) * (target - accept_stat)
    @. da.log_step = da.mu - sqrt(m) / da.gamma * da.h_bar
    w = m^(-da.kappa)
    @. da.log_step_bar = w * da.log_step + (1 - w) * da.log_step_bar
    return exp.(da.log_step)
end

finalize_step(da::DualAveraging) = exp.(da.log_step_bar)

@inline kinetic_energy(r::AbstractVector{T}) where {T} = T(0.5) * dot(r, r)

# ---------------------------------------------------------------------------------------------
# Diagonal mass matrix. `inv_mass` is the diagonal of M^-1 (length nparams, shared by
# every walker) or `nothing` for the identity metric. The `nothing` methods are the ORIGINAL unit-
# mass code (same `dot`, no extra multiplies) so the default path is bit-for-bit unchanged.
#   momentum:   r ~ N(0, M)            -> r = randn / sqrt(inv_mass)
#   kinetic:    K(r) = 0.5 r' M^-1 r
#   drift:      theta += eps * M^-1 r
#   U-turn:     rho . (M^-1 r_edge)    (Betancourt 2017: criterion uses the velocity, not the momentum)
# Reflection/wrapping in `_apply_bounds!` stays valid: it only flips the sign of r_i, and both K and
# the drift are invariant/equivariant under that sign flip for a diagonal M.
# ---------------------------------------------------------------------------------------------
@inline _kinetic(r, ::Nothing) = kinetic_energy(r)
@inline function _kinetic(r::AbstractVector{T}, im::AbstractVector{T}) where {T}
    s = zero(T)
    @inbounds for i in eachindex(r)
        s += im[i] * r[i] * r[i]
    end
    return T(0.5) * s
end

@inline _mdot(a, b, ::Nothing) = dot(a, b)
@inline function _mdot(a::AbstractVector{T}, b::AbstractVector{T}, im::AbstractVector{T}) where {T}
    s = zero(T)
    @inbounds for i in eachindex(a)
        s += a[i] * im[i] * b[i]
    end
    return s
end

_draw_momentum(rng, ::Type{T}, nparams, nwalkers, ::Nothing) where {T} = randn(rng, T, nparams, nwalkers)
function _draw_momentum(rng, ::Type{T}, nparams, nwalkers, im::AbstractArray{T}) where {T}
    return randn(rng, T, nparams, nwalkers) ./ sqrt.(im)
end

# Walker `w`'s diagonal: the shared vector (or `nothing`) as-is, or column `w` of a per-walker
# (nparams x nwalkers) matrix -- the latter is how `nuts_sample!`'s `metric_rank` (per-temperature
# metric for parallel tempering) reaches the draw.
@inline _im_col(::Nothing, w) = nothing
@inline _im_col(im::AbstractVector, w) = im
@inline _im_col(im::AbstractMatrix, w) = view(im, :, w)

@inline _drift!(theta, r, eff_step, ::Nothing) = (theta .+= eff_step .* r; nothing)
@inline _drift!(theta, r, eff_step, im) = (theta .+= eff_step .* im .* r; nothing)

# ---------------------------------------------------------------------------------------------
# Dense (full-covariance) mass matrix. A diagonal metric can only rescale axes; a GW
# posterior's narrowest direction is a correlated combination (chirp mass / mass ratio / spin /
# `t_c` / `phase`) that no axis rescaling isolates, so the step size stays pinned by it and the
# trajectories need thousands of leapfrog steps to cross the broad directions (a GW150914
# run hit `max_treedepth` on 100% of draws and never moved). `DenseMetric` holds `Minv`
# = M^-1 (covariance-like, SPD) and its Cholesky factor `L` (`Minv = L L'`):
#   momentum:   r ~ N(0, M)         -> r = L'^-1 z, z ~ N(0, I)     (Cov = L'^-1 L^-1 = Minv^-1 = M)
#   kinetic:    K(r) = 0.5 r' Minv r
#   drift:      theta += eps * Minv r
#   U-turn:     rho . (Minv r_edge)
#   reflection: the elastic wall at x_i = c must conserve K, which a plain `r_i -> -r_i` does NOT once
#               Minv has off-diagonals. The K-preserving unit-determinant involution that flips the
#               velocity component v_i = (Minv r)_i is r -> r - 2 (v_i / Minv_ii) e_i
#               (see `_apply_bounds!`'s `DenseMetric` method). Periodic wrapping needs no change.
# ---------------------------------------------------------------------------------------------
"""
    DenseMetric(Minv::AbstractMatrix)

A dense inverse mass matrix (`Minv` symmetric positive definite -- the posterior covariance the
sampler should whiten by). Pass it as `nuts_sample!`'s `inv_mass0`; `result.inv_mass` is one when
`NUTSConfig(dense_mass=true)` was used. Throws `PosDefException` if `Minv` is not positive definite.
"""
struct DenseMetric{T}
    Minv::Matrix{T}
    L::Matrix{T}     # lower Cholesky factor: Minv = L * L'
end

function DenseMetric(Minv::AbstractMatrix{T}) where {T}
    A = Matrix{T}(Symmetric(Matrix{T}(Minv)))
    return DenseMetric{T}(A, Matrix{T}(cholesky(Symmetric(A)).L))
end

# `NUTSConfig.mass_blocks`: zero every `A[i,j]` where `i`/`j` fall in different blocks (an index
# not mentioned in any block is its own singleton block, decoupled from everything). See
# `NUTSConfig`'s own `mass_blocks` docstring for why this is always safe to apply to a symmetric
# positive definite `A` (the result is still PD) and why `DenseMetric` needs no other changes to
# support it (a block-diagonal SPD matrix's Cholesky factor is itself block-diagonal, so momentum
# draw/kinetic energy/drift/U-turn/reflection all decouple correctly with zero new code).
_block_mask(A::AbstractMatrix, ::Nothing) = A
function _block_mask(A::AbstractMatrix{T}, blocks::Vector{Vector{Int}}) where {T}
    n = size(A, 1)
    block_of = zeros(Int, n)
    for (k, b) in enumerate(blocks), i in b
        block_of[i] = k
    end
    next_singleton = length(blocks)
    for i in 1:n
        if block_of[i] == 0
            next_singleton += 1
            block_of[i] = next_singleton
        end
    end
    B = copy(Matrix{T}(A))
    @inbounds for j in 1:n, i in 1:n
        block_of[i] == block_of[j] || (B[i, j] = zero(T))
    end
    return B
end

_draw_momentum(rng, ::Type{T}, nparams, nwalkers, dm::DenseMetric{T}) where {T} =
    UpperTriangular(collect(dm.L')) \ randn(rng, T, nparams, nwalkers)

@inline function _mdot(a::AbstractVector{T}, b::AbstractVector{T}, dm::DenseMetric{T}) where {T}
    s = zero(T)
    @inbounds for j in eachindex(b)
        t = zero(T)
        for i in eachindex(a)
            t += a[i] * dm.Minv[i, j]
        end
        s += t * b[j]
    end
    return s
end
@inline _kinetic(r::AbstractVector{T}, dm::DenseMetric{T}) where {T} = T(0.5) * _mdot(r, r, dm)
@inline _im_col(dm::DenseMetric, w) = dm
@inline _drift!(theta, r, eff_step, dm::DenseMetric) = (theta .+= eff_step .* (dm.Minv * r); nothing)

# Whitened length of a proposed move `d` (sqrt(d' M d), M = Minv^-1): ~1 means "moved about one
# posterior standard deviation". Used for the per-draw `disp` diagnostic; a chain whose `disp` sits
# orders of magnitude below 1 is frozen, whatever its acceptance rate says.
_whitened_norm(d::AbstractVector{T}, ::Nothing) where {T} = sqrt(dot(d, d))
_whitened_norm(d::AbstractVector{T}, im::AbstractVector{T}) where {T} = sqrt(sum(abs2(d[i]) / im[i] for i in eachindex(d)))
_whitened_norm(d::AbstractVector{T}, dm::DenseMetric{T}) where {T} = sqrt(sum(abs2, LowerTriangular(dm.L) \ d))

function _whitened_displacement(before::AbstractMatrix{T}, after::AbstractMatrix{T}, inv_mass) where {T}
    return [_whitened_norm(after[:, w] .- before[:, w], _im_col(inv_mass, w)) for w in axes(after, 2)]
end

# ---------------------------------------------------------------------------------------------
# Windowed diagonal-mass adaptation schedule (Stan reference manual, "HMC algorithm parameters" /
# "Automatic parameter tuning"): an initial fast buffer (step size only), then "slow" windows whose
# lengths double, each ending in a metric update (and a dual-averaging restart), then a terminal
# fast buffer (step size only, with the now-final metric). Returns the slow windows as 1-based
# warmup-draw index ranges, or an empty vector when `nwarmup < 20` (no metric adaptation, as in Stan).
# ---------------------------------------------------------------------------------------------
function _mass_adaptation_windows(nwarmup::Int; init_buffer::Int=75, term_buffer::Int=50, base_window::Int=25)
    nwarmup < 20 && return UnitRange{Int}[]
    if init_buffer + base_window + term_buffer > nwarmup
        init_buffer = floor(Int, 0.15 * nwarmup)
        term_buffer = floor(Int, 0.10 * nwarmup)
        base_window = nwarmup - init_buffer - term_buffer
    end
    windows = UnitRange{Int}[]
    wstart = init_buffer + 1
    wsize = base_window
    last_slow = nwarmup - term_buffer
    while wstart <= last_slow
        wend = wstart + wsize - 1
        # If the NEXT (doubled) window would not fit before the terminal buffer, stretch this one
        # to absorb the remainder (Stan's behaviour), rather than leaving a short useless window.
        if wend + 2 * wsize > last_slow
            wend = last_slow
        end
        wend = min(wend, last_slow)
        push!(windows, wstart:wend)
        wstart = wend + 1
        wsize *= 2
    end
    return windows
end

# Pooled-within-walker running variance over one slow window: per-walker Welford (mean, M2) over
# that walker's own draws, combined at window end by averaging the per-walker variances. Within-
# walker (not total) variance is used deliberately: if walkers are NOT yet mixed (the situation
# this adaptation exists to help with), between-walker spread would inflate the scale estimate.
mutable struct _WindowVariance{T}
    n::Int
    mean::Matrix{T}
    m2::Matrix{T}
end
_WindowVariance(::Type{T}, nparams, nwalkers) where {T} =
    _WindowVariance{T}(0, zeros(T, nparams, nwalkers), zeros(T, nparams, nwalkers))

function _accumulate!(wv::_WindowVariance{T}, x::AbstractMatrix{T}) where {T}
    wv.n += 1
    @. wv.m2 += (x - wv.mean) * (x - (wv.mean + (x - wv.mean) / wv.n))
    @. wv.mean += (x - wv.mean) / wv.n
    return nothing
end

# Per-column (NOT pooled) variance, for the per-temperature-rank metric: each column is one rank's
# own stream, and ranks target different tempered densities, so averaging them would be wrong.
function _column_variance(wv::_WindowVariance{T}) where {T}
    wv.n < 2 && return zeros(T, size(wv.m2))
    return wv.m2 ./ (wv.n - 1)
end

function _pooled_variance(wv::_WindowVariance{T}; walkers=1:size(wv.m2, 2)) where {T}
    nparams = size(wv.m2, 1)
    v = zeros(T, nparams)
    wv.n < 2 && return v
    @inbounds for w in walkers, i in 1:nparams
        v[i] += wv.m2[i, w] / (wv.n - 1)
    end
    return v ./ length(walkers)
end

# Dense analogue of `_WindowVariance`: per-walker Welford mean and cross-product matrix.
mutable struct _WindowCov{T}
    n::Int
    mean::Matrix{T}       # nparams x nwalkers
    m2::Array{T,3}        # nparams x nparams x nwalkers
end
_WindowCov(::Type{T}, nparams, nwalkers) where {T} =
    _WindowCov{T}(0, zeros(T, nparams, nwalkers), zeros(T, nparams, nparams, nwalkers))

function _accumulate!(wc::_WindowCov{T}, x::AbstractMatrix{T}) where {T}
    wc.n += 1
    @inbounds for w in axes(x, 2)
        d1 = x[:, w] .- wc.mean[:, w]
        wc.mean[:, w] .+= d1 ./ wc.n
        d2 = x[:, w] .- wc.mean[:, w]
        wc.m2[:, :, w] .+= d1 * d2'
    end
    return nothing
end

# Pooled within-walker covariance (same deliberate within-walker choice as `_WindowVariance`),
# regularized with weight `lam = 5/(n+5)` (Stan's idea). WHAT it is shrunk toward matters a great
# deal here: shrinking toward `diag(estimate)` (Stan-style) adds ~`lam * variance` to EVERY
# eigenvalue and so erases a narrow correlated direction (a GW posterior has eigenvalues 8+ decades
# apart; the first version of this function turned an exact Hessian metric with a 1e-8 eigenvalue
# into one with a ~1e-2 floor and froze the chain). So when the previous metric `prior` is
# informed (Hessian bootstrap, an earlier window, or a caller-supplied one) the shrinkage is
# applied in the WHITENED frame of `prior` -- `L^-1 S L^-T` shrunk toward `I` -- which preserves
# `prior`'s conditioning. Only with no informed prior does it fall back to shrinking toward
# `diag(estimate)`. Returns `nothing` if the estimate is unusable (too few draws, non-finite,
# non-positive diagonal, or not positive definite).
function _pooled_covariance(wc::_WindowCov{T}; prior::Union{Nothing,DenseMetric{T}}=nothing,
                            walkers=1:size(wc.m2, 3)) where {T}
    wc.n < 3 && return nothing
    nparams = size(wc.m2, 1)
    S = zeros(T, nparams, nparams)
    @inbounds for w in walkers
        S .+= wc.m2[:, :, w] ./ (wc.n - 1)
    end
    S ./= length(walkers)
    all(isfinite, S) || return nothing
    all(i -> S[i, i] > 0, 1:nparams) || return nothing
    lam = T(5) / (wc.n + 5)
    if prior === nothing
        D = zero(S)
        for i in 1:nparams
            D[i, i] = S[i, i]
        end
        S = (1 - lam) .* S .+ lam .* D
    else
        Lp = LowerTriangular(prior.L)
        C = Lp \ (S / Lp')                       # sample covariance in prior's whitened frame
        C = (1 - lam) .* C .+ lam .* Matrix{T}(I, nparams, nparams)
        S = Lp * C * Lp'
    end
    S = (S .+ S') ./ 2
    isposdef(Symmetric(S)) || return nothing
    return S
end

# ---------------------------------------------------------------------------------------------
# Finite-difference Hessian bootstrap of the dense metric (`NUTSConfig.hessian_init`).
#
# Sample-based adaptation is circular from a bad start: the metric it learns is the scale of the
# chain's own motion, which is whatever the initial step size allowed. This estimates the local
# curvature of `logp` directly instead, by central-differencing the SUPPLIED GRADIENT along each
# coordinate (column j of the Hessian = (g(x + h_j e_j) - g(x - h_j e_j)) / 2 h_j). One batched
# gradient call per perturbation covers every walker at once (2*nparams calls total), so each
# walker yields its own Hessian at its own position. Those are pooled by an ELEMENTWISE MEDIAN
# (robust to a walker that happens to sit somewhere with atypically large curvature), then
# eigen-projected onto a positive definite matrix: eigenvalues -> |lambda|, floored at
# `max(1e-12*max|lambda|, 1/max_span^2)` (a flat/unconstrained direction should not get a metric
# wider than the parameter box). The result is precision P = -H; the metric is Minv = P^-1.
# It is only a preconditioner: `logp` is not quadratic away from the mode, and later slow windows
# replace it with a sample covariance once the walkers are moving.
# Returns `(metric, nvalid)`; `metric === nothing` if no walker gave a finite Hessian.
# ---------------------------------------------------------------------------------------------
function _hessian_metric(logp_and_grad!::F, positions::AbstractMatrix{T}, cfg::NUTSConfig{T};
                         walkers=nothing) where {T,F}
    nparams, nwalkers = size(positions)
    b = cfg.bounds
    h = Vector{T}(undef, nparams)
    max_span = zero(T)
    for i in 1:nparams
        if b !== nothing && isfinite(b.lo[i]) && isfinite(b.hi[i])
            h[i] = cfg.hessian_rel_step * (b.hi[i] - b.lo[i])
            max_span = max(max_span, b.hi[i] - b.lo[i])
        else
            h[i] = cfg.hessian_rel_step * max(one(T), abs(sum(positions[i, :]) / nwalkers))
        end
    end
    mask = trues(nwalkers)
    gp = similar(positions); gm = similar(positions)
    lpp = Vector{T}(undef, nwalkers); lpm = Vector{T}(undef, nwalkers)
    Hs = zeros(T, nparams, nparams, nwalkers)
    valid = trues(nwalkers)
    X = similar(positions)
    for j in 1:nparams
        X .= positions; X[j, :] .+= h[j]
        logp_and_grad!(gp, lpp, X, mask)
        X .= positions; X[j, :] .-= h[j]
        logp_and_grad!(gm, lpm, X, mask)
        for w in 1:nwalkers
            col = (gp[:, w] .- gm[:, w]) ./ (2 * h[j])
            if !(isfinite(lpp[w]) && isfinite(lpm[w]) && all(isfinite, col))
                valid[w] = false
            else
                Hs[:, j, w] .= col
            end
        end
    end
    walkers === nothing || (valid .&= in.(1:nwalkers, Ref(Set(walkers))))
    nvalid = count(valid)
    nvalid == 0 && return nothing, 0
    P = zeros(T, nparams, nparams)
    idx = findall(valid)
    for i in 1:nparams, j in 1:nparams
        vals = sort!([-(Hs[i, j, w] + Hs[j, i, w]) / 2 for w in idx])
        m = length(vals)
        P[i, j] = isodd(m) ? vals[(m + 1) ÷ 2] : (vals[m ÷ 2] + vals[m ÷ 2 + 1]) / 2
    end
    P = (P .+ P') ./ 2
    F_ = eigen(Symmetric(P))
    lam = abs.(F_.values)
    floor_ = max(T(1e-12) * maximum(lam), max_span > 0 ? one(T) / max_span^2 : zero(T))
    lam = max.(lam, floor_)
    all(isfinite, lam) && all(>(0), lam) || return nothing, nvalid
    Sigma = F_.vectors * Diagonal(one(T) ./ lam) * F_.vectors'
    Sigma = _block_mask((Sigma .+ Sigma') ./ 2, cfg.mass_blocks)
    return DenseMetric(Sigma), nvalid
end

# ---------------------------------------------------------------------------------------------
# Stan's `find_reasonable_step_size`, batched: per walker, one leapfrog step from the current
# position with a fresh momentum; double the step while the single-step acceptance ratio stays above
# 1/2, halve it while it stays below, stop at the crossing. Dual averaging alone takes ~one draw per
# factor ~2 to climb from a tiny step size, and a draw here costs minutes (each leapfrog is a full
# waveform-gradient evaluation), so after a metric change this is far cheaper than letting it climb.
# Mutates `step_size`; every iteration is one batched gradient evaluation.
#
# `max_change` (kwarg, NOT read from `cfg` -- see call sites in `nuts_sample!`) optionally bounds
# the total multiplicative move from the step size this function was called with: a single test
# leapfrog's accept/reject is a noisy one-sample read of the crossing point, and right after a
# metric change it can drift unluckily for several doublings in a row before the loop's own
# crossing criterion fires -- observed as a 30-45x overshoot on real data. The search freezes (marks the walker `done`) once it would move
# `step_size` past the bound, in either direction, rather than let it run further on that one
# noisy sample. Left uncapped (`Inf`) by default: the INITIAL call (before any draws, bootstrapping
# from whatever `step_size0` the caller supplied, possibly many orders of magnitude off on purpose
# -- see "step-size heuristic lands near the right scale from either side" in
# `test_nuts_core_dense_mass.jl`) still needs to be able to climb arbitrarily far in one call; only
# the mid-warmup RESTART call site (after a metric window ends, where the incoming `step_size` was
# already dual-averaging-tuned by real draws and a large jump is the anomaly, not the intent) passes
# `cfg.step_size_max_change`.
# ---------------------------------------------------------------------------------------------
function _find_reasonable_step_size!(step_size::AbstractVector{T}, positions::AbstractMatrix{T},
                                      logp_and_grad!::F, cfg::NUTSConfig{T}, rng::AbstractRNG,
                                      inv_mass; maxiter::Int=40, max_change::T=T(Inf)) where {T,F}
    nparams, nwalkers = size(positions)
    mask = trues(nwalkers)
    grad0 = similar(positions); logp0 = Vector{T}(undef, nwalkers)
    logp_and_grad!(grad0, logp0, positions, mask)
    r0 = _draw_momentum(rng, T, nparams, nwalkers, inv_mass)
    H0 = [-logp0[w] + _kinetic(view(r0, :, w), _im_col(inv_mass, w)) for w in 1:nwalkers]
    dir = zeros(Int, nwalkers)
    done = .!isfinite.(H0)             # a walker with non-finite logp cannot be tuned; leave it
    step_size0 = copy(step_size)       # pre-search value; bounds how far this call may move step_size
    lo_bound = step_size0 ./ max_change
    hi_bound = step_size0 .* max_change
    theta1 = similar(positions); r1 = similar(r0); grad1 = similar(grad0); logp1 = Vector{T}(undef, nwalkers)
    logthresh = log(T(0.5))
    for _ in 1:maxiter
        all(done) && break
        for w in 1:nwalkers
            @views r1[:, w] .= r0[:, w] .+ T(0.5) .* step_size[w] .* grad0[:, w]
            @views theta1[:, w] .= positions[:, w]
            _drift_and_bound!(view(theta1, :, w), view(r1, :, w), step_size[w], _im_col(inv_mass, w), cfg.bounds)
        end
        logp_and_grad!(grad1, logp1, theta1, mask)
        for w in 1:nwalkers
            done[w] && continue
            @views r1[:, w] .+= T(0.5) .* step_size[w] .* grad1[:, w]
            H1 = -logp1[w] + _kinetic(view(r1, :, w), _im_col(inv_mass, w))
            good = isfinite(H1) && (H0[w] - H1) > logthresh
            dir[w] == 0 && (dir[w] = good ? 1 : -1)
            if dir[w] == 1 && !good
                step_size[w] /= 2; done[w] = true
            elseif dir[w] == -1 && good
                done[w] = true
            else
                step_size[w] *= dir[w] == 1 ? T(2) : T(0.5)
            end
            # `step_size_max_change` cap: freeze here rather than let a single test leapfrog's
            # accept/reject read drive the search further past a bounded multiple of step_size0.
            if step_size[w] >= hi_bound[w]
                step_size[w] = hi_bound[w]; done[w] = true
            elseif step_size[w] <= lo_bound[w]
                step_size[w] = lo_bound[w]; done[w] = true
            end
        end
    end
    return step_size
end

# ---------------------------------------------------------------------------------------------
# Boundary handling for leapfrog integration. Applied to a single (theta, r) SCALAR
# component immediately after that component's position has been drift-updated within one
# leapfrog half-step, i.e. exactly the point in `nuts_draw!`'s inner loop where a naive
# implementation would let `theta_work[i, w]` wander outside `[lo, hi]` until the NEXT
# `logp_and_grad!` call reports `-Inf` (see `NUTSBounds`'s docstring for why that alone is
# correctness-safe but wasteful, not why it's wrong).
#
# Why reflection preserves what NUTS needs (not just "it compiles" -- see also
# `test/test_nuts_core_bounds.jl` for the empirical check): the leapfrog step this augments is a
# standard symplectic (volume-preserving, time-reversible) map on (theta, r). Reflection off a
# hard wall at `lo`/`hi` composes an ADDITIONAL orthogonal linear map onto the raw drifted point:
# fold `x -> 2*hi - x` (or `x -> 2*lo - x`) together with `r -> -r`. Three facts make this
# composition still valid for NUTS:
#   1. Hamiltonian-preserving: the kinetic energy term is `0.5*r^2` (`kinetic_energy` above),
#      which is invariant under `r -> -r` -- reflecting changes no walker's `H = -logp + KE`
#      at all (position is unchanged in magnitude from the wall, and `logp` only depends on
#      the reflected/in-bounds `theta`, not on which side of the wall the raw drift landed).
#   2. Volume-preserving: a coordinate flip `r -> -r` and a fold `x -> 2c - x` are each
#      individually an orthogonal (indeed involutive) linear map with Jacobian determinant
#      exactly `-1`; the reflecting-into-an-interval map (possibly folding more than once, for
#      a step so large it would cross the interval width) is a finite composition of these,
#      so it preserves Lebesgue measure on (theta, r) exactly, same as the un-reflected leapfrog
#      step already does.
#   3. Reversible: the fold map is a literal involution (`fold(fold(x)) == x` algebraically);
#      composed with the momentum-negation NUTS/leapfrog already relies on for its own
#      reversibility proof (running a step backward means negating momentum, applying the same
#      step, and negating again), the reflected step remains its own correct inverse under
#      that same recipe. Physically this is exactly the elastic-wall/billiard-ball construction
#      classical mechanics uses for a "particle in a box" -- not a novel trick.
#
# This is the reason a NAIVE "clamp `theta` to `[lo,hi]` and continue with the OLD, un-flipped
# momentum" is NOT an equivalent fix (the task's own caution: "a naive clip-and-continue breaks
# the algorithm's correctness in ways a bare boundary check might not obviously reveal"): clamping
# alone discards the momentum's kinetic energy that carried the walker past the wall without
# giving it anywhere to go, which is neither Hamiltonian-preserving (H changes) nor reversible
# (there is no way to recover the pre-clamp state by reversing momentum and re-integrating) --
# it introduces a spurious drift toward the boundary instead of a true bounce.
# ---------------------------------------------------------------------------------------------

@inline function _reflect_component(x::T, r::T, lo::T, hi::T) where {T}
    guard = 0
    while x > hi || x < lo
        if x > hi
            x = 2hi - x
            r = -r
        elseif x < lo
            x = 2lo - x
            r = -r
        end
        guard += 1
        # Safety valve only: with a finite step size and finite (lo,hi), a handful of folds
        # always suffices (at most ceil(overshoot / (hi-lo)) of them). This guards only against
        # a pathological/misconfigured call (e.g. lo==hi, or a NaN/Inf step) spinning forever;
        # it is not part of the normal reflecting-boundary path and is not expected to trigger
        # in any of this module's tests.
        if guard > 10_000
            x = clamp(x, lo, hi)
            break
        end
    end
    return x, r
end

@inline function _wrap_component(x::T, lo::T, hi::T) where {T}
    # Periodic coordinate: `lo` and `hi` are the same point on the circle, so wrap via `mod`
    # rather than reflect (see `NUTSBounds`'s docstring). Momentum is untouched.
    span = hi - lo
    return lo + mod(x - lo, span)
end

"""
    _apply_bounds!(theta::AbstractVector{T}, r::AbstractVector{T}, b::NUTSBounds{T}) where {T}

In-place per-component boundary enforcement for one walker's drifted position `theta` and
momentum `r` (both length `nparams`), called once per leapfrog half-step from `nuts_draw!` when
`cfg.bounds !== nothing`. Rows with non-finite `lo`/`hi` (either or both) are left untouched
(unconstrained by this mechanism). See `NUTSBounds`'s docstring for reflect-vs-wrap semantics.
"""
function _apply_bounds!(theta::AbstractVector{T}, r::AbstractVector{T}, b::NUTSBounds{T}) where {T}
    @inbounds for i in eachindex(theta)
        lo_i = b.lo[i]
        hi_i = b.hi[i]
        (isfinite(lo_i) && isfinite(hi_i)) || continue
        if b.periodic[i]
            theta[i] = _wrap_component(theta[i], lo_i, hi_i)
        else
            theta[i], r[i] = _reflect_component(theta[i], r[i], lo_i, hi_i)
        end
    end
end

# Position drift + boundary handling for one walker's leapfrog half-step, as `nuts_draw!` needs it.
# Unit / diagonal metric: the original two-step "drift, then fold back" -- exactly reversible there
# because a fold of coordinate i commutes with the drift of coordinate i and touches no other.
_drift_and_bound!(theta, r, eff_step, im, ::Nothing) = _drift!(theta, r, eff_step, im)
function _drift_and_bound!(theta, r, eff_step, im, b::NUTSBounds)
    _drift!(theta, r, eff_step, im)
    _apply_bounds!(theta, r, b)
    return nothing
end
_drift_and_bound!(theta, r, eff_step, dm::DenseMetric, ::Nothing) = _drift!(theta, r, eff_step, dm)

# Dense metric: that two-step version is NOT reversible (a dense Minv couples the coordinates, so
# folding coordinate i does not commute with the drift of coordinate j != i -- observed as a ~15%
# variance bias on a truncated correlated Gaussian). The exact fix is the billiard: follow the
# straight-line drift theta + s*eps*v (v = Minv r) only until the first wall is reached, flip the
# velocity component normal to that wall with the K-conserving unit-determinant involution
# r -> r - 2 (v_i/Minv_ii) e_i (v_i -> -v_i; other velocity components change through Minv's
# off-diagonals), and continue the REMAINING part of the step with the new velocity. Periodic rows
# just drift and wrap at the end (no wall; wrapping commutes with the drift).
function _drift_and_bound!(theta::AbstractVector{T}, r::AbstractVector{T}, eff_step::T, dm::DenseMetric{T},
                           b::NUTSBounds{T}) where {T}
    n = length(theta)
    v = Vector{T}(undef, n)
    rem = one(T)
    for _ in 1:(16 * n)                        # each pass consumes one wall hit; guard against pathologies
        mul!(v, dm.Minv, r)
        s_hit = rem
        i_hit = 0
        @inbounds for i in 1:n
            (b.periodic[i] || !(isfinite(b.lo[i]) && isfinite(b.hi[i]))) && continue
            dx = rem * eff_step * v[i]
            target = theta[i] + dx
            if target > b.hi[i] && dx > 0
                s = max((b.hi[i] - theta[i]) / (eff_step * v[i]), zero(T))
            elseif target < b.lo[i] && dx < 0
                s = max((b.lo[i] - theta[i]) / (eff_step * v[i]), zero(T))
            else
                continue
            end
            if s < s_hit
                s_hit = s
                i_hit = i
            end
        end
        if i_hit == 0
            theta .+= (rem * eff_step) .* v
            break
        end
        theta .+= (s_hit * eff_step) .* v
        theta[i_hit] = v[i_hit] * eff_step > 0 ? b.hi[i_hit] : b.lo[i_hit]    # exactly on the wall
        r[i_hit] -= 2 * v[i_hit] / dm.Minv[i_hit, i_hit]
        rem -= s_hit
    end
    @inbounds for i in 1:n
        b.periodic[i] && isfinite(b.lo[i]) && isfinite(b.hi[i]) && (theta[i] = _wrap_component(theta[i], b.lo[i], b.hi[i]))
    end
    return nothing
end

# ---------------------------------------------------------------------------------------------
# Single NUTS draw: builds one trajectory (iterative doubling) for the WHOLE batch of walkers at
# once, updating `positions` in place to the multinomially-sampled new state for every walker,
# and returning each walker's mean Metropolis-style acceptance statistic (for dual averaging).
# ---------------------------------------------------------------------------------------------
function nuts_draw!(positions::AbstractMatrix{T}, logp_out::AbstractVector{T},
                     step_size::AbstractVector{T}, logp_and_grad!::F,
                     cfg::NUTSConfig{T}, rng::AbstractRNG,
                     inv_mass::Union{Nothing,AbstractVector{T},AbstractMatrix{T},DenseMetric{T}}=nothing) where {T,F}
    nparams, nwalkers = size(positions)

    r0 = _draw_momentum(rng, T, nparams, nwalkers, inv_mass)
    grad0 = similar(positions)
    logp0 = Vector{T}(undef, nwalkers)
    logp_and_grad!(grad0, logp0, positions, trues(nwalkers))

    H0 = Vector{T}(undef, nwalkers)
    @inbounds for w in 1:nwalkers
        H0[w] = -logp0[w] + _kinetic(view(r0, :, w), _im_col(inv_mass, w))
    end

    theta_minus = copy(positions); theta_plus = copy(positions)
    r_minus = copy(r0);            r_plus = copy(r0)
    grad_minus = copy(grad0);      grad_plus = copy(grad0)

    candidate = copy(positions)
    candidate_logp = copy(logp0)
    log_weight_total = zeros(T, nwalkers)   # initial point has log-weight log(exp(H0-H0)) = 0
    rho_total = copy(r0)

    active = trues(nwalkers)
    accept_stat_sum = zeros(T, nwalkers)
    accept_stat_count = zeros(Int, nwalkers)

    # Reused scratch buffers for the leapfrog integration (full width every step; see module
    # docstring for why the batch is never resized).
    theta_work = similar(positions)
    r_work = similar(r0)
    grad_work = similar(grad0)
    logp_work = Vector{T}(undef, nwalkers)
    eff_step = Vector{T}(undef, nwalkers)
    v = Vector{Int}(undef, nwalkers)

    # Per-draw diagnostics: scalars this function computes internally, surfaced to
    # `nuts_sample!`'s caller via its return value / the `callback` hook, instead of having to be
    # inferred indirectly from draw wall-clock timing.
    # `depth_reached[w]`: the last doubling depth whose subtree this walker's trajectory actually
    # incorporated (see the assignment sites below and the fixup after the loop for the
    # "exhausted max_treedepth" case). `n_leapfrog[w]`: total leapfrog integrator calls (gradient
    # evaluations) across the whole draw -- the real per-draw cost driver. `diverged[w]`: true iff
    # a non-finite or `max_delta_energy`-exceeding Hamiltonian was hit anywhere in this draw's tree
    # (as opposed to ordinary U-turn termination) -- a genuine numerical-error signal, not just
    # "the trajectory stopped growing".
    depth_reached = fill(-1, nwalkers)
    n_leapfrog = zeros(Int, nwalkers)
    diverged = falses(nwalkers)

    # `quorum_count`: minimum number of still-active walkers required to keep deepening the WHOLE
    # batch's tree (see `NUTSConfig.quorum_frac`'s docstring). Floored at 1 so `quorum_frac=0`
    # reproduces `any(active)` exactly -- `count(active) >= 1` is `any(active)`.
    quorum_count = max(1, ceil(Int, cfg.quorum_frac * nwalkers))

    depth = 0
    while count(active) >= quorum_count && depth <= cfg.max_treedepth
        @inbounds for w in 1:nwalkers
            active[w] && (v[w] = rand(rng) < T(0.5) ? -1 : 1)
        end

        @inbounds for w in 1:nwalkers
            if active[w]
                if v[w] == 1
                    @views theta_work[:, w] .= theta_plus[:, w]
                    @views r_work[:, w] .= r_plus[:, w]
                    @views grad_work[:, w] .= grad_plus[:, w]
                else
                    @views theta_work[:, w] .= theta_minus[:, w]
                    @views r_work[:, w] .= r_minus[:, w]
                    @views grad_work[:, w] .= grad_minus[:, w]
                end
            end
        end

        subtree_active = copy(active)
        # `subtree_valid[w]`: whether the subtree accumulated so far for walker `w` is usable at
        # all. Standard NUTS discards the ENTIRE newly-built subtree of size 2^depth if divergence
        # or a U-turn occurs ANYWHERE within it -- not merely the leaves from the failure point
        # onward -- because the detailed-balance proof relies on reconstructing the same trajectory
        # (and therefore the same validity verdict for the whole subtree) regardless of which point
        # within it you start reconstructing from; partial-credit truncation breaks that symmetry.
        # Keeping the partial prefix (this module's first implementation) reproducibly inflated
        # variance by ~5-10% on toy Gaussian targets even at a correctly-tuned (target-accept-
        # matching) step size, which is what this flag fixes.
        subtree_valid = copy(active)
        leaf_count = zeros(Int, nwalkers)
        rho_new = zeros(T, nparams, nwalkers)
        log_weight_sub = fill(T(-Inf), nwalkers)
        candidate_sub = similar(positions)
        candidate_sub_logp = Vector{T}(undef, nwalkers)
        theta_last_valid = similar(positions)
        r_last_valid = similar(r0)
        grad_last_valid = similar(grad0)
        logp_last_valid = Vector{T}(undef, nwalkers)

        # Per-walker binary-counter stack for the FULL multi-scale generalized U-turn check
        # within this subtree, following the standard non-recursive equivalent of the original
        # paper's recursive per-subtree check (see e.g. Stan's `build_tree`): checking momentum
        # running-sums only between the subtree's overall first and latest leaf (as a single
        # top-level check) is NOT sufficient to preserve detailed balance -- it misses U-turns
        # that form within smaller balanced sub-spans before the whole subtree finishes, which
        # empirically manifested as a small (~5-7%) systematic variance inflation on a toy
        # Gaussian target during development of this module. `stack_rho[:,k,w]`/`stack_redge[:,k,w]`
        # hold the momentum sum / far-edge momentum of a completed-but-not-yet-merged balanced
        # sub-span of size 2^(k-1) at level k (1-indexed); `stack_count[w]` is how many levels are
        # currently filled. Leaf `i` (1-indexed within this subtree) triggers exactly
        # `trailing_zeros(i)` cascading merges, mirroring binary-counter carry propagation and
        # therefore checking every balanced sub-span exactly once, in iterative (not recursive)
        # form. Sized `depth+1` (not `depth`) because processing the very last leaf of a full
        # subtree cascades through all `depth` pending levels and then pushes the fully-merged
        # result one level higher, even though nothing ever reads that final push back.
        nlevels = depth + 1
        stack_rho = zeros(T, nparams, nlevels, nwalkers)
        stack_redge = zeros(T, nparams, nlevels, nwalkers)

        nsub = 1 << depth
        for _step in 1:nsub
            entering = copy(subtree_active)
            any(entering) || break

            @inbounds for w in 1:nwalkers
                eff_step[w] = entering[w] ? T(v[w]) * step_size[w] : zero(T)
            end

            # Leapfrog: half-step momentum, full-step position, (gradient eval), half-step momentum.
            # Boundary reflection/wrapping (`cfg.bounds`), when configured, is applied
            # immediately after the position drift and before the gradient is evaluated at the new
            # position -- see `NUTSBounds`'s and `_apply_bounds!`'s docstrings for why this ordering
            # and this specific transformation preserve the leapfrog map's Hamiltonian-invariance,
            # volume-preservation, and reversibility instead of merely clamping.
            @inbounds for w in 1:nwalkers
                entering[w] || continue
                n_leapfrog[w] += 1
                @views r_work[:, w] .+= T(0.5) .* eff_step[w] .* grad_work[:, w]
                _drift_and_bound!(view(theta_work, :, w), view(r_work, :, w), eff_step[w], _im_col(inv_mass, w), cfg.bounds)
            end

            logp_and_grad!(grad_work, logp_work, theta_work, entering)

            @inbounds for w in 1:nwalkers
                entering[w] || continue
                @views r_work[:, w] .+= T(0.5) .* eff_step[w] .* grad_work[:, w]
            end

            @inbounds for w in 1:nwalkers
                entering[w] || continue
                Hw = -logp_work[w] + _kinetic(view(r_work, :, w), _im_col(inv_mass, w))

                if isfinite(Hw)
                    accept_stat_sum[w] += min(one(T), exp(H0[w] - Hw))
                    accept_stat_count[w] += 1
                end

                valid = isfinite(Hw) && abs(Hw - H0[w]) <= cfg.max_delta_energy
                if !valid
                    subtree_active[w] = false
                    subtree_valid[w] = false
                    diverged[w] = true
                    continue
                end

                leaf_count[w] += 1
                @views rho_new[:, w] .+= r_work[:, w]

                @views theta_last_valid[:, w] .= theta_work[:, w]
                @views r_last_valid[:, w] .= r_work[:, w]
                @views grad_last_valid[:, w] .= grad_work[:, w]
                logp_last_valid[w] = logp_work[w]

                # Proportional progressive (multinomial) update of this subtree's running candidate.
                lw = H0[w] - Hw
                new_sum = _logaddexp(log_weight_sub[w], lw)
                p_take = exp(lw - new_sum)
                if rand(rng) < p_take
                    @views candidate_sub[:, w] .= theta_work[:, w]
                    candidate_sub_logp[w] = logp_work[w]
                end
                log_weight_sub[w] = new_sum

                # Full multi-scale generalized U-turn check via binary-counter stack cascade (see
                # the comment above `stack_rho`'s allocation). `cur_rho`/`cur_redge` start as the
                # trivial singleton subtree containing just this leaf, then cascade-merge with
                # however many pending stack levels this leaf's index completes.
                cur_rho = copy(view(r_work, :, w))
                cur_redge = copy(view(r_work, :, w))
                nmerges = trailing_zeros(leaf_count[w])
                turned = false
                for lvl in 1:nmerges
                    rho_top = view(stack_rho, :, lvl, w)
                    redge_top = view(stack_redge, :, lvl, w)
                    cur_rho .+= rho_top
                    if _mdot(cur_rho, redge_top, _im_col(inv_mass, w)) < 0 || _mdot(cur_rho, view(r_work, :, w), _im_col(inv_mass, w)) < 0
                        turned = true
                    end
                    cur_redge = redge_top
                end
                @views stack_rho[:, nmerges + 1, w] .= cur_rho
                @views stack_redge[:, nmerges + 1, w] .= cur_redge

                if turned
                    subtree_active[w] = false
                    subtree_valid[w] = false
                end
            end
        end

        @inbounds for w in 1:nwalkers
            active[w] || continue

            if !subtree_valid[w]
                # Divergence or a U-turn occurred somewhere in this subtree: the WHOLE subtree is
                # discarded (see `subtree_valid`'s docstring above), not merged, and the whole
                # trajectory stops growing for this walker.
                active[w] = false
                depth_reached[w] = depth
                continue
            end

            if v[w] == 1
                @views theta_plus[:, w] .= theta_last_valid[:, w]
                @views r_plus[:, w] .= r_last_valid[:, w]
                @views grad_plus[:, w] .= grad_last_valid[:, w]
            else
                @views theta_minus[:, w] .= theta_last_valid[:, w]
                @views r_minus[:, w] .= r_last_valid[:, w]
                @views grad_minus[:, w] .= grad_last_valid[:, w]
            end
            @views rho_total[:, w] .+= rho_new[:, w]

            new_total = _logaddexp(log_weight_total[w], log_weight_sub[w])
            p_take = exp(log_weight_sub[w] - new_total)
            if rand(rng) < p_take
                @views candidate[:, w] .= candidate_sub[:, w]
                candidate_logp[w] = candidate_sub_logp[w]
            end
            log_weight_total[w] = new_total

            # subtree_valid[w] being true here already implies subtree_active[w] never flipped to
            # false during this subtree's construction (they're always set together on failure),
            # so only the global (old-boundary vs new-subtree) check can still stop growth here.
            global_uturn = _mdot(view(rho_total, :, w), view(r_minus, :, w), _im_col(inv_mass, w)) < 0 ||
                            _mdot(view(rho_total, :, w), view(r_plus, :, w), _im_col(inv_mass, w)) < 0

            if global_uturn
                active[w] = false
                depth_reached[w] = depth
            end
        end

        depth += 1
    end

    # Any walker still marked active here never hit a divergence/U-turn -- either it grew
    # trajectories right up to (and through) `depth == cfg.max_treedepth`'s subtree (exhausted the
    # treedepth cap), or the round was cut short because too few OTHER walkers remained active
    # (`quorum_count`, see `NUTSConfig.quorum_frac`) while this walker itself was still going.
    # `depth` has already been incremented past the last subtree actually processed, so `depth - 1`
    # is that last completed level, for either reason. `stopped_by_quorum`: the loop exited with
    # `depth <= cfg.max_treedepth` still true, so quorum (not the depth cap) ended the round --
    # with the default `quorum_frac=0` this is never true (quorum_count=1 makes the quorum
    # condition identical to the old `any(active)`, which only becomes false together with the
    # depth-cap check on the very same walkers).
    stopped_by_quorum = depth <= cfg.max_treedepth
    quorum_forced = falses(nwalkers)
    @inbounds for w in 1:nwalkers
        if depth_reached[w] == -1
            depth_reached[w] = depth - 1
            stopped_by_quorum && (quorum_forced[w] = true)
        end
    end

    positions .= candidate
    logp_out .= candidate_logp

    accept_stat = Vector{T}(undef, nwalkers)
    @inbounds for w in 1:nwalkers
        accept_stat[w] = accept_stat_count[w] == 0 ? zero(T) : accept_stat_sum[w] / accept_stat_count[w]
    end
    return (; accept_stat, depth=depth_reached, n_leapfrog, diverged, quorum_forced)
end

"""
    pt_position_swaps!(positions, ll, betas, rungs, accepts, rng)

Replica-exchange move for parallel tempering with SEVERAL walkers per temperature. `rungs[r]` lists the
walkers held at inverse temperature `betas[rungs[r][1]]` (every walker keeps a fixed temperature, so its
own step size / dual-averaging state stays tuned to it and one metric can be shared); `ll` is each
walker's current UNTEMPERED log-likelihood. For each adjacent pair of rungs the two walker lists are
randomly paired and each pair's POSITIONS (and `ll` entries) are exchanged with probability
`min(1, exp((beta_i - beta_j) (ll_j - ll_i)))` -- the standard PT swap, exact for a target that is the
product of the tempered posteriors (priors untempered). `accepts[r]` counts accepted swaps between
rungs r and r+1. Positions change in place; NUTS recomputes log-densities at the start of each draw.
"""
function pt_position_swaps!(positions::AbstractMatrix{T}, ll::AbstractVector{T}, betas::AbstractVector{T},
                            rungs::AbstractVector{<:AbstractVector{<:Integer}}, accepts::AbstractVector{Int},
                            rng::AbstractRNG; parity::Union{Nothing,Integer}=nothing,
                            rej_sum::Union{Nothing,AbstractVector{T}}=nothing,
                            attempts::Union{Nothing,AbstractVector{Int}}=nothing) where {T}
    for r in 1:length(rungs)-1
        # `parity` (deterministic even/odd, "DEO", Syed et al. 2021): only pairs (r, r+1) with
        # (r - 1) % 2 == parity are attempted. Alternating parity every round makes positions travel
        # the ladder ballistically instead of diffusively (round trips O(N) instead of O(N^2) rounds).
        parity === nothing || (r - 1) % 2 == parity || continue
        lo = shuffle(rng, collect(rungs[r]))
        hi = shuffle(rng, collect(rungs[r+1]))
        for k in 1:min(length(lo), length(hi))
            i, j = lo[k], hi[k]
            a = (betas[i] - betas[j]) * (ll[j] - ll[i])
            # Rao-Blackwellized rejection statistic 1 - min(1, e^a) (lower variance than 0/1 counts);
            # feeds the ladder adaptation (`pt_adapt_ladder`).
            if rej_sum !== nothing
                rej_sum[r] += isfinite(a) ? one(T) - min(one(T), exp(a)) : one(T)
                attempts[r] += 1
            end
            if isfinite(a) && (a >= 0 || log(rand(rng, T)) < a)
                tmp = positions[:, i]
                positions[:, i] .= view(positions, :, j)
                positions[:, j] .= tmp
                ll[i], ll[j] = ll[j], ll[i]
                accepts[r] += 1
            end
        end
    end
    return positions
end

"""
    pt_adapt_ladder(betas, rejection) -> (betas_new, barrier)

One round of the Syed-Bouchard-Côté-Deligiannidis-Doucet (2021, JRSS-B, "Non-reversible parallel
tempering") ladder update. `betas` is the current ladder (decreasing, `betas[1] = 1`), `rejection[r]`
the mean swap rejection probability between rungs r and r+1 measured on it. The cumulative sum of the
rejection rates estimates the global communication barrier `Lambda(beta)`; the new ladder places the
same number of rungs so that every adjacent pair gets an equal share of it (equal rejection rates, which
maximizes the round-trip rate). `Lambda` is interpolated linearly in `log(beta)` (exact for the
`1/beta` local barrier of a Gaussian target). Endpoints are kept. Rates are floored at `1e-3` so a
barrier-free stretch still keeps strictly decreasing rungs. Returns the new ladder and the barrier
estimate `Lambda(beta_min)` (see [`pt_recommended_ntemps`](@ref)).
"""
function pt_adapt_ladder(betas::AbstractVector{T}, rejection::AbstractVector{T}) where {T}
    n = length(betas)
    length(rejection) == n - 1 || throw(ArgumentError("need one rejection rate per adjacent pair"))
    n <= 2 && return (collect(betas), n == 2 ? rejection[1] : zero(T))
    x = log.(betas)
    lam = zeros(T, n)
    for r in 1:n-1
        lam[r+1] = lam[r] + max(rejection[r], T(1e-3))
    end
    out = similar(x)
    out[1], out[n] = x[1], x[n]
    for k in 2:n-1
        target = lam[n] * (k - 1) / (n - 1)
        r = searchsortedlast(lam, target)          # lam[r] <= target < lam[r+1]
        r = clamp(r, 1, n - 1)
        f = (target - lam[r]) / (lam[r+1] - lam[r])
        out[k] = x[r] + f * (x[r+1] - x[r])
    end
    return (exp.(out), lam[n])
end

"""
    pt_gaussian_barrier(d, beta_min) -> Lambda

Global communication barrier of a `d`-dimensional Gaussian likelihood tempered from `beta = 1` down to
`beta_min`: the energy `-logL` at inverse temperature beta has standard deviation `sqrt(d/2)/beta`, the
local barrier is `sd/sqrt(pi)`, and integrating gives `sqrt(d/(2 pi)) log(1/beta_min)`. A LOWER bound
for multimodal posteriors; used to choose the initial rung count before any swap has been measured.
"""
pt_gaussian_barrier(d::Integer, beta_min::Real) = sqrt(d / (2pi)) * log(1 / beta_min)

"""
    pt_recommended_ntemps(barrier; lo=2, hi=16) -> ntemps

Rung count giving a swap rejection of about 0.5 per adjacent pair with non-reversible (DEO) swaps:
`rejection ~ barrier/(ntemps - 1)`, so `ntemps = ceil(2 * barrier) + 1`, clamped to `[lo, hi]`.
"""
pt_recommended_ntemps(barrier::Real; lo::Integer=2, hi::Integer=16) = clamp(ceil(Int, 2 * barrier) + 1, lo, hi)

"""
    _de_jump_step!(positions, logp, accepts, logp_and_grad!, bounds, rng)

Differential-evolution mode-hopping move between NUTS draws (`nuts_sample!(...; de_jumps=true)`).
The walkers are split into two halves; each walker `w` of one half proposes
`x' = x_w + (x_b - x_c)` with `b != c` drawn from the OTHER half (gamma = 1, the value that maps one
mode onto another when `x_b` and `x_c` sit in different modes), accepted with probability
`min(1, p(x')/p(x_w))`; then the halves swap roles. Given the complementary half held fixed, the proposal
is a symmetric random translation, so each half-update leaves the joint target of that half invariant
(the complementary-ensemble construction used by emcee / ter Braak's DE-MC). Periodic rows of `bounds`
are wrapped (translations on a circle stay symmetric); proposals outside the box are rejected via the
target's own `-Inf`. Lets walkers reach separated modes (e.g. low-SNR chirp-mass cycle-slip peaks) that
other walkers have already found, which HMC trajectories cannot cross. Costs two batched gradient
evaluations per draw.
"""
function _de_jump_step!(positions::AbstractMatrix{T}, logp::AbstractVector{T}, accepts,
                        logp_and_grad!, bounds, rng) where {T}
    nparams, nwalkers = size(positions)
    perm = randperm(rng, nwalkers)
    halves = (perm[1:nwalkers ÷ 2], perm[nwalkers ÷ 2 + 1:end])
    prop = copy(positions)
    grad = similar(positions)
    lp = similar(logp)
    for (mine, other) in (halves, reverse(halves))
        copyto!(prop, positions)
        for w in mine
            b = other[rand(rng, 1:length(other))]
            c = other[rand(rng, 1:length(other))]
            while c == b
                c = other[rand(rng, 1:length(other))]
            end
            @inbounds for i in 1:nparams
                x = positions[i, w] + (positions[i, b] - positions[i, c])
                if bounds !== nothing && bounds.periodic[i]
                    x = bounds.lo[i] + mod(x - bounds.lo[i], bounds.hi[i] - bounds.lo[i])
                end
                prop[i, w] = x
            end
        end
        # Never evaluate the target outside the box (a model may be undefined there, not just -Inf):
        # such proposals are rejected outright, and the column is reset to its current position.
        inbox = trues(nwalkers)
        for w in mine
            if bounds !== nothing && !all(i -> bounds.lo[i] <= prop[i, w] <= bounds.hi[i], 1:nparams)
                inbox[w] = false
                prop[:, w] .= view(positions, :, w)
            end
        end
        logp_and_grad!(grad, lp, prop, trues(nwalkers))
        for w in mine
            a = lp[w] - logp[w]
            if inbox[w] && isfinite(lp[w]) && (a >= 0 || log(rand(rng, T)) < a)
                positions[:, w] .= view(prop, :, w)
                logp[w] = lp[w]
                accepts[] += 1
            end
        end
    end
    return nothing
end

"""
    _jump_step!(positions, logp, jumps, accepts, scratch, logp_and_grad!, rng)

Metropolis "jump" moves between NUTS draws (`nuts_sample!`'s `jumps` keyword). Each element of `jumps`
is a function `f!(x::AbstractVector)` mapping one walker's column in place by a deterministic
INVOLUTION (`f(f(x)) == x`) with unit Jacobian in the sampled coordinates; the proposal `x' = f(x)` is
accepted with probability `min(1, p(x')/p(x))`, which leaves the target exactly invariant. Intended for
known discrete (near-)symmetries of the posterior that HMC trajectories cannot cross -- e.g. the
two-detector sky reflection or the inclination flip of a GW posterior (`BatchedNUTS.gw_jump_moves`).
Each move costs one batched `logp_and_grad!` evaluation. `logp` is updated for accepted walkers.
"""
function _jump_step!(positions::AbstractMatrix{T}, logp::AbstractVector{T}, jumps, accepts,
                     scratch, logp_and_grad!, rng) where {T}
    prop, grad, lp = scratch
    nwalkers = size(positions, 2)
    for (k, f!) in enumerate(jumps)
        copyto!(prop, positions)
        for w in 1:nwalkers
            f!(view(prop, :, w))
        end
        logp_and_grad!(grad, lp, prop, trues(nwalkers))
        for w in 1:nwalkers
            a = lp[w] - logp[w]
            if isfinite(lp[w]) && (a >= 0 || log(rand(rng, T)) < a)
                positions[:, w] .= view(prop, :, w)
                logp[w] = lp[w]
                accepts[k] += 1
            end
        end
    end
    return nothing
end

"""
    nuts_sample!(positions, logp_and_grad!; nsteps, step_size0, cfg=NUTSConfig{T}(...),
                 rng=Random.default_rng(), record=false, warmup_logp_and_grad!=logp_and_grad!)

Run `nsteps` NUTS draws in place on `positions` (`nparams x nwalkers`), using dual-averaging
step-size adaptation for the first `cfg.nwarmup` draws and a frozen (dual-averaging-converged)
step size thereafter.

`logp_and_grad!` must conform to the module-level interface documented above. Returns a
NamedTuple `(positions, logp, step_size, chain, inv_mass)`; `chain` is `nothing` unless `record=true`, in
which case it is a `(nparams, nwalkers, nsteps)` array of every draw's positions (used by the
Gaussian-recovery test to pool post-warmup samples across both steps and walkers).

`warmup_logp_and_grad!` (defaults to `logp_and_grad!` itself, i.e. a no-op -- exactly the
pre-existing single-closure-for-the-whole-chain behavior): an optional SEPARATE callable, same
interface, used only for draws `it <= cfg.nwarmup`; `logp_and_grad!` is used for every draw after
that, same as always. This exists for callers whose main (post-warmup) kernel is unsafe to use
during warmup itself -- e.g. `BatchedNUTS`'s relative-binning kernel for `SEOBNRv5AlignedSpinF2`,
where a real NUTS chain's UNADAPTED warmup step size was measured to transiently breach the binned kernel's
own aliasing-safety ceiling before dual-averaging shrinks the step size, even though the CONVERGED
post-warmup step size stays deeply inside the safe region. Passing the dense (always-safe) kernel
as `warmup_logp_and_grad!` and the binned (fast, but only post-warmup-safe) kernel as the main
`logp_and_grad!` makes that pattern directly expressible: dense during warmup, binned once the
step size has converged. When `warmup_logp_and_grad!` is left at its default, `it <= cfg.nwarmup`
selects the SAME closure either way, so this keyword changes nothing about the default call path
(confirmed by `test/test_nuts_core_bounds.jl`'s own switch-mechanism testset, including a
bitwise-identical-trajectory check at a fixed seed).

`callback` (default `nothing`): an optional `(it, positions, logp, step_size, stats) -> nothing`
called once at the end of every draw `it` (after that draw's step-size update, if any), given this
module's own read-only view of the just-completed draw. Exists so a caller can hook in per-draw
side effects (parallel-tempering swap rounds via [`pt_swap!`](@ref) being the motivating case --
see `BatchedNUTS.batched_pt_nuts!`) WITHOUT `NUTSCore` itself knowing anything about what the
callback does -- kept generic/GW-agnostic per this module's own design (see the module docstring
above), the same reason `logp_and_grad!` itself is a caller-supplied closure rather than something
this module constructs. `positions` is passed as the SAME array `nuts_sample!` mutates in place
(not a copy), so a callback that mutates it (e.g. `pt_swap!` mutating a `PTLadderState`'s
`betas`/`lane_of_rank` that a tempered `logp_and_grad!` closure captured by reference) takes effect
starting with the NEXT draw, exactly like updating `logp_and_grad!`'s other captured state would.

`stats` is the NamedTuple `nuts_draw!` returns for this draw:
`(; accept_stat, depth, n_leapfrog, diverged)`, each field a length-`nwalkers` vector --
`accept_stat`: the Metropolis-style acceptance statistic dual averaging targets; `depth`: the
tree-doubling depth this walker's trajectory reached before terminating (U-turn, divergence, or
exhausting `cfg.max_treedepth`); `n_leapfrog`: total leapfrog integrator calls (gradient
evaluations) this walker's draw made; `diverged`: whether a non-finite or
`max_delta_energy`-exceeding Hamiltonian was hit anywhere in this draw's tree, as opposed to
ordinary U-turn termination. Near-zero overhead (these are scalars `nuts_draw!` already computes
internally; no new gradient/likelihood evaluations). Answers directly, instead of by inference from
draw wall-clock timing, questions like "is this run
still maxing out `max_treedepth` every draw post-warmup" (`depth` at or near `cfg.max_treedepth`
well past `cfg.nwarmup`) or "did this draw diverge" (`any(diverged)`).

`nothing` (default) preserves the exact pre-existing behavior with zero overhead.

**Diagonal mass matrix**. With `cfg.adapt_mass=true` and `cfg.nwarmup >= 20`, warmup
follows Stan's windowed schedule (`_mass_adaptation_windows`): a step-size-only initial buffer,
slow windows of doubling length, a step-size-only terminal buffer. At the end of each slow window
the inverse mass matrix (diagonal, one entry per parameter, SHARED by all walkers) is set to the
pooled within-walker sample variance of that window's positions, and dual averaging restarts from
the current step size (`mu = log(10*step)`). Momentum is drawn as `N(0, M)`, kinetic energy is
`0.5 r'M^-1 r`, the position drift is `eps*M^-1 r`, and the U-turn criterion uses `M^-1 r`
(Betancourt 2017) -- so the target is unchanged; only efficiency is. Differences from Stan:
variance is NOT shrunk toward `1e-3` (that constant assumes O(1) parameters; the GW sampled space
spans many decades), and no find-reasonable-step-size heuristic runs after a metric update
(dual averaging recovers within a few draws; validated in `test/test_nuts_core_mass_matrix.jl`).
By default the metric is shared across walkers, which is wrong for parallel tempering (each lane
targets a different tempered density, and the beta LABEL moves between lanes on a swap -- see
`metric_rank`).

`metric_rank` (default `nothing`; PT only): a length-`nwalkers` vector, read by reference every draw,
giving each walker's current temperature rank -- pass `pt.rank_of_lane`, a permutation of
`1:nwalkers`. The metric is then attached to the RANK, not the lane: `inv_mass` becomes an
`(nparams, nranks)` matrix, each draw gathers the columns for the walkers' current ranks, and
adaptation accumulates rank `r`'s own stream (whichever lane held rank `r` on each draw) with NO
pooling across ranks -- hot ranks legitimately learn broader scales than the cold one. Correct
because the rank is part of the extended (position, label) state and the metric is a fixed function
of it within a draw (and frozen after warmup). A matrix `inv_mass0` requires `metric_rank`.

`inv_mass0` (default `nothing` = identity metric, the exact old code path -- bit-for-bit identical
chains): initial inverse-mass-matrix diagonal, length `nparams` (or `(nparams, nranks)` with
`metric_rank`). Together with the returned
`inv_mass` this lets a caller carry an adapted metric from a warmup call into a later
`nwarmup=0` sampling call (see `scripts/run_from_config.jl`'s two-phase relative-binning runs). The
returned NamedTuple's `inv_mass` is the metric in effect at the end (ones if none was ever set).
"""
function nuts_sample!(positions::AbstractMatrix{T}, logp_and_grad!::F;
                       nsteps::Int, step_size0::AbstractVector{T},
                       cfg::NUTSConfig{T}=NUTSConfig{T}(; nwarmup=max(nsteps ÷ 2, 1)),
                       rng::AbstractRNG=Random.default_rng(),
                       record::Bool=false,
                       warmup_logp_and_grad!::G=logp_and_grad!,
                       callback=nothing,
                       inv_mass0::Union{Nothing,AbstractVector{T},AbstractMatrix{T},DenseMetric{T}}=nothing,
                       metric_rank::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                       jumps=(), de_jumps::Bool=false,
                       metric_walkers::Union{Nothing,AbstractVector{<:Integer}}=nothing) where {T,F,G}
    nparams, nwalkers = size(positions)
    (de_jumps && nwalkers < 6) && throw(ArgumentError("de_jumps needs at least 6 walkers (two halves of >= 3)"))
    de_accepts = Ref(0)
    jump_scratch = isempty(jumps) ? nothing : (similar(positions), similar(positions), Vector{T}(undef, nwalkers))
    jump_accepts = zeros(Int, length(jumps))
    @assert length(step_size0) == nwalkers
    per_rank = metric_rank !== nothing
    (per_rank && (cfg.dense_mass || inv_mass0 isa DenseMetric)) && throw(ArgumentError(
        "a dense mass matrix cannot be combined with `metric_rank` (parallel tempering)"))
    (per_rank && cfg.cull_logp_gap > 0) && throw(ArgumentError(
        "`cull_logp_gap` cannot be combined with `metric_rank` (parallel tempering): lanes are at different temperatures"))
    inv_mass0 isa DenseMetric && size(inv_mass0.Minv) != (nparams, nparams) && throw(ArgumentError(
        "`DenseMetric` must be $nparams x $nparams, got $(size(inv_mass0.Minv))"))
    if inv_mass0 isa AbstractMatrix
        per_rank || throw(ArgumentError("a matrix `inv_mass0` requires `metric_rank`"))
        size(inv_mass0) == (nparams, nwalkers) || throw(ArgumentError(
            "matrix `inv_mass0` must be (nparams, nranks) = ($nparams, $nwalkers), got $(size(inv_mass0))"))
    elseif inv_mass0 !== nothing && !(inv_mass0 isa DenseMetric)
        length(inv_mass0) == nparams || throw(ArgumentError("`inv_mass0` must have length nparams=$nparams"))
    end
    per_rank && (length(metric_rank) == nwalkers || throw(ArgumentError(
        "`metric_rank` must have one entry per walker (length $nwalkers)")))

    logp = Vector{T}(undef, nwalkers)
    step_size = copy(step_size0)
    da = DualAveraging(step_size)
    chain = record ? Array{T}(undef, nparams, nwalkers, nsteps) : nothing

    # Diagonal mass matrix state. `inv_mass === nothing` is the untouched unit-metric path.
    # Shared mode: a length-nparams vector. Per-rank mode (`metric_rank`): an (nparams, nranks)
    # matrix, column `r` being the metric of temperature rank `r`; each draw gathers the columns of
    # the walkers' CURRENT ranks (`metric_rank` is read by reference every draw, so PT label swaps
    # made by a callback take effect on the next draw, exactly like `pt.betas`).
    inv_mass = if inv_mass0 === nothing
        nothing
    elseif inv_mass0 isa DenseMetric
        inv_mass0
    elseif per_rank && inv_mass0 isa AbstractVector
        repeat(reshape(Vector{T}(inv_mass0), :, 1), 1, nwalkers)
    else
        copy(inv_mass0)
    end
    windows = cfg.adapt_mass ? _mass_adaptation_windows(cfg.nwarmup; init_buffer=cfg.mass_init_buffer,
                                                         term_buffer=cfg.mass_term_buffer,
                                                         base_window=cfg.mass_base_window) : UnitRange{Int}[]
    dense = cfg.dense_mass
    metric_informed = inv_mass0 isa DenseMetric     # true once `inv_mass` is more than the identity placeholder
    if !isempty(windows) && inv_mass === nothing
        inv_mass = dense ? DenseMetric(Matrix{T}(I, nparams, nparams)) :
                   per_rank ? ones(T, nparams, nwalkers) : ones(T, nparams)
    end
    wvar = isempty(windows) ? nothing : (dense ? _WindowCov(T, nparams, nwalkers) : _WindowVariance(T, nparams, nwalkers))
    next_window = 1

    # Dense-metric bootstrap (see `_hessian_metric`): replace the identity metric by the local
    # curvature before the first draw, then pick a step size that suits it (dual averaging alone
    # would need ~1 draw per factor of 2 from a tiny `step_size0`).
    if dense && cfg.hessian_init && cfg.nwarmup > 0
        hm, nvalid = _hessian_metric(warmup_logp_and_grad!, positions, cfg; walkers=metric_walkers)
        if hm === nothing
            @warn "NUTSCore: hessian_init found no walker with a finite Hessian; keeping the current metric"
        else
            inv_mass = hm
            metric_informed = true
        end
    end
    if dense && cfg.nwarmup > 0 && (cfg.hessian_init || inv_mass0 isa DenseMetric)
        _find_reasonable_step_size!(step_size, positions, warmup_logp_and_grad!, cfg, rng, inv_mass)
        da = DualAveraging(copy(step_size))
    end

    for it in 1:nsteps
        active_logp_and_grad! = it <= cfg.nwarmup ? warmup_logp_and_grad! : logp_and_grad!
        ranks = per_rank ? Int.(Array(metric_rank)) : nothing
        per_rank && (isperm(ranks) || throw(ArgumentError("`metric_rank` must be a permutation of 1:nwalkers")))
        im_draw = (per_rank && inv_mass !== nothing) ? inv_mass[:, ranks] : inv_mass
        pos_before = copy(positions)
        stats = nuts_draw!(positions, logp, step_size, active_logp_and_grad!, cfg, rng, im_draw)
        isempty(jumps) || _jump_step!(positions, logp, jumps, jump_accepts, jump_scratch, active_logp_and_grad!, rng)
        de_jumps && _de_jump_step!(positions, logp, de_accepts, active_logp_and_grad!, cfg.bounds, rng)
        # `disp`: per-walker whitened distance moved this draw (~1 = one posterior sd; << 1 = frozen).
        stats = merge(stats, (; disp=_whitened_displacement(pos_before, positions, im_draw)))
        if it <= cfg.nwarmup
            step_size .= update!(da, stats.accept_stat, cfg.target_accept)
        elseif it == cfg.nwarmup + 1 && cfg.nwarmup > 0
            # Freeze to the dual-averaging-converged step size. Guarded by `cfg.nwarmup > 0` so
            # that `nwarmup=0` (no adaptation requested) leaves `step_size0` untouched instead of
            # overwriting it with `exp(0) == 1` from `DualAveraging`'s never-updated initial state.
            step_size .= finalize_step(da)
        end
        if wvar !== nothing && next_window <= length(windows) && it in windows[next_window]
            # Per-rank mode: column r of the accumulator is rank r's own stream (whichever lane held
            # rank r during this draw), so re-order lanes into rank order first.
            _accumulate!(wvar, per_rank ? positions[:, invperm(ranks)] : positions)
            if it == last(windows[next_window])
                # Warmup-only culling (`cfg.cull_logp_gap`): walkers stuck far below the best logp are in a
                # secondary local mode HMC cannot leave. Exclude them from this window's metric estimate
                # (their local geometry is not the posterior's) and re-seed them from good walkers below.
                # `metric_walkers` (parallel tempering): only the beta = 1 walkers estimate the metric and
                # take part in culling; tempered walkers see a different (flatter) target.
                base = metric_walkers === nothing ? collect(1:nwalkers) : collect(Int, metric_walkers)
                good = copy(base)
                if cfg.cull_logp_gap > 0
                    finite = [w for w in base if isfinite(logp[w])]
                    if !isempty(finite)
                        lmax = maximum(logp[finite])
                        good = [w for w in finite if logp[w] >= lmax - cfg.cull_logp_gap]
                    end
                end
                # Guard: keep the previous scale for any parameter whose window variance is
                # non-finite or non-positive (e.g. a stalled row); Stan's own shrinkage-to-1e-3
                # target is deliberately NOT used -- this package's sampled parameters span wildly
                # different physical scales, so a fixed absolute target would swamp small-scale rows.
                if dense
                    S = _pooled_covariance(wvar; prior=metric_informed ? inv_mass : nothing, walkers=good)
                    if S !== nothing
                        inv_mass = DenseMetric(_block_mask(S, cfg.mass_blocks))
                        metric_informed = true
                    end
                elseif per_rank
                    v = _column_variance(wvar)
                    @inbounds for r in 1:nwalkers, i in 1:nparams
                        (isfinite(v[i, r]) && v[i, r] > 0) && (inv_mass[i, r] = v[i, r])
                    end
                else
                    v = _pooled_variance(wvar; walkers=good)
                    @inbounds for i in 1:nparams
                        (isfinite(v[i]) && v[i] > 0) && (inv_mass[i] = v[i])
                    end
                end
                if length(good) < length(base) && !isempty(good)
                    goodset = Set(good)
                    ncull = 0
                    for w in base
                        w in goodset && continue
                        donor = good[rand(rng, 1:length(good))]
                        positions[:, w] .= positions[:, donor]
                        logp[w] = logp[donor]
                        step_size[w] = step_size[donor]
                        ncull += 1
                    end
                    @info "NUTSCore: warmup draw $it: culled $ncull walker(s) more than $(cfg.cull_logp_gap) below the best logp and re-seeded them from good walkers"
                end
                wvar = dense ? _WindowCov(T, nparams, nwalkers) : _WindowVariance(T, nparams, nwalkers)
                next_window += 1
                # Metric changed => the old step size's tuning is stale. Dense mode re-finds a
                # reasonable step (Stan's heuristic; the metric can change the right step by orders
                # of magnitude and a draw is expensive); either way restart dual averaging
                # (mu = log(10 * current step), Stan's convention) from the current step size.
                # `max_change=cfg.step_size_max_change` bounds
                # how far this ONE restart may move the step size -- see the function's own comment
                # for why this call site is capped and the initial bootstrap call above is not.
                dense && _find_reasonable_step_size!(step_size, positions, warmup_logp_and_grad!, cfg, rng, inv_mass;
                                                      max_change=cfg.step_size_max_change)
                da = DualAveraging(copy(step_size))
            end
        end
        record && (chain[:, :, it] .= positions)
        callback === nothing || callback(it, positions, logp, step_size, stats)
    end

    inv_mass_out = inv_mass === nothing ? (per_rank ? ones(T, nparams, nwalkers) : ones(T, nparams)) : inv_mass
    dense && !(inv_mass_out isa DenseMetric) && (inv_mass_out = DenseMetric(Matrix{T}(I, nparams, nparams)))
    return (; positions, logp, step_size, chain, inv_mass=inv_mass_out,
            jump_acceptance=isempty(jumps) ? Float64[] : jump_accepts ./ (nsteps * nwalkers),
            de_acceptance=de_jumps ? de_accepts[] / (nsteps * nwalkers) : NaN)
end

# ---------------------------------------------------------------------------------------------
# Parallel tempering: swap accept/reject only needs each lane's own current
# UNTEMPERED log-likelihood plus the pairwise `beta`s already resident wherever `positions` itself
# lives, so the whole swap step -- deterministic even/odd adjacent-in-temperature-RANK pairing,
# Metropolis accept/reject, then the swap itself -- is written here as a small `KernelAbstractions`
# kernel (`_pt_swap_kernel!` below), exactly like every other batched kernel in this package
# (`KernelFused.jl`'s `_chirp_mass_eta_kernel!` is the closest analogue: a small, non-waveform
# `@kernel` used for a piece of bookkeeping that still benefits from running on whatever backend
# the caller's arrays live on). This is GW-agnostic (it never touches `positions`/waveform data at
# all, only the tiny length-`ntemps` `betas`/log-likelihood arrays), so it lives here alongside
# `NUTSBounds`/`DualAveraging` rather than in `BatchedNUTS.jl`.
#
# **Swaps the temperature LABEL per lane, not the physical state** (rather than a host-orchestrated exchange of states): each lane `w`
# holds a mutable `betas[w]` rather than being permanently bound to one temperature, so an accepted
# swap between two lanes is just exchanging two scalars, never touching (or even knowing the shape
# of) that lane's `(nparams, nwalkers)` position column. The tradeoff this accepts, same as the
# design note: no adaptive temperature-schedule tuning, no non-reversible communication scheme, no
# built-in diagnostics beyond the plain `naccept`/`nattempt` counters kept here -- Pigeons.jl's own
# machinery for those, not reimplemented.
#
# **Scope note on "on-device" here**: this kernel is written with `KernelAbstractions.get_backend`
# so it runs on whatever backend `betas`/`ll` live on (`CPU()` today when driven by
# `BatchedNUTS.batched_pt_nuts!`'s default `pt_backend`, `CUDABackend()` once `PTLadderState` is
# constructed with one and fed device-resident `ll`) -- the SAME portability convention every other
# `@kernel` in this package already follows (write once, backend inferred from the arrays handed
# in), not a claim that today's `NUTSCore`/`BatchedNUTS` sampler LOOP itself is GPU-resident: that
# loop's `positions`/`logp`/`grad` are host arrays regardless (see
# `BatchedNUTS.gw_logp_and_grad_gpu_staged`'s own docstring for why -- the expensive PER-STEP
# waveform+likelihood kernel is what actually runs on GPU, staged across the host/device boundary
# once per leapfrog step; this swap kernel's `betas`/`ll` inputs are already exactly as
# host/device-resident as everything else surrounding them). The `u`/`accepted` arrays this
# function allocates each round are `ntemps÷2`-length -- tiny scalars crossing the boundary, the
# same "small scalars are fine, large per-parameter buffers are not" precedent
# `BatchedNUTS._forward_mode_gradient!`'s `geom_flat`/`gmst_arr` staging already established.
# ---------------------------------------------------------------------------------------------

"""
    PTLadderState{T}(ntemps; T=Float64, beta_min=T(0.05), backend=KernelAbstractions.CPU())

Bookkeeping for a geometric temperature ladder over `ntemps` lanes (one NUTS walker per
temperature -- see `BatchedNUTS.batched_pt_nuts!`'s docstring for why `ntemps == nwalkers` is
required there, unlike `batched_nuts!`'s ordinary independent-walkers-all-at-`beta=1` design).

`beta_ladder[k] = beta_min^((k-1)/(ntemps-1))` for `k=1:ntemps`, so `beta_ladder[1] == 1`
(the untempered target posterior, "cold") descending to `beta_ladder[ntemps] == beta_min` ("hot",
flattened) -- the standard geometric PT spacing. Kept as a plain host `Vector{T}` regardless of
`backend`: it is fixed reference data, read only for diagnostics/driver bookkeeping (e.g.
"what beta does rank 1 correspond to"), never touched inside `_pt_swap_kernel!` itself (the kernel
only ever reads/writes `betas`/`lane_of_rank`/`rank_of_lane`, from which the assigned beta is
always recoverable as `betas[lane]`).

`betas[w]`: lane `w`'s CURRENTLY assigned temperature (mutated by `pt_swap!`) -- this is the array
[`BatchedNUTS.gw_logp_and_grad`](@ref)'s `beta` keyword should be given (by reference, so swaps
take effect on the very next likelihood evaluation with no closure rebuild needed).

`lane_of_rank[k]`/`rank_of_lane[w]`: mutually inverse permutations (`lane_of_rank[rank_of_lane[w]]
== w`) tracking which lane currently holds which ladder rank, maintained incrementally by
`pt_swap!` rather than re-sorted every round -- `lane_of_rank[1]` is always the lane currently
running the `beta=1` ("cold", target-posterior) chain, the one a caller should record for actual
posterior inference (see `BatchedNUTS.batched_pt_nuts!`'s `cold_chain` return value).

`betas`/`lane_of_rank`/`rank_of_lane` all live on `backend` (default `KernelAbstractions.CPU()`);
`naccept`/`nattempt` (length `ntemps-1`, one entry per adjacent ladder pair, indexed by the pair's
lower rank) are always plain host `Vector{Int}` diagnostics, updated from the kernel's tiny
`accepted` output after each round -- `naccept[k] / nattempt[k]` is that pair's running swap
acceptance rate.

`parity` alternates `0`/`1` each `pt_swap!` call: an even/odd adjacent-pairing schedule (ranks
`(1,2),(3,4),...` on `parity=0`, `(2,3),(4,5),...` on `parity=1`) so every adjacent pair
eventually gets proposed, not just one fixed pairing forever.
"""
mutable struct PTLadderState{T,V<:AbstractVector{T},I<:AbstractVector{Int}}
    beta_ladder::Vector{T}
    betas::V
    lane_of_rank::I
    rank_of_lane::I
    parity::Int
    naccept::Vector{Int}
    nattempt::Vector{Int}
end

function PTLadderState(ntemps::Int; T::Type=Float64, beta_min=T(0.05),
                        backend=KernelAbstractions.CPU())
    ntemps >= 1 || throw(ArgumentError("PTLadderState requires ntemps >= 1"))
    beta_min = T(beta_min)
    (beta_min > zero(T) && beta_min <= one(T)) ||
        throw(ArgumentError("beta_min must lie in (0, 1]"))
    beta_ladder = ntemps == 1 ? T[one(T)] :
        T[beta_min^(T(k - 1) / T(ntemps - 1)) for k in 1:ntemps]

    betas = KernelAbstractions.allocate(backend, T, ntemps)
    copyto!(betas, beta_ladder)
    lane_of_rank = KernelAbstractions.allocate(backend, Int, ntemps)
    copyto!(lane_of_rank, collect(1:ntemps))
    rank_of_lane = KernelAbstractions.allocate(backend, Int, ntemps)
    copyto!(rank_of_lane, collect(1:ntemps))

    npairs = max(ntemps - 1, 0)
    return PTLadderState{T,typeof(betas),typeof(lane_of_rank)}(
        beta_ladder, betas, lane_of_rank, rank_of_lane, 0, zeros(Int, npairs), zeros(Int, npairs))
end

# One thread per candidate adjacent pair at the current parity; `ntemps` is passed so the kernel
# can guard the boundary pair that parity=1 (or an odd ntemps) can generate with no partner --
# see `pt_swap!`'s `npairs = cld(ntemps, 2)` launch width for why that guard is needed rather than
# sized away.
@kernel function _pt_swap_kernel!(betas, @Const(ll), lane_of_rank, rank_of_lane, accepted,
                                   @Const(u), parity::Int, ntemps::Int)
    p = @index(Global, Linear)
    @inbounds begin
        r_lo = 2 * (p - 1) + 1 + parity
        r_hi = r_lo + 1
        if r_hi <= ntemps
            lane_lo = lane_of_rank[r_lo]
            lane_hi = lane_of_rank[r_hi]
            beta_lo = betas[lane_lo]
            beta_hi = betas[lane_hi]
            ll_lo = ll[lane_lo]
            ll_hi = ll[lane_hi]
            # Standard PT swap acceptance (see this section's module comment): detailed balance for
            # exchanging which beta labels which position reduces to only these two lanes' own raw
            # log-likelihoods at their own (unmoved) positions -- the log-prior term cancels exactly
            # since it is untempered and neither position moves.
            log_ratio = (beta_lo - beta_hi) * (ll_hi - ll_lo)
            accept = log(u[p]) < log_ratio
            accepted[p] = accept
            if accept
                betas[lane_lo] = beta_hi
                betas[lane_hi] = beta_lo
                lane_of_rank[r_lo] = lane_hi
                lane_of_rank[r_hi] = lane_lo
                rank_of_lane[lane_lo] = r_hi
                rank_of_lane[lane_hi] = r_lo
            end
        else
            accepted[p] = false
        end
    end
end

"""
    pt_swap!(pt::PTLadderState{T}, ll::AbstractVector{T}, rng::AbstractRNG) where {T}

One parallel-tempering swap round: proposes an accept/reject exchange of the `beta` LABEL (not
the physical position) between every adjacent-in-rank pair at `pt`'s current `parity`, using each
lane's own raw (untempered) log-likelihood `ll[w]` at its own current position (the caller must
compute this immediately before calling, e.g. via `BatchedNUTS._raw_loglikelihood!` -- it is NOT
the tempered `logp` `NUTSCore.nuts_sample!` tracks, and NOT recoverable from that, since the
tempered value already has `beta` baked in and the log-prior added). Mutates `pt.betas`,
`pt.lane_of_rank`, `pt.rank_of_lane` in place (via `_pt_swap_kernel!`, see this section's module
comment for the backend-portability note), updates `pt.naccept`/`pt.nattempt`, flips `pt.parity`
for the next round, and returns `pt`.

`ll` must live on the same backend as `pt.betas` (both derived from `KernelAbstractions.get_backend`
internally). A no-op (returns `pt` unchanged, no swap possible) when `ntemps < 2`.
"""
function pt_swap!(pt::PTLadderState{T}, ll::AbstractVector{T}, rng::AbstractRNG) where {T}
    ntemps = length(pt.betas)
    ntemps < 2 && return pt

    backend = KernelAbstractions.get_backend(pt.betas)
    npairs = cld(ntemps, 2)

    u = KernelAbstractions.allocate(backend, T, npairs)
    copyto!(u, rand(rng, T, npairs))
    accepted = KernelAbstractions.allocate(backend, Bool, npairs)

    kernel! = _pt_swap_kernel!(backend)
    kernel!(pt.betas, ll, pt.lane_of_rank, pt.rank_of_lane, accepted, u, pt.parity, ntemps;
            ndrange=npairs)
    KernelAbstractions.synchronize(backend)

    accepted_host = Array(accepted)
    @inbounds for p in 1:npairs
        r_lo = 2 * (p - 1) + 1 + pt.parity
        r_hi = r_lo + 1
        r_hi <= ntemps || continue
        pt.nattempt[r_lo] += 1
        accepted_host[p] && (pt.naccept[r_lo] += 1)
    end

    pt.parity = 1 - pt.parity
    return pt
end


include("SMCCore.jl")

end
