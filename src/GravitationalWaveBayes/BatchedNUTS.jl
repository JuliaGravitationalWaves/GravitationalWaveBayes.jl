module BatchedNUTS

export gw_jump_moves, reconstruct_marginalized!, PEState, batched_nuts!, run_gw150914_estimation, gw_logp_and_grad, gw_logp_and_grad_gpu_staged, tempered_logp_and_grad,
       batched_pt_nuts!, run_gw150914_pt_estimation

using ..Types: PEConfig, PEState, LikelihoodData, DetectorData, with_likelihood_options
import ..KernelFused
import LinearAlgebra
using ..KernelFused: batched_log_likelihood_kernel!
using ..RelativeBinning: RelativeBinningSetup, binned_log_likelihood_kernel!
import ..RelativeBinning
using GravitationalWaveforms: IMRPhenomD, IMRPhenomXAS, IMRPhenomXP, IMRPhenomXHM, IMRPhenomXPHM, SEOBNRv5AlignedSpinF2, SEOBNRv5PHM
using ..Types: PAF2Model, IMRTableModel
using ..KernelFused: _Prec15
import ..SEOBNRv5AlignedSpinF2Adjoint
import GravitationalWaveforms
using ..GPU: compatible_array, to_device
import ..GPU
using ..NUTSCore: nuts_sample!, NUTSConfig, NUTSBounds, PTLadderState, pt_swap!
using ..Priors: GWPrior, prior_logpdf_and_grad!, prior_bounds
using GravitationalWaveData: gmst, geometry, antenna_response_gmst, time_delay_from_geocenter_gmst
using Enzyme
using KernelAbstractions
using Random
using StaticArrays: SVector

# ------------------------------------------------------------------------------------------------
# Sampling-layer reparametrization (11 sampled parameters for the aligned-spin models).
# NUTS samples, in `positions` row order:
#
#   1. mc            -- chirp mass, solar masses (Jim's convention)
#   2. q             -- mass ratio m2/m1, (0.125, 1] (Jim's convention)
#   3. chi1          -- primary aligned spin, [-1, 1]
#   4. chi2          -- secondary aligned spin, [-1, 1]
#   5. dL            -- luminosity distance, Gpc -- sampled DIRECTLY (uniform prior stated
#                       directly in dL, no reparametrization)
#   6. t_c           -- coalescence-time offset from the fixed event GPS trigger time, seconds --
#                       sampled directly (small uniform window, e.g. +/-0.1s)
#   7. phase         -- reference/coalescence phase, radians -- sampled directly, uniform
#                       [0, 2*pi)
#   8. ra            -- right ascension, radians -- sampled directly, uniform [0, 2*pi)
#   9. sin_dec       -- NOT declination itself: NUTS samples `sin(dec)` directly, uniform on
#                       [-1, 1]. This is the isotropic-sky-prior reparametrization choice (see
#                       the block comment above `_log_prior` below for why sampling `sin(dec)`
#                       directly -- rather than sampling `dec` and adding a Jacobian term to
#                       `_log_prior` -- was chosen, matching this file's existing `(mc, q)`
#                       "state the prior directly in the sampled coordinate" style).
#  10. cos_theta_jn  -- NOT theta_jn itself: NUTS samples `cos(theta_jn)` directly, uniform on
#                       [-1, 1], for the isotropic-orientation prior. Same reasoning as sin_dec.
#  11. psi           -- polarization angle, radians -- sampled directly, uniform [0, pi)
#
# `batched_log_likelihood_kernel!`/`KernelFused.jl` (and `RelativeBinning.jl`) are UNCHANGED in
# the sense that they still expect the PHYSICAL 11-row convention documented at the top of
# `KernelFused.jl` (`m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi` -- physical `dec`/
# `theta_jn`, not `sin_dec`/`cos_theta_jn`) -- the conversion happens once here, at the
# sampler/likelihood boundary, via plain host-level array arithmetic (not inside a
# KernelAbstractions kernel body), which is exactly the pattern this codebase's Enzyme
# integration already relies on elsewhere in this file (`_sum_ll` below allocates `ll` at this
# same host level, not inside `@kernel`).
#
# The precessing models (`model = IMRPhenomXP()`, single- and multi-detector) append 4 more SAMPLED rows, matching
# `KernelFused.jl`'s PHYSICAL row convention extension one-for-one (see that file's module
# docstring) -- sampled DIRECTLY, no reparametrization needed (same pass-through style as rows
# 3-4/5-8/11 above):
#
#  12. chi1x -- primary in-plane spin x-component, [-1, 1]
#  13. chi1y -- primary in-plane spin y-component, [-1, 1]
#  14. chi2x -- secondary in-plane spin x-component, [-1, 1]
#  15. chi2y -- secondary in-plane spin y-component, [-1, 1]
#
# `positions`/`physical` is `(11, nwalkers)` for `IMRPhenomD`/`IMRPhenomXAS`, `(15, nwalkers)` for
# `IMRPhenomXP` -- `_reparam_to_physical!` copies rows 12-15 through unchanged when present (dispatch
# on `size(positions, 1)`, not on `model`, so it stays a plain array-arithmetic function); `_log_prior`
# gets a dedicated 15-argument method (see below) rather than optional/defaulted arguments, adding
# the bounded-box prior on the 4 new components PLUS the `chi1x^2+chi1y^2+chi1z^2 <= 1` (and chi2
# analogue) physical-magnitude constraint.
# ------------------------------------------------------------------------------------------------

@inline function _q_to_eta(q::T) where {T}
    return min(q / (one(T) + q)^2, one(T) / 4)
end

@inline function _mc_q_to_m1_m2(mc::T, q::T) where {T}
    eta = _q_to_eta(q)
    Mtot = mc / eta^(T(3) / T(5))
    m1 = Mtot / (one(T) + q)
    m2 = Mtot * q / (one(T) + q)
    return m1, m2
end

@inline function _m1_m2_to_mc_q(m1::T, m2::T) where {T}
    mu = (m1 * m2) / (m1 + m2)
    mc = (mu / (m1 + m2))^(T(3) / T(5)) * (m1 + m2)
    q = m2 / m1
    return mc, q
end

# `asin`/`acos` throw a `DomainError` (or return a `Complex`) for arguments outside [-1, 1].
# `sin_dec`/`cos_theta_jn` are hard-bounded to [-1, 1] by `_log_prior` (an out-of-bounds point
# gets `-Inf` log-posterior and is rejected/redirected by NUTS), but a leapfrog step's PROPOSED
# position can transiently land just outside [-1, 1] before that -Inf rejection is applied --
# without clamping, that would crash `Enzyme.autodiff` (and even a plain forward-pass logp
# evaluation) with a `DomainError` instead of just producing a (correctly rejected) `-Inf`.
#
# A SECOND, subtler issue (found empirically while validating the Enzyme gradients): `d(asin)/dx = 1/sqrt(1-x^2)` and `d(acos)/dx =
# -1/sqrt(1-x^2)` are GENUINELY, MATHEMATICALLY unbounded as `x -> +-1` -- this isn't a bug, it's
# an inherent property of this reparametrization at its coordinate poles (`dec = +-pi/2`,
# `theta_jn = 0` or `pi`). Clamping only to the literal closed interval `[-1, 1]` (as a first
# version of this function did) still produces `dparams = +-Inf` (not NaN, not a crash, but not
# useful to NUTS either) whenever a proposed `sin_dec`/`cos_theta_jn` lands EXACTLY at +-1 --
# confirmed via `Enzyme.autodiff` at `sin_dec = 1.0` returning an `Inf` gradient entry, which
# `test_batched_nuts_gw_smoke.jl`'s boundary testset originally caught as a test failure. Since
# `+-1` are themselves valid (non-zero-prior-density) points of the closed `[-1,1]` prior support
# -- not just a transient overshoot -- clamping to a margin strictly inside the mathematical
# domain (rather than exactly to it) is the correct fix: it trades a physically negligible loss of
# prior mass at the exact poles (a set of measure zero) for a guarantee that NUTS never sees an
# infinite gradient anywhere in its actual numerically-representable support. The margin
# (`sqrt(eps(T))`, ~1.5e-8 for Float64) keeps the worst-case `1/sqrt(1-x^2)` derivative bounded to
# a large-but-finite ~1/sqrt(2*sqrt(eps(T))) (~5800 for Float64) rather than literally `Inf`.
@inline function _clamp_unit(x::T) where {T}
    margin = sqrt(eps(T))
    return clamp(x, T(-1) + margin, T(1) - margin)
end

# Small dispatch helper so call sites that need "a" frequency grid to build a
# `PEConfig`/pass positionally to `gw_logp_and_grad` don't need to branch on whether `data` is
# single- or multi-detector themselves: single-detector uses `data.freqs` directly; multi-detector
# uses the first detector's `freqs` (only used where the caller needs SOME representative grid,
# e.g. `PEConfig.ntime`/`dt` in `run_gw150914_estimation` below, or the now-unused positional
# `freqs` argument `_eval_ll!`'s multi-detector methods accept for call-signature symmetry -- see
# `_eval_ll!`'s comment above). Detectors may have differently-SIZED grids (see
# `Types.DetectorData`'s docstring), so this is deliberately just "a" representative grid, not a
# claim that every detector shares it.
@inline _representative_freqs(data::LikelihoodData) = data.freqs
@inline _representative_freqs(data::AbstractVector{<:LikelihoodData}) = data[1].freqs

# Enzyme reverse-mode config dispatch. Empirically found (not assumed): the
# single-detector `Enzyme.Reverse` mode works fine for
# `data::LikelihoodData` -- but multi-detector's host-side setup code in
# `KernelFused.batched_log_likelihood_kernel!`/`RelativeBinning.binned_log_likelihood_kernel!`
# (which, unlike the single-detector methods, iterates `data::Vector{<:LikelihoodData}` via a
# `for`/`ntuple` loop to build the per-detector kernel-argument tuples) triggers a genuine
# `EnzymeRuntimeActivityError` ("Mismatched activity ... store atomic ptr ... Julia value causing
# error: Unknown object of type Vector{Float64}") when reverse-mode-compiled under plain
# `Enzyme.Reverse`. Root cause (confirmed via the error's own analysis, not guessed): `Vector{Float64}`
# is BOTH the type of fields read out of `Const(data)` (provably-constant, e.g. `data[i].freqs`)
# AND the type of genuinely-active values elsewhere in the same differentiated program (`ll`/
# `grad_out`); Enzyme's static whole-program activity analysis cannot always separate these when a
# dynamic loop/`ntuple`-with-closure indexes into the Vector, even though each individual value IS
# provably Const via the `Const(data)` wrapping at this call site.
#
# Fix: `Enzyme.set_runtime_activity(Enzyme.Reverse)` -- Enzyme's OWN documented, correctness-
# preserving resolution for exactly this diagnostic (suggested directly in the error message
# itself, not a workaround invented here): it defers the constant-vs-active determination to
# runtime instead of relying on static analysis, at a modest performance cost, rather than
# guaranteeing wrong results. Empirically confirmed to fix the multi-detector IMRPhenomD/IMRPhenomXAS
# fused-kernel and relative-binning Enzyme gradients --
# the resulting gradient matches finite differences to the same `rtol=1e-4, atol=1e-6` tolerance
# already used throughout this codebase's Enzyme validation, i.e. this is NOT a precision or
# correctness compromise, only a (measured) small compile-time-mode change.
#
# Scoped to ONLY the multi-detector path (not applied to `data::LikelihoodData`) so the
# single-detector Enzyme call sites and their timings are unaffected.
@inline _enzyme_reverse_mode(::LikelihoodData) = Enzyme.Reverse
@inline _enzyme_reverse_mode(::AbstractVector{<:LikelihoodData}) = Enzyme.set_runtime_activity(Enzyme.Reverse)

"""
    _reparam_to_physical!(physical, positions)

Converts the 11-row (or, for `IMRPhenomXP`, 15-row) SAMPLED parametrization (see this file's
module-level row-convention comment above) into the matching PHYSICAL parametrization
`KernelFused.jl`/`RelativeBinning.jl` consume. Only rows 1-2 (`mc,q -> m1,m2`) and rows 9-10
(`sin_dec -> dec`, `cos_theta_jn -> theta_jn`) actually transform; rows 3-4 (spins) and 5-8/11
(dL, t_c, phase, ra, psi) are direct pass-through since they're sampled in the same coordinates
the likelihood kernel consumes -- as are rows 12-15 (`chi1x, chi1y, chi2x, chi2y`), copied through
when present (`size(positions, 1) >= 15`, i.e. `IMRPhenomXP`'s 15-row space; dispatch on the array's
own shape rather than on `model`, since this function only ever sees plain arrays).

Implemented as an explicit per-walker `KernelAbstractions.@kernel`, not a whole-array
broadcast -- broadcast `.=` was tried first (see this function's own git history) and worked for
every backend/mode combination EXCEPT `Enzyme.Reverse` on `CUDABackend()`: reverse-mode
differentiating a broadcast assignment needs Enzyme's "create tape kernel" machinery to save
intermediate broadcast state for the backward pass, which wraps the `Broadcasted` object in an
`EnzymeCore.MixedDuplicated` and then asks `Adapt.jl` to convert it for the GPU kernel launch --
`Adapt` correctly converts a plain `Base.RefValue` shadow into CUDA's own `KernelRefValue`, but
`MixedDuplicated`'s `dval` field is hardcoded to the literal type `Base.RefValue{T}` (not a type
parameter), so the adapted `KernelRefValue` can never satisfy that constructor -- a genuine
EnzymeCore bug (`lib/EnzymeCore/ext/AdaptExt.jl`/`lib/EnzymeCore/src/EnzymeCore.jl`'s `MixedDuplicated`
struct), confirmed directly against a real CUDA GPU, not paraphrased from an issue tracker.
Generalizing `MixedDuplicated`'s shadow field to accept any backend's Ref-like type is a real
change to a core `Annotation` type used throughout Enzyme's whole type system -- much larger blast
radius than the narrow, already-applied `CUDACore` `EnzymeRules.width` fix. An explicit kernel
sidesteps the problem entirely instead: Enzyme differentiating an ORDINARY `KernelAbstractions.jl`
kernel (not a broadcast assignment) never invokes this "tape a broadcast" machinery at all -- it's
the exact same kernel-differentiation path already proven to work, both modes, both backends, for
every real physics kernel in `KernelFused.jl`/`RelativeBinning.jl`. Semantically identical to the
old broadcast version: copy every row of `positions` into `physical`, then overwrite rows 1-2
(`mc,q -> m1,m2`) and 9-10 (`sin_dec -> dec`, `cos_theta_jn -> theta_jn`) in place, per walker,
handling both the 11-row and (`IMRPhenomXP`'s) 15-row shapes as before.
"""
@kernel function _reparam_to_physical_kernel!(physical, positions)
    w = @index(Global, Linear)
    T = eltype(physical)
    nrows = size(positions, 1)
    @inbounds for r in 1:nrows
        physical[r, w] = positions[r, w]
    end
    one_T = one(T)
    three_fifths = T(3) / T(5)
    @inbounds mc = positions[1, w]
    @inbounds q = positions[2, w]
    @inbounds physical[1, w] = (mc / (q / (one_T + q)^2)^three_fifths) / (one_T + q)          # m1
    @inbounds physical[2, w] = (mc / (q / (one_T + q)^2)^three_fifths) * q / (one_T + q)      # m2
    @inbounds physical[9, w] = asin(_clamp_unit(positions[9, w]))    # dec, from sin_dec
    @inbounds physical[10, w] = acos(_clamp_unit(positions[10, w])) # theta_jn, from cos_theta_jn
end

function _reparam_to_physical!(physical::AbstractMatrix{T}, positions::AbstractMatrix{T}) where {T}
    backend = KernelAbstractions.get_backend(positions)
    kernel! = _reparam_to_physical_kernel!(backend)
    kernel!(physical, positions; ndrange = size(positions, 2))
    KernelAbstractions.synchronize(backend)
    return physical
end

# ------------------------------------------------------------------------------------------------
# Priors for the 11-parameter sampled space. Bounds for `(mc, q, chi1, chi2)` are unchanged from
# the pre-Task-4 4-parameter set. New bounds:
#
#   - `dL` in [0.05, 4.0] Gpc: plain uniform in distance (not the astrophysically-standard
#     uniform-in-comoving-volume / dL^2-weighted prior real GW PE pipelines use; `GWPrior` provides
#     those) covering GW150914's actual ~0.41 Gpc
#     with generous margin on both sides.
#   - `t_c` in [-0.1, 0.1] s: a small offset from the FIXED event GPS trigger time
#     (`LikelihoodData.gps_time`), not a wide search -- matches the task's explicit "+/-0.1s"
#     guidance.
#   - `phase`, `ra` in [0, 2*pi): flat priors on their natural domain, no Jacobian needed (sampled
#     directly).
#   - `sin_dec`, `cos_theta_jn` in [-1, 1]: THIS is the isotropic-sky / isotropic-orientation
#     prior. Worked through explicitly rather than assumed:
#       An isotropic (uniform-on-the-sphere) distribution over declination `dec` has probability
#       element `dP ~ cos(dec) d(dec) d(ra)` (the sphere's area element in spherical coordinates,
#       `dec` measured from the equator) -- NOT `dP ~ d(dec) d(ra)` (that would be uniform-in-
#       ANGLE, i.e. oversampling the poles relative to a true isotropic sky distribution, the
#       "classic, easy-to-miss bug" the task warns about). Changing variables to `u = sin(dec)`
#       gives `du = cos(dec) d(dec)`, so `dP ~ du d(ra)` -- i.e. isotropic-on-the-sphere is EXACTLY
#       uniform in `u = sin(dec)` (and separately uniform in `ra`), with `u` ranging over
#       `[-1, 1]` as `dec` ranges over `[-pi/2, pi/2]`. Sampling `sin_dec` directly with a flat
#       `[-1,1]` prior (this file's approach) therefore IS the isotropic sky prior with NO
#       Jacobian correction needed, by construction -- the alternative (sample `dec` directly, add
#       a `+log(cos(dec))` Jacobian term to `_log_prior`) would give the identical distribution
#       but was not the style chosen here (see the module-level comment above
#       `_reparam_to_physical!` for why the direct-coordinate style was preferred, matching this
#       file's existing `(mc, q)` pattern).
#       The identical argument applies to `theta_jn` (`dP ~ sin(theta_jn) d(theta_jn) d(psi)` for
#       an isotropic ORIENTATION distribution over the unit sphere of possible orbital angular
#       momentum directions relative to the line of sight; `v = cos(theta_jn)`, `dv =
#       -sin(theta_jn) d(theta_jn)`, so `dP ~ |dv| d(psi)` -- uniform in `v = cos(theta_jn)` on
#       `[-1, 1]`).
#   - `psi` in [0, pi): the polarization angle's natural fundamental domain (LALSuite/bilby
#     convention: `psi` and `psi+pi` are physically degenerate for this dominant-mode-only,
#     non-precessing waveform, so `[0, pi)` rather than `[0, 2*pi)` avoids double-covering).
# ------------------------------------------------------------------------------------------------

const _MC_PRIOR_BOUNDS = (10.0, 80.0)
const _Q_PRIOR_BOUNDS = (0.125, 1.0)
const _CHI_PRIOR_BOUNDS = (-1.0, 1.0)
const _DL_PRIOR_BOUNDS = (0.05, 4.0)      # Gpc
const _TC_PRIOR_BOUNDS = (-0.1, 0.1)      # seconds, offset from gps_time
const _PHASE_PRIOR_BOUNDS = (0.0, 2pi)
const _RA_PRIOR_BOUNDS = (0.0, 2pi)
const _SIN_DEC_PRIOR_BOUNDS = (-1.0, 1.0)
const _COS_THETA_JN_PRIOR_BOUNDS = (-1.0, 1.0)
const _PSI_PRIOR_BOUNDS = (0.0, pi)
# In-plane spin components (`IMRPhenomXP` only): a plain bounded-box prior, [-1, 1] each --
# documented as a simplification, NOT the standard isotropic-in-3D-spin-volume prior real PE
# pipelines use, matching this file's existing `dL` uniform-in-distance (not volume-weighted)
# precedent above. The physical constraint `chi1x^2+chi1y^2+chi1z^2 <= 1` (and the chi2 analogue)
# is enforced separately in the 15-argument `_log_prior` method below, not via these bounds alone.
const _CHI_INPLANE_PRIOR_BOUNDS = (-1.0, 1.0)

@inline function _in_bounds(x::T, bounds::Tuple) where {T}
    lo, hi = T(bounds[1]), T(bounds[2])
    return lo <= x <= hi
end

# ------------------------------------------------------------------------------------------------
# Leapfrog-time boundary enforcement, on top of (not
# instead of) the `-Inf`-outside-bounds `_log_prior` above. `_log_prior` alone is already
# correctness-safe (NUTS treats a non-finite Hamiltonian as a divergence and discards the whole
# offending subtree, see `NUTSCore.nuts_draw!`'s `subtree_valid` handling) but wasteful: a
# trajectory that would cross a tight bound gets truncated there instead of continuing to explore.
# `_sampled_space_bounds` builds the `NUTSCore.NUTSBounds`
# that lets `NUTSCore`'s leapfrog integration reflect/wrap in-place instead, using EXACTLY the same
# per-row bounds as `_log_prior`/`_..._PRIOR_BOUNDS` above so the two mechanisms never disagree.
#
# Reflect vs wrap, per row (see `NUTSBounds`'s docstring in NUTSCore.jl for the general reflect/wrap
# distinction): `phase` (row 7), `ra` (row 8), and `psi` (row 11) are periodic/angular coordinates
# -- the fused likelihood kernels' dependence on each is exactly periodic over its stated bounds
# (phase and ra with period `2*pi`; psi with period `pi`, since this non-precessing-mode waveform is
# invariant under `psi -> psi + pi`, see `_PSI_PRIOR_BOUNDS`'s comment above) -- so they use `wrap`.
# Every other row (`mc, q, chi1, chi2, dL, t_c, sin_dec, cos_theta_jn`, and, for `IMRPhenomXP`,
# `chi1x, chi1y, chi2x, chi2y`) is a genuine hard box constraint with no such identification at its
# endpoints, so they use `reflect`.
#
# The 15-row `IMRPhenomXP` case's ADDITIONAL nonlinear constraint (`chi1x^2+chi1y^2+chi1z^2 <= 1`,
# see `_log_prior`'s 15-argument method) is NOT a per-coordinate box and has no natural
# reflect/wrap treatment here -- it stays enforced only via `_log_prior`'s `-Inf` fallback (a
# divergence when violated). This is a known, accepted gap: box
# reflection handles the common/dominant case (each spin component's own linear range) and leaves
# the comparatively rare joint-norm violation to the pre-existing safety net.
# ------------------------------------------------------------------------------------------------
function _sampled_space_bounds(::Type{T}, nrows::Int) where {T}
    lo = Vector{T}(undef, nrows)
    hi = Vector{T}(undef, nrows)
    periodic = falses(nrows)

    box_rows = ((1, _MC_PRIOR_BOUNDS), (2, _Q_PRIOR_BOUNDS), (3, _CHI_PRIOR_BOUNDS), (4, _CHI_PRIOR_BOUNDS),
                (5, _DL_PRIOR_BOUNDS), (6, _TC_PRIOR_BOUNDS), (9, _SIN_DEC_PRIOR_BOUNDS),
                (10, _COS_THETA_JN_PRIOR_BOUNDS))
    wrap_rows = ((7, _PHASE_PRIOR_BOUNDS), (8, _RA_PRIOR_BOUNDS), (11, _PSI_PRIOR_BOUNDS))

    for (i, b) in box_rows
        lo[i], hi[i] = T(b[1]), T(b[2])
    end
    for (i, b) in wrap_rows
        lo[i], hi[i] = T(b[1]), T(b[2])
        periodic[i] = true
    end
    if nrows >= 15
        for i in 12:15
            lo[i], hi[i] = T(_CHI_INPLANE_PRIOR_BOUNDS[1]), T(_CHI_INPLANE_PRIOR_BOUNDS[2])
        end
    end

    return NUTSBounds(lo, hi; periodic=periodic)
end

# Bounds matching a configurable `Priors.GWPrior`'s support (same reflect/wrap split as above).
function _sampled_space_bounds(prior::GWPrior{T}) where {T}
    lo, hi, periodic = prior_bounds(prior)
    return NUTSBounds(lo, hi; periodic=periodic)
end

"""
    _log_prior(mc, q, chi1, chi2, dL, t_c, phase, ra, sin_dec, cos_theta_jn, psi)

Uniform-bounds log-prior over the full 11-parameter SAMPLED space (see this file's module-level
row-convention comment above `_reparam_to_physical!`). Every argument is checked against its own
flat `[lo, hi]` bounds and the whole thing is `0` if all are in-bounds, `-Inf` otherwise -- no
Jacobian term anywhere, because every sampled coordinate here (including `sin_dec`/`cos_theta_jn`)
already IS the coordinate a flat prior should be uniform in; see the block comment above the
`_..._PRIOR_BOUNDS` constants for the explicit isotropic-sky/isotropic-orientation
change-of-variables argument.
"""
@inline function _log_prior(mc::T, q::T, chi1::T, chi2::T, dL::T, t_c::T, phase::T, ra::T,
                             sin_dec::T, cos_theta_jn::T, psi::T) where {T}
    in_bounds = _in_bounds(mc, _MC_PRIOR_BOUNDS) && _in_bounds(q, _Q_PRIOR_BOUNDS) &&
                _in_bounds(chi1, _CHI_PRIOR_BOUNDS) && _in_bounds(chi2, _CHI_PRIOR_BOUNDS) &&
                _in_bounds(dL, _DL_PRIOR_BOUNDS) && _in_bounds(t_c, _TC_PRIOR_BOUNDS) &&
                _in_bounds(phase, _PHASE_PRIOR_BOUNDS) && _in_bounds(ra, _RA_PRIOR_BOUNDS) &&
                _in_bounds(sin_dec, _SIN_DEC_PRIOR_BOUNDS) &&
                _in_bounds(cos_theta_jn, _COS_THETA_JN_PRIOR_BOUNDS) &&
                _in_bounds(psi, _PSI_PRIOR_BOUNDS)
    return in_bounds ? zero(T) : T(-Inf)
end

"""
    _log_prior(mc, q, chi1, chi2, dL, t_c, phase, ra, sin_dec, cos_theta_jn, psi,
               chi1x, chi1y, chi2x, chi2y)

15-argument overload for the `IMRPhenomXP` SAMPLED space: the same 11-argument uniform-bounds
prior above, PLUS a bounded-box prior on the 4 in-plane spin components (`_CHI_INPLANE_PRIOR_BOUNDS`,
`[-1,1]` each) AND the physical-magnitude constraint `chi1x^2+chi1y^2+chi1z^2 <= 1` (and the `chi2`
analogue, using `chi1`/`chi2` -- rows 3-4, the aligned/z-component -- as the third component of each
spin vector) -- rejected (`-Inf`) if violated, matching this function's existing all-or-nothing
uniform-bounds style rather than e.g. clamping or renormalizing.
"""
@inline function _log_prior(mc::T, q::T, chi1::T, chi2::T, dL::T, t_c::T, phase::T, ra::T,
                             sin_dec::T, cos_theta_jn::T, psi::T,
                             chi1x::T, chi1y::T, chi2x::T, chi2y::T) where {T}
    base = _log_prior(mc, q, chi1, chi2, dL, t_c, phase, ra, sin_dec, cos_theta_jn, psi)
    isfinite(base) || return base

    in_bounds = _in_bounds(chi1x, _CHI_INPLANE_PRIOR_BOUNDS) && _in_bounds(chi1y, _CHI_INPLANE_PRIOR_BOUNDS) &&
                _in_bounds(chi2x, _CHI_INPLANE_PRIOR_BOUNDS) && _in_bounds(chi2y, _CHI_INPLANE_PRIOR_BOUNDS)
    in_bounds || return T(-Inf)

    chi1_norm2 = chi1x * chi1x + chi1y * chi1y + chi1 * chi1
    chi2_norm2 = chi2x * chi2x + chi2y * chi2y + chi2 * chi2
    (chi1_norm2 <= one(T) && chi2_norm2 <= one(T)) || return T(-Inf)

    return base
end

# Dispatch helper so `_sum_ll`/`gw_logp_and_grad` can call `_log_prior` across every walker in
# `positions` without knowing ahead of time whether it's the 11-row (IMRPhenomD/IMRPhenomXAS) or
# 15-row (IMRPhenomXP) SAMPLED space -- branches on `size(positions, 1)` (the array's own shape,
# identical for every walker in a call, hence safe to branch on inside Enzyme-differentiated
# code; it is `positions`' VALUES, not its shape, that Enzyme differentiates through) rather than
# on `model`, matching `_reparam_to_physical!`'s own dispatch-on-shape style above.
#
@inline function _call_log_prior(positions::AbstractMatrix{T}, w::Int) where {T}
    @inbounds if size(positions, 1) >= 15
        return _log_prior(positions[1, w], positions[2, w], positions[3, w], positions[4, w],
                           positions[5, w], positions[6, w], positions[7, w], positions[8, w],
                           positions[9, w], positions[10, w], positions[11, w],
                           positions[12, w], positions[13, w], positions[14, w], positions[15, w])
    else
        return _log_prior(positions[1, w], positions[2, w], positions[3, w], positions[4, w],
                           positions[5, w], positions[6, w], positions[7, w], positions[8, w],
                           positions[9, w], positions[10, w], positions[11, w])
    end
end

# Broadcasts the scalar `_log_prior` over each ROW-VIEW of `positions` (one view per sampled
# parameter) and returns the resulting length-`nwalkers` vector, rather than looping
# column-by-column with scalar `positions[i,w]` reads (`_call_log_prior` above) -- needed for
# GPU-portability (scalar `getindex` into a `CuArray`-backed `positions` throws outside
# `@allowscalar`) in the plain (non-Enzyme) forward-pass code that computes `logp_out`. NOT safe
# to `sum(...)` inside an `Enzyme`-differentiated function, though (see `_sum_ll`'s own comment on
# why it keeps the scalar loop instead) -- use `_call_log_prior` there.
@inline function _call_log_prior_vec(positions::AbstractMatrix{T}) where {T}
    if size(positions, 1) >= 15
        return _log_prior.(view(positions, 1, :), view(positions, 2, :), view(positions, 3, :),
                            view(positions, 4, :), view(positions, 5, :), view(positions, 6, :),
                            view(positions, 7, :), view(positions, 8, :), view(positions, 9, :),
                            view(positions, 10, :), view(positions, 11, :), view(positions, 12, :),
                            view(positions, 13, :), view(positions, 14, :), view(positions, 15, :))
    else
        return _log_prior.(view(positions, 1, :), view(positions, 2, :), view(positions, 3, :),
                            view(positions, 4, :), view(positions, 5, :), view(positions, 6, :),
                            view(positions, 7, :), view(positions, 8, :), view(positions, 9, :),
                            view(positions, 10, :), view(positions, 11, :))
    end
end

# Relative-binning dispatch point: whether `_sum_ll`/`gw_logp_and_grad` evaluate the likelihood
# on the full dense frequency grid (`rb === nothing`, the original path) or via the sparse
# bin-edge approximation (`rb::RelativeBinningSetup`, see RelativeBinning.jl) is resolved by
# ordinary Julia multiple dispatch on `rb`'s type, not a runtime `if` -- consistent with this
# codebase's existing model-dispatch convention (`batched_log_likelihood_kernel!` dispatching on
# `model isa IMRPhenomD`/`IMRPhenomXAS`) and deliberately avoiding a value-dependent branch inside code
# Enzyme differentiates through, matching this file's general care around Enzyme activity
# analysis (see e.g. the `Phi`/`Ampl` wrapper docstrings in IMRPhenomD.jl/IMRPhenomXAS.jl).
#
# `backend=KernelAbstractions.get_backend(physical)` on every call below (all 4 methods) --
# NOT the `batched_log_likelihood_kernel!`/`binned_log_likelihood_kernel!` default
# (`KernelAbstractions.CPU()`) -- found necessary (SEOBNR forward-mode GPU work) the
# first time `gw_logp_and_grad` itself ran with `CuArray`-backed `positions`/`data`: without an
# explicit `backend`, every `_eval_ll!` call site here silently launched the CPU-compiled kernel
# body on `CuArray` arguments (`@index`-driven `getindex` into a `CuArray` from CPU-side code),
# throwing "Scalar indexing is disallowed" instead of running on the device the arrays actually
# live on. `get_backend` infers the right `KernelAbstractions.Backend` from `physical`'s own
# array type (`CPU()` for `Array`, `CUDABackend()` for `CuArray`), so this layer never needs to
# branch on which backend is active, matching `GPU.jl`'s own `to_device`/`compatible_array`
# convention. This had never been exercised before: every prior real-GPU validation in this
# codebase called `batched_log_likelihood_kernel!`/`binned_log_likelihood_kernel!` directly with
# an explicit `backend=` kwarg, bypassing this `gw_logp_and_grad`/`_eval_ll!` layer entirely.
_eval_ll!(ll::AbstractVector{T}, physical::AbstractMatrix{T}, freqs::AbstractVector{T},
          cfg::PEConfig{T}, data::LikelihoodData, model, ::Nothing) where {T} =
    batched_log_likelihood_kernel!(ll, physical, freqs, cfg, data;
        backend=KernelAbstractions.get_backend(physical), model=model)

_eval_ll!(ll::AbstractVector{T}, physical::AbstractMatrix{T}, freqs::AbstractVector{T},
          cfg::PEConfig{T}, data::LikelihoodData, model, rb::RelativeBinningSetup{T}) where {T} =
    (_check_no_rb_options(cfg); binned_log_likelihood_kernel!(ll, physical, rb, data;
        backend=KernelAbstractions.get_backend(physical), model=model))

# Summary data of `build_xas_relative_binning` (XAS relative binning with the likelihood options).
const XASRelBin = NamedTuple{(:edges, :href, :A0, :A1, :B0, :B1, :dd, :dets, :gmst, :geo, :tref,
                              :reference, :nbins)}
const XHMRelBin = NamedTuple{(:edges, :href, :A0, :A1, :B0, :B1, :dd, :dets, :gmst, :geo, :tref,
                              :reference, :nbins, :fref0, :C0, :C1)}
# XHM relative binning with the likelihood options (PHYSICAL columns, host arrays).
function _eval_ll!(ll::AbstractVector{T}, physical::AbstractMatrix{T}, freqs::AbstractVector{T},
                   cfg::PEConfig{T}, data, model::IMRPhenomXHM, rb::XHMRelBin) where {T}
    KernelFused._check_likelihood_options(cfg, model, length(rb.edges))
    lik = KernelFused._lik_opts(cfg)
    Threads.@threads for w in axes(physical, 2)
        p = ntuple(k -> Float64(physical[k, w]), 11)
        ll[w] = T(KernelFused._xhm_ll_rb(model, p..., rb, lik))
    end
    return ll
end
# XAS relative binning with the likelihood options (PHYSICAL columns, host arrays).
function _eval_ll!(ll::AbstractVector{T}, physical::AbstractMatrix{T}, freqs::AbstractVector{T},
                   cfg::PEConfig{T}, data, model::IMRPhenomXAS, rb::XASRelBin) where {T}
    lik = KernelFused._lik_opts(cfg)
    Threads.@threads for w in axes(physical, 2)
        p = ntuple(k -> Float64(physical[k, w]), 11)
        ll[w] = T(KernelFused._xas_ll_rb(model, p..., rb, lik))
    end
    return ll
end

# Relative binning has no phase-marginalized / time-reference variant yet (except XAS, above).
_check_no_rb_options(cfg::PEConfig) =
    (cfg.marginalize_phase || cfg.marginalize_distance || cfg.time_reference != 0 || cfg.time_ref_frequency > 0 || cfg.fmax_taper > 0) && throw(ArgumentError(
        "marginalize_phase/time_reference are not implemented for relative binning; use the dense likelihood"))

# Multi-detector methods: `data` is `AbstractVector{<:LikelihoodData}`, one
# entry per detector -- see `Types.DetectorData`'s docstring for the design. `freqs` is accepted
# (unused) purely so `_sum_ll`/`gw_logp_and_grad` can call `_eval_ll!` with the SAME positional
# argument list regardless of whether `data` is single- or multi-detector (ordinary Julia multiple
# dispatch on `data`'s/`rb`'s type then routes to the right method, matching this file's existing
# no-runtime-branch-inside-the-Enzyme-differentiated-call-graph discipline). Each detector's own
# `freqs` lives inside `data[i].freqs`; `KernelFused.batched_log_likelihood_kernel!`'s
# multi-detector method reads those directly and doesn't need a separate top-level `freqs` arg.
_eval_ll!(ll::AbstractVector{T}, physical::AbstractMatrix{T}, freqs::AbstractVector{T},
          cfg::PEConfig{T}, data::AbstractVector{<:LikelihoodData}, model, ::Nothing) where {T} =
    batched_log_likelihood_kernel!(ll, physical, cfg, data;
        backend=KernelAbstractions.get_backend(physical), model=model)

_eval_ll!(ll::AbstractVector{T}, physical::AbstractMatrix{T}, freqs::AbstractVector{T},
          cfg::PEConfig{T}, data::AbstractVector{<:LikelihoodData}, model,
          rb::AbstractVector{<:RelativeBinningSetup{T}}) where {T} =
    (_check_no_rb_options(cfg); binned_log_likelihood_kernel!(ll, physical, rb, data;
        backend=KernelAbstractions.get_backend(physical), model=model))

# Plain top-level (non-closure) helper so the Enzyme.autodiff call below mirrors, argument for
# argument, the exact pattern already validated in test/test_enzyme_ka.jl: a function that
# allocates its own `ll` locally, calls `batched_log_likelihood_kernel!` (kwargs resolved by
# ordinary Julia dispatch inside the function body -- Enzyme just sees a nested call, not a kwarg
# call at the autodiff boundary itself), and returns a plain scalar. Differentiating a
# scalar-returning function this way is the pattern independently confirmed (via
# `test_enzyme_ka.jl`/`test_enzyme_ka_xas.jl`) to complete rather than hang through this
# KernelAbstractions kernel -- see the notes in `KernelFused.jl`.
#
# Returns log-POSTERIOR (log-likelihood + log-prior), not bare log-likelihood: `positions` here
# are the sampled `(mc, q, chi1, chi2)`, converted to physical `(m1, m2, chi1, chi2)` for the
# likelihood kernel via `_reparam_to_physical!` (plain host-level array arithmetic, so
# Enzyme.Reverse differentiates straight through it back to `positions`, exactly as it already
# does through `_reparam_to_physical!`'s sibling host-level allocation of `ll`/`physical` below).
#
# `rb` (a `RelativeBinningSetup` or `nothing`) is always passed `Enzyme.Const` at the autodiff
# call site below: its summary coefficients are fixed reference-waveform-derived data, not
# something to differentiate with respect to -- only `positions` (the trial walker parameters)
# needs a gradient, which flows through `_eval_ll!`'s scalar-core evaluation regardless of which
# method is picked.
function _sum_ll(positions::AbstractMatrix{T}, freqs::AbstractVector{T}, cfg::PEConfig{T},
                  data::DetectorData, model, rb, ::Nothing) where {T}
    physical = similar(positions)
    _reparam_to_physical!(physical, positions)
    ll = zeros(T, size(positions, 2))
    _eval_ll!(ll, physical, freqs, cfg, data, model, rb)
    # Deliberately NOT `sum(_call_log_prior_vec(positions))` here: found (SEOBNR
    # forward-mode GPU work) to trigger a NEW `EnzymeRuntimeActivityError` under plain
    # `Enzyme.Reverse` for every model using this function -- `sum(...)` over a freshly
    # broadcast-allocated `Vector{T}`, combined with `Const(data)` in the same call graph, hits
    # the same "Vector{Float64} ambiguity" class this file's `_enzyme_reverse_mode` comment
    # already documents for the (unrelated) multi-detector case, just via a different code path
    # (`mapreduce`, not a `for`/`ntuple` loop). `_call_log_prior_vec` is used elsewhere (GPU-safe
    # broadcast, needed there); THIS call site keeps the plain scalar loop it always had, since
    # `_sum_ll` is CPU-only regardless (blocked on GPU by the `TypeAnalysis.cpp` SIGABRT this
    # whole SEOBNR-forward-mode track exists to sidestep) and the scalar loop is already proven
    # correct here.
    lp = zero(T)
    @inbounds for w in 1:size(positions, 2)
        lp += _call_log_prior(positions, w)
    end
    return sum(ll) + lp
end

# Parallel-tempering sibling of the method above: `beta[w]` scales lane `w`'s
# log-LIKELIHOOD term only (`log_L_tempered = beta * log_L`) -- the log-prior stays untempered, exactly like every other PT scheme (a tempered prior
# would need its own per-beta renormalization; an untempered flat-box prior needs none). Dispatched
# on `beta::AbstractVector{T}` (a genuine second method, not a runtime `beta === nothing` branch
# inside the shared body) for the same reason `_eval_ll!` dispatches on `rb`'s type rather than
# branching at runtime -- see that function's own comment: avoids a value-dependent branch inside
# code Enzyme differentiates through. Kept as a single fused scalar loop (not
# `sum(beta .* ll) + lp`) for the identical reason the `::Nothing` method above keeps its own
# scalar loop instead of `sum(...)`: `beta::Vector{Float64}` (`Const` here) and the locally
# `zeros`-allocated `ll::Vector{Float64}` (active) are the same concrete type, exactly the
# "Vector{Float64} ambiguity" class `_enzyme_reverse_mode`'s comment documents -- avoiding a
# second broadcast-then-reduce over that pairing sidesteps needing `set_runtime_activity` here too
# (confirmed empirically, not assumed -- see `test/test_pt_nuts_tempering.jl`'s Enzyme-gradient
# cross-check against finite differences).
function _sum_ll(positions::AbstractMatrix{T}, freqs::AbstractVector{T}, cfg::PEConfig{T},
                  data::DetectorData, model, rb, beta::AbstractVector{T}) where {T}
    physical = similar(positions)
    _reparam_to_physical!(physical, positions)
    ll = zeros(T, size(positions, 2))
    _eval_ll!(ll, physical, freqs, cfg, data, model, rb)
    s = zero(T)
    @inbounds for w in 1:size(positions, 2)
        s += beta[w] * ll[w] + _call_log_prior(positions, w)
    end
    return s
end

# ------------------------------------------------------------------------------------------------
# SEOBNR-track-only forward-mode Enzyme gradient path: `Enzyme.autodiff(Enzyme.Reverse, ...)` through ANY KernelAbstractions kernel on
# `CUDABackend()` that differentiates `exp(im*phase)`/`cis`/`sincos` of an active argument is a
# hard SIGABRT inside Enzyme's TypeAnalysis.cpp,
# confirmed for every model in this codebase. `Enzyme.Forward` does NOT crash there (that
# machinery is reverse-mode-only), and replacing `exp(im*phase_val)` with the algebraically
# identical `Complex(cos(phase_val), sin(phase_val))` in ONLY `SEOBNRv5AlignedSpinF2`'s own kernel
# bodies (`KernelFused.jl`'s two `_fused_waveform_likelihood_kernel_seobnrv5alignedspinf2[_multi]!`,
# `RelativeBinning.jl`'s two `_relative_binning_kernel_seobnrv5alignedspinf2[_multi]!`) sidesteps a
# separate, narrower gap: `exp(::Complex)` lowers to CUDA's fused `__nv_sincosf` intrinsic, which
# has no forward-mode Enzyme rule, while plain `cos`/`sin` do. This workaround is scoped to
# SEOBNR ONLY -- every other model keeps `exp(im*...)` and
# `Enzyme.Reverse` unchanged, pending the upstream `TypeAnalysis.cpp` fix (the Phenom family/
# `AlignedSpinPNHamF2` have no standalone reason to adopt a forward-mode-only workaround now).
#
# `Enzyme.BatchDuplicated` (multiple seed directions batched into ONE autodiff call/kernel launch,
# the natural way to get a full multi-parameter gradient from forward mode cheaply) was tried
# first and found NOT to work on `CUDABackend()` at all, for any model: CUDA.jl's own
# `CUDACore.EnzymeCoreExt` forward rules for `Base.fill!` and `mapreduce`/`sum` reference
# `EnzymeRules.batch_width(A)`, an accessor that no longer exists in the installed `EnzymeCore`
# (renamed to `EnzymeRules.width(config)`) -- an upstream version-skew bug in CUDA.jl
# itself (seen with CUDACore 6.2), not a missing rule for this codebase's own kernels. Plain (non-batch) `Enzyme.Duplicated` never hits that code path (only the
# `isa BatchDuplicated` branch of those rules references `batch_width`), so it is unaffected.
#
# The actual gradient computation therefore uses `Enzyme.Forward` with a PLAIN `Duplicated` seed
# per SAMPLED PARAMETER ROW (11 rows -- `nparams` launches per leapfrog step, NOT
# `nparams*nwalkers`), exploiting the fact that every walker's likelihood is block-diagonal in
# `positions` (`_eval_ll!` never mixes one walker's columns with another's): a seed matrix that is
# 1 in row `p` across EVERY walker column and 0 elsewhere, pushed forward through a primal that
# returns the PER-WALKER LIKELIHOOD VECTOR (`_ll_vec!` below, NOT `_sum_ll`'s scalar sum), yields a
# tangent vector whose w'th entry is exactly `d(ll_w)/d(positions[p,w])` for every walker `w`
# simultaneously (off-diagonal Jacobian entries are structurally zero, so seeding all walkers' row
# `p` at once never mixes their derivatives). This is the forward-mode mirror of why `_sum_ll`'s
# reverse-mode call works: reverse mode gets the whole gradient from ONE scalar-output backward
# pass (via `sum(ll)+lp`); forward mode instead needs one push PER INPUT DIRECTION, but only
# `nparams` of them, not `nparams*nwalkers`, by never collapsing the per-walker output to a
# scalar. `_ll_vec!` omits the prior term entirely (see its own comment) rather than the
# `+lp`/`_call_log_prior` `_sum_ll` adds -- the prior's gradient is exactly zero wherever it isn't
# `-Inf`, so there is nothing for either mode to differentiate there.
@inline _uses_forward_mode_gradient(::PAF2Model) = true
@inline _uses_forward_mode_gradient(::Any) = false

# Mirrors `_sum_ll`'s physical-conversion + `_eval_ll!` pattern (same "plain top-level function,
# not a closure" Enzyme-call-site discipline) but leaves the per-walker LIKELIHOOD in `ll_out`
# rather than collapsing it to a scalar -- see the forward-mode gradient block comment above for
# why the vector, not the sum, is what forward mode needs.
#
# Deliberately does NOT add the prior term `_sum_ll` does (`... + lp`): `_log_prior` is a pure
# uniform-bounds indicator (0 inside the box, `-Inf` outside, see its definition above) -- its
# gradient is EXACTLY ZERO everywhere it is even differentiable (the interior of the box; at the
# boundary itself the posterior is `-Inf` and NUTS rejects the point using `logp_out`'s VALUE,
# already computed by `gw_logp_and_grad`'s own shared, non-Enzyme-differentiated forward pass --
# not this function). Skipping it isn't just an optimization: it also removes a second
# broadcast-then-reduce (`_call_log_prior_vec`) from inside the differentiated call graph, which
# was found to need `Enzyme.set_runtime_activity` to avoid a `Vector{Float64}`
# activity-ambiguity error when combined with `Const(data)` -- and `set_runtime_activity` in turn
# broke `_seobv5f2_coeffs`'s own static type analysis (`EnzymeNoTypeError`, unrelated to activity)
# under `Enzyme.Forward`. Dropping the (zero-gradient) prior term avoids needing
# `set_runtime_activity` at all: plain `Enzyme.Forward` differentiates `_reparam_to_physical!` +
# `_eval_ll!` cleanly on their own (confirmed in isolation and combined, CPU).
#
# `physical` is a caller-preallocated scratch buffer (`_forward_mode_gradient!`'s own
# `physical_scratch`/`dphysical_scratch`, built ONCE outside its `p in 1:nparams` loop), NOT
# allocated here via `similar(positions)` the way `_sum_ll` allocates its own `physical` -- found
# necessary on `CUDABackend()`: `Enzyme.autodiff(Enzyme.Forward, ...)` compiling a function that
# itself calls `similar` on a `CuArray`-typed `Duplicated` argument hits an internal LLVM-level
# assertion in Enzyme's forward-mode shadow-allocation code (`zero_single_allocation`/
# `create_recursive_stores`, "Number of non-ghost elements ... did not match") -- a genuine
# Enzyme/CUDA limitation around allocating NEW device arrays inside a differentiated function
# under forward mode, not something wrong with this codebase's own logic (the exact same
# `_reparam_to_physical!`+`_eval_ll!` combination already differentiates cleanly on CPU). Passing
# `physical` in as an already-allocated `Duplicated` argument instead sidesteps it entirely, the
# same "preallocate outside, mutate inside" discipline `ll_out`/`seed` already follow below.
function _ll_vec!(ll_out::AbstractVector{T}, physical::AbstractMatrix{T}, positions::AbstractMatrix{T},
                   freqs::AbstractVector{T}, cfg::PEConfig{T}, data::DetectorData, model, rb) where {T}
    _reparam_to_physical!(physical, positions)
    _eval_ll!(ll_out, physical, freqs, cfg, data, model, rb)
    return nothing
end

# Single-detector, dense-kernel-only fast path for `SEOBNRv5AlignedSpinF2`'s forward-mode
# gradient. It uses `KernelFused._fused_ll_kernel_seobnrv5alignedspinf2_precoeffs!` instead of
# `Enzyme.autodiff`-ing `_seobv5f2_coeffs`'s own construction: `coeffs` is built ONCE (ordinary,
# non-differentiated forward pass, `_seobv5f2_build_coeffs_soa_kernel!`) and reused across all
# `nparams` seed directions below, and its TANGENT (needed only for the 4 sampled rows that reach
# it -- `mc`, `q`, `chi1`, `chi2`) comes from `SEOBNRv5AlignedSpinF2Adjoint.seobv5f2_coeffs_jvp`
# instead of Enzyme tracing the PA/Newton-solve construction -- see that module's own docstring for
# why (sidesteps the three upstream Enzyme/CUDA bugs this file's own comments document elsewhere).
# Enzyme is still used for everything else (`_reparam_to_physical!`'s own sampled->physical
# Jacobian, and the bracket-search reconstruction/antenna-response/residual-sum kernel body) --
# exactly the code this codebase's existing tests already prove Enzyme differentiates safely.
#
# The coeffs-tangent computation runs on the HOST regardless of `positions`' own backend (mirrors
# `geom_flat`/`gmst_arr`'s own build-on-host-then-`copyto!` pattern below): `seobv5f2_coeffs_jvp`
# calls `Enzyme.autodiff` internally on several small functions of its own, which is not the
# supported way to call it from INSIDE an already-differentiated kernel launch (nested autodiff) --
# doing it as an ordinary top-level Julia call on `nwalkers` small scalars, then `copyto!`-ing the
# result to the device-resident SoA buffers, sidesteps that question entirely rather than assuming
# it would work.
#
# `binned_log_likelihood_kernel!`/multi-detector forward-mode gradients still go through the
# generic `_ll_vec!`/`_forward_mode_gradient!` pair above -- extending this precoeffs design to
# those kernels is a mechanical follow-up (same technique, different kernel bodies), not attempted
# here.
#
# **`Enzyme.set_runtime_activity(Enzyme.Forward)` IS needed on the call below** -- without it,
# `Enzyme.autodiff` throws `EnzymeRuntimeActivityError` ("Constant memory is stored to a
# differentiable variable"), the same activity-ambiguity class this file's own standing gotchas
# document elsewhere, here from mixing several same-concrete-type (`Vector{Float64}`,
# `Matrix{Float64}`) `Const` and `Duplicated` arguments in one call. Safe to use here: with the PA driver inside the
# differentiated call it broke `_seobv5f2_coeffs`'s OWN static type analysis (`EnzymeNoTypeError`,
# inside its `Val{256}`-parameterized PA driver), but that driver is not in this differentiated
# call graph, so there is nothing for `set_runtime_activity` to break.
#
# Test: `test/test_seobnr_forward_mode_gradient.jl` (full `gw_logp_and_grad` closure
# vs. finite differences, all 11 parameters, both walkers).
function _ll_given_coeffs_wrapper!(ll_out::AbstractVector{T}, physical::AbstractMatrix{T}, kernel!,
        ts_soa::AbstractMatrix{Float64}, phis_soa::AbstractMatrix{Float64}, omegas_soa::AbstractMatrix{Float64},
        Msec_soa::AbstractVector{Float64}, eta_soa::AbstractVector{Float64}, M_soa::AbstractVector{Float64},
        fmin_soa::AbstractVector{Float64}, fmax_soa::AbstractVector{Float64},
        nreal_soa::AbstractVector{Int}, success_soa::AbstractVector{Bool},
        freqs::AbstractVector{T}, strain::AbstractVector, psd::AbstractVector,
        geom_flat::AbstractVector{Float64}, gmst_arr::AbstractVector{Float64},
        model::PAF2Model, nwalkers::Int, lik_arr::AbstractVector{Float64}) where {T}
    kernel!(ll_out, physical, ts_soa, phis_soa, omegas_soa, Msec_soa, eta_soa, M_soa, fmin_soa, fmax_soa,
            nreal_soa, success_soa, freqs, strain, psd, geom_flat, gmst_arr, model, lik_arr; ndrange = nwalkers)
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(physical))
    return nothing
end

# Multi-detector sibling of `_ll_given_coeffs_wrapper!` (drives
# `KernelFused._fused_ll_kernel_seobnrv5alignedspinf2_precoeffs_multi!`).
function _ll_given_coeffs_multi_wrapper!(ll_out::AbstractVector{T}, physical::AbstractMatrix{T}, kernel!,
        ts_soa::AbstractMatrix{Float64}, phis_soa::AbstractMatrix{Float64}, omegas_soa::AbstractMatrix{Float64},
        Msec_soa::AbstractVector{Float64}, eta_soa::AbstractVector{Float64}, M_soa::AbstractVector{Float64},
        fmin_soa::AbstractVector{Float64}, fmax_soa::AbstractVector{Float64},
        nreal_soa::AbstractVector{Int}, success_soa::AbstractVector{Bool},
        freqs_tuple, strain_tuple, psd_tuple, geom_arr::AbstractVector{Float64},
        gmst_arr::AbstractVector{Float64}, model::PAF2Model, nwalkers::Int, lik_arr::AbstractVector{Float64}, tref::Int) where {T}
    kernel!(ll_out, physical, ts_soa, phis_soa, omegas_soa, Msec_soa, eta_soa, M_soa, fmin_soa, fmax_soa,
            nreal_soa, success_soa, freqs_tuple, strain_tuple, psd_tuple, geom_arr,
            gmst_arr, model, lik_arr, tref; ndrange = nwalkers)
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(physical))
    return nothing
end

function _chirp_mass_eta_jvp(m1, m2, dm1, dm2)
    (res,) = Enzyme.autodiff(Enzyme.Forward, KernelFused._chirp_mass_eta,
        Enzyme.Duplicated(m1, dm1), Enzyme.Duplicated(m2, dm2))
    return res  # (dmc, deta)
end

# Reverse counterpart of `_chirp_mass_eta_jvp`, used by `_reverse_mode_gradient!` below. Two-output
# scalar leaf -> seeded-wrapper technique (same as every multi-output reverse call in
# `SEOBNRv5AlignedSpinF2Adjoint.jl`): reverse-differentiate `b1*mc+b2*eta` to get the VJP
# contraction against seed `(b1,b2)` directly.
@inline function _chirp_mass_eta_seeded(m1, m2, b1, b2)
    mc, eta = KernelFused._chirp_mass_eta(m1, m2)
    return b1 * mc + b2 * eta
end
function _chirp_mass_eta_vjp(m1, m2, bar_mc, bar_eta)
    (g,) = Enzyme.autodiff(Enzyme.Reverse, _chirp_mass_eta_seeded,
        Enzyme.Active(m1), Enzyme.Active(m2), Enzyme.Const(bar_mc), Enzyme.Const(bar_eta))
    return g[1], g[2]  # (bar_m1, bar_m2)
end

# Opt-in switch (default `false`): flips the stage-2 CUDA branch, further down this function, from the
# sequential per-direction form to the same Enzyme.BatchDuplicated form the CPU branch uses.
# `Enzyme.BatchDuplicated` through a kernel launch on CUDABackend() fails when CUDACore's
# EnzymeCoreExt still calls `EnzymeRules.batch_width(A)` (renamed upstream to
# `EnzymeRules.width(config)`); with those call sites renamed it works. Set this to `true` only with
# a CUDACore that has the rename.
const _EXPERIMENTAL_BATCHED_CUDA_GRADIENT = Ref(false)

function _forward_mode_gradient!(grad_out::AbstractMatrix{T}, positions::AbstractMatrix{T},
                                  freqs::AbstractVector{T}, cfg::PEConfig{T}, data::LikelihoodData,
                                  model::PAF2Model, rb::Nothing) where {T}
    nparams, nwalkers = size(positions)
    backend = KernelAbstractions.get_backend(positions)
    geom_flat = KernelFused._geometry_array((data,), backend)
    gmst_arr = KernelAbstractions.allocate(backend, Float64, 1)
    copyto!(gmst_arr, [gmst(data.gps_time)])

    physical = similar(positions)
    _reparam_to_physical!(physical, positions)

    ts_soa = KernelAbstractions.allocate(backend, Float64, 256, nwalkers)
    phis_soa = similar(ts_soa)
    omegas_soa = similar(ts_soa)
    Msec_soa = KernelAbstractions.allocate(backend, Float64, nwalkers)
    eta_soa = similar(Msec_soa)
    M_soa = similar(Msec_soa)
    fmin_soa = similar(Msec_soa)
    fmax_soa = similar(Msec_soa)
    nreal_soa = KernelAbstractions.allocate(backend, Int, nwalkers)
    success_soa = KernelAbstractions.allocate(backend, Bool, nwalkers)
    build_kernel! = KernelFused._seobv5f2_build_coeffs_soa_kernel!(backend)
    build_kernel!(ts_soa, phis_soa, omegas_soa, Msec_soa, eta_soa, M_soa, fmin_soa, fmax_soa,
                  nreal_soa, success_soa, physical, model; ndrange = nwalkers)
    KernelAbstractions.synchronize(backend)

    m1_h = Array(view(physical, 1, :))
    m2_h = Array(view(physical, 2, :))
    chi1_h = Array(view(physical, 3, :))
    chi2_h = Array(view(physical, 4, :))
    mc_h = similar(m1_h)
    eta_h = similar(m1_h)
    for w in 1:nwalkers
        mc_h[w], eta_h[w] = KernelFused._chirp_mass_eta(m1_h[w], m2_h[w])
    end

    ll_kernel! = KernelFused._fused_ll_kernel_seobnrv5alignedspinf2_precoeffs!(backend)
    seed = similar(positions)
    dphysical = similar(physical)

    dts_h = zeros(Float64, 256, nwalkers)
    dphis_h = zeros(Float64, 256, nwalkers)
    domegas_h = zeros(Float64, 256, nwalkers)
    dMsec_h = zeros(Float64, nwalkers)
    deta_h = zeros(Float64, nwalkers)
    dM_h = zeros(Float64, nwalkers)
    dfmin_h = zeros(Float64, nwalkers)
    dfmax_h = zeros(Float64, nwalkers)

    # This model always samples exactly 11 parameters (fixed physics, not user-configurable --
    # see e.g. run_from_config.jl's own `nrows` convention). `Val(11)`, not the runtime `nparams`,
    # is load-bearing for the `ntuple` calls below: `ntuple(f, n::Integer)` falls back to Base's
    # generic, non-generated implementation for large-ish `n`, which previously produced invalid
    # LLVM IR under Enzyme on CUDABackend() for an unrelated SVector reconstruction in this same
    # model's kernel (see KernelFused.jl's own `Val(256)` fix) -- `Val(11)` uses the
    # `@generated`, fully-unrolled path instead, sidestepping that whole class of bug up front
    # rather than waiting to hit it.
    @assert nparams == 11 "$(nameof(typeof(model))) always samples 11 parameters, got $nparams"

    # Stage 1 (cheap): per-direction physical-space + coeffs sensitivities, exactly as before,
    # just stored into per-direction buffers instead of one reused scratch buffer -- BatchDuplicated
    # below needs all 11 directions' shadows alive simultaneously, not one at a time.
    dphysical_all = ntuple(_ -> similar(physical), Val(11))
    dts_all = ntuple(_ -> similar(ts_soa), Val(11))
    dphis_all = ntuple(_ -> similar(phis_soa), Val(11))
    domegas_all = ntuple(_ -> similar(omegas_soa), Val(11))
    dMsec_all = ntuple(_ -> similar(Msec_soa), Val(11))
    deta_all = ntuple(_ -> similar(eta_soa), Val(11))
    dM_all = ntuple(_ -> similar(M_soa), Val(11))
    dfmin_all = ntuple(_ -> similar(fmin_soa), Val(11))
    dfmax_all = ntuple(_ -> similar(fmax_soa), Val(11))

    @inbounds for p in 1:nparams
        fill!(seed, zero(T))
        seed[p, :] .= one(T)
        fill!(dphysical, zero(T))
        Enzyme.autodiff(Enzyme.Forward, _reparam_to_physical!,
            Enzyme.Duplicated(physical, dphysical), Enzyme.Duplicated(positions, seed))
        copyto!(dphysical_all[p], dphysical)

        dm1_h = Array(view(dphysical, 1, :))
        dm2_h = Array(view(dphysical, 2, :))
        dchi1_h = Array(view(dphysical, 3, :))
        dchi2_h = Array(view(dphysical, 4, :))

        if iszero(dm1_h) && iszero(dm2_h) && iszero(dchi1_h) && iszero(dchi2_h)
            fill!(dts_all[p], zero(T)); fill!(dphis_all[p], zero(T)); fill!(domegas_all[p], zero(T))
            fill!(dMsec_all[p], zero(T)); fill!(deta_all[p], zero(T)); fill!(dM_all[p], zero(T))
            fill!(dfmin_all[p], zero(T)); fill!(dfmax_all[p], zero(T))
        else
            # Walkers are independent; runs in parallel when Julia is started with -t N.
            Threads.@threads for w in 1:nwalkers
                dmc_w, deta_w = _chirp_mass_eta_jvp(m1_h[w], m2_h[w], dm1_h[w], dm2_h[w])
                dcoeffs_w = SEOBNRv5AlignedSpinF2Adjoint.seobv5f2_coeffs_jvp(model, mc_h[w], eta_h[w], chi1_h[w], chi2_h[w],
                    dmc_w, deta_w, dchi1_h[w], dchi2_h[w])
                @views dts_h[:, w] .= dcoeffs_w.ts
                @views dphis_h[:, w] .= dcoeffs_w.phis
                @views domegas_h[:, w] .= dcoeffs_w.omegas
                dMsec_h[w] = dcoeffs_w.M_sec
                deta_h[w] = dcoeffs_w.eta
                dM_h[w] = dcoeffs_w.M
                dfmin_h[w] = dcoeffs_w.fmin
                dfmax_h[w] = dcoeffs_w.fmax
            end
            copyto!(dts_all[p], dts_h); copyto!(dphis_all[p], dphis_h); copyto!(domegas_all[p], domegas_h)
            copyto!(dMsec_all[p], dMsec_h); copyto!(deta_all[p], deta_h); copyto!(dM_all[p], dM_h)
            copyto!(dfmin_all[p], dfmin_h); copyto!(dfmax_all[p], dfmax_h)
        end
    end

    # Stage 2 (the expensive part). On CPU: ONE Enzyme.BatchDuplicated autodiff call carrying all
    # 11 directions' shadows at once, instead of 11 separate Enzyme.autodiff calls each re-paying
    # Enzyme's own per-call dispatch/thunk-lookup overhead. Validated (standalone
    # cross-check against central finite differences, all 11 params, on this exact real GW150914 H1
    # band): max relative error ~1e-40 (bitwise-equivalent), ~2.9x faster post-compile than the
    # sequential form on CPU.
    #
    # On CUDA: this function is ALSO the dispatch target `gw_logp_and_grad_gpu_staged` reaches
    # (both call sites share this same `data::LikelihoodData` method -- there is no separate
    # GPU-specific method), so `backend` here can genuinely be `CUDABackend()`, not just `CPU()`.
    # `Enzyme.BatchDuplicated` is documented elsewhere in this file as broken on CUDABackend() for
    # every model (a CUDACore.EnzymeCoreExt/EnzymeCore version-skew bug in CUDA-specific forward
    # rules for `fill!`/`mapreduce`/`sum`) -- untested whether this SPECIFIC kernel body actually
    # exercises the broken rules, and not worth the risk of finding out via a silent wrong gradient
    # in the sampler that's supposed to be this project's real EOB/SEOBNR result. Branch on
    # `backend` and keep the original, already real-CUDA-validated sequential per-direction form
    # for anything that isn't `KernelAbstractions.CPU()`.
    ll_scratch = similar(positions, T, nwalkers)
    if backend isa KernelAbstractions.CPU || _EXPERIMENTAL_BATCHED_CUDA_GRADIENT[]
        dll_all = ntuple(_ -> similar(ll_scratch), Val(11))
        fill!(ll_scratch, zero(T))
        foreach(fill!, dll_all, ntuple(_ -> zero(T), Val(11)))
        Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Forward), _ll_given_coeffs_wrapper!,
            Enzyme.BatchDuplicated(ll_scratch, dll_all),
            Enzyme.BatchDuplicated(physical, dphysical_all),
            Enzyme.Const(ll_kernel!),
            Enzyme.BatchDuplicated(ts_soa, dts_all), Enzyme.BatchDuplicated(phis_soa, dphis_all), Enzyme.BatchDuplicated(omegas_soa, domegas_all),
            Enzyme.BatchDuplicated(Msec_soa, dMsec_all), Enzyme.BatchDuplicated(eta_soa, deta_all), Enzyme.BatchDuplicated(M_soa, dM_all),
            Enzyme.BatchDuplicated(fmin_soa, dfmin_all), Enzyme.BatchDuplicated(fmax_soa, dfmax_all),
            Enzyme.Const(nreal_soa), Enzyme.Const(success_soa),
            Enzyme.Const(freqs), Enzyme.Const(data.strain), Enzyme.Const(data.psd),
            Enzyme.Const(geom_flat), Enzyme.Const(gmst_arr), Enzyme.Const(model), Enzyme.Const(nwalkers),
            Enzyme.Const(KernelFused._lik_array(cfg, backend)))
        @inbounds for p in 1:nparams
            grad_out[p, :] .= dll_all[p]
        end
    else
        dll_scratch = similar(ll_scratch)
        @inbounds for p in 1:nparams
            fill!(ll_scratch, zero(T))
            fill!(dll_scratch, zero(T))
            Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Forward), _ll_given_coeffs_wrapper!,
                Enzyme.Duplicated(ll_scratch, dll_scratch),
                Enzyme.Duplicated(physical, dphysical_all[p]),
                Enzyme.Const(ll_kernel!),
                Enzyme.Duplicated(ts_soa, dts_all[p]), Enzyme.Duplicated(phis_soa, dphis_all[p]), Enzyme.Duplicated(omegas_soa, domegas_all[p]),
                Enzyme.Duplicated(Msec_soa, dMsec_all[p]), Enzyme.Duplicated(eta_soa, deta_all[p]), Enzyme.Duplicated(M_soa, dM_all[p]),
                Enzyme.Duplicated(fmin_soa, dfmin_all[p]), Enzyme.Duplicated(fmax_soa, dfmax_all[p]),
                Enzyme.Const(nreal_soa), Enzyme.Const(success_soa),
                Enzyme.Const(freqs), Enzyme.Const(data.strain), Enzyme.Const(data.psd),
                Enzyme.Const(geom_flat), Enzyme.Const(gmst_arr), Enzyme.Const(model), Enzyme.Const(nwalkers),
                Enzyme.Const(KernelFused._lik_array(cfg, backend)))
            grad_out[p, :] .= dll_scratch
        end
    end
    return nothing
end

# Whether to use the reverse-mode `_reverse_mode_gradient!` for SEOBNRv5AlignedSpinF2 (single-detector, dense
# data, no relative binning -- everything else falls back to forward mode in `gw_logp_and_grad`'s dispatch).
# DEFAULT: ON iff the loaded GravitationalWaveforms has the constant-trip-count `_seobv5f2_searchsortedlast`, OFF otherwise.
#
# Why that guard: on CUDA this path used to crash deterministically with `CUDA error: an illegal memory access`
# (code 700) after exactly ~11 repeated calls: a DEVICE-HEAP LEAK -- Enzyme's reverse
# pass caches a loop with a data-dependent trip count (here `Base.searchsortedlast` in GravitationalWaveforms'
# `_seobv5f2_dense_to_freq`) with its "exponential allocation" scheme, whose GPU grow step `malloc`s a bigger
# buffer, `memcpy`s and never `free`s the old one (~0.5MB per 16-walker call vs the 8MB default device heap).
# Fixed in GravitationalWaveforms by the constant-trip-count search; validated on real CUDA hardware (repeated-call
# regression test `test_seobnr_reverse_mode_gradient_cuda_repeated.jl`, and a real GW150914 H1 NUTS run of ~130
# reverse-mode gradient calls, ~1.4x faster per full-depth draw than forward mode). With an OLDER GravitationalWaveforms
# (no fix) the default silently stays forward mode instead of crashing.
#
# Set it explicitly to override (`scripts/run_from_config.jl`'s `sampler.gradient_mode = "forward"|"reverse"|"auto"`
# does exactly that). The name still says EXPERIMENTAL for compatibility with existing callers/tests; it no
# longer is. Any OTHER unknown-trip-count loop added inside a differentiated GPU kernel will leak the same
# way.
_gwwaveforms_has_constant_trip_search() =
    isdefined(parentmodule(KernelFused._seobv5f2_phi_scalar), :_seobv5f2_searchsortedlast)
const _EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT = Ref(_gwwaveforms_has_constant_trip_search())

"""
    _reverse_mode_gradient!(grad_out, positions, freqs, cfg, data, model::SEOBNRv5AlignedSpinF2, rb::Nothing)

Reverse-mode analogue of `_forward_mode_gradient!`'s SEOBNR-specific method above: ONE
`Enzyme.Reverse` call on `_ll_given_coeffs_wrapper!` (vs. that method's `nparams` forward calls)
gets the direct-dependence part of the gradient (rows 5-11's extrinsic parameters, PLUS whatever
direct `m1,m2,chi1,chi2` dependence `_seobv5f2_ll_given_coeffs`'s own `_extrinsic_setup` call has,
separate from `coeffs`) in one pass, together with cotangents on `coeffs`'s own differentiable
fields (`ts_soa`/`phis_soa`/`omegas_soa`/`Msec_soa`/`eta_soa`/`M_soa`/`fmin_soa`/`fmax_soa`).
`SEOBNRv5AlignedSpinF2Adjoint.seobv5f2_coeffs_vjp` then converts those coeffs-field cotangents into
`(bar_mc,bar_eta,bar_chi1,bar_chi2)` per walker in ONE call each (vs. `seobv5f2_coeffs_jvp` needing
one call per SAMPLED direction), `_chirp_mass_eta_vjp` pushes `(bar_mc,bar_eta)` back to
`(bar_m1,bar_m2)`, and a final `Enzyme.Reverse` call on `_reparam_to_physical!` (already made
reverse-mode-safe as an explicit `KernelAbstractions.@kernel`, not a broadcast -- see that
function's own docstring) converts the completed `bar_physical` into `grad_out`.

Device-generic (`backend = KernelAbstractions.get_backend(positions)`, any `KernelAbstractions`
backend), mirroring `_forward_mode_gradient!`'s own host/device split: the big per-walker
`ts_soa`/`phis_soa`/`omegas_soa`/... buffers and the two `Enzyme.autodiff` calls stay on-device
(`KernelAbstractions.allocate`/`similar`-of-a-device-array throughout, never a bare `zeros(...)`),
while the small, non-AD analytic `seobv5f2_coeffs_vjp`/`_chirp_mass_eta`/`_chirp_mass_eta_vjp` maths
round-trips through host `Array`s per walker -- exactly the same shape `_forward_mode_gradient!`
already uses for `seobv5f2_coeffs_jvp`, just mirrored for the reverse direction (host->device
staging instead of device->host). Single-detector only, matching `_forward_mode_gradient!`'s
own `data::LikelihoodData` (not `DetectorData`) scope. Selected by
`_EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[]` (see that flag's own docstring), which defaults to ON when the loaded
GravitationalWaveforms has the device-heap-leak fix and OFF otherwise.
"""
function _reverse_mode_gradient!(grad_out::AbstractMatrix{T}, positions::AbstractMatrix{T},
                                  freqs::AbstractVector{T}, cfg::PEConfig{T}, data::DetectorData,
                                  model::PAF2Model, rb::Nothing) where {T}
    nparams, nwalkers = size(positions)
    @assert nparams == 11 "$(nameof(typeof(model))) always samples 11 parameters, got $nparams"
    backend = KernelAbstractions.get_backend(positions)

    physical = similar(positions)
    _reparam_to_physical!(physical, positions)

    ts_soa = KernelAbstractions.allocate(backend, Float64, 256, nwalkers)
    phis_soa = similar(ts_soa)
    omegas_soa = similar(ts_soa)
    Msec_soa = KernelAbstractions.allocate(backend, Float64, nwalkers)
    eta_soa = similar(Msec_soa)
    M_soa = similar(Msec_soa)
    fmin_soa = similar(Msec_soa)
    fmax_soa = similar(Msec_soa)
    nreal_soa = KernelAbstractions.allocate(backend, Int, nwalkers)
    success_soa = KernelAbstractions.allocate(backend, Bool, nwalkers)
    build_kernel! = KernelFused._seobv5f2_build_coeffs_soa_kernel!(backend)
    build_kernel!(ts_soa, phis_soa, omegas_soa, Msec_soa, eta_soa, M_soa, fmin_soa, fmax_soa,
                  nreal_soa, success_soa, physical, model; ndrange = nwalkers)
    KernelAbstractions.synchronize(backend)

    m1_h = Array(view(physical, 1, :))
    m2_h = Array(view(physical, 2, :))
    chi1_h = Array(view(physical, 3, :))
    chi2_h = Array(view(physical, 4, :))
    mc_h = similar(m1_h)
    eta_h = similar(m1_h)
    for w in 1:nwalkers
        mc_h[w], eta_h[w] = KernelFused._chirp_mass_eta(m1_h[w], m2_h[w])
    end

    ll_scratch = similar(positions, T, nwalkers)
    fill!(ll_scratch, zero(T))
    bar_ll = similar(positions, T, nwalkers)
    fill!(bar_ll, one(T))       # sum_w ll[w]'s gradient == each ll[w]'s own gradient, per-walker,
                                # since the likelihood is block-diagonal in `physical`'s columns
                                # (same trick `_sum_ll` uses elsewhere in this file).

    bar_physical = similar(physical)
    fill!(bar_physical, zero(T))
    bar_ts = similar(ts_soa); fill!(bar_ts, zero(T))
    bar_phis = similar(phis_soa); fill!(bar_phis, zero(T))
    bar_omegas = similar(omegas_soa); fill!(bar_omegas, zero(T))
    bar_Msec = similar(Msec_soa); fill!(bar_Msec, zero(T))
    bar_eta_soa = similar(eta_soa); fill!(bar_eta_soa, zero(T))
    bar_M = similar(M_soa); fill!(bar_M, zero(T))
    bar_fmin = similar(fmin_soa); fill!(bar_fmin, zero(T))
    bar_fmax = similar(fmax_soa); fill!(bar_fmax, zero(T))

    _paf2_ll_reverse!(data, cfg, backend, model, nwalkers, freqs,
        ll_scratch, bar_ll, physical, bar_physical,
        ts_soa, bar_ts, phis_soa, bar_phis, omegas_soa, bar_omegas,
        Msec_soa, bar_Msec, eta_soa, bar_eta_soa, M_soa, bar_M,
        fmin_soa, bar_fmin, fmax_soa, bar_fmax, nreal_soa, success_soa)

    # ONE seobv5f2_coeffs_vjp call per walker (vs. seobv5f2_coeffs_jvp's own nparams calls) converts
    # the coeffs-field cotangents into (bar_mc,bar_eta,bar_chi1,bar_chi2), then _chirp_mass_eta_vjp
    # pushes (bar_mc,bar_eta) back to (bar_m1,bar_m2) -- both added ONTO whatever bar_physical[1:4,:]
    # the direct Enzyme.Reverse call above already accumulated (physical's rows 1-4 feed the
    # likelihood BOTH directly, via _extrinsic_setup, AND indirectly via coeffs; these two
    # contributions are independent and additive, not alternatives).
    # Pull the (small, 256-or-1-per-walker) coeffs-field cotangents to host for the analytic VJP --
    # `seobv5f2_coeffs_vjp` is plain scalar/SVector Julia, not itself Enzyme-differentiated or
    # GPU-kernelized, exactly like `seobv5f2_coeffs_jvp`'s own host round-trip in
    # `_forward_mode_gradient!` above (same reasoning, opposite direction: device->host here instead
    # of host->device there).
    bar_ts_h = Array(bar_ts); bar_phis_h = Array(bar_phis); bar_omegas_h = Array(bar_omegas)
    bar_Msec_h = Array(bar_Msec); bar_eta_soa_h = Array(bar_eta_soa); bar_M_h = Array(bar_M)
    bar_fmin_h = Array(bar_fmin); bar_fmax_h = Array(bar_fmax)
    bar_physical14_h = zeros(T, 4, nwalkers)
    # Walkers are independent; runs in parallel when Julia is started with -t N.
    Threads.@threads for w in 1:nwalkers
        bar_mc_w, bar_eta_w, bar_chi1_w, bar_chi2_w = SEOBNRv5AlignedSpinF2Adjoint.seobv5f2_coeffs_vjp(
            model, mc_h[w], eta_h[w], chi1_h[w], chi2_h[w],
            SVector{256,Float64}(view(bar_ts_h, :, w)), SVector{256,Float64}(view(bar_phis_h, :, w)),
            SVector{256,Float64}(view(bar_omegas_h, :, w)),
            bar_Msec_h[w], bar_eta_soa_h[w], bar_M_h[w], bar_fmin_h[w], bar_fmax_h[w])
        bar_m1_w, bar_m2_w = _chirp_mass_eta_vjp(m1_h[w], m2_h[w], bar_mc_w, bar_eta_w)
        bar_physical14_h[1, w] = bar_m1_w
        bar_physical14_h[2, w] = bar_m2_w
        bar_physical14_h[3, w] = bar_chi1_w
        bar_physical14_h[4, w] = bar_chi2_w
    end
    # `bar_physical` may be a device array (CUDA disallows scalar indexing) -- stage the coeffs-vjp
    # contribution on host, upload it, and ADD (not overwrite: rows 1-4 already carry whatever direct
    # `physical`-dependence the Enzyme.Reverse call above accumulated, independent of and additive
    # with this coeffs-mediated contribution, per this function's own docstring).
    bar_physical14 = copyto!(similar(bar_physical, 4, nwalkers), bar_physical14_h)
    view(bar_physical, 1:4, :) .+= bar_physical14

    fill!(grad_out, zero(T))
    Enzyme.autodiff(Enzyme.Reverse, _reparam_to_physical_novalue!,
        Enzyme.Duplicated(physical, bar_physical), Enzyme.Duplicated(positions, grad_out))

    return nothing
end

# The one data-shape-dependent step of `_reverse_mode_gradient!`: reverse-differentiate the
# precoeffs likelihood kernel, accumulating into `bar_physical` and the coeffs-field cotangents.
function _paf2_ll_reverse!(data::LikelihoodData, cfg::PEConfig{T}, backend, model, nwalkers, freqs,
        ll_scratch, bar_ll, physical, bar_physical,
        ts_soa, bar_ts, phis_soa, bar_phis, omegas_soa, bar_omegas,
        Msec_soa, bar_Msec, eta_soa, bar_eta_soa, M_soa, bar_M,
        fmin_soa, bar_fmin, fmax_soa, bar_fmax, nreal_soa, success_soa) where {T}
    geom_flat = KernelFused._geometry_array((data,), backend)
    gmst_arr = KernelAbstractions.allocate(backend, Float64, 1)
    copyto!(gmst_arr, [gmst(data.gps_time)])
    ll_kernel! = KernelFused._fused_ll_kernel_seobnrv5alignedspinf2_precoeffs!(backend)

    # Plain `Enzyme.Reverse`, NOT `set_runtime_activity` -- mirrors this file's own established
    # `_enzyme_reverse_mode(::LikelihoodData) = Enzyme.Reverse` convention (single-detector data
    # doesn't need it; only the multi-detector `AbstractVector{<:LikelihoodData}` path does, see that
    # dispatch's own docstring above). Confirmed empirically for THIS call site (not assumed by
    # analogy): removing it here produced a bitwise-identical gradient (max abs diff 0.0 across all 9
    # duplicated buffers) on real CUDA hardware, while cutting this call's own per-call cost ~47% (a
    # real GPU profiling pass on this exact kernel found `set_runtime_activity` roughly doubling the
    # cost of an already-expensive `ReverseModeSplit` two-kernel (`gpu_aug_fwd`+`gpu_rev`) launch).
    Enzyme.autodiff(Enzyme.Reverse, _ll_given_coeffs_wrapper!,
        Enzyme.Duplicated(ll_scratch, bar_ll),
        Enzyme.Duplicated(physical, bar_physical),
        Enzyme.Const(ll_kernel!),
        Enzyme.Duplicated(ts_soa, bar_ts), Enzyme.Duplicated(phis_soa, bar_phis), Enzyme.Duplicated(omegas_soa, bar_omegas),
        Enzyme.Duplicated(Msec_soa, bar_Msec), Enzyme.Duplicated(eta_soa, bar_eta_soa), Enzyme.Duplicated(M_soa, bar_M),
        Enzyme.Duplicated(fmin_soa, bar_fmin), Enzyme.Duplicated(fmax_soa, bar_fmax),
        Enzyme.Const(nreal_soa), Enzyme.Const(success_soa),
        Enzyme.Const(freqs), Enzyme.Const(data.strain), Enzyme.Const(data.psd),
        Enzyme.Const(geom_flat), Enzyme.Const(gmst_arr), Enzyme.Const(model), Enzyme.Const(nwalkers),
        Enzyme.Const(KernelFused._lik_array(cfg, backend)))
    return nothing
end

function _paf2_ll_reverse!(data::AbstractVector{<:LikelihoodData}, cfg::PEConfig{T}, backend, model, nwalkers, freqs,
        ll_scratch, bar_ll, physical, bar_physical,
        ts_soa, bar_ts, phis_soa, bar_phis, omegas_soa, bar_omegas,
        Msec_soa, bar_Msec, eta_soa, bar_eta_soa, M_soa, bar_M,
        fmin_soa, bar_fmin, fmax_soa, bar_fmax, nreal_soa, success_soa) where {T}
    ndet = length(data)
    gps_time = data[1].gps_time
    all(d -> d.gps_time == gps_time, data) || throw(ArgumentError("all detectors must share the same event gps_time"))
    0 <= cfg.time_reference <= ndet || throw(ArgumentError(
        "time_reference=$(cfg.time_reference) but only $ndet detector(s) were given"))
    freqs_tuple = ntuple(i -> data[i].freqs, ndet)
    strain_tuple = ntuple(i -> data[i].strain, ndet)
    psd_tuple = ntuple(i -> data[i].psd, ndet)
    geom_arr = KernelFused._geometry_array(data, backend)
    gmst_arr = KernelAbstractions.allocate(backend, Float64, 1)
    copyto!(gmst_arr, [gmst(gps_time)])
    ll_kernel! = KernelFused._fused_ll_kernel_seobnrv5alignedspinf2_precoeffs_multi!(backend)

    # `set_runtime_activity`: the detector tuples hold several same-typed `Vector{Float64}`s next to
    # the duplicated buffers -- the activity-ambiguity class `_enzyme_reverse_mode`'s comment
    # documents for every other multi-detector path.
    Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse), _ll_given_coeffs_multi_wrapper!,
        Enzyme.Duplicated(ll_scratch, bar_ll),
        Enzyme.Duplicated(physical, bar_physical),
        Enzyme.Const(ll_kernel!),
        Enzyme.Duplicated(ts_soa, bar_ts), Enzyme.Duplicated(phis_soa, bar_phis), Enzyme.Duplicated(omegas_soa, bar_omegas),
        Enzyme.Duplicated(Msec_soa, bar_Msec), Enzyme.Duplicated(eta_soa, bar_eta_soa), Enzyme.Duplicated(M_soa, bar_M),
        Enzyme.Duplicated(fmin_soa, bar_fmin), Enzyme.Duplicated(fmax_soa, bar_fmax),
        Enzyme.Const(nreal_soa), Enzyme.Const(success_soa),
        Enzyme.Const(freqs_tuple), Enzyme.Const(strain_tuple), Enzyme.Const(psd_tuple),
        Enzyme.Const(geom_arr),
        Enzyme.Const(gmst_arr), Enzyme.Const(model), Enzyme.Const(nwalkers),
        Enzyme.Const(KernelFused._lik_array(cfg, backend)), Enzyme.Const(cfg.time_reference))
    return nothing
end

# ------------------------------------------------------------------------------------------------
# IMR-table (SEOBNRv5IMRAlignedSpinF2) gradient: one Enzyme.Reverse call per walker (threaded) through the
# whole chain -- sampled -> physical conversion, the IMR table build (PA, RK4 plunge, NQC, merger-ringdown,
# FFT table) and the per-walker likelihood. The SoA hand-adjoint path of the other PA-F2 models assumes
# their 256-node (t, phase, omega) table; this model's table has more fields, and Enzyme differentiates
# its whole build directly. That needs Enzyme's `maxtypeoffset` raised to 32768 (the table build holds
# 768-node and 2048-sample fixed buffers; the default 512 gives EnzymeNoTypeError, and the end-to-end
# SEOBNRv5AlignedSpinF2 reverse pass of test_seobnr_enzyme_e2e_vs_adjoint.jl needs >= 4096), set in
# `__init__` below. The setting is global (it only makes type analysis more permissive) and is part of
# the CUDA kernel-cache key (`ext/GravitationalWaveBayesCUDAExt/kernel_cache.jl`). CPU only for now.
# ------------------------------------------------------------------------------------------------

function __init__()
    Enzyme.API.maxtypeoffset!(32768)
end

@inline function _eobimr_physical(mc, q, sdec, cth)
    s = (mc / (q / (1 + q)^2)^(3 / 5))
    return s / (1 + q), s * q / (1 + q), asin(_clamp_unit(sdec)), acos(_clamp_unit(cth))
end

# Single detector: `det = (freqs, strain, psd, response, gmst)`.
function _eobimr_walker_ll(mc, q, chi1, chi2, dL, t_c, phase, ra, sdec, cth, psi, model, det, lik)
    m1, m2, dec, th = _eobimr_physical(mc, q, sdec, cth)
    mcc, eta = KernelFused._chirp_mass_eta(m1, m2)
    coeffs = GravitationalWaveforms._seobv5f2_coeffs(model, mcc, eta, chi1, chi2)
    return KernelFused._seobv5f2_ll_given_coeffs(coeffs, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, th, psi,
        det[4], det[5], det[1], det[2], det[3], lik)
end

# Several detectors: `det = (freqs_tuple, strain_tuple, psd_tuple, det_tuple, gmst, tref)`.
function _eobimr_walker_ll_multi(mc, q, chi1, chi2, dL, t_c, phase, ra, sdec, cth, psi, model, det, lik)
    m1, m2, dec, th = _eobimr_physical(mc, q, sdec, cth)
    mcc, eta = KernelFused._chirp_mass_eta(m1, m2)
    coeffs = GravitationalWaveforms._seobv5f2_coeffs(model, mcc, eta, chi1, chi2)
    return KernelFused._seobv5f2_ll_given_coeffs_multi(coeffs, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, th, psi,
        det[1], det[2], det[3], det[4], det[5], lik, det[6])
end

_eobimr_det(data::LikelihoodData, freqs, cfg) =
    (freqs, data.strain, data.psd, geometry(data.detector), gmst(data.gps_time))
_eobimr_det(data::AbstractVector, freqs, cfg) =
    (length(data) == 1 ? _eobimr_det(data[1], freqs, cfg) :
     (Tuple(d.freqs for d in data), Tuple(d.strain for d in data), Tuple(d.psd for d in data),
      Tuple(geometry(d.detector) for d in data),
      gmst(data[1].gps_time), cfg.time_reference))
_eobimr_llfun(det) = length(det) == 5 ? _eobimr_walker_ll : _eobimr_walker_ll_multi

# IMRPhenomXAS through the same per-walker path: Enzyme on the scalar walker likelihood is
# ~10x cheaper on the CPU than Enzyme through the KernelAbstractions kernel launch.
function _xas_walker_ll(mc, q, chi1, chi2, dL, t_c, phase, ra, sdec, cth, psi, model, det, lik)
    m1, m2, dec, th = _eobimr_physical(mc, q, sdec, cth)
    return KernelFused._xas_ll_given_physical(model, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, th, psi,
        det[4], det[5], det[1], det[2], det[3], lik)
end
function _xas_walker_ll_multi(mc, q, chi1, chi2, dL, t_c, phase, ra, sdec, cth, psi, model, det, lik)
    m1, m2, dec, th = _eobimr_physical(mc, q, sdec, cth)
    return KernelFused._xas_ll_given_physical_multi(model, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, th, psi,
        det[1], det[2], det[3], det[4], det[5], lik, det[6])
end
# IMRPhenomXHM through the same per-walker path: likelihood options (no phase marginalization) and a
# scalar Enzyme gradient per walker instead of Enzyme through the KernelAbstractions launch.
function _xhm_walker_ll(mc, q, chi1, chi2, dL, t_c, phase, ra, sdec, cth, psi, model, det, lik)
    m1, m2, dec, th = _eobimr_physical(mc, q, sdec, cth)
    return KernelFused._xhm_ll_given_physical(model, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, th, psi,
        det[4], det[5], det[1], det[2], det[3], lik)
end
function _xhm_walker_ll_multi(mc, q, chi1, chi2, dL, t_c, phase, ra, sdec, cth, psi, model, det, lik)
    m1, m2, dec, th = _eobimr_physical(mc, q, sdec, cth)
    return KernelFused._xhm_ll_given_physical_multi(model, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, th, psi,
        det[1], det[2], det[3], det[4], det[5], lik, det[6])
end
# IMRPhenomD through the same per-walker path: likelihood options and a scalar Enzyme gradient per walker.
function _pd_walker_ll(mc, q, chi1, chi2, dL, t_c, phase, ra, sdec, cth, psi, model, det, lik)
    m1, m2, dec, th = _eobimr_physical(mc, q, sdec, cth)
    return KernelFused._pd_ll_given_physical(model, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, th, psi,
        det[4], det[5], det[1], det[2], det[3], lik)
end
function _pd_walker_ll_multi(mc, q, chi1, chi2, dL, t_c, phase, ra, sdec, cth, psi, model, det, lik)
    m1, m2, dec, th = _eobimr_physical(mc, q, sdec, cth)
    return KernelFused._pd_ll_given_physical_multi(model, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, th, psi,
        det[1], det[2], det[3], det[4], det[5], lik, det[6])
end
function _xhm_walker_ll_rb(mc, q, chi1, chi2, dL, t_c, phase, ra, sdec, cth, psi, model, rb, lik)
    m1, m2, dec, th = _eobimr_physical(mc, q, sdec, cth)
    return KernelFused._xhm_ll_rb(model, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, th, psi, rb, lik)
end
# Relative-binning XAS with the likelihood options: `rb` from `build_xas_relative_binning`.
function _xas_walker_ll_rb(mc, q, chi1, chi2, dL, t_c, phase, ra, sdec, cth, psi, model, rb, lik)
    m1, m2, dec, th = _eobimr_physical(mc, q, sdec, cth)
    return KernelFused._xas_ll_rb(model, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, th, psi, rb, lik)
end

"""
    build_xas_relative_binning(data, cfg, sampled_ref; nbins=200) -> NamedTuple

Relative-binning summary data for IMRPhenomXAS with `cfg`'s likelihood options, around the SAMPLED-space
reference point `sampled_ref` (11 rows, the sampler's parametrization). Per detector: bin edges placed at
equal increments of the reference's intrinsic phase (`RelativeBinning._phase_equal_bin_edges`), the
reference strain at the edges (built with exactly the dense kernels' `h`, including this run's time frame),
and the per-bin `A0, A1, B0, B1` sums plus `-2 df sum |d|^2/S`. Pass the result as `relative_binning` to
`gw_logp_and_grad(...; model=IMRPhenomXAS())` (CPU, per-walker Enzyme gradient).
"""
function build_xas_relative_binning(data, cfg::PEConfig, sampled_ref::AbstractVector; nbins::Int=200)
    model = IMRPhenomXAS()
    dets = data isa AbstractVector ? data : [data]
    geo = length(dets) > 1
    lik = KernelFused._lik_opts(cfg)
    x = Float64.(sampled_ref)
    m1, m2, dec, th = _eobimr_physical(x[1], x[2], x[9], x[10])
    chi1, chi2, dL, t_c, phase, ra, psi = x[3], x[4], x[5], x[6], x[7], x[8], x[11]
    mc, eta = KernelFused._chirp_mass_eta(m1, m2)
    pc = KernelFused._xas_phi_coeffs(model, mc, eta, chi1, chi2)
    ac = KernelFused._xas_ampl_coeffs(model, mc, eta, chi1, chi2, dL)
    g = gmst(dets[1].gps_time)
    tref = geo ? cfg.time_reference : 0
    geoms = Tuple(geometry(d.detector) for d in dets)
    dt_ref = tref > 0 ? time_delay_from_geocenter_gmst(geoms[tref], ra, dec, g) : 0.0
    t_c -= KernelFused._xas_time_shift(pc, lik)
    pfac = 0.5 * (1 + cos(th)^2); cfac = cos(th)
    edges_t = Vector{Float64}[]; href_t = Vector{ComplexF64}[]; A0_t = Vector{ComplexF64}[]; A1_t = Vector{ComplexF64}[]
    B0_t = Vector{Float64}[]; B1_t = Vector{Float64}[]; dd_t = Float64[]
    for (k, d) in enumerate(dets)
        dt_geo = geo ? time_delay_from_geocenter_gmst(geoms[k], ra, dec, g) - dt_ref : 0.0
        Fp, Fc = antenna_response_gmst(geoms[k], ra, dec, psi, g)
        combo = Fp * pfac - im * Fc * cfac
        f = d.freqs
        href = [KernelFused._xas_h(pc, ac, combo, t_c + dt_geo, phase, fi) for fi in f]
        bare = [KernelFused._xas_phase_amp(pc, ac, fi)[1] for fi in f]
        idx = RelativeBinning._phase_equal_bin_edges(bare, nbins)
        nb = length(idx) - 1
        df = f[2] - f[1]
        A0 = zeros(ComplexF64, nb); A1 = zeros(ComplexF64, nb); B0 = zeros(nb); B1 = zeros(nb)
        for b in 1:nb
            lo = idx[b]; hi = b == nb ? idx[b+1] : idx[b+1] - 1
            fm = 0.5 * (f[idx[b]] + f[idx[b+1]])
            for i in lo:hi
                w = d.psd[i] == 0 ? 0.0 : 4df / d.psd[i]
                dhc = d.strain[i] * conj(href[i])
                A0[b] += w * dhc; A1[b] += w * dhc * (f[i] - fm)
                B0[b] += w * abs2(href[i]); B1[b] += w * abs2(href[i]) * (f[i] - fm)
            end
        end
        push!(edges_t, f[idx]); push!(href_t, href[idx]); push!(A0_t, A0); push!(A1_t, A1); push!(B0_t, B0); push!(B1_t, B1)
        push!(dd_t, -2df * sum(d.psd[i] == 0 ? 0.0 : abs2(d.strain[i]) / d.psd[i] for i in eachindex(f)))
    end
    n = length(dets)
    return (edges=Tuple(edges_t), href=Tuple(href_t), A0=Tuple(A0_t), A1=Tuple(A1_t), B0=Tuple(B0_t), B1=Tuple(B1_t),
            dd=Tuple(dd_t), dets=geoms, gmst=g, geo=geo, tref=tref,
            reference=copy(x), nbins=Tuple(length(e) - 1 for e in edges_t))
end

"""
    build_xhm_relative_binning(data, cfg, sampled_ref; nbins=400) -> XHMRelBin

Per-mode relative-binning summary data for IMRPhenomXHM with `cfg`'s likelihood options, around the
SAMPLED-space reference point. Per detector: bin edges at equal increments of the reference's (2,2) phase (the
higher modes' phases run up to 2x faster: use ~1.5-2x the XAS bin count); for each of the five modes its
detector-frame reference strain u_k,ref at the edges (`href[d][k, edge]`) and `A0`, `A1` (data x mode, `[k, bin]`); for
each of the 15 mode pairs (k <= k') `C0`, `C1` (`[pair, bin]`: sum w u_k,ref conj(u_k',ref) and its first moment about
the bin centre). Mode-major matrices, not tuples of vectors: a runtime-indexed tuple of arrays made Enzyme's reverse
pass demand runtime activity. See
`KernelFused._xhm_ll_rb`. `fref0` = the dense kernels' phiRef frequency (first bin of the first detector).
"""
function build_xhm_relative_binning(data, cfg::PEConfig, sampled_ref::AbstractVector; nbins::Int=400)
    model = IMRPhenomXHM()
    dets = data isa AbstractVector ? data : [data]
    KernelFused._check_likelihood_options(cfg, model, length(dets))
    geo = length(dets) > 1
    lik = KernelFused._lik_opts(cfg)
    x = Float64.(sampled_ref)
    m1, m2, dec, th = _eobimr_physical(x[1], x[2], x[9], x[10])
    chi1, chi2, dL, t_c, phase, ra, psi = x[3], x[4], x[5], x[6], x[7], x[8], x[11]
    fref0 = Float64(dets[1].freqs[1])
    sys, ms21, ms33, ms44, m32 = KernelFused._xhm_setup(m1, m2, chi1, chi2, dL, phase, fref0)
    fac = KernelFused._xhm_mode_factors(th)
    mc, eta = KernelFused._chirp_mass_eta(m1, m2)
    pc = KernelFused._xas_phi_coeffs(IMRPhenomXAS(), mc, eta, chi1, chi2)
    ac = KernelFused._xas_ampl_coeffs(IMRPhenomXAS(), mc, eta, chi1, chi2, dL)
    g = gmst(dets[1].gps_time)
    tref = geo ? cfg.time_reference : 0
    geoms = Tuple(geometry(d.detector) for d in dets)
    dt_ref = tref > 0 ? time_delay_from_geocenter_gmst(geoms[tref], ra, dec, g) : 0.0
    t_c -= KernelFused._xhm_time_shift(sys, lik)
    pairs = [(k, kp) for k in 1:5 for kp in k:5]
    edges_t = []; href_t = []; A0_t = []; A1_t = []; C0_t = []; C1_t = []; dd_t = Float64[]
    for (kd, d) in enumerate(dets)
        dt_geo = geo ? time_delay_from_geocenter_gmst(geoms[kd], ra, dec, g) - dt_ref : 0.0
        Fp, Fc = antenna_response_gmst(geoms[kd], ra, dec, psi, g)
        f = d.freqs
        U = [KernelFused._xhm_u(KernelFused._xhm_modes(fi, sys, ms21, ms33, ms44, m32), fac, Fp, Fc, t_c + dt_geo, fi)
             for fi in f]
        bare = [KernelFused._xas_phase_amp(pc, ac, fi)[1] for fi in f]
        idx = RelativeBinning._phase_equal_bin_edges(bare, nbins)
        nb = length(idx) - 1
        df = f[2] - f[1]
        A0 = zeros(ComplexF64, 5, nb); A1 = zeros(ComplexF64, 5, nb)
        C0 = zeros(ComplexF64, length(pairs), nb); C1 = zeros(ComplexF64, length(pairs), nb)
        for b in 1:nb
            lo = idx[b]; hi = b == nb ? idx[b+1] : idx[b+1] - 1
            fm = 0.5 * (f[idx[b]] + f[idx[b+1]])
            for i in lo:hi
                w = d.psd[i] == 0 ? 0.0 : 4df / d.psd[i]
                u = U[i]
                for k in 1:5
                    dhc = d.strain[i] * conj(u[k])
                    A0[k, b] += w * dhc; A1[k, b] += w * dhc * (f[i] - fm)
                end
                for (p, (k, kp)) in enumerate(pairs)
                    c = w * u[k] * conj(u[kp])
                    C0[p, b] += c; C1[p, b] += c * (f[i] - fm)
                end
            end
        end
        push!(edges_t, f[idx]); push!(href_t, [U[i][k] for k in 1:5, i in idx])
        push!(A0_t, A0); push!(A1_t, A1); push!(C0_t, C0); push!(C1_t, C1)
        push!(dd_t, -2df * sum(d.psd[i] == 0 ? 0.0 : abs2(d.strain[i]) / d.psd[i] for i in eachindex(f)))
    end
    z = Tuple(Float64[] for _ in dets)
    return (edges=Tuple(edges_t), href=Tuple(href_t), A0=Tuple(A0_t), A1=Tuple(A1_t), B0=z, B1=z,
            dd=Tuple(dd_t), dets=geoms, gmst=g, geo=geo, tref=tref,
            reference=copy(x), nbins=Tuple(length(e) - 1 for e in edges_t), fref0=fref0,
            C0=Tuple(C0_t), C1=Tuple(C1_t))
end

# Relative binning with the likelihood options for the models that have it (the driver's `relative_binning = true`).
build_relative_binning_opts(::IMRPhenomXAS, data, cfg, x; nbins) = build_xas_relative_binning(data, cfg, x; nbins=nbins)
build_relative_binning_opts(::IMRPhenomXHM, data, cfg, x; nbins) = build_xhm_relative_binning(data, cfg, x; nbins=nbins)

_walker_llfun(::IMRTableModel, det) = _eobimr_llfun(det)
_walker_llfun(::IMRPhenomXAS, det) = length(det) == 5 ? _xas_walker_ll : _xas_walker_ll_multi
_walker_llfun(::IMRPhenomXHM, det) = length(det) == 5 ? _xhm_walker_ll : _xhm_walker_ll_multi
_walker_llfun(::IMRPhenomD, det) = length(det) == 5 ? _pd_walker_ll : _pd_walker_ll_multi
const PerWalkerGradModel = Union{IMRTableModel,IMRPhenomXAS,IMRPhenomXHM,IMRPhenomD}

include("XASBinParallel.jl")
include("XHMBinParallel.jl")
include("XPBinParallel.jl")
include("XPHMBinParallel.jl")
include("SEOBNRDeviceTable.jl")
include("SEOBNRDeviceLikelihood.jl")
include("SEOBNRHMDeviceTable.jl")
include("SEOBNRHMBinParallel.jl")
include("NTHelpers.jl")
include("SEOBNRPHMSampleRev.jl")
include("SEOBNRPHMDeviceGrad.jl")

# IMRPhenomXP / IMRPhenomXPHM (15 sampled rows) through the same per-walker scalar Enzyme gradient:
# `det = (freqs, strain, psd, DetectorGeometry) tuples, gmst, tref, geo` for `KernelFused._xprec_ll_given_physical`.
function _xprec_walker_ll(mc, q, chi1, chi2, dL, t_c, phase, ra, sdec, cth, psi, chi1x, chi1y, chi2x, chi2y, model, det, lik)
    m1, m2, dec, th = _eobimr_physical(mc, q, sdec, cth)
    return KernelFused._xprec_ll_given_physical(model, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, th, psi,
        chi1x, chi1y, chi2x, chi2y, det[1], det[2], det[3], det[4], det[5], lik, det[6], det[7])
end
function _xprec_det(data, cfg)
    dets = data isa AbstractVector ? data : [data]
    geo = length(dets) > 1
    return (Tuple(d.freqs for d in dets), Tuple(d.strain for d in dets), Tuple(d.psd for d in dets),
            Tuple(geometry(d.detector) for d in dets),
            gmst(dets[1].gps_time), geo ? cfg.time_reference : 0, geo)
end
function _eobimr_gradient!(grad_out::AbstractMatrix{T}, positions::AbstractMatrix{T}, freqs, cfg::PEConfig{T},
                          data, model::_Prec15; ll_out = nothing) where {T}
    nparams, nwalkers = size(positions)
    nparams == 15 || throw(ArgumentError("$(nameof(typeof(model))) samples 15 parameters, got $nparams"))
    positions isa Array || throw(ArgumentError("the per-walker gradient is CPU-only"))
    KernelFused._check_likelihood_options(cfg, model, data isa AbstractVector ? length(data) : 1)
    det = _xprec_det(data, cfg)
    lik = KernelFused._lik_opts(cfg)
    Threads.@threads for w in 1:nwalkers
        x = ntuple(k -> Float64(positions[k, w]), Val(15))
        res = Enzyme.autodiff(Enzyme.ReverseWithPrimal, _xprec_walker_ll, Enzyme.Active, map(Enzyme.Active, x)...,
            Enzyme.Const(model), Enzyme.Const(det), Enzyme.Const(lik))
        ll_out === nothing || (ll_out[w] = T(res[2]))
        @inbounds for k in 1:15
            gk = res[1][k]
            grad_out[k, w] = gk === nothing ? zero(T) : T(gk)
        end
    end
    return nothing
end

function _eobimr_gradient!(grad_out::AbstractMatrix{T}, positions::AbstractMatrix{T}, freqs, cfg::PEConfig{T},
                          data, model::PerWalkerGradModel; ll_out = nothing) where {T}
    nparams, nwalkers = size(positions)
    @assert nparams == 11 "IMR-table models / IMRPhenomXAS / IMRPhenomXHM sample 11 parameters, got $nparams"
    positions isa Array || throw(ArgumentError("the per-walker gradient (SEOBNRv5IMR/IMRPhenomXAS/XHM) is CPU-only"))
    model isa IMRPhenomXHM && KernelFused._check_likelihood_options(cfg, model, data isa AbstractVector ? length(data) : 1)
    det = _eobimr_det(data, freqs, cfg)
    return _perwalker_gradient!(grad_out, positions, _walker_llfun(model, det), model, det, KernelFused._lik_opts(cfg);
                                ll_out = ll_out)
end

function _perwalker_gradient!(grad_out::AbstractMatrix{T}, positions::AbstractMatrix{T}, f, model, det, lik;
                              ll_out = nothing) where {T}
    nwalkers = size(positions, 2)
    Threads.@threads for w in 1:nwalkers
        x = ntuple(k -> Float64(positions[k, w]), 11)
        res = Enzyme.autodiff(Enzyme.ReverseWithPrimal, f, Enzyme.Active, map(Enzyme.Active, x)...,
            Enzyme.Const(model), Enzyme.Const(det), Enzyme.Const(lik))
        g = res[1]
        ll_out === nothing || (ll_out[w] = T(res[2]))
        @inbounds for k in 1:11
            gk = g[k]
            grad_out[k, w] = gk === nothing ? zero(T) : T(gk)
        end
    end
    return nothing
end

# Forward-mode variant (one width-11 Enzyme forward pass per walker). Used for the XHM relative-binning likelihood: its
# reverse-mode gradient is WRONG in the intrinsic rows (q off by 30-45%, chi1/chi2 by 1-2%; mc and every extrinsic row
# right; with or without the likelihood options and with or without runtime activity), while forward mode matches
# ForwardDiff and finite differences to 1e-7 -- and is not slower (13.6 vs 16.6 ms per walker). The same
# reverse pass through the dense XHM likelihood is right; the per-mode values and loops over them differentiate
# correctly in isolation, so the fault is somewhere in how the binned sums accumulate the mode adjoints (not isolated).
@inline _fwd_unit(k) = ntuple(j -> ifelse(j == k, 1.0, 0.0), Val(11))
function _perwalker_gradient_fwd!(grad_out::AbstractMatrix{T}, positions::AbstractMatrix{T}, f, model, det, lik;
                                  ll_out = nothing) where {T}
    nwalkers = size(positions, 2)
    Threads.@threads for w in 1:nwalkers
        x = ntuple(k -> Float64(positions[k, w]), 11)
        res = Enzyme.autodiff(Enzyme.ForwardWithPrimal, Enzyme.Const(f), Enzyme.BatchDuplicated,
            ntuple(k -> Enzyme.BatchDuplicated(x[k], _fwd_unit(k)), Val(11))...,
            Enzyme.Const(model), Enzyme.Const(det), Enzyme.Const(lik))
        ll_out === nothing || (ll_out[w] = T(res[2]))
        @inbounds for k in 1:11
            grad_out[k, w] = T(res[1][k])
        end
    end
    return nothing
end

# ------------------------------------------------------------------------------------------------
# Extrinsic block with a cached waveform (IMR-table models). The intrinsic parameters (mc, q, chi1, chi2)
# fix the expensive coefficient table; distance, time, phase, sky and orientation only enter through
# `_seobv5f2_ll_given_coeffs`. With the table cached, the gradient with respect to the extrinsic rows is
# ~15x cheaper than the full one for SEOBNRv5 IMR (3.2 vs 47.9 ms per walker),
# so block moves on the extrinsic rows at fixed intrinsics (Metropolis-within-Gibbs) are cheap.
# ------------------------------------------------------------------------------------------------
function _eobimr_ext_ll(dL, t_c, phase, ra, sdec, cth, psi, coeffs, m1, m2, chi1, chi2, det, lik)
    dec = asin(_clamp_unit(sdec))
    th = acos(_clamp_unit(cth))
    return length(det) == 5 ?
        KernelFused._seobv5f2_ll_given_coeffs(coeffs, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, th, psi,
            det[4], det[5], det[1], det[2], det[3], lik) :
        KernelFused._seobv5f2_ll_given_coeffs_multi(coeffs, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, th, psi,
            det[1], det[2], det[3], det[4], det[5], lik, det[6])
end

"""
    IMRExtrinsicBlock(data, freqs, cfg, model::IMRTableModel)

Cached-waveform extrinsic block for the IMR-table models. `cache!(blk, positions)` builds each walker's
coefficient table from its intrinsic rows; `ext_ll_and_grad!(blk, ll, grad, positions)` then returns every
walker's FULL log-likelihood (identical to the full path) and its gradient with respect to the extrinsic
rows 5-11 (rows 1-4 of `grad` are set to zero), without rebuilding any table. Valid only while the
intrinsic rows equal the cached ones. CPU only (threads over walkers).
"""
mutable struct IMRExtrinsicBlock{D,L,M}
    det::D
    lik::L
    model::M
    coeffs::Vector{Any}
    m1::Vector{Float64}
    m2::Vector{Float64}
    chi1::Vector{Float64}
    chi2::Vector{Float64}
end

function IMRExtrinsicBlock(data, freqs, cfg::PEConfig, model::IMRTableModel)
    det = _eobimr_det(data, freqs, cfg)
    lik = KernelFused._lik_opts(cfg)
    return IMRExtrinsicBlock(det, lik, model, Any[], Float64[], Float64[], Float64[], Float64[])
end

function cache!(blk::IMRExtrinsicBlock, positions::AbstractMatrix)
    n = size(positions, 2)
    resize!(blk.coeffs, n); resize!(blk.m1, n); resize!(blk.m2, n); resize!(blk.chi1, n); resize!(blk.chi2, n)
    Threads.@threads for w in 1:n
        m1, m2, _, _ = _eobimr_physical(Float64(positions[1, w]), Float64(positions[2, w]), 0.0, 1.0)
        mcc, eta = KernelFused._chirp_mass_eta(m1, m2)
        c1, c2 = Float64(positions[3, w]), Float64(positions[4, w])
        blk.coeffs[w] = GravitationalWaveforms._seobv5f2_coeffs(blk.model, mcc, eta, c1, c2)
        blk.m1[w] = m1; blk.m2[w] = m2; blk.chi1[w] = c1; blk.chi2[w] = c2
    end
    return blk
end

function ext_ll_and_grad!(blk::IMRExtrinsicBlock, ll::AbstractVector{T}, grad::AbstractMatrix{T},
                          positions::AbstractMatrix{T}) where {T}
    n = size(positions, 2)
    length(blk.coeffs) == n || throw(ArgumentError("IMRExtrinsicBlock: call cache! for these walkers first"))
    Threads.@threads for w in 1:n
        x = ntuple(k -> Float64(positions[4 + k, w]), 7)
        res = Enzyme.autodiff(Enzyme.ReverseWithPrimal, _eobimr_ext_ll, Enzyme.Active, map(Enzyme.Active, x)...,
            Enzyme.Const(blk.coeffs[w]), Enzyme.Const(blk.m1[w]), Enzyme.Const(blk.m2[w]),
            Enzyme.Const(blk.chi1[w]), Enzyme.Const(blk.chi2[w]), Enzyme.Const(blk.det), Enzyme.Const(blk.lik))
        g = res[1]
        ll[w] = T(res[2])
        @inbounds for k in 1:4
            grad[k, w] = zero(T)
        end
        @inbounds for k in 1:7
            gk = g[k]
            grad[4 + k, w] = gk === nothing ? zero(T) : T(gk)
        end
    end
    return nothing
end

_reverse_mode_gradient!(grad_out::AbstractMatrix{T}, positions::AbstractMatrix{T}, freqs::AbstractVector{T},
                        cfg::PEConfig{T}, data::DetectorData, model::IMRTableModel, rb::Nothing) where {T} =
    _eobimr_gradient!(grad_out, positions, freqs, cfg, data, model)
_forward_mode_gradient!(grad_out::AbstractMatrix{T}, positions::AbstractMatrix{T}, freqs::AbstractVector{T},
                        cfg::PEConfig{T}, data::DetectorData, model::IMRTableModel, rb::Nothing) where {T} =
    _eobimr_gradient!(grad_out, positions, freqs, cfg, data, model)
# Single detector: also matches the PAF2Model method above (IMR table models are PAF2Models); the table path wins.
_forward_mode_gradient!(grad_out::AbstractMatrix{T}, positions::AbstractMatrix{T}, freqs::AbstractVector{T},
                        cfg::PEConfig{T}, data::LikelihoodData, model::IMRTableModel, rb::Nothing) where {T} =
    _eobimr_gradient!(grad_out, positions, freqs, cfg, data, model)

# `_reparam_to_physical!` returns `physical` (its own mutated argument) -- Enzyme.Reverse errors
# ("Duplicated Returns not yet handled") differentiating a function whose return value ALIASES one
# of its own Duplicated arguments without an explicit return-activity annotation. Sidestepped the
# same way `_ll_given_coeffs_wrapper!` already does for the same reason: a thin `nothing`-returning
# wrapper around the mutating call.
@inline function _reparam_to_physical_novalue!(physical, positions)
    _reparam_to_physical!(physical, positions)
    return nothing
end

function _forward_mode_gradient!(grad_out::AbstractMatrix{T}, positions::AbstractMatrix{T},
                                  freqs::AbstractVector{T}, cfg::PEConfig{T}, data::DetectorData,
                                  model, rb) where {T}
    nparams, nwalkers = size(positions)
    ll_scratch = similar(positions, T, nwalkers)
    dll_scratch = similar(ll_scratch)
    seed = similar(positions)
    physical_scratch = similar(positions)
    dphysical_scratch = similar(positions)
    @inbounds for p in 1:nparams
        fill!(seed, zero(T))
        seed[p, :] .= one(T)
        fill!(ll_scratch, zero(T))
        fill!(dll_scratch, zero(T))
        fill!(physical_scratch, zero(T))
        fill!(dphysical_scratch, zero(T))
        Enzyme.autodiff(Enzyme.Forward, _ll_vec!,
            Enzyme.Duplicated(ll_scratch, dll_scratch),
            Enzyme.Duplicated(physical_scratch, dphysical_scratch),
            Enzyme.Duplicated(positions, seed),
            Enzyme.Const(freqs), Enzyme.Const(cfg), Enzyme.Const(data), Enzyme.Const(model),
            Enzyme.Const(rb))
        grad_out[p, :] .= dll_scratch
    end
    return nothing
end

"""
    _raw_loglikelihood!(ll_out, positions, freqs, cfg, data, model, rb)

Plain forward-pass (no gradient, no Enzyme, no tempering) UNTEMPERED log-likelihood evaluation --
`_reparam_to_physical!` + `_eval_ll!`, the exact same two calls `gw_logp_and_grad`'s own forward
pass makes, minus the `beta` scaling and `+ log-prior` that turn it into a log-POSTERIOR. Exists
for `batched_pt_nuts!`'s swap rounds (parallel tempering): `NUTSCore.pt_swap!` needs each
lane's raw `logL(theta_w)` at its CURRENT (post-draw) position, which is NOT recoverable from
`NUTSCore.nuts_draw!`'s own `logp` output (that is the TEMPERED log-posterior, `beta*logL+logprior`,
and not necessarily even evaluated at exactly the final accepted position -- NUTS's multinomial
sampling can pick an EARLIER leaf of the trajectory as the draw's candidate, so the last
`logp_and_grad!` call of a draw is not reliably the one at `positions` after `nuts_draw!` returns).
Re-evaluating explicitly here, once per swap round (not once per leapfrog step), is cheap by
comparison and always correct regardless of NUTS's internal call pattern.
"""
function _raw_loglikelihood!(ll_out::AbstractVector{T}, positions::AbstractMatrix{T},
                              freqs::AbstractVector{T}, cfg::PEConfig{T}, data::DetectorData,
                              model, rb) where {T}
    physical = similar(positions)
    _reparam_to_physical!(physical, positions)
    _eval_ll!(ll_out, physical, freqs, cfg, data, model, rb)
    return ll_out
end

"""
    gw_logp_and_grad(data, cfg, freqs; model=IMRPhenomD(), relative_binning=nothing)

Build a `logp_and_grad!` closure conforming to `NUTSCore`'s generic interface
(`(grad_out, logp_out, positions, active_mask) -> nothing`), wired to the fused
waveform+likelihood KernelAbstractions kernel via `Enzyme.autodiff`.

`positions` here are the SAMPLED convention -- 11-row `(mc, q, chi1, chi2, dL, t_c, phase, ra,
sin_dec, cos_theta_jn, psi)` for `model isa IMRPhenomD`/`IMRPhenomXAS`, or 15-row (appending
`chi1x, chi1y, chi2x, chi2y`) for `model isa IMRPhenomXP` -- not the physical row convention the
likelihood kernels consume -- see the "Sampling-layer reparametrization" note above
`_reparam_to_physical!`. Both the forward pass (this function) and the reverse pass (`_sum_ll`,
via `Enzyme.autodiff`) convert to the physical row convention before calling the likelihood
kernel, and both add the uniform-bounds `_log_prior` term (dispatched to the matching-arity method
via `_call_log_prior`), so `logp_out` and its gradient consistently represent the log-POSTERIOR,
not bare log-likelihood.

`relative_binning`: `nothing` (default) evaluates the full-grid fused likelihood kernel
(`batched_log_likelihood_kernel!`) every leapfrog step, as before. Passing a
`RelativeBinning.RelativeBinningSetup` (built once, ahead of sampling, via
`RelativeBinning.build_relative_binning`) instead evaluates the sparse bin-edge approximation
(`RelativeBinning.binned_log_likelihood_kernel!`) every step -- see `_eval_ll!`'s docstring/
comment above for how this dispatches without a runtime branch inside the Enzyme-differentiated
call graph.

Because each walker's log-density depends only on its own column of `positions` (block-diagonal
across walkers), ONE `Enzyme.autodiff` call over the whole `(nparams, nwalkers)` batch yields
every walker's gradient simultaneously. NUTS calls this once per leapfrog step for the entire
currently-active set of walkers (never per-walker in a loop) -- this is the actual GPU-batching
payoff of the batched design.

`active_mask` is accepted for interface conformance but unused here: neither likelihood kernel
has a notion of a partial/masked launch, so inactive walkers' columns are simply computed and
then ignored by the caller (NUTSCore) -- wasted compute on those columns, traded for a constant
batch shape across every step of the tree build (see `NUTSCore`'s module docstring).

`beta` (parallel tempering, default `nothing`): `nothing` reproduces the exact pre-PT
behavior (every lane samples the untempered `beta=1` posterior, unchanged). An
`AbstractVector{T}` of length `nwalkers` instead tempers each lane's log-LIKELIHOOD term only --
`logp_out[w] = beta[w] * logL(theta_w) + logprior(theta_w)` (`log_L_tempered = beta * log_L`, prior left untempered). Pass the SAME mutable vector a
`NUTSCore.PTLadderState`'s `betas` field holds (see `batched_pt_nuts!`): since it is captured here
by reference, a `pt_swap!` mutating it in place between draws changes what THIS closure computes
on its very next call, with no need to rebuild it.

For the reverse-mode path (`Enzyme.autodiff` through `_sum_ll`), tempering is threaded through as
a genuine extra Enzyme.Const argument (`_sum_ll` has a dedicated `beta::AbstractVector{T}` method,
dispatched the same "no runtime branch inside Enzyme-differentiated code" way `_eval_ll!`'s own
`rb`-dispatch already works -- see that function's comment). For the forward-mode path
(`_uses_forward_mode_gradient(model) == true`), `_forward_mode_gradient!` itself is left
UNCHANGED (still computes plain `d(logL)/d(theta)`, no `beta`, no prior -- see its own docstring
for why the prior's gradient is exactly zero in-bounds): the tempered gradient
`d(beta*logL)/d(theta) = beta * d(logL)/d(theta)` is then just a per-COLUMN post-scale of
`grad_out`, since `beta` is a per-walker (per-column) constant and every walker's log-density is
already block-diagonal in `positions` -- cheaper than threading `beta` through the SEOBNR
precoeffs machinery itself, and does not touch any of that path's own standing Enzyme/CUDA
workarounds at all.
"""
function gw_logp_and_grad(data::DetectorData, cfg::PEConfig{T}, freqs::AbstractVector{T};
                           model=IMRPhenomD(), relative_binning=nothing,
                           beta::Union{Nothing,AbstractVector{T}}=nothing,
                           prior::Union{Nothing,GWPrior{T}}=nothing) where {T}
    prior_lp = prior === nothing ? nothing : Vector{T}(undef, cfg.nwalkers)
    return function (grad_out::AbstractMatrix{T}, logp_out::AbstractVector{T},
                      positions::AbstractMatrix{T}, active_mask)
        # SEOBNRv5IMRAlignedSpinF2: the per-walker reverse call returns the log-likelihood too
        # (ReverseWithPrimal), so there is no separate forward pass.
        if (model isa IMRPhenomXAS && relative_binning isa XASRelBin) || (model isa IMRPhenomXHM && relative_binning isa XHMRelBin)
            positions isa Array || throw(ArgumentError("relative binning with likelihood options is CPU-only here"))
            if model isa IMRPhenomXHM
                _perwalker_gradient_fwd!(grad_out, positions, _xhm_walker_ll_rb, model, relative_binning,
                                         KernelFused._lik_opts(cfg); ll_out = logp_out)
            else
                _perwalker_gradient!(grad_out, positions, _xas_walker_ll_rb, model, relative_binning,
                                     KernelFused._lik_opts(cfg); ll_out = logp_out)
            end
            beta === nothing || (logp_out .*= beta; grad_out .*= reshape(beta, 1, :))
            prior === nothing && (logp_out .+= _call_log_prior_vec(positions))
            if prior !== nothing
                length(prior_lp) == length(logp_out) || resize!(prior_lp, length(logp_out))
                prior_logpdf_and_grad!(prior_lp, grad_out, prior, positions)
                logp_out .+= prior_lp
            end
            return nothing
        end
        if (model isa IMRTableModel ||
            (model isa Union{IMRPhenomXAS,IMRPhenomXHM,IMRPhenomD,_Prec15} && positions isa Array)) &&
           relative_binning === nothing
            _eobimr_gradient!(grad_out, positions, freqs, cfg, data, model; ll_out = logp_out)
            if beta !== nothing
                logp_out .*= beta
                grad_out .*= reshape(beta, 1, :)
            end
            prior === nothing && (logp_out .+= _call_log_prior_vec(positions))
            if prior !== nothing
                length(prior_lp) == length(logp_out) || resize!(prior_lp, length(logp_out))
                prior_logpdf_and_grad!(prior_lp, grad_out, prior, positions)
                logp_out .+= prior_lp
            end
            return nothing
        end
        # Forward pass to populate per-walker log-posterior (cheap relative to the reverse-mode
        # call below, and keeps the autodiff call identical to the tested pattern rather than
        # trying to recover the primal `ll` from inside the Enzyme call). The `beta` scaling here
        # is plain host/device array arithmetic OUTSIDE the Enzyme call graph entirely (this whole
        # block is never differentiated -- see the docstring above), so the runtime
        # `beta === nothing` branch is safe unlike the dispatch-only discipline `_sum_ll` needs.
        physical = similar(positions)
        _reparam_to_physical!(physical, positions)
        _eval_ll!(logp_out, physical, freqs, cfg, data, model, relative_binning)
        beta === nothing || (logp_out .*= beta)
        prior === nothing && (logp_out .+= _call_log_prior_vec(positions))

        fill!(grad_out, zero(T))
        if model isa PAF2Model && _EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[] &&
           relative_binning === nothing
            # Reverse-mode path (default for this model when the loaded GravitationalWaveforms has the
            # device-heap-leak fix; see _EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT's own comment).
            # Anything it does not cover (multi-detector, relative binning) falls through to forward mode.
            _reverse_mode_gradient!(grad_out, positions, freqs, cfg, data, model, relative_binning)
            beta === nothing || (grad_out .*= reshape(beta, 1, :))
        elseif _uses_forward_mode_gradient(model)
            _forward_mode_gradient!(grad_out, positions, freqs, cfg, data, model, relative_binning)
            beta === nothing || (grad_out .*= reshape(beta, 1, :))
        else
            Enzyme.autodiff(_enzyme_reverse_mode(data), _sum_ll,
                Enzyme.Duplicated(positions, grad_out),
                Enzyme.Const(freqs), Enzyme.Const(cfg), Enzyme.Const(data), Enzyme.Const(model),
                Enzyme.Const(relative_binning), Enzyme.Const(beta))
        end
        # Configurable prior (`Priors.GWPrior`): closed-form value and gradient, untempered, added
        # outside every AD path. (`_sum_ll` above still adds the default box prior's zero gradient.)
        if prior !== nothing
            length(prior_lp) == length(logp_out) || resize!(prior_lp, length(logp_out))
            prior_logpdf_and_grad!(prior_lp, grad_out, prior, positions)
            logp_out .+= prior_lp
        end
        return nothing
    end
end

"""
    gw_logp_and_grad_gpu_staged(data::LikelihoodData, cfg::PEConfig{T}, freqs::AbstractVector{T};
                                 model=IMRPhenomD(), nparams::Int, nwalkers::Int) where {T}

Like [`gw_logp_and_grad`](@ref), but returns a closure taking ordinary HOST `positions`/`grad_out`/
`logp_out` (plain `Matrix`/`Vector`, exactly what `NUTSCore.nuts_sample!`/`nuts_draw!` already
allocate and index scalar-per-walker throughout their tree-doubling/reflection/dual-averaging
logic) while running the actual likelihood-and-gradient evaluation -- the expensive part -- on
whatever device `GravitationalWaveBayes.GPU.to_device`/`compatible_array` resolve to (a real GPU once `CUDA.jl` is
loaded and functional, plain `Array` otherwise).

**Why staged copies, not a fully device-resident sampler**: `NUTSCore.jl`'s tree-doubling/
reflection/U-turn/dual-averaging logic is written as ordinary per-walker Julia scalar loops
(`@inbounds for w in 1:nwalkers`, `BitVector` masks, `randn`-produced host momenta) -- this is the
right design for that logic (`nwalkers` is tiny, a handful to a few dozen; there is nothing to gain
from vectorizing it, and it is exactly the logic this package's whole existing NUTS test suite has
validated). Making THAT logic itself GPU-resident would mean rewriting and re-validating all of it
from scratch for no performance benefit. Staging only `positions`/`grad`/`logp` across the
host/device boundary once per leapfrog step, immediately around the one call that's actually worth
running on a GPU (`_forward_mode_gradient!`/`_eval_ll!`'s underlying `KernelAbstractions.jl`
kernels), reuses 100% of the already-validated `nuts_draw!`/`nuts_sample!`/`batched_nuts!` code
unchanged -- this function is a drop-in replacement for the `logp_and_grad!` closure they take, not
a new sampler.

`data`/`freqs` are copied to-device ONCE here (setup cost, matches `build_relative_binning`'s own
one-time-cost convention); `nparams`/`nwalkers` size the persistent device staging buffers (also
allocated once, reused across every call the returned closure receives, matching this file's
`similar`-outside-the-hot-loop discipline elsewhere). `active_mask` is accepted for interface
compatibility with `nuts_draw!`'s call signature but -- like `gw_logp_and_grad`'s own closure --
never read; every leapfrog step evaluates every walker's column regardless of `NUTSCore`'s active
mask (see `gw_logp_and_grad`'s own docstring for why).

**Scope, deliberately narrow, first attempt**: single-detector only (`data::
LikelihoodData`, not the multi-detector `Vector{LikelihoodData}` -- that path has a separate,
unresolved forward-mode Enzyme gap for SEOBNR). See
`scripts/check_seobnr_gpu_staged_nuts.jl` (a standalone script, since `test/runtests.jl` never loads
`CUDA.jl`) for real-CUDA validation.

`relative_binning` (added): `nothing` (default, unchanged) evaluates the dense kernel
every step, matching this function's original scope. Passing a `RelativeBinning.RelativeBinningSetup`
switches to the sparse bin-edge kernel instead -- this is pure plumbing (`inner!` is just
`gw_logp_and_grad`, which already dispatches on `relative_binning`'s type via `_eval_ll!`; nothing
about the host/device staging above changes). The setup does NOT need to be built with device-
resident `data` -- `build_relative_binning`'s own summary arrays already auto-promote via
`to_device` once CUDA is loaded/functional (see its docstring), independent of what device its
OWN input `data` argument was on. **Callers are responsible for the SAME dense-warmup-then-binned
safety discipline `batched_nuts!`'s `dense_warmup` keyword documents** (this function has no
`dense_warmup` flag of its own -- build two closures, one per kernel, and pass them to
`NUTSCore.nuts_sample!`'s `warmup_logp_and_grad!`/main `logp_and_grad!` respectively, exactly
mirroring `batched_nuts!`'s own composition; see `scripts/run_from_config.jl`'s relative-binning
code path for a worked example). Not yet validated on real CUDA hardware (GPU was in use by a
concurrent task when this was added) -- CPU-path correctness (dense-vs-binned agreement, via the
shared `gw_logp_and_grad` this wraps) is; real-CUDA validation of the RELATIVE-BINNING addition
specifically remains open.
"""
_data_to_device(d::LikelihoodData) =
    LikelihoodData(to_device(d.freqs), to_device(d.strain), to_device(d.psd), d.detector, d.gps_time)
_data_to_device(d::AbstractVector{<:LikelihoodData}) = [_data_to_device(x) for x in d]

function gw_logp_and_grad_gpu_staged(data::DetectorData, cfg::PEConfig{T}, freqs::AbstractVector{T};
                                      model=IMRPhenomD(), nparams::Int, nwalkers::Int,
                                      relative_binning=nothing, calibration=nothing, calmarg=nothing, box=nothing,
                                      timers=nothing) where {T}
    # calmarg: response curves (`calibration_curves(...).C`) for calibration MARGINALIZATION (SEOBNRv5PHM only); with
    # `calibration` (sampled) and neither (omitted) the three bilby modes
    (calmarg === nothing || model isa Union{SEOBNRv5PHM,IMRPhenomXPHM}) ||
        throw(ArgumentError("calibration marginalization: SEOBNRv5PHM / IMRPhenomXPHM only"))
    # box: a positions -> 0 / -Inf support function replacing the package box in the XPHM / SEOBNRv5PHM device closures
    # (`timers`: the mode-sum closures' 4-stage seconds accumulator, see `xhm_binparallel_logp_and_grad`)
    bkw = (; (box === nothing ? (;) : (; box))..., (timers === nothing ? (;) : (; timers))...)
    # calibration (`calibration_prior`): IMRPhenomXPHM bin-parallel and SEOBNRv5PHM device paths (positions:
    # 15 + 4 nn rows)
    (calibration === nothing || (model isa Union{IMRPhenomXPHM,SEOBNRv5PHM} && relative_binning === nothing)) ||
        throw(ArgumentError("calibration is implemented for the IMRPhenomXPHM / SEOBNRv5PHM device paths only"))
    # IMRPhenomXAS (dense): bin-parallel sums on the device, per-walker work on the host
    # (`xas_binparallel_logp_and_grad`); host arrays in and out like the path below.
    # With `relative_binning` from `build_xas_relative_binning` (an `XASRelBin`): relative binning on the device
    # (always resident).
    if model isa IMRPhenomXAS && (relative_binning === nothing || relative_binning isa XASRelBin)
        backend = KernelAbstractions.get_backend(to_device(zeros(T, 1)))
        # Forward-tangent single pass (`variant = :fwd`; V100, H1+L1 all options, 4096 walkers: dense 31 ->
        # ~5 ms, relative binning 10 -> ~3.6 ms) whenever it applies: identical frequency grids, <= 3 detectors (<= 2 with
        # relative binning). Otherwise the adjoint path, device-resident from 128 walkers (64 walkers 4.4 vs
        # 2.7 ms for the host-stage hybrid, 256: 5.0 vs 7.9, 4096: 31 vs 85).
        dets = data isa AbstractVector ? data : [data]
        fwd_ok = all(d -> d.freqs == dets[1].freqs, dets) && length(dets) <= (relative_binning === nothing ? 3 : 2)
        return xas_binparallel_logp_and_grad(data, cfg, freqs; backend=backend, nwalkers=nwalkers, model=model,
                                             resident=fwd_ok || nwalkers >= 128, relative_binning=relative_binning,
                                             variant=fwd_ok ? :fwd : :adjoint)
    end
    # SEOBNRv5PHM (dense, one shared frequency grid): stagewise reverse on the device (`SEOBNRPHMDeviceGrad.jl`).
    if model isa SEOBNRv5PHM && relative_binning === nothing
        backend = KernelAbstractions.get_backend(to_device(zeros(T, 1)))
        return seobnr_phm_logp_and_grad(data, cfg; backend=backend, nwalkers=nwalkers, calibration=calibration,
                                        calmarg=calmarg, (box === nothing ? (;) : (; boxf=box))...)
    end
    # IMRPhenomXHM (dense, one shared frequency grid): bin-parallel forward-tangent path (`XHMBinParallel.jl`;
    # V100, H1+L1 all options: 31 ms per call at 1024 walkers, 107 ms at 4096).
    if model isa IMRPhenomXHM && relative_binning === nothing
        backend = KernelAbstractions.get_backend(to_device(zeros(T, 1)))
        return xhm_binparallel_logp_and_grad(data, cfg, freqs; backend=backend, nwalkers=nwalkers, setup=:device, bkw...)
    end
    # IMRPhenomXP / XPHM (dense, one shared frequency grid): the same mode-sum path with 5 twisted (2,2) components
    # (`XPBinParallel.jl`) / the two polarizations plus 3 hand-differentiated walker constants (`XPHMBinParallel.jl`).
    if model isa IMRPhenomXP && relative_binning === nothing
        backend = KernelAbstractions.get_backend(to_device(zeros(T, 1)))
        return xp_binparallel_logp_and_grad(data, cfg, freqs; backend=backend, nwalkers=nwalkers, setup=:device)
    end
    if model isa IMRPhenomXPHM && relative_binning === nothing
        backend = KernelAbstractions.get_backend(to_device(zeros(T, 1)))
        # every per-walker stage on the device (PrecVersion 223's setup in tangent groups: at width 7 its
        # stack overflowed CUDA's local-memory reservation, "Out of GPU memory" at launch on a V100)
        return xphm_binparallel_logp_and_grad(data, cfg, freqs; model=model, backend=backend, nwalkers=nwalkers,
                                              setup=:device, calibration=calibration,
                                              calmarg=calmarg, bkw...)
    end
    # SEOBNRv5HM (dense, one shared frequency grid): the mode-sum path on the SEOBNRv5HM device tables
    # (`SEOBNRHMBinParallel.jl`).
    if model isa GravitationalWaveforms.SEOBNRv5HM && relative_binning === nothing
        backend = KernelAbstractions.get_backend(to_device(zeros(T, 1)))
        return seobnrhm_binparallel_logp_and_grad(data, cfg, freqs; backend=backend, nwalkers=nwalkers, bkw...)
    end
    # SEOBNRv5 IMR (dense): the whole table, likelihood and gradient on the device, walker x node
    # (`SEOBNRDeviceLikelihood`, SEOBNR_GPU_PLAN.md M4). Pays only at thousands of walkers.
    if model isa GravitationalWaveforms.SEOBNRv5IMRAlignedSpinF2 && relative_binning === nothing
        backend = KernelAbstractions.get_backend(to_device(zeros(T, 1)))
        return seobnr_device_closures(data, cfg; backend=backend, nwalkers=nwalkers)[1]
    end
    # Enzyme-in-kernel gradients keep their reverse-pass caches on the device malloc heap (CUDA default 8 MB, overflowed
    # by the PA-F2 H1+L1 + options gradient at 16 walkers; 64 MB suffices for 64 walkers, 256 MB for 256): 2 MB per walker.
    GPU.ensure_device_heap(max(64, 2 * nwalkers) << 20)
    # Multi-detector data (a Vector{LikelihoodData}) is staged per detector; the PAF2 reverse-mode
    # gradient has a multi-detector precoeffs path (`_paf2_ll_reverse!`).
    data_dev = _data_to_device(data)
    freqs_dev = to_device(freqs)
    inner! = gw_logp_and_grad(data_dev, cfg, freqs_dev; model=model, relative_binning=relative_binning)

    positions_dev = compatible_array(T, (nparams, nwalkers))
    grad_dev = compatible_array(T, (nparams, nwalkers))
    logp_dev = compatible_array(T, (nwalkers,))

    return function (grad_out::AbstractMatrix{T}, logp_out::AbstractVector{T},
                      positions::AbstractMatrix{T}, active_mask)
        copyto!(positions_dev, positions)
        inner!(grad_dev, logp_dev, positions_dev, active_mask)
        copyto!(grad_out, grad_dev)
        copyto!(logp_out, logp_dev)
        return nothing
    end
end

"""
    compact_active(full!, nwalkers; narrow=()) -> (logp_and_grad!, stats)

Batched-NUTS idle walkers: every leapfrog step of a tree-doubling round calls the gradient for all `nwalkers` columns
while only the `active_mask` ones are still integrating. `narrow` holds `(W, f!)` pairs, closures of the same
untempered log-density built `W < nwalkers` walkers wide; a call with `k <= W` active columns (smallest such `W`)
gathers them into a `W`-wide batch (padded with copies of the first active column), evaluates `f!` and scatters the
results back. Columns outside the mask keep whatever `grad_out` / `logp_out` held (callers read only active columns,
as `nuts_draw!` does). A `nothing` mask or `value_only` call goes to `full!`. `stats`: `calls[W]` per width and
`active` (sum of active counts over gradient calls), for the active-fraction measurement.
"""
function compact_active(full!::F, nwalkers::Int; narrow=()) where {F}
    ws = sort!([Int(w) for (w, _) in narrow])
    fs = Dict(Int(w) => f for (w, f) in narrow)
    all(<(nwalkers), ws) || throw(ArgumentError("narrow widths must be < nwalkers = $nwalkers"))
    stats = (; calls=Dict{Int,Int}(), active=Ref(0), hist=zeros(Int, 21))
    buf = Dict(w => (Matrix{Float64}(undef, 0, w), Matrix{Float64}(undef, 0, w), zeros(w), Int[]) for w in ws)
    f = function (grad_out::AbstractMatrix{T}, logp_out::AbstractVector{T}, positions::AbstractMatrix{T}, active_mask;
                  kw...) where {T}
        k = active_mask === nothing ? nwalkers : count(active_mask)
        if isempty(kw) && active_mask !== nothing
            stats.active[] += k
            stats.hist[1 + round(Int, 20k / nwalkers)] += 1
        end
        i = findfirst(>=(k), ws)
        if active_mask === nothing || !isempty(kw) || i === nothing || k == 0
            active_mask === nothing || !isempty(kw) || (stats.calls[nwalkers] = get(stats.calls, nwalkers, 0) + 1)
            return full!(grad_out, logp_out, positions, active_mask; kw...)
        end
        W = ws[i]
        stats.calls[W] = get(stats.calls, W, 0) + 1
        X, G, L, idx = buf[W]
        np = size(positions, 1)
        if size(X, 1) != np
            X = Matrix{Float64}(undef, np, W); G = similar(X); buf[W] = (X, G, L, idx)
        end
        empty!(idx)
        for w in eachindex(active_mask)
            active_mask[w] && push!(idx, w)
        end
        @inbounds for j in 1:W
            src = idx[min(j, k)]
            for r in 1:np
                X[r, j] = positions[r, src]
            end
        end
        fs[W](G, L, X, trues(W))
        @inbounds for (j, w) in enumerate(idx)
            logp_out[w] = L[j]
            for r in 1:np
                grad_out[r, w] = G[r, j]
            end
        end
        return nothing
    end
    return f, stats
end

"""
    tempered_logp_and_grad(base!, betas, prior::GWPrior) -> logp_and_grad!

Host-side tempering + configurable prior around an UNTEMPERED `logp_and_grad!` that was built without
a `prior` (its log-density is logL plus the package's flat box prior, 0 inside the box), e.g.
[`gw_logp_and_grad_gpu_staged`](@ref), which has no `beta`/`prior` options. Returns a closure computing
`betas[w] * logL + log p(theta)` and its gradient, exactly what `gw_logp_and_grad(...; beta, prior)`
computes on the CPU path, with no kernel change -- so SMC and PT (both read `betas` by reference) run
on any backend the base closure runs on. Outside the support the result is `-Inf` with zero gradient
(never `0 * -Inf = NaN`, which matters for a beta = 0 reference rung). `box`: the support function `base!` adds
(positions -> 0 / -Inf per column), the package box by default; pass the same `box` the base closure was built with
(e.g. `prior_support_vec` of a configured prior).
"""
function tempered_logp_and_grad(base!::F, betas::AbstractVector{T}, prior::GWPrior{T};
                                box::B=_call_log_prior_vec) where {F,T,B}
    lp = T[]
    return function (grad_out::AbstractMatrix{T}, logp_out::AbstractVector{T}, positions::AbstractMatrix{T}, active_mask)
        base!(grad_out, logp_out, positions, active_mask)
        bx = box(positions)
        length(lp) == length(logp_out) || resize!(lp, length(logp_out))
        @inbounds for w in eachindex(logp_out)
            ll = isfinite(bx[w]) ? logp_out[w] - bx[w] : T(-Inf)
            if isfinite(ll)
                logp_out[w] = betas[w] * ll
                @views grad_out[:, w] .*= betas[w]
            else
                logp_out[w] = T(-Inf)
                @views grad_out[:, w] .= zero(T)
            end
        end
        prior_logpdf_and_grad!(lp, grad_out, prior, positions)
        logp_out .+= lp
        @inbounds for w in eachindex(logp_out)
            if !isfinite(logp_out[w])
                logp_out[w] = T(-Inf)
                @views grad_out[:, w] .= zero(T)
            end
        end
        return nothing
    end
end

"""
    batched_nuts!(state::PEState{T,A,V}, data::LikelihoodData, cfg::PEConfig{T};
                  nwarmup=max(cfg.nsteps ÷ 2, 1), target_accept=T(0.8), max_treedepth=10,
                  rng=Random.default_rng(), model=IMRPhenomD(), relative_binning=nothing)

Run `cfg.nsteps` batched NUTS draws over every walker in `state`, sampling the log-POSTERIOR
surface defined by `data`/`cfg`/`model` plus the uniform-bounds prior (see `_log_prior`) --
`state.positions` holds the 11-row SAMPLED convention `(mc, q, chi1, chi2, dL, t_c, phase, ra,
sin_dec, cos_theta_jn, psi)` documented above `_reparam_to_physical!`, not physical component
masses/angles.

`state.positions` is updated in place to the final draw; `state.logp` is updated to match.
`state.energy` is intentionally left untouched: unlike the old single-leapfrog-step
implementation, NUTS resamples momentum fresh at the start of every draw and discards it once the
trajectory's multinomial sample is chosen, so there is no single leftover "energy" value that
means anything once a draw completes.

`relative_binning`: forwarded to `gw_logp_and_grad` unchanged -- `nothing` (default) uses the
full-grid likelihood every step; a `RelativeBinning.RelativeBinningSetup` uses the sparse
bin-edge approximation instead. See `RelativeBinning.build_relative_binning` for how to build one
(needs a reference parameter set and `data`; a natural choice is `state`'s own starting position,
converted to physical `(m1,m2,chi1,chi2)` via `_mc_q_to_m1_m2`, before this call runs).

`enforce_bounds` (default `true`): whether to pass a [`NUTSCore.NUTSBounds`](@ref)
(built by `_sampled_space_bounds` from this file's own `_..._PRIOR_BOUNDS` constants -- the exact
bounds `_log_prior` already enforces) so `NUTSCore`'s leapfrog integration reflects/wraps at the
sampled-space boundaries instead of relying solely on `_log_prior`'s `-Inf` fallback to reject a
trajectory that crosses one (see `_sampled_space_bounds`'s comment for the reflect-vs-wrap split
and its one known gap, `IMRPhenomXP`'s joint spin-norm constraint). `_log_prior`'s `-Inf` fallback
remains active regardless of this flag -- it is the correctness backstop, not an alternative;
`enforce_bounds=false` recovers the exact pre-Phase-5 behavior (useful for A/B comparison or if a
future caller needs the raw `logp_and_grad!` unmodified).

`dense_warmup` (default `false`): when `true` (and `relative_binning !== nothing`), runs WARMUP
draws (`it <= nwarmup`) against the dense (`relative_binning=nothing`) kernel and only switches to
the requested `relative_binning` kernel once `step_size` has converged, via `NUTSCore.nuts_sample!`'s
`warmup_logp_and_grad!` keyword. Exists because a real NUTS chain's UNADAPTED warmup step size can
transiently push `logp_and_grad!` evaluations into a binned kernel's own aliasing-unsafe region
even when the CONVERGED post-warmup step size stays deeply safe there -- see
`NUTSCore.nuts_sample!`'s own `warmup_logp_and_grad!` docstring. Throws
`ArgumentError` if `dense_warmup=true` is passed with `relative_binning === nothing` (already
dense throughout; the flag would be a silent no-op otherwise, which is more likely a caller
mistake than an intentional call). `false` (default) is the exact pre-existing single-kernel
behavior for the whole chain, unchanged.

`adapt_mass` (default `false`): forwarded to `NUTSConfig` -- adapts a diagonal (per-sampled-
parameter) inverse mass matrix during warmup (Stan's windowed schedule), so a stiff direction (e.g.
phase/`t_c`) no longer forces one tiny global step size on every other parameter. `false` is the
exact pre-existing unit-mass behavior. Needs `nwarmup >= 20`. The adapted metric itself is not
returned by this function (it returns `state`); call `NUTSCore.nuts_sample!` directly if you need
`result.inv_mass`.

Note on the `PEState{T,A,V}` type parameters: this signature deliberately spells out the bound
`A<:AbstractMatrix{T}` (mirroring the struct's own declaration in Types.jl) rather than writing
`PEState{T,AbstractMatrix{T}}`, which would pin the type parameter `A` to the literal type
`AbstractMatrix{T}` itself (Julia struct type parameters are invariant) and never match a
concrete instance like `PEState{Float64,Matrix{Float64}}`.
"""
function batched_nuts!(state::PEState{T,A,V}, data::DetectorData, cfg::PEConfig{T};
                        nwarmup::Int=max(cfg.nsteps ÷ 2, 1),
                        target_accept=T(0.8),
                        max_treedepth::Int=10,
                        rng::AbstractRNG=Random.default_rng(),
                        model=IMRPhenomD(),
                        relative_binning=nothing,
                        enforce_bounds::Bool=true,
                        dense_warmup::Bool=false,
                        adapt_mass::Bool=false) where {T,A<:AbstractMatrix{T},V<:AbstractVector{T}}
    dense_warmup && relative_binning === nothing && throw(ArgumentError(
        "dense_warmup=true has no effect without a `relative_binning` kernel to switch to post-warmup"))
    freqs = _representative_freqs(data)
    logp_and_grad! = gw_logp_and_grad(data, cfg, freqs; model=model, relative_binning=relative_binning)
    warmup_logp_and_grad! = dense_warmup ?
        gw_logp_and_grad(data, cfg, freqs; model=model, relative_binning=nothing) : logp_and_grad!
    bounds = enforce_bounds ? _sampled_space_bounds(T, size(state.positions, 1)) : nothing
    nuts_cfg = NUTSConfig{T}(; max_treedepth=max_treedepth, target_accept=target_accept, nwarmup=nwarmup, bounds=bounds, adapt_mass=adapt_mass)
    step_size0 = fill(cfg.step_size, cfg.nwalkers)

    result = nuts_sample!(state.positions, logp_and_grad!; nsteps=cfg.nsteps, step_size0=step_size0,
                           cfg=nuts_cfg, rng=rng, warmup_logp_and_grad!)
    state.logp .= result.logp
    return state
end


# Shared initial-position builder for `run_gw150914_estimation` (both single- and multi-detector
# methods below) -- one function so both methods share the same 11-row initialization block.
#
# Parameter layout is the 11-row SAMPLED convention documented above _reparam_to_physical!
# (mc, q, chi1, chi2, dL, t_c, phase, ra, sin_dec, cos_theta_jn, psi). A uniform
# `fill!(positions, 30.0)` (the original placeholder here, back when the layout was raw
# component masses) is wildly unphysical for a mass ratio (valid range (0.125, 1]) as well as
# for the spins (valid range [-1,1]) -- this was latent and harmless while `batched_nuts!` was
# uncallable, but with real NUTS gradient steps now running, an out-of-prior-bounds start
# produces a `-Inf` log-posterior on the very first leapfrog step. Initialize each row to a
# physically sensible GW150914-scale starting point instead (component masses m1=36, m2=29
# Msun, converted to (mc, q); extrinsic values are approximate/illustrative -- chosen to be
# comfortably in the INTERIOR of every prior bound, including away from the sin_dec/
# cos_theta_jn +/-1 boundaries where _reparam_to_physical!'s asin/acos derivatives are
# singular -- not a scientifically precise GW150914 sky/orientation estimate).
#
# `nrows`: defaults to `11`; pass `15` for `IMRPhenomXP`/`IMRPhenomXPHM` to also
# initialize rows 12-15 (`chi1x, chi1y, chi2x, chi2y`) to a SMALL NONZERO value (`1e-3`, not `0` --
# see the correction below), comfortably inside the `_CHI_INPLANE_PRIOR_BOUNDS`/norm-constraint
# interior (see `_log_prior`'s 15-argument method).
#
# **In-plane spins do not start at zero.** `0` for all four in-plane spin components
# is an internal singularity of the precessing models (`IMRPhenomXP`/`IMRPhenomXPHM`). `GravitationalWaveforms.jl`'s
# `IMRPhenomXPrecession.jl` (`_xp_precession_setup`) computes `J0x = m1_2*chi1x + m2_2*chi2x`,
# `J0y = m1_2*chi1y + m2_2*chi2y` (lines 345-346) and `phiJ = atan(J0y, J0x)` (line 352); the
# FORWARD value `atan(0,0) == 0.0` is fine, but its Enzyme-reverse-mode DERIVATIVE is `0/0` at the
# origin (like `d(sqrt(x))/dx` at `x=0`) -- confirmed to return literal `NaN` for
# `d(logp)/d(mc,q,chi1,chi2,chi1x,chi1y,chi2x,chi2y)` alike whenever all four in-plane components
# are exactly `0`. Every walker previously started this way would hit a NaN gradient on the very
# FIRST leapfrog step of a real `run_gw150914_estimation(...; model=IMRPhenomXP())` (or
# `IMRPhenomXPHM()`) call -- and since a smaller step size does not help (`0 * NaN == NaN`, not
# `0`), `NUTSCore.nuts_draw!`'s existing `isfinite(Hw)`-gated divergence handling would silently
# discard every subtree without ever advancing the walker's position: a 100%-stuck sampler from
# step 1, with no crash and no error, for the ENTIRE run. Fixed by using `1e-3` instead of `0` --
# small enough to stay a physically-negligible perturbation of the original "near-face-on,
# near-aligned-spin" intent, large enough (`J0x`/`J0y` then `~1e-3 * m{1,2}_2`, comfortably away
# from the origin) that `atan2`'s gradient is finite and well-conditioned there.
@inline function _fill_gw150914_base_point!(positions::AbstractMatrix{T}) where {T}
    mc0, q0 = _m1_m2_to_mc_q(T(36.0), T(29.0))
    @views positions[1, :] .= mc0       # mc (chirp mass, solar masses)
    @views positions[2, :] .= q0        # q (mass ratio m2/m1, (0.125, 1])
    @views positions[3, :] .= zero(T)   # chi1 (dimensionless spin, [-1,1])
    @views positions[4, :] .= zero(T)   # chi2 (dimensionless spin, [-1,1])
    @views positions[5, :] .= T(0.41)   # dL (Gpc), ~GW150914 scale
    @views positions[6, :] .= zero(T)   # t_c (s), offset from gps_time
    @views positions[7, :] .= zero(T)   # phase (rad)
    @views positions[8, :] .= T(1.95)   # ra (rad)
    @views positions[9, :] .= T(-0.9)   # sin_dec -> dec = asin(-0.9)
    @views positions[10, :] .= zero(T)  # cos_theta_jn -> theta_jn = pi/2
    @views positions[11, :] .= T(0.8)   # psi (rad)
    nrows = size(positions, 1)
    if nrows >= 15
        # NOT zero -- see this function's docstring's "Correction" note: exactly zero for all four
        # of these simultaneously is a real Enzyme-gradient singularity in GravitationalWaveforms.jl's
        # precession setup (IMRPhenomXPrecession.jl:345-352), not a safe interior point.
        @views positions[12, :] .= T(1e-3)  # chi1x
        @views positions[13, :] .= T(1e-3)  # chi1y
        @views positions[14, :] .= T(1e-3)  # chi2x
        @views positions[15, :] .= T(1e-3)  # chi2y
    end
    return positions
end

"""
    _gw150914_initial_positions(::Type{T}, nwalkers, nrows=11; rng=nothing, dispersion=0.02)

Builds the shared GW150914-scale starting point above. When `rng === nothing` (the default,
unchanged from before this dispersion feature), every walker starts at the IDENTICAL point, same
as always. When an `rng` is passed, each walker is additionally perturbed by independent Gaussian
noise, per row, with standard deviation `dispersion * (hi - lo)` of that row's own
`_sampled_space_bounds` range (default `dispersion=0.02`, i.e. 2% of each parameter's prior
width) -- small enough to stay a physically-negligible perturbation of the GW150914-scale intent
(same spirit as the `1e-3` in-plane-spin nudge above), large enough to break the exact-symmetry
start that made `Rhat` meaningless (identical walkers can never look non-mixed OR mixed; there is
nothing to compare). Periodic rows (`phase`, `ra`, `psi`) wrap via `mod`; box rows clamp into
`[lo, hi]`.

Dispersion is generated on the HOST (a plain `Matrix{T}`, then moved with `to_device` at the end)
rather than by broadcasting an `AbstractRNG` draw directly onto a `compatible_array`: the
no-dispersion path above may return a `CuArray`, and broadcasting a host-generated random `Array`
onto it would mix host and device arrays (not supported by CUDA.jl's broadcast machinery) --
matching this file's/`GPU.jl`'s own established "build on host, `to_device` once" convention for
exactly this kind of situation, rather than the scalar-fill-only path used when there's no host
data involved.
"""
function _gw150914_initial_positions(::Type{T}, nwalkers::Int, nrows::Int=11;
                                      rng::Union{Nothing,Random.AbstractRNG}=nothing,
                                      dispersion::Real=0.02) where {T}
    if rng === nothing
        positions = compatible_array(T, (nrows, nwalkers))
        _fill_gw150914_base_point!(positions)
        return positions
    end
    positions = Matrix{T}(undef, nrows, nwalkers)
    _fill_gw150914_base_point!(positions)
    bounds = _sampled_space_bounds(T, nrows)
    for i in 1:nrows
        lo, hi = bounds.lo[i], bounds.hi[i]
        span = hi - lo
        scale = T(dispersion) * span
        for w in 1:nwalkers
            x = positions[i, w] + scale * randn(rng, T)
            positions[i, w] = bounds.periodic[i] ? (mod(x - lo, span) + lo) : clamp(x, lo, hi)
        end
    end
    return to_device(positions)
end

"""
    run_gw150914_estimation(data::LikelihoodData; nwalkers=8, nsteps=32, step_size=1e-3,
                             nwarmup=max(nsteps÷2,1), max_treedepth=10, rng=Random.default_rng(),
                             model=IMRPhenomXAS(), relative_binning=nothing) -> PEState

One-call, single-detector convenience entry point: builds a `PEConfig` from `data`'s own
frequency grid, initializes `nwalkers` NUTS chains at a physically-sensible GW150914-scale
starting point (component masses `m1=36`, `m2=29` Msun, converted to the sampled `(mc, q)`
reparametrization; extrinsic values approximate/illustrative -- see `_gw150914_initial_positions`
for the full per-row rationale, including why they're chosen away from prior-boundary
singularities), then runs [`batched_nuts!`](@ref) to completion and returns the resulting
`PEState`. `model isa IMRPhenomXP` uses 15 sampled rows (adds 4 in-plane spin components,
initialized to `0`) instead of the aligned-spin default's 11. See the multi-detector method below
for the `data::AbstractVector{<:LikelihoodData}` (H1+L1[+V1]) version.

Walkers start DISPERSED around that point (using this function's own `rng`), not identically --
see `_gw150914_initial_positions`'s docstring; identical starts made `Rhat` vacuous (nothing to
distinguish "mixed" from "not mixed").
"""
function run_gw150914_estimation(data::LikelihoodData; nwalkers::Int=8, nsteps::Int=32, step_size::T=1e-3,
                                  nwarmup::Int=max(nsteps ÷ 2, 1), max_treedepth::Int=10,
                                  rng::AbstractRNG=Random.default_rng(), model=IMRPhenomXAS(),
                                  relative_binning=nothing, enforce_bounds::Bool=true,
                                  adapt_mass::Bool=false) where {T}
    cfg = PEConfig{T}(length(data.freqs), nwalkers, step_size, nsteps)

    # IMRPhenomXPHM also needs the 15-row
    # precessing-spin sampled space (KernelFused.jl/RelativeBinning.jl both assert
    # `size(params,1) == 15` for `model isa IMRPhenomXP || model isa IMRPhenomXPHM`) -- this branch
    # must cover both, since calling this convenience entry point with
    # `model=IMRPhenomXPHM()` built an 11-row `positions`/`momenta` and would have hit that
    # assertion on the very first likelihood evaluation (`gw_logp_and_grad`/`batched_nuts!` take
    # `positions` as given, whatever its row count).
    nrows = (model isa _Prec15) ? 15 : 11
    positions = _gw150914_initial_positions(T, nwalkers, nrows; rng=rng)
    momenta = compatible_array(T, (nrows, nwalkers))
    fill!(momenta, zero(T))

    # logp/energy built via compatible_array (not plain zeros(T, ...)) so they live on the same
    # device as positions/momenta -- see PEState's docstring in Types.jl.
    logp = compatible_array(T, (nwalkers,))
    energy = compatible_array(T, (nwalkers,))
    fill!(logp, zero(T))
    fill!(energy, zero(T))

    state = PEState(positions, momenta, logp, energy)
    batched_nuts!(state, data, cfg; nwarmup=nwarmup, max_treedepth=max_treedepth, rng=rng, model=model,
                  relative_binning=relative_binning, enforce_bounds=enforce_bounds, adapt_mass=adapt_mass)
    return state
end

"""
    run_gw150914_estimation(data::AbstractVector{<:LikelihoodData}; nwalkers=8, nsteps=32, ...)

Multi-detector method (also covers `IMRPhenomXP`):
identical to the single-detector method above (same prior/reparametrization, same GW150914-scale
starting point via `_gw150914_initial_positions`), except `data` is a vector of per-detector
`LikelihoodData` (e.g. `[h1_data, l1_data]`) and `cfg`'s representative frequency grid/`ntime` are
taken from the FIRST detector (`data[1]`) -- `cfg.ntime` is not actually consulted by the
multi-detector likelihood kernel (each detector's own grid length is read directly from that
detector's `LikelihoodData`), it only needs SOME representative value here for `PEConfig`'s other
fields (`dt`, `sample_rate`). Like the single-detector method, `positions`/`momenta` are built with
15 rows (not 11) when `model isa IMRPhenomXP`, initializing the 4 new in-plane spin components to `0`.
"""
function run_gw150914_estimation(data::AbstractVector{<:LikelihoodData}; nwalkers::Int=8, nsteps::Int=32,
                                  step_size::T=1e-3, nwarmup::Int=max(nsteps ÷ 2, 1), max_treedepth::Int=10,
                                  rng::AbstractRNG=Random.default_rng(), model=IMRPhenomXAS(),
                                  relative_binning=nothing, enforce_bounds::Bool=true,
                                  adapt_mass::Bool=false) where {T}
    @assert !isempty(data) "run_gw150914_estimation needs at least one detector"
    ref = data[1]
    cfg = PEConfig{T}(length(ref.freqs), nwalkers, step_size, nsteps)

    # As in the single-detector method: IMRPhenomXPHM needs 15 rows too.
    nrows = (model isa _Prec15) ? 15 : 11
    positions = _gw150914_initial_positions(T, nwalkers, nrows; rng=rng)
    momenta = compatible_array(T, (nrows, nwalkers))
    fill!(momenta, zero(T))

    logp = compatible_array(T, (nwalkers,))
    energy = compatible_array(T, (nwalkers,))
    fill!(logp, zero(T))
    fill!(energy, zero(T))

    state = PEState(positions, momenta, logp, energy)
    batched_nuts!(state, data, cfg; nwarmup=nwarmup, max_treedepth=max_treedepth, rng=rng, model=model,
                  relative_binning=relative_binning, enforce_bounds=enforce_bounds, adapt_mass=adapt_mass)
    return state
end

# ------------------------------------------------------------------------------------------------
# GPU-native parallel tempering. Built
# entirely on top of already-existing, already-validated pieces: `gw_logp_and_grad`'s new `beta`
# kwarg (tempers the likelihood term per lane), `_raw_loglikelihood!` (the swap round's own
# untempered-logL re-evaluation), and `NUTSCore.PTLadderState`/`NUTSCore.pt_swap!` (the generic,
# GW-agnostic ladder bookkeeping and swap kernel -- see that module's own section comment for why
# the swap step itself lives there, not here). This file's job is only the GW-specific wiring:
# building the tempered `logp_and_grad!` closure, running it through `NUTSCore.nuts_sample!`'s new
# `callback` hook to interleave swap rounds between NUTS draws, and tracking which lane currently
# holds the `beta=1` ("cold") chain -- the only lane whose draws are actual posterior samples.
# ------------------------------------------------------------------------------------------------

"""
    batched_pt_nuts!(state::PEState{T,A,V}, data::DetectorData, cfg::PEConfig{T};
                     ntemps=size(state.positions,2), beta_min=0.05, swap_interval=5,
                     nwarmup=max(cfg.nsteps÷2,1), target_accept=T(0.8), max_treedepth=10,
                     rng=Random.default_rng(), model=IMRPhenomD(), relative_binning=nothing,
                     enforce_bounds=true, pt_backend=KernelAbstractions.CPU(), adapt_mass=false)
                     -> (state, pt::PTLadderState, cold_chain)

Parallel-tempered batched NUTS. Unlike [`batched_nuts!`](@ref) (every lane an independent walker
sampling the SAME `beta=1` posterior), each lane here runs an independent NUTS chain at its OWN
currently-assigned temperature (`log_L_tempered = beta * log_L`, log-prior left untempered) --
`state.positions` must therefore have exactly `ntemps` columns, one lane per rung of the ladder
(`ntemps == nwalkers` is asserted, matching `NUTSCore`'s "each walker/temperature is already an
independent trajectory" design).

Every `swap_interval` draws (via `NUTSCore.nuts_sample!`'s `callback` hook), lane temperatures are
proposed to swap with their current adjacent-in-rank neighbor (`NUTSCore.pt_swap!`, even/odd
schedule) using each lane's freshly re-evaluated raw log-likelihood (`_raw_loglikelihood!` --
NOT `nuts_draw!`'s own tempered `logp`, see that function's docstring for why). `swap_interval=1`
proposes a swap after every single draw; larger values let each temperature's chain mix locally
between swap attempts, the usual PT tradeoff.

Step-size adaptation (dual averaging) is already PER-WALKER in `NUTSCore` (unrelated to this
feature), which is exactly what different-temperature lanes need anyway: a hot (small-`beta`,
flattened) lane's well-adapted step size can be very different from the cold lane's.

Returns `(state, pt, cold_chain)`: `state.positions`/`state.logp` are left as
`NUTSCore.nuts_sample!` leaves them (every lane's FINAL draw, at whatever temperature that lane
ended up holding -- NOT necessarily `beta=1`, since swaps permute temperature labels across
lanes over the run). `pt::NUTSCore.PTLadderState` is the final ladder/lane bookkeeping
(`pt.naccept ./ pt.nattempt`, one entry per adjacent pair indexed by its lower rank, gives each
pair's running swap acceptance rate -- the standard PT tuning diagnostic). `cold_chain` is a
`(nparams, cfg.nsteps)` array: the `beta=1` chain's position after EVERY draw, assembled by
following `pt.lane_of_rank[1]` (whichever lane currently holds `beta=1`) at each step -- THIS,
not any single fixed lane of `state.positions`, is the array a caller should treat as posterior
samples from the target distribution.

`pt_backend` (default `KernelAbstractions.CPU()`): backend `PTLadderState`'s `betas`/
`lane_of_rank`/`rank_of_lane` are allocated on -- pass `CUDABackend()` (after `using CUDA`) to run
the swap kernel itself on-device; independent of whatever backend `state.positions` lives on
(`_raw_loglikelihood!`/`gw_logp_and_grad` already dispatch their own kernel backend from
`positions`/`data`, same as `batched_nuts!`). Not validated on CUDA hardware; see `NUTSCore`'s PT
section comment.

`adapt_mass` (default `false`): adapt a diagonal mass matrix during warmup, attached to the
temperature RANK (via `pt.rank_of_lane`), not to the lane -- lanes exchange beta labels on a swap, so a
per-lane metric would go stale after every accepted swap. Each rank learns its own scales (hot ranks
legitimately broader). See `NUTSCore.nuts_sample!`'s `metric_rank`. Step sizes remain per-lane (they
are dual-averaged during warmup only and are not rank-attached).
"""
function batched_pt_nuts!(state::PEState{T,A,V}, data::DetectorData, cfg::PEConfig{T};
                           ntemps::Int=size(state.positions, 2),
                           beta_min=0.05,
                           swap_interval::Int=5,
                           nwarmup::Int=max(cfg.nsteps ÷ 2, 1),
                           target_accept=T(0.8),
                           max_treedepth::Int=10,
                           rng::AbstractRNG=Random.default_rng(),
                           model=IMRPhenomD(),
                           relative_binning=nothing,
                           enforce_bounds::Bool=true,
                           pt_backend=KernelAbstractions.CPU(),
                           adapt_mass::Bool=false) where {T,A<:AbstractMatrix{T},V<:AbstractVector{T}}
    nwalkers = size(state.positions, 2)
    ntemps == nwalkers || throw(ArgumentError(
        "batched_pt_nuts! needs one lane per temperature: ntemps=$ntemps must equal nwalkers=$nwalkers"))
    swap_interval >= 1 || throw(ArgumentError("swap_interval must be >= 1"))

    pt = PTLadderState(ntemps; T=T, beta_min=T(beta_min), backend=pt_backend)

    freqs = _representative_freqs(data)
    logp_and_grad! = gw_logp_and_grad(data, cfg, freqs; model=model, relative_binning=relative_binning,
                                       beta=pt.betas)
    bounds = enforce_bounds ? _sampled_space_bounds(T, size(state.positions, 1)) : nothing
    nuts_cfg = NUTSConfig{T}(; max_treedepth=max_treedepth, target_accept=target_accept, nwarmup=nwarmup, bounds=bounds, adapt_mass=adapt_mass)
    step_size0 = fill(cfg.step_size, cfg.nwalkers)

    nparams = size(state.positions, 1)
    cold_chain = Array{T}(undef, nparams, cfg.nsteps)
    ll_scratch = similar(state.positions, T, nwalkers)

    swap_round! = function (it::Int, positions::AbstractMatrix{T}, logp::AbstractVector{T},
                             step_size::AbstractVector{T}, stats)
        if it % swap_interval == 0
            _raw_loglikelihood!(ll_scratch, positions, freqs, cfg, data, model, relative_binning)
            pt_swap!(pt, ll_scratch, rng)
        end
        cold_lane = Array(pt.lane_of_rank)[1]
        @views cold_chain[:, it] .= Array(positions[:, cold_lane])
        return nothing
    end

    result = nuts_sample!(state.positions, logp_and_grad!; nsteps=cfg.nsteps, step_size0=step_size0,
                           cfg=nuts_cfg, rng=rng, callback=swap_round!,
                           metric_rank=adapt_mass ? pt.rank_of_lane : nothing)
    state.logp .= result.logp
    return state, pt, cold_chain
end

"""
    run_gw150914_pt_estimation(data::DetectorData; ntemps=4, nsteps=32, step_size=1e-3,
                               beta_min=1e-2, swap_interval=5, nwarmup=max(nsteps÷2,1),
                               max_treedepth=10, rng=Random.default_rng(), model=IMRPhenomXAS(),
                               relative_binning=nothing, enforce_bounds=true,
                               pt_backend=KernelAbstractions.CPU(), adapt_mass=false) -> (state, pt, cold_chain)

Parallel-tempering analogue of [`run_gw150914_estimation`](@ref): same GW150914-scale starting
point (`_gw150914_initial_positions`, one copy per temperature rung rather than per independent
walker), same single-/multi-detector genericity (`data::DetectorData`, dispatched exactly as
`_representative_freqs`/`_raw_loglikelihood!`/`_eval_ll!` already do throughout this file), but
runs [`batched_pt_nuts!`](@ref) instead of [`batched_nuts!`](@ref) -- see that function's
docstring for the full `ntemps`/`swap_interval`/`cold_chain` semantics.
"""
function run_gw150914_pt_estimation(data::DetectorData; ntemps::Int=4, nsteps::Int=32, step_size::T=1e-3,
                                     beta_min=1e-2, swap_interval::Int=5,
                                     nwarmup::Int=max(nsteps ÷ 2, 1), max_treedepth::Int=10,
                                     rng::AbstractRNG=Random.default_rng(), model=IMRPhenomXAS(),
                                     relative_binning=nothing, enforce_bounds::Bool=true,
                                     pt_backend=KernelAbstractions.CPU(),
                                     adapt_mass::Bool=false) where {T}
    freqs = _representative_freqs(data)
    cfg = PEConfig{T}(length(freqs), ntemps, step_size, nsteps)

    nrows = (model isa _Prec15) ? 15 : 11
    positions = _gw150914_initial_positions(T, ntemps, nrows)
    momenta = compatible_array(T, (nrows, ntemps))
    fill!(momenta, zero(T))

    logp = compatible_array(T, (ntemps,))
    energy = compatible_array(T, (ntemps,))
    fill!(logp, zero(T))
    fill!(energy, zero(T))

    state = PEState(positions, momenta, logp, energy)
    _, pt, cold_chain = batched_pt_nuts!(state, data, cfg; ntemps=ntemps, beta_min=beta_min,
                                          swap_interval=swap_interval, nwarmup=nwarmup,
                                          max_treedepth=max_treedepth, rng=rng, model=model,
                                          relative_binning=relative_binning, enforce_bounds=enforce_bounds,
                                          pt_backend=pt_backend, adapt_mass=adapt_mass)
    return state, pt, cold_chain
end


# ------------------------------------------------------------------------------------------------
# Phase reconstruction for phase-marginalized runs (`PEConfig.marginalize_phase = true`).
# ------------------------------------------------------------------------------------------------

# Best & Fisher (1979) von Mises sampler, valid for every kappa >= 0.
function _rand_von_mises(rng::AbstractRNG, mu::T, kappa::T) where {T}
    kappa < T(1e-8) && return mod(mu + 2 * T(pi) * rand(rng, T) - T(pi), 2 * T(pi))
    a = 1 + sqrt(1 + 4 * kappa^2)
    b = (a - sqrt(2 * a)) / (2 * kappa)
    r = (1 + b^2) / (2 * b)
    while true
        u1, u2, u3 = rand(rng, T), rand(rng, T), rand(rng, T)
        z = cos(T(pi) * u1)
        f = (1 + r * z) / (r + z)
        c = kappa * (r - f)
        if c * (2 - c) - u2 > 0 || log(c / u2) + 1 - c >= 0
            theta = sign(u3 - T(0.5)) * acos(clamp(f, -one(T), one(T)))
            return mod(mu + theta, 2 * T(pi))
        end
    end
end

"""
    reconstruct_marginalized!(positions, data, cfg, model; rng=Random.default_rng()) -> positions

For a run sampled with the phase and/or distance analytically marginalized (`cfg.marginalize_phase`,
`cfg.marginalize_distance`), overwrite the marginalized rows (7 = phase, 5 = dL) of every walker column
with exact draws from their conditional posterior given the other parameters. Nothing is changed for a
plain `cfg`.

With `h = h0 * exp(-i*phase) * (dL0/dL)` the plain log-likelihood is
`C - <h0,h0> u^2/2 + Re(z0 exp(-i*phase)) u` (`u = dL0/dL`, `z0 = <d,h0>`), read off five plain-likelihood
evaluations (phase 0, pi/2, pi, 3pi/2 at the walker's dL, and phase 0 at 2 dL). Distance is drawn first
(on a fine grid over its prior range, with the phase integrated out when it is marginalized), then the
phase given that distance from its von Mises conditional (mean `arg z0`, concentration `|z0| u`).
`positions` may also be a recorded chain `(nparams, nwalkers, ndraws)`.

`loglik` (precessing models): a batched `(ll, positions_sampled_space) -> ll` evaluating the PLAIN likelihood (no
marginalization) used instead of the CPU reference path, e.g. the SEOBNRv5PHM device likelihood built with
`marginalize_distance = false` -- required for SEOBNRv5PHM with `time_ref_frequency` / `phase_ref_frequency`, which the
CPU path does not implement (it would align the template with t_ref read as t_c: the distance pinned at its maximum).
"""
function reconstruct_marginalized!(positions::AbstractMatrix{T}, data::DetectorData, cfg::PEConfig{T}, model;
                                   rng::AbstractRNG=Random.default_rng(), ngrid::Int=4000, loglik=nothing,
                                   curve_loglik=nothing, curves=nothing) where {T}
    (cfg.marginalize_phase || cfg.marginalize_distance) || return positions
    if curve_loglik !== nothing
        # calibration marginalization: the likelihood is a mean over response curves, NOT one quadratic in u
        cfg.marginalize_phase && throw(ArgumentError("calibration marginalization: no phase marginalization"))
        return _calmarg_reconstruct_distance!(positions, cfg, curve_loglik, curves; rng, ngrid)
    end
    ((model isa SEOBNRv5PHM && (cfg.time_ref_frequency > 0 || cfg.phase_ref_frequency > 0) ||
      model isa IMRPhenomXHM && cfg.phase_ref_frequency > 0) && loglik === nothing) &&
        throw(ArgumentError("reconstruct_marginalized!: SEOBNRv5PHM with time_ref_frequency / phase_ref_frequency needs " *
                            "`loglik` (the device likelihood); the CPU path does not implement them"))
    nbase = model isa _Prec15 ? 15 : 11            # (+ calibration rows when the supplied `loglik` takes them)
    (size(positions, 1) == nbase || (loglik !== nothing && size(positions, 1) > nbase)) ||
        throw(ArgumentError("reconstruct_marginalized! expects the sampled space of $(typeof(model))"))
    freqs = _representative_freqs(data)
    plain = with_likelihood_options(cfg; marginalize_phase=false, marginalize_distance=false)
    n = size(positions, 2)
    trial = copy(positions)
    physical = similar(trial)
    if model isa Union{IMRPhenomXHM,GravitationalWaveforms.SEOBNRv5HM,_Prec15}
        # Higher modes / precession: the phase is sampled (no overall e^{-i phase}); only the distance is marginalized. At the
        # walker's own phase, logL(D) = c - hh u^2 / 2 + z u with u = dL0 / D: from D = dL0, 2 dL0 and ~infinity.
        cfg.marginalize_phase && throw(ArgumentError("marginalize_phase is not valid for $(typeof(model))"))
        llx = ntuple(_ -> zeros(T, n), 3)
        lo, hi, kpow = cfg.distance_lo, cfg.distance_hi, cfg.distance_power
        # three distances: the CPU path dL0 x (1, 2, 1e6); a supplied `loglik` (e.g. the device likelihood, which returns
        # -Inf outside the prior box) three inside the prior range, hi x (1, 1/2, 1/4), with dL0 = hi
        Ds = loglik === nothing ? nothing : (hi, hi / 2, hi / 4)
        for k in 1:3
            trial .= positions
            if loglik === nothing
                trial[5, :] .*= (one(T), T(2), T(1e6))[k]
                _reparam_to_physical!(physical, trial)
                _eval_ll!(llx[k], physical, freqs, plain, data, model, nothing)
            else
                trial[5, :] .= Ds[k]
                loglik(llx[k], trial)
            end
        end
        grid = collect(range(lo, hi; length=ngrid))
        logw = similar(grid)
        for w in 1:n
            if loglik === nothing
                a1 = llx[1][w] - llx[3][w]; a2 = llx[2][w] - llx[3][w]
                hh = 4 * (2 * a2 - a1)
                z = a1 + hh / 2
                dL0 = positions[5, w]
            else
                # u = 1, 2, 4 (dL0 = hi): L(u) = c - hh u^2 / 2 + z u  ->  L2 - L1 = -3 hh / 2 + z, L4 - L2 = -6 hh + 2 z
                d21 = llx[2][w] - llx[1][w]; d42 = llx[3][w] - llx[2][w]
                hh = (2 * d21 - d42) / 3
                z = d21 + 3 * hh / 2
                dL0 = hi
            end
            for g in eachindex(grid)
                u = dL0 / grid[g]
                logw[g] = kpow * log(grid[g]) - hh * u^2 / 2 + z * u
            end
            m = maximum(logw)
            cdf = cumsum(exp.(logw .- m))
            g = clamp(searchsortedfirst(cdf, rand(rng, T) * cdf[end]), 1, ngrid)
            positions[5, w] = clamp(grid[g] + (rand(rng, T) - T(0.5)) * (grid[2] - grid[1]), lo, hi)
        end
        return positions
    end
    ll = ntuple(_ -> zeros(T, n), 5)
    for k in 1:5
        trial .= positions
        trial[7, :] .= k == 5 ? zero(T) : (k - 1) * T(pi) / 2
        k == 5 && (trial[5, :] .*= 2)
        _reparam_to_physical!(physical, trial)
        _eval_ll!(ll[k], physical, freqs, plain, data, model, nothing)
    end
    lo, hi, kpow = cfg.distance_lo, cfg.distance_hi, cfg.distance_power
    grid = cfg.marginalize_distance ? collect(range(lo, hi; length=ngrid)) : T[]
    logw = similar(grid)
    for w in 1:n
        zr = (ll[1][w] - ll[3][w]) / 2
        zi = (ll[2][w] - ll[4][w]) / 2
        hh = (ll[5][w] - ll[1][w] + zr / 2) * 8 / 3
        absz = sqrt(zr^2 + zi^2)
        dL0 = positions[5, w]
        if cfg.marginalize_distance
            # Re(z0 e^{-i phase}) at the walker's own phase when the phase itself was sampled
            ph = positions[7, w]
            zeff = zr * cos(ph) + zi * sin(ph)
            for g in eachindex(grid)
                u = dL0 / grid[g]
                logw[g] = kpow * log(grid[g]) - hh * u^2 / 2 +
                          (cfg.marginalize_phase ? KernelFused._log_besseli0(absz * u) : zeff * u)
            end
            m = maximum(logw)
            cdf = cumsum(exp.(logw .- m))
            r = rand(rng, T) * cdf[end]
            g = clamp(searchsortedfirst(cdf, r), 1, ngrid)
            # jitter within the grid cell so draws are continuous
            positions[5, w] = clamp(grid[g] + (rand(rng, T) - T(0.5)) * (grid[2] - grid[1]), lo, hi)
        end
        if cfg.marginalize_phase
            u = dL0 / positions[5, w]
            # Re(z0 u e^{-i phase}) = |z0| u cos(phase - arg z0)
            positions[7, w] = _rand_von_mises(rng, atan(zi, zr), absz * u)
        end
    end
    return positions
end

function reconstruct_marginalized!(chain::AbstractArray{T,3}, data::DetectorData, cfg::PEConfig{T}, model;
                                   rng::AbstractRNG=Random.default_rng(), loglik=nothing, curve_loglik=nothing,
                                   curves=nothing) where {T}
    curves === nothing || size(curves) == (size(chain, 2), size(chain, 3)) ||
        throw(ArgumentError("curves must be nwalkers x ndraws"))
    for it in axes(chain, 3)
        draw = chain[:, :, it]
        reconstruct_marginalized!(draw, data, cfg, model; rng=rng, loglik=loglik, curve_loglik=curve_loglik,
                                  curves=curves === nothing ? nothing : view(curves, :, it))
        chain[:, :, it] .= draw
    end
    return chain
end

"""
    _calmarg_reconstruct_distance!(positions, cfg, curve_loglik, curves; rng, ngrid) -> positions

Distance reconstruction for a calibration-MARGINALIZED run (`curve_loglik` from the run's likelihood with response curves
`C_k`): a joint draw of (curve k, d_L) from p(k, D | theta) ~ p(D) exp(l_k(theta, D)). Every curve's plain
log-likelihood is exactly quadratic in u = D0 / D, l_k = c - hh_k u^2 / 2 + z_k u, but the curve-AVERAGED likelihood
log mean_k exp(l_k) is not, so the single quadratic `reconstruct_marginalized!` fits to it from far-away distances
(D0 = hi, hi/2, hi/4) misplaces the d_L conditional (GW150914 XPHM d_L median 454 vs 468 Mpc exact). Here
k is drawn first with weight exp(v_k), v_k = log int p(D) exp(l_k) dD the per-curve distance-marginalized
log-likelihood the sampler itself uses, then D from that curve's quadratic (read off three plain evaluations, as for the
uncalibrated case). `curve_loglik = (; ncurves, marg, plain)`: `marg(LK, pos)` fills the per-curve distance-marginalized and
`plain(LK, pos)` the per-curve plain log-likelihoods (ncurves x nwalkers) at the positions' d_L (row 5). `curves`
(optional, length nwalkers): receives the drawn curve indices, for the calibration posterior.
"""
function _calmarg_reconstruct_distance!(positions::AbstractMatrix{T}, cfg::PEConfig{T}, curve_loglik, curves;
                                        rng::AbstractRNG, ngrid::Int) where {T}
    n = size(positions, 2)
    lo, hi, kpow = cfg.distance_lo, cfg.distance_hi, cfg.distance_power
    k = zeros(Int, n)
    trial = copy(positions)
    nc = curve_loglik.ncurves
    LKm = zeros(T, nc, n)
    curve_loglik.marg(LKm, trial)
    for w in 1:n
        lw = view(LKm, :, w); m = maximum(lw)
        isfinite(m) || (k[w] = 1; continue)
        c = zero(T); s = sum(x -> exp(x - m), lw); r = rand(rng, T) * s
        kk = nc
        for i in 1:nc
            c += exp(lw[i] - m)
            c >= r && (kk = i; break)
        end
        k[w] = kk
    end
    # the chosen curve's plain log-likelihood at u = 1, 2, 4 (D0 = hi): l(u) = c - hh u^2 / 2 + z u
    Ds = (hi, hi / 2, hi / 4)
    lsel = ntuple(_ -> zeros(T, n), 3)
    for j in 1:3
        trial .= positions; trial[5, :] .= Ds[j]
        curve_loglik.plain(LKm, trial)
        for w in 1:n; lsel[j][w] = LKm[k[w], w]; end
    end
    grid = collect(range(lo, hi; length=ngrid))
    logw = similar(grid)
    for w in 1:n
        d21 = lsel[2][w] - lsel[1][w]; d42 = lsel[3][w] - lsel[2][w]
        hh = (2 * d21 - d42) / 3
        z = d21 + 3 * hh / 2
        for g in eachindex(grid)
            u = hi / grid[g]
            logw[g] = kpow * log(grid[g]) - hh * u^2 / 2 + z * u
        end
        m = maximum(logw)
        cdf = cumsum(exp.(logw .- m))
        g = clamp(searchsortedfirst(cdf, rand(rng, T) * cdf[end]), 1, ngrid)
        positions[5, w] = clamp(grid[g] + (rand(rng, T) - T(0.5)) * (grid[2] - grid[1]), lo, hi)
    end
    curves === nothing || (curves .= k)
    return positions
end


# ------------------------------------------------------------------------------------------------
# Discrete jump moves for NUTSCore.nuts_sample!(...; jumps) (see `NUTSCore._jump_step!`).
# ------------------------------------------------------------------------------------------------

"""
    gw_jump_moves(data; sky=true, inclination=true) -> Vector{Function}

Involutive, unit-Jacobian maps of the 11-row sampled space for the classic discrete degeneracies of an
aligned-spin GW posterior, to be used as Metropolis jumps between NUTS draws:

- **inclination flip** `cos_theta_jn -> -cos_theta_jn`: the antenna combination
  `F+ (1+c^2)/2 - i Fx c` becomes its complex conjugate, i.e. only a per-detector phase changes -- an
  exact symmetry for one detector with the phase marginalized, a near-symmetry for a network.
- **sky reflection** (exactly two detectors): reflect the source direction through the plane that
  contains the geocenter and both detector vertices. Every arrival-time delay (geocentric and relative)
  is preserved exactly, so this maps between the two lobes of the timing ring that a two-detector
  network cannot tell apart except through the antenna patterns. An isometry of the sphere, and
  `(ra, sin_dec)` is area-uniform, so the Jacobian is 1.
- their composition (commuting involutions, so also an involution).

The time row is untouched, which is consistent for every `PEConfig.time_reference` (the reflection
preserves all detector delays) and for `time_ref_frequency` (intrinsic parameters unchanged).
"""
function gw_jump_moves(data::DetectorData; sky::Bool=true, inclination::Bool=true)
    moves = Function[]
    flip! = x -> (x[10] = -x[10]; nothing)
    inclination && push!(moves, flip!)
    dets = data isa AbstractVector ? data : [data]
    if sky && length(dets) == 2
        g = gmst(dets[1].gps_time)
        r1 = collect(geometry(dets[1].detector).location)
        r2 = collect(geometry(dets[2].detector).location)
        nrm = [r1[2] * r2[3] - r1[3] * r2[2], r1[3] * r2[1] - r1[1] * r2[3], r1[1] * r2[2] - r1[2] * r2[1]]
        nrm ./= sqrt(sum(abs2, nrm))
        reflect! = function (x)
            ra, sd = x[8], clamp(x[9], -1.0, 1.0)
            cd = sqrt(max(1 - sd^2, 0.0))
            gha = g - ra
            e = (cd * cos(gha), -cd * sin(gha), sd)            # same frame as GravitationalWaveData.time_delay_from_geocenter_gmst
            d = e[1] * nrm[1] + e[2] * nrm[2] + e[3] * nrm[3]
            e2 = (e[1] - 2d * nrm[1], e[2] - 2d * nrm[2], e[3] - 2d * nrm[3])
            x[9] = clamp(e2[3], -1.0, 1.0)
            gha2 = atan(-e2[2], e2[1])
            x[8] = mod(g - gha2, 2pi)
            return nothing
        end
        push!(moves, reflect!)
        inclination && push!(moves, x -> (reflect!(x); flip!(x)))
    end
    return moves
end

end
