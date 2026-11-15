module AnalyticInit

# ---------------------------------------------------------------------------------------------
# Analytic pre-computation from a reference waveform. Everything in `NUTSCore`'s `hessian_init`/
# `cull_logp_gap` learns from walkers that already exist and have already taken steps -- circular
# when the very first steps are badly scaled (on GW150914: step_size stuck at
# ~1e-6, walkers barely move, the metric learns the scale of its own jitter). This module instead
# computes a metric, a set of dispersed starting positions, and a PT temperature floor directly
# from a single reference point and the data, BEFORE any walker exists or takes a step -- using
# IMRPhenomXHM (cheap, already Enzyme/KernelAbstractions-differentiable) rather than SEOBNR, since
# the whole point is to get a good starting metric before paying SEOBNR's per-draw cost at all.
# ---------------------------------------------------------------------------------------------

using Random
using LinearAlgebra: Diagonal

using ..Types: PEConfig, LikelihoodData, DetectorData
using GravitationalWaveforms: IMRPhenomXHM
using ..NUTSCore: DenseMetric, NUTSConfig, _hessian_metric
using ..BatchedNUTS: gw_logp_and_grad, _fill_gw150914_base_point!, _sampled_space_bounds

export analytic_reference_metric, informed_initial_positions, analytic_snr2, analytic_beta_min

"""
    analytic_reference_metric(data::LikelihoodData, freqs; model=IMRPhenomXHM(), nrows=11,
                               T=Float64, hessian_rel_step=1e-6) -> (metric, ref_point)

Finite-difference Hessian/Fisher metric of the log-posterior at the single GW150914-scale
reference point [`_fill_gw150914_base_point!`](@ref) builds (the same point
`_gw150914_initial_positions` disperses around), evaluated with `model` (default `IMRPhenomXHM`,
not the production SEOBNR model -- this needs to run once, cheaply, before any expensive-model
walker exists). Thin wrapper around [`NUTSCore._hessian_metric`](@ref), the SAME finite-difference
machinery `NUTSConfig(hessian_init=true)` already uses per-walker at mid-warmup windows, just
called ONCE at a single point instead of per-walker-per-window.

Returns `(metric::Union{Nothing,DenseMetric{T}}, ref_point::Vector{T})`. `metric === nothing`
means the Hessian was not usable at the reference point (non-finite gradient, not positive
definite after the eigen-floor -- see `_hessian_metric`'s own docstring); callers should fall back
to an uninformed start rather than error, since this is a preconditioner/convenience, not a
correctness requirement.

Single-detector only (`data::LikelihoodData`); multi-detector reference metrics are not yet
wired.
"""
function analytic_reference_metric(data::LikelihoodData, freqs::AbstractVector{T};
                                    model=IMRPhenomXHM(), nrows::Int=11,
                                    hessian_rel_step::Real=1e-6) where {T}
    ref_point = zeros(T, nrows, 1)
    _fill_gw150914_base_point!(ref_point)
    bounds = _sampled_space_bounds(T, nrows)
    cfg = NUTSConfig{T}(; bounds=bounds, hessian_rel_step=T(hessian_rel_step))
    logp_and_grad! = gw_logp_and_grad(data, _reference_pe_config(T, freqs), freqs; model=model)
    metric, _nvalid = _hessian_metric(logp_and_grad!, ref_point, cfg)
    return metric, vec(ref_point)
end

# `gw_logp_and_grad` takes a `PEConfig` for `dt`/`sample_rate`/`ntime`/`nsteps`/`device` metadata
# only -- the actual likelihood evaluation reads `size(positions, 2)` for the walker count and
# `freqs`/`data` for everything physical (same "cfg.ntime is not actually consulted" convention
# `run_gw150914_estimation` already relies on) -- so a single-walker-shaped `PEConfig` is exactly
# as valid here as the real one built for an actual sampling run.
_reference_pe_config(::Type{T}, freqs::AbstractVector{T}) where {T} =
    PEConfig{T}(length(freqs), 1, T(1e-3), 1)

"""
    informed_initial_positions(::Type{T}, nwalkers, metric, ref_point; rng, nrows=11, scale=1.0)

Disperse `nwalkers` starting positions around `ref_point` along `metric`'s actual correlated
directions (`ref_point .+ scale .* metric.L * randn(nparams)`, i.e. `scale` standard deviations of
the ESTIMATED posterior shape in a random direction) instead of `_gw150914_initial_positions`'s
independent per-parameter jitter. Motivated by a real-data
finding: independent per-parameter dispersion is blind to the mc/t_c/phase ridge and can land a
walker in a secondary mode that a correlation-aware draw would have avoided by staying on-ridge.

Periodic rows (phase/ra/psi) wrap via `mod`, box rows clamp into `[lo, hi]`, exactly mirroring
`_gw150914_initial_positions`'s own post-processing -- a correlated draw can push a coordinate out
of its physical bounds just as easily as independent jitter can.

`metric === nothing` (the reference Hessian was unusable, see `analytic_reference_metric`) falls
back to `_gw150914_initial_positions`'s plain independent-jitter dispersion at `dispersion=0.02`,
rather than erroring -- an informed start is a convenience, not a correctness requirement.
"""
function informed_initial_positions(::Type{T}, nwalkers::Int, metric::Union{Nothing,DenseMetric{T}},
                                     ref_point::AbstractVector{T}; rng::Random.AbstractRNG,
                                     nrows::Int=11, scale::Real=1.0) where {T}
    nparams = length(ref_point)
    bounds = _sampled_space_bounds(T, nrows)
    if metric === nothing
        positions = Matrix{T}(undef, nrows, nwalkers)
        for w in 1:nwalkers
            positions[:, w] .= ref_point
        end
        return _disperse_independent!(positions, bounds, rng, T(0.02))
    end
    draws = metric.L * randn(rng, T, nparams, nwalkers)
    positions = ref_point .+ T(scale) .* draws
    return _wrap_and_clamp!(positions, bounds)
end

function _disperse_independent!(positions::Matrix{T}, bounds, rng, dispersion::T) where {T}
    nrows, nwalkers = size(positions)
    for i in 1:nrows
        lo, hi = bounds.lo[i], bounds.hi[i]
        scale = dispersion * (hi - lo)
        for w in 1:nwalkers
            positions[i, w] += scale * randn(rng, T)
        end
    end
    return _wrap_and_clamp!(positions, bounds)
end

function _wrap_and_clamp!(positions::Matrix{T}, bounds) where {T}
    nrows, nwalkers = size(positions)
    for i in 1:nrows
        lo, hi = bounds.lo[i], bounds.hi[i]
        span = hi - lo
        for w in 1:nwalkers
            x = positions[i, w]
            positions[i, w] = bounds.periodic[i] ? (mod(x - lo, span) + lo) : clamp(x, lo, hi)
        end
    end
    return positions
end

"""
    analytic_snr2(data::LikelihoodData, freqs, ref_point; model=IMRPhenomXHM()) -> snr2

Optimal (noise-free) matched-filter SNR^2 of `model` at `ref_point` against `data`'s PSD:
`snr2 = <h,h> = 4 * Re(sum(|h(f)|^2 / psd(f))) * df`. Computed WITHOUT any model-specific
amplitude/phase machinery (several models, including `IMRPhenomXHM`, have no single-valued
`Ampl`/`Phi` to call directly -- see `RelativeBinning.jl`'s own `_bin_placement_model` comment for
the same issue) by reusing the generic likelihood evaluation against a COPY of `data` with `strain`
zeroed: with `dd=0` and `dh=0`, `logL(zero_strain_data, ref) = -0.5*hh`, so `hh = -2*logL`. Works
for any model `gw_logp_and_grad` already dispatches on.
"""
function analytic_snr2(data::LikelihoodData, freqs::AbstractVector{T}, ref_point::AbstractVector{T};
                        model=IMRPhenomXHM()) where {T}
    zero_data = LikelihoodData(data.freqs, zero(data.strain), data.psd, data.detector, data.gps_time)
    logp_and_grad! = gw_logp_and_grad(zero_data, _reference_pe_config(T, freqs), freqs; model=model)
    grad = similar(ref_point, length(ref_point), 1)
    logp = Vector{T}(undef, 1)
    logp_and_grad!(grad, logp, reshape(ref_point, :, 1), trues(1))
    # `logp` includes the (in-bounds, so exactly 0) log-prior term added inside `gw_logp_and_grad`
    # -- `ref_point` is `_fill_gw150914_base_point!`'s own point, always inside
    # `_sampled_space_bounds`, so no correction is needed; not asserted here to keep this a cheap
    # pure computation, but see this file's tests for a direct check that it holds.
    return -2 * logp[1]
end

"""
    analytic_beta_min(snr2, nparams; floor=1e-3) -> beta_min

First-pass heuristic PT ladder floor: the tempered log-likelihood curvature at inverse temperature
`beta` scales as `beta * snr2`, so `beta_min = nparams / snr2` puts the HOTTEST rung's curvature at
roughly `O(nparams)` -- flat enough that it explores close to the prior (the standard PT design
goal for the hottest chain) without being so flat it wastes the whole ladder on rungs indistinguishable
from the prior. Clamped to `[floor, 1]`. This is a heuristic starting point, not a validated
optimum -- unlike `analytic_reference_metric`/`informed_initial_positions` (directly justified by
real-data dense-metric runs), this needs real-data PT swap-acceptance validation
before being trusted on a production ladder.
"""
function analytic_beta_min(snr2::Real, nparams::Integer; floor::Real=1e-3)
    snr2 > 0 || throw(ArgumentError("analytic_beta_min requires snr2 > 0, got $snr2"))
    return clamp(nparams / snr2, floor, 1.0)
end

end
