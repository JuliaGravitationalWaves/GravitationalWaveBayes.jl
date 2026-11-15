module Types

export PEConfig, with_likelihood_options, PEState, LikelihoodData, DetectorData, PAF2Model, IMRTableModel

using GravitationalWaveData: Detector, H1
using GravitationalWaveforms: SEOBNRv5AlignedSpinF2, SEOBNRv5IMRAlignedSpinF2

# Models whose waveform is a post-adiabatic table + F2 reconstruction (`GravitationalWaveforms._seobv5f2_coeffs`
# builds the table for either). They share every SEOBNR kernel and gradient path in this package;
# `SEOBNRv5IMRAlignedSpinF2` adds the plunge and the SEOBNRv5 merger-ringdown (its table type,
# `GravitationalWaveforms.EOBIMRTable`, has its own amplitude/phase methods; its gradient is a per-walker Enzyme
# call, see `BatchedNUTS._reverse_mode_gradient!`, not the SoA hand-adjoint path).
# IMR models whose table is a GravitationalWaveforms.EOBIMRTable (SEOBNRv5IMRAlignedSpinF2; the
# alias is the dispatch name of the IMR-table paths).
const IMRTableModel = SEOBNRv5IMRAlignedSpinF2
const PAF2Model = Union{SEOBNRv5AlignedSpinF2, IMRTableModel}

"""
    PEConfig{T}(ntime, nwalkers, step_size, nsteps; likelihood options...)

Run configuration for batched parameter estimation: `ntime` (the frequency-domain grid length
every likelihood/kernel call asserts against, e.g. `KernelFused.jl`'s fused kernels), `nwalkers`
(batch size), `step_size`/`nsteps` (the sampler's initial leapfrog step and draw count -- see
`NUTSCore.jl`'s `NUTSConfig` for the full dual-averaging/adaptation config `batched_nuts!` seeds
with them).

The keyword fields select likelihood variants (all default to the plain likelihood):

- `marginalize_phase::Bool` (default `false`): analytically marginalize the reference phase (row 7)
  under its uniform `[0, 2pi)` prior, `logL = -1/2(<h,h> + <d,d>) + log I0(|<d,h>|)`. Exact for
  waveforms whose phase enters only as a common `exp(-i*phase)` factor (the dominant-mode-only
  `PAF2Model`s). The sampled phase row then has zero likelihood gradient and is drawn from its
  prior; `BatchedNUTS.reconstruct_marginalized!` redraws it from its conditional posterior afterwards.
- `time_reference::Int` (default `0`): multi-detector only. `0` means the sampled `t_c` is the
  geocenter arrival time; `k >= 1` means it is the arrival time at detector `data[k]` (bilby's
  `time_reference`), which decouples `t_c` from `ra`/`dec`. Single-detector likelihoods already
  treat `t_c` as that detector's arrival time and ignore this field.
- `marginalize_distance::Bool` (default `false`), `distance_lo`/`distance_hi` (Gpc, default
  `0.05`/`4.0`), `distance_power::Int` (default `0`): analytically marginalize the luminosity distance
  (row 5) under the prior `p(dL) ∝ dL^distance_power` on `[distance_lo, distance_hi]` (`2` = uniform
  in Euclidean volume). Uses `<h,h> ∝ dL^-2`, `<d,h> ∝ dL^-1` and a per-walker Gauss-Legendre
  quadrature (see `KernelFused._distance_marg`). Combines with `marginalize_phase`. The sampled dL row
  then only carries prior draws; `reconstruct_marginalized!` redraws it afterwards.
- `time_ref_frequency` (Hz, default `0` = off), `time_ref_offset` (s, default `0`): PAF2 models only.
  When on, the sampled time row is `t_ref = t_c + t*(theta; f_ref) + time_ref_offset`, where `t*` is the
  model's own stationary-phase time at which the GW frequency passes `f_ref` (for the PAF2 tables,
  `dPsi/df = 2 pi t*` exactly). An in-band arrival time is what the data measure; the coalescence time
  is an extrapolation past the band that slides along a long curved ridge with the masses and spins.
  The offset is a constant that just re-centres the prior window (e.g. `-t*` at a reference point).
- `fmax_taper` (Hz, default `0` = off): PAF2 models only. Multiplies the amplitude by a C1 `sin^2` taper
  over the last `fmax_taper` Hz below the table's own end frequency `fmax`. Without it the signal is
  cut off exactly at `fmax`, which moves with the masses/spins, so every time `fmax` crosses a
  frequency bin the log-likelihood JUMPS (measured up to ~1 on an SNR-13 injection) -- step-size
  independent energy errors that pin NUTS at tiny step sizes. Injections must use the same taper.
- `phase_ref_frequency` (Hz, default `0` = the data's lowest frequency): SEOBNRv5PHM, and IMRPhenomXHM / SEOBNRv5HM
  on the mode-sum device path (XHM: its phase reference frequency `f_ref`; SEOBNRv5HM: the modes rotated by the (2,2)
  orbital phase at that frequency); the sampled phase is then the phase there. The waveform's
  reference frequency -- where the reference phase, the spin components and the inclination frame are defined
  (pyseobnr's `f_ref`) -- separately from its start (0.9 x the lowest data frequency). A phase defined at a
  low frequency is coupled to the masses and spins by the whole inspiral phase up to the sensitive band (many
  wraps across a population: the dominant thin direction of the posterior); defined near the band it is not.
  Capped at M omega = 0.06 (below the end of any inspiral in the prior).
"""
struct PEConfig{T}
    ntime::Int
    nwalkers::Int
    step_size::T
    nsteps::Int
    marginalize_phase::Bool
    time_reference::Int
    marginalize_distance::Bool
    distance_lo::T
    distance_hi::T
    distance_power::Int
    time_ref_frequency::T
    time_ref_offset::T
    fmax_taper::T
    phase_ref_frequency::T
    function PEConfig{T}(ntime::Integer, nwalkers::Integer, step_size, nsteps::Integer;
                         marginalize_phase::Bool=false, time_reference::Integer=0,
                         marginalize_distance::Bool=false, distance_lo=0.05, distance_hi=4.0,
                         distance_power::Integer=0, time_ref_frequency=0.0, time_ref_offset=0.0,
                         fmax_taper=0.0, phase_ref_frequency=0.0) where {T}
        fmax_taper >= 0 || throw(ArgumentError("fmax_taper must be >= 0"))
        time_reference >= 0 || throw(ArgumentError("time_reference must be >= 0, got $time_reference"))
        0 < distance_lo < distance_hi || throw(ArgumentError("need 0 < distance_lo < distance_hi"))
        distance_power >= 0 || throw(ArgumentError("distance_power must be >= 0"))
        phase_ref_frequency >= 0 || throw(ArgumentError("phase_ref_frequency must be >= 0"))
        return new{T}(ntime, nwalkers, T(step_size), nsteps,
                      marginalize_phase, Int(time_reference), marginalize_distance, T(distance_lo),
                      T(distance_hi), Int(distance_power), T(time_ref_frequency), T(time_ref_offset), T(fmax_taper),
                      T(phase_ref_frequency))
    end
end

"""
    with_likelihood_options(cfg; likelihood options...)

Copy of `cfg` with the given likelihood-variant keyword fields (see [`PEConfig`](@ref)) replaced, the others kept.
"""
with_likelihood_options(cfg::PEConfig{T}; marginalize_phase::Bool=cfg.marginalize_phase,
                        time_reference::Integer=cfg.time_reference,
                        marginalize_distance::Bool=cfg.marginalize_distance,
                        distance_lo=cfg.distance_lo, distance_hi=cfg.distance_hi,
                        distance_power::Integer=cfg.distance_power,
                        time_ref_frequency=cfg.time_ref_frequency, time_ref_offset=cfg.time_ref_offset,
                        fmax_taper=cfg.fmax_taper, phase_ref_frequency=cfg.phase_ref_frequency) where {T} =
    PEConfig{T}(cfg.ntime, cfg.nwalkers, cfg.step_size, cfg.nsteps; marginalize_phase, time_reference,
                marginalize_distance, distance_lo, distance_hi, distance_power, time_ref_frequency, time_ref_offset,
                fmax_taper, phase_ref_frequency)

"""
    PEState{T,A,V}

Sampler state for batched parameter estimation. `positions`/`momenta` are 2-D
(`nparams x nwalkers`) device arrays of type `A`; `logp`/`energy` are 1-D arrays of type `V`.

`A` and `V` are independent type parameters (rather than `V` being derived from `A`, e.g. via
`Vector{T}`) specifically so that `logp`/`energy` can live on the same device as
`positions`/`momenta` — e.g. `A = CuMatrix{T}` paired with `V = CuVector{T}` — instead of being
hardcoded to host `Vector{T}`, which would force a host round-trip on every access. Construct
with plain `PEState(positions, momenta, logp, energy)` and let the type parameters be inferred.
"""
struct PEState{T,A<:AbstractArray{T,2},V<:AbstractVector{T}}
    positions::A
    momenta::A
    logp::V
    energy::V
end

"""
    LikelihoodData{F,S,P}

Single-detector frequency-domain data bundle: `freqs`/`strain`/`psd` (see
`KernelFused.batched_log_likelihood_kernel!`'s docstring for the likelihood convention they feed), plus `detector` (a
`GravitationalWaveData.Detector`, e.g. `GravitationalWaveData.H1`, for the antenna response and time delay) and `gps_time` (the FIXED event trigger
time, GPS seconds -- NOT a sampled parameter; only the small offset `t_c` sampled as physical-params row 6 is a free
parameter, see `KernelFused.jl`'s module docstring for the full row convention). Multi-detector data is a vector of
these (see [`DetectorData`](@ref)).

The 3-argument constructor `LikelihoodData(freqs, strain, psd)` (for synthetic data where the sky position doesn't
matter) defaults to `GravitationalWaveData.H1` and GW150914's trigger time (`1126259462.4` GPS seconds).
"""
struct LikelihoodData{F<:AbstractVector, S<:AbstractVector, P<:AbstractVector}
    freqs::F
    strain::S
    psd::P
    detector::Detector
    gps_time::Float64
end

LikelihoodData(freqs::AbstractVector, strain::AbstractVector, psd::AbstractVector) =
    LikelihoodData(freqs, strain, psd, H1, 1126259462.4)

"""
    DetectorData

Type alias for what this package's likelihood/kernel entry points accept as detector data: either
a single [`LikelihoodData`](@ref) (single-detector) or
`AbstractVector{<:LikelihoodData}` (multi-detector -- one entry per detector,
e.g. `[h1_data, l1_data]` for H1+L1).

**Design choice:** multi-detector data is represented as a plain
`Vector{LikelihoodData}` rather than a new wrapping struct. Reasoning:
- `LikelihoodData` already carries `gps_time` per entry; every multi-detector
  entry point (`KernelFused.batched_log_likelihood_kernel!`, `RelativeBinning.build_relative_binning`/
  `binned_log_likelihood_kernel!`, `DataIO.build_gwdata_multi`) asserts all entries share the same
  `gps_time` (same physical event) rather than silently assuming it.
- `freqs`/`strain`/`psd` may genuinely differ in length/grid per detector (independent data
  conditioning per detector is plausible even if today's real-data pipeline happens to produce
  matching grids for H1/L1) -- there is no shared-shape invariant to hoist into a new type.
- The one field that IS shared (`gps_time`, a single `Float64`) is cheap enough to duplicate per
  entry, and duplicating it keeps every `LikelihoodData` self-contained/independently constructible
  (e.g. by the existing, unchanged `DataIO.build_gwdata`) rather than requiring a two-step
  construction (build N `LikelihoodData`s, then wrap). A new `struct MultiLikelihoodData` bundling
  `gps_time` once was considered and rejected for this reason.
"""
const DetectorData = Union{LikelihoodData, AbstractVector{<:LikelihoodData}}

end
