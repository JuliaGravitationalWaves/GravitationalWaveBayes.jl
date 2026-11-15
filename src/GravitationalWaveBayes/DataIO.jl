module DataIO

export load_gwosc_strain, condition_segment, to_frequency_domain, build_gwdata, build_gwdata_multi, synthetic_gwdata

using FFTW
using GravitationalWaveData: GravitationalWaveData, load_gwosc_strain, condition_segment, psd, TimeSeries, FrequencySeries, Detector, H1
using ..Types: LikelihoodData

# ================================================================================================
# GWOSC/LOSC HDF5 loading, trigger-relative segmenting+windowing, and Welch/median PSD estimation
# now live in `GravitationalWaveData.jl` (`GravitationalWaveData/IO/GWOSC.jl`, `GravitationalWaveData/PSD.jl`) -- confirmed direct ports of what
# used to be hand-rolled here (same rounding/segmenting conventions, same median-Welch/Hann-window
# PSD convention, `DSP.Windows.hanning`/`DSP.Windows.tukey` verified to match the old hand-rolled
# windows to machine precision, see `GravitationalWaveData.jl`'s `PSD.jl`/`GWOSC.jl` docstrings). `load_gwosc_strain`
# and `condition_segment` are imported directly from `GravitationalWaveData` above and simply re-exported here --
# not redefined, so this is not a naming collision, just a pass-through of the same function objects
# (return types changed from bare `NamedTuple`/`Vector` to `GravitationalWaveData.TimeSeries`, which is the one
# externally-visible signature change from this migration; see `test/test_dataio.jl`).
#
# What stays local: `to_frequency_domain` (a single-shot windowed-segment FFT, not a Welch PSD
# estimate -- `GravitationalWaveData.jl` has no equivalent) and `build_gwdata`, the GravitationalWaveBayes-specific glue that
# assembles a `LikelihoodData` from the pieces above.
# ================================================================================================

"""
    to_frequency_domain(segment, dt) -> (freqs, strain_fd)

One-sided FFT of a (already windowed, if desired) time-domain segment, scaled by `dt` so that
`strain_fd` approximates the continuous Fourier transform (`strain_fd(f) ≈ dt * DFT[segment]`) --
the standard convention pairing with `GravitationalWaveData.psd`'s density-normalized PSD in a matched-filter
likelihood (see `build_gwdata`'s docstring for how the two combine).
"""
function to_frequency_domain(segment::AbstractVector{T}, dt::Real) where {T<:Real}
    n = length(segment)
    strain_fd = FFTW.rfft(segment) .* dt
    freqs = FFTW.rfftfreq(n, 1 / dt)
    return freqs, strain_fd
end

"""
    build_gwdata(strain_path; trigger_time, duration=4.0, post_trigger=2.0,
                 psd_segment=1024.0, f_min=20.0, f_max=1024.0, alpha=0.2, psd_method=:median,
                 window_power_correction=false) -> LikelihoodData

Glue `GravitationalWaveData.load_gwosc_strain`/`GravitationalWaveData.condition_segment`/`GravitationalWaveData.psd`/`to_frequency_domain` into
a `LikelihoodData`, following Jim's tutorial config as the default (4s analysis window / 2s
post-trigger, 1024s off-source PSD segment immediately before the analysis window, 20-1024 Hz
band).

`data.psd` is the one-sided PSD in the convention `gwpy`/bilby/LALInference use -- `GravitationalWaveData.psd`'s off-source
estimate, or `psd_file` (two columns, e.g. a PE-release PSD; see [`read_psd_file`](@ref)) when given. As in
bilby >= 2.6.0 it is NOT multiplied by the Tukey window factor mean(w^2): the signal sits in the flat part of the
window, where the noise is not reduced (bilby PR #967, issue #869). `window_power_correction=true` restores the
bilby <= 2.5 convention (PSD x mean(w^2), see [`window_factor`](@ref); every GWTC-2.1 / GWTC-3 bilby release), like
bilby's `BILBY_INCORRECT_PSD_NORMALIZATION=TRUE`; SNRs are then 1/sqrt(mean(w^2)) higher. `KernelFused.jl`/`RelativeBinning.jl` apply the standard `4 * df`
matched-filter normalization themselves (see `KernelFused.batched_log_likelihood_kernel!`'s docstring);
`data.psd` does not need to (and must not) have that folded in here.

`detector` (default `GravitationalWaveData.H1`) and `gps_time=trigger_time` are threaded
straight into the returned `LikelihoodData` for the single-detector antenna-response projection
`KernelFused.jl`'s kernels now perform -- see `LikelihoodData`'s docstring in Types.jl. Explicit
here (not relying on `LikelihoodData`'s own 3-arg-constructor default, which happens to also be
`H1`/GW150914's trigger time) so this stays correct for a future non-GW150914/non-H1 call.
"""
function build_gwdata(strain_path::AbstractString; trigger_time::Real,
                       duration::Real=4.0, post_trigger::Real=2.0, psd_segment::Real=1024.0,
                       f_min::Real=20.0, f_max::Real=1024.0, alpha::Real=0.2,
                       psd_method::Symbol=:median, detector::Detector=H1,
                       psd_file::Union{Nothing,AbstractString}=nothing, window_power_correction::Bool=false)
    strain_ts, _dq_mask = load_gwosc_strain(strain_path)
    dt = strain_ts.dt

    segment_ts = condition_segment(strain_ts; trigger_time=trigger_time, duration=duration,
                                    post_trigger=post_trigger, alpha=alpha)
    freqs, strain_fd = to_frequency_domain(segment_ts.data, dt)

    # Phase-reference correction: `to_frequency_domain`'s
    # plain `FFTW.rfft` implicitly treats the FIRST SAMPLE of `segment_ts` (i.e. `segment_ts.t0`,
    # which `condition_segment` sets to `trigger_time - (duration - post_trigger)`, sample-rounded)
    # as its own t'=0 -- NOT `trigger_time`. But `LikelihoodData.gps_time` (`trigger_time`) is the
    # FIXED reference every likelihood kernel's `t_c` is defined relative to (`t_c` = "coalescence
    # time OFFSET FROM `gps_time`", see `KernelFused.jl`'s module docstring / `Types.LikelihoodData`'s
    # docstring) -- i.e. the templates assume `strain_fd`'s phase is ALSO referenced to
    # `trigger_time`, not to the segment's own start. Left uncorrected, this is a phase error of
    # `exp(i*2*pi*f*(segment_ts.t0 - trigger_time))` on every frequency bin -- for GW150914's
    # defaults (`duration=4, post_trigger=2`, a 2s pre-trigger window) this is a 2-second phase
    # ramp; for a GW190412 window (`duration=8, post_trigger=2`, a 6s pre-trigger window)
    # it's 6 seconds -- both large enough, across a 20-1024 Hz band, to essentially decorrelate any
    # trial waveform from the real data (confirmed empirically: an FFT-based matched-filter time-
    # shift search against real GW190412 H1+L1 data found the correlation peak at almost EXACTLY
    # `trigger_time - segment_ts.t0` -- mod the analysis `duration`, since the correlation is
    # periodic with period `duration` -- before this fix, and at approximately zero offset (within
    # ~0.03s, attributable to the search's aligned-spin-proxy/MAP-point approximation of the real
    # precessing signal, not a remaining phase bug) after it). Multiplying `strain_fd` by
    # `cis(2*pi*f*(trigger_time - segment_ts.t0))` re-references its phase to `trigger_time`,
    # matching every likelihood kernel's own convention -- this is exactly the standard
    # `X'(f) = X(f) * exp(i*2*pi*f*Delta_t)` time-shift-by-`Delta_t` identity for a DFT computed as
    # `X(f) = sum_n x[n] exp(-i*2*pi*f*n*dt)` (FFTW's convention), with `Delta_t = trigger_time -
    # segment_ts.t0` the shift needed to move the phase origin from `segment_ts.t0` to
    # `trigger_time`.
    strain_fd = strain_fd .* cis.(2 * pi .* freqs .* (trigger_time - segment_ts.t0))

    band = findall(f -> f_min <= f <= f_max, freqs)
    isempty(band) && throw(ArgumentError("No frequency bins in [$f_min, $f_max] Hz"))

    if psd_file === nothing
        # Off-source stretch for PSD estimation: the psd_segment seconds immediately preceding the
        # analysis window (so it never overlaps the event itself), matching Jim's tutorial approach.
        # `GravitationalWaveData.crop`'s half-open-interval rounding is the same `round(Int, (t - t0) / dt) + 1`
        # convention.
        analysis_start = trigger_time - (duration - post_trigger)
        off_source_ts = GravitationalWaveData.crop(strain_ts, analysis_start - psd_segment, analysis_start)
        nseg = round(Int, duration / dt)  # same length as the analysis segment, per convention
        psd_fs = psd(off_source_ts; seglen=nseg, method=psd_method)
        # Sanity check only (not bit-exact `==`): `GravitationalWaveData.psd`'s frequency grid and `freqs` above are
        # both `FFTW.rfftfreq(nseg, 1/dt)`-equivalent by construction (same seglen/dt), but computed
        # through different intermediate divisions (`GravitationalWaveData.FrequencySeries`'s `f0 + df*k` vs FFTW's
        # own internal formula), which differ at the ~1 ULP level -- not a real discrepancy, just
        # non-associative floating point, so this is an `isapprox` check, not `==`.
        isapprox(collect(GravitationalWaveData.frequencies(psd_fs)), freqs; rtol=0, atol=1e-9) ||
            throw(AssertionError("PSD and analysis-segment frequency grids must match (same seglen/dt by construction)"))
        psd_band = psd_fs.data[band]
    else
        psd_band = read_psd_file(psd_file, freqs[band])
    end

    # bilby <= 2.5 multiplied the PSD by the window's power loss mean(w^2) (0.875 for alpha = 0.2); bilby 2.6.0
    # dropped it (the noise under the signal, in the flat part of the window, is not reduced). Default = bilby >= 2.6.
    window_power_correction &&
        (psd_band = psd_band .* window_factor(dt; duration=duration, post_trigger=post_trigger, alpha=alpha))

    return LikelihoodData(freqs[band], strain_fd[band], psd_band, detector, Float64(trigger_time))
end

"""
    window_factor(dt; duration, post_trigger, alpha) -> mean(w^2)

Mean square of the Tukey window `condition_segment` applies (`DSP.Windows.tukey(n, alpha)`, the same symmetric
window as scipy's `tukey`, which bilby uses for its `window_factor`), read off by conditioning a series of ones.
"""
function window_factor(dt::Real; duration::Real, post_trigger::Real, alpha::Real)
    n = round(Int, duration / dt)
    w = condition_segment(TimeSeries(ones(n), 0.0, dt); trigger_time=duration - post_trigger, duration=duration,
                          post_trigger=post_trigger, alpha=alpha).data
    length(w) == n || throw(AssertionError("window length $(length(w)) != $n"))
    return sum(abs2, w) / n
end

"""
    read_psd_file(path, freqs) -> Vector{Float64}

A two-column text PSD (frequency [Hz], one-sided PSD [1/Hz]; `#` comment lines allowed), e.g. a PSD from an LVK
PE release, linearly interpolated onto `freqs` (exact when the grids coincide, as for a release PSD with the
analysis segment's df). Every requested frequency must lie inside the file's range.
"""
function read_psd_file(path::AbstractString, freqs::AbstractVector{<:Real})
    rows = [parse.(Float64, split(l)) for l in eachline(path) if !isempty(strip(l)) && !startswith(lstrip(l), "#")]
    f = getindex.(rows, 1); S = getindex.(rows, 2)
    issorted(f) || throw(ArgumentError("PSD file $path: frequencies must be increasing"))
    (first(freqs) >= f[1] && last(freqs) <= f[end]) ||
        throw(ArgumentError("PSD file $path covers $(f[1])-$(f[end]) Hz, band needs $(first(freqs))-$(last(freqs)) Hz"))
    out = similar(f, length(freqs))
    for (i, x) in enumerate(freqs)
        j = clamp(searchsortedlast(f, x), 1, length(f) - 1)
        w = (x - f[j]) / (f[j+1] - f[j])
        out[i] = (1 - w) * S[j] + w * S[j+1]
    end
    all(>(0), out) || throw(ArgumentError("PSD file $path: non-positive PSD in the band"))
    return out
end

"""
    synthetic_gwdata(; detector, trigger_time, duration=4.0, f_min=20.0, f_max=1024.0, psd_file=nothing,
                     asd_file=nothing) -> LikelihoodData

A detector with no data: zero strain on the analysis grid `f = k / duration` within `[f_min, f_max]`, and the PSD of a
two-column noise-curve file (`psd_file`: one-sided PSD; `asd_file`: amplitude spectral density, squared), e.g. an
LVK observing-scenarios curve. For simulated signals in Gaussian noise drawn from that PSD (`simulate_injection`);
the noise is simulated in the frequency domain, so no window factor applies.
"""
function synthetic_gwdata(; detector, trigger_time::Real, duration::Real=4.0, f_min::Real=20.0, f_max::Real=1024.0,
                          psd_file::Union{Nothing,AbstractString}=nothing, asd_file::Union{Nothing,AbstractString}=nothing)
    (psd_file === nothing) == (asd_file === nothing) && throw(ArgumentError("give exactly one of psd_file, asd_file"))
    df = 1 / duration
    freqs = [k * df for k in ceil(Int, f_min / df - 1e-9):floor(Int, f_max / df + 1e-9)]
    S = psd_file === nothing ? read_psd_file(asd_file, freqs) .^ 2 : read_psd_file(psd_file, freqs)
    return LikelihoodData(freqs, zeros(ComplexF64, length(freqs)), S, detector, Float64(trigger_time))
end

"""
    build_gwdata_multi(paths::AbstractVector{<:AbstractString}, dets::AbstractVector{Detector};
                        trigger_time, kwargs...) -> Vector{LikelihoodData}

Multi-detector entry point: calls [`build_gwdata`](@ref) once per `(path,
detector)` pair (e.g. `build_gwdata_multi(["H-H1_....hdf5", "L-L1_....hdf5"], [GravitationalWaveData.H1,
GravitationalWaveData.L1]; trigger_time=1126259462.4)`), sharing `trigger_time` and every other keyword argument
across all detectors -- the same physical event, so every returned `LikelihoodData` entry carries
the identical `gps_time` (required by `KernelFused.jl`/`RelativeBinning.jl`'s multi-detector
methods, which assert this rather than assume it).

`length(paths) == length(dets)` is required (one strain file per detector). Each detector's
`freqs`/`strain`/`psd` are conditioned independently by its own `build_gwdata` call and are NOT
assumed to share a grid across detectors -- see `Types.DetectorData`'s docstring.
"""
function build_gwdata_multi(paths::AbstractVector{<:AbstractString}, dets::AbstractVector{Detector};
                             trigger_time::Real, kwargs...)
    length(paths) == length(dets) ||
        throw(ArgumentError("paths and dets must have the same length (one strain file per detector): got $(length(paths)) paths, $(length(dets)) detectors"))
    return [build_gwdata(paths[i]; trigger_time=trigger_time, detector=dets[i], kwargs...) for i in eachindex(paths)]
end

end
