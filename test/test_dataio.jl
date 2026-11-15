using Test
using GravitationalWaveBayes
using GravitationalWaveData
import GravitationalWaveforms

# Guarded by the real GW150914 HDF5 fixture's presence (see data/README.md) so CI/checkouts
# without the ~250MB gitignored data files don't fail -- they just skip this file's tests.
const _H1_PATH = joinpath(@__DIR__, "..", "data", "H-H1_LOSC_4_V1-1126256640-4096.hdf5")
const _L1_PATH = joinpath(@__DIR__, "..", "data", "L-L1_LOSC_4_V1-1126256640-4096.hdf5")
const _GW150914_TRIGGER = 1126259462.4

if isfile(_H1_PATH)
    @testset "DataIO: GWOSC HDF5 loader against the real GW150914 H1 file" begin
        # `GravitationalWaveBayes.load_gwosc_strain` is `GravitationalWaveData.load_gwosc_strain` re-exported:
        # returns (strain::TimeSeries, dq_mask::TimeSeries) rather than the old flat NamedTuple.
        strain, dq_mask = GravitationalWaveBayes.load_gwosc_strain(_H1_PATH)

        @test length(strain) == 16_777_216
        @test strain.dt ≈ 0.000244140625
        @test strain.t0 == 1126256640
        @test length(GravitationalWaveData.times(strain)) == length(strain)
        @test GravitationalWaveData.times(strain)[1] == strain.t0
        @test GravitationalWaveData.times(strain)[end] ≈ strain.t0 + (length(strain) - 1) * strain.dt
        @test length(dq_mask) == 4096
        @test all(isfinite, strain.data)
    end

    @testset "DataIO: condition_segment" begin
        strain, _ = GravitationalWaveBayes.load_gwosc_strain(_H1_PATH)
        segment = GravitationalWaveBayes.condition_segment(strain; trigger_time=_GW150914_TRIGGER,
                                             duration=4.0, post_trigger=2.0)
        @test length(segment) == round(Int, 4.0 / strain.dt)
        @test all(isfinite, segment.data)
        # Trigger should fall inside [segment.t0, segment.t0 + duration).
        @test segment.t0 <= _GW150914_TRIGGER < segment.t0 + 4.0
        # Tukey-windowed: first and last samples tapered toward zero, interior untouched shape-wise.
        @test abs(segment.data[1]) < abs(maximum(abs, segment.data))

        @test_throws ArgumentError GravitationalWaveBayes.condition_segment(strain; trigger_time=strain.t0 - 1000.0)
    end

    @testset "DataIO: GravitationalWaveData.psd sanity (no NaN/Inf, roughly flat-ish in-band)" begin
        strain, _ = GravitationalWaveBayes.load_gwosc_strain(_H1_PATH)
        nseg = round(Int, 4.0 / strain.dt)
        # Use the first ~1024s of the file (well away from the trigger, which is near the file's
        # midpoint at GPS 1126259462.4 vs file start 1126256640) as the off-source stretch.
        off_source = strain[1:round(Int, 1024.0 / strain.dt)]
        psd_fs = GravitationalWaveData.psd(off_source; seglen=nseg)

        @test all(isfinite, psd_fs.data)
        @test all(>(0), psd_fs.data)
        freqs = collect(GravitationalWaveData.frequencies(psd_fs))
        band = findall(f -> 20.0 <= f <= 1024.0, freqs)
        @test !isempty(band)
        # LIGO's sensitive band shouldn't vary by many orders of magnitude bin-to-bin (loose sanity
        # check, not a precision check -- precision cross-validation against gwpy is a separate,
        # still-open WSL2 task).
        @test maximum(psd_fs.data[band]) / minimum(psd_fs.data[band]) < 1e8
    end

    @testset "DataIO: to_frequency_domain" begin
        dt = 1 / 4096.0
        n = 4096 * 4
        t = (0:(n - 1)) .* dt
        f0 = 100.0
        segment = sin.(2pi * f0 .* t)
        freqs, strain_fd = GravitationalWaveBayes.to_frequency_domain(segment, dt)

        @test length(freqs) == length(strain_fd)
        @test freqs[1] == 0.0
        peak_idx = argmax(abs.(strain_fd))
        @test freqs[peak_idx] ≈ f0 atol=1 / (n * dt)
    end

    @testset "DataIO: build_gwdata end-to-end on the real H1 file" begin
        data = GravitationalWaveBayes.build_gwdata(_H1_PATH; trigger_time=_GW150914_TRIGGER)

        @test all(isfinite, data.freqs)
        @test all(f -> 20.0 <= f <= 1024.0, data.freqs)
        @test all(isfinite, data.strain)
        @test all(isfinite, data.psd)
        @test all(>(0), data.psd)
        @test length(data.freqs) == length(data.strain) == length(data.psd)
        @test issorted(data.freqs)
        # An earlier `build_gwdata` (hand-rolled Hann/Tukey windows, local Welch PSD) was compared
        # offline against this GravitationalWaveData.jl-backed version on this same file/trigger:
        # freqs bit-identical, strain max relative diff ~9.7e-12, psd max relative diff ~2.3e-12 --
        # both attributable to `DSP.Windows.{hanning,tukey}` replacing the hand-rolled windows.
    end

    @testset "DataIO: PSD window factor off by default (bilby >= 2.6), window_power_correction restores it" begin
        kw = (; trigger_time=_GW150914_TRIGGER, duration=4.0, post_trigger=2.0, alpha=0.2)
        d0 = GravitationalWaveBayes.build_gwdata(_H1_PATH; kw...)
        d1 = GravitationalWaveBayes.build_gwdata(_H1_PATH; kw..., window_power_correction=true)
        wf = GravitationalWaveBayes.DataIO.window_factor(1 / 4096; duration=4.0, post_trigger=2.0, alpha=0.2)
        @test wf ≈ 0.875 rtol = 1e-3
        @test d1.psd ≈ d0.psd .* wf
        @test d1.strain == d0.strain && d1.freqs == d0.freqs
    end
else
    @warn "Skipping test_dataio.jl: real GW150914 fixture not present at $_H1_PATH (see data/README.md)"
end

# Guarded separately (both H1 AND L1 fixtures required) so a checkout with only the H1 file
# (already required above) still runs everything it can -- multi-detector-specific tests
# just self-skip if L1 isn't also present.
if isfile(_H1_PATH) && isfile(_L1_PATH)
    @testset "DataIO: build_gwdata_multi end-to-end on real H1+L1 GW150914 data" begin
        data = GravitationalWaveBayes.build_gwdata_multi([_H1_PATH, _L1_PATH], [GravitationalWaveData.H1, GravitationalWaveData.L1];
                                           trigger_time=_GW150914_TRIGGER)

        @test length(data) == 2
        @test data[1].detector.prefix == "H1"
        @test data[2].detector.prefix == "L1"
        @test data[1].gps_time == data[2].gps_time == _GW150914_TRIGGER
        for d in data
            @test all(isfinite, d.freqs)
            @test all(isfinite, d.strain)
            @test all(isfinite, d.psd)
            @test all(>(0), d.psd)
            @test length(d.freqs) == length(d.strain) == length(d.psd)
        end
        # Not asserted to be identical in general (see `Types.DetectorData`'s docstring), but
        # H1/L1's shared conditioning pipeline/config in this call DOES produce matching grids in
        # practice -- confirmed, not assumed, since the multi-detector kernel's `eachindex(freqs_d)`
        # loop works either way.
        @test length(data[1].freqs) == length(data[2].freqs)

        # Evaluate the multi-detector likelihood at a GW150914-scale physical point and confirm it
        # comes out finite -- an end-to-end real-data smoke test for the whole multi-detector
        # pipeline (build_gwdata_multi -> batched_log_likelihood_kernel! multi-detector method).
        physical = [36.0, 29.0, 0.0, 0.0, 0.41, 0.0, 0.0, 1.95, -0.9, pi / 2, 0.8]
        ntime = length(data[1].freqs)
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)
        ll = zeros(Float64, 1)
        GravitationalWaveBayes.batched_log_likelihood_kernel!(ll, reshape(physical, 11, 1), cfg, data;
                                                model=GravitationalWaveforms.IMRPhenomXAS())
        @test isfinite(ll[1])
    end

    @testset "DataIO: multi-detector IMRPhenomXP real-data smoke test" begin
        # Cheap finite-likelihood smoke test only (no Enzyme, no recovery run). Mirrors the
        # IMRPhenomXAS smoke test directly above, but at a 15-row physical point (nonzero in-plane
        # spins) dispatched through the new multi-detector IMRPhenomXP kernel.
        data = GravitationalWaveBayes.build_gwdata_multi([_H1_PATH, _L1_PATH], [GravitationalWaveData.H1, GravitationalWaveData.L1];
                                           trigger_time=_GW150914_TRIGGER)
        physical = [36.0, 29.0, 0.05, -0.03, 0.41, 0.0, 0.0, 1.95, -0.9, pi / 2, 0.8, 0.1, -0.05, 0.03, 0.07]
        ntime = length(data[1].freqs)
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)
        ll = zeros(Float64, 1)
        GravitationalWaveBayes.batched_log_likelihood_kernel!(ll, reshape(physical, 15, 1), cfg, data;
                                                model=GravitationalWaveforms.IMRPhenomXP())
        @test isfinite(ll[1])
    end

    @testset "DataIO: multi-detector IMRPhenomXPHM real-data smoke test" begin
        # Cheap finite-likelihood smoke test only (no Enzyme, no recovery run), same scope as the
        # IMRPhenomXP smoke test directly above -- mirrors it exactly, just dispatched through the
        # new multi-detector IMRPhenomXPHM kernel (same 15-row physical point/convention, since
        # IMRPhenomXPHM needs exactly the same physical inputs IMRPhenomXP does).
        data = GravitationalWaveBayes.build_gwdata_multi([_H1_PATH, _L1_PATH], [GravitationalWaveData.H1, GravitationalWaveData.L1];
                                           trigger_time=_GW150914_TRIGGER)
        physical = [36.0, 29.0, 0.05, -0.03, 0.41, 0.0, 0.0, 1.95, -0.9, pi / 2, 0.8, 0.1, -0.05, 0.03, 0.07]
        ntime = length(data[1].freqs)
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)
        ll = zeros(Float64, 1)
        GravitationalWaveBayes.batched_log_likelihood_kernel!(ll, reshape(physical, 15, 1), cfg, data;
                                                model=GravitationalWaveforms.IMRPhenomXPHM())
        @test isfinite(ll[1])
    end
elseif isfile(_H1_PATH)
    @warn "Skipping multi-detector test_dataio.jl tests: L1 fixture not present at $_L1_PATH (see data/README.md)"
end
