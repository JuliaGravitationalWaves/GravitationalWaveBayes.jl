using Test
using GravitationalWaveBayes
using GravitationalWaveforms: IMRPhenomXAS, IMRPhenomD, SEOBNRv5IMRAlignedSpinF2
using GravitationalWaveData: H1, L1

# Phase / time sign conventions against REAL data (KernelFused `_PHASE_CONVENTION`):
# injections are self-consistent whatever the convention, so only real data can check it. GW150914
# reached L1 ~6.9 ms before H1; with the physical time convention each detector's best single-detector
# `t_c` must show that ordering, and every model family must fit the signal (large likelihood gain;
# a conjugated waveform gains little). Skipped when the GWOSC files are not present.

const _RDC_H1 = joinpath(@__DIR__, "..", "data", "H-H1_LOSC_4_V1-1126256640-4096.hdf5")
const _RDC_L1 = joinpath(@__DIR__, "..", "data", "L-L1_LOSC_4_V1-1126256640-4096.hdf5")

function _rdc_best_tc(model, data, tcs)
    fs = data.freqs
    phases = collect(range(0, 2pi; length = 17))[1:16]
    nw = length(tcs) * length(phases)
    P = zeros(11, nw)
    k = 0
    for tc in tcs, ph in phases
        k += 1
        P[:, k] .= (39.0, 32.0, 0.0, 0.0, 0.45, tc, ph, 1.95, -1.27, 2.7, 0.8)
    end
    cfg = PEConfig{Float64}(length(fs), nw, 1e-3, 1)
    ll = zeros(nw)
    GravitationalWaveBayes.BatchedNUTS._eval_ll!(ll, P, fs, cfg, data, model, nothing)
    best = vec(maximum(reshape(ll, length(phases), length(tcs)); dims = 1))
    i = argmax(best)
    return tcs[i], best[i] - max(best[1], best[end])
end

if isfile(_RDC_H1) && isfile(_RDC_L1)
    @testset "real GW150914: arrival order (L1 before H1) and signal found, every model family" begin
        kw = (; trigger_time = 1126259462.4, duration = 4.0, post_trigger = 2.0, psd_segment = 1024.0,
              f_min = 20.0, f_max = 512.0, alpha = 0.2, psd_method = :median)
        h1 = build_gwdata(_RDC_H1; detector = H1, kw...)
        l1 = build_gwdata(_RDC_L1; detector = L1, kw...)
        tcs = collect(-0.01:0.0002:0.045)
        for model in (IMRPhenomXAS(), IMRPhenomD(), SEOBNRv5IMRAlignedSpinF2())
            th, gh = _rdc_best_tc(model, h1, tcs)
            tl, gl = _rdc_best_tc(model, l1, tcs)
            @test isapprox(th - tl, 0.0069; atol = 0.0012)     # H1 - L1 arrival difference
            @test gh > 250 && gl > 100                         # SNR ~19.5 (H1), ~13.5 (L1)
        end
    end
else
    @testset "real GW150914 conventions" begin
        @test_skip "GWOSC GW150914 strain files not present under data/"
    end
end
