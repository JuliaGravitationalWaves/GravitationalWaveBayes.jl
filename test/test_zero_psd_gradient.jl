using Test
using Random
using GravitationalWaveBayes
using GravitationalWaveforms: IMRPhenomXAS, IMRPhenomD
using GravitationalWaveData: H1, L1

const BNz = GravitationalWaveBayes.BatchedNUTS

# A PSD bin that is exactly 0 carries no information, like an infinite-PSD bin. Before the guarded-denominator fix
# the logp was right but the Enzyme gradient was NaN: `_safe_div`'s ifelse sent a zero cotangent through
# `x / 0`. The gradient with a zero-PSD bin must now be finite and equal the gradient with that bin's PSD = Inf.

@testset "zero-PSD bin: finite gradient, same as an infinite-PSD bin" begin
    freqs = collect(20.0:0.5:300.0)
    gps = 1126259462.4
    function det_data(det, seed, badpsd)
        psd = fill(1e-46, length(freqs))
        psd[17] = badpsd
        strain = [1e-22 * complex(sin(1.3k + seed), cos(0.7k * seed)) for k in eachindex(freqs)]
        LikelihoodData(freqs, strain, psd, det, gps)
    end
    nw = 3
    pos = [30.0, 0.8, 0.1, -0.1, 0.5, 0.01, 1.0, 1.9, -0.9, -0.6, 1.0] .+
          [0.3, 0.05, 0.1, 0.1, 0.05, 0.002, 0.3, 0.1, 0.05, 0.1, 0.3] .* randn(Random.Xoshiro(5), 11, nw)
    opts = (; marginalize_phase=true, marginalize_distance=true, distance_lo=0.05, distance_hi=2.0, distance_power=2,
            time_ref_frequency=35.0, time_ref_offset=0.3, time_reference=1)
    for (model, o, label) in ((IMRPhenomXAS(), opts, "IMRPhenomXAS, all options"), (IMRPhenomD(), (;), "IMRPhenomD, plain"))
        cfg = with_likelihood_options(PEConfig{Float64}(length(freqs), nw, 1e-3, 1); o...)
        res = map((0.0, Inf)) do bad
            data = [det_data(H1, 1, bad), det_data(L1, 2, bad)]
            g = zeros(11, nw); l = zeros(nw)
            BNz.gw_logp_and_grad(data, cfg, freqs; model=model)(g, l, pos, nothing)
            (g, l)
        end
        @testset "$label" begin
            @test all(isfinite, res[1][2])
            @test all(isfinite, res[1][1])
            @test res[1][2] == res[2][2]
            @test isapprox(res[1][1], res[2][1]; rtol=1e-12, atol=0)
        end
    end
end
