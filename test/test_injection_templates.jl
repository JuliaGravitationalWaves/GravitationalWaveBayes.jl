using Test
using Random
using GravitationalWaveBayes
using GravitationalWaveBayes: LikelihoodData, PEConfig, with_likelihood_options, simulate_injection
using GravitationalWaveData: H1, L1, gmst
using GravitationalWaveforms: IMRPhenomXP, IMRPhenomXPHM, SEOBNRv5PHM

# Injections for the precessing models are the likelihood's own detector templates (`KernelFused.xprec_strain`,
# `KernelFused.seobnr_phm_device_strain`): a zero-noise injection must give logL = 0 at the truth (the parameters in the
# likelihood's time frame), with the time-reference options on, and logL < 0 a little away from it.
const KFi = GravitationalWaveBayes.KernelFused
_ipsd(f) = (x = f / 215; 1e-49 * (x^(-4.14) - 5 * x^(-2) + 111 * (1 - x^2 + 0.5 * x^4) / (1 + 0.5 * x^2)))
const _IGPS = 1126259462.4
function _itemplates()
    freqs = collect(20.0:0.25:512.0)
    return [LikelihoodData(freqs, zeros(ComplexF64, length(freqs)), _ipsd.(freqs), det, _IGPS) for det in (H1, L1)]
end
# physical: m1, m2, chi1z, chi2z, dL, t (likelihood frame), phase, ra, dec, theta_jn, psi, chi1x, chi1y, chi2x, chi2y
const _ITRUTH = [36.0, 29.0, 0.3, -0.2, 0.5, 0.012, 0.4, 1.95, -1.27, 0.7, 0.8, 0.3, 0.2, -0.4, 0.1]
function _iwalkers()
    P = repeat(_ITRUTH, 1, 3)
    P[1, 2] *= 1.002          # m1 + 0.2%
    P[12, 3] += 0.05          # chi1x + 0.05
    return P
end

@testset "precessing injection templates: logL = 0 at the truth" begin
    tmpl = _itemplates()
    opts = (; time_reference=1, time_ref_frequency=80.0, time_ref_offset=0.01)
    @testset "$(nameof(typeof(model)))" for model in (IMRPhenomXP(), IMRPhenomXPHM())
        cfg = with_likelihood_options(PEConfig{Float64}(length(tmpl[1].freqs), 3, 1e-3, 1); opts...)
        h = KFi.xprec_strain(model, _ITRUTH, cfg, tmpl)
        data, info = simulate_injection(h, tmpl; noise=false)
        @test info.network_optimal_snr > 10
        ll = zeros(3)
        batched_log_likelihood_kernel!(ll, _iwalkers(), cfg, data; model=model)
        @test abs(ll[1]) < 1e-8 * info.network_optimal_snr^2
        @test ll[2] < -0.1 && ll[3] < -1e-3
    end
    @testset "SEOBNRv5PHM (device pipeline, CPU backend)" begin
        cfg = with_likelihood_options(PEConfig{Float64}(length(tmpl[1].freqs), 3, 1e-3, 1); opts...,
                                      phase_ref_frequency=50.0)
        h, ok = KFi.seobnr_phm_device_strain(_ITRUTH, cfg, tmpl)
        @test ok
        data, info = simulate_injection(h, tmpl; noise=false)
        @test info.network_optimal_snr > 10
        ll = zeros(3)
        KFi.seobnr_phm_device_loglikelihood!(ll, _iwalkers(), cfg, data)
        @test abs(ll[1]) < 1e-8 * info.network_optimal_snr^2
        @test ll[2] < -0.1 && ll[3] < -1e-3
    end
    @testset "noise" begin
        cfg = PEConfig{Float64}(length(tmpl[1].freqs), 1, 1e-3, 1)
        h = KFi.xprec_strain(IMRPhenomXPHM(), _ITRUTH, cfg, tmpl)
        data, _ = simulate_injection(h, tmpl; noise=true, rng=Random.Xoshiro(1))
        df = 0.25
        # whitened residual = noise: <|n|^2 / S> * 4 df = 2 per bin
        r = [abs2(data[1].strain[i] - h[1][i]) / tmpl[1].psd[i] * 4 * df for i in eachindex(h[1])]
        @test isapprox(sum(r) / length(r), 2.0; rtol=0.05)
    end
end

@testset "synthetic_gwdata: noise-curve detector" begin
    path = joinpath(@__DIR__, "..", "data", "noise_curves", "aligo_O4high.txt")
    d = GravitationalWaveBayes.synthetic_gwdata(; detector=H1, trigger_time=_IGPS, duration=4.0, f_min=20.0, f_max=1024.0, asd_file=path)
    @test d.freqs[1] == 20.0 && d.freqs[end] == 1024.0 && d.freqs[2] - d.freqs[1] == 0.25
    @test all(iszero, d.strain)
    rows = [parse.(Float64, split(l)) for l in eachline(path) if !startswith(l, "#")]
    j = findfirst(r -> r[1] >= 100.0, rows)
    f0, a0 = rows[j - 1]; f1, a1 = rows[j]
    i = findfirst(==(100.0), d.freqs)
    @test isapprox(d.psd[i], (a0 + (100.0 - f0) / (f1 - f0) * (a1 - a0))^2; rtol=1e-12)
    @test_throws ArgumentError GravitationalWaveBayes.synthetic_gwdata(; detector=H1, trigger_time=_IGPS)
end
