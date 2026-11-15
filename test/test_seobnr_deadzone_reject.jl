using Test
using GravitationalWaveBayes
using GravitationalWaveBayes: LikelihoodData, PEConfig, batched_log_likelihood_kernel!, binned_log_likelihood_kernel!, build_relative_binning
using GravitationalWaveforms: SEOBNRv5AlignedSpinF2
using GravitationalWaveforms: _seobv5f2_coeffs
using GravitationalWaveData

# ---------------------------------------------------------------------------------------------
# `SEOBNRv5AlignedSpinF2`'s post-adiabatic table does
# not reach the analysis band (`coeffs.fmax < f_min`, or `!coeffs.success`) for a large fraction
# of this sampler's own `(mc, q)` prior. Before this fix, every likelihood kernel silently
# returned a walker-independent, ZERO-gradient noise-only constant there (`_seobv5f2_ampl_scalar`/
# `_seobv5f2_phi_scalar` both hard-return `0.0` out of band, so `h` is identically zero). The fix
# (`_seobv5f2_apply_deadzone_reject` in `KernelFused.jl`/`RelativeBinning.jl`) turns that flat
# constant into a large, finite, unconditional reject so NUTS's `max_delta_energy` divergence
# check (default `1000`) always fires there instead of silently accepting a flat region.
#
# Dead-zone/good points below are NOT guessed -- confirmed directly via `_seobv5f2_coeffs` before
# writing any assertion (same discipline as this repo's other `SEOBNRv5AlignedSpinF2` tests):
#   good:  m1=36, m2=29, chi=0     -> success=true,  fmin=20.0, fmax=31.5
#   dead:  m1=60, m2=60, chi=0     -> success=false, fmax=2.1   (q=1.0, high total mass)
#   dead:  m1=70, m2=8.75, chi=0   -> success=false, fmax=3.2   (q=0.125, extreme mass ratio)
# ---------------------------------------------------------------------------------------------

function _mc_eta(m1, m2)
    mu = m1 * m2 / (m1 + m2)
    mc = (mu / (m1 + m2))^0.6 * (m1 + m2)
    eta = mu / (m1 + m2)
    return mc, eta
end

const _DZ_GOOD = (36.0, 29.0, 0.0, 0.0)
const _DZ_DEAD_EQUALMASS = (60.0, 60.0, 0.0, 0.0)
const _DZ_DEAD_EXTREME_Q = (70.0, 8.75, 0.0, 0.0)

@testset "SEOBNRv5AlignedSpinF2 dead-zone hard-reject" begin
    @testset "sanity: coeffs.success/fmax at the fixed test points, matches what's hardcoded above" begin
        model = SEOBNRv5AlignedSpinF2()
        mc, eta = _mc_eta(_DZ_GOOD[1], _DZ_GOOD[2])
        c = _seobv5f2_coeffs(model, mc, eta, _DZ_GOOD[3], _DZ_GOOD[4])
        @test c.success
        @test c.fmax >= 20.0

        for pt in (_DZ_DEAD_EQUALMASS, _DZ_DEAD_EXTREME_Q)
            mc, eta = _mc_eta(pt[1], pt[2])
            c = _seobv5f2_coeffs(model, mc, eta, pt[3], pt[4])
            @test !c.success || c.fmax < 20.0
        end
    end

    T = Float64
    fmin, fmax, df = 20.5, 30.5, 0.05
    freqs = collect(fmin:df:fmax)
    ntime = length(freqs)
    model = SEOBNRv5AlignedSpinF2()
    reject_dense = GravitationalWaveBayes.KernelFused._SEOBV5F2_DEADZONE_LOGLIKELIHOOD
    reject_binned = GravitationalWaveBayes.RelativeBinning._SEOBV5F2_DEADZONE_LOGLIKELIHOOD
    @test reject_dense == reject_binned      # same constant, independently duplicated (matches `_safe_div`'s convention)

    _physical(m1, m2, chi1, chi2) = T[m1, m2, chi1, chi2, 0.4, 0.0, 0.5, 1.2, -0.4, 1.0, 0.6]

    @testset "single-detector dense kernel: dead zone rejects, good point unaffected" begin
        data0 = LikelihoodData(freqs, zeros(ComplexF64, ntime), fill(T(1e-46), ntime))
        cfg = PEConfig{T}(ntime, 1, 1e-3, 1)

        for pt in (_DZ_DEAD_EQUALMASS, _DZ_DEAD_EXTREME_Q)
            p = reshape(_physical(pt...), 11, 1)
            ll = zeros(T, 1)
            batched_log_likelihood_kernel!(ll, p, freqs, cfg, data0; model=model)
            @test ll[1] == reject_dense
        end

        # Good point: a real (noise-free-injection) likelihood, not the reject constant, and
        # matches what the kernel returns against zero strain (pure `-<h|h>`-style self term),
        # i.e. the ordinary computation still runs exactly as before this fix for an in-band point.
        p_good = reshape(_physical(_DZ_GOOD...), 11, 1)
        ll_good = zeros(T, 1)
        batched_log_likelihood_kernel!(ll_good, p_good, freqs, cfg, data0; model=model)
        @test isfinite(ll_good[1])
        @test ll_good[1] != reject_dense
        @test ll_good[1] < 0   # noise-only strain, nonzero template -> negative log-likelihood
    end

    @testset "single-detector relative-binning kernel: dead zone rejects, good point unaffected" begin
        physical_ref = _physical(_DZ_GOOD...)
        data0 = LikelihoodData(freqs, zeros(ComplexF64, ntime), fill(T(1e-46), ntime))
        rb = build_relative_binning(model, physical_ref, data0; nbins=30)
        cfg = PEConfig{T}(ntime, 1, 1e-3, 1)

        for pt in (_DZ_DEAD_EQUALMASS, _DZ_DEAD_EXTREME_Q)
            p = reshape(_physical(pt...), 11, 1)
            ll = zeros(T, 1)
            binned_log_likelihood_kernel!(ll, p, rb, data0; model=model)
            @test ll[1] == reject_binned
        end

        p_good = reshape(physical_ref, 11, 1)
        ll_dense = zeros(T, 1)
        batched_log_likelihood_kernel!(ll_dense, p_good, freqs, cfg, data0; model=model)
        ll_binned = zeros(T, 1)
        binned_log_likelihood_kernel!(ll_binned, p_good, rb, data0; model=model)
        @test isapprox(ll_dense[1], ll_binned[1]; atol=1e-6)   # unchanged regression check (theta_ref, matches test_relative_binning.jl's own tolerance)
        @test ll_binned[1] != reject_binned
    end

    @testset "multi-detector dense + binned kernels: dead zone rejects" begin
        gps = 1126259462.4
        dets = [GravitationalWaveData.H1, GravitationalWaveData.L1]
        physical_ref = _physical(_DZ_GOOD...)
        data0 = [LikelihoodData(freqs, zeros(ComplexF64, ntime), fill(T(1e-46), ntime), d, gps) for d in dets]
        rb = build_relative_binning(model, physical_ref, data0; nbins=30)
        cfg = PEConfig{T}(ntime, 1, 1e-3, 1)

        for pt in (_DZ_DEAD_EQUALMASS, _DZ_DEAD_EXTREME_Q)
            p = reshape(_physical(pt...), 11, 1)
            ll_dense = zeros(T, 1)
            batched_log_likelihood_kernel!(ll_dense, p, cfg, data0; model=model)
            @test ll_dense[1] == reject_dense
            ll_binned = zeros(T, 1)
            binned_log_likelihood_kernel!(ll_binned, p, rb, data0; model=model)
            @test ll_binned[1] == reject_binned
        end

        p_good = reshape(physical_ref, 11, 1)
        ll_dense_good = zeros(T, 1)
        batched_log_likelihood_kernel!(ll_dense_good, p_good, cfg, data0; model=model)
        @test ll_dense_good[1] != reject_dense
        @test isfinite(ll_dense_good[1])
    end
end

# The Enzyme reverse-mode gradient check for this same fix (confirms the `ifelse`-based reject is
# a genuine zero-gradient constant, not a NaN/crash risk, and doesn't contaminate a good walker
# sharing the same batch) is split into test_seobnr_deadzone_reject_gradient.jl -- Enzyme-compile-
# heavy, gated behind GRAVITATIONALWAVEBAYES_RUN_SLOW_TESTS in runtests.jl, matching every other Enzyme test here.
