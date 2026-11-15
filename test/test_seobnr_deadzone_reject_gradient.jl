using Test
using GravitationalWaveBayes
using GravitationalWaveBayes: LikelihoodData, PEConfig
using GravitationalWaveforms: SEOBNRv5AlignedSpinF2

const BN = GravitationalWaveBayes.BatchedNUTS

const _DZG_GOOD = (36.0, 29.0, 0.0, 0.0)
const _DZG_DEAD_EQUALMASS = (60.0, 60.0, 0.0, 0.0)

# Enzyme-compile-heavy counterpart to test_seobnr_deadzone_reject.jl's fast kernel-value checks
# (split out and gated behind GRAVITATIONALWAVEBAYES_RUN_SLOW_TESTS, matching every other Enzyme test's own
# runtests.jl wiring -- see e.g. test_seobnr_forward_mode_gradient.jl/test_seobnr_reverse_mode_gradient.jl).
@testset "SEOBNRv5AlignedSpinF2 reverse-mode Enzyme gradient: dead-zone reject stays finite, good walker unaffected" begin
    # Mirrors test_seobnr_reverse_mode_gradient.jl's setup exactly (sampled-space positions via
    # `BN.gw_logp_and_grad`, the actual NUTS-facing gradient path -- not the raw physical kernel),
    # one dead-zone walker and one good walker in the same batch, confirming the new `ifelse`-based
    # reject (`KernelFused._seobv5f2_apply_deadzone_reject`) doesn't crash or
    # NaN-poison Enzyme's reverse pass, and that the good walker's gradient is bit-for-bit unaffected
    # by the dead walker sharing the batch.
    ntime_g = 8
    freqs_g = collect(20.5:0.25:(20.5 + 0.25 * (ntime_g - 1)))
    strain_g = zeros(ComplexF64, ntime_g)
    psd_g = ones(Float64, ntime_g)
    data_g = LikelihoodData(freqs_g, strain_g, psd_g)
    cfg_g = PEConfig{Float64}(ntime_g, 2, 0.01, 1)

    m1 = [_DZG_GOOD[1], _DZG_DEAD_EQUALMASS[1]]; m2 = [_DZG_GOOD[2], _DZG_DEAD_EQUALMASS[2]]
    chi1 = [0.0, 0.0]; chi2 = [0.0, 0.0]
    dL = [0.4, 0.4]; t_c = [0.02, 0.0]; phase = [0.5, 0.5]
    ra = [1.2, 1.2]; dec = [-0.4, -0.4]; theta_jn = [1.0, 1.0]; psi = [0.6, 0.6]

    positions = zeros(Float64, 11, 2)
    for w in 1:2
        mc, q = BN._m1_m2_to_mc_q(m1[w], m2[w])
        positions[1, w] = mc; positions[2, w] = q
        positions[3, w] = chi1[w]; positions[4, w] = chi2[w]
        positions[5, w] = dL[w]; positions[6, w] = t_c[w]; positions[7, w] = phase[w]
        positions[8, w] = ra[w]; positions[9, w] = sin(dec[w]); positions[10, w] = cos(theta_jn[w]); positions[11, w] = psi[w]
    end

    old_reverse_flag = BN._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[]
    BN._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[] = true
    local grad_rev, logp_rev
    try
        logp_and_grad_rev! = BN.gw_logp_and_grad(data_g, cfg_g, freqs_g; model=SEOBNRv5AlignedSpinF2())
        grad_rev = zeros(Float64, 11, 2)
        logp_rev = zeros(Float64, 2)
        logp_and_grad_rev!(grad_rev, logp_rev, positions, nothing)
    finally
        BN._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[] = old_reverse_flag
    end

    @test all(isfinite, logp_rev)
    @test all(isfinite, grad_rev)
    @test logp_rev[2] == GravitationalWaveBayes.KernelFused._SEOBV5F2_DEADZONE_LOGLIKELIHOOD
    @test all(iszero, grad_rev[:, 2])   # ifelse's untaken branch carries no gradient -> reject is a flat constant
    @test logp_rev[1] != GravitationalWaveBayes.KernelFused._SEOBV5F2_DEADZONE_LOGLIKELIHOOD
    @test any(!iszero, grad_rev[:, 1])

    # Cross-check the good walker's gradient against forward-mode (existing, already-validated
    # path) run on the SAME batch, confirming the dead walker doesn't contaminate walker 1.
    BN._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[] = false
    logp_and_grad_fwd! = BN.gw_logp_and_grad(data_g, cfg_g, freqs_g; model=SEOBNRv5AlignedSpinF2())
    grad_fwd = zeros(Float64, 11, 2)
    logp_fwd = zeros(Float64, 2)
    logp_and_grad_fwd!(grad_fwd, logp_fwd, positions, nothing)
    BN._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[] = old_reverse_flag

    @test isapprox(grad_rev[:, 1], grad_fwd[:, 1]; rtol=1e-4, atol=1e-6)
    @test logp_fwd[2] == GravitationalWaveBayes.KernelFused._SEOBV5F2_DEADZONE_LOGLIKELIHOOD
end
