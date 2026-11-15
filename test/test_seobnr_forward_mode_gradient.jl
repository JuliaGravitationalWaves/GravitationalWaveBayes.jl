using Test
using GravitationalWaveBayes
using GravitationalWaveforms: SEOBNRv5AlignedSpinF2

const BN = GravitationalWaveBayes.BatchedNUTS

# Reverse mode is now the default for this model (2026-09-19); this file tests the FORWARD path, so force it.
const _fwd_test_old_reverse_flag = BN._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[]
BN._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[] = false

@testset "SEOBNRv5AlignedSpinF2 forward-mode Enzyme gradient (gw_logp_and_grad) vs finite differences" begin
    # HANDOFF.md (2026-09-13, part 7, item 1): `SEOBNRv5AlignedSpinF2` uses `Enzyme.Forward`
    # (via `BatchedNUTS._forward_mode_gradient!`), not `Enzyme.Reverse`, because
    # `Enzyme.autodiff(Enzyme.Reverse, ...)` through this model's `exp(im*phase)` kernel body on
    # `CUDABackend()` is a hard, uncatchable SIGABRT
    # (`ISSUE_enzyme_gpu_complex_typeanalysis_crash.md`) -- Forward mode doesn't build the
    # reverse-mode-only machinery that crashes there. This test exercises the FULL
    # `gw_logp_and_grad` closure (sampled-space `positions`, not the physical-row convention
    # `test_enzyme_ka_seobnrv5alignedspinf2.jl` uses directly), CPU only -- see
    # DEVELOPMENT_PLAN.md for why real CUDA hardware still fails here (three separate, genuinely
    # upstream Enzyme.jl/CUDA.jl issues, not a gap in this codebase's own logic).
    #
    # Same frequency band/physical configuration as `test_enzyme_ka_seobnrv5alignedspinf2.jl`
    # (comfortably inside the post-adiabatic table's valid `[fmin,fmax]` band for both walkers).
    ntime = 8
    nwalkers = 2
    freqs = collect(20.5:0.25:(20.5 + 0.25 * (ntime - 1)))
    strain = zeros(ComplexF64, ntime)
    psd = ones(Float64, ntime)
    data = LikelihoodData(freqs, strain, psd)
    cfg = PEConfig{Float64}(ntime, nwalkers, 0.01, 1)

    m1 = [36.0, 36.5]; m2 = [29.0, 29.2]
    chi1 = [0.0, 0.05]; chi2 = [0.0, -0.05]
    dL = [0.4, 0.42]; t_c = [0.02, -0.03]; phase = [0.5, 0.55]
    ra = [1.2, 1.25]; dec = [-0.4, -0.35]; theta_jn = [1.0, 1.05]; psi = [0.6, 0.65]

    positions = zeros(Float64, 11, nwalkers)
    for w in 1:nwalkers
        mc, q = BN._m1_m2_to_mc_q(m1[w], m2[w])
        positions[1, w] = mc
        positions[2, w] = q
        positions[3, w] = chi1[w]
        positions[4, w] = chi2[w]
        positions[5, w] = dL[w]
        positions[6, w] = t_c[w]
        positions[7, w] = phase[w]
        positions[8, w] = ra[w]
        positions[9, w] = sin(dec[w])
        positions[10, w] = cos(theta_jn[w])
        positions[11, w] = psi[w]
    end

    logp_and_grad! = BN.gw_logp_and_grad(data, cfg, freqs; model=SEOBNRv5AlignedSpinF2())

    grad = zeros(Float64, 11, nwalkers)
    logp = zeros(Float64, nwalkers)
    logp_and_grad!(grad, logp, positions, nothing)

    @test all(isfinite, logp)
    @test all(!iszero, logp)

    h = 1e-6
    fd = zeros(Float64, size(positions))
    tmp_grad = zeros(Float64, 11, nwalkers)
    tmp_logp = zeros(Float64, nwalkers)
    for i in eachindex(positions)
        p_plus = copy(positions); p_plus[i] += h
        p_minus = copy(positions); p_minus[i] -= h
        logp_and_grad!(tmp_grad, tmp_logp, p_plus, nothing)
        lp_plus = sum(tmp_logp)
        logp_and_grad!(tmp_grad, tmp_logp, p_minus, nothing)
        lp_minus = sum(tmp_logp)
        fd[i] = (lp_plus - lp_minus) / (2h)
    end

    @test isapprox(grad, fd; rtol=1e-4, atol=1e-6)

    row_names = ["mc", "q", "chi1", "chi2", "dL", "t_c", "phase", "ra", "sin_dec", "cos_theta_jn", "psi"]
    for (i, name) in enumerate(row_names)
        @test isapprox(grad[i, :], fd[i, :]; rtol=1e-4, atol=1e-6)
    end
end

BN._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[] = _fwd_test_old_reverse_flag
