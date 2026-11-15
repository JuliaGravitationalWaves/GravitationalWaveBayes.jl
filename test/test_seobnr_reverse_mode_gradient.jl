using Test
using GravitationalWaveBayes
using GravitationalWaveforms: SEOBNRv5AlignedSpinF2

const BN = GravitationalWaveBayes.BatchedNUTS

@testset "SEOBNRv5AlignedSpinF2 reverse-mode Enzyme gradient (gw_logp_and_grad) vs finite differences and forward-mode" begin
    # Same smoke setup as test_seobnr_forward_mode_gradient.jl (synthetic strain/PSD, comfortably
    # inside the post-adiabatic table's valid [fmin,fmax] band for both walkers) -- CPU only, per
    # _reverse_mode_gradient!'s own docstring (no real CUDA validation yet).
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

    # Reverse is the package default (2026-09-19): force forward for the baseline, restore afterwards.
    old_reverse_flag = BN._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[]
    BN._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[] = false
    logp_and_grad_fwd! = BN.gw_logp_and_grad(data, cfg, freqs; model=SEOBNRv5AlignedSpinF2())
    grad_fwd = zeros(Float64, 11, nwalkers)
    logp_fwd = zeros(Float64, nwalkers)
    logp_and_grad_fwd!(grad_fwd, logp_fwd, positions, nothing)

    BN._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[] = true
    local grad_rev, logp_rev
    try
        logp_and_grad_rev! = BN.gw_logp_and_grad(data, cfg, freqs; model=SEOBNRv5AlignedSpinF2())
        grad_rev = zeros(Float64, 11, nwalkers)
        logp_rev = zeros(Float64, nwalkers)
        logp_and_grad_rev!(grad_rev, logp_rev, positions, nothing)
    finally
        BN._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[] = old_reverse_flag
    end

    @test all(isfinite, logp_rev)
    @test all(!iszero, logp_rev)
    @test logp_rev == logp_fwd  # forward pass (_eval_ll!) is untouched by which gradient path runs

    h = 1e-6
    fd = zeros(Float64, size(positions))
    tmp_grad = zeros(Float64, 11, nwalkers)
    tmp_logp = zeros(Float64, nwalkers)
    for i in eachindex(positions)
        p_plus = copy(positions); p_plus[i] += h
        p_minus = copy(positions); p_minus[i] -= h
        logp_and_grad_fwd!(tmp_grad, tmp_logp, p_plus, nothing)
        lp_plus = sum(tmp_logp)
        logp_and_grad_fwd!(tmp_grad, tmp_logp, p_minus, nothing)
        lp_minus = sum(tmp_logp)
        fd[i] = (lp_plus - lp_minus) / (2h)
    end

    row_names = ["mc", "q", "chi1", "chi2", "dL", "t_c", "phase", "ra", "sin_dec", "cos_theta_jn", "psi"]
    for (i, name) in enumerate(row_names)
        @test isapprox(grad_rev[i, :], fd[i, :]; rtol=1e-4, atol=1e-6)
    end
    # Cross-check directly against the already-validated forward-mode path too, not just FD.
    @test isapprox(grad_rev, grad_fwd; rtol=1e-4, atol=1e-6)
end
