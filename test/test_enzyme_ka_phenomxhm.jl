# Enzyme-through-KernelAbstractions test for IMRPhenomXHM (Phase 3 follow-on).
#
# In the slow suite (GRAVITATIONALWAVEBAYES_RUN_SLOW_TESTS=true, see test/runtests.jl): re-validated at 14 GB together with the
# other slow tests on 2026-10-17 (3-5 min each on wsl-laptop).
#
# Run directly: `julia --project=. test/test_enzyme_ka_phenomxhm.jl`

using Test
using Enzyme
using GravitationalWaveBayes
using GravitationalWaveforms: IMRPhenomXHM
using KernelAbstractions
using GravitationalWaveData
using Random

# NOTE (2026-09-20): this test used to use zero strain and unit PSD. Then every gradient entry is ~1e-45 (the
# log-likelihood is ~ -0.5 <h|h> with h ~ 1e-22 and PSD 1), far below the `atol=1e-6` below, so the Enzyme-vs-FD
# assertions passed VACUOUSLY and never noticed that IMRPhenomXHM's (3,2) mode was mis-differentiated (Enzyme
# reverse-mode bug on GravitationalWaveforms' old recursive Cramer's-rule `_xas_linsolve`, N = 6; up to 3 % error in the
# full gradient). Data are now signal-scale (strain 1e-22, PSD 1e-46) so the gradients are O(1e2..1e4), and the
# comparison is relative. Requires a GravitationalWaveforms with the straight-line `_xas_linsolve` (2026-09-20 fix).

@testset "Enzyme gradient through the fused KernelAbstractions kernel (IMRPhenomXHM)" begin
    # Same tiny problem size used throughout this codebase's Enzyme-through-KA validation
    # (test_enzyme_ka.jl's IMRPhenomD testset, test_enzyme_ka_xas.jl's IMRPhenomXAS testset,
    # IMRPhenomXP's own standalone testset per DEVELOPMENT_PLAN.md): ntime=16, nwalkers=2.
    #
    # `params` is (11, nwalkers) -- IMRPhenomXHM is aligned-spin/non-precessing, so it uses the
    # EXISTING 11-row convention (KernelFused.jl's module docstring), not IMRPhenomXP's 15-row
    # extension. Uses a genuinely asymmetric mass ratio (like the injection-recovery test in
    # test_phenomxhm.jl) so the higher-order-mode content this model exists for is actually
    # exercised by the gradient, not just the dominant (2,2) mode.
    ntime = 16
    nwalkers = 2
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    rng = Random.Xoshiro(3)
    strain = ComplexF64.(1e-22 .* (randn(rng, ntime) .+ im .* randn(rng, ntime)))
    psd = fill(1e-46, ntime)
    data = LikelihoodData(freqs, strain, psd)
    cfg = PEConfig{Float64}(ntime, nwalkers, 0.01, 1)

    params = zeros(Float64, 11, nwalkers)
    params[1, :] .= [32.0, 40.0]     # m1
    params[2, :] .= [8.0, 12.0]      # m2 (asymmetric mass ratio -- HOM content matters)
    params[3, :] .= [0.3, -0.2]      # chi1
    params[4, :] .= [-0.1, 0.15]     # chi2
    params[5, :] .= [0.5, 1.1]       # dL (Gpc)
    params[6, :] .= [0.015, -0.02]   # t_c (s)
    params[7, :] .= [0.4, 2.0]       # phase (rad)
    params[8, :] .= [1.0, 4.0]       # ra (rad)
    params[9, :] .= [-0.3, 0.3]      # dec (rad)
    params[10, :] .= [1.0, 2.0]      # theta_jn (rad)
    params[11, :] .= [0.6, 2.5]      # psi (rad)

    function sum_ll_kernel(params, freqs, cfg, data)
        ll = zeros(eltype(params), size(params, 2))
        batched_log_likelihood_kernel!(ll, params, freqs, cfg, data; model=IMRPhenomXHM())
        return sum(ll)
    end

    dparams = zeros(Float64, size(params))
    t0 = time()
    Enzyme.autodiff(Enzyme.Reverse, sum_ll_kernel,
        Enzyme.Duplicated(params, dparams),
        Enzyme.Const(freqs), Enzyme.Const(cfg), Enzyme.Const(data))
    println("Enzyme.autodiff(IMRPhenomXHM) completed in $(time() - t0) s")

    # Central differences with a relative step; take the best agreement over a small step sweep so a single unlucky
    # step (roundoff for small h, truncation for large h) cannot fail or mask the test.
    function fd_entry(i)
        best = Inf
        val = 0.0
        for hf in (1e-4, 1e-5, 1e-6)
            h = hf * max(abs(params[i]), 1.0)
            p_plus = copy(params); p_plus[i] += h
            p_minus = copy(params); p_minus[i] -= h
            fd_i = (sum_ll_kernel(p_plus, freqs, cfg, data) - sum_ll_kernel(p_minus, freqs, cfg, data)) / (2h)
            err = abs(dparams[i] - fd_i)
            if err < best
                best = err
                val = fd_i
            end
        end
        return val
    end
    fd = zeros(Float64, size(params))
    for i in eachindex(params)
        fd[i] = fd_entry(i)
    end

    gscale = maximum(abs, fd)
    @test gscale > 1.0   # the test must be non-vacuous: gradients at signal scale, not ~1e-45
    println("max |gradient| = ", gscale, "; max abs diff (Enzyme vs FD): ", maximum(abs.(dparams .- fd)),
        "; worst relative to the entry itself: ", maximum(abs.(dparams .- fd) ./ max.(abs.(fd), 1e-8 * gscale)))
    @test isapprox(dparams, fd; rtol=1e-3, atol=1e-8 * gscale)

    row_names = ["m1", "m2", "chi1", "chi2", "dL", "t_c", "phase", "ra", "dec", "theta_jn", "psi"]
    for (i, name) in enumerate(row_names)
        @test isapprox(dparams[i, :], fd[i, :]; rtol=1e-3, atol=1e-8 * gscale)
    end
end
