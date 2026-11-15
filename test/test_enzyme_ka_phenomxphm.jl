# Enzyme-through-KernelAbstractions test for IMRPhenomXPHM (Phase 5 Task 5b).
#
# In the slow suite (GRAVITATIONALWAVEBAYES_RUN_SLOW_TESTS=true, see test/runtests.jl): re-validated at 14 GB together with the
# other slow tests on 2026-10-17 (3-5 min each on wsl-laptop).
#
# Run directly: `julia --project=. test/test_enzyme_ka_phenomxphm.jl`

using Test
using Enzyme
using GravitationalWaveBayes
using GravitationalWaveforms: IMRPhenomXPHM
using KernelAbstractions
using GravitationalWaveData

@testset "Enzyme gradient through the fused KernelAbstractions kernel (IMRPhenomXPHM)" begin
    # Same tiny problem size used throughout this codebase's Enzyme-through-KA validation
    # (test_enzyme_ka.jl's IMRPhenomD testset, test_enzyme_ka_xas.jl's IMRPhenomXAS testset,
    # IMRPhenomXP's and IMRPhenomXHM's own standalone testsets): ntime=16, nwalkers=2.
    #
    # `params` is (15, nwalkers) -- IMRPhenomXPHM needs IMRPhenomXP's own precessing-spin row
    # convention (KernelFused.jl's module docstring, rows 12-15), not IMRPhenomXHM's 11-row one.
    # Uses a genuinely precessing (nonzero chi1x/chi1y/chi2x/chi2y), asymmetric-mass-ratio
    # configuration (mirroring test_phenomxhm.jl's own injection-recovery config) so both the
    # precession AND higher-order-mode content this model exists for are actually exercised by the
    # gradient, not just the aligned-spin dominant mode.
    ntime = 16
    nwalkers = 2
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    strain = zeros(ComplexF64, ntime)
    psd = ones(Float64, ntime)
    data = LikelihoodData(freqs, strain, psd)
    cfg = PEConfig{Float64}(ntime, nwalkers, 0.01, 1)

    params = zeros(Float64, 15, nwalkers)
    params[1, :] .= [32.0, 40.0]     # m1
    params[2, :] .= [8.0, 12.0]      # m2 (asymmetric mass ratio -- HOM content matters)
    params[3, :] .= [0.3, -0.2]      # chi1 (aligned/z component)
    params[4, :] .= [-0.1, 0.15]     # chi2 (aligned/z component)
    params[5, :] .= [0.5, 1.1]       # dL (Gpc)
    params[6, :] .= [0.015, -0.02]   # t_c (s)
    params[7, :] .= [0.4, 2.0]       # phase (rad)
    params[8, :] .= [1.0, 4.0]       # ra (rad)
    params[9, :] .= [-0.3, 0.3]      # dec (rad)
    params[10, :] .= [1.0, 2.0]      # theta_jn (rad)
    params[11, :] .= [0.6, 2.5]      # psi (rad)
    params[12, :] .= [0.25, -0.15]   # chi1x (in-plane -- genuinely precessing)
    params[13, :] .= [0.1, 0.2]      # chi1y
    params[14, :] .= [-0.2, 0.1]     # chi2x
    params[15, :] .= [0.15, -0.1]    # chi2y

    function sum_ll_kernel(params, freqs, cfg, data)
        ll = zeros(eltype(params), size(params, 2))
        batched_log_likelihood_kernel!(ll, params, freqs, cfg, data; model=IMRPhenomXPHM())
        return sum(ll)
    end

    dparams = zeros(Float64, size(params))
    t0 = time()
    Enzyme.autodiff(Enzyme.Reverse, sum_ll_kernel,
        Enzyme.Duplicated(params, dparams),
        Enzyme.Const(freqs), Enzyme.Const(cfg), Enzyme.Const(data))
    println("Enzyme.autodiff(IMRPhenomXPHM) completed in $(time() - t0) s")

    h = 1e-6
    fd = zeros(Float64, size(params))
    for i in eachindex(params)
        p_plus = copy(params); p_plus[i] += h
        p_minus = copy(params); p_minus[i] -= h
        fd[i] = (sum_ll_kernel(p_plus, freqs, cfg, data) - sum_ll_kernel(p_minus, freqs, cfg, data)) / (2h)
    end

    println("max abs diff (Enzyme vs FD): ", maximum(abs.(dparams .- fd)))
    @test isapprox(dparams, fd; rtol=1e-4, atol=1e-6)

    row_names = ["m1", "m2", "chi1", "chi2", "dL", "t_c", "phase", "ra", "dec", "theta_jn", "psi",
                 "chi1x", "chi1y", "chi2x", "chi2y"]
    for (i, name) in enumerate(row_names)
        @test isapprox(dparams[i, :], fd[i, :]; rtol=1e-4, atol=1e-6)
    end
end
