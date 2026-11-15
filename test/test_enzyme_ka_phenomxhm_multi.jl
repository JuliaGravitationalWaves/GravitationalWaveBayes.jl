# Enzyme-through-KernelAbstractions test for multi-detector IMRPhenomXHM (companion of the
# single-detector `test_enzyme_ka_phenomxhm.jl` testset).
#
# In the slow suite (GRAVITATIONALWAVEBAYES_RUN_SLOW_TESTS=true, see test/runtests.jl); 3-5 min.
#
# Run directly: `julia --project=. test/test_enzyme_ka_phenomxhm_multi.jl`

using Test
using Enzyme
using GravitationalWaveBayes
using GravitationalWaveforms: IMRPhenomXHM
using KernelAbstractions
using GravitationalWaveData

@testset "Enzyme gradient through the fused KernelAbstractions kernel (IMRPhenomXHM, multi-detector H1+L1)" begin
    # Same tiny problem size used throughout this codebase's Enzyme-through-KA validation
    # (test_enzyme_ka.jl's IMRPhenomD testset, test_enzyme_ka_xas.jl's IMRPhenomXAS testset,
    # IMRPhenomXP's multi-detector standalone check, and the
    # single-detector IMRPhenomXHM testset above): ntime=16, nwalkers=2.
    #
    # `params` is (11, nwalkers) -- IMRPhenomXHM is aligned-spin/non-precessing, so it uses the
    # EXISTING 11-row convention (KernelFused.jl's module docstring), not IMRPhenomXP's 15-row
    # extension. Genuinely asymmetric mass ratio (like the single-detector Enzyme testset) so the
    # higher-order-mode content this model exists for is actually exercised by the gradient.
    # Genuinely nonzero ra/dec (not on-axis/degenerate) so H1's and L1's own `dt_geo` differ and
    # the new per-detector time-delay term is actually exercised by the gradient, not just present.
    ntime = 16
    nwalkers = 2
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    strain_h1 = zeros(ComplexF64, ntime)
    strain_l1 = zeros(ComplexF64, ntime)
    psd = ones(Float64, ntime)
    gps = 1126259462.4
    data = [LikelihoodData(freqs, strain_h1, psd, GravitationalWaveData.H1, gps),
            LikelihoodData(freqs, strain_l1, psd, GravitationalWaveData.L1, gps)]
    cfg = PEConfig{Float64}(ntime, nwalkers, 0.01, 1)

    params = zeros(Float64, 11, nwalkers)
    params[1, :] .= [32.0, 40.0]     # m1
    params[2, :] .= [8.0, 12.0]      # m2 (asymmetric mass ratio -- HOM content matters)
    params[3, :] .= [0.3, -0.2]      # chi1
    params[4, :] .= [-0.1, 0.15]     # chi2
    params[5, :] .= [0.5, 1.1]       # dL (Gpc)
    params[6, :] .= [0.015, -0.02]   # t_c (s)
    params[7, :] .= [0.4, 2.0]       # phase (rad)
    params[8, :] .= [2.1, 4.0]       # ra (rad) -- away from H1/L1-degenerate sky positions
    params[9, :] .= [0.4, 0.3]       # dec (rad)
    params[10, :] .= [1.0, 2.0]      # theta_jn (rad)
    params[11, :] .= [0.6, 2.5]      # psi (rad)

    function sum_ll_kernel(params, cfg, data)
        ll = zeros(eltype(params), size(params, 2))
        batched_log_likelihood_kernel!(ll, params, cfg, data; model=IMRPhenomXHM())
        return sum(ll)
    end

    # Host-side loop over the dynamic `Vector{LikelihoodData}` in `data` inside
    # `batched_log_likelihood_kernel!`'s multi-detector method is the SAME pattern that needed
    # `Enzyme.set_runtime_activity(Enzyme.Reverse)` for every other multi-detector model in this
    # codebase (IMRPhenomD/IMRPhenomXAS's multi-detector testsets, IMRPhenomXP's own multi-detector
    # testset) -- applied here too, confirmed empirically (not just assumed by analogy) via the
    # numerical agreement check below.
    dparams = zeros(Float64, size(params))
    t0 = time()
    Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse), sum_ll_kernel,
        Enzyme.Duplicated(params, dparams),
        Enzyme.Const(cfg), Enzyme.Const(data))
    println("Enzyme.autodiff(IMRPhenomXHM multi-detector) completed in $(time() - t0) s")

    h = 1e-6
    fd = zeros(Float64, size(params))
    for i in eachindex(params)
        p_plus = copy(params); p_plus[i] += h
        p_minus = copy(params); p_minus[i] -= h
        fd[i] = (sum_ll_kernel(p_plus, cfg, data) - sum_ll_kernel(p_minus, cfg, data)) / (2h)
    end

    println("max abs diff (Enzyme vs FD): ", maximum(abs.(dparams .- fd)))
    @test isapprox(dparams, fd; rtol=1e-4, atol=1e-6)

    row_names = ["m1", "m2", "chi1", "chi2", "dL", "t_c", "phase", "ra", "dec", "theta_jn", "psi"]
    for (i, name) in enumerate(row_names)
        @test isapprox(dparams[i, :], fd[i, :]; rtol=1e-4, atol=1e-6)
    end
end
