# Enzyme-through-KernelAbstractions test for the IMRPhenomXHM relative-binning kernel
# (Phase 3 follow-on).
#
# In the slow suite (GRAVITATIONALWAVEBAYES_RUN_SLOW_TESTS=true, see test/runtests.jl): re-validated at 14 GB together with the
# other slow tests on 2026-10-17 (3-5 min each on wsl-laptop).
#
# Also worth noting up front (per this task's own validation brief): before this file was written,
# `RelativeBinning.jl`'s relative-binning kernels had an Enzyme test for IMRPhenomD (committed,
# tiny problem) and standalone-only tests for IMRPhenomXAS/IMRPhenomXP (per DEVELOPMENT_PLAN.md) --
# so there IS established precedent for Enzyme-testing this file's relative-binning kernels
# per-model, and this file follows that precedent for IMRPhenomXHM specifically.
#
# Run directly: `julia --project=. test/test_enzyme_ka_relative_binning_phenomxhm.jl`

using Test
using Enzyme
using GravitationalWaveBayes
using GravitationalWaveBayes: build_relative_binning, binned_log_likelihood_kernel!
using GravitationalWaveforms: IMRPhenomXHM
using KernelAbstractions
using GravitationalWaveData

@testset "Enzyme gradient through the binned relative-binning kernel (IMRPhenomXHM)" begin
    # Same tiny problem size used throughout this codebase's Enzyme-through-KA validation
    # (test_enzyme_ka_phenomxhm.jl's dense-kernel testset, IMRPhenomXP's own standalone
    # relative-binning testset per DEVELOPMENT_PLAN.md): ntime=16, nwalkers=2, nbins=8.
    #
    # `params`/`params_ref` are (11, nwalkers) -- IMRPhenomXHM's existing aligned-spin convention
    # (not IMRPhenomXP's 15-row extension). Genuinely asymmetric mass ratio so higher-order-mode
    # content is actually exercised by the gradient, matching test_phenomxhm.jl's own convention.
    ntime = 16
    nwalkers = 2
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    strain = zeros(ComplexF64, ntime)
    psd = ones(Float64, ntime)
    data = LikelihoodData(freqs, strain, psd)

    model = IMRPhenomXHM()
    physical_ref = [32.0, 8.0, 0.3, -0.1, 0.5, 0.015, 0.4, 1.0, -0.3, 1.0, 0.6]
    rb = build_relative_binning(model, physical_ref, data; nbins=8)

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

    function sum_ll_binned(params, rb, data, model)
        ll = zeros(eltype(params), size(params, 2))
        binned_log_likelihood_kernel!(ll, params, rb, data; model=model)
        return sum(ll)
    end

    dparams = zeros(Float64, size(params))
    t0 = time()
    Enzyme.autodiff(Enzyme.Reverse, sum_ll_binned,
        Enzyme.Duplicated(params, dparams),
        Enzyme.Const(rb), Enzyme.Const(data), Enzyme.Const(model))
    println("Enzyme.autodiff(binned IMRPhenomXHM) completed in $(time() - t0) s")

    h = 1e-6
    fd = zeros(Float64, size(params))
    for i in eachindex(params)
        p_plus = copy(params); p_plus[i] += h
        p_minus = copy(params); p_minus[i] -= h
        fd[i] = (sum_ll_binned(p_plus, rb, data, model) - sum_ll_binned(p_minus, rb, data, model)) / (2h)
    end

    println("max abs diff (Enzyme vs FD): ", maximum(abs.(dparams .- fd)))
    @test isapprox(dparams, fd; rtol=1e-4, atol=1e-6)

    row_names = ["m1", "m2", "chi1", "chi2", "dL", "t_c", "phase", "ra", "dec", "theta_jn", "psi"]
    for (i, name) in enumerate(row_names)
        @test isapprox(dparams[i, :], fd[i, :]; rtol=1e-4, atol=1e-6)
    end
end
