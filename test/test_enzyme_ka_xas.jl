using Test
using Enzyme
using GravitationalWaveBayes
using KernelAbstractions
using GravitationalWaveforms: IMRPhenomXAS

@testset "Enzyme gradient through the fused KernelAbstractions kernel (IMRPhenomXAS)" begin
    # Same tiny problem size (and same rationale) as the IMRPhenomD testset in test_enzyme_ka.jl.
    # IMRPhenomXAS now has its own non-allocating scalar core (_xas_phi_coeffs/_xas_phi_scalar/
    # _xas_ampl_coeffs/_xas_ampl_scalar, see GravitationalWaveforms.jl's IMRPhenomXAS.jl) and a dedicated fused-kernel
    # body (_fused_waveform_likelihood_kernel_phenomxas! in KernelFused.jl), mirroring the fix
    # already landed for IMRPhenomD -- see the "Enzyme-through-KernelAbstractions" note in
    # DEVELOPMENT_PLAN.md.
    #
    # `params` is (11, nwalkers), the full physical row convention (Phase 2b Task 4) -- see
    # test_enzyme_ka.jl's matching testset docstring for why the full 11-row space (not just the
    # original 4) is what's actually being validated here.
    ntime = 16
    nwalkers = 2
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    strain = zeros(ComplexF64, ntime)
    psd = ones(Float64, ntime)
    data = LikelihoodData(freqs, strain, psd)
    cfg = PEConfig{Float64}(ntime, nwalkers, 0.01, 1)

    params = zeros(Float64, 11, nwalkers)
    params[1, :] .= [30.0, 40.0]     # m1
    params[2, :] .= [25.0, 20.0]     # m2
    params[3, :] .= [0.5, -0.2]      # chi1
    params[4, :] .= [0.3, 0.1]       # chi2
    params[5, :] .= [0.4, 1.1]       # dL (Gpc)
    params[6, :] .= [0.02, -0.03]    # t_c (s)
    params[7, :] .= [0.3, 2.0]       # phase (rad)
    params[8, :] .= [1.2, 4.0]       # ra (rad)
    params[9, :] .= [-0.5, 0.3]      # dec (rad)
    params[10, :] .= [1.0, 2.0]      # theta_jn (rad)
    params[11, :] .= [0.7, 2.5]      # psi (rad)

    function sum_ll_kernel_xas(params, freqs, cfg, data)
        ll = zeros(eltype(params), size(params, 2))
        batched_log_likelihood_kernel!(ll, params, freqs, cfg, data; model = IMRPhenomXAS())
        return sum(ll)
    end

    dparams = zeros(Float64, size(params))
    Enzyme.autodiff(Enzyme.Reverse, sum_ll_kernel_xas,
        Enzyme.Duplicated(params, dparams),
        Enzyme.Const(freqs), Enzyme.Const(cfg), Enzyme.Const(data))

    h = 1e-6
    fd = zeros(Float64, size(params))
    for i in eachindex(params)
        p_plus = copy(params); p_plus[i] += h
        p_minus = copy(params); p_minus[i] -= h
        fd[i] = (sum_ll_kernel_xas(p_plus, freqs, cfg, data) - sum_ll_kernel_xas(p_minus, freqs, cfg, data)) / (2h)
    end

    @test isapprox(dparams, fd; rtol=1e-4, atol=1e-6)

    row_names = ["m1", "m2", "chi1", "chi2", "dL", "t_c", "phase", "ra", "dec", "theta_jn", "psi"]
    for (i, name) in enumerate(row_names)
        @test isapprox(dparams[i, :], fd[i, :]; rtol=1e-4, atol=1e-6)
    end
end

# Multi-detector IMRPhenomXAS fused-kernel Enzyme gradient (Phase 2b Task 5): validated STANDALONE,
# not included in this committed suite, matching Task 2's already-established precedent for this
# exact file (its own relative-binning IMRPhenomXAS Enzyme test is also standalone-only, see
# DEVELOPMENT_PLAN.md Phase 2b) -- this machine previously had only ~6GB RAM (upgraded to 14GB on
# 2026-09-18) and reproducibly hit `LLVM ERROR: out of memory` under the old constraint when
# compiling too many heavy Enzyme.autodiff call graphs back-to-back within a single `Pkg.test()`
# process; this has NOT yet been re-tested at 14GB, so this remains standalone-only for now pending
# re-validation. The multi-detector IMRPhenomXAS fused kernel (the heaviest kernel body in
# this codebase: two-join-point XAS scalar core INSIDE a `for d in 1:ndet` detector loop) is
# exactly the kind of addition that risks tipping it over, on top of the suite's other already-
# accumulated Enzyme compiles (single-detector IMRPhenomD/IMRPhenomXAS fused kernels, IMRPhenomD relative
# binning, the GW smoke test, the multi-detector IMRPhenomD fused kernel added to test_enzyme_ka.jl
# by this same task).
#
# Standalone validation (same tiny problem size/tolerance pattern as every other Enzyme testset in
# this file, `rtol=1e-4, atol=1e-6`, H1+L1, `ntime=16, nwalkers=2`): PASS, ~254s to compile+run.
# See DEVELOPMENT_PLAN.md Phase 2b Task 5 for the exact numbers and how to rerun this
# (`_fused_waveform_likelihood_kernel_phenomxas_multi!` in KernelFused.jl, same
# `set_runtime_activity` requirement as the multi-detector IMRPhenomD testset in test_enzyme_ka.jl).
