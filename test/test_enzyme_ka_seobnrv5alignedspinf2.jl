using Test
using Enzyme
using GravitationalWaveBayes
using KernelAbstractions
using GravitationalWaveforms: SEOBNRv5AlignedSpinF2

@testset "Enzyme gradient through the fused KernelAbstractions kernel (SEOBNRv5AlignedSpinF2)" begin
    # The isolated-scalar-core Enzyme validation in GravitationalWaveforms.jl's tests held the dense post-adiabatic
    # table fixed and only differentiated the reconstruction formula w.r.t. a `coeffs` field --
    # it does NOT prove Enzyme survives composition inside
    # `_fused_waveform_likelihood_kernel_seobnrv5alignedspinf2!` itself, where `_seobv5f2_coeffs`
    # (the two-stage IC Newton solve + `Vector{Float64}`-allocating post-adiabatic driver) runs
    # INSIDE the differentiated call graph, driven by the actual `mc`/`eta`/`chi1`/`chi2` physical
    # parameters (via `_extrinsic_setup`). This testset is that missing check, mirroring
    # test_enzyme_ka_alignedspinpnhamf2.jl's structure exactly (full 11-row Enzyme.Reverse vs.
    # central-finite-difference comparison through `batched_log_likelihood_kernel!`).
    #
    # Frequency band and physical configuration are NOT arbitrary: chosen to sit safely inside the
    # validated regime from test_relative_binning.jl's "SEOBNRv5AlignedSpinF2: binned likelihood
    # matches full-grid likelihood" testset (m1=36/m2=29 solar masses, post-adiabatic table
    # confirmed there to reach `fmax~31.5`Hz from `f_min=20`Hz) -- 20.5-22.5Hz is comfortably
    # inside `[coeffs.fmin, coeffs.fmax]` for both walkers below, including after every `+-1e-6`
    # finite-difference perturbation used to check the gradient (unlike the relative-binning
    # aliasing issue, which is specific to BINNED-vs-dense agreement, not to
    # this dense kernel's own differentiability -- so nonzero `t_c` is fine to use here).
    ntime = 8
    nwalkers = 2
    freqs = collect(20.5:0.25:(20.5 + 0.25 * (ntime - 1)))
    strain = zeros(ComplexF64, ntime)
    psd = ones(Float64, ntime)
    data = LikelihoodData(freqs, strain, psd)
    cfg = PEConfig{Float64}(ntime, nwalkers, 0.01, 1)

    params = zeros(Float64, 11, nwalkers)
    params[1, :] .= [36.0, 36.5]      # m1
    params[2, :] .= [29.0, 29.2]      # m2
    params[3, :] .= [0.0, 0.05]       # chi1
    params[4, :] .= [0.0, -0.05]      # chi2
    params[5, :] .= [0.4, 0.42]       # dL (Gpc)
    params[6, :] .= [0.02, -0.03]     # t_c (s)
    params[7, :] .= [0.5, 0.55]       # phase (rad)
    params[8, :] .= [1.2, 1.25]       # ra (rad)
    params[9, :] .= [-0.4, -0.35]     # dec (rad)
    params[10, :] .= [1.0, 1.05]      # theta_jn (rad)
    params[11, :] .= [0.6, 0.65]      # psi (rad)

    function sum_ll_kernel_seobnrv5f2(params, freqs, cfg, data)
        ll = zeros(eltype(params), size(params, 2))
        batched_log_likelihood_kernel!(ll, params, freqs, cfg, data; model = SEOBNRv5AlignedSpinF2())
        return sum(ll)
    end

    # Sanity check first: both walkers must land inside the post-adiabatic table's valid band
    # (nonzero, finite likelihood) before trusting either the Enzyme or FD result below.
    ll0 = zeros(Float64, nwalkers)
    batched_log_likelihood_kernel!(ll0, params, freqs, cfg, data; model = SEOBNRv5AlignedSpinF2())
    @test all(isfinite, ll0)
    @test all(!iszero, ll0)

    dparams = zeros(Float64, size(params))
    Enzyme.autodiff(Enzyme.Reverse, sum_ll_kernel_seobnrv5f2,
        Enzyme.Duplicated(params, dparams),
        Enzyme.Const(freqs), Enzyme.Const(cfg), Enzyme.Const(data))

    h = 1e-6
    fd = zeros(Float64, size(params))
    for i in eachindex(params)
        p_plus = copy(params); p_plus[i] += h
        p_minus = copy(params); p_minus[i] -= h
        fd[i] = (sum_ll_kernel_seobnrv5f2(p_plus, freqs, cfg, data) - sum_ll_kernel_seobnrv5f2(p_minus, freqs, cfg, data)) / (2h)
    end

    @test isapprox(dparams, fd; rtol=1e-4, atol=1e-6)

    row_names = ["m1", "m2", "chi1", "chi2", "dL", "t_c", "phase", "ra", "dec", "theta_jn", "psi"]
    for (i, name) in enumerate(row_names)
        @test isapprox(dparams[i, :], fd[i, :]; rtol=1e-4, atol=1e-6)
    end
end
