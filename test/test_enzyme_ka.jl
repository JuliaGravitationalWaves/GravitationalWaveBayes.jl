using Test
using Enzyme
using GravitationalWaveBayes
using KernelAbstractions
using GravitationalWaveData

@testset "Enzyme gradient through the fused KernelAbstractions kernel (IMRPhenomD)" begin
    # Same tiny problem size that made Enzyme.autodiff hang indefinitely (>5min, confirmed
    # twice) when the kernel body called the array-in/array-out Ampl/Phi. With the IMRPhenomD
    # scalar-core kernel (_fused_waveform_likelihood_kernel_phenomd!, using _phi_coeffs/
    # _phi_scalar/_ampl_coeffs/_ampl_scalar), the same autodiff call completes in a few minutes.
    #
    # `params` is now (11, nwalkers) -- the full physical row convention documented at the top
    # of KernelFused.jl (m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi). This is
    # the highest-risk validation: if Enzyme silently zeroed or mis-computed the
    # gradient for any of rows 5-11 (most plausibly the antenna-response/plus-cross rows 8-11,
    # ra/dec/theta_jn/psi), NUTS's dynamics for exactly those parameters would be corrupted.
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

    function sum_ll_kernel(params, freqs, cfg, data)
        ll = zeros(eltype(params), size(params, 2))
        batched_log_likelihood_kernel!(ll, params, freqs, cfg, data)
        return sum(ll)
    end

    dparams = zeros(Float64, size(params))
    Enzyme.autodiff(Enzyme.Reverse, sum_ll_kernel,
        Enzyme.Duplicated(params, dparams),
        Enzyme.Const(freqs), Enzyme.Const(cfg), Enzyme.Const(data))

    h = 1e-6
    fd = zeros(Float64, size(params))
    for i in eachindex(params)
        p_plus = copy(params); p_plus[i] += h
        p_minus = copy(params); p_minus[i] -= h
        fd[i] = (sum_ll_kernel(p_plus, freqs, cfg, data) - sum_ll_kernel(p_minus, freqs, cfg, data)) / (2h)
    end

    @test isapprox(dparams, fd; rtol=1e-4, atol=1e-6)

    # Per-row breakdown so a regression in just the NEW extrinsic rows (5-11) doesn't get masked
    # by the aggregate isapprox above (which could in principle pass even with one bad row, if
    # the other rows' larger magnitudes dominate a matrix-wide tolerance check in some future
    # refactor -- being explicit here costs nothing and documents exactly what was checked).
    row_names = ["m1", "m2", "chi1", "chi2", "dL", "t_c", "phase", "ra", "dec", "theta_jn", "psi"]
    for (i, name) in enumerate(row_names)
        @test isapprox(dparams[i, :], fd[i, :]; rtol=1e-4, atol=1e-6)
    end
end

@testset "Enzyme gradient near the dec/theta_jn coordinate boundaries" begin
    # The task's explicit worry: a branch/ifelse near a coordinate singularity like dec = +-pi/2
    # could make Enzyme silently produce a wrong (or NaN/Inf) gradient. This kernel operates on
    # PHYSICAL dec/theta_jn directly (not the sin_dec/cos_theta_jn NUTS reparametrization -- that
    # layer's own boundary behavior is separately covered by
    # test_batched_nuts_gw_smoke.jl's "sin_dec/cos_theta_jn near the reparametrization boundary"
    # testset), so this specifically isolates whether the antenna-response/plus-cross scalar math
    # itself (compute_antenna_response's cos/sin, the plus/cross pfac/cfac trig) stays
    # well-behaved at dec's actual physical poles.
    ntime = 16
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    strain = zeros(ComplexF64, ntime)
    psd = ones(Float64, ntime)
    data = LikelihoodData(freqs, strain, psd)
    cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)

    function sum_ll_kernel1(params, freqs, cfg, data)
        ll = zeros(eltype(params), size(params, 2))
        batched_log_likelihood_kernel!(ll, params, freqs, cfg, data)
        return sum(ll)
    end

    for (dec0, theta_jn0) in [(pi/2 - 1e-6, 1e-6), (-pi/2 + 1e-6, pi - 1e-6), (0.0, pi/2)]
        params = reshape([32.0, 28.0, 0.1, -0.05, 0.6, 0.01, 0.5, 2.0, dec0, theta_jn0, 1.0], 11, 1)
        dparams = zeros(Float64, size(params))
        Enzyme.autodiff(Enzyme.Reverse, sum_ll_kernel1,
            Enzyme.Duplicated(params, dparams),
            Enzyme.Const(freqs), Enzyme.Const(cfg), Enzyme.Const(data))
        @test all(isfinite, dparams)

        h = 1e-6
        fd = zeros(Float64, size(params))
        for i in eachindex(params)
            p_plus = copy(params); p_plus[i] += h
            p_minus = copy(params); p_minus[i] -= h
            fd[i] = (sum_ll_kernel1(p_plus, freqs, cfg, data) - sum_ll_kernel1(p_minus, freqs, cfg, data)) / (2h)
        end
        @test isapprox(dparams, fd; rtol=1e-3, atol=1e-5)
    end
end

@testset "Enzyme gradient through the MULTI-DETECTOR fused kernel (IMRPhenomD, H1+L1)" begin
    # The highest-risk multi-detector validation step, same rationale as the single-detector
    # testset above: if Enzyme silently zeroed or mis-computed the gradient for `ra`/`dec`
    # flowing through the `_time_delay_scalar` term (on top of the
    # already-validated `_antenna_response_scalar` term), NUTS's dynamics for exactly those
    # parameters would be corrupted for any multi-detector run. `data` is now
    # `Vector{LikelihoodData}` (H1 then L1), and `batched_log_likelihood_kernel!` dispatches to
    # `_fused_waveform_likelihood_kernel_phenomd_multi!` accordingly.
    ntime = 16
    nwalkers = 2
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    strain = zeros(ComplexF64, ntime)
    psd = ones(Float64, ntime)
    data = [LikelihoodData(freqs, strain, psd, GravitationalWaveData.H1, 1126259462.4),
            LikelihoodData(freqs, strain, psd, GravitationalWaveData.L1, 1126259462.4)]
    cfg = PEConfig{Float64}(ntime, nwalkers, 0.01, 1)

    params = zeros(Float64, 11, nwalkers)
    params[1, :] .= [30.0, 40.0]
    params[2, :] .= [25.0, 20.0]
    params[3, :] .= [0.5, -0.2]
    params[4, :] .= [0.3, 0.1]
    params[5, :] .= [0.4, 1.1]
    params[6, :] .= [0.02, -0.03]
    params[7, :] .= [0.3, 2.0]
    params[8, :] .= [1.2, 4.0]
    params[9, :] .= [-0.5, 0.3]
    params[10, :] .= [1.0, 2.0]
    params[11, :] .= [0.7, 2.5]

    function sum_ll_kernel_multi(params, cfg, data)
        ll = zeros(eltype(params), size(params, 2))
        batched_log_likelihood_kernel!(ll, params, cfg, data)
        return sum(ll)
    end

    # `set_runtime_activity` is required here (NOT for the single-detector testset above) --
    # see `BatchedNUTS._enzyme_reverse_mode`'s docstring/comment for the full root-cause analysis:
    # multi-detector's host-side `for`/`ntuple` loop over `data::Vector{LikelihoodData}` (building
    # per-detector kernel-argument tuples) triggers a genuine `EnzymeRuntimeActivityError` under
    # plain `Enzyme.Reverse` (confirmed empirically), because `Vector{Float64}` is both the
    # provably-Const type read out of `data` here AND the type of the genuinely-active `ll`
    # elsewhere in this same differentiated program. This is Enzyme's own documented,
    # correctness-preserving fix for that diagnostic (suggested directly in the error message),
    # not a precision/correctness compromise -- confirmed below by the same finite-difference
    # tolerance used everywhere else in this file.
    dparams = zeros(Float64, size(params))
    Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse), sum_ll_kernel_multi,
        Enzyme.Duplicated(params, dparams),
        Enzyme.Const(cfg), Enzyme.Const(data))

    h = 1e-6
    fd = zeros(Float64, size(params))
    for i in eachindex(params)
        p_plus = copy(params); p_plus[i] += h
        p_minus = copy(params); p_minus[i] -= h
        fd[i] = (sum_ll_kernel_multi(p_plus, cfg, data) - sum_ll_kernel_multi(p_minus, cfg, data)) / (2h)
    end

    @test isapprox(dparams, fd; rtol=1e-4, atol=1e-6)

    row_names = ["m1", "m2", "chi1", "chi2", "dL", "t_c", "phase", "ra", "dec", "theta_jn", "psi"]
    for (i, name) in enumerate(row_names)
        @test isapprox(dparams[i, :], fd[i, :]; rtol=1e-4, atol=1e-6)
    end
end
