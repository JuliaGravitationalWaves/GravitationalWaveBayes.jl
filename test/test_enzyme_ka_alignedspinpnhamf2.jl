using Test
using Enzyme
using GravitationalWaveBayes
using KernelAbstractions
using GravitationalWaveforms: AlignedSpinPNHamF2

@testset "Enzyme gradient through the fused KernelAbstractions kernel (AlignedSpinPNHamF2)" begin
    # Same tiny problem size (and same rationale) as the IMRPhenomD/IMRPhenomXAS testsets in
    # test_enzyme_ka.jl/test_enzyme_ka_xas.jl. AlignedSpinPNHamF2
    # (GravitationalWaveforms.jl's closed-form SPA sibling of AlignedSpinPNHam) has its own non-allocating scalar core
    # (_asphf2_phi_coeffs/_asphf2_phi_scalar/_asphf2_ampl_coeffs/_asphf2_ampl_scalar) and a
    # dedicated fused-kernel body (_fused_waveform_likelihood_kernel_alignedspinpnhamf2! in
    # KernelFused.jl), mirroring the fix already landed for IMRPhenomD/IMRPhenomXAS.
    #
    # `params` is (11, nwalkers), the full physical row convention -- see
    # test_enzyme_ka.jl's matching testset docstring for why the full 11-row space (not just the
    # original 4) is what's actually being validated here.
    #
    # Frequency band (20-35 Hz) is well below this model's hard Schwarzschild-ISCO cutoff (~80 Hz
    # for the ~55 Msun total mass used here -- see the "AlignedSpinPNHamF2" testsets in
    # test_relative_binning.jl for the exact reasoning), so no `fCut`-related discontinuity sits
    # inside the tiny finite-difference perturbations used below.
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

    function sum_ll_kernel_asphf2(params, freqs, cfg, data)
        ll = zeros(eltype(params), size(params, 2))
        batched_log_likelihood_kernel!(ll, params, freqs, cfg, data; model = AlignedSpinPNHamF2())
        return sum(ll)
    end

    dparams = zeros(Float64, size(params))
    Enzyme.autodiff(Enzyme.Reverse, sum_ll_kernel_asphf2,
        Enzyme.Duplicated(params, dparams),
        Enzyme.Const(freqs), Enzyme.Const(cfg), Enzyme.Const(data))

    h = 1e-6
    fd = zeros(Float64, size(params))
    for i in eachindex(params)
        p_plus = copy(params); p_plus[i] += h
        p_minus = copy(params); p_minus[i] -= h
        fd[i] = (sum_ll_kernel_asphf2(p_plus, freqs, cfg, data) - sum_ll_kernel_asphf2(p_minus, freqs, cfg, data)) / (2h)
    end

    @test isapprox(dparams, fd; rtol=1e-4, atol=1e-6)

    row_names = ["m1", "m2", "chi1", "chi2", "dL", "t_c", "phase", "ra", "dec", "theta_jn", "psi"]
    for (i, name) in enumerate(row_names)
        @test isapprox(dparams[i, :], fd[i, :]; rtol=1e-4, atol=1e-6)
    end
end
