using Test
using GravitationalWaveBayes
using GravitationalWaveforms: SEOBNRv5AlignedSpinF2
using GravitationalWaveBayes.BatchedNUTS: gw_logp_and_grad, _m1_m2_to_mc_q
using Random

@testset "SEOBNRv5AlignedSpinF2 dense-kernel NUTS smoke test" begin
    # The Enzyme-through-fused-kernel validation
    # (test_enzyme_ka_seobnrv5alignedspinf2.jl) proved the gradient itself is correct through
    # `batched_log_likelihood_kernel!`, but not that a real NUTS run (repeated leapfrog steps,
    # each a fresh Enzyme.autodiff call through the SAME kernel, plus the reparametrization/prior
    # machinery around it) survives end-to-end for this model. This is that missing check --
    # mirrors test_batched_nuts_gw_smoke.jl's "runs end-to-end and produces finite output" scope
    # exactly (NOT a convergence/recovery check), using `gw_logp_and_grad(...; model=
    # SEOBNRv5AlignedSpinF2())` directly rather than `run_gw150914_estimation` (which is hardcoded
    # to `IMRPhenomD()`).
    #
    # Deliberately uses `relative_binning=nothing` (the dense kernel only) -- test_relative_binning
    # .jl's "SEOBNRv5AlignedSpinF2: binned likelihood matches full-grid likelihood" testset already
    # documents a real, unfixed aliasing hazard in the binned kernel for nonzero `t_c` with this
    # model's narrow post-adiabatic band: a NUTS chain free to explore `t_c` (row 6) is exactly the case flagged as
    # dangerous with relative binning. Running the AlignedSpinPNHamF2-style "full vs binned NUTS
    # posteriors agree" comparison for THIS model is deliberately NOT attempted here or anywhere
    # else yet -- it needs `_phase_equal_bin_edges` extended for the extrinsic time-shift gradient
    # first.
    #
    # Frequency band (20.5-22.5Hz) sits inside this system's validated post-adiabatic table range
    # (fmax~31.5Hz for m1=36/m2=29, confirmed in test_relative_binning.jl and reused unchanged
    # here) with margin to spare, and step_size/nsteps are small enough that the walkers (started
    # exactly at this reference system) can't leapfrog their way out of that valid band within this
    # short smoke test.
    ntime = 8
    freqs = collect(20.5:0.25:(20.5 + 0.25 * (ntime - 1)))
    strain = zeros(ComplexF64, ntime)
    psd = ones(Float64, ntime)
    data = LikelihoodData(freqs, strain, psd)
    cfg = PEConfig{Float64}(ntime, 2, 0.01, 1)

    nwalkers = 2
    positions = zeros(Float64, 11, nwalkers)
    mc_inj, q_inj = _m1_m2_to_mc_q(36.0, 29.0)
    @views positions[1, :] .= mc_inj
    @views positions[2, :] .= q_inj
    @views positions[3, :] .= 0.0    # chi1
    @views positions[4, :] .= 0.0    # chi2
    @views positions[5, :] .= 0.4    # dL
    @views positions[6, :] .= 0.0    # t_c
    @views positions[7, :] .= 0.5    # phase
    @views positions[8, :] .= 1.2    # ra
    @views positions[9, :] .= sin(-0.4)   # sin_dec
    @views positions[10, :] .= cos(1.0)   # cos_theta_jn
    @views positions[11, :] .= 0.6   # psi

    logp_and_grad! = gw_logp_and_grad(data, cfg, freqs; model = SEOBNRv5AlignedSpinF2())
    nuts_cfg = NUTSConfig{Float64}(; max_treedepth = 3, target_accept = 0.8, nwarmup = 2)
    step_size0 = fill(1e-4, nwalkers)

    rng = Random.Xoshiro(1)
    result = nuts_sample!(positions, logp_and_grad!; nsteps = 3, step_size0 = step_size0,
        cfg = nuts_cfg, rng = rng, record = true)

    @test size(result.positions) == (11, nwalkers)
    @test all(isfinite, result.positions)
    @test all(isfinite, result.logp)
end
