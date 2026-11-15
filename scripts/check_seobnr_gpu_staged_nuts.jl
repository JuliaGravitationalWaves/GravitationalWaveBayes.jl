# Real (not toy) GPU-staged NUTS smoke test for `BatchedNUTS.gw_logp_and_grad_gpu_staged`.
# `positions`/`grad`/`logp`/`chain` all stay ordinary HOST arrays throughout -- exactly what
# `NUTSCore.nuts_sample!`/`nuts_draw!` already allocate and index scalar-per-walker -- while the
# actual likelihood-and-gradient evaluation each leapfrog step runs on whatever device
# `GravitationalWaveBayes.GPU.to_device`/`compatible_array` resolve to (real CUDA here, once loaded+functional).
# Mirrors test/test_batched_nuts_seobnrv5alignedspinf2_smoke.jl's fixture exactly (same reference
# system, same frequency band inside the validated post-adiabatic table range) but swaps
# `gw_logp_and_grad` for `gw_logp_and_grad_gpu_staged` -- this is the first real end-to-end check
# that a genuine NUTS chain (not just a single gradient evaluation) survives with the likelihood
# staged onto real GPU hardware every step.

using GravitationalWaveBayes
using GravitationalWaveforms: SEOBNRv5AlignedSpinF2
using GravitationalWaveBayes.BatchedNUTS: gw_logp_and_grad_gpu_staged, _m1_m2_to_mc_q
using CUDA
using Random

CUDA.functional() || error("CUDA not functional -- stopping")

ntime = 8
freqs = collect(20.5:0.25:(20.5 + 0.25 * (ntime - 1)))
strain = zeros(ComplexF64, ntime)
psd = ones(Float64, ntime)
data = LikelihoodData(freqs, strain, psd)
# `:cpu` here is correct, not a mistake -- `cfg`/`state.positions` stay host-resident throughout;
# only `gw_logp_and_grad_gpu_staged`'s own internal closure state (built once, see its docstring)
# is device-resident.
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

println("=== Building GPU-staged closure ===")
logp_and_grad! = gw_logp_and_grad_gpu_staged(data, cfg, freqs; model=SEOBNRv5AlignedSpinF2(),
                                              nparams=11, nwalkers=nwalkers)

nuts_cfg = NUTSConfig{Float64}(; max_treedepth=3, target_accept=0.8, nwarmup=2)
step_size0 = fill(1e-4, nwalkers)
rng = Random.Xoshiro(1)

println("=== Running 3-step NUTS chain, likelihood staged to CUDA each leapfrog call ===")
t0 = time()
try
    result = nuts_sample!(positions, logp_and_grad!; nsteps=3, step_size0=step_size0,
        cfg=nuts_cfg, rng=rng, record=true)
    elapsed = time() - t0
    println("GPU-staged NUTS chain completed in $(elapsed) s")

    ok_shape = size(result.positions) == (11, nwalkers)
    ok_finite_pos = all(isfinite, result.positions)
    ok_finite_logp = all(isfinite, result.logp)
    println("shape ok: ", ok_shape)
    println("positions finite: ", ok_finite_pos)
    println("logp finite: ", ok_finite_logp)
    println("final positions: ", result.positions)
    println("final logp: ", result.logp)
    println("final step_size: ", result.step_size)

    if ok_shape && ok_finite_pos && ok_finite_logp
        println("PASSED")
    else
        println("FAILED (non-finite or wrong-shape output)")
    end
catch e
    println("GPU-staged NUTS chain FAILED after ", round(time() - t0, digits=1), "s")
    showerror(stdout, e, catch_backtrace())
    println()
end

println("\nDONE")
