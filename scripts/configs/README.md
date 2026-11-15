 # Run configs

TOML configs for `scripts/run_from_config.jl`. Run with:

```
julia --project=. scripts/run_from_config.jl scripts/configs/<name>.toml
```

## Fields

### `[event]`
Shared defaults for every detector (each can be overridden per-detector, see below).

| key | default | meaning |
|---|---|---|
| `trigger_time` | *required* | GPS seconds of the event trigger |
| `duration` | `4.0` | analysis window length, seconds |
| `post_trigger` | `2.0` | seconds of the window after the trigger |
| `psd_segment` | `1024.0` | seconds of off-source data used for the PSD estimate |
| `f_min` / `f_max` | `20.0` / `1024.0` | analysis band, Hz |
| `alpha` | `0.2` | Tukey window parameter |
| `psd_method` | `"median"` | `"median"` or `"welch"` |
| `window_power_correction` | `false` | `true`: PSD x Tukey window factor mean(w^2), the bilby <= 2.5 convention (GWTC-2.1 / GWTC-3 bilby releases); bilby >= 2.6 dropped it (`BILBY_INCORRECT_PSD_NORMALIZATION` restores it there) |

### `[[detectors]]`
One table per detector -- any number, one to N. Required: `name` (`"H1"`, `"L1"`, or `"V1"`) and
`strain_path` (a GWOSC-format HDF5 strain file). Any `[event]` key may also be set here to
override that detector's own value (e.g. a different `f_max` for one interferometer). Optional
`psd_file`: a two-column (frequency, PSD) text file, e.g. a PE-release PSD, used instead of the
off-source estimate (`DataIO.read_psd_file`).

### `[model]`
`name`: one of `IMRPhenomD`, `IMRPhenomXAS`, `IMRPhenomXP`, `IMRPhenomXHM`, `IMRPhenomXPHM`,
`SEOBNRv5HM`, `SEOBNRv5PHM`, `SEOBNRv5AlignedSpinF2`, `SEOBNRv5IMRAlignedSpinF2`,
`AlignedSpinPNHamF2`. `IMRPhenomXP`/`IMRPhenomXPHM`/`SEOBNRv5PHM` sample 15 parameters
(add in-plane spins `chi1x,chi1y,chi2x,chi2y`); everything else samples 11 (aligned-spin only).
`SEOBNRv5AlignedSpinF2` is EOB-track, inspiral only (its table ends at r = 10 M).
`SEOBNRv5IMRAlignedSpinF2` carries the same dynamics through plunge, NQC and the SEOBNRv5 merger-ringdown
(SPA + FFT frequency domain) -- see `DEVELOPMENT_PLAN.md`. (The cheap EOB-lite stand-ins
`EOBLiteAlignedSpinF2`/`EOBLiteIMRAlignedSpinF2`, their `*eoblite*` configs and the SEOBNRv5 injection configs
built on an EOB-lite pass 1 were retired 2026-10-29.)

### `[sampler]`
`type`: `"nuts"` (independent walkers) or `"pt"` (parallel tempering). With `type = "nuts"`,
`method = "smc"` runs adaptive tempered SMC instead of NUTS warmup + sampling (see
[SMC](#smc-samplermethod--smc) below).

| key | applies to | default | meaning |
|---|---|---|---|
| `nwalkers` | nuts | `8` | independent chains |
| `ntemps` | pt | `4` | temperature rungs (one lane each) |
| `nsteps` | both | `200` | NUTS draws |
| `nwarmup` | both | `nsteps÷2` | dual-averaging step-size adaptation draws |
| `step_size` | both | `1e-3` | initial leapfrog step size |
| `max_treedepth` | both | `10` | NUTS max tree depth |
| `adapt_mass` | both | `false` | adapt a diagonal (per-parameter) mass matrix during warmup, Stan windowed schedule; needs `nwarmup >= 20`. For `pt` the metric is attached to the temperature rank, not the lane. nuts: incompatible with `resume=true` (metric is not checkpointed) |
| `dense_mass` | nuts | `false` | with `adapt_mass=true`: adapt a DENSE (full-covariance) metric instead of a diagonal one. Needed for correlated posteriors (all GW ones): a diagonal metric leaves the step size pinned by the narrowest correlated direction. Not for `pt`. |
| `hessian_init` | nuts | `false` | with `dense_mass=true`: before warmup, bootstrap the metric from a finite-difference Hessian of the gradient (2×nparams batched gradient evals) and pick a step size for it. Fixes the "tiny `step_size`, walkers freeze, metric learns the frozen jitter" failure (2026-09-20 GW150914 run). |
| `hessian_rel_step` | nuts | `1e-6` | central-difference step for `hessian_init`, as a fraction of each parameter's box width |
| `mass_init_buffer` / `mass_base_window` / `mass_term_buffer` | nuts | `75` / `25` / `50` | warmup schedule of the metric adaptation (Stan's). With `hessian_init` a short init buffer (~10) is appropriate. |
| `step_size_max_change` | nuts | `8` | with `dense_mass=true`: caps how far a single post-metric-change step-size search (`_find_reasonable_step_size!`) may move `step_size` from its value going in, in either direction. Found needed 2026-09-23/24: a single test leapfrog's accept/reject can drift unluckily for several doublings in a row right after a metric change, overshooting 30-45x before the search's own crossing criterion fires, which then costs many expensive draws of dual-averaging correction (and a divergence-rate spike) to recover from. Must be `>= 1`. |
| `mass_blocks` | nuts | none (full dense) | with `dense_mass=true`: restrict the adapted metric to a block-diagonal structure, e.g. `mass_blocks = [[1,2,6,7], [3,4]]` (1-indexed rows into the sampled parameter vector) to group `mc`/`q`/`t_c`/`phase` separately from `chi1`/`chi2` -- fewer entries to estimate from few walkers than full dense, at zero cost to correctness (rows not mentioned are their own independent singleton). |
| `quorum_frac` | nuts | `0` | once fewer than `ceil(quorum_frac * nwalkers)` walkers remain active mid-draw, stop the whole tree-doubling round early instead of deepening it for every walker until the LAST one finishes (100% of draws hitting `max_treedepth` was the dominant real-run cost, 2026-09-19). `0` (default) reproduces the old behaviour exactly. E.g. `0.75` with 16 walkers stops once fewer than 12 remain active. Applies to both the warmup and the frozen sampling phase. |
| `seed` | both | none (default RNG) | integer seed for reproducibility |
| `enforce_bounds` | both | `true` | reject proposals outside the prior box |
| `beta_min` | pt | `1e-2` | coldest-to-hottest temperature ladder floor |
| `swap_interval` | pt | `5` | draws between temperature-swap attempts |
| `resume` | nuts | `false` | continue from `[output].dir`'s checkpoint instead of restarting from scratch |
| `device` | nuts | `"cpu"` | `"cpu"` or `"cuda"` (the environment must load `CUDA.jl`; which models have device paths is decided in `run_from_config.jl`) |
| `method` | nuts | `"nuts"` | `"nuts"` or `"smc"` (see [SMC](#smc-samplermethod--smc)) |
| `gradient_mode` | nuts | `"auto"` | PAF2 models: `"auto"`, `"forward"` or `"reverse"` Enzyme mode for the SEOBNR-path gradient (`"reverse"` is rejected with relative binning) |
| `pt_ntemps_max` | nuts (CPU) | `12` | cap on the rung count when `pt_ntemps = "auto"` |
| `relative_binning_rebuild_at` | nuts (IMRPhenomXAS on CUDA) | `[10, 25, 50, 100, nwarmup÷2, 3nwarmup÷4]` (those < `nwarmup`) | warmup draws at which the relative-binning summary data are rebuilt around the best walker (frozen for sampling) |
| `relative_binning` | nuts | `false` | two-phase dense-warmup-then-binned run (see below) |
| `relative_binning_nbins` | nuts | `150` | number of relative-binning bins |
| `relative_binning_max_tc_deviation` | nuts | `0.0` | passed through to `build_relative_binning` |
| `relative_binning_max_extrinsic_phase` | nuts | `0.0` | passed through to `build_relative_binning` |
| `init` | nuts | `"gw150914"` | walker start: `"gw150914"` (fixed GW150914-scale point + jitter), `"injection"` (the `[injection]` truth + `init_scale` jitter), `"prior"` (exact prior draws) |
| `jumps` | nuts (CPU) | `false` | Metropolis jump moves between NUTS draws for the discrete GW degeneracies (`gw_jump_moves`: inclination flip; for two detectors, the time-delay-preserving sky reflection and their composition). Exact (involutive, unit-Jacobian proposals). |
| `de_jumps` | nuts (CPU) | `false` | differential-evolution mode-hopping move between NUTS draws: each walker proposes `x + (x_b - x_c)` from two walkers of the other half of the ensemble (exact, complementary-ensemble construction). Lets walkers reach separated modes other walkers have found (e.g. low-SNR chirp-mass cycle-slip peaks). Needs >= 6 walkers. |
| `pt_ntemps` | nuts (CPU) | `1` | parallel tempering with `nwalkers` walkers PER temperature (total `nwalkers * pt_ntemps`); geometric ladder from 1 down to `pt_beta_min` (default `0.05`); positions exchanged between adjacent rungs every `pt_swap_interval` (default 1) draws; metric adapted from the beta=1 walkers only; only beta=1 walkers are written out. Combines with `jumps`/`de_jumps`. |
| `pt_beta_min` / `pt_ntemps` = `"auto"` | nuts (CPU) | - | ladder from the reference waveform: `beta_min = d/SNR^2` (d = non-marginalized sampled parameters; SNR = the injection's optimal network SNR, else the amplitude/phase-maximized matched-filter SNR at the best starting walker, PAF2 models), `ntemps = ceil(2*Lambda)+1` from the Gaussian communication barrier `Lambda = sqrt(d/2pi) log(1/beta_min)`, capped by `pt_ntemps_max` (default 12). |
| `pt_adapt` | nuts (CPU) | `true` | re-space the ladder during warmup (at 1/8, 1/4, 1/2, 3/4 of `nwarmup`) so every adjacent pair has the same swap rejection rate (Syed et al. 2021); ends fixed; frozen after warmup. Needs `pt_ntemps >= 3`. |
| `pt_deo` | nuts (CPU) | `true` | deterministic even/odd swap schedule (non-reversible PT): alternate rounds attempt pairs (1,2),(3,4),... then (2,3),(4,5),...; `false` = every pair every round. |
| `init_scale` | nuts | `0.01` | with `init="injection"`: per-row Gaussian jitter as a fraction of that row's prior width |

### `[likelihood]` (optional)
Every model except `AlignedSpinPNHamF2`. The physics of each option is in `PEConfig`'s docstring
(`src/GravitationalWaveBayes/Types.jl`).

| key | default | meaning |
|---|---|---|
| `marginalize_phase` | `false` | analytically marginalize the reference phase (`log I0(|<d,h>|)`). Dominant-mode models only (not the precessing or higher-mode ones). The phase row then carries prior draws during sampling and is replaced by exact conditional-posterior draws (`reconstruct_marginalized!`) before the chain is written. |
| `marginalize_distance` | `false` | analytically marginalize the luminosity distance over the `[prior]` `dL` range and `distance` shape (Gauss-Legendre quadrature); the dL row is redrawn afterwards like the phase |
| `time_reference` | `"geocenter"` | multi-detector only: frame of the sampled `t_c` -- `"geocenter"` or a detector name (e.g. `"H1"`, bilby's `time_reference`) |
| `time_ref_frequency` | `0` (off) | Hz: sample the arrival time at which the signal passes this frequency instead of the coalescence time |
| `time_ref_offset` | `"auto"` | s, with `time_ref_frequency`: constant added to the sampled time; `"auto"` centres it on a reference point (the injection truth, else the prior box centre with zero spins) |
| `fmax_taper` | `0` (off) | Hz: C1 taper of the PAF2 models' amplitude below their table end frequency (removes logL jumps when `fmax` crosses a bin) |
| `phase_ref_frequency` | `0` (lowest data frequency) | Hz: reference frequency of the sampled phase (and, for SEOBNRv5PHM, the spins and inclination frame). SEOBNRv5PHM, SEOBNRv5HM, and IMRPhenomXHM on the CUDA path |

### `[prior]` (optional)
Bounds overrides `mc`, `q`, `chi1`, `chi2`, `dL`, `t_c` as `[lo, hi]` (sampled coordinates; `t_c` in
the `time_reference` frame); `distance = "uniform" | "volume"` (∝ dL²) `| "source_frame"` (bilby
`UniformSourceFrame`, Planck15_LAL: sampled under dL², then importance weights per chain sample in
`prior_weights.jls` and `logZ_source_frame` in `smc_summary.txt`; needs `output.record_chain`);
`masses = "uniform_mc_q" | "components"` (uniform in detector-frame m1, m2); `m1`, `m2 = [lo, hi]`
(detector-frame component-mass cuts, as bilby's `Constraint`s); `spins = "ball"` (default) or
`"isotropic"` (the LVK isotropic spin prior, magnitudes up to `spin_max` = one cap or per body
`[a1, a2]`, default `0.99`; for aligned-spin models its z-projection,
p(chi) = log(a / |chi|) / (2a)). Without this section the historical flat box is used.

### `[calibration]` (optional; needs `[prior]`)
Detector calibration uncertainty, bilby's model: `priors` is the path of a bilby prior file (one
`key = "value"` per line: `recalib_<det>_frequency_k`, `recalib_<det>_amplitude_k`,
`recalib_<det>_phase_k`, as extracted from a PE release). `mode = "sample"` (default) adds the
spline node values as sampled rows; `mode = "marginalize"` averages the likelihood over `ncurves`
(default `1000`) prior response curves drawn with `curve_seed` (default `1`) instead -- SEOBNRv5PHM
and IMRPhenomXPHM only.

### `[injection]` (optional; PAF2 models and SEOBNRv5PHM)
Replaces each detector's strain by the model signal plus stationary Gaussian noise drawn from that
detector's real-data PSD (`simulate_injection`). Keys: `mc, q, chi1, chi2, dL, t_c, phase, ra, dec,
theta_jn, psi` (angles in radians, `t_c` geocentric for multi-detector data), `noise` (default
`true`), `seed`. Writes `injection.txt` (SNRs) and `injection_sampled_truth.csv` (truth in the sampled
coordinates/time frame) to the output dir.

### SMC (`sampler.method = "smc"`)
Adaptive tempered SMC from the prior (`init = "prior"`) to the posterior with `nwalkers` particles,
then `smc_nfinal` recorded draws per particle at beta = 1. Every `sampler.smc_<name>` key is passed
through as the `<name>` keyword of `GravitationalWaveBayes.smc_sample` (`src/GravitationalWaveBayes/SMCCore.jl`), whose docstring
documents them; the list of accepted keys is in `run_from_config.jl`. The ones the current best
configs set:

| key | meaning |
|---|---|
| `smc_target_cess` | conditional ESS fraction kept per tempering step (sets the beta ladder) |
| `smc_nfinal` | recorded draws per particle at beta = 1 |
| `smc_moves`, `smc_nuts_beta` | `"nuts"` (default) or `"rw"`: random-walk (likelihood-only) mutations; with `smc_nuts_beta`, a hybrid that switches to NUTS from that beta on |
| `smc_max_mutate` | cap on mutation draws per stage |
| `smc_online_kde`, `smc_online_flow`, `smc_flow_*` | independence proposals from a KDE / normalizing flow fitted to the current particles |
| `smc_reference = "<run dir>"` | anneal from a density fitted to a previous run's chain (multi-fidelity) instead of the prior |
| `smc_final_ess` | go straight to beta = 1 once the incremental weights' ESS fraction there reaches this |
| `smc_final_max_treedepth` | NUTS tree-depth cap of the recorded beta = 1 phase |

`scripts/configs/gw150914_h1l1_seobnrv5hm_speed_jaxprior_best2_s44_cuda.toml` is a complete
example.

### P-P (calibration) tests
`scripts/pp_test.jl base.toml N outdir` draws `N` truths from the base config's `[prior]`, injects
each and runs the sampler, appending the truth's posterior percentile per parameter to
`outdir/pp_results.csv` (restartable). `julia scripts/pp_summary.jl outdir/pp_results.csv` prints
per-parameter KS p-values against Uniform(0,1). Walkers start around the truth (`init_scale`), so a
sampler that fails to move would show as percentiles clustered near 0.5 -- the test still catches it.

### `[output]`
`dir`: where results are written (created if missing). `record_chain` (nuts only, default
`true`): keep the full `(nparams, nwalkers, nsteps)` chain and compute Rhat/ESS diagnostics --
set `false` to skip if you only want the final state.

## What gets written

- `config_used.toml` -- a copy of the input config, for reproducibility.
- `final_positions.csv` / `final_logp.csv` -- last draw of every walker.
- nuts + `record_chain=true`: `chain.jls` (full chain, `Serialization`-readable as
  `(; names, chain)`), `diagnostics.txt` (per-parameter Rhat/ESS).
- pt: `cold_chain.csv` (the beta=1 chain, one posterior draw per step) and
  `pt_swap_acceptance.csv` (per-adjacent-pair swap acceptance rate).

## Checkpointing & resume

Every draw, `[output].dir` gets `checkpoint_positions.csv`/`checkpoint_step_size.csv`/
`checkpoint_meta.txt` overwritten with the latest state (a multi-hour real-data run has no other
recovery path if the process is killed partway). Set `sampler.resume = true` and re-run the SAME
config (same `dir`, same `nsteps`/`nwalkers`) to continue from there instead of restarting --
draws already completed are skipped, and if the checkpoint was taken mid-warmup, dual averaging
re-adapts over the remaining warmup draws rather than resuming its internal state exactly (see
`_apply_resume!`'s docstring in `scripts/run_from_config.jl`). Works for both plain and
relative-binning `nuts` runs, CPU or GPU.

For `relative_binning = true` runs specifically (two `nuts_sample!` calls -- dense warmup, then
binned sampling from a reference point derived from warmup's own result): resuming mid-warmup
re-runs the remaining warmup draws then proceeds through the binned phase fresh from that result;
resuming mid-binned-phase skips re-running warmup entirely and rebuilds the identical binned kernel
from a `relative_binning_ref.csv` persisted the moment warmup completed (see
`_run_nuts_relative_binning`'s docstring for the full case breakdown, including the narrow
warmup-just-finished boundary case).

## Relative binning (`sampler.relative_binning = true`)

Wired up as of 2026-09-18: set `relative_binning = true` under `[sampler]` and the run becomes
two `nuts_sample!` calls -- a dense-grid warmup phase, whose final mean walker position becomes
the physical-space reference point `RelativeBinning.build_relative_binning` needs, then a second,
much cheaper binned-likelihood phase from that reference. No manual reference point is needed;
it's derived from the run's own warmup, not a placeholder. See `_run_nuts_relative_binning`/
`_run_nuts_gpu_relative_binning` in `scripts/run_from_config.jl` for the exact mechanics, and
"Checkpointing & resume" above for how resume interacts with the two phases. `nuts` only (not
`pt`); single-detector only, 11-parameter aligned-spin models only.

Not yet wired up for `[sampler] type = "pt"` -- call `run_gw150914_pt_estimation` directly with
`relative_binning=...` if you need it there.
