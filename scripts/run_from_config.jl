# Config-driven parameter-estimation runner: takes a TOML file describing an event (one or more
# detectors, a waveform model, and sampler settings) and drives a real `run_gw150914_estimation`/
# `run_gw150914_pt_estimation` run from it.
#
# This is a thin orchestration layer over already-existing, already-validated pieces --
# `DataIO.build_gwdata_multi` (real per-detector data conditioning) and `BatchedNUTS`'s two public
# entry points (both already generic over "1..N detectors", see `Types.DetectorData`'s docstring in
# src/GravitationalWaveBayes/Types.jl) -- NOT a new inference engine. Its only job is: parse TOML -> build a
# `Vector{LikelihoodData}` of whatever length the config's `[[detectors]]` array has -> call the
# right public entry point -> write results to disk. There is deliberately no special-casing on the
# NUMBER of detectors anywhere below: `build_gwdata_multi`/`run_gw150914_estimation`/
# `run_gw150914_pt_estimation` already accept a `Vector{LikelihoodData}` of any length >= 1, so a
# 1-entry `[[detectors]]` array and a 5-entry one go through the exact same code path here.
#
# Usage:
#   julia --project=. scripts/run_from_config.jl path/to/config.toml
#
# See scripts/configs/gw150914_h1l1.toml for a fully worked example (real GW150914 H1+L1 data,
# already present under data/) and scripts/configs/README.md for the full field reference.

using GravitationalWaveBayes
using GravitationalWaveBayes.BatchedNUTS: _gw150914_initial_positions, _sampled_space_bounds, gw_logp_and_grad,
                            _representative_freqs, _reparam_to_physical!, gw_logp_and_grad_gpu_staged,
                            _m1_m2_to_mc_q, _mc_q_to_m1_m2, reconstruct_marginalized!, gw_jump_moves,
                            _raw_loglikelihood!
using GravitationalWaveBayes.NUTSCore: nuts_sample!, NUTSConfig, pt_position_swaps!
using GravitationalWaveBayes: NUTSCore
using GravitationalWaveforms: IMRPhenomXPHM, SEOBNRv5PHM, SEOBNRv5AlignedSpinF2, SEOBNRv5IMRAlignedSpinF2, AlignedSpinPNHamF2
using GravitationalWaveBayes: PAF2Model
using GravitationalWaveBayes.RelativeBinning: build_relative_binning
using GravitationalWaveData: Detector, H1, L1, V1
import GravitationalWaveData
import GravitationalWaveforms
using TOML
using Random
using DelimitedFiles
import LinearAlgebra
using Serialization

# `[sampler] device = "cuda"` (see main()) needs CUDA.jl actually loaded for GravitationalWaveBayesCUDAExt to
# activate -- CUDA is a weakdep (see Project.toml), not a hard dep, so this script stays runnable
# on a machine with no GPU at all as long as the config never asks for device="cuda". Loaded here
# at top level (not lazily inside main()) because `using` inside a function body has real world-age
# footguns in Julia; a plain CPU run pays only the (cheap) cost of CUDA.jl failing to find a device
# at `CUDA.functional()` time, not a hard failure, if CUDA happens to be installed but unusable.
try
    @eval using CUDA
catch e
    @debug "CUDA.jl not available -- device=\"cuda\" configs will error with a clear message" exception=e
end

# ------------------------------------------------------------------------------------------------
# Name -> value lookup tables (config files reference everything by string name).
# ------------------------------------------------------------------------------------------------

const _DETECTORS = Dict{String,Detector}("H1" => H1, "L1" => L1, "V1" => V1)

const _MODELS = Dict{String,Any}(
    "IMRPhenomD" => IMRPhenomD(),
    "IMRPhenomXAS" => IMRPhenomXAS(),
    "IMRPhenomXP" => IMRPhenomXP(),
    "IMRPhenomXHM" => IMRPhenomXHM(),
    "IMRPhenomXPHM" => IMRPhenomXPHM(),
    "SEOBNRv5PHM" => SEOBNRv5PHM(),
    "SEOBNRv5HM" => GravitationalWaveforms.SEOBNRv5HM(),
    "SEOBNRv5AlignedSpinF2" => SEOBNRv5AlignedSpinF2(),
    "SEOBNRv5IMRAlignedSpinF2" => SEOBNRv5IMRAlignedSpinF2(),
    "AlignedSpinPNHamF2" => AlignedSpinPNHamF2(),
)

const _PARAM_NAMES_11 = ["mc", "q", "chi1", "chi2", "dL", "t_c", "phase", "ra", "sin_dec", "cos_theta_jn", "psi"]
const _PARAM_NAMES_15 = vcat(_PARAM_NAMES_11, ["chi1x", "chi1y", "chi2x", "chi2y"])
# default `smc_mode_jump_rows` (those the model samples): masses, spins, time, phase, inclination -- the rows along which
# the GW150914 XPHM bulk and its low-Mc / anti-aligned corner differ (2026-10-29/30 widthcheck runs)
const _MODE_JUMP_ROWS = ["mc", "q", "chi1", "chi2", "t_c", "phase", "cos_theta_jn", "chi1x", "chi1y", "chi2x", "chi2y"]

_lookup(dict, key, kind) =
    haskey(dict, key) ? dict[key] : throw(ArgumentError("unknown $kind \"$key\" -- known: $(sort(collect(keys(dict))))"))

# ------------------------------------------------------------------------------------------------
# Progress logging. Real per-draw timing has repeatedly mattered here (NUTS warmup can visit
# arbitrarily deep trees while dual-averaging searches for a step size, and stdout is fully
# buffered when this script's output is redirected to a log file, so without this a run's own log
# stays completely empty until the process exits or crashes -- both bit us directly on 2026-09-17).
# Uses `nuts_sample!`'s existing generic `callback(it, positions, logp, step_size, stats)` hook
# (added for Phase 3 parallel tempering; `stats` added 2026-09-18 for the "summary"-tier NUTS
# instrumentation, DEVELOPMENT_PLAN.md Phase 6) rather than adding anything NUTSCore-specific.
# ------------------------------------------------------------------------------------------------

function _progress_callback(label::AbstractString, ntotal::Int; every::Int=1,
                             outdir::Union{Nothing,AbstractString}=nothing)
    t_start = time()
    t_last = Ref(t_start)
    recent_move = Float64[]
    return function (it, positions, logp, step_size, stats)
        now = time()
        # `stats.depth`/`stats.n_leapfrog`/`stats.diverged` (per-walker) answer, directly and every
        # draw, the question that previously required inferring tree behavior from draw timing
        # alone (see the 2026-09-18 "Check post-warmup tree depth" TODO in DEVELOPMENT_PLAN.md) --
        # in particular whether NUTS is still maxing out `max_treedepth` well past warmup.
        mean_depth = round(sum(stats.depth) / length(stats.depth), digits=2)
        max_depth = maximum(stats.depth)
        mean_nleap = round(sum(stats.n_leapfrog) / length(stats.n_leapfrog), digits=1)
        ndiverged = count(stats.diverged)
        # `stats.quorum_forced` (NUTSCore, opt-in via `NUTSConfig(quorum_frac=...)`): walkers whose
        # tree-doubling was cut short because too few OTHER walkers were still active, not because
        # they hit max_treedepth or diverged themselves -- administrative, not numerical.
        nquorum = hasproperty(stats, :quorum_forced) ? count(stats.quorum_forced) : 0
        # `stats.disp` (NUTSCore): per-walker distance moved this draw in units of the CURRENT metric's
        # posterior sd -- ~1 is healthy, << 1 is a frozen chain regardless of acceptance rate or depth.
        # Added after the 2026-09-20 run, whose every draw hit max_treedepth with walkers ~1e-4 sd
        # from where they started and nothing in the log said so.
        move = hasproperty(stats, :disp) ? sort(stats.disp)[cld(length(stats.disp), 2)] : NaN
        push!(recent_move, move); length(recent_move) > 10 && popfirst!(recent_move)
        println("  $label $it/$ntotal  step took $(round(now - t_last[], digits=2))s  ",
                "elapsed=$(round(now - t_start, digits=1))s  mean_logp=$(round(sum(logp)/length(logp), digits=3))  ",
                "step_size[1]=$(round(step_size[1], sigdigits=4))  ",
                "depth(mean/max)=$mean_depth/$max_depth  mean_nleap=$mean_nleap  ndiverged=$ndiverged  nquorum=$nquorum  move(median)=$(round(move, sigdigits=3))")
        if length(recent_move) == 10 && it % 10 == 0 && all(isfinite, recent_move) && maximum(recent_move) < 0.05
            println("  WARNING: $label chain looks FROZEN -- median whitened displacement per draw < 0.05 for ",
                    "the last 10 draws (max=$(round(maximum(recent_move), sigdigits=2))); the sampler is not ",
                    "exploring. Check the metric / step size before spending more time on this run.")
        end
        flush(stdout)
        # Overnight/multi-hour runs (this file has none of nuts_sample!'s own checkpointing --
        # _write_outputs only ever runs once, after the WHOLE run returns) previously had no way
        # to recover anything if interrupted partway. Overwritten every draw, so it's always the
        # latest state regardless of which phase (dense-warmup vs binned, etc.) is currently running.
        if outdir !== nothing
            mkpath(outdir)
            nrows = size(positions, 1)
            names = nrows == 15 ? _PARAM_NAMES_15 : _PARAM_NAMES_11
            open(joinpath(outdir, "checkpoint_positions.csv"), "w") do io
                writedlm(io, vcat(reshape(names, 1, nrows), permutedims(positions)), ',')
            end
            writedlm(joinpath(outdir, "checkpoint_logp.csv"), logp, ',')
            writedlm(joinpath(outdir, "checkpoint_step_size.csv"), step_size, ',')
            open(joinpath(outdir, "checkpoint_meta.txt"), "w") do io
                println(io, "label=$label draw=$it/$ntotal elapsed=$(round(now - t_start, digits=1))s ",
                             "mean_depth=$mean_depth max_depth=$max_depth mean_nleap=$mean_nleap ",
                             "ndiverged=$ndiverged nquorum=$nquorum")
            end
        end
        t_last[] = now
        return nothing
    end
end

"""
    _load_checkpoint(outdir) -> Union{Nothing,NamedTuple}

Reads back `checkpoint_positions.csv`/`checkpoint_step_size.csv`/`checkpoint_meta.txt` as written
by `_progress_callback` above, for the "resume-from-checkpoint" TODO in DEVELOPMENT_PLAN.md
(2026-09-18). Returns `nothing` if any of the three files is missing (nothing to resume from --
callers should fall back to a fresh start, not error, since this is also what a first-time run into
a fresh `outdir` looks like). Returns `(; positions, step_size, label, it, ntotal)` otherwise:
`positions` is `(nrows, nwalkers)` (the on-disk file is walkers-as-rows, transposed back here to
match `nuts_sample!`'s own `(nparams, nwalkers)` convention), `label`/`it`/`ntotal` are parsed back
out of `checkpoint_meta.txt`'s `label=... draw=it/ntotal ...` line.

Scope note: only the `label`/`it`/`ntotal` fields needed to resume a SINGLE `nuts_sample!` call are
parsed -- see `_run_nuts`'s/`_run_nuts_gpu`'s `resume` handling for the single-phase case, and
`_run_nuts_relative_binning`'s/`_run_nuts_gpu_relative_binning`'s (which additionally branch on
`label` to tell which of their two phases a killed run was in, and load a separately-persisted
relative-binning reference point -- see [`_load_relative_binning_ref`](@ref) -- when resuming into
the second phase).
"""
function _load_checkpoint(outdir::AbstractString)
    pos_path = joinpath(outdir, "checkpoint_positions.csv")
    step_path = joinpath(outdir, "checkpoint_step_size.csv")
    meta_path = joinpath(outdir, "checkpoint_meta.txt")
    (isfile(pos_path) && isfile(step_path) && isfile(meta_path)) || return nothing

    raw = readdlm(pos_path, ',')
    positions = permutedims(Matrix{Float64}(raw[2:end, :]))  # back to (nrows, nwalkers)
    step_size = vec(Float64.(readdlm(step_path, ',')))

    meta = read(meta_path, String)
    # `(.+?)` (non-greedy), not `\S+` -- some labels are multiple words ("CPU dense-warmup", "GPU
    # binned"; see _run_nuts_relative_binning's label_a/label_b), and `\S+` would stop at the first
    # space and truncate those, silently misparsing the label (caught 2026-09-18 by an actual
    # two-phase relative-binning resume smoke test, not by inspection).
    m = match(r"label=(.+?)\s+draw=(\d+)/(\d+)", meta)
    m === nothing && throw(ErrorException("checkpoint_meta.txt at $meta_path doesn't match the " *
        "expected \"label=... draw=it/ntotal\" format -- can't resume from it"))
    label = String(m.captures[1])
    it = parse(Int, m.captures[2])
    ntotal = parse(Int, m.captures[3])

    return (; positions, step_size, label, it, ntotal)
end

"""
    _apply_resume!(positions, step_size0, outdir, expected_label, nsteps, nwarmup) -> (nsteps_eff, nwarmup_eff)

Loads a checkpoint from `outdir` (via [`_load_checkpoint`](@ref)) and, if one exists, overwrites
`positions`/`step_size0` IN PLACE with the checkpointed state and returns the reduced
`(nsteps, nwarmup)` a caller should pass to `nuts_sample!` to continue from there instead of
restarting. Returns `(nsteps, nwarmup)` UNCHANGED (and leaves `positions`/`step_size0` untouched)
if `outdir` has no checkpoint yet -- resuming into a fresh `outdir` is exactly a fresh start, not
an error.

Refuses (throws) rather than silently resuming something wrong: a `label` mismatch (e.g. a
checkpoint written by a CPU run being resumed into a `device="cuda"` config), a `positions` shape
mismatch (nwalkers or the parameter count changed since the checkpoint was written), or
`ckpt.ntotal != nsteps` (the config's `sampler.nsteps` changed since the checkpoint was written --
the "how many draws remain" arithmetic below assumes it didn't).

**Warmup caveat, stated plainly rather than glossed over**: if the checkpoint was taken DURING
warmup (`ckpt.it < nwarmup`), `NUTSCore.DualAveraging`'s own internal adaptation state
(`mu`/`h_bar`/`log_step_bar`/draw count `m`) is NOT checkpointed -- only the per-walker step size
`nuts_sample!` had reached at that draw is. Resuming restarts dual averaging FRESH for the
remaining `nwarmup - ckpt.it` warmup draws, using the checkpointed step size as its new starting
point rather than picking up mid-adaptation exactly. This is safe (dual averaging still converges)
but not bit-identical to an uninterrupted run. If the checkpoint was taken AFTER warmup completed
(`ckpt.it >= nwarmup`), there is no such caveat: `nwarmup_eff` comes back `0`, so
`step_size0` (the checkpointed, already-converged step size) is used as-is, exactly matching
`NUTSConfig`'s own documented `nwarmup=0` semantics.

**A second, separate non-bit-identity caveat**: `main()`'s `rng` is also not checkpointed, so a
resumed run's `nuts_sample!` call restarts its random draws (directions, multinomial-selection
`rand(rng)` calls, etc.) from the SAME seeded stream a fresh run would have used for draws
`1:nsteps_eff`, not from wherever an uninterrupted run's stream would have been at draw
`ckpt.it+1`. Confirmed by direct comparison (2026-09-18 smoke test): a resumed run's post-resume
positions/logp are close in magnitude to, but NOT identical to, the corresponding draws of an
uninterrupted run with the same seed -- statistically fine (still valid NUTS draws targeting the
same posterior), just not reproducible against an uninterrupted run's exact trajectory.
"""
function _apply_resume!(positions, step_size0, outdir, expected_label, nsteps, nwarmup)
    outdir === nothing && throw(ArgumentError("resume=true requires an [output] dir to resume from"))
    ckpt = _load_checkpoint(outdir)
    if ckpt === nothing
        println("  resume=true but no checkpoint found in $outdir -- starting fresh")
        return nsteps, nwarmup
    end
    ckpt.label == expected_label || throw(ArgumentError(
        "checkpoint in $outdir was written by a \"$(ckpt.label)\" run, not \"$expected_label\" -- " *
        "refusing to resume a mismatched run (e.g. a CPU checkpoint into a device=\"cuda\" config)"))
    size(ckpt.positions) == size(positions) || throw(ArgumentError(
        "checkpoint positions shape $(size(ckpt.positions)) doesn't match this config's " *
        "$(size(positions)) (nwalkers or parameter-count mismatch) -- refusing to resume"))
    ckpt.ntotal == nsteps || throw(ArgumentError(
        "checkpoint was written for nsteps=$(ckpt.ntotal) but this config has nsteps=$nsteps -- " *
        "refusing to resume against a changed step count"))
    nsteps_eff = nsteps - ckpt.it
    nsteps_eff > 0 || throw(ArgumentError(
        "checkpoint already completed all $nsteps draws -- nothing left to resume"))

    positions .= ckpt.positions
    step_size0 .= ckpt.step_size
    nwarmup_eff = max(nwarmup - ckpt.it, 0)
    println("  Resuming from checkpoint: draw $(ckpt.it)/$(ckpt.ntotal) already done, ",
            "$nsteps_eff draws remaining ($nwarmup_eff of them warmup)")
    return nsteps_eff, nwarmup_eff
end

"""
    _save_relative_binning_ref(outdir, physical_ref)
    _load_relative_binning_ref(outdir) -> Union{Nothing,Vector{Float64}}

Persist/reload the physical-space reference point [`RelativeBinning.build_relative_binning`](@ref)
was built from, so a two-phase relative-binning run (see `_run_nuts_relative_binning`) interrupted
DURING Phase B can rebuild an IDENTICAL `RelativeBinningSetup` on resume without re-running Phase
A's warmup -- Phase A already ran, before the interruption, and this is exactly the value it
produced. Written once, right after Phase A completes (whether fresh or itself resumed) and before
Phase B starts; `nothing` if it was never written (e.g. resuming a checkpoint that's still mid-
Phase-A, or a pre-2026-09-18 outdir from before this file existed).
"""
function _save_relative_binning_ref(outdir::AbstractString, physical_ref::AbstractVector)
    mkpath(outdir)
    writedlm(joinpath(outdir, "relative_binning_ref.csv"), physical_ref, ',')
end

function _load_relative_binning_ref(outdir::AbstractString)
    path = joinpath(outdir, "relative_binning_ref.csv")
    isfile(path) || return nothing
    return vec(Float64.(readdlm(path, ',')))
end

# ------------------------------------------------------------------------------------------------
# TOML -> data
# ------------------------------------------------------------------------------------------------

"""
    _build_detector_data(cfg) -> Vector{LikelihoodData}

Reads `cfg["event"]` (shared defaults) and `cfg["detectors"]` (one entry per detector, each at
least `name`/`strain_path`, everything else optional and falling back to `[event]`'s value), and
calls `build_gwdata` once per entry -- exactly what `build_gwdata_multi` does, just with per-
detector overrides `build_gwdata_multi` itself doesn't support (it shares every kwarg across all
detectors). Length of the returned vector is however many `[[detectors]]` tables the config has;
nothing here assumes or special-cases any particular count.
"""
function _build_detector_data(cfg)
    event = get(cfg, "event", Dict{String,Any}())
    dets = get(cfg, "detectors", nothing)
    dets === nothing && throw(ArgumentError("config needs at least one [[detectors]] table"))
    isempty(dets) && throw(ArgumentError("config needs at least one [[detectors]] table"))

    trigger_time = get(event, "trigger_time", nothing)
    trigger_time === nothing && throw(ArgumentError("[event] needs a trigger_time (GPS seconds)"))

    data = Vector{LikelihoodData}(undef, length(dets))
    for (i, d) in enumerate(dets)
        haskey(d, "name") || throw(ArgumentError("detectors[$i] needs a name (H1/L1/V1)"))
        detector = _lookup(_DETECTORS, d["name"], "detector")
        g(key, default) = get(d, key, get(event, key, default))
        if !haskey(d, "strain_path")
            # synthetic detector (P-P tests): no strain, the PSD of a noise-curve file (`asd_file` or `psd_file`)
            (haskey(d, "asd_file") || haskey(d, "psd_file")) ||
                throw(ArgumentError("detectors[$i] needs a strain_path, or an asd_file / psd_file for a synthetic detector"))
            data[i] = synthetic_gwdata(; detector=detector, trigger_time=g("trigger_time", trigger_time),
                                       duration=g("duration", 4.0), f_min=g("f_min", 20.0), f_max=g("f_max", 1024.0),
                                       asd_file=get(d, "asd_file", nothing), psd_file=get(d, "psd_file", nothing))
            continue
        end
        data[i] = build_gwdata(d["strain_path"];
                                trigger_time=g("trigger_time", trigger_time),
                                duration=g("duration", 4.0),
                                post_trigger=g("post_trigger", 2.0),
                                psd_segment=g("psd_segment", 1024.0),
                                f_min=g("f_min", 20.0),
                                f_max=g("f_max", 1024.0),
                                alpha=g("alpha", 0.2),
                                psd_method=Symbol(g("psd_method", "median")),
                                psd_file=get(d, "psd_file", nothing),
                                window_power_correction=g("window_power_correction", false),
                                detector=detector)
    end
    return data
end

# ------------------------------------------------------------------------------------------------
# NUTS run, with optional full-chain recording (`run_gw150914_estimation` itself only returns the
# final state -- this mirrors its internal setup, purely to additionally expose `record=true`).
# ------------------------------------------------------------------------------------------------

function _run_nuts(data, model; nwalkers, nsteps, step_size, nwarmup, max_treedepth, rng,
                    enforce_bounds, record_chain, outdir=nothing, resume=false, adapt_mass=false, mass_opts::NamedTuple=(;), quorum_opts::NamedTuple=(;),
                    lik_opts::NamedTuple=(;), prior=nothing, init=nothing, jumps::Bool=false, de_jumps::Bool=false,
                    pt_ntemps=1, pt_beta_min=0.05, pt_swap_interval::Int=1,
                    pt_adapt::Bool=true, pt_deo::Bool=true, pt_ntemps_max::Int=12, pt_snr2=nothing,
                    pt_reference_opts::NamedTuple=(;), xas_rb=nothing, device::String="cpu")
    T = Float64
    freqs = _representative_freqs(data)
    nrows = (model isa GravitationalWaveBayes.KernelFused._Prec15) ? 15 : 11
    # "auto" ladder (reference-waveform estimate): beta_min = d/snr2 (`analytic_beta_min`, d = number
    # of non-marginalized sampled parameters) and ntemps from the Gaussian communication barrier
    # (`pt_gaussian_barrier` -> `pt_recommended_ntemps`). snr2 is the injection's optimal network SNR^2
    # when there is one, else the zero-noise <h,h> at the best starting walker.
    if pt_beta_min == "auto" || pt_ntemps == "auto"
        d = nrows - lik_opts_count_marginalized(lik_opts)
        snr2 = pt_snr2 === nothing ? _reference_snr2(data, model, freqs, lik_opts, init, nwalkers, nrows, rng) : T(pt_snr2)
        bmin = pt_beta_min == "auto" ? analytic_beta_min(snr2, d; floor=1e-4) : T(pt_beta_min)
        nt = pt_ntemps == "auto" ? NUTSCore.pt_recommended_ntemps(NUTSCore.pt_gaussian_barrier(d, bmin); hi=pt_ntemps_max) : Int(pt_ntemps)
        println("PT ladder from the reference waveform: SNR = ", round(sqrt(snr2); digits=2), ", d = ", d,
                ", beta_min = ", round(bmin; sigdigits=3), ", Gaussian barrier = ",
                round(NUTSCore.pt_gaussian_barrier(d, bmin); digits=2), ", ntemps = ", nt)
        pt_beta_min, pt_ntemps = bmin, nt
    end
    pt_ntemps = Int(pt_ntemps)
    # PT with a reference (`sampler.pt_reference = "<run dir>"`, same density options as the SMC reference):
    # the path q0^(1-beta) (prior L)^beta, with one extra rung at beta = 0 whose walkers are redrawn iid
    # from q0 after every draw (a "regeneration": each round trip through it restarts from an exact
    # independent draw, Syed et al. 2021). Swaps use r = logL + log p - log q0.
    ref_dir = get(pt_reference_opts, :reference, nothing)
    use_ref = ref_dir !== nothing
    if use_ref
        pt_ntemps >= 2 || throw(ArgumentError("sampler.pt_reference needs pt_ntemps >= 2"))
        prior_ref = prior === nothing ? GWPrior{Float64}() : prior
        refobj, reference = _smc_reference(ref_dir, prior_ref, lik_opts, pt_reference_opts, rng)
        prior = prior_ref
        pt_ntemps += 1
    end
    # Parallel tempering (pt_ntemps > 1): `nwalkers` walkers PER temperature, rung r holding walkers
    # (r-1)*nwalkers+1 : r*nwalkers at beta_r = pt_beta_min^((r-1)/(ntemps-1)); rung 1 is beta = 1.
    # Each walker keeps its temperature; positions are exchanged between adjacent rungs every
    # `pt_swap_interval` draws (NUTSCore.pt_position_swaps!). Only rung 1 is returned.
    pt = pt_ntemps > 1
    pt && resume && throw(ArgumentError("sampler.resume is not supported with parallel tempering"))
    nwalkers_cold = nwalkers
    nwalkers = nwalkers_cold * pt_ntemps
    cfg = _pe_config(T, freqs, nwalkers, step_size, nsteps, lik_opts)
    # Array(...) forces host residency regardless of whether CUDA happens to be loaded in this
    # process (compatible_array, which this helper uses internally, auto-promotes to CuArray once
    # CUDA.jl is loaded+functional ANYWHERE in the process -- including for a device="cpu" run, if
    # CUDA.jl was ALSO loaded for an earlier/other device="cuda" config in the same session).
    # nuts_sample!/nuts_draw!'s own host-side scalar per-walker logic needs positions to stay a
    # plain Array unconditionally; gw_logp_and_grad_gpu_staged handles its OWN device staging
    # internally and never expects this array to already be device-resident.
    positions = use_ref ? Matrix{T}(sample_reference(rng, refobj, nwalkers)) :
                init === nothing ? Array(_gw150914_initial_positions(T, nwalkers, nrows; rng=rng)) :
                (size(init, 2) == nwalkers ? Matrix{T}(init) : repeat(Matrix{T}(init), 1, pt_ntemps))
    step_size0 = fill(T(step_size), nwalkers)
    nsteps_eff, nwarmup_eff = nsteps, nwarmup
    if resume
        nsteps_eff, nwarmup_eff = _apply_resume!(positions, step_size0, outdir, "CPU", nsteps, nwarmup)
    end
    rungs = [collect((r - 1) * nwalkers_cold + 1 : r * nwalkers_cold) for r in 1:pt_ntemps]
    ladder = if use_ref
        nt = pt_ntemps - 1
        T[[T(pt_beta_min)^(T(r - 1) / T(nt - 1)) for r in 1:nt]; zero(T)]
    else
        T[pt_ntemps == 1 ? one(T) : T(pt_beta_min)^(T(r - 1) / T(pt_ntemps - 1)) for r in 1:pt_ntemps]
    end
    betas = T[ladder[r] for r in 1:pt_ntemps for _ in 1:nwalkers_cold]

    # Unwrap a 1-detector Vector{LikelihoodData} to a bare LikelihoodData: gw_logp_and_grad's
    # forward-mode SEOBNR dispatch has a specialized, hand-derived-JVP method keyed on
    # `data::LikelihoodData` specifically (see BatchedNUTS.jl's two _forward_mode_gradient!
    # methods) -- a length-1 Vector never matches it and falls through to the generic
    # DetectorData method instead, which has a real, currently-unresolved gap differentiating
    # SEOBNRv5AlignedSpinF2 (found 2026-09-14: both plain Enzyme.Forward and
    # set_runtime_activity(Enzyme.Forward) fail, differently, for that path). Every other model
    # dispatches identically either way (DetectorData's Union already covers both shapes), so
    # this is a pure widening, not a behavior change for non-SEOBNR runs.
    data_for_grad = (data isa AbstractVector && length(data) == 1) ? data[1] : data
    # XAS relative binning with the likelihood options (`sampler.relative_binning = true`, 2026-09-28): the
    # summary data are built around the walker with the highest DENSE likelihood, rebuilt at `xas_rb.rebuild_at`
    # warmup draws (the target changes only between draws; `nuts_draw!` re-evaluates logp every draw) and frozen
    # for sampling, so the sampling phase targets one fixed relative-binning likelihood.
    rb_cur = Ref{Any}(nothing)
    if xas_rb !== nothing
        model isa Union{IMRPhenomXAS,IMRPhenomXHM} || throw(ArgumentError("relative binning with options needs IMRPhenomXAS or IMRPhenomXHM"))
        rb_rebuild! = function (pos)
            lld = zeros(T, size(pos, 2))
            _raw_loglikelihood!(lld, pos, freqs, cfg, data_for_grad, model, nothing)
            best = argmax(map(x -> isfinite(x) ? x : -Inf, lld))
            rb_cur[] = GravitationalWaveBayes.BatchedNUTS.build_relative_binning_opts(model, data_for_grad, cfg, pos[:, best]; nbins=xas_rb.nbins)
            println(nameof(typeof(model)), " relative binning: fiducial = walker $best (dense logL ", round(lld[best]; digits=2), "), bins ",
                    rb_cur[].nbins, ", fiducial ", round.(pos[:, best]; sigdigits=5))
            outdir === nothing || writedlm(joinpath(outdir, "relative_binning_ref.csv"), pos[:, best], ',')
            flush(stdout)
        end
        outdir === nothing || mkpath(outdir)
        rb_rebuild!(positions)
    end
    # device = "cuda" (IMRPhenomXAS, 2026-09-30; with `relative_binning = true` on the device too, 2026-10-01): the
    # bin-parallel device closure, tempered and with the prior added on the host (`tempered_logp_and_grad`); positions
    # stay host arrays.
    # SEOBNRv5 IMR on the device (2026-10-03): one device likelihood serves the gradient closure and the PT swaps'
    # untempered log-likelihoods (`seobnr_device_closures`).
    seob_dev = (device == "cuda" && model isa SEOBNRv5IMRAlignedSpinF2 && xas_rb === nothing) ?
        GravitationalWaveBayes.BatchedNUTS.seobnr_device_closures(data_for_grad, cfg; nwalkers=nwalkers,
            backend=GravitationalWaveBayes.BatchedNUTS.KernelAbstractions.get_backend(GravitationalWaveBayes.GPU.to_device(zeros(T, 1)))) : nothing
    make_lpg(rb) = if device == "cuda"
        ((model isa IMRPhenomXAS || seob_dev !== nothing) && prior !== nothing) || throw(ArgumentError(
            "device=\"cuda\" in the PT/NUTS runner needs IMRPhenomXAS or SEOBNRv5IMRAlignedSpinF2 and a [prior]"))
        base! = seob_dev !== nothing ? seob_dev[1] :
            gw_logp_and_grad_gpu_staged(data_for_grad, cfg, freqs; model=model, nparams=nrows, nwalkers=nwalkers,
                                        relative_binning=rb)
        GravitationalWaveBayes.BatchedNUTS.tempered_logp_and_grad(base!, pt ? betas : ones(T, nwalkers), prior)
    else
        gw_logp_and_grad(data_for_grad, cfg, freqs; model=model, prior=prior,
                         beta=pt ? betas : nothing, relative_binning=rb)
    end
    lpg_cur = Ref{Any}(make_lpg(rb_cur[]))
    logp_and_grad! = xas_rb === nothing ? lpg_cur[] : ((g, l, p, m) -> lpg_cur[](g, l, p, m))
    raw_ll! = seob_dev !== nothing ? seob_dev[2] :
        ((ll, pos) -> _raw_loglikelihood!(ll, pos, freqs, cfg, data_for_grad, model, rb_cur[]))
    if use_ref
        base_lpg! = logp_and_grad!
        lpb = Vector{T}(undef, nwalkers); lqb = Vector{T}(undef, nwalkers)
        gpb = Matrix{T}(undef, nrows, nwalkers); gqb = Matrix{T}(undef, nrows, nwalkers)
        logp_and_grad! = function (grad, logp, pos, mask)
            # reference terms on another task while the (possibly device) likelihood runs
            ref_task = Threads.@spawn begin
                reference.logprior_and_grad!(lpb, gpb, pos)
                reference.logq_and_grad!(lqb, gqb, pos)
            end
            base_lpg!(grad, logp, pos, mask)
            fetch(ref_task)
            c = one(T) .- betas
            logp .+= c .* (lqb .- lpb)
            grad .+= reshape(c, 1, :) .* (gqb .- gpb)
            return nothing
        end
        base_raw_ll! = raw_ll!
        raw_ll! = function (ll, pos)
            base_raw_ll!(ll, pos)
            reference.logprior_and_grad!(lpb, gpb, pos)
            reference.logq_and_grad!(lqb, gqb, pos)
            ll .+= lpb .- lqb
            return ll
        end
    end
    bounds = enforce_bounds ? (prior === nothing ? _sampled_space_bounds(T, nrows) : _sampled_space_bounds(prior)) : nothing
    nuts_cfg = NUTSConfig{T}(; max_treedepth=max_treedepth, nwarmup=nwarmup_eff, bounds=bounds, adapt_mass=adapt_mass, mass_opts..., quorum_opts...)

    moves = jumps ? gw_jump_moves(data_for_grad) : ()
    # Relative binning with log-likelihood-based jump moves: jumps use `logp_and_grad!` (the RB target).
    progress = _progress_callback("CPU", nsteps_eff; outdir=outdir)
    npairs = max(pt_ntemps - 1, 0)
    swap_accepts = zeros(Int, npairs)
    swap_attempts = zeros(Int, npairs)       # whole run (acceptance report)
    round_rej = zeros(T, npairs)             # current adaptation round
    round_att = zeros(Int, npairs)
    ll_scratch = zeros(T, nwalkers)
    swap_round = Ref(0)
    # Ladder adaptation (warmup only; frozen afterwards so the sampling phase is exact): rounds end at
    # 1/8, 1/4, 1/2 and 3/4 of warmup (doubling-length rounds, as in Syed et al. 2021), leaving the last
    # quarter for the step sizes to settle on the final ladder.
    adapt_at = (pt && pt_adapt && pt_ntemps > (use_ref ? 3 : 2)) ?
        unique(filter(>(0), round.(Int, nwarmup_eff .* (0.125, 0.25, 0.5, 0.75)))) : Int[]
    barrier = Ref(NaN)
    callback = if pt
        function (it, pos, logp, ss, stats)
            progress(it, pos, logp, ss, stats)
            if it % pt_swap_interval == 0
                if use_ref
                    # beta = 0 rung: exact iid refresh from q0 (a Gibbs step on that rung's marginal).
                    pos[:, rungs[end]] .= sample_reference(rng, refobj, length(rungs[end]))
                end
                raw_ll!(ll_scratch, pos)
                acc0 = copy(round_att)
                pt_position_swaps!(pos, ll_scratch, betas, rungs, swap_accepts, rng;
                                   parity=pt_deo ? swap_round[] % 2 : nothing, rej_sum=round_rej, attempts=round_att)
                swap_attempts .+= round_att .- acc0
                swap_round[] += 1
            end
            if it in adapt_at && all(>(0), round_att)
                rej = round_rej ./ round_att
                # With a reference rung (beta = 0, fixed) only the positive part of the ladder is re-spaced.
                newladder, barrier[] = use_ref ?
                    (vcat(NUTSCore.pt_adapt_ladder(ladder[1:end-1], rej[1:end-1])[1], zero(T)), sum(rej)) :
                    NUTSCore.pt_adapt_ladder(ladder, rej)
                println("PT ladder update at warmup draw $it: rejection ", round.(rej; digits=2),
                        " barrier ", round(barrier[]; digits=2), " (recommended ntemps ",
                        NUTSCore.pt_recommended_ntemps(barrier[]), ") betas ", round.(newladder; sigdigits=3))
                ladder .= newladder
                for r in 1:pt_ntemps, w in rungs[r]
                    betas[w] = ladder[r]
                end
                round_rej .= 0
                round_att .= 0
            end
            return nothing
        end
    else
        progress
    end
    if xas_rb !== nothing
        cb_inner = callback
        callback = function (it, pos, logp, ss, stats)
            cb_inner === nothing || cb_inner(it, pos, logp, ss, stats)
            if it in xas_rb.rebuild_at && it <= nwarmup_eff
                rb_rebuild!(pos)
                lpg_cur[] = make_lpg(rb_cur[])
            end
        end
    end
    result = nuts_sample!(positions, logp_and_grad!; nsteps=nsteps_eff, step_size0=step_size0,
                           cfg=nuts_cfg, rng=rng, record=record_chain, callback=callback, jumps=moves,
                           de_jumps=de_jumps, metric_walkers=pt ? rungs[1] : nothing)
    isempty(moves) || println("Jump-move acceptance rates: ", round.(result.jump_acceptance; digits=3))
    de_jumps && println("DE-jump acceptance rate: ", round(result.de_acceptance; digits=3))
    if pt
        acc = swap_accepts ./ max.(swap_attempts, 1)
        println("PT betas: ", round.(betas[first.(rungs)]; digits=4), "  swap acceptance (r,r+1): ",
                round.(acc; digits=3), pt_deo ? " (DEO)" : "")
        cold = rungs[1]
        result = merge(result, (; positions=result.positions[:, cold], logp=result.logp[cold],
                                  step_size=result.step_size[cold],
                                  chain=result.chain === nothing ? nothing : result.chain[:, cold, :],
                                  pt_swap_acceptance=acc, pt_betas=copy(ladder)))
        cfg = _pe_config(T, freqs, nwalkers_cold, step_size, nsteps, lik_opts)
    end
    # A resumed chain only has nsteps_eff (not the pre-interruption nwarmup) draws recorded -- see
    # main()'s use of this for _write_outputs' diagnostics warmup cutoff.
    return merge(result, (; nwarmup_eff, pe_cfg=cfg, data_for_grad)), nrows
end

# Rows a sample-based KDE should cover: every likelihood-constrained row. Marginalized rows keep the prior,
# and so do periodic rows the samples barely constrain (circular sd above period/8, e.g. psi). Returns
# (rows, covariance of the samples on those rows -- periodic deviations about the circular mean -- and the
# per-row period, 0 = not periodic).
function _kde_rows_and_cov(X::AbstractMatrix{T}, pr, lik_opts) where {T}
    marg = Int[]
    get(lik_opts, :marginalize_distance, false) && push!(marg, 5)
    get(lik_opts, :marginalize_phase, false) && push!(marg, 7)
    lo_, hi_, per_ = prior_bounds(pr)
    dev = similar(X)
    rows = Int[]
    for r in 1:size(X, 1)
        r in marg && continue
        if per_[r]
            P = hi_[r] - lo_[r]
            cmean = angle(sum(cis.(2pi .* (X[r, :] .- lo_[r]) ./ P))) * P / (2pi) + lo_[r]
            d = X[r, :] .- cmean
            dev[r, :] .= d .- P .* round.(d ./ P)
            sqrt(sum(abs2, dev[r, :]) / size(X, 2)) < P / 8 || continue
        else
            dev[r, :] .= X[r, :] .- sum(X[r, :]) / size(X, 2)
        end
        push!(rows, r)
    end
    D = dev[rows, :]
    C = D * D' ./ (size(D, 2) - 1)
    period = T[per_[r] ? hi_[r] - lo_[r] : zero(T) for r in rows]
    return rows, C, period
end

# SMC block move for the IMR-table models (`sampler.smc_extrinsic_sweeps = k > 0`): after every full NUTS
# draw, `k` NUTS draws on the extrinsic rows only (t_c, sky, orientation; plus distance/phase when they are
# not marginalized) at fixed intrinsics, with each particle's waveform table cached
# (`BatchedNUTS.IMRExtrinsicBlock`, ~15x cheaper gradients for SEOBNRv5 IMR). Metropolis-within-Gibbs: each
# draw leaves prior * L^beta invariant conditional on the intrinsic rows. Metric = the population metric's
# extrinsic sub-block; step size adapted from the population's mean acceptance (O(1/N) coupling).
function _imr_extrinsic_block_move(data_for_grad, freqs, cfg, model, pr, lik_opts; nsweeps::Int=3, T=Float64)
    blk = GravitationalWaveBayes.BatchedNUTS.IMRExtrinsicBlock(data_for_grad, freqs, cfg, model)
    rows = [r for r in 5:11 if !(r == 5 && get(lik_opts, :marginalize_distance, false)) &&
                                 !(r == 7 && get(lik_opts, :marginalize_phase, false))]
    lo, hi, per = prior_bounds(pr)
    sub_cfg = NUTSConfig{T}(; max_treedepth=8, nwarmup=0, bounds=NUTSBounds(lo[rows], hi[rows]; periodic=per[rows]))
    eps = Ref(T(0.1))
    println("SMC: extrinsic block moves on rows ", rows, " (", nsweeps, " cached-waveform NUTS draws per full draw)")
    return function (positions, betas, metric, rng)
        n = size(positions, 2)
        GravitationalWaveBayes.BatchedNUTS.cache!(blk, positions)
        full = copy(positions)
        ll = zeros(T, n); g = zeros(T, 11, n); lpr = zeros(T, n); gpr = zeros(T, 11, n)
        lpg! = function (grad, logp, sp, mask)
            full[rows, :] .= sp
            GravitationalWaveBayes.BatchedNUTS.ext_ll_and_grad!(blk, ll, g, full)
            fill!(gpr, zero(T))
            prior_logpdf_and_grad!(lpr, gpr, pr, full)
            for w in 1:n
                if isfinite(ll[w]) && isfinite(lpr[w])
                    logp[w] = betas[w] * ll[w] + lpr[w]
                    @views grad[:, w] .= betas[w] .* g[rows, w] .+ gpr[rows, w]
                else
                    logp[w] = T(-Inf)
                    @views grad[:, w] .= zero(T)
                end
            end
            return nothing
        end
        sub = positions[rows, :]
        lp = zeros(T, n)
        subm = DenseMetric(metric.Minv[rows, rows])
        evals = 0
        for _ in 1:nsweeps
            step = eps[] .* (T(0.8) .+ T(0.4) .* rand(rng, T, n))
            st = NUTSCore.nuts_draw!(sub, lp, step, lpg!, sub_cfg, rng, subm)
            evals += sum(st.n_leapfrog)
            a = sum(st.accept_stat) / n
            eps[] *= exp(clamp(T(2) * (a - T(0.8)), T(-1), T(0.5)))
        end
        positions[rows, :] .= sub
        return evals
    end
end

# Online SMC proposal (`sampler.smc_online_kde = true`): a function (pool, weights) -> (; sample, logq_and_grad!) that
# builds a LOCAL KDE (kernels on the pool, nearest-neighbour covariances, + prior component) from one SMC
# stage's mutation states, for the next stage's independence moves. Returns `nothing` (keep the previous
# proposal) if the pool is too small or the construction fails.
function _online_kde_proposal(pr, lik_opts, smc_opts, rng; T=Float64)
    alpha = get(smc_opts, :online_alpha, 0.1)
    kmax = get(smc_opts, :online_kernels, 300)
    return function (pool::AbstractMatrix, W::AbstractVector)
        size(pool, 2) >= 16 || return nothing
        try
            X = Matrix{T}(pool)
            rows, C, period = _kde_rows_and_cov(X, pr, lik_opts)
            nk = min(kmax, size(X, 2))
            # kernel centres: systematic resampling of the pool by its weights toward the new beta (the
            # neighbour covariances below still come from the whole pool)
            cw = cumsum(W ./ sum(W)); u0 = rand(rng) / nk
            pick = [min(searchsortedfirst(cw, u0 + (j - 1) / nk), size(X, 2)) for j in 1:nk]
            covs = local_kde_covariances(X[rows, :], pick, period; k=min(get(smc_opts, :reference_neighbours, 24), size(X, 2) - 1),
                                         h=get(smc_opts, :reference_local_h, 1.0))
            ref = KDEPriorMixture(pr, rows, X[rows, pick], covs; alpha=alpha, rng=rng, nmc=500)
            return (; sample = (r, n) -> Matrix{T}(sample_reference(r, ref, n)),
                      logq_and_grad! = (lq, g, pos) -> reference_logpdf_and_grad!(lq, g, ref, pos))
        catch err
            @warn "online KDE proposal: construction failed, keeping the previous proposal" exception=err
            return nothing
        end
    end
end

# Seconds per stage of the mode-sum likelihood closures (RW / hybrid CUDA runs): (a) setup incl. the device tables,
# (b) host mode factors, (c) bin sums incl. copies, (d) host combination (and gradient surrogate); printed and reset
# per SMC stage.
const _LIK_TIMERS = zeros(4)
const _COMPACT_STATS = Ref{Any}(nothing)        # `compact_active` stats (`sampler.smc_nuts_narrow`)

# Run `f()`; if it throws (e.g. a device exception in a likelihood kernel), first save the batch's positions to
# `gwbayes_failed_positions_<pid>.jls` in the working directory, for a replay on the CPU model.
function _save_on_error(f, pos)
    try
        return f()
    catch
        fn = abspath("gwbayes_failed_positions_$(getpid()).jls")
        Serialization.serialize(fn, Matrix{Float64}(pos))
        println("likelihood threw: positions saved to ", fn); flush(stdout)
        rethrow()
    end
end

# Online SMC flow proposal (`sampler.smc_online_flow = true`, in place of the online KDE): a normalizing flow
# (`GravitationalWaveBayes.Flows`, RealNVP) fitted to one stage's mutation states weighted to the next beta, mixed with the
# normalized prior (weight `online_alpha`), for the next stage's independence moves (flowMC's global jumps, but each
# one an exact MH step against the stage target). `smc_flow_steps` / `smc_flow_layers` / `smc_flow_hidden` set the
# training; give `smc_online_pool_max` a few thousand states to train on.
function _online_flow_proposal(pr, smc_opts, rng; T=Float64)
    alpha = T(get(smc_opts, :online_alpha, 0.1))
    steps = Int(get(smc_opts, :flow_steps, 1500))
    nl = Int(get(smc_opts, :flow_layers, 8)); hid = Int(get(smc_opts, :flow_hidden, 64))
    # `smc_flow_batch`: training minibatch (default 512; on the GPU a step costs about the same at 4096 -- launch-bound)
    fbatch = Int(get(smc_opts, :flow_batch, 512))
    lo, hi, per = prior_bounds(pr)
    lpn = prior_log_normalizer(pr)
    # `smc_flow_warm = true`: each stage's fit starts from the previous stage's flow (fewer steps to the early stop)
    warm = Bool(get(smc_opts, :flow_warm, false))
    # `smc_flow_device = true`: training, and the per-batch sampling / density of the proposal, on the GPU
    fdev = Bool(get(smc_opts, :flow_device, false))
    todev = GravitationalWaveBayes.GPU.to_device
    prev = Ref{Any}(nothing)
    return function (pool::AbstractMatrix, W::AbstractVector)
        size(pool, 2) >= 256 || return nothing
        try
            t0 = time()
            F = fit_flow(Matrix{Float64}(pool), Float64.(W); lo=lo, hi=hi, periodic=per, nlayers=nl, hidden=hid,
                         steps=steps, batch=fbatch, init=warm ? prev[] : nothing, (fdev ? (; device=todev) : (;))..., rng=rng)
            prev[] = F
            Fe = fdev ? GravitationalWaveBayes.Flows.flow_to_device(F, todev) : F
            println("  online flow: fitted to ", size(pool, 2), " states (ESS ", round(1 / sum(abs2, W ./ sum(W)); digits=0),
                    ") in ", round(time() - t0; digits=1), " s"); flush(stdout)
            sample = function (r, n)
                X = Matrix{T}(Array(flow_sample(r, Fe, n)))
                P = sample_prior(r, pr, n)
                for w in 1:n
                    rand(r) < alpha && (@views X[:, w] .= P[:, w])
                end
                return X
            end
            logq! = function (lq, g, pos)
                lf = fdev ? Array(flow_logpdf!(todev(zeros(Float64, length(lq))), Fe, todev(Matrix{Float64}(pos)))) :
                            flow_logpdf!(zeros(T, length(lq)), F, pos)
                lp = prior_logpdf!(zeros(T, length(lq)), pr, pos) .- lpn
                @inbounds for w in eachindex(lq)
                    a, b = log1p(-alpha) + lf[w], log(alpha) + lp[w]
                    m = max(a, b)
                    lq[w] = isfinite(m) ? m + log(exp(a - m) + exp(b - m)) : m
                end
                fill!(g, zero(T))
                return nothing
            end
            # the flow's latent space, for `smc_latent_nuts` (NeuTra: the gradient kernel runs on z, x = T(z))
            to_latent = X -> flow_to_latent(F, X)
            from_latent = function (Z)
                X, ld, c = flow_from_latent(F, Z)
                return X, ld, GX -> flow_latent_pull(F, c, GX)
            end
            return (; sample, logq_and_grad! = logq!, to_latent, from_latent)
        catch err
            # (printed to stdout: a silent fallback hid a broken flow for a whole run, 2026-10-12)
            println("  online flow: FIT FAILED, keeping the previous proposal: ", sprint(showerror, err)[1:min(end, 300)])
            flush(stdout)
            return nothing
        end
    end
end

# SMC reference density from a previous run's chain (`sampler.smc_reference`): returns
# `(ref, reference)` -- the density object (for `sample_reference`) and the `(; logq_and_grad!,
# logprior_and_grad!)` NamedTuple `smc_sample` takes -- or `(nothing, nothing)` without one.
function _smc_reference(ref_dir, pr, lik_opts, smc_opts, rng; T=Float64)
    ref_dir === nothing && return (nothing, nothing)
    ch = deserialize(joinpath(ref_dir, "chain.jls")).chain
    # A NUTS/PT chain.jls records every draw, warmup included (the first ones start far from the
    # posterior); drop them. SMC chains are all post-ladder draws.
    nwarm = 0
    cfgpath = joinpath(ref_dir, "config_used.toml")
    if isfile(cfgpath)
        sref = get(TOML.parsefile(cfgpath), "sampler", Dict{String,Any}())
        get(sref, "method", "nuts") == "smc" || (nwarm = Int(get(sref, "nwarmup", 0)))
    else
        @warn "SMC/PT reference: no config_used.toml in $ref_dir; using every draw of chain.jls (warmup included?)"
    end
    nwarm < size(ch, 3) || throw(ArgumentError("reference chain has no post-warmup draws"))
    Xall = reshape(ch[:, :, nwarm+1:end], size(ch, 1), :)
    # `reference_time_shift` (s): added to the reference's time row, to bring a pass-1 run analysed with a
    # different `time_ref_offset` (e.g. "auto" with another model) into this run's time frame:
    # shift = this run's time_ref_offset - the reference run's.
    tshift = T(get(smc_opts, :reference_time_shift, 0.0))
    tshift == 0 || (Xall[6, :] .+= tshift; println("SMC/PT reference: time row shifted by ", tshift, " s"))
    println("SMC/PT reference: ", size(Xall, 2), " draws from ", ref_dir, " (dropped ", nwarm, " warmup draws per walker)")
    infl = T(get(smc_opts, :reference_inflate, 1.5))
    alpha = get(smc_opts, :reference_alpha, 0.1)
    kind = get(smc_opts, :reference_kind, "kde")
    if kind == "gaussian"
        rows = [1, 2, 3, 4, 6]
        X = Xall[rows, :]
        mu = vec(sum(X; dims=2)) ./ size(X, 2)
        C = (X .- mu) * (X .- mu)' ./ (size(X, 2) - 1)
        ref = GaussianPriorMixture(pr, rows, mu, infl^2 .* C; alpha=alpha, rng=rng)
        println("SMC reference (Gaussian) from ", ref_dir, ": rows ", rows, " mean ", round.(mu; sigdigits=5), " sd ",
                round.(sqrt.([C[i, i] for i in eachindex(rows)]); sigdigits=3), " (x", infl, "), alpha ", alpha)
    elseif kind == "kde" || kind == "kde_global"
        # Kernel density estimate over every likelihood-constrained row: marginalized rows keep the
        # prior, and so do periodic rows the cheap posterior barely constrains (circular sd above
        # period/8, e.g. psi). Periodic deviations are taken about the circular mean.
        rows, C, period = _kde_rows_and_cov(Xall, pr, lik_opts)
        # `reference_prior_rows`: rows kept at the prior although the pass-1 run constrains them (a convention the two
        # models do not share, e.g. the reference phase between IMRPhenomXPHM and SEOBNRv5PHM)
        keep = findall(r -> !(r in get(smc_opts, :reference_prior_rows, Int[])), rows)
        rows, C, period = rows[keep], C[keep, keep], period[keep]
        lo_, hi_, per_ = prior_bounds(pr)
        nk = min(get(smc_opts, :reference_kernels, 400), size(Xall, 2))
        pick = round.(Int, range(1, size(Xall, 2); length=nk))
        if kind == "kde_global"
            h = infl * T(nk)^(-1 / (length(rows) + 4))          # Scott's rule, inflated
            ref = KDEPriorMixture(pr, rows, Xall[rows, pick], h^2 .* C; alpha=alpha, rng=rng)
            println("SMC reference (global KDE, ", nk, " kernels) from ", ref_dir, ": rows ", rows, " bandwidth sd ",
                    round.(h .* sqrt.([C[i, i] for i in eachindex(rows)]); sigdigits=3), ", alpha ", alpha)
        else
            # Local KDE (default): each kernel's covariance from its nearest posterior samples, so the
            # mixture follows narrow curved ridges (a global bandwidth fills them in: on a thin test ring
            # the sd of log(target/q0) is 89 with Scott's rule and 1.0 with local covariances).
            # Neighbours default to ~1% of the pool (min 24, the value tuned on ~2000-2400-draw pass-1 chains):
            # at a fixed k the nearest-neighbour kernels shrink as the pool gets denser, and a 24000-draw pool with
            # k = 24 gave a reference too tight to cover the expensive model's shifted posterior (2026-09-28).
            kdef = max(24, round(Int, 0.01 * size(Xall, 2)))
            covs = local_kde_covariances(Xall[rows, :], pick, period; k=get(smc_opts, :reference_neighbours, kdef),
                                         h=get(smc_opts, :reference_local_h, 1.0))
            ref = KDEPriorMixture(pr, rows, Xall[rows, pick], covs; alpha=alpha, rng=rng, nmc=2000)
            med = [sort([sqrt(covs[i, i, j]) for j in 1:nk])[cld(nk, 2)] for i in eachindex(rows)]
            println("SMC reference (local KDE, ", nk, " kernels) from ", ref_dir, ": rows ", rows,
                    " median kernel sd ", round.(med; sigdigits=3), " (posterior sd ",
                    round.(sqrt.([C[i, i] for i in eachindex(rows)]); sigdigits=3), "), alpha ", alpha)
        end
    else
        throw(ArgumentError("sampler.smc_reference_kind must be \"kde\", \"kde_global\" or \"gaussian\", got \"$kind\""))
    end
    flush(stdout)                                  # a file-redirected stdout is block-buffered
    lpn = prior_log_normalizer(pr)
    reference = (; sample = (r, n) -> Matrix{T}(sample_reference(r, ref, n)),
                   logq_and_grad! = (lq, g, pos) -> reference_logpdf_and_grad!(lq, g, ref, pos),
                   logprior_and_grad! = function (lp, g, pos)
                       fill!(g, zero(T))
                       prior_logpdf_and_grad!(lp, g, pr, pos)
                       lp .-= lpn
                       return nothing
                   end)
    return ref, reference
end

"""
    _run_smc(data, model; nwalkers, ..., smc_opts) -> (result, nrows)

Adaptive tempered SMC (`NUTSCore.smc_sample`) in place of NUTS warmup + parallel tempering:
`nwalkers` particles drawn from the prior are annealed to the posterior through a CESS-adapted beta
schedule, each stage mutated by batched NUTS with a population-estimated block-dense metric, then
`smc_nfinal` draws at beta = 1 are all kept (the returned chain; no warmup to discard). Also returns
the log-evidence `logZ` (relative to the prior) and the total gradient evaluations `n_leapfrog`.
Selected with `sampler.method = "smc"`; `[prior]` (or the default `GWPrior`) is both the initial
distribution and the prior term of the target, so they are exactly consistent.
"""
function _run_smc(data, model; nwalkers, step_size, max_treedepth, rng, enforce_bounds, outdir=nothing,
                  mass_opts::NamedTuple=(;), quorum_opts::NamedTuple=(;), lik_opts::NamedTuple=(;),
                  prior=nothing, smc_opts::NamedTuple=(;), device::AbstractString="cpu", kwargs...)
    T = Float64
    freqs = _representative_freqs(data)
    nrows = (model isa GravitationalWaveBayes.KernelFused._Prec15) ? 15 : 11
    cal = get(smc_opts, :calibration, nothing)
    # [calibration] mode = "marginalize": response curves averaged in the likelihood (SEOBNRv5PHM), no sampled rows
    cm = get(smc_opts, :calmarg, nothing)
    calm = cm === nothing ? nothing : cm.C
    (calm === nothing || (device == "cuda" && model isa Union{SEOBNRv5PHM,IMRPhenomXPHM})) ||
        throw(ArgumentError("[calibration] mode = \"marginalize\" is implemented for SEOBNRv5PHM / IMRPhenomXPHM on CUDA only"))
    if cal !== nothing
        (device == "cuda" && model isa Union{IMRPhenomXPHM,SEOBNRv5PHM}) ||
            throw(ArgumentError("[calibration] is implemented for IMRPhenomXPHM / SEOBNRv5PHM on CUDA only"))
        nrows += length(cal.mu)
    end
    pr = prior === nothing ? GWPrior{Float64}(; nrows=nrows) : prior
    length(pr.lo) == nrows || throw(ArgumentError("the prior has $(length(pr.lo)) rows, the model samples $nrows"))
    cfg = _pe_config(T, freqs, nwalkers, step_size, 1, lik_opts)
    # the device closures' support: the configured prior's, not the package box (d_L <= 4 Gpc, q >= 0.125 there)
    boxf = pos -> prior_support_vec(pr, pos)
    # Optional reference (`sampler.smc_reference = "<run dir>"`): a Gaussian fitted to that run's
    # chain on the intrinsic + t_c rows, sd inflated by `smc_reference_inflate`, mixed with the prior
    # (weight `smc_reference_alpha`) -- e.g. a cheap model's posterior as the start for an expensive one.
    ref, reference = _smc_reference(get(smc_opts, :reference, nothing), pr, lik_opts, smc_opts, rng)
    positions = Matrix{T}(reference === nothing ? sample_prior(rng, pr, nwalkers) : sample_reference(rng, ref, nwalkers))
    betas = zeros(T, nwalkers)
    data_for_grad = (data isa AbstractVector && length(data) == 1) ? data[1] : data
    loglik! = (ll, pos) -> _raw_loglikelihood!(ll, pos, freqs, cfg, data_for_grad, model, nothing)
    rw = Symbol(get(smc_opts, :moves, "nuts")) === :rw
    logp_and_grad! = if rw
        # Gradient-free random-walk moves (`sampler.smc_moves = "rw"`): a likelihood closure only, no gradient path
        # compiled. SEOBNRv5PHM on CUDA: the device likelihood (`seobnr_phm_loglik`); other CUDA models keep their
        # stage-reweighting closure below (which may compute a gradient).
        xphm_base! = nothing
        if device == "cuda" && model isa SEOBNRv5PHM
            loglik! = GravitationalWaveBayes.BatchedNUTS.seobnr_phm_loglik(data_for_grad, cfg; nwalkers=nwalkers, calibration=cal, calmarg=calm, boxf=boxf,
                backend=GravitationalWaveBayes.BatchedNUTS.KernelAbstractions.get_backend(GravitationalWaveBayes.GPU.to_device(zeros(T, 1))))
        elseif device == "cuda" && model isa Union{IMRPhenomXPHM,IMRPhenomXHM,GravitationalWaveforms.SEOBNRv5HM}
            # (2026-10-11) the bin-parallel closure also serves the RW stages' likelihoods (2026-10-12: also the aligned
            # higher-mode models, same row layout), value only (no tangents / surrogate reverse)
            xphm_base! = gw_logp_and_grad_gpu_staged(data_for_grad, cfg, freqs; model=model, nparams=nrows,
                                                     nwalkers=nwalkers, calibration=cal, calmarg=calm, box=boxf,
                                                     timers=_LIK_TIMERS)
            gscr_rw = zeros(T, nrows, nwalkers)
            loglik! = (ll, pos) -> (_save_on_error(() -> xphm_base!(gscr_rw, ll, pos, nothing; value_only=true), pos); ll)
        elseif device == "cuda"
            throw(ArgumentError("sampler.smc_moves = \"rw\" on CUDA is wired up for SEOBNRv5PHM / IMRPhenomXPHM / XHM / SEOBNRv5HM only"))
        end
        if haskey(smc_opts, :nuts_beta)
            # hybrid (`sampler.smc_nuts_beta`): NUTS stages from that beta on, with the device gradient
            (device == "cuda" && model isa Union{SEOBNRv5PHM,IMRPhenomXPHM,IMRPhenomXHM,GravitationalWaveforms.SEOBNRv5HM}) ||
                throw(ArgumentError("sampler.smc_nuts_beta is wired up for SEOBNRv5PHM / IMRPhenomXPHM / XHM / SEOBNRv5HM on CUDA only"))
            base! = xphm_base! !== nothing ? xphm_base! :
                    gw_logp_and_grad_gpu_staged(data_for_grad, cfg, freqs; model=model, nparams=nrows, nwalkers=nwalkers,
                                                calibration=cal, calmarg=calm, box=boxf)
            # `sampler.smc_nuts_narrow = [W, ...]` (2026-10-15): NUTS leapfrog calls with at most W active walkers run on
            # a closure W wide (`compact_active`); [] still records the active fraction per call
            if haskey(smc_opts, :nuts_narrow)
                narrow = [(W, gw_logp_and_grad_gpu_staged(data_for_grad, cfg, freqs; model=model, nparams=nrows, nwalkers=W,
                                                          calibration=cal, calmarg=calm, box=boxf))
                          for W in smc_opts.nuts_narrow]
                base!, cstats = GravitationalWaveBayes.BatchedNUTS.compact_active(base!, nwalkers; narrow=narrow)
                _COMPACT_STATS[] = cstats
                println("SMC: NUTS calls compacted to widths ", collect(smc_opts.nuts_narrow), " when few walkers are active")
            end
            println("SMC: hybrid -- NUTS mutations from beta >= ", smc_opts.nuts_beta)
            GravitationalWaveBayes.BatchedNUTS.tempered_logp_and_grad(base!, betas, pr; box=boxf)
        else
            nothing
        end
    elseif device == "cuda" && model isa SEOBNRv5IMRAlignedSpinF2
        # SEOBNRv5 IMR: table, likelihood and gradient on the device (any detectors on one grid); the stage
        # reweighting log-likelihoods too. Untempered + box prior -> tempered with the configured prior on the host.
        base!, loglik! = GravitationalWaveBayes.BatchedNUTS.seobnr_device_closures(data_for_grad, cfg; nwalkers=nwalkers,
            backend=GravitationalWaveBayes.BatchedNUTS.KernelAbstractions.get_backend(GravitationalWaveBayes.GPU.to_device(zeros(T, 1))))
        GravitationalWaveBayes.BatchedNUTS.tempered_logp_and_grad(base!, betas, pr)
    elseif device == "cuda"
        # GPU-staged gradient (single detector, PA-F2 models) has no beta/prior options: temper and add
        # the prior on the host (`tempered_logp_and_grad`, validated equal to the CPU path).
        (data_for_grad isa LikelihoodData ||
         model isa Union{IMRPhenomXAS,IMRPhenomXHM,IMRPhenomXP,IMRPhenomXPHM,SEOBNRv5PHM,GravitationalWaveforms.SEOBNRv5HM}) ||
            throw(ArgumentError("SMC on CUDA: GPU-staged path is single-detector only (except IMRPhenomXAS / XHM / XP / XPHM / SEOBNRv5PHM / SEOBNRv5HM)"))
        base! = gw_logp_and_grad_gpu_staged(data_for_grad, cfg, freqs; model=model, nparams=nrows, nwalkers=nwalkers,
                                            calibration=cal, calmarg=calm, box=boxf)
        if model isa SEOBNRv5PHM
            # stage-reweighting / independence-move log-likelihoods: the device likelihood alone (~3.5x cheaper than
            # logL + gradient at 1024 walkers; the same forward kernels as the gradient path)
            loglik! = GravitationalWaveBayes.BatchedNUTS.seobnr_phm_loglik(data_for_grad, cfg; nwalkers=nwalkers, calibration=cal, calmarg=calm, boxf=boxf,
                backend=GravitationalWaveBayes.BatchedNUTS.KernelAbstractions.get_backend(GravitationalWaveBayes.GPU.to_device(zeros(T, 1))))
        elseif model isa Union{IMRPhenomXHM,IMRPhenomXP,IMRPhenomXPHM,GravitationalWaveforms.SEOBNRv5HM}
            # stage-reweighting log-likelihoods from the device path too (the dense CPU XHM likelihood is ~100x slower);
            # particles stay inside the box, where its box prior is 0
            gscr = zeros(T, nrows, nwalkers)
            loglik! = (ll, pos) -> (_save_on_error(() -> base!(gscr, ll, pos, nothing; value_only=true), pos); ll)
        end
        GravitationalWaveBayes.BatchedNUTS.tempered_logp_and_grad(base!, betas, pr; box=boxf)
    else
        gw_logp_and_grad(data_for_grad, cfg, freqs; model=model, prior=pr, beta=betas)
    end
    fsh = T(get(smc_opts, :phase_shear, 0.0))
    fsh > 0 && ((loglik!, logp_and_grad!) = _phase_shear(loglik!, logp_and_grad!, fsh, pr))
    fsh > 0 && println("SMC: sampling the phase sheared along the time row (phase' = phase - pi ", fsh, " Hz x t)")
    amax = get(smc_opts, :spin_isotropic_max, nothing)
    if amax !== nothing
        (loglik!, logp_and_grad!) = _spin_isotropic(loglik!, logp_and_grad!, amax)
        println("SMC: LVK spin prior (|chi| uniform on [0, ", amax, "], isotropic", nrows >= 15 ?
                "): spin rows sampled as y, chi = amax |y|^2 y" : ", z-projection): chi_z rows sampled as y, the inverse CDF")
    end
    (loglik!, logp_and_grad!) = _support_guard(loglik!, logp_and_grad!, pr)
    # `sampler.smc_ind_batch = W` (IMRPhenomXHM / SEOBNRv5HM on CUDA): all of a draw's independence proposals evaluated
    # in one batch by a value-only closure W walkers wide (W divides smc_independence_tries x nwalkers; SEOBNRv5HM's is
    # built without tangent tables), wrapped like `loglik!`. Wider device calls amortize the latency-bound serial stages.
    ind_llk = nothing
    Wb = Int(get(smc_opts, :ind_batch, 0))
    if Wb > 0
        (device == "cuda" && model isa Union{IMRPhenomXHM,GravitationalWaveforms.SEOBNRv5HM}) ||
            throw(ArgumentError("sampler.smc_ind_batch: IMRPhenomXHM / SEOBNRv5HM on CUDA only"))
        ntot = Int(get(smc_opts, :independence_tries, 1)) * nwalkers
        ntot % Wb == 0 || throw(ArgumentError("sampler.smc_ind_batch = $Wb must divide smc_independence_tries x nwalkers = $ntot"))
        be = GravitationalWaveBayes.BatchedNUTS.KernelAbstractions.get_backend(GravitationalWaveBayes.GPU.to_device(zeros(T, 1)))
        # (held only through `fb_ref`, so `smc_ind_release` can drop it)
        fb_ref = Ref{Any}(model isa IMRPhenomXHM ?
            GravitationalWaveBayes.BatchedNUTS.xhm_binparallel_logp_and_grad(data_for_grad, cfg, freqs; backend=be, nwalkers=Wb,
                                                              setup=:device, box=boxf) :
            GravitationalWaveBayes.BatchedNUTS.seobnrhm_binparallel_logp_and_grad(data_for_grad, cfg, freqs; backend=be, nwalkers=Wb,
                                                                   box=boxf, value_build=true))
        ind_gb = zeros(T, nrows, Wb); ind_lb = zeros(T, Wb)
        nb_base = rw && @isdefined(xphm_base!) ? xphm_base! : nothing
        rel_g = zeros(T, nrows, nwalkers)
        raw = function (ll, X)
            fb! = fb_ref[]
            if fb! === nothing
                # released (`smc_ind_release`): the hybrid's own closure, nwalkers wide, value only
                for c in 1:(size(X, 2) ÷ nwalkers)
                    cols = ((c - 1) * nwalkers + 1):(c * nwalkers)
                    Xc = X[:, cols]; lc = zeros(T, nwalkers)
                    _save_on_error(() -> nb_base(rel_g, lc, Xc, nothing; value_only=true), Xc)
                    ll[cols] .= lc
                end
                return ll
            end
            for c in 1:(size(X, 2) ÷ Wb)
                cols = ((c - 1) * Wb + 1):(c * Wb)
                Xc = X[:, cols]
                _save_on_error(() -> fb!(ind_gb, ind_lb, Xc, nothing; value_only=true), Xc)
                ll[cols] .= ind_lb
            end
            return ll
        end
        # `sampler.smc_ind_release = true` (hybrid, 2026-10-15): the wide value-only closure is dropped (GC + pool reclaim)
        # before the first gradient call, whose forward kernels then reserve their stack (~42 KB per resident thread);
        # the NUTS stages' independence tries use the hybrid's own closure. Lets a 5120-wide batch fit on a 16 GB card.
        if Bool(get(smc_opts, :ind_release, false))
            (nb_base !== nothing && logp_and_grad! !== nothing) ||
                throw(ArgumentError("sampler.smc_ind_release needs the hybrid sampler (smc_moves = \"rw\", smc_nuts_beta)"))
            lpg_in = logp_and_grad!
            logp_and_grad! = function (g, lp, pos, mask)
                if fb_ref[] !== nothing
                    fb_ref[] = nothing
                    GC.gc(true); GravitationalWaveBayes.GPU.reclaim()
                    println("SMC: released the ", Wb, "-wide value-only closure before the first gradient call"); flush(stdout)
                end
                return lpg_in(g, lp, pos, mask)
            end
        end
        fsh > 0 && (raw = _phase_shear(raw, nothing, fsh, pr)[1])
        amax === nothing || (raw = _spin_isotropic(raw, nothing, amax)[1])
        ind_llk = _support_guard(raw, nothing, pr)[1]
        println("SMC: independence proposals evaluated in batches of ", Wb, " (", ntot ÷ Wb, " device calls per draw)")
    end
    # Learned mode jumps (`sampler.smc_mode_jumps = true`, `SMCCore.smc_sample`'s `mode_jumps`): jump rows by name
    # (`smc_mode_jump_rows`; default the intrinsic rows, time, phase and inclination the model samples), in the sampled
    # coordinates (sheared phase, spin rows as sampled). `smc_mode_jump_batch = m > 1`: the tries evaluated m per particle
    # in one call m x nwalkers wide (CPU: the likelihood itself; CUDA: a value-only staged closure that wide).
    mj_llk = nothing
    if Bool(get(smc_opts, :mode_jumps, false))
        rnames = nrows >= 15 ? vcat(_PARAM_NAMES_15, ["cal_z$(k)" for k in 1:nrows-15]) : _PARAM_NAMES_11
        want = haskey(smc_opts, :mode_jump_rows) ? String.(smc_opts.mode_jump_rows) :
               filter(in(rnames), _MODE_JUMP_ROWS)
        bad = filter(!in(rnames), want)
        isempty(bad) || throw(ArgumentError("sampler.smc_mode_jump_rows: unknown rows $bad (the model samples $rnames)"))
        rows = [findfirst(==(nm), rnames) for nm in want]
        smc_opts = merge(smc_opts, (; mode_jump_rows=rows))
        haskey(smc_opts, :mode_jump_learn) && (smc_opts = merge(smc_opts, (; mode_jump_learn=Symbol(smc_opts.mode_jump_learn))))
        haskey(smc_opts, :mode_jump_tail_frac) &&
            (smc_opts = merge(smc_opts, (; mode_jump_tail_frac=T.(collect(smc_opts.mode_jump_tail_frac)))))
        tnames = haskey(smc_opts, :mode_jump_tails) ? String.(smc_opts.mode_jump_tails) : String[]
        if haskey(smc_opts, :mode_jump_tails)        # tail quantities by name: "mc", "chi_eff" or any sampled row name
            tq = [_mode_jump_tail_quantity(String(nm), rnames, amax) for nm in smc_opts.mode_jump_tails]
            smc_opts = merge(smc_opts, (; mode_jump_tails=tq))
        end
        haskey(smc_opts, :mode_jump_scale) &&
            (smc_opts = merge(smc_opts, (; mode_jump_scale=(T(smc_opts.mode_jump_scale[1]), T(smc_opts.mode_jump_scale[2])))))
        mjb = Int(get(smc_opts, :mode_jump_batch, 1))
        if mjb > 1 && device == "cuda"
            model isa Union{IMRPhenomXP,IMRPhenomXPHM,IMRPhenomXHM,GravitationalWaveforms.SEOBNRv5HM} ||
                throw(ArgumentError("sampler.smc_mode_jump_batch > 1 on CUDA: IMRPhenomXP / XPHM / XHM / SEOBNRv5HM only"))
            wide! = gw_logp_and_grad_gpu_staged(data_for_grad, cfg, freqs; model=model, nparams=nrows,
                                                nwalkers=mjb * nwalkers, calibration=cal, calmarg=calm, box=boxf)
            gwide = zeros(T, nrows, mjb * nwalkers)
            mj_raw = (ll, pos) -> (_save_on_error(() -> wide!(gwide, ll, pos, nothing; value_only=true), pos); ll)
            fsh > 0 && (mj_raw = _phase_shear(mj_raw, nothing, fsh, pr)[1])
            amax === nothing || (mj_raw = _spin_isotropic(mj_raw, nothing, amax)[1])
            mj_llk = _support_guard(mj_raw, nothing, pr)[1]
        end
        println("SMC: learned mode jumps on rows ", join(want, ", "), " (", get(smc_opts, :mode_jump_tries, 12),
                " tries per draw", mjb > 1 ? ", evaluated $mjb per particle per call" : "", ", vectors from ",
                get(smc_opts, :mode_jump_learn, :kmeans),
                isempty(tnames) ? "" : " along $(join(tnames, ", "))", ")")
    end
    bounds = enforce_bounds ? _sampled_space_bounds(pr) : nothing
    nuts_cfg = NUTSConfig{T}(; max_treedepth=max_treedepth, nwarmup=0, bounds=bounds, quorum_opts...)
    # The configured prior: random-walk moves gate every likelihood evaluation with it; NUTS independence /
    # online-proposal moves use it with `loglik!` instead of a gradient call. NOT the package box (d_L <= 4 Gpc,
    # q >= 0.125): with the LVK d_L prior it froze the ~94% of particles outside it (R4/R2 stage 1, 2026-10-11).
    logprior! = (lp, pos) -> prior_logpdf!(lp, pr, pos)
    println("SMC: ", nwalkers, " particles, device ", device, rw ? ", random-walk moves (likelihood only)" : "")
    t0 = time()
    dump = Bool(get(smc_opts, :dump_stages, false))
    cb = function (stage, info)
        println("  SMC stage $stage$(info.nuts ? (info.latent ? " (NUTS, latent)" : " (NUTS)") : "")$(isnan(info.heuristic_steps[1]) ? "" : "  heuristic_steps(latent,x)=$(round.(info.heuristic_steps; sigdigits=3))")  beta=$(round(info.beta; sigdigits=4))  logZ=$(round(info.logZ; digits=3))  ",
                "draws=$(info.ndraws)  step=$(round(info.step_size; sigdigits=3))  accept=$(round(info.accept; digits=2))  ",
                "grad_evals=$(info.total_leap)  loglik_evals=$(info.total_ll)  unique_ancestors=$(info.unique)  slowest_move=$(round(info.slowest_move; digits=2))",
                "(row $(info.slowest_row))  indep_accept=$(round(info.independence_accept; digits=3))  de_accept=$(round(info.de_accept; digits=3))  move_accept=$(round(info.move_accept; digits=3))  $(info.mode_jumps > 0 || info.mode_jumps_added > 0 ? "mode_jumps=$(info.mode_jumps)(+$(info.mode_jumps_added) from $(info.mode_jump_cells) cells or tail quantities) jump_accept=$(round(info.mode_jump_accept; digits=4)) jump_evals=$(info.mode_jump_evals)  " : "")block_evals=$(info.block_evals)  elapsed=$(round(time() - t0; digits=1))s")
        if any(>(0), _LIK_TIMERS)
            println("    likelihood closure (s): setup ", round(_LIK_TIMERS[1]; digits=1), ", host mode factors ",
                    round(_LIK_TIMERS[2]; digits=1), ", bin sums ", round(_LIK_TIMERS[3]; digits=1), ", host combine ",
                    round(_LIK_TIMERS[4]; digits=1))
            fill!(_LIK_TIMERS, 0.0)
        end
        cs = _COMPACT_STATS[]
        if cs !== nothing && !isempty(cs.calls)
            nc = sum(values(cs.calls))
            println("    NUTS gradient calls: ", nc, ", mean active fraction ", round(cs.active[] / (nc * nwalkers); digits=3),
                    ", calls per width ", sort(collect(cs.calls)), ", active-fraction histogram (0:0.05:1) ", cs.hist)
            empty!(cs.calls); cs.active[] = 0; fill!(cs.hist, 0)
        end
        if haskey(info, :time_split)
            ts = info.time_split; tot = get(ts, :total, 0.0)
            other = tot - sum(v for (k, v) in ts if k !== :total; init=0.0)
            println("    time split (s): ", join(["$k $(round(v; digits=1))" for (k, v) in sort(collect(ts); by=last, rev=true)
                                                 if k !== :total], ", "), ", other $(round(other; digits=1)) of $(round(tot; digits=1))")
        end
        flush(stdout)
        if dump && outdir !== nothing                  # `sampler.smc_dump_stages`: the population after each stage
            mkpath(joinpath(outdir, "stages"))
            Serialization.serialize(joinpath(outdir, "stages", "stage_$stage.jls"),
                                    (; positions=copy(info.positions), ll=copy(info.ll), beta=info.beta))
        end
    end
    rw && (smc_opts = merge(smc_opts, (; moves=:rw)))
    sopts = Base.structdiff(smc_opts, NamedTuple{(:reference, :reference_inflate, :reference_alpha,
                                                  :reference_kind, :reference_kernels, :reference_neighbours,
                                                  :reference_local_h, :online_kde, :online_alpha, :online_kernels, :phase_shear, :dump_stages, :reference_prior_rows, :sky_ring, :sky_phase, :incl_flip,
                                                  :online_flow, :flow_steps, :flow_layers, :flow_hidden, :flow_warm, :flow_device, :flow_batch,
                                                  :extrinsic_sweeps, :reference_time_shift, :spin_isotropic_max, :calibration, :calmarg, :ind_batch, :nuts_narrow, :ind_release)})
    online = get(smc_opts, :online_flow, false) ? _online_flow_proposal(pr, smc_opts, rng) :
             get(smc_opts, :online_kde, false) ? _online_kde_proposal(pr, lik_opts, smc_opts, rng) : nothing
    online === nothing || println("SMC: online ", get(smc_opts, :online_flow, false) ? "normalizing-flow" : "local-KDE",
                                  " proposal for independence moves (rebuilt every stage)")
    rw_moves = if Bool(get(smc_opts, :sky_ring, false))
        rw || throw(ArgumentError("sampler.smc_sky_ring needs smc_moves = \"rw\""))
        (data_for_grad isa AbstractVector && length(data_for_grad) >= 2) ||
            throw(ArgumentError("sampler.smc_sky_ring needs at least two detectors"))
        _sky_moves(data_for_grad, get(lik_opts, :time_reference, 0), pr; phase_sign=Int(get(smc_opts, :sky_phase, 0)),
                   incl_flip=get(smc_opts, :incl_flip, nothing))
    else
        ()
    end
    nsw = Int(get(smc_opts, :extrinsic_sweeps, 0))
    block = nsw > 0 ? (model isa GravitationalWaveBayes.IMRTableModel && device == "cpu" ||
                       throw(ArgumentError("smc_extrinsic_sweeps needs an IMR-table model on the CPU"));
                       _imr_extrinsic_block_move(data_for_grad, freqs, cfg, model, pr, lik_opts; nsweeps=nsw)) : nothing
    res = smc_sample(positions, loglik!, logp_and_grad!, betas; cfg=nuts_cfg, rng=rng,
                     mass_blocks=get(mass_opts, :mass_blocks, nothing), callback=cb, reference=reference,
                     online_proposal=online, block_move=block, logprior! = logprior!, rw_moves=rw_moves, ind_llk=ind_llk,
                     mode_jump_llk=mj_llk, sopts...)
    elapsed = time() - t0
    println("SMC done: ", res.nstages, " stages, logZ = ", round(res.logZ; digits=3), ", gradient evaluations = ",
            res.n_leapfrog, " (", round(res.n_leapfrog / nwalkers; digits=1), " per particle), likelihood evaluations ",
            res.n_loglik, " (", round(res.n_loglik / nwalkers; digits=1), " per particle), block (cached-waveform) evaluations ",
            res.block_evals, ", ",
            "unique ancestors ", res.ancestors_unique, "/", nwalkers, ", independence acceptance ",
            round(res.independence_accept; digits=3), ", wall ", round(elapsed; digits=1), " s")
    if outdir !== nothing
        mkpath(outdir)
        open(joinpath(outdir, "smc_summary.txt"), "w") do io
            println(io, "logZ = ", res.logZ)
            println(io, "nstages = ", res.nstages)
            println(io, "n_leapfrog = ", res.n_leapfrog)
            println(io, "n_loglik = ", res.n_loglik)
            println(io, "moves = ", rw ? "rw" : "nuts")
            println(io, "wall_seconds = ", elapsed)
            println(io, "ancestors_unique = ", res.ancestors_unique)
            println(io, "beta_schedule = ", res.beta_schedule)
            println(io, "accept = ", res.accept)
            println(io, "independence_accept = ", res.independence_accept)
            println(io, "block_evals = ", res.block_evals)
            println(io, "mode_jumps = ", res.mode_jumps)
            println(io, "mode_jump_accept = ", res.mode_jump_accept)
            println(io, "mode_jump_evals = ", res.mode_jump_evals)
        end
    end
    if fsh > 0                                   # outputs in the physical phase
        _phase_unshear!(res.positions, fsh, pr)
        for k in axes(res.chain, 3)
            _phase_unshear!(view(res.chain, :, :, k), fsh, pr)
        end
    end
    return (; positions=res.positions, logp=res.logp, step_size=fill(res.step_size, nwalkers), chain=res.chain,
            inv_mass=res.metric, nwarmup_eff=0, pe_cfg=cfg, data_for_grad, logZ=res.logZ,
            n_leapfrog=res.n_leapfrog), nrows
end

# Phase-time shear (`sampler.smc_phase_shear = fbar`, Hz): the sampler sees phase' = phase - pi fbar t (row 7 against
# the time row 6, wrapped into the phase window), a volume-preserving map that keeps the uniform periodic phase prior.
# For the dominant m = 2 content a time shift dt is compensated by a phase shift pi fbar dt (fbar ~ the SNR-weighted
# frequency), a ridge that wraps the 2 pi phase several times across the population's time spread: no population
# covariance or KDE kernel captures it, and the moves' steps collapse once beta resolves it (the PHM SMC with t_ref:
# slowest row t_ref at beta ~0.095). Wraps `loglik!` and `logp_and_grad!` (d/dt gains pi fbar d/dphase).
function _phase_shear(loglik!, lpg!, fsh, pr)
    lo, hi, _ = prior_bounds(pr)
    plo, P, c = lo[7], hi[7] - lo[7], pi * fsh
    function shear(pos)
        Y = copy(pos)
        @inbounds for w in axes(Y, 2)
            Y[7, w] = plo + mod(pos[7, w] + c * pos[6, w] - plo, P)
        end
        return Y
    end
    ll2 = (ll, pos) -> loglik!(ll, shear(pos))
    lpg2 = lpg! === nothing ? nothing : function (g, lp, pos, mask)
        lpg!(g, lp, shear(pos), mask)
        @inbounds for w in axes(g, 2)
            g[6, w] += c * g[7, w]
        end
        return nothing
    end
    return ll2, lpg2
end
# Support guard (outermost wrapper, sampled coordinates): columns outside the prior's support are evaluated at a
# clamped in-support point and returned as logL = -Inf, gradient 0. NUTS trajectories (unlike the RW / DE /
# independence moves, which substitute the current state) hand the closures arbitrary points, and the device kernels
# throw on unphysical ones -- e.g. |y| > 1 maps to |chi| = amax |y|^3 > 1, sqrt of a negative number (R4/R2, 2026-10-11).
function _support_guard(loglik!, lpg!, pr)
    lo, hi, per = prior_bounds(pr)
    function safe(pos)
        sup = prior_support_vec(pr, pos)
        bad = .!isfinite.(sup)
        any(bad) || return pos, bad
        Y = copy(pos)
        @inbounds for w in findall(bad)
            for r in axes(Y, 1)
                x = Y[r, w]
                isfinite(x) || (x = (lo[r] + hi[r]) / 2)
                Y[r, w] = per[r] ? lo[r] + mod(x - lo[r], hi[r] - lo[r]) : clamp(x, lo[r], hi[r])
            end
            if size(Y, 1) >= 15
                for (rows, a) in (((3, 12, 13), pr.spin_max[1]), ((4, 14, 15), pr.spin_max[2]))
                    n = sqrt(sum(r -> Y[r, w]^2, rows))
                    n > a * (1 - 1e-9) && for r in rows; Y[r, w] *= a * (1 - 1e-9) / n; end
                end
            end
        end
        return Y, bad
    end
    ll2 = function (ll, pos)
        Y, bad = safe(pos)
        loglik!(ll, Y)
        ll[bad] .= -Inf
        return ll
    end
    lpg2 = lpg! === nothing ? nothing : function (g, lp, pos, mask)
        Y, bad = safe(pos)
        lpg!(g, lp, Y, mask)
        lp[bad] .= -Inf
        g[:, bad] .= 0
        return nothing
    end
    return ll2, lpg2
end
# LVK spin prior (`prior.spins = "isotropic"`, `prior.spin_max` = one cap or per body [a1, a2]): the sampler's spin
# rows are y in the unit ball, the likelihood sees chi = amax |y|^2 y (`GravitationalWaveBayes.isotropic_spins!`; magnitude uniform on
# [0, amax], isotropic); aligned-spin (11-row) models: y in [-1, 1] -> the isotropic prior's z-projection. Wraps
# `loglik!` and `logp_and_grad!` (gradient pulled back through the Jacobian); the prior (flat in y) is unchanged.
# Recorded chains / stage dumps hold y: apply `isotropic_spins!` with `_spin_amax(config_used.toml)`.
# A mode-jump tail quantity (`sampler.smc_mode_jump_tails`) of the population in the SAMPLED coordinates (columns =
# particles; sheared phase, spin rows as y): "chi_eff" = (chi1z + q chi2z) / (1 + q) with the spins mapped through the
# LVK isotropic spin map when `amax` is set (`isotropic_spins!`; 15 rows: chi = amax |y|^2 y; 11 rows: z-projection),
# else the rows themselves; any other name = that sampled row ("mc" = row 1).
function _mode_jump_tail_quantity(name::String, rnames, amax)
    if name == "chi_eff"
        return let a = amax
            function (X)
                C = a === nothing ? X : isotropic_spins!(similar(X), X, a)
                return [(C[3, w] + X[2, w] * C[4, w]) / (1 + X[2, w]) for w in axes(X, 2)]
            end
        end
    end
    r = findfirst(==(name), rnames)
    r === nothing && throw(ArgumentError("sampler.smc_mode_jump_tails: unknown quantity \"$name\" (chi_eff or one of $rnames)"))
    return let r = r; X -> X[r, :]; end
end
function _spin_amax(cfg)
    pr = get(cfg, "prior", Dict{String,Any}())
    get(pr, "spins", "ball") == "isotropic" || return nothing
    a = get(pr, "spin_max", 0.99)
    a isa AbstractVector && length(a) == 2 && return (Float64(a[1]), Float64(a[2]))
    a isa Real || throw(ArgumentError("prior.spin_max must be a number or [a1, a2]"))
    return Float64(a)
end
function _spin_isotropic(loglik!, lpg!, amax)
    mapped(pos) = isotropic_spins!(similar(pos), pos, amax)
    ll2 = (ll, pos) -> loglik!(ll, mapped(pos))
    lpg2 = lpg! === nothing ? nothing : function (g, lp, pos, mask)
        lpg!(g, lp, mapped(pos), mask)
        isotropic_spins_pullback!(g, pos, amax)
        return nothing
    end
    return ll2, lpg2
end
# The plain (unmarginalized) likelihood for `reconstruct_marginalized!`: SEOBNRv5PHM on CUDA, or with time_ref_frequency /
# phase_ref_frequency, uses the device likelihood (on the CPU backend without CUDA; the CPU reference path implements
# neither option); others the CPU path.
# With the LVK spin map (`spin_amax`) the closure maps the sampled y to chi first; with `calibration` (IMRPhenomXPHM on
# the device) the bin-parallel closure evaluates the calibrated likelihood (inside the box its prior term is 0).
function _recon_loglik(model, use_gpu, data, pe_cfg, nw; calibration=nothing, calmarg=nothing, spin_amax=nothing,
                       prior=nothing)
    # with a configured prior: its support replaces the package box (the reconstruction tries d_L up to the prior's hi)
    bk = prior === nothing ? (;) : (; box=(pos -> prior_support_vec(prior, pos)))
    plain = with_likelihood_options(pe_cfg; marginalize_phase=false, marginalize_distance=false)
    backend() = GravitationalWaveBayes.BatchedNUTS.KernelAbstractions.get_backend(GravitationalWaveBayes.GPU.to_device(zeros(1)))
    ll = if model isa SEOBNRv5PHM && (use_gpu || pe_cfg.time_ref_frequency > 0 || pe_cfg.phase_ref_frequency > 0)
        GravitationalWaveBayes.BatchedNUTS.seobnr_phm_loglik(data, plain; nwalkers=nw,
                                              backend=use_gpu ? backend() : GravitationalWaveBayes.BatchedNUTS.KernelAbstractions.CPU(),
                                              calibration, calmarg,
                                              (prior === nothing ? (;) : (; boxf=bk.box))...)
    elseif model isa IMRPhenomXPHM && use_gpu
        f! = GravitationalWaveBayes.BatchedNUTS.xphm_binparallel_logp_and_grad(data, plain, _representative_freqs(data); model,
                                                                backend=backend(), nwalkers=nw, calibration, calmarg,
                                                                setup=:device, bk...)
        (out, pos) -> (f!(zeros(size(pos)), out, pos, nothing; value_only=true); out)
    elseif model isa IMRPhenomXHM && use_gpu && pe_cfg.phase_ref_frequency > 0
        # (the CPU XHM likelihood keeps the phase at the lowest data frequency: the device path, value only)
        calibration === nothing && calmarg === nothing ||
            throw(ArgumentError("IMRPhenomXHM with phase_ref_frequency: no calibration support"))
        f! = GravitationalWaveBayes.BatchedNUTS.xhm_binparallel_logp_and_grad(data, plain, _representative_freqs(data);
                                                              backend=backend(), nwalkers=nw, setup=:device, bk...)
        (out, pos) -> (f!(zeros(size(pos)), out, pos, nothing; value_only=true); out)
    elseif model isa GravitationalWaveforms.SEOBNRv5HM && use_gpu
        # (no CPU batched SEOBNRv5HM likelihood: the device path, value only)
        calibration === nothing && calmarg === nothing ||
            throw(ArgumentError("SEOBNRv5HM on the device has no calibration support"))
        f! = GravitationalWaveBayes.BatchedNUTS.seobnrhm_binparallel_logp_and_grad(data, plain, _representative_freqs(data);
                                                                    backend=backend(), nwalkers=nw, bk...)
        (out, pos) -> (f!(zeros(size(pos)), out, pos, nothing; value_only=true); out)
    else
        calibration === nothing || throw(ArgumentError("calibration needs the IMRPhenomXPHM device likelihood"))
        calmarg === nothing || throw(ArgumentError("calibration marginalization needs the SEOBNRv5PHM device likelihood"))
        nothing
    end
    (spin_amax === nothing || ll === nothing) && return ll
    return (out, pos) -> ll(out, isotropic_spins!(similar(pos), pos, spin_amax))
end

# Sky-ring move (`sampler.smc_sky_ring = true`, RW SMC, >= 2 detectors): rotate the sky position (rows 8 ra, 9 sin_dec)
# about the baseline of the first two detectors by an angle drawn from a symmetric law (uniform on (-pi, pi] half the
# time, N(0, 0.3^2) otherwise). The rotation keeps the arrival-time difference between the two detectors, and the time
# row is the arrival time at the reference detector (`likelihood.time_reference`, one of the two), so it moves along
# the two-detector time-delay ring that random-walk steps crawl around. Rotations preserve area on the sphere, and area
# is flat in (ra, sin_dec), so the proposal is symmetric there (plain MH, `SMCCore.smc_sample`'s `rw_moves`).
#
# `phase_det` + `phase_sign = +-1`: also shift the phase row by phase_sign (arg R(x') - arg R(x)) / 2, R = F+ (1 + c^2) / 2
# - i Fx c the (2,2) response at that detector (c = cos inclination, row 10; psi row 11), keeping its signal phase:
# the phase is tied to sky / psi / inclination along a wrapped manifold, and an uncompensated sky move lands off it.
# A shift by a function of the other coordinates keeps volume and the reverse rotation undoes it: still symmetric.
@inline function _resp22_arg(det, ra, sd, psi, c, g)
    fp, fx = GravitationalWaveData.antenna_response_gmst(det, ra, asin(clamp(sd, -1.0, 1.0)), psi, g)
    return angle(complex(fp * (1 + c^2) / 2, -fx * c))
end
function _sky_ring_move(loc1, loc2, g, pr; phase_det=nothing, phase_sign=0)
    lo, hi, _ = prior_bounds(pr)
    b = collect(Float64, loc2) .- collect(Float64, loc1)
    b ./= sqrt(sum(abs2, b))
    ralo, P = lo[8], hi[8] - lo[8]
    plo, PP = lo[7], hi[7] - lo[7]
    return function (xp, pos, rng)
        xp .= pos
        @inbounds for w in axes(pos, 2)
            phi = rand(rng) < 0.5 ? pi * (2 * rand(rng) - 1) : 0.3 * randn(rng)
            sd = pos[9, w]; cd = sqrt(max(1 - sd^2, 0.0)); h = pos[8, w] - g
            v = (cd * cos(h), cd * sin(h), sd)                         # Earth-fixed unit vector (`_time_delay_scalar`)
            bv = b[1] * v[1] + b[2] * v[2] + b[3] * v[3]
            bxv = (b[2] * v[3] - b[3] * v[2], b[3] * v[1] - b[1] * v[3], b[1] * v[2] - b[2] * v[1])
            c, s = cos(phi), sin(phi)
            u = ntuple(i -> v[i] * c + bxv[i] * s + b[i] * bv * (1 - c), 3)   # Rodrigues
            xp[9, w] = clamp(u[3], -1.0, 1.0)
            xp[8, w] = ralo + mod(atan(u[2], u[1]) + g - ralo, P)
            if phase_det !== nothing && phase_sign != 0
                a0 = _resp22_arg(phase_det, pos[8, w], pos[9, w], pos[11, w], pos[10, w], g)
                a1 = _resp22_arg(phase_det, xp[8, w], xp[9, w], pos[11, w], pos[10, w], g)
                xp[7, w] = plo + mod(pos[7, w] + phase_sign * (a1 - a0) / 2 - plo, PP)
            end
        end
        return xp
    end
end

# Inclination flip (`sampler.smc_incl_flip`): cos(inclination) -> -cos (row 10), psi -> psi_map(psi) (row 11; one of
# :same, :neg, :plus_quarter, :quarter_minus = psi, -psi, psi + pi/2, pi/2 - psi, wrapped to the psi window), and the
# phase shifted by phase_sign (arg R(x') - arg R(x)) / 2 at `phase_det` (as the sky move) -- face-on and face-off
# signals differ by the sense of circular polarization, a degeneracy NUTS can only diffuse across. Each psi map is an
# involution and the phase shift a function of the other rows: a deterministic, volume-preserving involution, so
# plain MH is exact.
function _incl_flip_move(det, g, pr; psi_map=:neg, phase_sign=-1)
    lo, hi, _ = prior_bounds(pr)
    plo, PP = lo[7], hi[7] - lo[7]
    slo, SP = lo[11], hi[11] - lo[11]
    fpsi = psi_map === :same ? (x -> x) : psi_map === :neg ? (x -> -x) : psi_map === :plus_quarter ? (x -> x + pi / 2) :
           psi_map === :quarter_minus ? (x -> pi / 2 - x) : throw(ArgumentError("psi_map $psi_map"))
    return function (xp, pos, rng)
        xp .= pos
        @inbounds for w in axes(pos, 2)
            xp[10, w] = -pos[10, w]
            xp[11, w] = slo + mod(fpsi(pos[11, w]) - slo, SP)
            if phase_sign != 0
                a0 = _resp22_arg(det, pos[8, w], pos[9, w], pos[11, w], pos[10, w], g)
                a1 = _resp22_arg(det, xp[8, w], xp[9, w], xp[11, w], xp[10, w], g)
                xp[7, w] = plo + mod(pos[7, w] + phase_sign * (a1 - a0) / 2 - plo, PP)
            end
        end
        return xp
    end
end

# Sky reflection (>= 3 detectors): mirror the sky direction through the plane of the detectors, n' = n - 2 (n.m) m with
# m the plane's unit normal. With three detectors m is perpendicular to every baseline, so all arrival-time differences
# are kept exactly: the ring collapses to two mirror-image patches and this jumps between them (LALInference's sky
# reflection). With four or more, m is the normal of the least-squares plane (an approximate degeneracy; the move stays
# exact). A deterministic area-preserving involution on the sphere: plain MH applies.
function _detector_plane_normal(locs)
    R = reduce(hcat, [collect(Float64, l) for l in locs])
    if size(R, 2) == 3
        a, b = R[:, 2] .- R[:, 1], R[:, 3] .- R[:, 1]
        m = [a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1]]
    else
        C = R .- sum(R; dims=2) ./ size(R, 2)
        m = LinearAlgebra.eigen(LinearAlgebra.Symmetric(C * C')).vectors[:, 1]   # smallest eigenvalue
    end
    return m ./ sqrt(sum(abs2, m))
end
function _sky_reflect_move(locs, g, pr)
    lo, hi, _ = prior_bounds(pr)
    m = _detector_plane_normal(locs)                   # assigned once: captured below (no Core.Box)
    ralo, P = lo[8], hi[8] - lo[8]
    return function (xp, pos, rng)
        xp .= pos
        @inbounds for w in axes(pos, 2)
            sd = pos[9, w]; cd = sqrt(max(1 - sd^2, 0.0)); h = pos[8, w] - g
            v = (cd * cos(h), cd * sin(h), sd)
            mv = m[1] * v[1] + m[2] * v[2] + m[3] * v[3]
            u = ntuple(i -> v[i] - 2 * mv * m[i], 3)
            xp[9, w] = clamp(u[3], -1.0, 1.0)
            xp[8, w] = ralo + mod(atan(u[2], u[1]) + g - ralo, P)
        end
        return xp
    end
end

# The RW-SMC sky moves (`sampler.smc_sky_ring = true`): the ring rotation about the baseline of the two most sensitive
# detectors (inspiral-weighted sensitivity sum f^(-7/3) / S_n over the band), plus the reflection through the detector
# plane with three or more detectors. Both keep the arrival-time differences; the arrival times themselves stay put
# only if the time row is referenced to a detector of the pair (`likelihood.time_reference`), else a warning.
function _sky_pair(dets)
    score = map(dets) do d
        ok = [isfinite(s) && s > 0 for s in d.psd]
        sum(d.freqs[ok] .^ (-7 / 3) ./ d.psd[ok])
    end
    p = sortperm(score; rev=true)
    return p[1], p[2]
end
function _sky_moves(dets, tref, pr; phase_sign=0, incl_flip=nothing)
    i, j = _sky_pair(dets)
    g = GravitationalWaveBayes.BatchedNUTS.gmst(dets[1].gps_time)
    names = [d.detector.prefix for d in dets]
    tref in (i, j) || @warn "smc_sky_ring: the time row is referenced to " *
        (tref == 0 ? "the geocenter" : names[tref]) * ", not to $(names[i]) or $(names[j]); sky moves then shift the " *
        "arrival times and are accepted less often (set likelihood.time_reference to one of them)"
    moves = Any[_sky_ring_move(GravitationalWaveData.geometry(dets[i].detector).location, GravitationalWaveData.geometry(dets[j].detector).location, g, pr;
                               phase_det=dets[i].detector, phase_sign=phase_sign)]
    msg = "SMC: sky-ring moves about the $(names[i])-$(names[j]) baseline" *
          (phase_sign == 0 ? "" : " (phase kept at $(names[i]), sign $phase_sign)")
    if incl_flip !== nothing                     # `smc_incl_flip = "<psi map>"` (see `_incl_flip_move`)
        push!(moves, _incl_flip_move(dets[i].detector, g, pr; psi_map=Symbol(incl_flip),
                                     phase_sign=phase_sign == 0 ? -1 : phase_sign))
        msg *= ", inclination flips (psi map $incl_flip)"
    end
    if length(dets) >= 3
        push!(moves, _sky_reflect_move([GravitationalWaveData.geometry(d.detector).location for d in dets], g, pr))
        msg *= ", reflections through the plane of " * join(names, "-") * (length(dets) > 3 ? " (least squares)" : "")
    end
    println(msg)
    return Tuple(moves)
end

# Per-curve log-likelihoods for `reconstruct_marginalized!(...; curve_loglik)` (calibration marginalization with the
# distance marginalized): `marg` the sampler's per-curve distance-marginalized values, `plain` the per-curve plain ones
# (both ncurves x nwalkers). SEOBNRv5PHM / IMRPhenomXPHM device likelihoods (the KA CPU backend without CUDA).
function _recon_curve_loglik(model, use_gpu, data, pe_cfg, nw, C; prior=nothing, spin_amax=nothing)
    model isa Union{SEOBNRv5PHM,IMRPhenomXPHM} ||
        throw(ArgumentError("calibration marginalization needs the SEOBNRv5PHM / IMRPhenomXPHM device likelihood"))
    be = use_gpu ? GravitationalWaveBayes.BatchedNUTS.KernelAbstractions.get_backend(GravitationalWaveBayes.GPU.to_device(zeros(1))) :
                   GravitationalWaveBayes.BatchedNUTS.KernelAbstractions.CPU()
    plain = with_likelihood_options(pe_cfg; marginalize_phase=false, marginalize_distance=false)
    marg = with_likelihood_options(pe_cfg; marginalize_phase=false, marginalize_distance=true)
    box = pos -> prior_support_vec(prior, pos)
    mk(c) = if model isa SEOBNRv5PHM
        f = GravitationalWaveBayes.BatchedNUTS.seobnr_phm_loglik(data, c; nwalkers=nw, backend=be, calmarg=C,
                                                  (prior === nothing ? (;) : (; boxf=box))...)
        (LK, pos) -> (f(zeros(size(pos, 2)), pos; curve_ll=LK); LK)
    else
        g! = GravitationalWaveBayes.BatchedNUTS.xphm_binparallel_logp_and_grad(data, c, _representative_freqs(data); model, backend=be,
                 nwalkers=nw, calmarg=C, setup=:device, (prior === nothing ? (;) : (; box))...)
        (LK, pos) -> (g!(zeros(size(pos)), zeros(size(pos, 2)), pos, nothing; curve_ll=LK, value_only=true); LK)
    end
    wrap(f) = spin_amax === nothing ? f : ((LK, pos) -> f(LK, isotropic_spins!(similar(pos), pos, spin_amax)))
    return (; ncurves=size(C, 1), marg=wrap(mk(marg)), plain=wrap(mk(plain)))
end

# calibration_draws.jls from the curve indices of the joint (curve, d_L) reconstruction (nwalkers x ndraws)
function _write_calmarg_draws(outdir, kidx, calmarg)
    cp = calmarg.prior
    nodes = zeros(length(cp.mu), size(kidx)...)
    for i in CartesianIndices(kidx)
        nodes[:, i] .= cp.mu .+ cp.sigma .* calmarg.Z[:, kidx[i]]
    end
    serialize(joinpath(outdir, "calibration_draws.jls"), (; nodes, curve=kidx, mu=cp.mu, sigma=cp.sigma, nn=cp.nn,
                                                          ncurves=size(calmarg.C, 1)))
    println("Calibration posterior: the curve of each joint (curve, d_L) draw (", length(unique(kidx)), " distinct of ",
            size(calmarg.C, 1), ") -> calibration_draws.jls")
    return nothing
end

# Calibration marginalization WITHOUT distance marginalization (d_L sampled): per sample a curve given (theta, d_L).
# (With the distance marginalized, `reconstruct_marginalized!(...; curve_loglik)` draws (curve, d_L) jointly instead:
# a curve drawn given a d_L from the wrong conditional is biased with it.)
function _calmarg_curve_draws(outdir, chain, data, pe_cfg, calmarg, prior, spin_amax, rng; model=SEOBNRv5PHM())
    nr, nw, nd = size(chain)
    plain = with_likelihood_options(pe_cfg; marginalize_phase=false, marginalize_distance=false)
    backend = GravitationalWaveBayes.BatchedNUTS.KernelAbstractions.get_backend(GravitationalWaveBayes.GPU.to_device(zeros(1)))
    f = if model isa SEOBNRv5PHM
        GravitationalWaveBayes.BatchedNUTS.seobnr_phm_loglik(data, plain; nwalkers=nw, backend, calmarg=calmarg.C,
                                              boxf=(pos -> prior_support_vec(prior, pos)))
    else                        # IMRPhenomXPHM bin-parallel (logL + gradient closure; the gradient is discarded)
        g! = GravitationalWaveBayes.BatchedNUTS.xphm_binparallel_logp_and_grad(data, plain, _representative_freqs(data); model, backend,
                 nwalkers=nw, calmarg=calmarg.C, setup=:device,
                 box=(pos -> prior_support_vec(prior, pos)))
        gs = zeros(15, nw)
        (ll, pos; curve_ll) -> (g!(gs, ll, pos, nothing; curve_ll); ll)
    end
    nc = size(calmarg.C, 1); LK = zeros(nc, nw); ll = zeros(nw)
    cp = calmarg.prior
    nodes = zeros(length(cp.mu), nw, nd); kidx = zeros(Int, nw, nd)
    for d in 1:nd
        pos = chain[1:15, :, d]
        spin_amax === nothing || (pos = isotropic_spins!(similar(pos), pos, spin_amax))
        f(ll, pos; curve_ll=LK)
        for w in 1:nw
            lw = view(LK, :, w); m = maximum(lw)
            p = exp.(lw .- m); p ./= sum(p)
            k = something(findfirst(>=(rand(rng)), cumsum(p)), nc)
            kidx[w, d] = k
            nodes[:, w, d] .= cp.mu .+ cp.sigma .* calmarg.Z[:, k]
        end
    end
    serialize(joinpath(outdir, "calibration_draws.jls"), (; nodes, curve=kidx, mu=cp.mu, sigma=cp.sigma, nn=cp.nn,
                                                          ncurves=nc))
    println("Calibration posterior: one response curve per sample drawn (", length(unique(kidx)), " distinct of ", nc,
            ") -> calibration_draws.jls")
    return nothing
end

function _phase_unshear!(X, fsh, pr)
    lo, hi, _ = prior_bounds(pr)
    @inbounds for w in axes(X, 2)
        X[7, w] = lo[7] + mod(X[7, w] + pi * fsh * X[6, w] - lo[7], hi[7] - lo[7])
    end
    return X
end

"""
    _run_nuts_relative_binning(data, model; ..., rb_nbins, rb_max_tc_deviation, rb_max_extrinsic_phase)

Same contract as `_run_nuts`, but builds and uses a `RelativeBinning.RelativeBinningSetup` for
post-warmup sampling instead of the dense kernel throughout. Solves the chicken-and-egg problem
`scripts/configs/README.md` names as the reason relative binning wasn't originally exposed here:
`build_relative_binning` needs a real physical-space reference point, which this script has no
principled way to default -- so this function RUNS the dense-kernel warmup phase first (exactly
the same warmup any dense-only run would do: real dual-averaging step-size adaptation over
`nwarmup` draws), uses ITS OWN resulting position (averaged across walkers, converted to physical)
as the reference, builds the binned kernel from that, then continues sampling on the binned
kernel for the remaining draws -- two `nuts_sample!` calls, not one, the second starting from
where the first left off (final positions AND the dual-averaging-converged step size, frozen via
`nwarmup=0`; see `NUTSCore.nuts_sample!`'s own `nwarmup=0` semantics). This is the SAME
dense-then-binned safety discipline `batched_nuts!`'s `dense_warmup` keyword documents, just
composed from two separate `nuts_sample!` calls instead of one call given two closures up front
-- `dense_warmup` itself can't be used directly here because it requires `relative_binning` to
already exist BEFORE warmup starts, which is exactly what this function has no way to provide.

Single-detector only (`data::LikelihoodData`), matching `RelativeBinning.build_relative_binning`'s
own single-detector method and this script's existing SEOBNR single-detector-only scoping (see
`_run_nuts`'s own comment) -- a length-1 `Vector{LikelihoodData}` is unwrapped the same way.

`chain` (when `record_chain=true`) is the concatenation of both phases' draws along the step axis
(`(nparams, nwalkers, nwarmup + nsteps_binned)`), so diagnostics/output see one continuous chain,
not two.

**Resume (2026-09-18)**: unlike `_run_nuts`'s single `nuts_sample!` call, resuming here has to know
WHICH phase a killed run was in, and Phase B additionally needs the SAME relative-binning reference
point Phase A produced (not a freshly-recomputed one) to rebuild an identical
`RelativeBinningSetup` -- so this function persists that reference point
([`_save_relative_binning_ref`](@ref)) the moment Phase A finishes, and branches on the checkpoint's
own `label` (via [`_load_checkpoint`](@ref)) to decide what to resume:
- label is Phase B's (`"... binned"`): Phase A already ran (before the interruption) -- load the
  persisted reference point, rebuild `rb` from it, and resume ONLY the remaining Phase B draws.
  Phase A is not re-run at all.
- label is Phase A's (`"... dense-warmup"`) and its `it < nwarmup`: resume the remaining warmup
  draws (same mechanism as `_run_nuts`'s resume), then proceed through Phase B fresh from THAT
  result -- Phase A's real output, whether the process ran it in one sitting or two, is what
  Phase B's reference point is built from either way, so this stays exactly as correct as an
  uninterrupted run.
- label is Phase A's and its `it >= nwarmup`: a narrow boundary case -- Phase A's draws all
  finished, but the process died before `_save_relative_binning_ref` (or Phase B's own first
  checkpoint) ever got written. The checkpoint IS Phase A's true final state, so it's reused
  directly as `warmup_result` with zero further `nuts_sample!` calls, rather than either re-running
  Phase A (wasted work) or refusing to resume at all.
- no checkpoint at outdir (or `resume=false`): a fresh start, exactly the pre-2026-09-18 behavior.
"""
function _run_nuts_relative_binning(data, model; nwalkers, nsteps, step_size, nwarmup, max_treedepth,
                                     rng, enforce_bounds, record_chain,
                                     rb_nbins, rb_max_tc_deviation, rb_max_extrinsic_phase,
                                     outdir=nothing, resume=false, adapt_mass=false, mass_opts::NamedTuple=(;), quorum_opts::NamedTuple=(;))
    T = Float64
    freqs = _representative_freqs(data)
    cfg = PEConfig{T}(length(freqs), nwalkers, T(step_size), nsteps)
    nrows = (model isa GravitationalWaveBayes.KernelFused._Prec15) ? 15 : 11
    nrows == 11 || throw(ArgumentError("relative binning here is only wired for the 11-parameter " *
        "aligned-spin models (got a $(nrows)-parameter model) -- IMRPhenomXP/XPHM's own " *
        "build_relative_binning methods take a 15-row params_ref with a different physical " *
        "layout, not handled by this function"))
    data1 = (data isa AbstractVector && length(data) == 1) ? data[1] : data
    data1 isa LikelihoodData || throw(ArgumentError(
        "relative binning here is single-detector only, got $(length(data)) detectors"))

    bounds = enforce_bounds ? _sampled_space_bounds(T, nrows) : nothing
    label_a, label_b = "CPU dense-warmup", "CPU binned"
    nsteps_binned = nsteps - nwarmup
    step_size0 = fill(T(step_size), nwalkers)
    ckpt = (resume && outdir !== nothing) ? _load_checkpoint(outdir) : nothing

    if ckpt !== nothing && ckpt.label == label_b
        # Phase A (and its relative-binning reference point) already completed before the
        # interruption -- skip re-running it entirely, rebuild the SAME RelativeBinningSetup from
        # the persisted reference point, and resume only the remaining Phase B draws.
        physical_ref = _load_relative_binning_ref(outdir)
        physical_ref === nothing && throw(ErrorException(
            "checkpoint in $outdir is mid-\"$label_b\" but relative_binning_ref.csv is missing -- " *
            "can't rebuild an identical RelativeBinningSetup, refusing to resume"))
        rb = build_relative_binning(model, physical_ref, data1; nbins=rb_nbins,
                                     max_tc_deviation=rb_max_tc_deviation,
                                     max_extrinsic_phase=rb_max_extrinsic_phase)
        binned_logp_and_grad! = gw_logp_and_grad(data1, cfg, freqs; model=model, relative_binning=rb)
        positions = Array(_gw150914_initial_positions(T, nwalkers, nrows; rng=rng))  # overwritten by resume below
        nsteps_binned_eff, _ = _apply_resume!(positions, step_size0, outdir, label_b, nsteps_binned, 0)
        sample_cfg = NUTSConfig{T}(; max_treedepth=max_treedepth, nwarmup=0, bounds=bounds, quorum_opts...)
        result = nuts_sample!(positions, binned_logp_and_grad!; nsteps=nsteps_binned_eff,
                               step_size0=step_size0, cfg=sample_cfg, rng=rng, record=record_chain,
                               callback=_progress_callback(label_b, nsteps_binned_eff; outdir=outdir))
        # No Phase A draws in this call's chain (it wasn't rerun) -- nwarmup_eff=0 tells
        # _write_outputs the WHOLE returned chain is post-warmup, same convention _run_nuts's own
        # resume path uses.
        return merge(result, (; nwarmup_eff=0)), nrows
    end

    # Phase A: dense kernel, real warmup (dual-averaging adaptation over all `nwarmup` draws --
    # `cfg.nwarmup == nsteps` here means every draw in THIS call is a warmup draw). `warmup_result`
    # either comes from a genuine nuts_sample! call (fresh start, or resuming mid-Phase-A) or, in
    # the narrow "Phase A finished but the process died before Phase B started" case
    # (`ckpt.it >= nwarmup`), directly from the checkpoint -- Phase A's draws are already done,
    # nothing left to re-run.
    local warmup_result
    if ckpt !== nothing && ckpt.label == label_a && ckpt.it >= nwarmup
        warmup_result = (; positions=ckpt.positions, step_size=ckpt.step_size, chain=nothing)
    else
        # Array(...) forces host residency regardless of whether CUDA happens to be loaded in this
        # process (compatible_array, which this helper uses internally, auto-promotes to CuArray
        # once CUDA.jl is loaded+functional ANYWHERE in the process -- including for a device="cpu"
        # run, if CUDA.jl was ALSO loaded for an earlier/other device="cuda" config in the same
        # session). nuts_sample!/nuts_draw!'s own host-side scalar per-walker logic needs positions
        # to stay a plain Array unconditionally; gw_logp_and_grad_gpu_staged handles its OWN device
        # staging internally and never expects this array to already be device-resident.
        positions = Array(_gw150914_initial_positions(T, nwalkers, nrows; rng=rng))
        nwarmup_eff = nwarmup
        if resume
            _, nwarmup_eff = _apply_resume!(positions, step_size0, outdir, label_a, nwarmup, nwarmup)
        end
        dense_logp_and_grad! = gw_logp_and_grad(data1, cfg, freqs; model=model)
        warmup_cfg = NUTSConfig{T}(; max_treedepth=max_treedepth, nwarmup=nwarmup_eff, bounds=bounds, adapt_mass=adapt_mass, mass_opts..., quorum_opts...)
        warmup_result = nuts_sample!(positions, dense_logp_and_grad!; nsteps=nwarmup_eff, step_size0=step_size0,
                                      cfg=warmup_cfg, rng=rng, record=record_chain,
                                      callback=_progress_callback(label_a, nwarmup_eff; outdir=outdir))
    end

    # Reference point for relative binning: mean physical-space position across walkers after
    # warmup -- a real point estimate, not a placeholder (this is the actual fix for the gap
    # scripts/configs/README.md names). Persisted so a Phase-B interruption can rebuild the SAME
    # RelativeBinningSetup on resume (see this function's own docstring) without re-running Phase A.
    physical_after_warmup = similar(warmup_result.positions)
    _reparam_to_physical!(physical_after_warmup, warmup_result.positions)
    physical_ref = vec(sum(physical_after_warmup; dims=2)) ./ nwalkers
    outdir !== nothing && _save_relative_binning_ref(outdir, physical_ref)

    rb = build_relative_binning(model, physical_ref, data1; nbins=rb_nbins,
                                 max_tc_deviation=rb_max_tc_deviation,
                                 max_extrinsic_phase=rb_max_extrinsic_phase)

    # Phase B: binned kernel, step size FROZEN at Phase A's dual-averaging-converged value
    # (`nwarmup=0` -- see NUTSCore.nuts_sample!'s own nwarmup=0 semantics: step_size0 is used
    # as-is, never overwritten). Continues from Phase A's final positions, not a fresh start. Runs
    # in full here (never itself resumed) -- reaching this point means either Phase A just ran (so
    # Phase B hasn't started yet) or there was no checkpoint at all.
    binned_logp_and_grad! = gw_logp_and_grad(data1, cfg, freqs; model=model, relative_binning=rb)
    sample_cfg = NUTSConfig{T}(; max_treedepth=max_treedepth, nwarmup=0, bounds=bounds, quorum_opts...)
    result = nuts_sample!(warmup_result.positions, binned_logp_and_grad!; nsteps=nsteps_binned,
                           step_size0=warmup_result.step_size, cfg=sample_cfg, rng=rng,
                           inv_mass0=adapt_mass ? warmup_result.inv_mass : nothing,
                           record=record_chain,
                           callback=_progress_callback(label_b, nsteps_binned; outdir=outdir))

    # nwarmup_eff is how many draws warmup_result.chain ACTUALLY has -- the full `nwarmup` for a
    # fresh Phase A, fewer if Phase A itself was resumed partway, or `0` if Phase A was reused
    # directly from a checkpoint (chain === nothing, nothing to concatenate below).
    chain = if record_chain && warmup_result.chain !== nothing
        cat(warmup_result.chain, result.chain; dims=3)
    elseif record_chain
        result.chain
    else
        nothing
    end
    nwarmup_eff = warmup_result.chain === nothing ? 0 : size(warmup_result.chain, 3)
    return (; positions=result.positions, logp=result.logp, step_size=result.step_size, chain,
            nwarmup_eff), nrows
end

# ------------------------------------------------------------------------------------------------
# GPU (real CUDA hardware) variants of the two functions above. `gw_logp_and_grad_gpu_staged`
# (2026-09-13/14, real-CUDA-validated for SEOBNRv5AlignedSpinF2 -- see
# scripts/check_seobnr_gpu_staged_nuts.jl and DEVELOPMENT_PLAN.md) keeps `positions`/`grad`/`logp` as
# ordinary HOST arrays (exactly what `nuts_sample!` already allocates and indexes scalar-per-
# walker throughout its tree-doubling/dual-averaging logic) while staging only those buffers to
# the GPU once per leapfrog step, around the actual differentiated likelihood-and-gradient kernel
# call -- so `nuts_sample!`'s own call signature here is IDENTICAL to the CPU functions above,
# only the closure-builder differs. Single-detector only, matching that function's own documented
# scope (a `Vector{LikelihoodData}` multi-detector gradient has a separate, unresolved forward-mode
# Enzyme gap for this model -- see DEVELOPMENT_PLAN.md's 2026-09-14 entry -- not this GPU staging
# itself). Requires CUDA.jl loaded and functional (checked in `main()` before either is called).
# ------------------------------------------------------------------------------------------------

function _run_nuts_gpu(data, model; nwalkers, nsteps, step_size, nwarmup, max_treedepth, rng,
                        enforce_bounds, record_chain, outdir=nothing, resume=false, adapt_mass=false, mass_opts::NamedTuple=(;), quorum_opts::NamedTuple=(;),
                        lik_opts::NamedTuple=(;), prior=nothing, init=nothing)
    T = Float64
    data1 = (data isa AbstractVector && length(data) == 1) ? data[1] : data
    data1 isa LikelihoodData || throw(ArgumentError(
        "GPU-staged SEOBNR is single-detector only, got $(length(data)) detectors"))
    prior === nothing || throw(ArgumentError("[prior] is not wired into the GPU-staged path yet; use device=\"cpu\""))
    freqs = _representative_freqs(data1)
    cfg = _pe_config(T, freqs, nwalkers, step_size, nsteps, lik_opts)
    nrows = 11
    # Array(...) forces host residency regardless of whether CUDA happens to be loaded in this
    # process (compatible_array, which this helper uses internally, auto-promotes to CuArray once
    # CUDA.jl is loaded+functional ANYWHERE in the process -- including for a device="cpu" run, if
    # CUDA.jl was ALSO loaded for an earlier/other device="cuda" config in the same session).
    # nuts_sample!/nuts_draw!'s own host-side scalar per-walker logic needs positions to stay a
    # plain Array unconditionally; gw_logp_and_grad_gpu_staged handles its OWN device staging
    # internally and never expects this array to already be device-resident.
    positions = init === nothing ? Array(_gw150914_initial_positions(T, nwalkers, nrows; rng=rng)) : Matrix{T}(init)
    step_size0 = fill(T(step_size), nwalkers)
    nsteps_eff, nwarmup_eff = nsteps, nwarmup
    if resume
        nsteps_eff, nwarmup_eff = _apply_resume!(positions, step_size0, outdir, "GPU", nsteps, nwarmup)
    end

    logp_and_grad! = gw_logp_and_grad_gpu_staged(data1, cfg, freqs; model=model, nparams=nrows,
                                                  nwalkers=nwalkers)
    bounds = enforce_bounds ? _sampled_space_bounds(T, nrows) : nothing
    nuts_cfg = NUTSConfig{T}(; max_treedepth=max_treedepth, nwarmup=nwarmup_eff, bounds=bounds, adapt_mass=adapt_mass, mass_opts..., quorum_opts...)

    result = nuts_sample!(positions, logp_and_grad!; nsteps=nsteps_eff, step_size0=step_size0,
                           cfg=nuts_cfg, rng=rng, record=record_chain,
                           callback=_progress_callback("GPU", nsteps_eff; outdir=outdir))
    # A resumed chain only has nsteps_eff (not the pre-interruption nwarmup) draws recorded -- see
    # main()'s use of this for _write_outputs' diagnostics warmup cutoff.
    return merge(result, (; nwarmup_eff, pe_cfg=cfg, data_for_grad=data1)), nrows
end

"""
    _run_nuts_gpu_relative_binning(...)

GPU-staged analogue of [`_run_nuts_relative_binning`](@ref) -- see that function's docstring for
the two-phase structure and its `resume` handling (identical here, just with `gw_logp_and_grad_gpu_staged`
and `"GPU dense-warmup"`/`"GPU binned"` labels in place of the CPU closures/labels).
"""
function _run_nuts_gpu_relative_binning(data, model; nwalkers, nsteps, step_size, nwarmup,
                                         max_treedepth, rng, enforce_bounds, record_chain,
                                         rb_nbins, rb_max_tc_deviation, rb_max_extrinsic_phase,
                                         outdir=nothing, resume=false, adapt_mass=false, mass_opts::NamedTuple=(;), quorum_opts::NamedTuple=(;))
    T = Float64
    data1 = (data isa AbstractVector && length(data) == 1) ? data[1] : data
    data1 isa LikelihoodData || throw(ArgumentError(
        "GPU-staged SEOBNR is single-detector only, got $(length(data)) detectors"))
    freqs = _representative_freqs(data1)
    cfg = PEConfig{T}(length(freqs), nwalkers, T(step_size), nsteps)
    nrows = 11
    bounds = enforce_bounds ? _sampled_space_bounds(T, nrows) : nothing
    label_a, label_b = "GPU dense-warmup", "GPU binned"
    nsteps_binned = nsteps - nwarmup
    step_size0 = fill(T(step_size), nwalkers)
    ckpt = (resume && outdir !== nothing) ? _load_checkpoint(outdir) : nothing

    if ckpt !== nothing && ckpt.label == label_b
        physical_ref = _load_relative_binning_ref(outdir)
        physical_ref === nothing && throw(ErrorException(
            "checkpoint in $outdir is mid-\"$label_b\" but relative_binning_ref.csv is missing -- " *
            "can't rebuild an identical RelativeBinningSetup, refusing to resume"))
        rb = build_relative_binning(model, physical_ref, data1; nbins=rb_nbins,
                                     max_tc_deviation=rb_max_tc_deviation,
                                     max_extrinsic_phase=rb_max_extrinsic_phase)
        binned_logp_and_grad! = gw_logp_and_grad_gpu_staged(data1, cfg, freqs; model=model, nparams=nrows,
                                                             nwalkers=nwalkers, relative_binning=rb)
        positions = Array(_gw150914_initial_positions(T, nwalkers, nrows; rng=rng))  # overwritten by resume below
        nsteps_binned_eff, _ = _apply_resume!(positions, step_size0, outdir, label_b, nsteps_binned, 0)
        sample_cfg = NUTSConfig{T}(; max_treedepth=max_treedepth, nwarmup=0, bounds=bounds, quorum_opts...)
        result = nuts_sample!(positions, binned_logp_and_grad!; nsteps=nsteps_binned_eff,
                               step_size0=step_size0, cfg=sample_cfg, rng=rng, record=record_chain,
                               callback=_progress_callback(label_b, nsteps_binned_eff; outdir=outdir))
        return merge(result, (; nwarmup_eff=0)), nrows
    end

    # Phase A: dense GPU-staged kernel, real warmup -- same dense-warmup-then-binned safety
    # discipline `_run_nuts_relative_binning` uses on CPU, needed for the same reason (a NUTS
    # chain's own early, oversized warmup steps can visit `t_c` excursions the binned kernel isn't
    # safe for; see RelativeBinning.jl's `_cap_bin_width_for_extrinsic_ramp` docs). `warmup_result`
    # either comes from a real nuts_sample! call (fresh, or resumed mid-Phase-A) or, in the narrow
    # "Phase A finished but the process died before Phase B started" case (`ckpt.it >= nwarmup`),
    # directly from the checkpoint -- see _run_nuts_relative_binning's identical branch.
    local warmup_result
    if ckpt !== nothing && ckpt.label == label_a && ckpt.it >= nwarmup
        warmup_result = (; positions=ckpt.positions, step_size=ckpt.step_size, chain=nothing)
    else
        # Array(...) forces host residency regardless of whether CUDA happens to be loaded in this
        # process (compatible_array, which this helper uses internally, auto-promotes to CuArray
        # once CUDA.jl is loaded+functional ANYWHERE in the process -- including for a device="cpu"
        # run, if CUDA.jl was ALSO loaded for an earlier/other device="cuda" config in the same
        # session). nuts_sample!/nuts_draw!'s own host-side scalar per-walker logic needs positions
        # to stay a plain Array unconditionally; gw_logp_and_grad_gpu_staged handles its OWN device
        # staging internally and never expects this array to already be device-resident.
        positions = Array(_gw150914_initial_positions(T, nwalkers, nrows; rng=rng))
        nwarmup_eff = nwarmup
        if resume
            _, nwarmup_eff = _apply_resume!(positions, step_size0, outdir, label_a, nwarmup, nwarmup)
        end
        dense_logp_and_grad! = gw_logp_and_grad_gpu_staged(data1, cfg, freqs; model=model, nparams=nrows,
                                                            nwalkers=nwalkers)
        warmup_cfg = NUTSConfig{T}(; max_treedepth=max_treedepth, nwarmup=nwarmup_eff, bounds=bounds, adapt_mass=adapt_mass, mass_opts..., quorum_opts...)
        warmup_result = nuts_sample!(positions, dense_logp_and_grad!; nsteps=nwarmup_eff, step_size0=step_size0,
                                      cfg=warmup_cfg, rng=rng, record=record_chain,
                                      callback=_progress_callback(label_a, nwarmup_eff; outdir=outdir))
    end

    physical_after_warmup = similar(warmup_result.positions)
    _reparam_to_physical!(physical_after_warmup, warmup_result.positions)
    physical_ref = vec(sum(physical_after_warmup; dims=2)) ./ nwalkers
    outdir !== nothing && _save_relative_binning_ref(outdir, physical_ref)

    rb = build_relative_binning(model, physical_ref, data1; nbins=rb_nbins,
                                 max_tc_deviation=rb_max_tc_deviation,
                                 max_extrinsic_phase=rb_max_extrinsic_phase)

    # Phase B: binned GPU-staged kernel, step size FROZEN at Phase A's converged value (nwarmup=0).
    binned_logp_and_grad! = gw_logp_and_grad_gpu_staged(data1, cfg, freqs; model=model, nparams=nrows,
                                                         nwalkers=nwalkers, relative_binning=rb)
    sample_cfg = NUTSConfig{T}(; max_treedepth=max_treedepth, nwarmup=0, bounds=bounds, quorum_opts...)
    result = nuts_sample!(warmup_result.positions, binned_logp_and_grad!; nsteps=nsteps_binned,
                           step_size0=warmup_result.step_size, cfg=sample_cfg, rng=rng,
                           inv_mass0=adapt_mass ? warmup_result.inv_mass : nothing,
                           record=record_chain,
                           callback=_progress_callback(label_b, nsteps_binned; outdir=outdir))

    chain = if record_chain && warmup_result.chain !== nothing
        cat(warmup_result.chain, result.chain; dims=3)
    elseif record_chain
        result.chain
    else
        nothing
    end
    nwarmup_eff = warmup_result.chain === nothing ? 0 : size(warmup_result.chain, 3)
    return (; positions=result.positions, logp=result.logp, step_size=result.step_size, chain,
            nwarmup_eff), nrows
end

# ------------------------------------------------------------------------------------------------
# Output
# ------------------------------------------------------------------------------------------------

function _write_outputs(outdir, config_path, nrows; positions=nothing, logp=nothing, chain=nothing,
                         nwarmup=0, cold_chain=nothing, pt=nothing)
    mkpath(outdir)
    cp(config_path, joinpath(outdir, "config_used.toml"); force=true)
    names = nrows >= 15 ? vcat(_PARAM_NAMES_15, ["cal_z$(k)" for k in 1:nrows-15]) : _PARAM_NAMES_11

    if positions !== nothing
        # positions is (nrows params, nwalkers); write one row per walker, one column per param.
        open(joinpath(outdir, "final_positions.csv"), "w") do io
            writedlm(io, vcat(reshape(names, 1, nrows), permutedims(positions)), ',')
        end
    end
    logp !== nothing && writedlm(joinpath(outdir, "final_logp.csv"), logp, ',')

    if chain !== nothing
        serialize(joinpath(outdir, "chain.jls"), (; names, chain))
        # `warmup` excludes not-yet-converged draws from Rhat/ESS (see chain_diagnostics' docstring
        # in src/GravitationalWaveBayes/Diagnostics.jl); needs at least 4 post-warmup draws to run at all.
        if size(chain, 3) - nwarmup >= 4
            diags = chain_diagnostics(chain; warmup=nwarmup, names=names)
            open(joinpath(outdir, "diagnostics.txt"), "w") do io
                print_diagnostics(io, diags)
            end
            print_diagnostics(diags)
        else
            @warn "Too few post-warmup draws ($(size(chain, 3) - nwarmup)) for diagnostics; skipping Rhat/ESS"
        end
    end

    if cold_chain !== nothing
        # cold_chain is (nrows params, nsteps); write one row per step, one column per param.
        open(joinpath(outdir, "cold_chain.csv"), "w") do io
            writedlm(io, vcat(reshape(names, 1, nrows), permutedims(cold_chain)), ',')
        end
    end
    if pt !== nothing
        acc = Array(pt.naccept) ./ max.(Array(pt.nattempt), 1)
        writedlm(joinpath(outdir, "pt_swap_acceptance.csv"), acc, ',')
    end

    println("Wrote results to ", outdir)
end


# `prior.distance = "source_frame"`: the run sampled d_L^2 on the same bounds; `prior_weights.jls` holds
# (; logw, logZ_shift, ess): per chain sample (walker, draw) log p_sf(d_L) / p_vol(d_L) (both normalized), and the
# evidence shift log mean(w) (logZ_source_frame = logZ + logZ_shift, also appended to smc_summary.txt). Weighted
# chain = the UniformSourceFrame posterior (GWTC `_cosmo` releases are this reweighting of their d_L^2 runs).
function _source_frame_weights(outdir, cfg, prior, result)
    get(get(cfg, "prior", Dict{String,Any}()), "distance", "") == "source_frame" || return nothing
    result.chain === nothing && throw(ArgumentError("prior.distance = \"source_frame\" needs output.record_chain = true"))
    D = 1000 .* vec(result.chain[5, :, :])
    logw = reshape(source_frame_log_weights(D, 1000 * prior.lo[5], 1000 * prior.hi[5]), size(result.chain)[2:3])
    m = maximum(logw)
    shift = m + log(sum(exp.(logw .- m)) / length(logw))
    w = exp.(logw .- m)
    ess = sum(w)^2 / sum(abs2, w)
    serialize(joinpath(outdir, "prior_weights.jls"), (; logw, logZ_shift=shift, ess))
    println("UniformSourceFrame weights: logZ shift ", round(shift; digits=4), ", weight ESS ", round(ess; digits=1),
            " / ", length(logw))
    haskey(result, :logZ) && open(joinpath(outdir, "smc_summary.txt"), "a") do io
        println(io, "logZ_source_frame = ", result.logZ + shift)
        println(io, "source_frame_weight_ess = ", ess)
    end
    return nothing
end

# ------------------------------------------------------------------------------------------------
# [likelihood], [prior], [injection] sections and sampler.init (see scripts/configs/README.md).
# ------------------------------------------------------------------------------------------------

# PEConfig carrying the [likelihood] options (lik_opts from _parse_likelihood_opts; `(;)` = plain).
# Number of sampled rows the likelihood marginalizes over (they carry prior draws only).
lik_opts_count_marginalized(lik_opts) =
    Int(get(lik_opts, :marginalize_phase, false)) + Int(get(lik_opts, :marginalize_distance, false))

# Reference SNR^2 for the "auto" PT ladder when there is no injection: the matched-filter SNR^2 at the
# starting walker with the highest likelihood, maximized over a common amplitude and phase
# (|sum_k <d_k,h_k>|^2 / sum_k <h_k,h_k>, complex inner products), so it does not depend on that walker's
# distance or phase rows. PAF2 models only (uses `detector_strain`).
function _reference_snr2(data, model, freqs, lik_opts, init, nwalkers, nrows, rng)
    T = Float64
    model isa PAF2Model || throw(ArgumentError(
        "sampler.pt_beta_min/pt_ntemps = \"auto\" without an [injection] is implemented for PAF2 models only"))
    dets = data isa AbstractVector ? data : [data]
    pos = init === nothing ? Array(_gw150914_initial_positions(T, nwalkers, nrows; rng=copy(rng))) : Matrix{T}(init)
    cfg = _pe_config(T, freqs, size(pos, 2), 1e-3, 1, lik_opts)
    ll = zeros(T, size(pos, 2))
    _raw_loglikelihood!(ll, pos, freqs, cfg, length(dets) == 1 ? dets[1] : dets, model, nothing)
    physical = similar(pos)
    _reparam_to_physical!(physical, pos)
    best = argmax(map(x -> isfinite(x) ? x : -Inf, ll))
    p = physical[:, best]
    geo = length(dets) > 1
    dh = zero(ComplexF64)
    hh = zero(T)
    for d in dets
        h = detector_strain(model, d, p; geocentric=geo, fmax_taper=get(lik_opts, :fmax_taper, 0.0))
        df = d.freqs[2] - d.freqs[1]
        dh += 4 * df * sum(conj(h[i]) * d.strain[i] / d.psd[i] for i in eachindex(h) if d.psd[i] > 0)
        hh += optimal_snr(h, d)^2
    end
    return abs2(dh) / hh
end

_pe_config(::Type{T}, freqs, nwalkers, step_size, nsteps, lik_opts) where {T} =
    PEConfig{T}(length(freqs), nwalkers, T(step_size), nsteps;
                marginalize_phase=get(lik_opts, :marginalize_phase, false),
                time_reference=get(lik_opts, :time_reference, 0),
                marginalize_distance=get(lik_opts, :marginalize_distance, false),
                distance_lo=get(lik_opts, :distance_lo, 0.05), distance_hi=get(lik_opts, :distance_hi, 4.0),
                distance_power=get(lik_opts, :distance_power, 0),
                time_ref_frequency=get(lik_opts, :time_ref_frequency, 0.0),
                time_ref_offset=get(lik_opts, :time_ref_offset, 0.0), fmax_taper=get(lik_opts, :fmax_taper, 0.0),
                phase_ref_frequency=get(lik_opts, :phase_ref_frequency, 0.0))

# `prior` supplies the distance prior the distance marginalization integrates over (the [prior] dL
# bounds and `distance` shape, or the package defaults). `ref_intrinsic = (m1, m2, chi1, chi2)` is the
# point `time_ref_offset = "auto"` centres the reference-frequency time on (offset = -t*(ref)).
function _parse_likelihood_opts(cfg, data, prior=nothing; model=nothing, ref_intrinsic=nothing)
    lik = get(cfg, "likelihood", Dict{String,Any}())
    # (IMRPhenomXHM: the CUDA mode-sum path only -- its CPU likelihoods keep the phase at the lowest data frequency)
    (get(lik, "phase_ref_frequency", 0.0) > 0 && !(model isa Union{SEOBNRv5PHM,IMRPhenomXHM,GravitationalWaveforms.SEOBNRv5HM})) &&
        throw(ArgumentError("likelihood.phase_ref_frequency is implemented for SEOBNRv5PHM / IMRPhenomXHM / SEOBNRv5HM only"))
    marg = get(lik, "marginalize_phase", false)
    tref_raw = get(lik, "time_reference", "geocenter")
    tref = if tref_raw isa Integer
        Int(tref_raw)
    elseif lowercase(String(tref_raw)) == "geocenter"
        0
    else
        k = findfirst(d -> d.detector.prefix == tref_raw, data)
        k === nothing && throw(ArgumentError("likelihood.time_reference=\"$tref_raw\" is not one of the configured detectors"))
        k
    end
    margdist = Bool(get(lik, "marginalize_distance", false))
    pr = prior === nothing ? GWPrior{Float64}() : prior
    fref = Float64(get(lik, "time_ref_frequency", 0.0))
    toff_raw = get(lik, "time_ref_offset", "auto")
    toff = if fref <= 0
        0.0
    elseif toff_raw isa Real
        Float64(toff_raw)
    else
        String(toff_raw) == "auto" || throw(ArgumentError("likelihood.time_ref_offset must be a number or \"auto\""))
        (model === nothing || ref_intrinsic === nothing) && throw(ArgumentError("time_ref_offset=\"auto\" needs a model and reference point"))
        -time_at_frequency(model, ref_intrinsic..., fref)
    end
    return (; marginalize_phase=Bool(marg), time_reference=tref, marginalize_distance=margdist,
            distance_lo=pr.lo[5], distance_hi=pr.hi[5], distance_power=(pr.distance === :volume ? 2 : 0),
            time_ref_frequency=fref, time_ref_offset=toff, fmax_taper=_fmax_taper(cfg),
            phase_ref_frequency=Float64(get(lik, "phase_ref_frequency", 0.0)))
end

_fmax_taper(cfg) = Float64(get(get(cfg, "likelihood", Dict{String,Any}()), "fmax_taper", 0.0))

# Reference intrinsic point (m1, m2, chi1, chi2) for time_ref_offset="auto": the injection truth when
# there is one, else the centre of the prior's (mc, q) box with zero spins.
function _reference_intrinsic(truth_sampled, prior)
    if truth_sampled !== nothing
        m1, m2 = _mc_q_to_m1_m2(truth_sampled[1], truth_sampled[2])
        return (m1, m2, truth_sampled[3], truth_sampled[4])
    end
    pr = prior === nothing ? GWPrior{Float64}() : prior
    m1, m2 = _mc_q_to_m1_m2((pr.lo[1] + pr.hi[1]) / 2, (pr.lo[2] + pr.hi[2]) / 2)
    return (m1, m2, 0.0, 0.0)
end

# Coalescence time (sampled frame, i.e. after any detector time_reference) -> sampled time row.
function _tc_to_time_row(tc, sampled, model, lik_opts)
    get(lik_opts, :time_ref_frequency, 0.0) > 0 || return tc
    m1, m2 = _mc_q_to_m1_m2(sampled[1], sampled[2])
    return tc + time_at_frequency(model, m1, m2, sampled[3], sampled[4], lik_opts.time_ref_frequency) + lik_opts.time_ref_offset
end

function _parse_prior(cfg; ncal::Int=0)
    haskey(cfg, "prior") || (ncal == 0 ? (return nothing) :
                             throw(ArgumentError("[calibration] needs an explicit [prior] table")))
    pr = cfg["prior"]
    bounds = Dict{String,Any}(k => v for (k, v) in pr if k in ("mc", "q", "chi1", "chi2", "dL", "t_c"))
    if get(pr, "spins", "ball") == "isotropic"          # LVK spin prior: y in the unit ball (see `_spin_isotropic`)
        bounds["chi1"] = (-1.0, 1.0); bounds["chi2"] = (-1.0, 1.0)
    end
    # precessing models: 15 rows, each spin vector uniform in the ball |chi_i| <= max(|chi_i bounds|)
    nrows = get(_MODELS, get(get(cfg, "model", Dict()), "name", ""), nothing) isa GravitationalWaveBayes.KernelFused._Prec15 ? 15 : 11
    # component-mass constraints (`prior.m1 = [lo, hi]`, `prior.m2`; detector frame, as bilby's mass_1 / mass_2 Constraint)
    cut(k) = haskey(pr, k) ? (Float64(pr[k][1]), Float64(pr[k][2])) : (0.0, Inf)
    # distance = "source_frame" (bilby UniformSourceFrame, Planck15_LAL): sampled under d_L^2, then reweighted
    # (`_source_frame_weights`; the distance marginalization assumes a power law)
    dist = get(pr, "distance", "uniform") == "source_frame" ? "volume" : get(pr, "distance", "uniform")
    return GWPrior{Float64}(; bounds=bounds, distance=Symbol(dist),
                            masses=Symbol(get(pr, "masses", "uniform_mc_q")), nrows=nrows, ncal=ncal, m1=cut("m1"), m2=cut("m2"))
end

const _INJECTION_KEYS = ("mc", "q", "chi1", "chi2", "dL", "t_c", "phase", "ra", "dec", "theta_jn", "psi")

# SEOBNRv5PHM [injection] (zero noise on the real PSDs): mc, q (detector frame), chi1 / chi2 (z) and chi1x, chi1y, chi2x,
# chi2y (at `likelihood.phase_ref_frequency`, like the sampled spins), dL (Gpc), t_c (s, the arrival time at the
# likelihood's time-reference detector, relative to the trigger), phase (at the same reference), ra, dec, theta_jn, psi.
# The strain is GravitationalWaveforms.hphc(...; f_ref = phase_ref_frequency) projected on each detector -- the device likelihood's
# convention: logL = 0 at the injection. Replaces the data strain (the real
# signal and noise are discarded; the PSDs are kept). Writes injection.txt (truth, optimal SNRs) to the run directory.
const _PHM_INJECTION_KEYS = ("mc", "q", "chi1", "chi2", "chi1x", "chi1y", "chi2x", "chi2y", "dL", "t_c", "phase", "ra",
                             "dec", "theta_jn", "psi")
function _phm_injection!(data, cfg, outdir)
    inj = cfg["injection"]
    for k in _PHM_INJECTION_KEYS
        haskey(inj, k) || throw(ArgumentError("SEOBNRv5PHM [injection] needs $(join(_PHM_INJECTION_KEYS, ", ")); missing $k"))
    end
    get(inj, "noise", false) && throw(ArgumentError("SEOBNRv5PHM [injection]: only noise = false (zero noise) is implemented"))
    v = Dict(k => Float64(inj[k]) for k in _PHM_INJECTION_KEYS)
    m1, m2 = _mc_q_to_m1_m2(v["mc"], v["q"])
    likc = get(cfg, "likelihood", Dict{String,Any}())
    fphi = Float64(get(likc, "phase_ref_frequency", 0.0))
    freqs = Array(data[1].freqs)
    fref = fphi > 0 ? fphi : freqs[1]
    hp, hc = GravitationalWaveforms.hphc(SEOBNRv5PHM(), freqs, m1, m2, (v["chi1x"], v["chi1y"], v["chi1"]),
                              (v["chi2x"], v["chi2y"], v["chi2"]), v["dL"], v["theta_jn"], v["phase"];
                              f_min = 0.9 * freqs[1], f_ref = fref)
    g = GravitationalWaveBayes.BatchedNUTS.gmst(data[1].gps_time)
    tr = get(likc, "time_reference", 0)
    kref = tr isa AbstractString ? findfirst(d -> d.detector.prefix == tr, data) : Int(tr)
    tgeo = v["t_c"] - (kref === nothing || kref == 0 ? 0.0 :
                       GravitationalWaveData.time_delay_from_geocenter_gmst(data[kref].detector, v["ra"], v["dec"], g))
    df = freqs[2] - freqs[1]
    snr2 = Float64[]
    out = map(data) do d
        Fp, Fc = GravitationalWaveData.antenna_response_gmst(d.detector, v["ra"], v["dec"], v["psi"], g)
        dt = GravitationalWaveData.time_delay_from_geocenter_gmst(d.detector, v["ra"], v["dec"], g)
        h = [(Fp * hp[i] + Fc * hc[i]) * cis(-2pi * freqs[i] * (tgeo + dt)) for i in eachindex(freqs)]
        S = Array(d.psd)
        push!(snr2, 4 * df * sum(abs2(h[i]) / S[i] for i in eachindex(h) if isfinite(S[i]) && S[i] > 0))
        strain = similar(d.strain); copyto!(strain, ComplexF64.(h))
        GravitationalWaveBayes.LikelihoodData(d.freqs, strain, d.psd, d.detector, d.gps_time)
    end |> collect
    println("SEOBNRv5PHM zero-noise injection (f_ref $fref Hz): m1 ", round(m1; digits=3), " m2 ", round(m2; digits=3),
            ", optimal SNR ", round.(sqrt.(snr2); digits=2), " network ", round(sqrt(sum(snr2)); digits=2))
    mkpath(outdir)
    open(joinpath(outdir, "injection.txt"), "w") do io
        for k in _PHM_INJECTION_KEYS
            println(io, k, " = ", v[k])
        end
        println(io, "m1 = ", m1, "
m2 = ", m2, "
t_c_geocentric = ", tgeo, "
f_ref = ", fref)
        println(io, "optimal_snr = ", sqrt.(snr2), "
network_optimal_snr = ", sqrt(sum(snr2)))
    end
    return out
end

# Physical 11-tuple and sampled 11-vector for an [injection] table (mc, q, chi1, chi2, dL, t_c, phase,
# ra, dec, theta_jn, psi -- dec/theta_jn as angles; t_c is geocentric for multi-detector data).
function _injection_truth(inj)
    for k in _INJECTION_KEYS
        haskey(inj, k) || throw(ArgumentError("[injection] needs $(join(_INJECTION_KEYS, ", ")); missing $k"))
    end
    v = Dict(k => Float64(inj[k]) for k in _INJECTION_KEYS)
    m1, m2 = _mc_q_to_m1_m2(v["mc"], v["q"])
    physical = (m1, m2, v["chi1"], v["chi2"], v["dL"], v["t_c"], v["phase"], v["ra"], v["dec"], v["theta_jn"], v["psi"])
    sampled = [v["mc"], v["q"], v["chi1"], v["chi2"], v["dL"], v["t_c"], v["phase"], v["ra"], sin(v["dec"]),
               cos(v["theta_jn"]), v["psi"]]
    return physical, sampled
end

# The injection's sampled-space truth expressed in the frame the likelihood samples `t_c` in.
function _sampled_truth_in_frame(sampled, data, lik_opts, model=nothing)
    out = copy(sampled)
    ra, dec = sampled[8], asin(sampled[9])
    g = GravitationalWaveBayes.BatchedNUTS.gmst(data[1].gps_time)
    if length(data) > 1 && lik_opts.time_reference > 0
        out[6] += GravitationalWaveData.time_delay_from_geocenter_gmst(data[lik_opts.time_reference].detector, ra, dec, g)
    end
    model === nothing || (out[6] = _tc_to_time_row(out[6], out, model, lik_opts))
    return out
end

function _init_positions(kind, rng, nwalkers; truth=nothing, prior=nothing, scale=0.01)
    if kind == "prior"
        pr = prior === nothing ? GWPrior{Float64}() : prior
        return sample_prior(rng, pr, nwalkers)
    elseif kind == "injection"
        truth === nothing && throw(ArgumentError("sampler.init=\"injection\" needs an [injection] section"))
        pr = prior === nothing ? GWPrior{Float64}() : prior
        lo, hi, periodic = prior_bounds(pr)
        pos = repeat(reshape(truth, :, 1), 1, nwalkers)
        for w in 1:nwalkers, r in 1:11
            x = truth[r] + scale * (hi[r] - lo[r]) * randn(rng)
            if periodic[r]
                x = lo[r] + mod(x - lo[r], hi[r] - lo[r])
            else
                span = hi[r] - lo[r]
                x = clamp(x, lo[r] + 1e-3 * span, hi[r] - 1e-3 * span)
            end
            pos[r, w] = x
        end
        return pos
    end
    return nothing   # "gw150914": the runners' built-in start
end

# ------------------------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------------------------

# The configured sampler on `data` (the sampler half of `main`, shared with scripts/pp_test.jl): `[sampler]` / `[output]`
# of `cfg`, with the run directory and seed overridable. Writes the run outputs when `write`. Returns
# `(; result, nrows, nwarmup)` for sampler.type = "nuts" (NUTS or SMC), `nothing` for "pt".
function _run_sampler(cfg, config_path, data, model, prior, lik_opts; calibration=nothing, calmarg=nothing,
                      truth_sampled=nothing, ref_snr2=nothing, outdir_override=nothing, seed_override=nothing,
                      init_override=nothing, write::Bool=true)
    sampler = get(cfg, "sampler", Dict{String,Any}())
    sampler_type = get(sampler, "type", "nuts")
    nsteps = get(sampler, "nsteps", 200)
    step_size = get(sampler, "step_size", 1e-3)
    max_treedepth = get(sampler, "max_treedepth", 10)
    enforce_bounds = get(sampler, "enforce_bounds", true)
    seed = seed_override === nothing ? get(sampler, "seed", nothing) : seed_override
    rng = seed === nothing ? Random.default_rng() : Random.Xoshiro(Int(seed))

    output = get(cfg, "output", Dict{String,Any}())
    outdir = outdir_override === nothing ? get(output, "dir", "run_output") : outdir_override

    if sampler_type == "nuts"
        nwalkers = get(sampler, "nwalkers", 8)
        init_kind = get(sampler, "init", "gw150914")
        init_kind in ("gw150914", "injection", "prior") || throw(ArgumentError(
            "sampler.init must be \"gw150914\", \"injection\" or \"prior\", got \"$init_kind\""))
        init = init_override !== nothing ? init_override : _init_positions(init_kind, rng, nwalkers;
            truth=truth_sampled === nothing ? nothing : _sampled_truth_in_frame(truth_sampled, data, lik_opts, model),
            prior=prior, scale=get(sampler, "init_scale", 0.01))
        nwarmup = get(sampler, "nwarmup", max(nsteps ÷ 2, 1))
        record_chain = get(output, "record_chain", true)
        use_rb = get(sampler, "relative_binning", false)
        resume = get(sampler, "resume", false)
        adapt_mass = get(sampler, "adapt_mass", false)
        # Dense-metric options (NUTSConfig, warmup-only -- `dense_mass`/`hessian_init`/
        # `cull_logp_gap` only affect metric ADAPTATION, which only runs during warmup; the frozen
        # sampling-phase configs below don't set these, since the actual per-step kinematics
        # dispatch on the `inv_mass` VALUE carried forward from warmup, not on these config flags):
        # only keys actually present in the TOML are forwarded, so a config without them gets
        # NUTSConfig's own defaults (the unchanged diagonal behaviour).
        mass_opts = (; (Symbol(k) => sampler[k] for k in ("dense_mass", "hessian_init", "hessian_rel_step",
                                                          "mass_init_buffer", "mass_base_window", "mass_term_buffer", "cull_logp_gap",
                                                          "step_size_max_change")
                        if haskey(sampler, k))...)
        # `mass_blocks` (NUTSConfig, warmup-only, same reasoning as `mass_opts` above): TOML parses
        # a nested array-of-arrays as `Vector{Any}` of `Vector{Any}`, not the concrete
        # `Vector{Vector{Int}}` NUTSConfig's constructor requires -- converted explicitly rather
        # than folded into `mass_opts`'s generic pass-through, which would hand it through
        # untouched and hit a `MethodError` at `NUTSConfig{T}(...)` construction.
        if haskey(sampler, "mass_blocks")
            mass_opts = merge(mass_opts, (; mass_blocks=[Vector{Int}(b) for b in sampler["mass_blocks"]]))
        end
        # `quorum_frac` (NUTSConfig): unlike the dense-metric options above, this shapes every
        # draw's tree-doubling loop directly, not just warmup adaptation -- forwarded to BOTH the
        # warmup and the frozen sampling-phase configs below.
        quorum_opts = haskey(sampler, "quorum_frac") ? (; quorum_frac=sampler["quorum_frac"]) : NamedTuple()
        # The adapted diagonal metric (NUTSCore `adapt_mass`) is not checkpointed, so resuming
        # would run a converged step size against a unit metric it was never tuned for. Refuse
        # rather than silently resume something wrong (see scripts/configs/README.md).
        (adapt_mass && resume) && throw(ArgumentError(
            "sampler.adapt_mass=true cannot be combined with sampler.resume=true: the adapted mass " *
            "matrix is not checkpointed"))
        device = get(sampler, "device", "cpu")
        device in ("cpu", "cuda") || throw(ArgumentError("sampler.device must be \"cpu\" or \"cuda\", got \"$device\""))
        use_gpu = device == "cuda"
        # sampler.method = "smc": adaptive tempered SMC (`_run_smc`) instead of NUTS warmup (+ PT).
        method = get(sampler, "method", "nuts")
        method in ("nuts", "smc") || throw(ArgumentError("sampler.method must be \"nuts\" or \"smc\", got \"$method\""))
        (method == "smc" && get(sampler, "relative_binning", false)) &&
            throw(ArgumentError("sampler.method = \"smc\" does not support relative binning yet"))
        smc_opts = (; (Symbol(k[5:end]) => sampler[k] for k in ("smc_target_cess", "smc_nfinal", "smc_min_mutate",
                                                                 "smc_max_mutate", "smc_disp_target", "smc_dense",
                                                                 "smc_step_size0", "smc_reference", "smc_reference_inflate",
                                                                 "smc_reference_alpha", "smc_reference_kind",
                                                                 "smc_reference_kernels", "smc_independence",
                                                                 "smc_reference_neighbours", "smc_reference_local_h",
                                                                 "smc_online_kde", "smc_online_alpha", "smc_online_kernels", "smc_online_pool_max", "smc_independence_tries", "smc_online_nuts", "smc_ind_batch", "smc_nuts_narrow", "smc_final_ess", "smc_final_max_treedepth", "smc_ind_release",
                                                                 "smc_de_tries", "smc_de_jump", "smc_phase_shear", "smc_dump_stages", "smc_reference_prior_rows", "smc_sky_ring",
                                                                 "smc_nuts_beta", "smc_nuts_max_mutate", "smc_metric_core", "smc_hessian_metric", "smc_sky_phase", "smc_incl_flip",
                                                                 "smc_online_flow", "smc_flow_steps", "smc_flow_layers", "smc_flow_hidden", "smc_flow_warm", "smc_flow_device", "smc_flow_batch",
                                                                 "smc_latent_nuts", "smc_local_kernel", "smc_mala_steps",
                                                                 "smc_persistent", "smc_extrinsic_sweeps",
                                                                 "smc_reference_time_shift", "smc_move_stat", "smc_moves",
                                                                 "smc_rw_thin", "smc_rw_target_accept", "smc_rw_scale0", "smc_rw_de", "smc_rw_de_jump",
                                                                 "smc_mode_jumps", "smc_mode_jump_rows", "smc_mode_jump_tries", "smc_mode_jump_k",
                                                                 "smc_mode_jump_learn_beta", "smc_mode_jump_scale", "smc_mode_jump_min_cell",
                                                                 "smc_mode_jump_batch", "smc_mode_jump_learn", "smc_mode_jump_tails",
                                                                 "smc_mode_jump_tail_frac", "smc_mode_jump_tail_min", "smc_mode_jump_tail_pcs")
                      if haskey(sampler, k))...)
        haskey(smc_opts, :move_stat) && (smc_opts = merge(smc_opts, (; move_stat=Symbol(smc_opts.move_stat))))
        haskey(smc_opts, :local_kernel) && (smc_opts = merge(smc_opts, (; local_kernel=Symbol(smc_opts.local_kernel))))
        _spin_amax(cfg) === nothing || (smc_opts = merge(smc_opts, (; spin_isotropic_max=_spin_amax(cfg))))
        calibration === nothing || (smc_opts = merge(smc_opts, (; calibration)))
        calmarg === nothing || (smc_opts = merge(smc_opts, (; calmarg)))
        if use_gpu
            # IMRPhenomXAS: dense, or relative binning on the device (2026-10-01; SMC + relative binning is rejected above).
            (model isa Union{PAF2Model,IMRPhenomXAS,IMRPhenomXHM,IMRPhenomXP,IMRPhenomXPHM,SEOBNRv5IMRAlignedSpinF2,SEOBNRv5PHM,GravitationalWaveforms.SEOBNRv5HM}) ||
                throw(ArgumentError(
                "device=\"cuda\" is only wired up for SEOBNRv5AlignedSpinF2, SEOBNRv5IMRAlignedSpinF2, SEOBNRv5PHM, SEOBNRv5HM, IMRPhenomXAS, IMRPhenomXHM, IMRPhenomXP and IMRPhenomXPHM " *
                "(gw_logp_and_grad_gpu_staged), got model=$(typeof(model))"))
            isdefined(Main, :CUDA) || throw(ErrorException(
                "device=\"cuda\" requested but CUDA.jl failed to load -- see the warning printed at " *
                "startup, or add it: `julia --project=. -e 'import Pkg; Pkg.add(\"CUDA\")'`"))
            CUDA.functional() || throw(ErrorException("device=\"cuda\" requested but CUDA.functional() is false"))
        end
        # `sampler.gradient_mode` = "auto" (default) | "forward" | "reverse" for SEOBNRv5AlignedSpinF2:
        #   "auto"    -- follow BatchedNUTS._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT's default: reverse mode
        #                when the loaded GravitationalWaveforms has the 2026-09-19 device-heap-leak fix, forward otherwise;
        #                unsupported cases (relative binning, multi-detector, other models) use forward.
        #   "forward" -- force the forward-mode path (the long-standing production default before 2026-09-19).
        #   "reverse" -- force reverse mode; rejected if the run can't use it (relative binning, other models).
        # Reverse mode on CUDA used to crash after ~11 repeated gradient calls (Enzyme exponential-allocation
        # device-heap leak via Base.searchsortedlast in GravitationalWaveforms' bracket search); fixed by GravitationalWaveforms'
        # constant-trip-count `_seobv5f2_searchsortedlast` (see DEVELOPMENT_PLAN.md, 2026-09-19).
        gradient_mode = get(sampler, "gradient_mode", "auto")
        gradient_mode in ("auto", "forward", "reverse") || throw(ArgumentError(
            "sampler.gradient_mode must be \"auto\", \"forward\" or \"reverse\", got \"$gradient_mode\""))
        if gradient_mode == "reverse"
            model isa PAF2Model || throw(ArgumentError(
                "gradient_mode=\"reverse\" is only wired up for SEOBNRv5AlignedSpinF2/SEOBNRv5IMRAlignedSpinF2, got model=$(typeof(model))"))
            use_rb && throw(ArgumentError(
                "gradient_mode=\"reverse\" is not supported with relative_binning=true -- " *
                "_reverse_mode_gradient! doesn't implement that path"))
            # Reverse mode on CUDA used to leak device heap (illegal memory access after ~11 repeated
            # gradient calls) via Base.searchsortedlast inside GravitationalWaveforms' SEOBNR bracket search;
            # fixed by GravitationalWaveforms' constant-trip-count `_seobv5f2_searchsortedlast` (2026-09-19, see
            # DEVELOPMENT_PLAN.md). Only warn if the loaded GravitationalWaveforms predates that fix.
            gww = parentmodule(GravitationalWaveBayes.KernelFused._seobv5f2_phi_scalar)
            (device == "cuda" && !isdefined(gww, :_seobv5f2_searchsortedlast)) && @warn(
                "gradient_mode=\"reverse\" on device=\"cuda\" is KNOWN TO CRASH (CUDA illegal memory " *
                "access after ~11 repeated gradient calls): the loaded GravitationalWaveforms predates the " *
                "`_seobv5f2_searchsortedlast` fix for Enzyme's reverse-mode device-heap leak. Update " *
                "GravitationalWaveforms; see DEVELOPMENT_PLAN.md's 2026-09-19 entry.")
        end

        println("Sampler: NUTS, nwalkers=$nwalkers nsteps=$nsteps nwarmup=$nwarmup, device=$device",
                isempty(mass_opts) ? "" : " mass_opts=$(mass_opts)",
                use_rb ? " (dense-warmup-then-binned)" : "", resume ? " (resume=true)" : "",
                gradient_mode == "auto" ? "" : " (gradient_mode=$gradient_mode)")

        # Remember the package default so it can be restored afterwards (a config run must not leak its
        # override into a later run in the same session).
        reverse_default = GravitationalWaveBayes.BatchedNUTS._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[]
        gradient_mode == "auto" || (GravitationalWaveBayes.BatchedNUTS._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[] = (gradient_mode == "reverse"))
        model isa PAF2Model && println("SEOBNR-path gradient preference: ",
            GravitationalWaveBayes.BatchedNUTS._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[] ?
            "reverse-mode (falls back to forward where unsupported: relative binning, multi-detector)" : "forward-mode")
        local result, nrows
        try
            # relative binning with the likelihood options: XAS (CPU, or on the device, 2026-10-01); XHM (CPU, 2026-10-03)
            use_rb_xas = use_rb && (model isa IMRPhenomXAS || (model isa IMRPhenomXHM && !use_gpu)) && method != "smc"
            if use_rb && !use_rb_xas
                (lik_opts.marginalize_phase || lik_opts.marginalize_distance || lik_opts.time_reference != 0 ||
                 lik_opts.time_ref_frequency > 0 || lik_opts.fmax_taper > 0 || prior !== nothing || init !== nothing) &&
                    throw(ArgumentError("[likelihood]/[prior]/sampler.init are not supported with relative_binning=true yet"))
                rb_nbins = get(sampler, "relative_binning_nbins", 150)
                rb_max_tc_deviation = get(sampler, "relative_binning_max_tc_deviation", 0.0)
                rb_max_extrinsic_phase = get(sampler, "relative_binning_max_extrinsic_phase", 0.0)
                runner = use_gpu ? _run_nuts_gpu_relative_binning : _run_nuts_relative_binning
                result, nrows = runner(data, model; nwalkers=nwalkers, nsteps=nsteps,
                    step_size=step_size, nwarmup=nwarmup, max_treedepth=max_treedepth, rng=rng,
                    enforce_bounds=enforce_bounds, record_chain=record_chain, rb_nbins=rb_nbins,
                    rb_max_tc_deviation=rb_max_tc_deviation, rb_max_extrinsic_phase=rb_max_extrinsic_phase,
                    outdir=outdir, resume=resume, adapt_mass=adapt_mass, mass_opts=mass_opts, quorum_opts=quorum_opts)
            else
                # Dense IMRPhenomXAS on CUDA goes through the CPU runner (PT, jumps, ...) with a device closure.
                xas_gpu = use_gpu && model isa IMRPhenomXAS && method != "smc"
                runner = method == "smc" ? _run_smc : (use_gpu && !xas_gpu) ? _run_nuts_gpu : _run_nuts
                result, nrows = runner(data, model; nwalkers=nwalkers, nsteps=nsteps, step_size=step_size,
                                        nwarmup=nwarmup, max_treedepth=max_treedepth, rng=rng,
                                        enforce_bounds=enforce_bounds, record_chain=record_chain,
                                        outdir=outdir, resume=resume, adapt_mass=adapt_mass, mass_opts=mass_opts, quorum_opts=quorum_opts,
                                        lik_opts=lik_opts, prior=prior, init=init,
                                        (method == "smc" ? (; smc_opts=smc_opts, device=device) : (;))...,
                                        ((use_gpu && !xas_gpu) || method == "smc" ? (;) : (; jumps=get(sampler, "jumps", false),
                                                               device=device,
                                                               de_jumps=get(sampler, "de_jumps", false),
                                                               pt_ntemps=get(sampler, "pt_ntemps", 1),
                                                               pt_beta_min=get(sampler, "pt_beta_min", 0.05),
                                                               pt_swap_interval=get(sampler, "pt_swap_interval", 1),
                                                               pt_adapt=get(sampler, "pt_adapt", true),
                                                               pt_deo=get(sampler, "pt_deo", true),
                                                               pt_ntemps_max=get(sampler, "pt_ntemps_max", 12),
                                                               pt_snr2=ref_snr2,
                                                               pt_reference_opts=(; (Symbol(k[4:end]) => sampler[k] for k in
                                                                   ("pt_reference", "pt_reference_kind", "pt_reference_inflate",
                                                                    "pt_reference_alpha", "pt_reference_kernels",
                                                                    "pt_reference_neighbours", "pt_reference_local_h",
                                                                    "pt_reference_time_shift") if haskey(sampler, k))...),
                                                               xas_rb=use_rb_xas ? (; nbins=Int(get(sampler, "relative_binning_nbins", 300)),
                                                                   rebuild_at=Int.(get(sampler, "relative_binning_rebuild_at",
                                                                       filter(<(nwarmup), unique([10, 25, 50, 100, nwarmup ÷ 2, (3 * nwarmup) ÷ 4]))))) :
                                                                   nothing))...)
            end
        finally
            GravitationalWaveBayes.BatchedNUTS._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[] = reverse_default
        end
        println("Final logp: ", result.logp, "  (all finite? ", all(isfinite, result.logp), ")")
        if (lik_opts.marginalize_phase || lik_opts.marginalize_distance) && haskey(result, :pe_cfg)
            # The marginalized rows only carried prior draws; replace them with exact draws from
            # their conditional posterior so the written chain is a joint posterior sample.
            recon_rng = Random.Xoshiro(hash((seed, "marginalized")))
            recon_ll = _recon_loglik(model, use_gpu, result.data_for_grad, result.pe_cfg, size(result.positions, 2);
                                     calibration, calmarg=calmarg === nothing ? nothing : calmarg.C, prior, spin_amax=_spin_amax(cfg))
            # calibration marginalization + distance marginalization: (curve, d_L) drawn jointly per sample
            # (`_calmarg_reconstruct_distance!`), the curves written as the calibration posterior
            curve_ll = (calmarg !== nothing && lik_opts.marginalize_distance) ?
                _recon_curve_loglik(model, use_gpu, result.data_for_grad, result.pe_cfg, size(result.positions, 2), calmarg.C;
                                    prior, spin_amax=_spin_amax(cfg)) : nothing
            reconstruct_marginalized!(result.positions, result.data_for_grad, result.pe_cfg, model; rng=recon_rng,
                                      loglik=recon_ll, curve_loglik=curve_ll)
            kchain = (curve_ll === nothing || result.chain === nothing) ? nothing :
                     zeros(Int, size(result.chain, 2), size(result.chain, 3))
            result.chain === nothing ||
                reconstruct_marginalized!(result.chain, result.data_for_grad, result.pe_cfg, model; rng=recon_rng,
                                          loglik=recon_ll, curve_loglik=curve_ll, curves=kchain)
            kchain === nothing || _write_calmarg_draws(outdir, kchain, calmarg)
            println("Reconstructed the marginalized parameters from their conditional posterior.")
        end
        if calmarg !== nothing && result.chain !== nothing && !lik_opts.marginalize_distance
            # marginalized calibration: per chain sample (theta, d_L reconstructed), a response curve drawn with weight
            # exp(ll_plain_k) -- a joint posterior draw of the calibration (written to calibration_draws.jls)
            _calmarg_curve_draws(outdir, result.chain, result.data_for_grad, result.pe_cfg, calmarg, prior,
                                 _spin_amax(cfg), Random.Xoshiro(hash((seed, "curves"))); model)
        end
        # get(result, :nwarmup_eff, nwarmup): a resumed run's returned chain only covers the
        # draws remaining after the checkpoint, so diagnostics need the REDUCED post-resume
        # warmup cutoff (see _run_nuts's/_run_nuts_gpu's nwarmup_eff), not the original config
        # value -- a non-resumed result has no such field and falls back to `nwarmup` unchanged.
        if write
            _write_outputs(outdir, config_path, nrows; positions=result.positions, logp=result.logp,
                            chain=result.chain, nwarmup=get(result, :nwarmup_eff, nwarmup))
            _source_frame_weights(outdir, cfg, prior, result)
            # Adapted metric + step sizes, for diagnosing sampler efficiency afterwards.
            haskey(result, :inv_mass) && serialize(joinpath(outdir, "adapted_metric.jls"),
                                                  (; inv_mass=result.inv_mass, step_size=result.step_size))
        end
        return (; result, nrows, nwarmup=get(result, :nwarmup_eff, nwarmup))

    elseif sampler_type == "pt"
        ntemps = get(sampler, "ntemps", 4)
        nwarmup = get(sampler, "nwarmup", max(nsteps ÷ 2, 1))
        beta_min = get(sampler, "beta_min", 1e-2)
        swap_interval = get(sampler, "swap_interval", 5)
        adapt_mass = get(sampler, "adapt_mass", false)
        println("Sampler: parallel tempering, ntemps=$ntemps nsteps=$nsteps nwarmup=$nwarmup")

        nrows = (model isa GravitationalWaveBayes.KernelFused._Prec15) ? 15 : 11
        state, pt, cold_chain = run_gw150914_pt_estimation(data; ntemps=ntemps, nsteps=nsteps,
                                                             step_size=step_size, beta_min=beta_min,
                                                             swap_interval=swap_interval,
                                                             nwarmup=nwarmup, max_treedepth=max_treedepth,
                                                             rng=rng, model=model,
                                                             enforce_bounds=enforce_bounds,
                                                             adapt_mass=adapt_mass)
        println("Final cold-chain logp: ", state.logp, "  (all finite? ", all(isfinite, state.logp), ")")
        _write_outputs(outdir, config_path, nrows; positions=state.positions, logp=state.logp,
                        cold_chain=cold_chain, pt=pt)
        return nothing
    else
        throw(ArgumentError("sampler.type must be \"nuts\" or \"pt\", got \"$sampler_type\""))
    end
end

function main(config_path::AbstractString)
    cfg = TOML.parsefile(config_path)

    println("Loading detector data...")
    data = _build_detector_data(cfg)
    println("  ", length(data), " detector(s): ", join([d.detector.prefix for d in data], ", "))

    model_name = get(get(cfg, "model", Dict()), "name", "IMRPhenomXAS")
    model = _lookup(_MODELS, model_name, "model")
    # [model] prec_version: LAL's IMRPhenomXPrecVersion for IMRPhenomXPHM (223 MSA = LAL default, or 102 NNLO)
    pv = get(get(cfg, "model", Dict()), "prec_version", nothing)
    if pv !== nothing
        model isa IMRPhenomXPHM || throw(ArgumentError("[model] prec_version is for IMRPhenomXPHM only"))
        model = IMRPhenomXPHM(pv)
    end
    println("Model: ", model_name, model isa IMRPhenomXPHM ? " (PrecVersion $(typeof(model).parameters[1]))" : "")

    output0 = get(cfg, "output", Dict{String,Any}())
    outdir0 = get(output0, "dir", "run_output")
    truth_sampled = nothing
    ref_snr2 = nothing
    if haskey(cfg, "injection") && model isa SEOBNRv5PHM
        data = _phm_injection!(data, cfg, outdir0)
    elseif haskey(cfg, "injection")
        inj = cfg["injection"]
        physical, truth_sampled = _injection_truth(inj)
        inj_rng = Random.Xoshiro(Int(get(inj, "seed", 1)))
        # `[injection] model` (default: the recovery model) injects with a different waveform, e.g. SEOBNRv5
        # IMR data analysed with a cheaper PA-F2 model.
        inj_model = haskey(inj, "model") ? _lookup(_MODELS, inj["model"], "injection.model") : model
        inj_model === model || println("Injecting with ", typeof(inj_model), " (recovery model ", typeof(model), ")")
        inj_model isa PAF2Model || throw(ArgumentError("[injection] needs a PA-F2/IMR-table injection model (got $(typeof(inj_model)))"))
        data, info = simulate_injection(inj_model, physical, data; rng=inj_rng, noise=get(inj, "noise", true),
                                        fmax_taper=_fmax_taper(cfg))
        ref_snr2 = info.network_optimal_snr^2
        println("Injection (simulated Gaussian noise from the real PSDs): optimal SNR ",
                round.(info.optimal_snr; digits=2), " network ", round(info.network_optimal_snr; digits=2),
                ", matched-filter network ", round(info.network_matched_filter_snr; digits=2))
        mkpath(outdir0)
        open(joinpath(outdir0, "injection.txt"), "w") do io
            println(io, "network_optimal_snr = ", info.network_optimal_snr)
            println(io, "network_matched_filter_snr = ", info.network_matched_filter_snr)
        end
    end
    # [calibration] priors = "<bilby prior strings>": the spline calibration model of that analysis, sampled (standard
    # normal rows 16:15 + 4 nn after the 15 precessing rows; IMRPhenomXPHM device path), see `GravitationalWaveBayes.calibration_prior`
    # [calibration] mode (bilby's three workflows): no table = omitted; "sample" (default) = the node values sampled;
    # "marginalize" = averaged in the likelihood over `ncurves` (1000, bilby's default) prior response curves drawn with
    # `curve_seed` (Payne et al. 2020; SEOBNRv5PHM device path), no sampled rows.
    calibration = nothing; calmarg = nothing
    if haskey(cfg, "calibration")
        cc = cfg["calibration"]
        calp = calibration_prior(cc["priors"], [d.detector.prefix for d in data], data[1].freqs)
        mode = get(cc, "mode", "sample")
        if mode == "sample"
            calibration = calp
            println("Calibration: ", calp.nn, " spline nodes per detector (", round(calp.nodes[1]; digits=1), "-",
                    round(calp.nodes[end]; digits=1), " Hz), ", length(calp.mu), " sampled rows")
        elseif mode == "marginalize"
            nc = Int(get(cc, "ncurves", 1000))
            calmarg = (; calibration_curves(calp, nc; rng=Random.Xoshiro(Int(get(cc, "curve_seed", 1))))..., prior=calp)
            println("Calibration: ", calp.nn, " spline nodes per detector (", round(calp.nodes[1]; digits=1), "-",
                    round(calp.nodes[end]; digits=1), " Hz), MARGINALIZED over ", nc, " prior response curves")
        else
            throw(ArgumentError("[calibration] mode must be \"sample\" or \"marginalize\", got \"$mode\""))
        end
    end
    prior = _parse_prior(cfg; ncal=calibration === nothing ? 0 : length(calibration.mu))
    lik_opts = _parse_likelihood_opts(cfg, data, prior; model=model,
                                      ref_intrinsic=GravitationalWaveBayes.KernelFused._has_likelihood_options(model) ? _reference_intrinsic(truth_sampled, prior) : nothing)
    (lik_opts.marginalize_phase || lik_opts.marginalize_distance || lik_opts.time_reference != 0 ||
     lik_opts.time_ref_frequency > 0 || lik_opts.fmax_taper > 0) &&
        println("Likelihood options: ", lik_opts)
    prior === nothing || println("Prior: distance=", prior.distance, " masses=", prior.masses,
                                 " lo=", prior.lo, " hi=", prior.hi)
    if truth_sampled !== nothing
        truth_frame = _sampled_truth_in_frame(truth_sampled, data, lik_opts, model)
        writedlm(joinpath(outdir0, "injection_sampled_truth.csv"),
                 vcat(reshape(_PARAM_NAMES_11, 1, :), reshape(truth_frame, 1, :)), ',')
    end

    return _run_sampler(cfg, config_path, data, model, prior, lik_opts; calibration, calmarg, truth_sampled, ref_snr2)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("Usage: julia --project=. scripts/run_from_config.jl path/to/config.toml")
    main(ARGS[1])
end
