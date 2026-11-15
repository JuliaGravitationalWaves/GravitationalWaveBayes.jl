# Probability-probability (simulation-based calibration) test.
#
# Draws `N` injections from the config's own [prior], simulates each into stationary Gaussian noise
# (real-data PSDs from the config's [[detectors]]), runs the config's sampler on each, and records
# the posterior CDF value of the truth for every sampled parameter. If the likelihood, prior and sampler are
# all correct those percentiles are Uniform(0,1) per parameter -- the standard end-to-end calibration
# check for a PE code (bilby's P-P plot). Deviations show up in the KS p-values `pp_summary.jl` prints.
# A 1-D P-P is nearly blind to a posterior that under-weights a small correlated region, so each row also records
# derived chi_eff / chi_p percentiles, the posterior mass of prior-quantile tails (`mass_*`, mean must equal the prior
# mass) and a TARP coverage value (`tarp`, joint coverage) -- see `_pp_reference` / `_pp_tarp`.
#
# The sampler is exactly `run_from_config.jl`'s (`_run_sampler`: NUTS, PT or SMC with all its [sampler] options, CPU or
# CUDA), so the test covers the configuration the analyses use. Injections:
#   - PA-F2 / IMR-table models: `simulate_injection(model, physical, ...)`; walkers start `init_scale` prior-widths
#     around the truth (sampler.init is forced to "injection" for NUTS -- see the caveat in scripts/configs/README.md:
#     a sampler that never leaves its start shows up as percentiles piled up near 0.5, an S-shaped P-P curve).
#   - IMRPhenomXP / IMRPhenomXPHM / SEOBNRv5PHM (15 rows): the likelihood's own detector templates
#     (`KernelFused.xprec_strain`, `KernelFused.seobnr_phm_device_strain`) at the truth, which is drawn in the sampled
#     space and therefore already in the likelihood's time frame (time_reference / time_ref_frequency) and spin
#     parameterization (`spins = "isotropic"`: y in the unit ball). The sampler starts as configured (SMC: the prior).
#
# One Julia process loops over injections (compile once). Results are appended to
# `<outdir>/pp_results.csv` after every injection, so a killed run keeps what it finished and a rerun
# with the same arguments skips injections already done. Several processes (e.g. one per GPU) can share an outdir
# with disjoint index ranges: each writes `pp_results_<first_index>.csv`; `pp_summary.jl` reads them all.
#
# Usage:
#   julia -t 8 --project=. scripts/pp_test.jl <base_config.toml> <N> <outdir> [first_index=1] [keep_chains=0]
#
# The base config needs [prior], [likelihood] (optional), [model] and a [sampler] section; its [injection] section, if
# any, is ignored (truths come from the prior). keep_chains=1 writes each injection's run outputs to <outdir>/inj_<i>/.

include(joinpath(@__DIR__, "run_from_config.jl"))

using GravitationalWaveforms: _seobv5f2_coeffs
using Printf
using Statistics: quantile
using GravitationalWaveData: time_delay_from_geocenter_gmst

const _KF = GravitationalWaveBayes.KernelFused

function _truth_is_usable(model::PAF2Model, sampled, f_min)
    m1, m2 = _mc_q_to_m1_m2(sampled[1], sampled[2])
    mc, eta = _KF._chirp_mass_eta(m1, m2)
    c = _seobv5f2_coeffs(model, mc, eta, sampled[3], sampled[4])
    return c.success && c.fmax > f_min
end

# PA-F2: sampled-space draw (in the likelihood's time frame) -> physical tuple with GEOCENTRIC t_c for injection.
function _physical_from_sampled(sampled, data, lik_opts, model)
    m1, m2 = _mc_q_to_m1_m2(sampled[1], sampled[2])
    ra, dec = sampled[8], asin(sampled[9])
    t = sampled[6]
    if lik_opts.time_ref_frequency > 0
        t -= time_at_frequency(model, m1, m2, sampled[3], sampled[4], lik_opts.time_ref_frequency) + lik_opts.time_ref_offset
    end
    if length(data) > 1 && lik_opts.time_reference > 0
        g = GravitationalWaveBayes.BatchedNUTS.gmst(data[1].gps_time)
        t -= time_delay_from_geocenter_gmst(data[lik_opts.time_reference].detector, ra, dec, g)
    end
    return (m1, m2, sampled[3], sampled[4], sampled[5], t, sampled[7], ra, dec, acos(sampled[10]), sampled[11])
end

# Precessing models: sampled 15-vector -> the likelihood's physical rows (time row unchanged: the likelihood's frame).
function _prec_kernel_params(sampled, spin_amax)
    pos = reshape(Float64.(sampled[1:15]), 15, 1)
    spin_amax === nothing || (pos = isotropic_spins!(similar(pos), pos, spin_amax))
    phys = similar(pos)
    GravitationalWaveBayes.BatchedNUTS._phm_to_physical!(phys, pos)
    return phys[:, 1]
end

# One injection: (data, info) or `nothing` when the truth has no usable waveform (redraw).
function _pp_injection(model, truth, templates, lik_opts, pe_cfg, spin_amax, f_min, rng, cfg)
    if model isa PAF2Model
        _truth_is_usable(model, truth, f_min) || return nothing
        physical = _physical_from_sampled(truth, templates, lik_opts, model)
        return simulate_injection(model, physical, templates; rng=rng, noise=true, fmax_taper=_fmax_taper(cfg))
    end
    p = _prec_kernel_params(truth, spin_amax)
    h, ok = model isa SEOBNRv5PHM ? _KF.seobnr_phm_device_strain(p, pe_cfg, templates) :
                                    (_KF.xprec_strain(model, p, pe_cfg, templates), true)
    (ok && all(hk -> all(isfinite, hk), h)) || return nothing
    return simulate_injection(h, templates; rng=rng, noise=true)
end

# Derived quantities of sampled-space draws (columns): chi_eff, plus chi_p for the precessing 15-row layout. A 1-D P-P
# on the sampled rows hardly sees a sampler that under-weights a correlated region (the 2026-10 XPHM GW150914 low-Mc /
# anti-aligned corner); these feed derived P-P columns and the region-mass test.
function _pp_derived(x::AbstractMatrix, spin_amax)
    p = spin_amax === nothing ? x : isotropic_spins!(similar(x), x, spin_amax)
    q = p[2, :]
    d = Dict("q" => q, "chi_eff" => (p[3, :] .+ q .* p[4, :]) ./ (1 .+ q))
    size(p, 1) == 15 && (d["chi_p"] = max.(hypot.(p[12, :], p[13, :]),
                                           (4 .* q .+ 3) ./ (4 .+ 3 .* q) .* q .* hypot.(p[14, :], p[15, :])))
    return d
end

# Fixed (seeded) prior reference shared by every process writing to one outdir: per-row ranges for TARP, and region
# thresholds at prior quantiles of the derived quantities. A region {f < t} (or {f > t}) with prior mass m has, by the
# SBC identity (the posterior averaged over prior-drawn data is the prior), mean posterior mass m over injections; a
# sampler that under-samples a tail shows a deficit there. Column `mass_<f>_<lt|gt>_<m>` = posterior mass of the region.
const _PP_TAILS = (0.02, 0.1)
function _pp_reference(prior, spin_amax)
    x = sample_prior(Random.Xoshiro(1), prior, 2_000_000)      # threshold mass error ~2e-4
    lo, hi = vec(minimum(x; dims=2)), vec(maximum(x; dims=2))
    regions = Tuple{String,String,Float64,Float64}[]          # (quantity, "lt"/"gt", prior mass, threshold)
    for (f, v) in sort(collect(_pp_derived(x, spin_amax)); by=first), m in _PP_TAILS
        push!(regions, (f, "lt", m, quantile(v, m)), (f, "gt", m, quantile(v, 1 - m)))
    end
    return (; lo, hi, regions)
end
_pp_region_name(r) = @sprintf("mass_%s_%s_%g", r[1], r[2], r[3])
_pp_region_mass(d, r) = r[2] == "lt" ? count(<(r[4]), d[r[1]]) / length(d[r[1]]) : count(>(r[4]), d[r[1]]) / length(d[r[1]])

# TARP (Lemos et al. 2023, arXiv:2302.03026): one reference point per injection, uniform in the prior box, drawn
# independently of the truth; f = posterior fraction closer to it than the truth (range-normalized Euclidean distance
# over all sampled rows). Uniform(0,1) for a correct posterior; tests JOINT coverage, sensitive to over-confidence.
function _pp_tarp(post::AbstractMatrix, truth, ref, rng)
    s = 1 ./ (ref.hi .- ref.lo)
    r = ref.lo .+ rand(rng, length(ref.lo)) .* (ref.hi .- ref.lo)
    dt = sum(abs2, (truth .- r) .* s)
    return count(j -> sum(abs2, (view(post, :, j) .- r) .* s) < dt, axes(post, 2)) / size(post, 2)
end

function pp_main(config_path, N, outdir, first_index; keep_chains::Bool=false)
    cfg = TOML.parsefile(config_path)
    haskey(cfg, "calibration") && error("pp_test.jl: [calibration] is not supported (injections carry no calibration error)")
    templates = _build_detector_data(cfg)
    model = _lookup(_MODELS, get(get(cfg, "model", Dict()), "name", "SEOBNRv5AlignedSpinF2"), "model")
    pv = get(get(cfg, "model", Dict()), "prec_version", nothing)
    pv === nothing || (model = IMRPhenomXPHM(pv))
    model isa Union{PAF2Model,IMRPhenomXP,IMRPhenomXPHM,SEOBNRv5PHM} ||
        error("pp_test.jl injects with the PA-F2 / IMR-table models, IMRPhenomXP, IMRPhenomXPHM and SEOBNRv5PHM only")
    nrows = model isa _KF._Prec15 ? 15 : 11
    names = nrows == 15 ? _PARAM_NAMES_15 : _PARAM_NAMES_11
    prior = _parse_prior(cfg)
    prior === nothing && error("pp_test.jl needs a [prior] section (truths are drawn from it)")
    lik_opts = _parse_likelihood_opts(cfg, templates, prior; model=model, ref_intrinsic=_reference_intrinsic(nothing, prior))
    pe_cfg = _pe_config(Float64, Array(templates[1].freqs), 1, 1e-3, 1, lik_opts)
    pcfg = get(cfg, "prior", Dict())
    spin_amax = _spin_amax(cfg)
    s = cfg["sampler"]
    nwalkers = get(s, "nwalkers", 16)
    f_min = get(get(cfg, "event", Dict()), "f_min", 20.0)
    smc = get(s, "method", "nuts") == "smc"

    mkpath(outdir)
    cp(config_path, joinpath(outdir, "base_config.toml"); force=true)
    csv = joinpath(outdir, first_index == 1 ? "pp_results.csv" : "pp_results_$(first_index).csv")
    ref = _pp_reference(prior, spin_amax)
    dnames = sort(setdiff(collect(keys(_pp_derived(zeros(nrows, 1), nothing))), names))    # derived P-P columns (q is sampled)
    header = join(vcat(["index", "net_snr", "mf_snr", "seconds", "max_rhat", "min_ess"],
                       ["truth_" * n for n in names], ["pct_" * n for n in vcat(names, dnames)],
                       ["rhat_" * n for n in names], _pp_region_name.(ref.regions), ["tarp"]), ',')
    done = Set{Int}()
    if isfile(csv)
        lines = collect(eachline(csv))
        first(lines) == header || error("$csv was written with different columns (older pp_test.jl?): use a new outdir")
        foreach(l -> push!(done, parse(Int, split(l, ',')[1])), lines[2:end])
    else
        open(io -> println(io, header), csv, "w")
    end

    for i in first_index:(first_index + N - 1)
        i in done && continue
        rng = Random.Xoshiro(10_000 + i)
        local truth, inj
        while true
            truth = sample_prior(rng, prior, 1)[:, 1]
            inj = _pp_injection(model, truth, templates, lik_opts, pe_cfg, spin_amax, f_min, rng, cfg)
            inj === nothing || break
        end
        data, info = inj
        # NUTS (the PA-F2 P-P): walkers start around the truth, as before; SMC starts from the prior.
        init = smc ? nothing : _init_positions("injection", rng, nwalkers; truth=truth, prior=prior,
                                               scale=get(s, "init_scale", 0.02))
        t0 = time()
        run = _run_sampler(cfg, config_path, data, model, prior, lik_opts; ref_snr2=info.network_optimal_snr^2,
                           outdir_override=joinpath(outdir, "inj_$i"), seed_override=10_000 + i, init_override=init,
                           write=keep_chains)
        secs = time() - t0
        chain = run.result.chain
        nw = run.nwarmup
        post = chain[1:nrows, :, (nw + 1):end]
        # R-hat / ESS need >= 4 draws per chain (SMC: smc_nfinal); with fewer, NaN (the percentiles are unaffected)
        diags = size(chain, 3) - nw >= 4 ? chain_diagnostics(chain[1:nrows, :, :]; warmup=nw, names=names) : nothing
        max_rhat = diags === nothing ? NaN : worst_rhat(diags).rhat
        ess = diags === nothing ? NaN : min_ess(diags).ess
        rhats = diags === nothing ? fill(NaN, nrows) : [d.rhat for d in diags]
        pct = [count(<(truth[r]), view(post, r, :, :)) / length(view(post, r, :, :)) for r in 1:nrows]
        flat = reshape(post, nrows, :)
        dpost, dtruth = _pp_derived(flat, spin_amax), _pp_derived(reshape(Float64.(truth[1:nrows]), nrows, 1), spin_amax)
        append!(pct, [count(<(dtruth[n][1]), dpost[n]) / length(dpost[n]) for n in dnames])
        masses = [_pp_region_mass(dpost, r) for r in ref.regions]
        tarp = _pp_tarp(flat, truth[1:nrows], ref, Random.Xoshiro(20_000 + i))
        open(csv, "a") do io
            println(io, join(vcat([string(i), @sprintf("%.3f", info.network_optimal_snr), @sprintf("%.3f", info.network_matched_filter_snr),
                                   @sprintf("%.1f", secs), @sprintf("%.4f", max_rhat), @sprintf("%.1f", ess)],
                                  [@sprintf("%.8g", x) for x in truth], [@sprintf("%.6f", x) for x in pct],
                                  [@sprintf("%.4f", x) for x in rhats], [@sprintf("%.6f", x) for x in masses],
                                  [@sprintf("%.6f", tarp)]), ','))
        end
        println(@sprintf("[pp %d] snr=%.2f  %.0fs  max_rhat=%.3f  pct=%s", i, info.network_optimal_snr, secs,
                         max_rhat, join([@sprintf("%.2f", x) for x in pct], " ")))
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) in 3:5 || error("Usage: julia --project=. scripts/pp_test.jl base.toml N outdir [first_index] [keep_chains]")
    pp_main(ARGS[1], parse(Int, ARGS[2]), ARGS[3], length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1;
            keep_chains=length(ARGS) == 5 && ARGS[5] == "1")
end
