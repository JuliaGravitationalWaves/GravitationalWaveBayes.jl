# Importance reweighting of a finished run's samples to another model's likelihood (multi-fidelity without the
# expensive model's gradients): w = L_target(x') / L_source(x), same data and prior, x' = x with the time row moved
# from the source run's time convention to the target's (each `likelihood.time_ref_frequency` / offset; the other
# rows, incl. phase and in-plane spins at f_ref, mean the same in both). Reports the effective sample size and
# weighted vs unweighted summaries; writes the weights and an equal-weight resample.
#   julia -t 8 --project=<env with CUDA> scripts/reweight_posterior.jl <target config.toml> <source run dir> [outdir]
# e.g. an IMRPhenomXPHM SMC run reweighted to SEOBNRv5PHM:
#   scripts/reweight_posterior.jl scripts/configs/gw150914_h1l1_seobnrv5phm_smc1024_cuda.toml runs/.../xphm_smc1024_cuda
# `--chain`: reweight every recorded draw of the source run's chain.jls instead of its final positions (more samples,
# larger ESS); likelihoods are evaluated in chunks of 1024.
include(joinpath(@__DIR__, "run_from_config.jl"))
using Statistics, Printf

# (nrows, nsamples): the run's final positions, or with `chain = true` every recorded draw of `chain.jls` (an SMC run's
# beta = 1 phase: all draws are posterior samples)
function _read_positions(dir; chain::Bool=false)
    if chain
        c = deserialize(joinpath(dir, "chain.jls")).chain
        return Matrix{Float64}(reshape(c, size(c, 1), :))
    end
    raw = readdlm(joinpath(dir, "final_positions.csv"), ',')
    return permutedims(Matrix{Float64}(raw[2:end, :]))         # (nrows, nsamples)
end

# t*(f) of a sample, as the likelihood kernel computes it (IMRPhenomXPHM: the XHM (2,2) time with the precessing final
# spin, `BatchedNUTS._xphmbp_setup`; the others: `time_at_frequency`)
function _tstar(model, x, f)
    if model isa IMRPhenomXPHM
        BN = GravitationalWaveBayes.BatchedNUTS
        th = BN._mbp_theta(BN._XPSpec(), x)
        _, sys = BN._xphmbp_setup(th..., ntuple(_ -> 0.0, Val(44)), 20.0)
        return GravitationalWaveBayes.KernelFused._xhm_time_shift(sys, (fref = Float64(f), toff = 0.0))
    end
    m1, m2 = _mc_q_to_m1_m2(x[1], x[2])
    return time_at_frequency(model, m1, m2, x[3], x[4], f)
end
# time row in `lik` convention -> time row in `lik2` convention
function _convert_time_row(row, x, model, lik, model2, lik2)
    tc = lik.time_ref_frequency > 0 ? row - _tstar(model, x, lik.time_ref_frequency) - lik.time_ref_offset : row
    return lik2.time_ref_frequency > 0 ? tc + _tstar(model2, x, lik2.time_ref_frequency) + lik2.time_ref_offset : tc
end

function _loglik(data, model, lik, X; chunk::Int=1024)
    T = Float64
    freqs = _representative_freqs(data)
    n = size(X, 2)
    nc = min(n, chunk)
    cfg = _pe_config(T, freqs, nc, 1e-4, 1, lik)
    dfg = (data isa AbstractVector && length(data) == 1) ? data[1] : data
    # SEOBNRv5PHM: the likelihood-only device closure (its gradient costs ~21 likelihoods); else the staged closure.
    f! = if model isa SEOBNRv5PHM
        GravitationalWaveBayes.BatchedNUTS.seobnr_phm_loglik(dfg, cfg; nwalkers=nc,
            backend=GravitationalWaveBayes.BatchedNUTS.KernelAbstractions.get_backend(GravitationalWaveBayes.GPU.to_device(zeros(T, 1))))
    else
        base! = gw_logp_and_grad_gpu_staged(dfg, cfg, freqs; model=model, nparams=size(X, 1), nwalkers=nc)
        g = zeros(T, size(X, 1), nc)
        (l, x) -> (base!(g, l, x, nothing); l)
    end
    ll = zeros(T, n); buf = zeros(T, nc); Xb = zeros(T, size(X, 1), nc)
    function run!()
        for s in 1:nc:n
            e = min(s + nc - 1, n); m = e - s + 1
            Xb[:, 1:m] .= view(X, :, s:e)
            m < nc && (Xb[:, m+1:end] .= view(X, :, s:s))        # pad the last chunk with a valid sample
            f!(buf, Xb)
            ll[s:e] .= view(buf, 1:m)
        end
    end
    run!()
    t = @elapsed run!()
    return ll, t
end

function wquantile(x, w, p)
    o = sortperm(x); c = cumsum(w[o]) ./ sum(w)
    return x[o[clamp(searchsortedfirst(c, p), 1, length(x))]]
end

function reweight(tgt_path, srcdir, outdir; chain::Bool=false)
    tgt = TOML.parsefile(tgt_path)
    src = TOML.parsefile(joinpath(srcdir, "config_used.toml"))
    data = _build_detector_data(tgt)
    prior = _parse_prior(tgt)
    msrc = _MODELS[src["model"]["name"]]; mtgt = _MODELS[tgt["model"]["name"]]
    lsrc = _parse_likelihood_opts(src, data, prior; model=msrc)
    ltgt = _parse_likelihood_opts(tgt, data, prior; model=mtgt)
    X = _read_positions(srcdir; chain=chain)
    n = size(X, 2)
    println("source: ", typeof(msrc), " (", n, " samples from ", srcdir, "); target: ", typeof(mtgt))
    Xt = copy(X)
    for j in 1:n
        Xt[6, j] = _convert_time_row(X[6, j], view(X, :, j), msrc, lsrc, mtgt, ltgt)
    end
    lls, ts = _loglik(data, msrc, lsrc, X)
    llt, tt = _loglik(data, mtgt, ltgt, Xt)
    @printf("log-likelihoods: source %.2f s, target %.2f s (second calls, %d samples)\n", ts, tt, n)
    # the two models' raw coalescence times differ (e.g. XHM (2,2) time vs SEOBNRv5PHM's frame-invariant peak): a
    # constant shift of the time row (any deterministic shift keeps the importance weights valid) chosen for the ESS
    essof(l) = (lv = l .- lls; lv[.!isfinite.(lv)] .= -Inf; wv = exp.(lv .- maximum(lv)); 1 / sum(abs2, wv ./ sum(wv)))
    best = (0.0, essof(llt), llt)
    for d in -4e-3:5e-4:4e-3
        d == 0 && continue
        Xd = copy(Xt); Xd[6, :] .+= d
        l, _ = _loglik(data, mtgt, ltgt, Xd)
        e = essof(l)
        @printf("  time shift %+.1f ms: ESS %.1f\n", 1e3 * d, e)
        e > best[2] && (best = (d, e, l))
    end
    tshift, _, llt = best
    Xt[6, :] .+= tshift
    @printf("time shift chosen %+.1f ms\n", 1e3 * tshift)
    lw = llt .- lls
    ok = isfinite.(lw)
    lw[.!ok] .= -Inf
    w = exp.(lw .- maximum(lw)); w ./= sum(w)
    ess = 1 / sum(abs2, w)
    @printf("log w: mean %.3f sd %.3f (finite %d/%d); ESS %.1f of %d (%.1f%%)\n", mean(lw[ok]), std(lw[ok]), count(ok), n, ess,
        n, 100 * ess / n)
    names = size(X, 1) == 15 ? _PARAM_NAMES_15 : _PARAM_NAMES_11
    println(rpad("param", 14), "   unweighted median [5%, 95%]        reweighted median [5%, 95%]")
    u = fill(1.0 / n, n)
    for (k, nm) in enumerate(names)
        x = Xt[k, :]
        @printf("%-14s %10.4f [%9.4f, %9.4f]   %10.4f [%9.4f, %9.4f]\n", nm, wquantile(x, u, 0.5), wquantile(x, u, 0.05),
            wquantile(x, u, 0.95), wquantile(x, w, 0.5), wquantile(x, w, 0.05), wquantile(x, w, 0.95))
    end
    mkpath(outdir)
    writedlm(joinpath(outdir, "reweight.csv"), vcat(reshape(vcat(names, ["loglik_source", "loglik_target", "weight"]), 1, :),
        hcat(permutedims(Xt), lls, llt, w)), ',')
    rng = Random.Xoshiro(1)
    idx = [searchsortedfirst(cumsum(w), rand(rng)) for _ in 1:n]
    writedlm(joinpath(outdir, "reweighted_resample.csv"), vcat(reshape(names, 1, :), permutedims(Xt[:, clamp.(idx, 1, n)])), ',')
    open(joinpath(outdir, "reweight_summary.txt"), "w") do io
        println(io, "source = ", srcdir); println(io, "target = ", tgt_path)
        println(io, "nsamples = ", n); println(io, "ess = ", ess); println(io, "time_shift_s = ", tshift)
        println(io, "logw_mean = ", mean(lw[ok])); println(io, "logw_sd = ", std(lw[ok]))
        # log Z_target - log Z_source estimate (same prior): log mean of the likelihood ratio over the source posterior
        mx = maximum(lw); println(io, "log_bayes_target_over_source = ", mx + log(mean(exp.(lw .- mx))))
    end
    # the source samples in the target's time convention (incl. the fitted shift), unweighted, as a run directory an SMC
    # can use as `sampler.smc_reference` (multi-fidelity: anneal from the cheap model's posterior)
    refdir = joinpath(outdir, "source_in_target_frame")
    mkpath(refdir)
    serialize(joinpath(refdir, "chain.jls"), (; names, chain=reshape(Xt, size(Xt, 1), n, 1)))
    cp(joinpath(srcdir, "config_used.toml"), joinpath(refdir, "config_used.toml"); force=true)
    println("wrote ", outdir, " (and ", refdir, ")")
end

if abspath(PROGRAM_FILE) == @__FILE__
    use_chain = "--chain" in ARGS
    args = filter(!=("--chain"), ARGS)
    length(args) >= 2 || error("Usage: scripts/reweight_posterior.jl <target config.toml> <source run dir> [outdir] [--chain]")
    reweight(args[1], args[2], length(args) >= 3 ? args[3] :
             joinpath(args[2], "reweight_" * splitext(basename(args[1]))[1] * (use_chain ? "_chain" : "")); chain=use_chain)
end
