# Per-call logp+gradient cost vs walker count, real GW150914 H1+L1 (1969 bins each), all likelihood options
# (phase + distance marginalized, f_ref timing, H1 time reference):
#   xas_cpu    IMRPhenomXAS, CPU per-walker Enzyme path (gw_logp_and_grad), Threads.nthreads() threads
#   xas_rb_cpu IMRPhenomXAS relative binning (300 requested bins), CPU per-walker
#   xas_gpu    IMRPhenomXAS bin-parallel (xas_binparallel_logp_and_grad) on CUDA   [needs `cuda` arg]
#   xas_res    the same with `resident = true` (per-walker stages as kernels)      [`cuda` + `resident` args]
#   xas_res_rb relative binning (300 bins) on the device, resident                   [`cuda` + `resident` args]
#   seob_cpu   SEOBNRv5 IMR, CPU per-walker                                           [needs `seob` arg]
#   seobf2_*   SEOBNRv5 inspiral-only PA-F2, CPU and existing GPU path (table build 1 thread/walker) [`seobf2`]
# `julia --project=<env> -t N scripts/bench_gpu_models.jl [cuda] [seob] [seobf2] [hand] [nocpu] [walkers=16,64,256]
#   [chunk=4] [reps=10] [heap=MB]`
const USE_CUDA = "cuda" in ARGS
USE_CUDA && using CUDA
using KernelAbstractions, GravitationalWaveBayes, Random, Printf
import GravitationalWaveforms
using GravitationalWaveforms: IMRPhenomXAS
using GravitationalWaveData: H1, L1
const BN = GravitationalWaveBayes.BatchedNUTS
kv = Dict(split(a, "=")[1] => split(a, "=")[2] for a in ARGS if occursin("=", a))
walkers = parse.(Int, split(get(kv, "walkers", "16,64,256"), ","))
chunk = parse(Int, get(kv, "chunk", "4"))
USE_CUDA && haskey(kv, "heap") && CUDA.limit!(CUDA.LIMIT_MALLOC_HEAP_SIZE, parse(Int, kv["heap"]) * 1024^2)
# One session builds closures for several walker counts; the heap can only grow before the first malloc-using kernel runs,
# so reserve it for the largest count now (what `gw_logp_and_grad_gpu_staged` would do for a single run).
USE_CUDA && GravitationalWaveBayes.GPU.ensure_device_heap(max(64, 2 * maximum(walkers)) << 20)
USE_CUDA && println("GPU: ", CUDA.name(CUDA.device()), "  device malloc heap ", CUDA.limit(CUDA.LIMIT_MALLOC_HEAP_SIZE) ÷ 1024^2, " MB")
println("CPU threads: ", Threads.nthreads(), "  (", Sys.cpu_info()[1].model, ")")

kw = (; trigger_time=1126259462.4, duration=4.0, post_trigger=2.0, psd_segment=1024.0, f_min=20.0,
      f_max=512.0, alpha=0.2, psd_method=:median)
h1 = build_gwdata("data/H-H1_LOSC_4_V1-1126256640-4096.hdf5"; detector=H1, kw...)
l1 = build_gwdata("data/L-L1_LOSC_4_V1-1126256640-4096.hdf5"; detector=L1, kw...)
data = [h1, l1]
freqs = h1.freqs
opts = (; marginalize_phase=true, marginalize_distance=true, distance_lo=0.05, distance_hi=1.5, distance_power=2,
        time_reference=1, time_ref_frequency=35.0, time_ref_offset=0.12283266110168832)
center = [31.0, 0.85, 0.0, 0.0, 0.45, 0.029, 1.0, 1.9, -0.9, -0.8, 1.0]
spread = [0.5, 0.05, 0.1, 0.1, 0.05, 0.001, 0.3, 0.1, 0.05, 0.1, 0.3]

function timeit(f!, g, l, pos; reps)
    f!(g, l, pos, nothing)                                   # compile / warm
    USE_CUDA && CUDA.synchronize()
    ts = [(@elapsed (f!(g, l, pos, nothing); USE_CUDA && CUDA.synchronize())) for _ in 1:reps]
    return ts
end

println(rpad("case", 12), rpad("walkers", 9), rpad("mean ms", 10), rpad("min ms", 10), "min ms/walker-grad")
for nw in walkers
    pos = center .+ spread .* randn(Random.Xoshiro(nw), 11, nw)
    pos[2, :] .= clamp.(pos[2, :], 0.3, 1.0)
    cfg = with_likelihood_options(PEConfig{Float64}(length(freqs), nw, 1e-3, 1); opts...)
    g = zeros(11, nw); l = zeros(nw)
    cases = Pair{String,Any}["xas_cpu" => BN.gw_logp_and_grad(data, cfg, freqs; model=IMRPhenomXAS())]
    rb = BN.build_xas_relative_binning(data, cfg, center; nbins=300)
    push!(cases, "xas_rb_cpu" => BN.gw_logp_and_grad(data, cfg, freqs; model=IMRPhenomXAS(), relative_binning=rb))
    tim = zeros(5)
    USE_CUDA && push!(cases, "xas_gpu" => BN.xas_binparallel_logp_and_grad(data, cfg, freqs; backend=CUDABackend(),
                                                                           nwalkers=nw, chunk=chunk, timers=tim))
    timr = zeros(5)
    "resident" in ARGS && USE_CUDA && push!(cases, "xas_res" => BN.xas_binparallel_logp_and_grad(data, cfg, freqs;
                                                  backend=CUDABackend(), nwalkers=nw, chunk=chunk, resident=true, timers=timr))
    timrb = zeros(5)
    "resident" in ARGS && USE_CUDA && push!(cases, "xas_res_rb" => BN.xas_binparallel_logp_and_grad(data, cfg, freqs;
                                                  backend=CUDABackend(), nwalkers=nw, chunk=chunk, relative_binning=rb, timers=timrb))
    "hand" in ARGS && "resident" in ARGS && USE_CUDA && push!(cases, "xas_res_hand" => BN.xas_binparallel_logp_and_grad(data, cfg, freqs;
                                                  backend=CUDABackend(), nwalkers=nw, chunk=chunk, resident=true, adjoint=:hand))
    "hand" in ARGS && USE_CUDA && push!(cases, "xas_gpu_hand" => BN.xas_binparallel_logp_and_grad(data, cfg, freqs;
                                                  backend=CUDABackend(), nwalkers=nw, chunk=chunk, adjoint=:hand))
    "nocpu" in ARGS && filter!(c -> !endswith(c.first, "_cpu"), cases)
    if "seob" in ARGS
        push!(cases, "seob_cpu" => BN.gw_logp_and_grad(data, cfg, freqs; model=GravitationalWaveforms.SEOBNRv5IMRAlignedSpinF2()))
    end
    # SEOBNRv5 inspiral-only (PA-F2): the existing one-thread-per-walker table build on the GPU vs the CPU path. This is
    # the measurement that says whether FP64 GPUs pay off for the EOB table build (the IMR model has no GPU path yet).
    if "seobf2" in ARGS
        m = GravitationalWaveforms.SEOBNRv5AlignedSpinF2()
        "nocpu" in ARGS || push!(cases, "seobf2_cpu" => BN.gw_logp_and_grad(data, cfg, freqs; model=m))
        USE_CUDA && push!(cases, "seobf2_gpu" => BN.gw_logp_and_grad_gpu_staged(data, cfg, freqs; model=m, nparams=11, nwalkers=nw))
    end
    for (name, f!) in cases
        ts = timeit(f!, g, l, pos; reps=startswith(name, "seob") ? 3 : parse(Int, get(kv, "reps", "10")))
        t = sum(ts) / length(ts); tmin = minimum(ts)
        @printf("%-12s%-9d%-10.2f%-10.2f%.4f\n", name, nw, 1000t, 1000tmin, 1000tmin / nw)
        (startswith(name, "xas_gpu") || name == "xas_res") && println("    per-call ms: ", round.(1000 .* ts; digits=1))
        if name == "xas_gpu"
            fill!(tim, 0.0); f!(g, l, pos, nothing)
            @printf("    stages ms: (a) coeffs %.2f  (b) sums kernel %.2f  (c) host ll grad %.2f  (d) adjoint kernel %.2f  (e) coeff VJP %.2f\n",
                    (1000 .* tim)...)
        end
        if name == "xas_res_rb"
            fill!(timrb, 0.0); f!(g, l, pos, nothing)
            @printf("    stages ms: (a+e') coeffs+jacobian %.2f  (b) edge eval+sums %.2f  (c) ll kernel %.2f  (d) adjoint %.2f  (e) grad + copy %.2f
",
                    (1000 .* timrb)...)
        end
        if name == "xas_res"
            fill!(timr, 0.0); f!(g, l, pos, nothing)
            @printf("    stages ms: (a+e') coeffs+jacobian %.2f  (b) sums %.2f  (c) ll kernel %.2f  (d) adjoint %.2f  (e) grad + copy %.2f
",
                    (1000 .* timr)...)
        end
        flush(stdout)
    end
end
println("BENCH_DONE")
