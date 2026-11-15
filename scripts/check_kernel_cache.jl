# GPU kernel cache check (GravitationalWaveBayes.KernelCache): cold-start breakdown of the SEOBNRv5 (2,2) device likelihood and the
# SEOBNRv5HM bin-parallel likelihood on CUDA, and a bit-level dump of logL / gradient for fixed walkers, compared
# against a reference dump from another process.
#   GRAVITATIONALWAVEBAYES_KERNEL_CACHE=<dir> [GRAVITATIONALWAVEBAYES_KERNEL_CACHE_LOG=1] julia --project=<env> scripts/check_kernel_cache.jl \
#       <out.jls> [ref=<reference.jls>] [models=seobnr,hm]
t_start = time()
Base.cumulative_compile_timing(true)
ctime() = Base.cumulative_compile_time_ns()[1] / 1e9   # host-side Julia compile time so far
using GravitationalWaveBayes, GravitationalWaveData, KernelAbstractions, CUDA, Random, Printf, Serialization
using GravitationalWaveData: H1, L1
import GravitationalWaveforms
t_load = time() - t_start
c_load = ctime()
const BN = GravitationalWaveBayes.BatchedNUTS

out = ARGS[1]
kv = Dict(split(a, "=")[1] => split(a, "=")[2] for a in ARGS[2:end] if occursin("=", a))
ref = get(kv, "ref", nothing)
models = split(get(kv, "models", "seobnr,hm"), ",")
CUDA.device()
@printf("package load %.1f s (host compile %.1f s), CUDA init %.1f s\n", t_load, c_load, time() - t_start - t_load)

kw = (; trigger_time = 1126259462.4, duration = 4.0, post_trigger = 2.0, psd_segment = 1024.0, f_min = 20.0,
      f_max = 512.0, alpha = 0.2, psd_method = :median)
h1 = build_gwdata("data/H-H1_LOSC_4_V1-1126256640-4096.hdf5"; detector = H1, kw...)
l1 = build_gwdata("data/L-L1_LOSC_4_V1-1126256640-4096.hdf5"; detector = L1, kw...)
freqs = h1.freqs
df = freqs[2] - freqs[1]
backend = CUDA.CUDABackend()
results = Dict{String,Any}()

function timed(f, label)
    t0 = time(); c0 = ctime()
    r = f()
    CUDA.synchronize()
    @printf("  %-34s %8.1f s (host compile %.1f s)\n", label, time() - t0, ctime() - c0)
    return r
end

if "seobnr" in models
    println("SEOBNRv5 (2,2) device likelihood")
    allopts = (; marginalize_phase = true, marginalize_distance = true, distance_lo = 0.05, distance_hi = 2.0,
               distance_power = 2, time_reference = 1, time_ref_frequency = 35.0, time_ref_offset = 0.305)
    nw = 64
    rng = Random.Xoshiro(1)
    pos = repeat([30.5, 0.8, 0.1, -0.1, 0.45, 0.012, 1.0, 1.9, -0.9, -0.6, 1.0], 1, nw) .+
          [0.6, 0.08, 0.15, 0.15, 0.08, 0.002, 0.5, 0.1, 0.05, 0.15, 0.4] .* randn(rng, 11, nw)
    pos[2, :] .= clamp.(pos[2, :], 0.25, 1.0)
    pos[3:4, :] .= clamp.(pos[3:4, :], -0.95, 0.95)
    pos[9:10, :] .= clamp.(pos[9:10, :], -0.99, 0.99)
    cfg = with_likelihood_options(PEConfig{Float64}(length(freqs), nw, 1e-3, 1); allopts...)
    L = timed("likelihood setup") do
        BN.SEOBNRDeviceLikelihood(backend, [h1, l1], cfg; nwalkers = nw, grad = true)
    end
    results["seobnr_ll"] = timed(() -> Array(BN.seobnr_device_logp!(L, pos)), "first logL")
    TKW = (; dirpar = (base = true, plunge = nothing, nodes = false), base_ift = true, plunge_split = false)
    llg, gg = timed(() -> BN.seobnr_device_logp_and_grad!(L, pos; TKW...), "first logL+grad")
    results["seobnr_llg"] = Array(llg); results["seobnr_gg"] = Array(gg)
    timed(() -> BN.seobnr_device_logp_and_grad!(L, pos; TKW...), "second logL+grad")
    timed(() -> BN.seobnr_device_logp_and_grad!(L, pos; TKW...), "third logL+grad")
    timed(() -> BN.seobnr_device_logp_and_grad!(L, pos; TKW...), "fourth logL+grad")
end

if "hm" in models
    println("SEOBNRv5HM bin-parallel likelihood")
    opts = (; marginalize_distance = true, distance_lo = 0.05, distance_hi = 1.5, distance_power = 2,
            time_reference = 1, time_ref_frequency = 35.0, time_ref_offset = 0.12283266110168832)
    n = 8
    center = [31.0, 0.85, 0.1, -0.2, 0.45, 0.029, 1.0, 1.9, -0.9, -0.8, 1.0]
    spread = [0.5, 0.05, 0.1, 0.1, 0.05, 0.001, 0.3, 0.1, 0.05, 0.1, 0.3]
    pos = center .+ spread .* randn(Random.Xoshiro(n), 11, n); pos[2, :] .= clamp.(pos[2, :], 0.3, 1.0)
    cfg = with_likelihood_options(PEConfig{Float64}(length(freqs), n, 1e-3, 1); opts...)
    f! = timed(() -> BN.seobnrhm_binparallel_logp_and_grad([h1, l1], cfg, freqs; backend = backend, nwalkers = n),
        "likelihood setup")
    gr = zeros(11, n); l = zeros(n)
    timed(() -> f!(gr, l, pos, nothing), "first logL+grad")
    results["hm_ll"] = copy(l); results["hm_gg"] = copy(gr)
    timed(() -> f!(gr, l, pos, nothing), "second logL+grad")
    timed(() -> f!(gr, l, pos, nothing), "third logL+grad")
    timed(() -> f!(gr, l, pos, nothing), "fourth logL+grad")
end

ext = Base.get_extension(GravitationalWaveBayes, :GravitationalWaveBayesCUDAExt)
println("kernel cache stats: ", ext.kernel_cache_stats())
@printf("total script time %.1f s\n", time() - t_start)

serialize(out, results)
if ref !== nothing
    r = deserialize(ref)
    bits(x) = reinterpret(UInt64, vec(x))
    for k in sort!(collect(keys(results)))
        haskey(r, k) || continue
        a, b = results[k], r[k]
        same = size(a) == size(b) && bits(a) == bits(b)
        @printf("%-12s bit-identical to reference: %s (max |diff| %.3e)\n", k, same, maximum(abs.(a .- b)))
    end
end
println("DONE")
