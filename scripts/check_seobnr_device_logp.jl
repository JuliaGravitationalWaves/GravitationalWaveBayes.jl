# The device SEOBNRv5 IMR log-likelihood (`seobnr_device_logp!`) vs the CPU
# per-walker likelihood (`_eobimr_walker_ll[_multi]`, the function the CPU gradient differentiates), on real
# GW150914 data: H1+L1 and H1 alone, plain and with all likelihood options; then timing.
#   julia --project=<env> scripts/check_seobnr_device_logp.jl [cuda|cpu] [nwalkers for timing...]
using GravitationalWaveBayes, GravitationalWaveData, KernelAbstractions, Random, Printf
using GravitationalWaveData: H1, L1
const BN = GravitationalWaveBayes.BatchedNUTS
const KF = GravitationalWaveBayes.KernelFused
import GravitationalWaveforms
const MODEL = GravitationalWaveforms.SEOBNRv5IMRAlignedSpinF2()

which = isempty(ARGS) ? "cpu" : ARGS[1]
sizes = length(ARGS) > 1 ? parse.(Int, ARGS[2:end]) : Int[]
if get(ENV, "SDT_NOSTRICT", "0") == "1"
    using Enzyme
    Enzyme.API.strictAliasing!(false)
    println("Enzyme strictAliasing = false")
end
if which == "cuda"
    using CUDA
    # SDT_INLINE=1: compile every device function inline (KernelAbstractions `always_inline`)
    backend = CUDA.CUDABackend(; always_inline = get(ENV, "SDT_INLINE", "0") == "1")
    println("always_inline = ", get(ENV, "SDT_INLINE", "0") == "1")
    println("device: ", CUDA.name(CUDA.device()))
else
    backend = KernelAbstractions.CPU()
    println("KernelAbstractions CPU backend, ", Threads.nthreads(), " threads")
end
flush(stdout)
# SDT_DIRPAR="b,p,n" (0/1 each, default "1,a,0"): forward mode of the base / plunge / node stages with one width-1
# thread per tangent direction (1) or one width-4 thread (0)
# ("a" = automatic: `nothing`; plunge "2": two width-2 threads per walker)
const DPV = map(x -> x == "a" ? nothing : x == "2" ? 2 : x == "1", split(get(ENV, "SDT_DIRPAR", "1,a,0"), ","))
const TKW = (; dirpar = (base = DPV[1], plunge = DPV[2], nodes = DPV[3]), base_ift = get(ENV, "SDT_BASEIFT", "1") == "1",
    plunge_split = get(ENV, "SDT_PSPLIT", "0") == "1")
println("table keywords: ", TKW)

kw = (; trigger_time = 1126259462.4, duration = 4.0, post_trigger = 2.0, psd_segment = 1024.0, f_min = 20.0,
      f_max = 512.0, alpha = 0.2, psd_method = :median)
h1 = build_gwdata("data/H-H1_LOSC_4_V1-1126256640-4096.hdf5"; detector = H1, kw...)
l1 = build_gwdata("data/L-L1_LOSC_4_V1-1126256640-4096.hdf5"; detector = L1, kw...)
freqs = h1.freqs
allopts = (; marginalize_phase = true, marginalize_distance = true, distance_lo = 0.05, distance_hi = 2.0,
           distance_power = 2, time_reference = 1, time_ref_frequency = 35.0, time_ref_offset = 0.305)

function positions(nw; seed = 1)
    rng = Random.Xoshiro(seed)
    pos = repeat([30.5, 0.8, 0.1, -0.1, 0.45, 0.012, 1.0, 1.9, -0.9, -0.6, 1.0], 1, nw) .+
          [0.6, 0.08, 0.15, 0.15, 0.08, 0.002, 0.5, 0.1, 0.05, 0.15, 0.4] .* randn(rng, 11, nw)
    pos[2, :] .= clamp.(pos[2, :], 0.25, 1.0)
    pos[3:4, :] .= clamp.(pos[3:4, :], -0.95, 0.95)
    pos[9:10, :] .= clamp.(pos[9:10, :], -0.99, 0.99)
    pos
end

function cfg_for(nw, opts)
    df = freqs[2] - freqs[1]
    c = PEConfig{Float64}(length(freqs), nw, 1e-3, 1)
    return isempty(opts) ? c : with_likelihood_options(c; opts...)
end

ncheck = 64
pos = positions(ncheck)
for (name, data, opts) in (("H1+L1 plain", [h1, l1], (;)), ("H1+L1 all options", [h1, l1], allopts),
                           ("H1 plain", [h1], (;)), ("H1 all options", [h1], merge(allopts, (; time_reference = 0))))
    cfg = cfg_for(ncheck, opts)
    lik = KF._lik_opts(cfg)
    det = BN._eobimr_det(length(data) == 1 ? data[1] : data, freqs, cfg)
    llf = BN._eobimr_llfun(det)
    ref = [llf(pos[:, w]..., MODEL, det, lik) for w in 1:ncheck]
    L = BN.SEOBNRDeviceLikelihood(backend, data, cfg; nwalkers = ncheck)
    t0 = time()
    got = Array(BN.seobnr_device_logp!(L, pos))
    el = time() - t0
    err = maximum(abs.(got .- ref))
    @printf("%-18s max |dlogL| %.2e (logL range %.1f .. %.1f; first call %.1f s)\n", name, err, minimum(ref),
        maximum(ref), el)
    flush(stdout)
    # gradient vs the production CPU gradient (host Enzyme reverse per walker through the whole build)
    ng = 16
    gref = zeros(11, ng); lref = zeros(ng)
    BN._eobimr_gradient!(gref, pos[:, 1:ng], freqs, cfg, length(data) == 1 ? data[1] : data, MODEL; ll_out = lref)
    Lg = BN.SEOBNRDeviceLikelihood(backend, data, cfg; nwalkers = ng, grad = true)
    t0 = time()
    llg, gg = BN.seobnr_device_logp_and_grad!(Lg, pos[:, 1:ng]; TKW...)
    el = time() - t0
    llg = Array(llg); gg = Array(gg)
    scale = [maximum(abs.(gref[:, w])) for w in 1:ng]
    rows = [maximum(abs.(gg[k, w] - gref[k, w]) / scale[w] for w in 1:ng) for k in 1:11]
    @printf("%-18s gradient: logL %.2e, max rel err on each walker's gradient scale %.2e (first call %.1f s)\n",
        "", maximum(abs.(llg .- lref)), maximum(rows), el)
    @printf("%-18s   per row: %s\n", "", join((@sprintf("%.1e", r) for r in rows), " "))
    flush(stdout)
end

for nw in sizes
    cfg = cfg_for(nw, allopts)
    L = BN.SEOBNRDeviceLikelihood(backend, [h1, l1], cfg; nwalkers = nw, grad = true)
    p = positions(nw; seed = 7)
    BN.seobnr_device_logp!(L, p)
    tm = Dict{Symbol,Float64}()
    ncall = 5
    tt = @elapsed for _ in 1:ncall
        BN.seobnr_device_logp!(L, p; timers = tm)
    end
    t2 = @elapsed for _ in 1:ncall
        BN.seobnr_device_logp!(L, p)
    end
    @printf("\n=== %d walkers, H1+L1 all options: %.2f ms per call = %.5f ms per walker (untimed %.5f)\n", nw,
        1000tt / ncall, 1000tt / ncall / nw, 1000t2 / ncall / nw)
    for k in (:base, :pa, :plunge, :nodes, :build, :spa, :segment, :fft, :fourier, :likelihood)
        @printf("    %-10s %.3f ms per call\n", k, 1000 * get(tm, k, 0.0) / ncall)
    end
    flush(stdout)
    BN.seobnr_device_logp_and_grad!(L, p; TKW...)
    tmg = Dict{Symbol,Float64}()
    ttg = @elapsed for _ in 1:ncall
        BN.seobnr_device_logp_and_grad!(L, p; timers = tmg, TKW...)
    end
    tg2 = Float64[]
    for _ in 1:ncall
        push!(tg2, @elapsed BN.seobnr_device_logp_and_grad!(L, p; TKW...))
    end
    @printf("  logL + gradient: %.2f ms per call = %.5f ms per walker (untimed: min %.5f, last %.5f -- growth check)\n",
        1000ttg / ncall, 1000ttg / ncall / nw, 1000minimum(tg2) / nw, 1000tg2[end] / nw)
    for k in (:base, :pa, :plunge, :nodes, :build, :spa, :segment, :fft, :fourier, :likelihood)
        @printf("    %-10s %.3f ms per call\n", k, 1000 * get(tmg, k, 0.0) / ncall)
    end
    flush(stdout)
end
println("\nDONE")
