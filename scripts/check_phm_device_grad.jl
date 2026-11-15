# SEOBNRv5PHM device log-likelihood gradient (`BatchedNUTS.seobnr_phm_device_logl_grad!`, stagewise reverse) end to
# end: the gradient along the 15 physical rows against central finite differences of the device log-likelihood
# itself (same FFT length), zero-noise H1+L1 injection, walkers off it; then
# timings at W walkers.
#   julia -t 1 --project=<env> scripts/check_phm_device_grad.jl cpu|cuda [W=2] [bench=64,1024] [fref=35] [fphi=50]
using GravitationalWaveBayes, GravitationalWaveData, Printf, Random, KernelAbstractions
import GravitationalWaveforms
const KF = GravitationalWaveBayes.KernelFused
const BN = GravitationalWaveBayes.BatchedNUTS
dev = ARGS[1]
kv = Dict(split(a, "=")[1] => split(a, "=")[2] for a in ARGS[2:end] if occursin("=", a))
if dev == "cuda"
    using CUDA
    be = CUDA.CUDABackend(); toarr(x) = CuArray(x)
    # heap=<MB>: the device malloc heap (Enzyme's reverse tapes for loops of runtime length come from it)
    haskey(kv, "heap") && CUDA.limit!(CUDA.LIMIT_MALLOC_HEAP_SIZE, parse(Int, kv["heap"]) * 2^20)
    # fixheap=1: keep that heap (the driver would raise it to 1 MB per walker)
    get(kv, "fixheap", "0") == "1" && (GravitationalWaveBayes.GPU._ensure_device_heap[] = b -> nothing)
    @printf("device malloc heap %.0f MB
", CUDA.limit(CUDA.LIMIT_MALLOC_HEAP_SIZE) / 2^20)
else
    be = KernelAbstractions.CPU(); toarr(x) = copy(x)
end

psd_fn(f) = (x = f / 215; 1e-49 * (x^(-4.14) - 5 * x^(-2) + 111 * (1 - x^2 + 0.5 * x^4) / (1 + 0.5 * x^2)))
freqs = collect(20.0:0.25:512.0)
gps = 1126259462.4
gm = GravitationalWaveData.gmst(gps)
inj = [36.0, 29.0, 0.4, 0.2, 0.5, 0.01, 0.4, 1.95, -1.27, 0.7, 0.8, 0.3, 0.2, -0.4, 0.1]
model = GravitationalWaveforms.SEOBNRv5PHM()
hp, hc = GravitationalWaveforms.hphc(model, freqs, inj[1], inj[2], (inj[12], inj[13], inj[3]), (inj[14], inj[15], inj[4]), inj[5],
    inj[10], inj[7]; f_min = 0.9 * freqs[1], f_ref = freqs[1])
dets = map((GravitationalWaveData.H1, GravitationalWaveData.L1)) do det
    Fp, Fc = GravitationalWaveData.antenna_response_gmst(det, inj[8], inj[9], inj[11], gm)
    dt = GravitationalWaveData.time_delay_from_geocenter_gmst(det, inj[8], inj[9], gm)
    strain = [(Fp * hp[i] + Fc * hc[i]) * cis(-2pi * freqs[i] * (inj[6] + dt)) for i in eachindex(freqs)]
    GravitationalWaveBayes.LikelihoodData(toarr(freqs), toarr(strain), toarr(psd_fn.(freqs)), det, gps)
end |> collect
fref_ = parse(Float64, get(kv, "fref", "0"))         # time_ref_frequency (Hz; offset 0.17 s when on)
fphi_ = parse(Float64, get(kv, "fphi", "0"))         # phase_ref_frequency (Hz)
cfg = GravitationalWaveBayes.PEConfig{Float64}(length(freqs), 3, 1e-3, 1; time_ref_frequency=fref_,
                                time_ref_offset=fref_ > 0 ? 0.17 : 0.0, phase_ref_frequency=fphi_)

function walkers(W; seed = 3)
    rng = MersenneTwister(seed)
    P = repeat(inj, 1, W)
    for w in 1:W
        P[1, w] *= 1 + 0.003 * randn(rng); P[2, w] *= 1 + 0.003 * randn(rng)
        P[3, w] += 0.02 * randn(rng); P[4, w] += 0.02 * randn(rng)
        P[6, w] += 2e-4 * randn(rng); P[7, w] += 0.05 * randn(rng); P[10, w] += 0.05 * randn(rng)
        P[8, w] += 0.01 * randn(rng); P[9, w] += 0.01 * randn(rng); P[11, w] += 0.05 * randn(rng)
        for k in 12:15
            P[k, w] += 0.02 * randn(rng)
        end
    end
    return P
end

W = parse(Int, get(kv, "W", "2"))
Ph = walkers(W; seed = parse(Int, get(kv, "seed", "3")))
if get(kv, "gradonly", "0") == "1"           # one gradient call at W walkers, stage timers (a sync per stage), no FD
    N = KF._phm_nfft(freqs, 0.9 * minimum(Ph[1, :] .+ Ph[2, :]))
    ll = toarr(zeros(W)); G = toarr(zeros(15, W)); tm = Dict{Symbol,Float64}()
    Pd = toarr(Ph)
    t1 = @elapsed BN.seobnr_phm_device_logl_grad!(ll, G, Pd, cfg, dets; backend = be, nfft = N)
    t = @elapsed BN.seobnr_phm_device_logl_grad!(ll, G, Pd, cfg, dets; backend = be, nfft = N, timers = tm)
    tp = @elapsed KF.seobnr_phm_device_loglikelihood!(ll, Pd, cfg, dets; backend = be, nfft = N)
    @printf("gradonly W = %d: first %.1f s, second %.3f s (%.3f ms/walker), logL alone %.3f s; finite grads %d/%d
", W, t1, t,
        1e3 * t / W, tp, count(isfinite, Array(G)), 15W)
    println("   ", join([@sprintf("%s %.3f", k, v) for (k, v) in sort(collect(tm), by = x -> -x[2])], ", "))
    exit(0)
end
N = KF._phm_nfft(freqs, 0.9 * minimum(Ph[1, :] .+ Ph[2, :]))
ll = toarr(zeros(W)); G = toarr(zeros(15, W))
t = @elapsed BN.seobnr_phm_device_logl_grad!(ll, G, toarr(Ph), cfg, dets; backend = be, nfft = N)
@printf("first call (compile): %.1f s; logL %s\n", t, string(Array(ll)))
Gh = Array(G)
llf = toarr(zeros(W))
L(P) = (KF.seobnr_phm_device_loglikelihood!(llf, toarr(P), cfg, dets; backend = be, nfft = N); Array(llf))
names = ("m1", "m2", "chi1z", "chi2z", "dL", "t_c", "phase", "ra", "dec", "incl", "psi", "chi1x", "chi1y", "chi2x", "chi2y")
worst = 0.0
for k in 1:15
    errs = Float64[]
    for h in (1e-3, 1e-4, 1e-5, 1e-6)
        hk = h * (k == 6 ? 1e-2 : max(abs(inj[k]), 0.1))
        Pp = copy(Ph); Pp[k, :] .+= hk; Pm = copy(Ph); Pm[k, :] .-= hk
        fd = (L(Pp) .- L(Pm)) ./ (2hk)
        push!(errs, maximum(abs.(fd .- Gh[k, :]) ./ max.(abs.(Gh[k, :]), 1e-3 * maximum(abs.(Gh)))))
    end
    global worst = max(worst, minimum(errs))
    @printf("  %-6s grad %s   rel err by step %s\n", names[k], string(round.(Gh[k, :], sigdigits = 6)), string(round.(errs, sigdigits = 2)))
end
@printf("worst best-step rel err %.2e\n", worst)

for Wb in parse.(Int, split(get(kv, "bench", ""), ","; keepempty = false))
    Pb = walkers(Wb; seed = 7)
    llb = toarr(zeros(Wb)); Gb = toarr(zeros(15, Wb)); Pd = toarr(Pb)
    BN.seobnr_phm_device_logl_grad!(llb, Gb, Pd, cfg, dets; backend = be, nfft = N)
    tm = Dict{Symbol,Float64}()
    t = @elapsed BN.seobnr_phm_device_logl_grad!(llb, Gb, Pd, cfg, dets; backend = be, nfft = N, timers = tm)
    tp = @elapsed KF.seobnr_phm_device_loglikelihood!(llb, Pd, cfg, dets; backend = be, nfft = N)
    @printf("W = %d: logL+grad %.3f s (%.3f ms/walker; logL alone %.3f s); finite grads %d/%d\n", Wb, t, 1e3 * t / Wb, tp,
        count(isfinite, Array(Gb)), 15 * Wb)
    println("   ", join([@sprintf("%s %.3f", k, v) for (k, v) in sort(collect(tm), by = x -> -x[2])], ", "))
end
