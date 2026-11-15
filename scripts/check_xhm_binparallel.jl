# IMRPhenomXHM bin-parallel likelihood + gradient (`xhm_binparallel_logp_and_grad`, 2026-10-04): exactness on a backend
# against the CPU per-walker dense likelihood (value) and central finite differences of it (gradient), then timings on
# real GW150914 H1+L1 with all likelihood options.
#   julia -t 15 --project=<env with CUDA> scripts/check_xhm_binparallel.jl [cuda] [walkers=64,256,1024,4096] [setup=host|device]
using GravitationalWaveBayes, KernelAbstractions, Random, Printf
using GravitationalWaveforms: IMRPhenomXHM
using GravitationalWaveData: H1, L1
const BN = GravitationalWaveBayes.BatchedNUTS
const XHM = IMRPhenomXHM()
usecuda = "cuda" in ARGS
if usecuda
    using CUDA
    backend = CUDABackend()
else
    backend = KernelAbstractions.CPU()
end
kv = Dict(split(a, "=")[1] => split(a, "=")[2] for a in ARGS if occursin("=", a))
setup = Symbol(get(kv, "setup", "host"))
kw = (; trigger_time=1126259462.4, duration=4.0, post_trigger=2.0, psd_segment=1024.0, f_min=20.0,
      f_max=512.0, alpha=0.2, psd_method=:median)
data = [build_gwdata("data/H-H1_LOSC_4_V1-1126256640-4096.hdf5"; detector=H1, kw...),
        build_gwdata("data/L-L1_LOSC_4_V1-1126256640-4096.hdf5"; detector=L1, kw...)]
freqs = data[1].freqs
opts = (; marginalize_distance=true, distance_lo=0.05, distance_hi=1.5, distance_power=2,
        time_reference=1, time_ref_frequency=35.0, time_ref_offset=0.12283266110168832)
center = [31.0, 0.85, 0.1, -0.2, 0.45, 0.029, 1.0, 1.9, -0.9, -0.8, 1.0]
spread = [0.5, 0.05, 0.1, 0.1, 0.05, 0.001, 0.3, 0.1, 0.05, 0.1, 0.3]
mkpos(n) = (p = center .+ spread .* randn(Random.Xoshiro(n), 11, n); p[2, :] .= clamp.(p[2, :], 0.3, 1.0); p)
mkcfg(n) = with_likelihood_options(PEConfig{Float64}(length(freqs), n, 1e-3, 1); opts...)
llfun(p, cfg) = (ph = similar(p); BN._reparam_to_physical!(ph, p); o = zeros(size(p, 2));
                 BN._eval_ll!(o, ph, freqs, cfg, data, XHM, nothing); o)

# exactness on 4 walkers
n = 4; pos = mkpos(n); cfg = mkcfg(n)
t0 = time()
f! = BN.xhm_binparallel_logp_and_grad(data, cfg, freqs; backend=backend, nwalkers=n, setup=setup)
g = zeros(11, n); l = zeros(n)
f!(g, l, pos, nothing)
@printf("first call %.1f s (%s)\n", time() - t0, usecuda ? "CUDA" : "CPU")
lld = llfun(pos, cfg)
inbox = isfinite.(BN._call_log_prior_vec(pos))     # (outside the prior box both sides are -Inf)
@printf("logp max rel diff vs dense CPU: %.2e (%d of %d walkers inside the box)\n",
        maximum(abs.(l[inbox] .- lld[inbox] .- BN._call_log_prior_vec(pos)[inbox]) ./ abs.(lld[inbox])), count(inbox), n)
worst = 0.0
for r in 1:11
    h = r == 6 ? 1e-8 : 1e-6
    pp = copy(pos); pp[r, :] .+= h; pm = copy(pos); pm[r, :] .-= h
    fd = (llfun(pp, cfg) .- llfun(pm, cfg)) ./ (2h)
    e = maximum(abs.(g[r, :] .- fd) ./ max.(abs.(fd), 1e-2 * maximum(abs.(fd)), 1e-3))
    global worst = max(worst, e)
    @printf("row %2d  max rel err vs FD %.2e\n", r, e)
end
@printf("worst gradient rel err %.2e\n", worst)

for nw in parse.(Int, split(get(kv, "walkers", "64,256,1024"), ","))
    p = mkpos(nw); c = mkcfg(nw)
    fb! = BN.xhm_binparallel_logp_and_grad(data, c, freqs; backend=backend, nwalkers=nw, setup=setup)
    gg = zeros(11, nw); ll = zeros(nw)
    fb!(gg, ll, p, nothing)
    ts = [(@elapsed fb!(gg, ll, p, nothing)) for _ in 1:parse(Int, get(ENV, "REPS", "10"))]
    @printf("walkers %5d  min %.2f ms  median %.2f ms  (%.2f us/walker)\n", nw, 1000minimum(ts),
            1000sort(ts)[cld(length(ts), 2)], 1e6minimum(ts) / nw)
    tm = zeros(4)
    ft! = BN.xhm_binparallel_logp_and_grad(data, c, freqs; backend=backend, nwalkers=nw, setup=setup, timers=tm)
    ft!(gg, ll, p, nothing); fill!(tm, 0.0)
    for _ in 1:5; ft!(gg, ll, p, nothing); end
    @printf("   per call: setup %.2f  ext %.2f  sums %.2f  surrogate %.2f ms\n", (200 .* tm)...)
end
println("CHECK_DONE")
