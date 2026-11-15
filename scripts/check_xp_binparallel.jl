# IMRPhenomXP / XPHM bin-parallel likelihood + gradient (`xp[hm]_binparallel_logp_and_grad`): exactness on a backend
# against the CPU per-walker dense likelihood (value) and central finite differences of it (gradient), then timings on
# real GW150914 H1+L1 with all likelihood options.
#   julia -t 15 --project=<env with CUDA> scripts/check_xp_binparallel.jl [cuda] [model=XP|XPHM|XPHM102] [walkers=64,256,1024,4096] [setup=host|device] [data=matched]
# model=XPHM is PrecVersion 223 (MSA, the default), XPHM102 the NNLO angles; data=matched: the GWTC-2.1 inputs (release PSDs,
# 16 kHz strain resampled to 2048 Hz, 20-896 Hz, trigger .391; see data/README.md).
using GravitationalWaveBayes, KernelAbstractions, Random, Printf
using GravitationalWaveforms: IMRPhenomXP, IMRPhenomXPHM
using GravitationalWaveData: H1, L1
const BN = GravitationalWaveBayes.BatchedNUTS
usecuda = "cuda" in ARGS
if usecuda
    using CUDA
    backend = CUDABackend()
else
    backend = KernelAbstractions.CPU()
end
kv = Dict(split(a, "=")[1] => split(a, "=")[2] for a in ARGS if occursin("=", a))
setup = Symbol(get(kv, "setup", "host"))
const XP = Dict("XP" => IMRPhenomXP(), "XPHM" => IMRPhenomXPHM(), "XPHM102" => IMRPhenomXPHM(102))[get(kv, "model", "XP")]
const BPF0 = XP isa IMRPhenomXPHM ? BN.xphm_binparallel_logp_and_grad : BN.xp_binparallel_logp_and_grad
# cal=1 (XPHM, data=matched): the GWTC-2.1 spline calibration (data/gwtc21_gw150914/priors.toml), 40 extra rows
const CAL = get(kv, "cal", "0") == "1"
BPF(a...; k...) = XP isa IMRPhenomXPHM ? BPF0(a...; model=XP, k..., (CAL ? (; calibration=CALM) : (;))...) : BPF0(a...; k...)
println("model ", typeof(XP))
if get(kv, "data", "") == "matched"
    kw = (; trigger_time=1126259462.391, duration=4.0, post_trigger=2.0, f_min=20.0, f_max=896.0, alpha=0.2)
    data = [build_gwdata("data/$(d)_GWOSC_16KHZ_R1-1126259447-32_lal2048.hdf5"; detector=det,
                         psd_file="data/gwtc21_gw150914/psd_$(det.prefix).txt", kw...) for (d, det) in (("H-H1", H1), ("L-L1", L1))]
else
    kw = (; trigger_time=1126259462.4, duration=4.0, post_trigger=2.0, psd_segment=1024.0, f_min=20.0,
          f_max=512.0, alpha=0.2, psd_method=:median)
    data = [build_gwdata("data/H-H1_LOSC_4_V1-1126256640-4096.hdf5"; detector=H1, kw...),
            build_gwdata("data/L-L1_LOSC_4_V1-1126256640-4096.hdf5"; detector=L1, kw...)]
end
freqs = data[1].freqs
const CALM = CAL ? calibration_prior("data/gwtc21_gw150914/priors.toml", ["H1", "L1"], freqs) : nothing
const NCAL = CAL ? length(CALM.mu) : 0
opts = (; marginalize_distance=true, distance_lo=0.05, distance_hi=1.5, distance_power=2,
        time_reference=1, time_ref_frequency=35.0, time_ref_offset=0.12283266110168832)
center = [31.0, 0.85, 0.1, -0.2, 0.45, 0.029, 1.0, 1.9, -0.9, -0.8, 1.0, 0.2, -0.1, 0.1, 0.15]
spread = [0.5, 0.05, 0.1, 0.1, 0.05, 0.001, 0.3, 0.1, 0.05, 0.1, 0.3, 0.15, 0.15, 0.15, 0.15]
mkpos(n) = (p = center .+ spread .* randn(Random.Xoshiro(n), 15, n); p[2, :] .= clamp.(p[2, :], 0.3, 1.0);
            vcat(p, randn(Random.Xoshiro(n + 1), NCAL, n)))
mkcfg(n) = with_likelihood_options(PEConfig{Float64}(length(freqs), n, 1e-3, 1); opts...)
# CPU reference; with calibration, per walker on the calibrated-equivalent data (strain d / C, PSD S / |C|^2)
function llfun(p, cfg)
    p15 = p[1:15, :]
    ph = similar(p15); BN._reparam_to_physical!(ph, p15); o = zeros(size(p, 2))
    if !CAL
        BN._eval_ll!(o, ph, freqs, cfg, data, XP, nothing)
        return o
    end
    nn = CALM.nn
    for w in axes(p, 2)
        node = CALM.mu .+ CALM.sigma .* p[16:end, w]
        dm = [begin
                  o2 = (d - 1) * 2nn
                  C = calibration_factor(CALM.W, node[o2+1:o2+nn], node[o2+nn+1:o2+2nn])
                  LikelihoodData(dd.freqs, dd.strain ./ C, dd.psd ./ abs2.(C), dd.detector, dd.gps_time)
              end for (d, dd) in enumerate(data)]
        c1 = with_likelihood_options(PEConfig{Float64}(length(freqs), 1, 1e-3, 1); opts...)
        oo = zeros(1); BN._eval_ll!(oo, ph[:, w:w], freqs, c1, dm, XP, nothing); o[w] = oo[1]
    end
    return o
end

# exactness on 4 walkers (the 4th with zero in-plane spins: chi_p = 0, the NUTS initialization)
n = 4; pos = mkpos(n); pos[12:15, 4] .= 0.0; cfg = mkcfg(n)
t0 = time()
f! = BPF(data, cfg, freqs; backend=backend, nwalkers=n, setup=setup)
g = zeros(15 + NCAL, n); l = zeros(n)
f!(g, l, pos, nothing)
@printf("first call %.1f s (%s)\n", time() - t0, usecuda ? "CUDA" : "CPU")
lld = llfun(pos, cfg)
inbox = isfinite.(BN._call_log_prior_vec(pos[1:15, :]))     # (outside the prior box both sides are -Inf)
@printf("logp max rel diff vs dense CPU: %.2e (%d of %d walkers inside the box)\n",
        maximum(abs.(l[inbox] .- lld[inbox] .- BN._call_log_prior_vec(pos[1:15, :])[inbox]) ./ abs.(lld[inbox])), count(inbox), n)
println("all gradients finite: ", all(isfinite, g))
worst = 0.0
for r in vcat(1:15, CAL ? collect(16:5:15 + NCAL) : Int[])
    ws = r >= 12 ? (1:3) : (1:4)        # chi_p = |S_perp| has a kink at zero in-plane spin: no FD there
    # Per walker, the smallest error over several steps: one fixed small step is round-off on a small gradient (0.45 on
    # the zero in-plane spin walker at h = 1e-6, where h = 1e-4 agrees to 3e-3), and truncation on a sharp one.
    E = fill(Inf, n); fdbest = zeros(n)
    for h in (r == 6 ? (1e-7, 1e-8, 1e-9) : (1e-3, 1e-4, 1e-5, 1e-6))
        pp = copy(pos); pp[r, :] .+= h; pm = copy(pos); pm[r, :] .-= h
        fd = (llfun(pp, cfg) .- llfun(pm, cfg)) ./ (2h)
        floor_ = max(1e-2 * maximum(abs.(fd[ws])), 1e-3)
        for w in ws
            ew = abs(g[r, w] - fd[w]) / max(abs(fd[w]), floor_)
            ew < E[w] && (E[w] = ew; fdbest[w] = fd[w])
        end
    end
    e = maximum(E[ws])
    global worst = max(worst, e)
    @printf("row %2d  max rel err vs FD %.2e\n", r, e)
    e > 1e-3 && println("    grad ", round.(g[r, :]; sigdigits=6), "\n    fd   ", round.(fdbest; sigdigits=6))
end
@printf("worst gradient rel err %.2e\n", worst)

for nw in parse.(Int, split(get(kv, "walkers", "64,256,1024"), ","))
    p = mkpos(nw); c = mkcfg(nw)
    fb! = BPF(data, c, freqs; backend=backend, nwalkers=nw, setup=setup)
    gg = zeros(15 + NCAL, nw); ll = zeros(nw)
    fb!(gg, ll, p, nothing)
    ts = [(@elapsed fb!(gg, ll, p, nothing)) for _ in 1:parse(Int, get(ENV, "REPS", "10"))]
    @printf("walkers %5d  min %.2f ms  median %.2f ms  (%.2f us/walker)\n", nw, 1000minimum(ts),
            1000sort(ts)[cld(length(ts), 2)], 1e6minimum(ts) / nw)
    tm = zeros(4)
    ft! = BPF(data, c, freqs; backend=backend, nwalkers=nw, setup=setup, timers=tm)
    ft!(gg, ll, p, nothing); fill!(tm, 0.0)
    for _ in 1:5; ft!(gg, ll, p, nothing); end
    @printf("   per call: setup %.2f  ext %.2f  sums %.2f  surrogate %.2f ms\n", (200 .* tm)...)
end
println("CHECK_DONE")
