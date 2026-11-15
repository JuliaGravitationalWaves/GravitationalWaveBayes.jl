# XAS device-path variants vs the CPU per-walker path (logp and gradient), real GW150914 H1+L1 and H1, plain and all
# likelihood options. `julia --project=<env> scripts/check_xas_variants.jl [cuda] [walkers=8] [variant=fwd] [rb]`
const USE_CUDA = "cuda" in ARGS
USE_CUDA && using CUDA
using KernelAbstractions, GravitationalWaveBayes, Random, Printf
using GravitationalWaveforms: IMRPhenomXAS
using GravitationalWaveData: H1, L1
const BN = GravitationalWaveBayes.BatchedNUTS
kv = Dict(split(a, "=")[1] => split(a, "=")[2] for a in ARGS if occursin("=", a))
nw = parse(Int, get(kv, "walkers", "8"))
variants = Symbol.(split(get(kv, "variant", "fwd"), ","))
backend = USE_CUDA ? CUDABackend() : KernelAbstractions.CPU()
USE_CUDA && GravitationalWaveBayes.GPU.ensure_device_heap(max(64, 2 * nw) << 20)

kw = (; trigger_time=1126259462.4, duration=4.0, post_trigger=2.0, psd_segment=1024.0, f_min=20.0,
      f_max=512.0, alpha=0.2, psd_method=:median)
h1 = build_gwdata("data/H-H1_LOSC_4_V1-1126256640-4096.hdf5"; detector=H1, kw...)
l1 = build_gwdata("data/L-L1_LOSC_4_V1-1126256640-4096.hdf5"; detector=L1, kw...)
freqs = h1.freqs
allopts = (; marginalize_phase=true, marginalize_distance=true, distance_lo=0.05, distance_hi=1.5, distance_power=2,
           time_reference=1, time_ref_frequency=35.0, time_ref_offset=0.12283266110168832)
center = [31.0, 0.85, 0.0, 0.0, 0.45, 0.029, 1.0, 1.9, -0.9, -0.8, 1.0]
spread = [0.5, 0.05, 0.1, 0.1, 0.05, 0.001, 0.3, 0.1, 0.05, 0.1, 0.3]
pos = center .+ spread .* randn(Random.Xoshiro(1), 11, nw)
pos[2, :] .= clamp.(pos[2, :], 0.3, 1.0)
pos[:, end] .= center                               # one walker exactly at the centre
pos[3, 1] = 0.9; pos[4, 1] = -0.9                    # one high-spin walker (different segment boundaries)

worst = 0.0
for (dname, data) in (("H1+L1", [h1, l1]), ("H1", h1)), (oname, opts) in (("plain", (;)), ("all", allopts))
    cfg = with_likelihood_options(PEConfig{Float64}(length(freqs), nw, 1e-3, 1); opts...)
    rb = "rb" in ARGS ? BN.build_xas_relative_binning(data, cfg, center; nbins=300) : nothing
    cpu = rb === nothing ? BN.gw_logp_and_grad(data, cfg, freqs; model=IMRPhenomXAS()) :
                           BN.gw_logp_and_grad(data, cfg, freqs; model=IMRPhenomXAS(), relative_binning=rb)
    gref = zeros(11, nw); lref = zeros(nw)
    cpu(gref, lref, pos, nothing)
    for v in variants
        kwv = v === :fwdj ? (; variant=:fwd, jac=:split) : v === :fwd3 ? (; variant=:fwd, jac=:split3) : v === :fwd5 ? (; variant=:fwd, jac=:split5) : (; variant=v)
        f! = BN.xas_binparallel_logp_and_grad(data, cfg, freqs; backend=backend, nwalkers=nw, resident=true,
                                              relative_binning=rb, kwv...)
        g = zeros(11, nw); l = zeros(nw)
        t = @elapsed f!(g, l, pos, nothing)
        f!(g, l, pos, nothing)
        dl = maximum(ifelse.(l .== lref, 0.0, abs.(l .- lref) ./ max.(abs.(lref), 1)))
        any(isinf, lref) && println("    (", count(isinf, lref), " walker(s) outside the prior box, logp -Inf on both paths: ", isinf.(l) == isinf.(lref), ")")
        # marginalized rows are exactly flat: compare each walker's gradient on its overall scale
        dg = maximum(abs.(g .- gref) ./ max.(maximum(abs.(gref); dims=1), 1e-300))
        global worst = max(worst, dl, dg)
        @printf("%-6s %-6s %-8s first %.1f s   logp rel %.2e   grad rel %.2e\n", dname, oname, string(v), t, dl, dg)
        flush(stdout)
    end
end
println(worst < 1e-8 ? "CHECK_OK" : "CHECK_FAIL", "  worst ", worst)
