# Bin-parallel IMRPhenomXAS likelihood + gradient (BatchedNUTS.xas_binparallel_logp_and_grad) vs the CPU
# per-walker path (gw_logp_and_grad), real GW150914 H1+L1 and H1 only, plain and all likelihood options.
# `julia --project=<env> -t N scripts/check_xas_binparallel.jl [cuda] [nwalkers] [chunk] [resident] [hand] [unshared]`
# `rb`: relative binning (300 bins around the walkers' centre) on the device vs the CPU relative-binning path.
# `resident`: device-resident per-walker stages; also times 30 repeated calls (a device-heap leak shows as growth).
const USE_CUDA = "cuda" in ARGS
USE_CUDA && using CUDA
using KernelAbstractions, GravitationalWaveBayes, Random, Test
using GravitationalWaveforms: IMRPhenomXAS
using GravitationalWaveData: H1, L1
const BN = GravitationalWaveBayes.BatchedNUTS
backend = USE_CUDA ? CUDABackend() : KernelAbstractions.CPU()
nums = [parse(Int, a) for a in ARGS if all(isdigit, a)]
nw = length(nums) >= 1 ? nums[1] : 64
chunk = length(nums) >= 2 ? nums[2] : 16
USE_CUDA && println("GPU: ", CUDA.name(CUDA.device()))
println("threads ", Threads.nthreads(), ", walkers ", nw, ", chunk ", chunk)

kw = (; trigger_time=1126259462.4, duration=4.0, post_trigger=2.0, psd_segment=1024.0, f_min=20.0,
      f_max=512.0, alpha=0.2, psd_method=:median)
h1 = build_gwdata("data/H-H1_LOSC_4_V1-1126256640-4096.hdf5"; detector=H1, kw...)
l1 = build_gwdata("data/L-L1_LOSC_4_V1-1126256640-4096.hdf5"; detector=L1, kw...)
freqs = h1.freqs
pos = repeat([30.0, 0.8, 0.1, -0.1, 0.5, 0.01, 1.0, 1.9, -0.9, -0.6, 1.0], 1, nw) .+
      [0.3, 0.05, 0.1, 0.1, 0.05, 0.002, 0.3, 0.1, 0.05, 0.1, 0.3] .* randn(Random.Xoshiro(1), 11, nw)
pos[2, :] .= clamp.(pos[2, :], 0.2, 1.0)
model = IMRPhenomXAS()
sync() = KernelAbstractions.synchronize(backend)

for (dlabel, data) in (("H1+L1", [h1, l1]), ("H1", h1)),
    (label, opts) in (("plain", (;)),
                      ("all options", (; marginalize_phase=true, marginalize_distance=true, distance_lo=0.05,
                                        distance_hi=2.0, distance_power=2, time_reference=1,
                                        time_ref_frequency=35.0, time_ref_offset=0.305)))
    dlabel == "H1" && get(opts, :time_reference, 0) == 1 && (opts = merge(opts, (; time_reference=0)))
    cfg = with_likelihood_options(PEConfig{Float64}(length(freqs), nw, 1e-3, 1); opts...)
    g1 = zeros(11, nw); l1v = zeros(nw)
    rb = "rb" in ARGS ? BN.build_xas_relative_binning(data, cfg, [30.0, 0.8, 0.1, -0.1, 0.5, 0.01, 1.0, 1.9, -0.9, -0.6, 1.0];
                                                     nbins=300) : nothing
    ref! = rb === nothing ? BN.gw_logp_and_grad(data, cfg, freqs; model=model) :
           BN.gw_logp_and_grad(data, cfg, freqs; model=model, relative_binning=rb)
    ref!(g1, l1v, pos, nothing)
    tref = @elapsed for _ in 1:3; ref!(g1, l1v, pos, nothing); end
    g2 = zeros(11, nw); l2v = zeros(nw)
    bp! = BN.xas_binparallel_logp_and_grad(data, cfg, freqs; backend=backend, nwalkers=nw, chunk=chunk,
                                           shared=!("unshared" in ARGS), adjoint="hand" in ARGS ? :hand : :enzyme,
                                           resident="resident" in ARGS, relative_binning=rb)
    t1 = @elapsed (bp!(g2, l2v, pos, nothing); sync())
    t2 = @elapsed for _ in 1:3; bp!(g2, l2v, pos, nothing); sync(); end
    ok = isfinite.(l1v)
    dl = maximum(abs.(l1v[ok] .- l2v[ok]))
    # relative to the gradient's overall scale: marginalized rows (phase, dL) are exactly flat, so their
    # entries are rounding noise in both paths
    dg = maximum(abs.(g1[:, ok] .- g2[:, ok])) / maximum(abs.(g1[:, ok]))
    println(rpad("$dlabel $label", 22), " max|dlogp|=", round(dl; sigdigits=3), " max rel|dgrad|=", round(dg; sigdigits=3),
            "  bin-parallel: first ", round(t1; digits=1), " s, then ", round(1000t2 / 3; digits=1),
            " ms/call; CPU per-walker ", round(1000tref / 3; digits=1), " ms/call")
    if "resident" in ARGS
        ts = [1000 * @elapsed((bp!(g2, l2v, pos, nothing); sync())) for _ in 1:30]
        println("    30 calls ms: first 5 ", round.(ts[1:5]; digits=1), " last 5 ", round.(ts[26:30]; digits=1),
                "  min ", round(minimum(ts); digits=2))
    end
    @test isapprox(l1v, l2v; rtol=1e-9)
    @test dg < 1e-6
end
println("XAS_BINPARALLEL_DONE")
