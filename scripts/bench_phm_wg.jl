# SEOBNRv5PHM device logL+grad: walker-kernel workgroup size (`GRAVITATIONALWAVEBAYES_PHM_WG`) against the batch size (V100).
#   julia -t 1 --project=<env with CUDA> scripts/bench_phm_wg.jl [W=256,1024,4096] [wg=32,16,8,4,2,1]
lines = readlines(joinpath(@__DIR__, "check_phm_sample_rev.jl"))
stop = findfirst(l -> startswith(l, "function check_samples"), lines)
args = copy(ARGS); empty!(ARGS); push!(ARGS, "cuda", "W=2")
include_string(Main, join(lines[1:stop-1], "\n"))
kvb = Dict(split(a, "=")[1] => split(a, "=")[2] for a in args if occursin("=", a))
for Wb in parse.(Int, split(get(kvb, "W", "256,1024,4096"), ","))
    Pb = toarr(walkers(Wb; seed = 5))
    Nb = KF._phm_nfft(freqs, 0.9 * minimum(Array(Pb)[1, :] .+ Array(Pb)[2, :]))
    llb = toarr(zeros(Wb)); Gb = toarr(zeros(15, Wb))
    for wg in parse.(Int, split(get(kvb, "wg", "32,16,8,4,2,1"), ","))
        ENV["GRAVITATIONALWAVEBAYES_PHM_WG"] = string(wg)
        BN.seobnr_phm_device_logl_grad!(llb, Gb, Pb, cfg, dets; backend = be, nfft = Nb)
        tm = Dict{Symbol,Float64}()
        t = @elapsed BN.seobnr_phm_device_logl_grad!(llb, Gb, Pb, cfg, dets; backend = be, nfft = Nb)
        BN.seobnr_phm_device_logl_grad!(llb, Gb, Pb, cfg, dets; backend = be, nfft = Nb, timers = tm)
        tp = @elapsed KF.seobnr_phm_device_loglikelihood!(llb, Pb, cfg, dets; backend = be, nfft = Nb)
        @printf("W %5d wg %2d: logL+grad %.3f s (%.3f ms/walker), logL %.3f s; plunge rev %.3f, dynamics %.3f, pn rev %.3f\n",
            Wb, wg, t, 1e3 * t / Wb, tp, get(tm, :rev_plunge, NaN), get(tm, :dynamics, NaN), get(tm, :rev_pn, NaN))
        flush(stdout)
    end
end
