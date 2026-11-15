# SEOBNRv5PHM gradient, the plunge and PN-table reverses: node-parallel (`chain_rev = :par`) against the serial
# step-by-step reverses (`:serial`) on the same walkers, then timings of both. FD end-to-end: scripts/check_phm_device_grad.jl.
#   julia -t 1 --project=<env> scripts/check_phm_chain_rev.jl cpu|cuda [W=3] [bench=256,1024] (:par up to the `_phm_chain_par_max()` walker counts)
lines = readlines(joinpath(@__DIR__, "check_phm_sample_rev.jl"))
stop = findfirst(l -> startswith(l, "function check_samples"), lines)
args = copy(ARGS); empty!(ARGS); push!(ARGS, args[1], "W=2")
include_string(Main, join(lines[1:stop-1], "\n"))
kvp = Dict(split(a, "=")[1] => split(a, "=")[2] for a in args if occursin("=", a))

function cmp_chain(W)
    P = toarr(walkers(W; seed = 7))
    N = KF._phm_nfft(freqs, 0.9 * minimum(Array(P)[1, :] .+ Array(P)[2, :]))
    ll = toarr(zeros(W)); Gs = toarr(zeros(15, W)); Gp = toarr(zeros(15, W))
    t = @elapsed BN.seobnr_phm_device_logl_grad!(ll, Gs, P, cfg, dets; backend = be, nfft = N, chain_rev = :serial)
    @printf("serial first call %.1f s\n", t); flush(stdout)
    t = @elapsed BN.seobnr_phm_device_logl_grad!(ll, Gp, P, cfg, dets; backend = be, nfft = N, chain_rev = :par)
    @printf("par first call %.1f s; logL %s\n", t, Array(ll)); flush(stdout)
    a = Array(Gs); b = Array(Gp)
    worst = 0.0
    for k in 1:15
        e = maximum(abs.(a[k, :] .- b[k, :])) / max(maximum(abs.(a[k, :])), 1e-300)
        worst = max(worst, e)
        @printf("  row %2d  serial %s  par %s  rel %.1e\n", k, a[k, 1], b[k, 1], e)
    end
    @printf("worst row rel diff par vs serial %.2e\n", worst); flush(stdout)
end
cmp_chain(parse(Int, get(kvp, "W", "3")))

for Wb in parse.(Int, split(get(kvp, "bench", "256,1024"), ","))
    Pb = toarr(walkers(Wb; seed = 5))
    Nb = KF._phm_nfft(freqs, 0.9 * minimum(Array(Pb)[1, :] .+ Array(Pb)[2, :]))
    llb = toarr(zeros(Wb)); Gb = toarr(zeros(15, Wb))
    for mode in (:par, :serial)
        BN.seobnr_phm_device_logl_grad!(llb, Gb, Pb, cfg, dets; backend = be, nfft = Nb, chain_rev = mode)
        t = @elapsed BN.seobnr_phm_device_logl_grad!(llb, Gb, Pb, cfg, dets; backend = be, nfft = Nb, chain_rev = mode)
        tm = Dict{Symbol,Float64}()
        BN.seobnr_phm_device_logl_grad!(llb, Gb, Pb, cfg, dets; backend = be, nfft = Nb, chain_rev = mode, timers = tm)
        @printf("W %5d %-6s: logL+grad %.3f s (%.3f ms/walker); plunge rev %.3f, pn rev %.3f, sts rev %.3f; finite %d/%d\n", Wb, mode, t,
            1e3 * t / Wb, get(tm, :rev_plunge, NaN) + get(tm, :rev_plunge_jac, 0.0),
            get(tm, :rev_pn, NaN) + get(tm, :rev_pn_jac, 0.0), get(tm, :rev_sts, NaN), count(isfinite, Array(Gb)), length(Gb))
        flush(stdout)
    end
end
