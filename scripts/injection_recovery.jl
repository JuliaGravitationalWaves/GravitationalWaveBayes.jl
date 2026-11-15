# Recovery of a SEOBNRv5PHM [injection] run: truth vs posterior median, 90% interval and the truth's posterior
# percentile per parameter (the time row is t_ref in the sampler's frame and is skipped), plus chi_eff and chi_p.
#   julia --project=<env> scripts/injection_recovery.jl <run dir>   (reads <run dir>/injection.txt and chain.jls)
using Serialization, Statistics, Printf
dir = ARGS[1]
txt = Dict(strip(a) => strip(b) for (a, b) in (split(l, "=") for l in eachline(joinpath(dir, "injection.txt")) if occursin("=", l)))
v(k) = parse(Float64, txt[k])
X = reshape(deserialize(joinpath(dir, "chain.jls")).chain, 15, :)
mc, q = X[1, :], X[2, :]
m1 = mc .* (1 .+ q) .^ (1 / 5) ./ q .^ (3 / 5); m2 = q .* m1
chieff(c1, c2, a, b) = (c1 .* a .+ c2 .* b) ./ (a .+ b)
function chip(q, c1x, c1y, c2x, c2y)
    a1 = sqrt.(c1x .^ 2 .+ c1y .^ 2); a2 = sqrt.(c2x .^ 2 .+ c2y .^ 2)
    return max.(a1, (4 .* q .+ 3) ./ (4 .+ 3 .* q) .* q .* a2)
end
rows = [("mc", 1, v("mc")), ("q", 2, v("q")), ("chi1z", 3, v("chi1")), ("chi2z", 4, v("chi2")), ("dL", 5, v("dL")),
        ("phase", 7, v("phase")), ("ra", 8, v("ra")), ("sin_dec", 9, sin(v("dec"))), ("cos_theta_jn", 10, cos(v("theta_jn"))),
        ("psi", 11, v("psi")), ("chi1x", 12, v("chi1x")), ("chi1y", 13, v("chi1y")), ("chi2x", 14, v("chi2x")),
        ("chi2y", 15, v("chi2y"))]
cols = [(n, X[r, :], t) for (n, r, t) in rows]
tm1, tm2 = v("m1"), v("m2")
push!(cols, ("chi_eff", chieff(X[3, :], X[4, :], m1, m2), (v("chi1") * tm1 + v("chi2") * tm2) / (tm1 + tm2)))
push!(cols, ("chi_p", chip(q, X[12, :], X[13, :], X[14, :], X[15, :]),
             chip([v("q")], [v("chi1x")], [v("chi1y")], [v("chi2x")], [v("chi2y")])[1]))
@printf("%-14s %10s %10s %22s %10s\n", "param", "truth", "median", "90% interval", "truth pct")
inside = 0
for (n, x, t) in cols
    lo, md, hi = quantile(x, 0.05), median(x), quantile(x, 0.95)
    pct = mean(x .< t)
    global inside += lo <= t <= hi
    @printf("%-14s %10.4f %10.4f   [%8.4f, %8.4f] %9.3f%s\n", n, t, md, lo, hi, pct, lo <= t <= hi ? "" : "  <- outside 90%")
end
@printf("%d of %d parameters inside their 90%% interval; network optimal SNR %s; %d draws\n", inside, length(cols),
        txt["network_optimal_snr"], size(X, 2))
