# Summarize a P-P run written by scripts/pp_test.jl: per-parameter Kolmogorov-Smirnov test of the
# truth percentiles against Uniform(0,1), plus Fisher's combined p-value.
#
#   julia scripts/pp_summary.jl runs/pp_o4*/pp_results.csv         (several files are pooled)
#
# No package dependencies (reads the CSV by hand) so it runs outside the project environment.

using Printf

# Asymptotic Kolmogorov distribution with Stephens' small-n correction.
function ks_pvalue(pcts)
    x = sort(pcts)
    n = length(x)
    d = maximum(max(i / n - x[i], x[i] - (i - 1) / n) for i in 1:n)
    t = (sqrt(n) + 0.12 + 0.11 / sqrt(n)) * d
    p = 2 * sum((-1)^(k - 1) * exp(-2 * k^2 * t^2) for k in 1:100)
    return d, clamp(p, 0.0, 1.0)
end

# chi-square survival function for even degrees of freedom 2m (closed form).
chisq_sf_even(x, m) = exp(-x / 2) * sum((x / 2)^k / factorial(big(k)) for k in 0:(m - 1))

function main(paths)
    header = nothing
    rows = Vector{Vector{Float64}}()
    for path in paths
        lines = collect(eachline(path))
        h = split(lines[1], ',')
        header === nothing ? (header = h) : (h == header || error("$path has a different header"))
        append!(rows, [parse.(Float64, split(l, ',')) for l in lines[2:end] if !isempty(l)])
    end
    n = length(rows)
    println("P-P summary for $n injections ($(join(paths, ", ")))")
    snr = [r[findfirst(==("net_snr"), header)] for r in rows]
    rh = [r[findfirst(==("max_rhat"), header)] for r in rows]
    @printf("  network SNR: median %.1f, range [%.1f, %.1f];  max Rhat: median %.3f, worst %.3f\n",
            sort(snr)[cld(n, 2)], minimum(snr), maximum(snr), sort(rh)[cld(n, 2)], maximum(rh))
    ps = Float64[]
    for (j, h) in enumerate(header)
        startswith(h, "pct_") || continue
        d, p = ks_pvalue([r[j] for r in rows])
        push!(ps, p)
        @printf("  %-14s KS D=%.3f  p=%.3f\n", h[5:end], d, p)
    end
    x = -2 * sum(log.(max.(ps, 1e-300)))
    @printf("  combined (Fisher) p = %.3g over %d parameters\n", Float64(chisq_sf_even(x, length(ps))), length(ps))

    # Region masses: the mean posterior mass of a prior-quantile tail must equal its prior mass (SBC identity).
    # z < 0 = the sampler under-fills that tail. The regions overlap, so the z values are not independent; no flag
    # below 20 injections (the s.e. itself is unreliable).
    println("  region mass (mean posterior mass vs prior mass; z = (mean - prior) / s.e.):")
    for (j, h) in enumerate(header)
        startswith(h, "mass_") || continue
        v = [r[j] for r in rows]
        m = parse(Float64, split(h, '_')[end])
        mu = sum(v) / n
        se = sqrt(sum(abs2, v .- mu) / (n - 1) / n)
        @printf("    %-24s %.4f vs %.4f  z=%+.2f%s\n", h[6:end], mu, m, (mu - m) / se, n >= 20 && abs(mu - m) > 3se ? "  <--" : "")
    end

    # TARP: f ~ Uniform(0,1) for correct joint coverage. 12 var(f) > 1 = U-shaped f = over-confident (too narrow);
    # < 1 = under-confident. ECP(a) = P(f < a) should equal a.
    jt = findfirst(==("tarp"), header)
    if jt !== nothing
        f = [r[jt] for r in rows]
        d, p = ks_pvalue(f)
        mu = sum(f) / n
        @printf("  TARP: KS D=%.3f p=%.3f;  mean f %.3f (0.5);  12 var(f) = %.2f (1; > 1 over-confident)\n",
                d, p, mu, 12 * sum(abs2, f .- mu) / (n - 1))
        println("    ECP: ", join([@sprintf("%.2f->%.2f", a, count(<(a), f) / n) for a in (0.1, 0.25, 0.5, 0.75, 0.9)], "  "),
                @sprintf("  (s.e. <= %.2f)", 0.5 / sqrt(n)))
    end
end

main(ARGS)
