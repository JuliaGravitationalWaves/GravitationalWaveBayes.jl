# Compare recorded posteriors of several run directories (chain.jls): per-parameter mean, sd and the
# two-sample Kolmogorov-Smirnov distance to the first run. PT chains include warmup draws: pass
# `dir:skip` to drop the first `skip` draws (e.g. `runs/x_pt:200`).
#   julia --project=. scripts/compare_posteriors.jl runs/a_pt:200 runs/b_smc [runs/c ...]
using Serialization, Statistics, Printf

function load_draws(spec)
    dir, skip = occursin(':', spec) && !occursin(r"^[A-Za-z]:[\\/]", spec) ? (split(spec, ':')[1], parse(Int, split(spec, ':')[2])) :
                (spec, 0)
    c = deserialize(joinpath(dir, "chain.jls"))
    x = c.chain[:, :, skip+1:end]
    return String(dir), c.names, reshape(x, size(x, 1), :)
end

function ks(a, b)
    a = sort(a); b = sort(b); na, nb = length(a), length(b)
    i = j = 0; d = 0.0
    while i < na && j < nb
        v = min(a[i+1], b[j+1])
        while i < na && a[i+1] <= v; i += 1; end
        while j < nb && b[j+1] <= v; j += 1; end
        d = max(d, abs(i / na - j / nb))
    end
    return d
end

runs = [load_draws(s) for s in ARGS]
names = runs[1][2]
println(rpad("param", 14), join((rpad(basename(r[1]), 34) for r in runs), ""))
for (k, n) in enumerate(names)
    print(rpad(n, 14))
    for (i, r) in enumerate(runs)
        v = r[3][k, :]
        s = @sprintf("%.4g ± %.3g", mean(v), std(v))
        i > 1 && (s *= @sprintf(" KS %.3f", ks(runs[1][3][k, :], v)))
        print(rpad(s, 34))
    end
    println()
end
println(rpad("P(cos>0)", 14), join((rpad(@sprintf("%.3f", mean(r[3][10, :] .> 0)), 34) for r in runs), ""))
println(rpad("draws", 14), join((rpad(string(size(r[3], 2)), 34) for r in runs), ""))
