# Our pooled SEOBNRv5PHM posterior vs the LVK GWTC-2.1 GW150914 release (cached by GravitationalWaveData): detector-frame chirp mass,
# mass ratio, chi_eff (approximately conserved, so comparable although our spins are defined at 50 Hz and LVK's at
# 20 Hz), luminosity distance, sky position, cos theta_jn. Medians and 90% intervals.
#   julia --project=<env with GravitationalWaveData> scripts/compare_lvk.jl <release .h5> <run dir> [<run dir> ...]
using GravitationalWaveData, Serialization, Statistics, Printf
path = ARGS[1]
dirs = ARGS[2:end]
println("LVK analyses in the file: ", join(list_posterior_analyses(path), ", "))

function ours(dirs)
    X = reduce(hcat, [reshape(deserialize(joinpath(d, "chain.jls")).chain, 15, :) for d in dirs])
    mc, q = X[1, :], X[2, :]
    m1 = mc .* (1 .+ q) .^ (1 / 5) ./ q .^ (3 / 5); m2 = q .* m1
    chieff = (X[3, :] .* m1 .+ X[4, :] .* m2) ./ (m1 .+ m2)
    return Dict("mc_det" => mc, "q" => q, "chi_eff" => chieff, "dL_Mpc" => 1000 .* X[5, :], "ra" => X[8, :],
                "dec" => asin.(clamp.(X[9, :], -1, 1)), "cos_theta_jn" => X[10, :])
end

function lvk(path, analysis)
    s = load_posterior_samples(path, analysis)
    get1(ks...) = (for k in ks; haskey(s, k) && return s[k]; end; nothing)
    mc = get1("chirp_mass")                          # detector frame in the PE releases
    d = Dict{String,Vector{Float64}}()
    mc === nothing || (d["mc_det"] = mc)
    for (o, ks) in (("q", ("mass_ratio",)), ("chi_eff", ("chi_eff",)), ("dL_Mpc", ("luminosity_distance",)),
                    ("ra", ("ra",)), ("dec", ("dec",)), ("cos_theta_jn", ("cos_theta_jn",)))
        v = get1(ks...); v === nothing || (d[o] = v)
    end
    if !haskey(d, "cos_theta_jn") && haskey(s, "theta_jn")
        d["cos_theta_jn"] = cos.(s["theta_jn"])
    end
    return d
end

q5(v) = (quantile(v, 0.05), median(v), quantile(v, 0.95))
fmt(v) = (a = q5(v); @sprintf("%8.3f [%7.3f, %7.3f]", a[2], a[1], a[3]))
O = ours(dirs)
want = filter(a -> occursin(r"SEOBNRv4PHM|IMRPhenomXPHM|Mixed", a) && !occursin("prior", lowercase(a)),
              list_posterior_analyses(path))
L = [(a, lvk(path, a)) for a in want]
println(rpad("param", 14), rpad("ours (SEOBNRv5PHM)", 30), join((rpad(a, 30) for (a, _) in L), ""))
for k in ("mc_det", "q", "chi_eff", "dL_Mpc", "ra", "dec", "cos_theta_jn")
    print(rpad(k, 14), rpad(fmt(O[k]), 30))
    for (_, d) in L
        print(rpad(haskey(d, k) ? fmt(d[k]) : "-", 30))
    end
    println()
end
@printf("P(cos theta_jn > 0): ours %.4f; %s\n", mean(O["cos_theta_jn"] .> 0),
        join((@sprintf("%s %.4f", a, mean(d["cos_theta_jn"] .> 0)) for (a, d) in L if haskey(d, "cos_theta_jn")), "; "))
println("ours: ", length(O["q"]), " draws pooled from ", length(dirs), " runs")
