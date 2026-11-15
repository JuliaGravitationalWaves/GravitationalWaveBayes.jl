# Reweight a run's posterior from the sampling prior p(d_L) ∝ d_L^2 to LVK's cosmological prior (uniform merger rate in
# the source's comoving frame, Planck15_LAL; GravitationalWaveBayes.Cosmology) by rejection sampling -- how GWTC `_cosmo` releases are
# made from `_nocosmo` (checked against GWTC-2.1 GW150914: KS D = 0.0009 on d_L). Compare the output with `_cosmo`
# release samples, the raw chain with `_nocosmo`.
#   julia --project=. scripts/reweight_cosmology.jl <run_dir> [warmup=0] [seed=1]
# Reads <run_dir>/chain.jls (d_L row "dL" in Gpc), writes <run_dir>/posterior_cosmo.csv (post-warmup draws kept).
using GravitationalWaveBayes, Serialization, DelimitedFiles, Random, Printf

function main(dir; warmup::Int=0, seed::Int=1)
    c = deserialize(joinpath(dir, "chain.jls"))
    names, chain = c.names, c.chain                       # (nrows, nwalkers, nsteps)
    post = reshape(chain[:, :, (warmup + 1):end], size(chain, 1), :)
    r = findfirst(==("dL"), names)
    r === nothing && error("no dL row in $(names)")
    keep = reweight_to_source_frame(Xoshiro(seed), 1000 .* post[r, :])
    open(joinpath(dir, "posterior_cosmo.csv"), "w") do io
        writedlm(io, vcat(reshape(names, 1, :), permutedims(post[:, keep])), ',')
    end
    @printf("%d of %d samples kept; median d_L %.1f -> %.1f Mpc\n", length(keep), size(post, 2),
            1000 * sort(post[r, :])[cld(size(post, 2), 2)], 1000 * sort(post[r, keep])[cld(length(keep), 2)])
end

if abspath(PROGRAM_FILE) == @__FILE__
    kv = Dict(split(a, "=")[1] => split(a, "=")[2] for a in ARGS[2:end] if occursin("=", a))
    main(ARGS[1]; warmup=parse(Int, get(kv, "warmup", "0")), seed=parse(Int, get(kv, "seed", "1")))
end
