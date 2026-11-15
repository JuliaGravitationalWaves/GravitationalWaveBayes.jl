# Posterior summary of a run_from_config output dir: per-parameter mean/sd (and truth z-score when an
# injection truth file exists), the largest posterior correlations, and -- when adapted_metric.jls is
# present -- the condition number of the posterior covariance in the metric's whitened frame (1 = the
# metric matches the posterior; NUTS step size scales roughly like 1/sqrt of that condition number).
#
#   julia --project=. scripts/analyze_run.jl runs/<dir> [nwarmup]
using Serialization, LinearAlgebra, Printf, DelimitedFiles
using GravitationalWaveBayes   # DenseMetric type, for adapted_metric.jls

function main(dir, nwarmup)
    c = deserialize(joinpath(dir, "chain.jls"))
    names, chain = c.names, c.chain
    np, nw, nd = size(chain)
    post = reshape(chain[:, :, (nwarmup + 1):end], np, :)
    mu = vec(sum(post; dims=2)) ./ size(post, 2)
    C = cov(post)
    sd = sqrt.(diag(C))
    truth = nothing
    tf = joinpath(dir, "injection_sampled_truth.csv")
    isfile(tf) && (truth = Float64.(readdlm(tf, ',')[2, :]))
    println("param          mean        sd        ", truth === nothing ? "" : "truth    z")
    for i in 1:np
        @printf("%-13s %10.5g %10.4g", names[i], mu[i], sd[i])
        truth === nothing || @printf("  %10.5g %6.2f", truth[i], (mu[i] - truth[i]) / sd[i])
        println()
    end
    R = C ./ (sd * sd')
    pairs = [(abs(R[i, j]), i, j) for i in 1:np for j in (i + 1):np]
    sort!(pairs; rev=true)
    println("largest |correlations|:")
    for (r, i, j) in pairs[1:8]
        @printf("  %-13s %-13s %6.3f\n", names[i], names[j], R[i, j])
    end
    mf = joinpath(dir, "adapted_metric.jls")
    if isfile(mf)
        m = deserialize(mf)
        M = hasproperty(m.inv_mass, :Minv) ? m.inv_mass.Minv : Diagonal(vec(m.inv_mass[:, 1]))
        L = cholesky(Symmetric(Matrix(M))).L
        W = L \ C / L'
        ev = eigvals(Symmetric(W))
        @printf("whitened posterior covariance eigenvalues: min %.3g max %.3g  (condition %.3g)\n", minimum(ev), maximum(ev), maximum(ev) / minimum(ev))
        println("step sizes: ", m.step_size)
    end
end

cov(X) = (Y = X .- sum(X; dims=2) ./ size(X, 2); Y * Y' ./ (size(X, 2) - 1))
main(ARGS[1], length(ARGS) > 1 ? parse(Int, ARGS[2]) : 0)
