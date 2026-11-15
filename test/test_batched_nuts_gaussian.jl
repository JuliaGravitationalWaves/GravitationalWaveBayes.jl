using Test
using GravitationalWaveBayes
using GravitationalWaveBayes: nuts_sample!, NUTSConfig
using LinearAlgebra
using Random

# Small inline mean/covariance helpers (avoids adding Statistics as a test dependency): `samples`
# is (nparams, ndraws).
function _empirical_mean(samples::AbstractMatrix{T}) where {T}
    vec(sum(samples; dims=2)) ./ size(samples, 2)
end

function _empirical_cov(samples::AbstractMatrix{T}, mu::AbstractVector{T}) where {T}
    n = size(samples, 2)
    centered = samples .- mu
    return (centered * centered') ./ (n - 1)
end

# Toy Gaussian targets for validating the generic NUTSCore engine, per DEVELOPMENT_PLAN.md Phase
# 1d's requirement to validate on toy targets before pointing NUTS at a GW likelihood. Zero
# waveform/KernelAbstractions/Enzyme machinery is involved here: the gradient is a plain analytic
# closure conforming to NUTSCore's `(grad_out, logp_out, positions, active_mask) -> nothing`
# interface, run against `nuts_sample!` directly (the same generic engine `BatchedNUTS.jl` wires
# to the GW likelihood).
function make_gaussian_logp_and_grad(mu::Vector{T}, Sigma::Matrix{T}) where {T}
    Sigma_inv = inv(Sigma)
    return function (grad_out::AbstractMatrix{T}, logp_out::AbstractVector{T},
                      positions::AbstractMatrix{T}, active_mask)
        nwalkers = size(positions, 2)
        @inbounds for w in 1:nwalkers
            diff = @view(positions[:, w]) .- mu
            g = Sigma_inv * diff
            @views grad_out[:, w] .= -g
            logp_out[w] = -T(0.5) * dot(diff, g)
        end
        return nothing
    end
end

@testset "Batched NUTS recovers Gaussian posteriors" begin
    @testset "diagonal covariance" begin
        rng = Random.Xoshiro(42)
        T = Float64
        mu = [1.0, -2.0]
        Sigma = [1.0 0.0; 0.0 4.0]
        logp_and_grad! = make_gaussian_logp_and_grad(mu, Sigma)

        nwalkers = 8
        positions = randn(rng, T, 2, nwalkers)
        step_size0 = fill(0.5, nwalkers)
        nwarmup = 800
        nsamples = 1200
        cfg = NUTSConfig{T}(; max_treedepth=8, target_accept=0.8, nwarmup=nwarmup)

        result = nuts_sample!(positions, logp_and_grad!; nsteps=nwarmup + nsamples,
                               step_size0=step_size0, cfg=cfg, rng=rng, record=true)

        @test all(isfinite, result.positions)
        @test all(isfinite, result.logp)

        post = result.chain[:, :, (nwarmup+1):end]
        samples = reshape(post, 2, :)  # pool across walkers and post-warmup draws

        emp_mean = _empirical_mean(samples)
        emp_cov = _empirical_cov(samples, emp_mean)

        @test isapprox(emp_mean, mu; atol=0.15)
        @test isapprox(emp_cov, Sigma; atol=0.5)
    end

    @testset "correlated covariance" begin
        rng = Random.Xoshiro(7)
        T = Float64
        mu = [0.0, 0.0]
        Sigma = [1.0 0.6; 0.6 1.0]
        logp_and_grad! = make_gaussian_logp_and_grad(mu, Sigma)

        nwalkers = 8
        positions = randn(rng, T, 2, nwalkers)
        step_size0 = fill(0.5, nwalkers)
        nwarmup = 800
        nsamples = 1200
        cfg = NUTSConfig{T}(; max_treedepth=8, target_accept=0.8, nwarmup=nwarmup)

        result = nuts_sample!(positions, logp_and_grad!; nsteps=nwarmup + nsamples,
                               step_size0=step_size0, cfg=cfg, rng=rng, record=true)

        @test all(isfinite, result.positions)
        @test all(isfinite, result.logp)

        post = result.chain[:, :, (nwarmup+1):end]
        samples = reshape(post, 2, :)

        emp_mean = _empirical_mean(samples)
        emp_cov = _empirical_cov(samples, emp_mean)

        @test isapprox(emp_mean, mu; atol=0.15)
        @test isapprox(emp_cov, Sigma; atol=0.5)
        # Off-diagonal correlation specifically: makes sure gradient/mass handling isn't
        # accidentally axis-aligned-only.
        @test emp_cov[1, 2] > 0.3
    end
end
