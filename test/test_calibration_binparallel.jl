# Calibration in the bin-parallel IMRPhenomXPHM likelihood (2026-10-11, matched-gw150914): value against the CPU
# likelihood on calibrated-equivalent data (<d, C h>, <C h, C h> with strain d, PSD S == the uncalibrated sums with strain
# d / C, PSD S / |C|^2), gradients (15 + 40 rows) against central finite differences. KA CPU backend.
using Test, Random
src = read(joinpath(@__DIR__, "test_xprec_likelihood_options.jl"), String)
include_string(Main, src[1:first(findfirst("@testset", src)) - 1])   # its helpers: _pmake_data, _ploglike, _ppositions, ...

@testset "IMRPhenomXPHM bin-parallel with calibration (CPU backend, setup = $setup)" for model in (IMRPhenomXPHM(), IMRPhenomXPHM(102)),
                                                                                         setup in (:host, :device)
    data = [_pmake_data(model, H1, 150.0, 1.0), _pmake_data(model, L1, 150.0, 2.0)]
    nn = 10
    nodes = GravitationalWaveBayes.calibration_nodes(20.0, 300.0, nn)
    W = GravitationalWaveBayes.calibration_weights(nodes, FREQSp)
    rng = Random.Xoshiro(3)
    mu = 0.01 .* randn(rng, 4nn); sigma = 0.03 .+ 0.02 .* rand(rng, 4nn)
    cal = (; nn, W, mu, sigma, nodes)
    kw = (; marginalize_distance = true, distance_lo = 0.05, distance_hi = 1.5, distance_power = 2,
          time_reference = 1, time_ref_frequency = 35.0, time_ref_offset = 0.3)
    nw = 2
    cfg = _pcfg(; nw, kw...)
    pos15 = _ppositions([TRUTHp, OTHERp])
    z = randn(rng, 4nn, nw)
    pos = vcat(pos15, z)
    f! = BNp.xphm_binparallel_logp_and_grad(data, cfg, FREQSp; model, backend = KernelAbstractions.CPU(), nwalkers = nw,
                                            calibration = cal, setup)
    grad = zeros(15 + 4nn, nw); logp = zeros(nw)
    f!(grad, logp, pos, nothing)
    # value: per walker, the CPU likelihood on the calibrated-equivalent data
    for w in 1:nw
        node = mu .+ sigma .* z[:, w]
        dmod = map(enumerate(data)) do (d, dd)
            o = (d - 1) * 2nn
            C = GravitationalWaveBayes.calibration_factor(W, node[o+1:o+nn], node[o+nn+1:o+2nn])
            LikelihoodData(dd.freqs, dd.strain ./ C, dd.psd ./ abs2.(C), dd.detector, dd.gps_time)
        end
        ref = _ploglike(model, dmod, _pcfg(; nw = 1, kw...), pos15[:, w:w])[1] + BNp._call_log_prior_vec(pos15[:, w:w])[1]
        # (223: angle table, ~1e-5 from the exact angles; 102: exact)
        @test model isa IMRPhenomXPHM{223} ? isapprox(logp[w], ref; atol = 1e-4) : isapprox(logp[w], ref; rtol = 1e-10)
    end
    @test all(isfinite, grad)
    # the log-likelihood alone (`value_only`, the RW / reweighting path) is the same number
    lv = zeros(nw); gv = ones(15 + 4nn, nw); f!(gv, lv, pos, nothing; value_only = true)
    @test isapprox(lv, logp; rtol = 1e-12) && all(iszero, gv)
    # gradients: FD of the same closure
    fval(p) = (g = zeros(size(p)); l = zeros(nw); f!(g, l, p, nothing); l)
    for r in vcat(1:15, 16:3:15 + 4nn)
        h = r == 6 ? 1e-8 : 1e-6
        pp = copy(pos); pp[r, :] .+= h; pm = copy(pos); pm[r, :] .-= h
        fd = (fval(pp) .- fval(pm)) ./ (2h)
        ok = isapprox(grad[r, :], fd; rtol = 1e-4, atol = 1e-3)
        ok || @info "calibrated bin-parallel gradient mismatch" model r grad[r, :] fd
        @test ok
    end
end

# Calibration MARGINALIZATION (2026-10-12): the log-mean-exp over response curves of the sampled-calibration likelihood
# at each curve's node values (the brute force), and gradients against central finite differences of the closure.
@testset "IMRPhenomXPHM bin-parallel with calibration marginalization (CPU backend, setup = $setup)" for model in (IMRPhenomXPHM(), IMRPhenomXPHM(102)),
                                                                                                         setup in (:host, :device)
    data = [_pmake_data(model, H1, 150.0, 1.0), _pmake_data(model, L1, 150.0, 2.0)]
    nn = 10
    nodes = GravitationalWaveBayes.calibration_nodes(20.0, 300.0, nn)
    W = GravitationalWaveBayes.calibration_weights(nodes, FREQSp)
    rng = Random.Xoshiro(5)
    mu = 0.01 .* randn(rng, 4nn); sigma = 0.03 .+ 0.02 .* rand(rng, 4nn)
    cal = (; nn, W, mu, sigma, nodes)
    nc = 6
    curves = GravitationalWaveBayes.calibration_curves(cal, nc; rng = Random.Xoshiro(9))
    kw = (; marginalize_distance = true, distance_lo = 0.05, distance_hi = 1.5, distance_power = 2,
          time_reference = 1, time_ref_frequency = 35.0, time_ref_offset = 0.3)
    nw = 2
    pos = _ppositions([TRUTHp, OTHERp])
    fm! = BNp.xphm_binparallel_logp_and_grad(data, _pcfg(; nw, kw...), FREQSp; model, backend = KernelAbstractions.CPU(),
                                             nwalkers = nw, calmarg = curves.C, setup)
    grad = zeros(15, nw); logp = zeros(nw)
    fm!(grad, logp, pos, nothing)
    fs! = BNp.xphm_binparallel_logp_and_grad(data, _pcfg(; nw = nc, kw...), FREQSp; model, backend = KernelAbstractions.CPU(),
                                             nwalkers = nc, calibration = cal, setup)
    for w in 1:nw
        P = vcat(repeat(pos[:, w:w], 1, nc), curves.Z)
        lk = zeros(nc); fs!(zeros(size(P)), lk, P, nothing)
        m = maximum(lk); ref = m + log(sum(exp.(lk .- m))) - log(nc)
        @test isapprox(logp[w], ref; atol = 1e-8)
    end
    @test all(isfinite, grad)
    lv = zeros(nw); fm!(zeros(15, nw), lv, pos, nothing; value_only = true)
    @test isapprox(lv, logp; rtol = 1e-12)
    # the per-curve log-likelihoods (`curve_ll`, for the calibration posterior) log-mean-exp to the value
    LK = zeros(nc, nw); fm!(zeros(15, nw), zeros(nw), pos, nothing; curve_ll = LK)
    for w in 1:nw
        m = maximum(LK[:, w])
        @test isapprox(m + log(sum(exp.(LK[:, w] .- m))) - log(nc), logp[w]; atol = 1e-10)
    end
    fval(p) = (g = zeros(size(p)); l = zeros(nw); fm!(g, l, p, nothing); l)
    for r in 1:15
        h = r == 6 ? 1e-8 : 1e-6
        pp = copy(pos); pp[r, :] .+= h; pm = copy(pos); pm[r, :] .-= h
        fd = (fval(pp) .- fval(pm)) ./ (2h)
        ok = isapprox(grad[r, :], fd; rtol = 1e-4, atol = 1e-3)
        ok || @info "calmarg bin-parallel gradient mismatch" model r grad[r, :] fd
        @test ok
    end
end
