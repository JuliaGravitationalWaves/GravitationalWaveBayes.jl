using Test
using Random
using GravitationalWaveBayes
using GravitationalWaveforms: IMRPhenomXAS
using GravitationalWaveData: H1, L1, antenna_response_gmst, time_delay_from_geocenter_gmst

const BNr = GravitationalWaveBayes.BatchedNUTS
const KFr = GravitationalWaveBayes.KernelFused

# Relative binning for IMRPhenomXAS WITH the likelihood options: against the dense likelihood
# near the reference point (plain / phase + distance marginalized, f_ref timing, detector time reference),
# 1 and 2 detectors, and the per-walker Enzyme gradient against finite differences.

const XASr = IMRPhenomXAS()
const FREQSr = collect(20.0:0.25:512.0)
const PSDr = [1e-48 * ((f / 60)^-4 + 2 + (f / 250)^2) for f in FREQSr]
const GPSr = 1126259462.4

function _rb_data(dets, cfg, truth; seed = 3)
    tmpl = [LikelihoodData(FREQSr, zeros(ComplexF64, length(FREQSr)), PSDr, d, GPSr) for d in dets]
    # noise-free signal from the dense model at `truth` (the likelihood's own h), plus Gaussian noise
    rng = Xoshiro(seed)
    hs = Vector{ComplexF64}[]
    for (k, t) in enumerate(tmpl)
        rb = BNr.build_xas_relative_binning(length(tmpl) == 1 ? t : tmpl, cfg, truth; nbins = 10)  # only for geometry
        x = truth
        m1, m2, dec, th = BNr._eobimr_physical(x[1], x[2], x[9], x[10])
        mc, eta = KFr._chirp_mass_eta(m1, m2)
        pc = KFr._xas_phi_coeffs(XASr, mc, eta, x[3], x[4]); ac = KFr._xas_ampl_coeffs(XASr, mc, eta, x[3], x[4], x[5])
        g = rb.gmst
        dt_ref = rb.geo && rb.tref > 0 ? time_delay_from_geocenter_gmst(rb.dets[rb.tref], x[8], dec, g) : 0.0
        dt = rb.geo ? time_delay_from_geocenter_gmst(rb.dets[k], x[8], dec, g) - dt_ref : 0.0
        tsec = x[6] - KFr._xas_time_shift(pc, KFr._lik_opts(cfg)) + dt
        Fp, Fc = antenna_response_gmst(rb.dets[k], x[8], dec, x[11], g)
        combo = Fp * 0.5 * (1 + cos(th)^2) - im * Fc * cos(th)
        push!(hs, [KFr._xas_h(pc, ac, combo, tsec, x[7], f) for f in FREQSr])
    end
    # PSD scaled so the network optimal SNR is 25
    snr2 = sum(4 * 0.25 * sum(abs2.(h) ./ PSDr) for h in hs)
    psd = PSDr .* (snr2 / 25^2)
    out = LikelihoodData[]
    for (k, t) in enumerate(tmpl)
        n = sqrt.(psd ./ (4 * 0.25)) .* (randn(rng, length(FREQSr)) .+ im .* randn(rng, length(FREQSr))) ./ sqrt(2)
        push!(out, LikelihoodData(FREQSr, hs[k] .+ n, psd, t.detector, GPSr))
    end
    return length(out) == 1 ? out[1] : out
end

function _rb_dense_ll(data, cfg, pos)
    phys = similar(pos); BNr._reparam_to_physical!(phys, pos)
    ll = zeros(size(pos, 2))
    BNr._eval_ll!(ll, phys, FREQSr, cfg, data, XASr, nothing)
    return ll
end

@testset "XAS relative binning with likelihood options (ndet=$ndet, opts=$opts)" for ndet in (1, 2), opts in (:plain, :all)
    m1, m2 = 36.0, 29.0
    mc, q = BNr._m1_m2_to_mc_q(m1, m2)
    kw = opts == :plain ? (;) :
         (; marginalize_phase = true, marginalize_distance = true, distance_lo = 0.05, distance_hi = 2.0,
            distance_power = 2, time_reference = ndet, time_ref_frequency = 35.0, time_ref_offset = 0.3)
    cfg = with_likelihood_options(PEConfig{Float64}(length(FREQSr), 3, 0.01, 1); kw...)
    truth = [mc, q, 0.2, -0.1, 0.5, 0.004, 0.7, 1.3, sin(-0.5), cos(0.9), 0.4]
    data = _rb_data(ndet == 1 ? [H1] : [H1, L1], cfg, truth)
    rb = BNr.build_xas_relative_binning(data, cfg, truth; nbins = 400)
    @test all(n -> 250 <= n <= 400, rb.nbins)
    # walkers inside the posterior (network SNR 25): logL 0.4-15 below the reference. Measured error vs nbins:
    # ~0.01 at 200 bins, ~0.0025 at 400; ~0.05-0.2 only 70+ below the peak.
    pos = hcat(truth, truth .+ [0.03, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], truth .+ [0, 0, 0, 0, 0, 3e-4, 0, 0, 0, 0, 0])
    dense = _rb_dense_ll(data, cfg, pos)
    @test all(dense[1] .- dense[2:3] .< 20)
    lgr! = BNr.gw_logp_and_grad(data, cfg, FREQSr; model = XASr, relative_binning = rb)
    grad = zeros(11, 3); logp = zeros(3)
    lgr!(grad, logp, pos, nothing)
    @test isapprox(logp, dense .+ BNr._call_log_prior_vec(pos); atol = 0.01)
    @info "RB vs dense" ndet opts maxerr = maximum(abs.(logp .- dense .- BNr._call_log_prior_vec(pos)))
    # _eval_ll! (used by reconstruct_marginalized! etc.) agrees with the gradient path's primal
    phys = similar(pos); BNr._reparam_to_physical!(phys, pos)
    ll2 = zeros(3); BNr._eval_ll!(ll2, phys, FREQSr, cfg, data, XASr, rb)
    @test isapprox(ll2 .+ BNr._call_log_prior_vec(pos), logp; rtol = 1e-12)
    # gradient vs finite differences of the RB likelihood itself
    steps = [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-8, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]
    fd = zeros(11, 3); tg = zeros(11, 3); tl = zeros(3)
    for i in eachindex(pos)
        h = steps[(i - 1) % 11 + 1]
        pp = copy(pos); pp[i] += h; lgr!(tg, tl, pp, nothing); lp = sum(tl)
        pm = copy(pos); pm[i] -= h; lgr!(tg, tl, pm, nothing)
        fd[i] = (lp - sum(tl)) / (2h)
    end
    for r in 1:11
        ok = isapprox(grad[r, :], fd[r, :]; rtol = 1e-4, atol = 1e-3)
        ok || @info "RB gradient mismatch" r grad[r, :] fd[r, :]
        @test ok
    end
end
