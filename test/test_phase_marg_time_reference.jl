using Test
using Random
using GravitationalWaveBayes
import GravitationalWaveforms
using GravitationalWaveforms: IMRPhenomXAS, SEOBNRv5AlignedSpinF2, Ampl, Phi
using GravitationalWaveData: H1, L1, gmst, geometry, time_delay_from_geocenter_gmst
using GravitationalWaveforms: _seobv5f2_coeffs, _seobv5f2_ampl_scalar, _seobv5f2_phi_scalar, GMsun_over_c2_Gpc, GMsun_over_c3

const BN = GravitationalWaveBayes.BatchedNUTS
const KF = GravitationalWaveBayes.KernelFused

# `PEConfig.marginalize_phase` / `PEConfig.time_reference` & co.:
#   - log I0 approximation against direct quadrature,
#   - marginalized logL against a brute-force phase average of the plain likelihood (single + H1/L1),
#   - time_reference=k: identical likelihood to the geocentric one after shifting t_c by dt_k(ra, dec),
#   - gw_logp_and_grad (reverse precoeffs path, now also multi-detector) against finite differences
#     and against the forward-mode path.
# The brute-force identities (thousands to ~10^6 likelihood calls) use the cheap IMRPhenomXAS (`MODEL`); the
# PAF2Model kernels (`PAF2`, SEOBNRv5AlignedSpinF2) get the phase-average, time-reference, fmax_taper and gradient
# checks.

const _pmtr_old_reverse_flag = BN._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[]
const GPS = 1126259462.4

function _strain(model::GravitationalWaveBayes.PAF2Model, freqs, det, phys)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = phys
    mc, eta = KF._chirp_mass_eta(m1, m2)
    coeffs = _seobv5f2_coeffs(model, mc, eta, chi1, chi2)
    g = gmst(GPS)
    phiRef, combo = KF._extrinsic_setup_phase_only(phase, ra, dec, theta_jn, psi, geometry(det), g)
    return [combo * _seobv5f2_ampl_scalar(coeffs, f, dL, GMsun_over_c2_Gpc) *
            cis(-(_seobv5f2_phi_scalar(coeffs, f) + 2pi * t_c * f - phiRef)) for f in freqs]
end

# XAS family (KernelFused `_PHASE_CONVENTION`): h = combo A exp(i (Phi - 2 pi f t - phiRef)) via Phi's t0/phiRef.
function _strain(model::IMRPhenomXAS, freqs, det, phys)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = phys
    mc, eta = KF._chirp_mass_eta(m1, m2)
    phiRef, combo = KF._extrinsic_setup_phase_only(phase, ra, dec, theta_jn, psi, geometry(det), gmst(GPS))
    M_sec = mc / eta^0.6 * GMsun_over_c3
    A = Ampl(model, freqs, mc, eta, chi1, chi2, dL)
    P = Phi(model, freqs, mc, eta, chi1, chi2; t0 = 2pi * t_c / M_sec, phiRef = phiRef)
    return combo .* A .* cis.(P)
end

# Sampled-space walker columns from physical tuples.
function _positions(phys_list)
    positions = zeros(Float64, 11, length(phys_list))
    for (w, p) in enumerate(phys_list)
        m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = p
        mc, q = BN._m1_m2_to_mc_q(m1, m2)
        positions[:, w] .= (mc, q, chi1, chi2, dL, t_c, phase, ra, sin(dec), cos(theta_jn), psi)
    end
    return positions
end

function _loglike(data, cfg, model, positions)
    physical = similar(positions)
    BN._reparam_to_physical!(physical, positions)
    ll = zeros(size(positions, 2))
    BN._eval_ll!(ll, physical, data isa AbstractVector ? data[1].freqs : data.freqs, cfg, data, model, nothing)
    return ll
end

logsumexp(x) = (m = maximum(x); m + log(sum(exp.(x .- m))))

const MODEL = IMRPhenomXAS()
const PAF2 = SEOBNRv5AlignedSpinF2()
const FREQS = collect(20.0:0.25:30.0)
const TRUTH = (36.0, 29.0, 0.1, -0.1, 0.4, 0.004, 1.3, 1.2, -0.4, 0.7, 0.6)

# Deterministic "noise": a fixed pseudo-random complex series, so <d,h> has a generic phase.
_noise(n, seed) = [complex(sin(1.7 * k + seed), cos(2.3 * k * seed + 0.1)) for k in 1:n]

function _make_data(det, snr2_target, seed; model = MODEL)
    h = _strain(model, FREQS, det, TRUTH)
    df = FREQS[2] - FREQS[1]
    s = 4 * df * sum(abs2, h) / snr2_target        # constant PSD giving <h,h> = snr2_target
    noise = sqrt(s / (4 * df)) .* _noise(length(FREQS), seed)
    return LikelihoodData(FREQS, h .+ noise, fill(s, length(FREQS)), det, GPS)
end

@testset "log I0 approximation" begin
    for x in (0.0, 1e-3, 0.5, 2.0, 3.7499, 3.7501, 10.0, 50.0, 800.0, 5000.0)
        # log I0(x) = x + log((1/pi) int_0^pi exp(x (cos t - 1)) dt), midpoint rule (spectrally accurate here)
        n = 20000
        quad = sum(exp(x * (cos((k - 0.5) * pi / n) - 1)) for k in 1:n) / n
        @test isapprox(KF._log_besseli0(x), x + log(quad); atol=1e-6)
    end
end

@testset "phase-marginalized likelihood == phase average ($(nameof(typeof(model))), ndet=$(ndet), snr2=$(snr2))" for
        model in (MODEL, PAF2), ndet in (1, 2), snr2 in (4.0, 400.0)
    data = ndet == 1 ? _make_data(H1, snr2, 1.0; model) : [_make_data(H1, snr2 / 2, 1.0; model), _make_data(L1, snr2 / 2, 2.0; model)]
    cfg0 = PEConfig{Float64}(length(FREQS), 1, 0.01, 1)
    cfgm = with_likelihood_options(cfg0; marginalize_phase=true)
    walkers = [TRUTH, (37.0, 28.0, 0.0, 0.0, 0.5, 0.003, 0.2, 1.0, -0.2, 1.0, 0.3)]
    for p in walkers
        pos = _positions([p])
        lm = _loglike(data, cfgm, model, pos)[1]
        nph = 512
        lls = map(0:nph-1) do k
            q = copy(pos); q[7, 1] = 2pi * k / nph
            _loglike(data, cfg0, model, q)[1]
        end
        @test isapprox(lm, logsumexp(lls) - log(nph); atol=2e-6)
        # the marginal value must not depend on the walker's own phase row
        q = copy(pos); q[7, 1] = 4.0
        @test isapprox(_loglike(data, cfgm, model, q)[1], lm; atol=1e-9)
    end
    # default config is untouched by the new code path's bookkeeping
    @test _loglike(data, cfg0, model, _positions([TRUTH]))[1] ≈
          _loglike(data, with_likelihood_options(cfg0; marginalize_phase=false), model, _positions([TRUTH]))[1]
end

@testset "time_reference shifts t_c by the reference detector's delay ($(nameof(typeof(model))))" for model in (MODEL, PAF2)
    data = [_make_data(H1, 200.0, 1.0; model), _make_data(L1, 200.0, 2.0; model)]
    cfg0 = PEConfig{Float64}(length(FREQS), 1, 0.01, 1)
    g = gmst(GPS)
    for (k, det) in ((1, H1), (2, L1))
        cfgk = with_likelihood_options(cfg0; time_reference=k)
        pos_geo = _positions([TRUTH])
        ra, dec = TRUTH[8], TRUTH[9]
        pos_det = copy(pos_geo)
        pos_det[6, 1] += time_delay_from_geocenter_gmst(det, ra, dec, g)
        @test isapprox(_loglike(data, cfgk, model, pos_det)[1], _loglike(data, cfg0, model, pos_geo)[1]; rtol=1e-12)
    end
    @test_throws ArgumentError _loglike(data, with_likelihood_options(cfg0; time_reference=3), model, _positions([TRUTH]))
    # (a model without likelihood options; IMRPhenomD has them since 2026-10-04)
    @test_throws ArgumentError _loglike(data, with_likelihood_options(cfg0; marginalize_phase=true),
        GravitationalWaveforms.AlignedSpinPNHamF2(), _positions([TRUTH]))
end

@testset "time_ref_frequency: sampled time = t_c + t*(f_ref) + offset ($(nameof(typeof(model))))" for model in (MODEL, PAF2)
    data = [_make_data(H1, 200.0, 1.0; model), _make_data(L1, 200.0, 2.0; model)]
    cfg0 = PEConfig{Float64}(length(FREQS), 1, 0.01, 1)
    for (fref, off) in ((24.0, 0.0), (27.5, 0.37))
        cfgf = with_likelihood_options(cfg0; time_ref_frequency=fref, time_ref_offset=off)
        pos = _positions([TRUTH])
        tstar = time_at_frequency(model, TRUTH[1], TRUTH[2], TRUTH[3], TRUTH[4], fref)
        @test isfinite(tstar)
        posf = copy(pos); posf[6, 1] += tstar + off
        @test isapprox(_loglike(data, cfgf, model, posf)[1], _loglike(data, cfg0, model, pos)[1]; rtol=1e-12)
    end
end

@testset "fmax_taper makes logL continuous as the PAF2 table end crosses frequency bins" begin
    # Band to 40 Hz so the table's own end (~31 Hz for these masses) lies inside it; scanning mc moves
    # fmax across several 0.25 Hz bins. Without the taper each crossing is a jump in logL.
    freqs40 = collect(20.0:0.25:40.0)
    h = _strain(PAF2, freqs40, H1, TRUTH)
    s = 4 * 0.25 * sum(abs2, h) / 200.0
    d = LikelihoodData(freqs40, h .+ sqrt(s / 1.0) .* _noise(length(freqs40), 1.0), fill(s, length(freqs40)), H1, GPS)
    n = 1601
    pos = repeat(_positions([TRUTH]), 1, n)
    pos[1, :] .= range(pos[1, 1] - 0.5, pos[1, 1] + 0.5; length=n)
    maxjump(taper) = begin
        cfg = PEConfig{Float64}(length(freqs40), n, 0.01, 1; fmax_taper=taper)
        g = zeros(11, n); l = zeros(n)
        BN.gw_logp_and_grad(d, cfg, freqs40; model=PAF2)(g, l, pos, nothing)
        dx = pos[1, 2] - pos[1, 1]
        maximum(abs(l[i+1] - l[i] - dx * (g[1, i] + g[1, i+1]) / 2) for i in 1:n-1)
    end
    @test maxjump(0.0) > 1e-2       # the discontinuity is real without the taper
    # and gone with it (FD/trapezoid agreement): untapered jumps are up to 2.6 here, tapered <= 1.9e-4. The residual
    # sits where the PA grid loses a node (nreal 19 -> 18 at mc ~ 27.97), with or without the taper: the regridded
    # t/phi quadrature and the nodal d omega/dt (`GravitationalWaveforms._spa_node_wdot`). It was 1.4e-3 while the SPA amplitude
    # used the phase Hermite's second derivative (until 2026-10-29).
    @test maxjump(3.0) < 5e-4
end

@testset "distance-marginalized likelihood == brute-force integral (ndet=$(ndet), snr2=$(snr2), margphase=$(mp), dpow=$(dpow))" for
        ndet in (1, 2), snr2 in (4.0, 400.0), mp in (false, true), dpow in (0, 2)
    data = ndet == 1 ? _make_data(H1, snr2, 1.0) : [_make_data(H1, snr2 / 2, 1.0), _make_data(L1, snr2 / 2, 2.0)]
    lo, hi = 0.05, 1.5
    cfg0 = PEConfig{Float64}(length(FREQS), 1, 0.01, 1)
    cfgp = with_likelihood_options(cfg0; marginalize_phase=mp)
    cfgd = with_likelihood_options(cfg0; marginalize_phase=mp, marginalize_distance=true,
                                   distance_lo=lo, distance_hi=hi, distance_power=dpow)
    for p in (TRUTH, (37.0, 28.0, 0.0, 0.0, 0.9, 0.003, 0.2, 1.0, -0.2, 1.0, 0.3))
        pos = _positions([p])
        ld = _loglike(data, cfgd, MODEL, pos)[1]
        # brute force: fine midpoint rule over D of the (phase-marginalized or plain) likelihood
        nD = 20000
        Ds = [lo + (hi - lo) * (k - 0.5) / nD for k in 1:nD]
        lls = map(Ds) do D
            q = copy(pos); q[5, 1] = D
            _loglike(data, cfgp, MODEL, q)[1] + dpow * log(D)
        end
        lognorm = log((hi^(dpow + 1) - lo^(dpow + 1)) / (dpow + 1))
        brute = logsumexp(lls) + log((hi - lo) / nD) - lognorm
        @test isapprox(ld, brute; atol=1e-4)
        # independent of the walker's own dL row
        q = copy(pos); q[5, 1] = 0.37
        @test isapprox(_loglike(data, cfgd, MODEL, q)[1], ld; atol=1e-8)
    end
end

@testset "reconstruct_marginalized! draws from the conditional posterior" begin
    data = [_make_data(H1, 150.0, 1.0), _make_data(L1, 150.0, 2.0)]
    lo, hi = 0.05, 1.5
    cfgd = PEConfig{Float64}(length(FREQS), 1, 0.01, 1; marginalize_phase=true, marginalize_distance=true,
                             distance_lo=lo, distance_hi=hi, distance_power=2)
    pos = _positions([TRUTH])
    ndraw = 4000
    many = repeat(pos, 1, ndraw)
    BN.reconstruct_marginalized!(many, data, cfgd, MODEL; rng=Random.Xoshiro(3))
    # brute-force joint conditional of (dL, phase) on a grid
    cfg0 = PEConfig{Float64}(length(FREQS), 1, 0.01, 1)
    Ds = range(lo, hi; length=300); phs = range(0, 2pi; length=121)[1:end-1]
    lw = [begin q = copy(pos); q[5, 1] = D; q[7, 1] = ph; _loglike(data, cfg0, MODEL, q)[1] + 2log(D) end for D in Ds, ph in phs]
    w = exp.(lw .- maximum(lw)); w ./= sum(w)
    meanD = sum(w .* [D for D in Ds, _ in phs])
    sdD = sqrt(sum(w .* [(D - meanD)^2 for D in Ds, _ in phs]))
    cmean = sum(w .* [cis(ph) for _ in Ds, ph in phs])
    @test isapprox(sum(many[5, :]) / ndraw, meanD; atol=4 * sdD / sqrt(ndraw) + 2e-3)
    @test isapprox(sqrt(sum(abs2, many[5, :] .- meanD) / ndraw), sdD; rtol=0.1)
    @test abs(sum(cis.(many[7, :])) / ndraw - cmean) < 0.05
end

@testset "PAF2 gw_logp_and_grad vs finite differences (ndet=$(ndet), marg=$(marg), tref=$(tref), $(mode))" for
        ndet in (1, 2), marg in (false, true, :dist, :fref), tref in (0, 1), mode in (:reverse, :forward)
    ndet == 1 && tref == 1 && continue
    # Multi-detector PAF2 forward mode hits a pre-existing EnzymeRuntimeActivityError (known since
    # 2026-09-14, predates this file); the multi-detector reverse precoeffs path replaces it.
    ndet == 2 && mode == :forward && continue
    BN._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[] = (mode == :reverse)
    data = ndet == 1 ? _make_data(H1, 300.0, 1.0; model = PAF2) : [_make_data(H1, 150.0, 1.0; model = PAF2), _make_data(L1, 150.0, 2.0; model = PAF2)]
    cfg = PEConfig{Float64}(length(FREQS), 2, 0.01, 1; marginalize_phase=marg in (true, :dist, :fref),
                            time_reference=tref, marginalize_distance=marg in (:dist, :fref), distance_hi=1.5,
                            distance_power=2, time_ref_frequency=marg == :fref ? 25.0 : 0.0,
                            time_ref_offset=marg == :fref ? 1.2 : 0.0, fmax_taper=marg == :fref ? 3.0 : 0.0)
    positions = _positions([TRUTH, (36.5, 29.2, 0.05, -0.05, 0.42, 0.002, 0.55, 1.25, -0.35, 1.05, 0.65)])
    logp_and_grad! = BN.gw_logp_and_grad(data, cfg, FREQS; model=PAF2)
    grad = zeros(11, 2); logp = zeros(2)
    logp_and_grad!(grad, logp, positions, nothing)
    @test all(isfinite, logp)
    marg != false && @test all(abs.(grad[7, :]) .< 1e-8)
    marg in (:dist, :fref) && @test all(abs.(grad[5, :]) .< 1e-6)

    h = 1e-6
    fd = zeros(size(positions))
    tg = zeros(11, 2); tl = zeros(2)
    for i in eachindex(positions)
        pp = copy(positions); pp[i] += h
        pm = copy(positions); pm[i] -= h
        logp_and_grad!(tg, tl, pp, nothing); lp = sum(tl)
        logp_and_grad!(tg, tl, pm, nothing)
        fd[i] = (lp - sum(tl)) / (2h)
    end
    # Intrinsic rows (mc, q, chi1, chi2) go through SEOBNRv5's PA solve, whose finite differences carry ~1e-5..2e-4
    # relative solver noise (see test_seobnr_enzyme_e2e_vs_adjoint.jl; measured 2.2e-4 on the mc/q rows here):
    # 1e-3 there, 1e-4 for the extrinsic rows.
    for r in 1:11
        @test isapprox(grad[r, :], fd[r, :]; rtol=(r <= 4 ? 1e-3 : 1e-4), atol=1e-4)
    end
end

BN._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[] = _pmtr_old_reverse_flag
