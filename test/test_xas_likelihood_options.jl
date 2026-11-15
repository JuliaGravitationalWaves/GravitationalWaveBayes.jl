using Test
using Random
using GravitationalWaveBayes
using GravitationalWaveforms: IMRPhenomXAS, Ampl, Phi
using GravitationalWaveData: H1, L1, gmst, geometry, time_delay_from_geocenter_gmst
using GravitationalWaveforms: GMsun_over_c3, _xas_phi_coeffs

const BNx = GravitationalWaveBayes.BatchedNUTS
const KFx = GravitationalWaveBayes.KernelFused

# Likelihood options on the IMRPhenomXAS kernels: the plain likelihood is unchanged
# (against the array API `Ampl`/`Phi`), phase / distance marginalization equal brute-force integrals,
# time_reference shifts t_c by the reference detector's delay, time_ref_frequency shifts it by
# t*(f_ref) = -Phi'(f_ref) / 2 pi, and the Enzyme gradient through all of them matches finite differences.

const XAS = IMRPhenomXAS()
const GPSx = 1126259462.4
const FREQSx = collect(20.0:0.5:400.0)
const TRUTHx = (36.0, 29.0, 0.1, -0.1, 0.4, 0.004, 1.3, 1.2, -0.4, 0.7, 0.6)
const OTHERx = (36.5, 29.2, 0.05, -0.05, 0.42, 0.0045, 0.55, 1.25, -0.35, 1.05, 0.65)

# Independent host-side strain from the array API, in the kernel convention (see KernelFused
# `_PHASE_CONVENTION`: XAS family, h = combo A exp(i (Phi - 2 pi f t - phiRef)) via Phi's t0/phiRef).
function _xas_strain(freqs, det, phys; geocentric = false)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = phys
    mc, eta = KFx._chirp_mass_eta(m1, m2)
    g = gmst(GPSx)
    t = geocentric ? t_c + time_delay_from_geocenter_gmst(det, ra, dec, g) : t_c
    phiRef, combo = KFx._extrinsic_setup_phase_only(phase, ra, dec, theta_jn, psi, geometry(det), g)
    M_sec = mc / eta^0.6 * GMsun_over_c3
    A = Ampl(XAS, freqs, mc, eta, chi1, chi2, dL)
    P = Phi(XAS, freqs, mc, eta, chi1, chi2; t0 = 2pi * t / M_sec, phiRef = phiRef)
    return combo .* A .* cis.(P)
end

function _xpositions(phys_list)
    positions = zeros(Float64, 11, length(phys_list))
    for (w, p) in enumerate(phys_list)
        m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = p
        mc, q = BNx._m1_m2_to_mc_q(m1, m2)
        positions[:, w] .= (mc, q, chi1, chi2, dL, t_c, phase, ra, sin(dec), cos(theta_jn), psi)
    end
    return positions
end

function _xloglike(data, cfg, positions)
    physical = similar(positions)
    BNx._reparam_to_physical!(physical, positions)
    ll = zeros(size(positions, 2))
    BNx._eval_ll!(ll, physical, data isa AbstractVector ? data[1].freqs : data.freqs, cfg, data, XAS, nothing)
    return ll
end

_xlogsumexp(x) = (m = maximum(x); m + log(sum(exp.(x .- m))))
_xnoise(n, seed) = [complex(sin(1.7 * k + seed), cos(2.3 * k * seed + 0.1)) for k in 1:n]

function _xmake_data(det, snr2_target, seed; geocentric = false)
    h = _xas_strain(FREQSx, det, TRUTHx; geocentric)
    df = FREQSx[2] - FREQSx[1]
    s = 4 * df * sum(abs2, h) / snr2_target
    return LikelihoodData(FREQSx, h .+ sqrt(s / (4 * df)) .* _xnoise(length(FREQSx), seed), fill(s, length(FREQSx)), det, GPSx)
end

_xcfg(; kw...) = with_likelihood_options(PEConfig{Float64}(length(FREQSx), 1, 0.01, 1); kw...)

@testset "IMRPhenomXAS branching evaluator == _xas_phi_scalar / _xas_ampl_scalar" begin
    for (m1, m2, chi1, chi2) in ((36.0, 29.0, 0.1, -0.1), (12.0, 4.0, 0.8, -0.5), (80.0, 70.0, -0.6, 0.9))
        mc, eta = KFx._chirp_mass_eta(m1, m2)
        pc = _xas_phi_coeffs(XAS, mc, eta, chi1, chi2)
        ac = GravitationalWaveBayes.KernelFused._xas_ampl_coeffs(XAS, mc, eta, chi1, chi2, 0.7)
        for f in range(5.0, 4000.0; length = 4001)
            ph, a = KFx._xas_phase_amp(pc, ac, f)
            @test ph === GravitationalWaveBayes.KernelFused._xas_phi_scalar(pc, f)
            @test a === GravitationalWaveBayes.KernelFused._xas_ampl_scalar(ac, f)
        end
    end
end

@testset "IMRPhenomXAS plain likelihood unchanged (vs the Ampl/Phi array API)" begin
    for ndet in (1, 2), p in (TRUTHx, OTHERx)
        data = ndet == 1 ? _xmake_data(H1, 300.0, 1.0) :
               [_xmake_data(H1, 150.0, 1.0; geocentric = true), _xmake_data(L1, 150.0, 2.0; geocentric = true)]
        dets = ndet == 1 ? [data] : data
        ref = sum(dets) do d
            h = _xas_strain(FREQSx, d.detector, p; geocentric = ndet > 1)
            -2 * (FREQSx[2] - FREQSx[1]) * sum(abs2.(h .- d.strain) ./ d.psd)
        end
        @test isapprox(_xloglike(data, _xcfg(), _xpositions([p]))[1], ref; rtol = 1e-10)
    end
end

@testset "IMRPhenomXAS phase marginalization == phase average (ndet=$ndet)" for ndet in (1, 2)
    data = ndet == 1 ? _xmake_data(H1, 300.0, 1.0) : [_xmake_data(H1, 150.0, 1.0), _xmake_data(L1, 150.0, 2.0)]
    for p in (TRUTHx, OTHERx)
        pos = _xpositions([p])
        lm = _xloglike(data, _xcfg(; marginalize_phase = true), pos)[1]
        nph = 512
        lls = map(0:nph-1) do k
            q = copy(pos); q[7, 1] = 2pi * k / nph
            _xloglike(data, _xcfg(), q)[1]
        end
        @test isapprox(lm, _xlogsumexp(lls) - log(nph); atol = 2e-6)
        q = copy(pos); q[7, 1] = 4.0
        @test isapprox(_xloglike(data, _xcfg(; marginalize_phase = true), q)[1], lm; atol = 1e-9)
    end
end

@testset "IMRPhenomXAS distance marginalization == brute force (margphase=$mp)" for mp in (false, true)
    data = [_xmake_data(H1, 150.0, 1.0), _xmake_data(L1, 150.0, 2.0)]
    lo, hi, dpow = 0.05, 1.5, 2
    cfgp = _xcfg(; marginalize_phase = mp)
    cfgd = _xcfg(; marginalize_phase = mp, marginalize_distance = true, distance_lo = lo, distance_hi = hi,
                 distance_power = dpow)
    pos = _xpositions([OTHERx])
    ld = _xloglike(data, cfgd, pos)[1]
    nD = 20000
    lls = map(1:nD) do k
        D = lo + (hi - lo) * (k - 0.5) / nD
        q = copy(pos); q[5, 1] = D
        _xloglike(data, cfgp, q)[1] + dpow * log(D)
    end
    brute = _xlogsumexp(lls) + log((hi - lo) / nD) - log((hi^(dpow + 1) - lo^(dpow + 1)) / (dpow + 1))
    @test isapprox(ld, brute; atol = 1e-4)
end

@testset "IMRPhenomXAS time_reference and time_ref_frequency" begin
    data = [_xmake_data(H1, 150.0, 1.0), _xmake_data(L1, 150.0, 2.0)]
    g = gmst(GPSx)
    pos = _xpositions([TRUTHx])
    base = _xloglike(data, _xcfg(), pos)[1]
    for (k, det) in ((1, H1), (2, L1))
        q = copy(pos); q[6, 1] += time_delay_from_geocenter_gmst(det, TRUTHx[8], TRUTHx[9], g)
        @test isapprox(_xloglike(data, _xcfg(; time_reference = k), q)[1], base; rtol = 1e-12)
    end
    for (fref, off) in ((30.0, 0.0), (45.0, 0.37), (300.0, 0.0))   # 300 Hz: past the inspiral join
        m1, m2, chi1, chi2 = TRUTHx[1:4]
        tstar = time_at_frequency(XAS, m1, m2, chi1, chi2, fref)
        # t* = -Phi'(f) / 2 pi of the intrinsic phase (central difference of the array API)
        mc, eta = KFx._chirp_mass_eta(m1, m2)
        dfd = 1e-3
        dphi = (Phi(XAS, fref + dfd, mc, eta, chi1, chi2) - Phi(XAS, fref - dfd, mc, eta, chi1, chi2)) / (2dfd)
        @test isapprox(tstar, -dphi / 2pi; atol = 1e-7)
        q = copy(pos); q[6, 1] += tstar + off
        @test isapprox(_xloglike(data, _xcfg(; time_ref_frequency = fref, time_ref_offset = off), q)[1], base; rtol = 1e-10)
    end
    # the inspiral reaches 30 Hz before 45 Hz, a few tenths of a second before the late-inspiral frequencies
    t30, t45 = (time_at_frequency(XAS, TRUTHx[1:4]..., f) for f in (30.0, 45.0))
    @test t30 < t45 && 0.05 < t45 - t30 < 0.5
    @test_throws ArgumentError _xloglike(data, _xcfg(; fmax_taper = 2.0), pos)
end

@testset "IMRPhenomXAS gw_logp_and_grad vs finite differences (ndet=$ndet, opts=$opts)" for
        ndet in (1, 2), opts in (:plain, :phase, :all)
    data = ndet == 1 ? _xmake_data(H1, 300.0, 1.0) : [_xmake_data(H1, 150.0, 1.0), _xmake_data(L1, 150.0, 2.0)]
    kw = opts == :plain ? (;) : opts == :phase ? (; marginalize_phase = true) :
         (; marginalize_phase = true, marginalize_distance = true, distance_lo = 0.05, distance_hi = 1.5,
            distance_power = 2, time_reference = ndet, time_ref_frequency = 35.0, time_ref_offset = 0.3)
    cfg = with_likelihood_options(PEConfig{Float64}(length(FREQSx), 2, 0.01, 1); kw...)
    positions = _xpositions([TRUTHx, OTHERx])
    lg! = BNx.gw_logp_and_grad(data, cfg, FREQSx; model = XAS)
    grad = zeros(11, 2); logp = zeros(2)
    lg!(grad, logp, positions, nothing)
    @test all(isfinite, logp)
    # the per-walker (scalar Enzyme) primal equals the kernel likelihood
    @test isapprox(logp, _xloglike(data, cfg, positions) .+ BNx._call_log_prior_vec(positions); rtol = 1e-10)
    opts != :plain && @test all(abs.(grad[7, :]) .< 1e-8)
    opts == :all && @test all(abs.(grad[5, :]) .< 1e-6)
    steps = [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-8, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]
    fd = zeros(size(positions)); tg = zeros(11, 2); tl = zeros(2)
    for i in eachindex(positions)
        h = steps[(i - 1) % 11 + 1]
        pp = copy(positions); pp[i] += h
        pm = copy(positions); pm[i] -= h
        lg!(tg, tl, pp, nothing); lp = sum(tl)
        lg!(tg, tl, pm, nothing)
        fd[i] = (lp - sum(tl)) / (2h)
    end
    for r in 1:11
        ok = isapprox(grad[r, :], fd[r, :]; rtol = 1e-4, atol = 1e-3)
        ok || @info "gradient mismatch" r grad[r, :] fd[r, :]
        @test ok
    end
end
