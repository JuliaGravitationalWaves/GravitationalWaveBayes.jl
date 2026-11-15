using Test
using Random
using GravitationalWaveBayes
using GravitationalWaveforms: IMRPhenomXHM
using GravitationalWaveData: H1, L1, gmst, antenna_response_gmst, time_delay_from_geocenter_gmst
import GravitationalWaveforms

const BNh = GravitationalWaveBayes.BatchedNUTS
const KFh = GravitationalWaveBayes.KernelFused

# Likelihood options on IMRPhenomXHM (2026-10-03): the plain likelihood equals the `hphc` array API (in the XHM
# kernels' convention: f_ref = first bin, theta_jn as inclination, t_c as exp(-2 pi i f t)), distance
# marginalization equals a brute-force integral, time_reference shifts t_c by the reference detector's delay,
# time_ref_frequency by t*(f_ref) of the (2,2) mode (checked against a finite difference of its phase), phase
# marginalization is refused, the per-walker Enzyme gradient matches finite differences, and the reconstructed
# distance draws follow the conditional posterior.

const XHM = IMRPhenomXHM()
const GPSh = 1126259462.4
const FREQSh = collect(20.0:0.5:300.0)
const TRUTHh = (36.0, 22.0, 0.3, -0.1, 0.4, 0.004, 1.3, 1.2, -0.4, 0.9, 0.6)
const OTHERh = (36.5, 22.5, 0.25, -0.05, 0.42, 0.0045, 0.55, 1.25, -0.35, 1.05, 0.65)

function _xhm_strain(freqs, det, phys; geocentric = false)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = phys
    g = gmst(GPSh)
    t = geocentric ? t_c + time_delay_from_geocenter_gmst(det, ra, dec, g) : t_c
    hp, hc = GravitationalWaveforms.hphc(XHM, freqs, m1, m2, chi1, chi2, dL, theta_jn, phase; f_ref = freqs[1])
    Fp, Fc = antenna_response_gmst(det, ra, dec, psi, g)
    return (Fp .* hp .+ Fc .* hc) .* cis.(-2pi .* freqs .* t)
end

function _hpositions(phys_list)
    positions = zeros(Float64, 11, length(phys_list))
    for (w, p) in enumerate(phys_list)
        m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = p
        mc, q = BNh._m1_m2_to_mc_q(m1, m2)
        positions[:, w] .= (mc, q, chi1, chi2, dL, t_c, phase, ra, sin(dec), cos(theta_jn), psi)
    end
    return positions
end

function _hloglike(data, cfg, positions)
    physical = similar(positions)
    BNh._reparam_to_physical!(physical, positions)
    ll = zeros(size(positions, 2))
    BNh._eval_ll!(ll, physical, data isa AbstractVector ? data[1].freqs : data.freqs, cfg, data, XHM, nothing)
    return ll
end

_hlogsumexp(x) = (m = maximum(x); m + log(sum(exp.(x .- m))))
_hnoise(n, seed) = [complex(sin(1.7 * k + seed), cos(2.3 * k * seed + 0.1)) for k in 1:n]

function _hmake_data(det, snr2_target, seed; geocentric = false)
    h = _xhm_strain(FREQSh, det, TRUTHh; geocentric)
    df = FREQSh[2] - FREQSh[1]
    s = 4 * df * sum(abs2, h) / snr2_target
    return LikelihoodData(FREQSh, h .+ sqrt(s / (4 * df)) .* _hnoise(length(FREQSh), seed), fill(s, length(FREQSh)), det, GPSh)
end

_hcfg(; kw...) = with_likelihood_options(PEConfig{Float64}(length(FREQSh), 1, 0.01, 1); kw...)

@testset "IMRPhenomXHM plain likelihood (vs the hphc array API)" begin
    for ndet in (1, 2), p in (TRUTHh, OTHERh)
        data = ndet == 1 ? _hmake_data(H1, 300.0, 1.0) :
               [_hmake_data(H1, 150.0, 1.0; geocentric = true), _hmake_data(L1, 150.0, 2.0; geocentric = true)]
        dets = ndet == 1 ? [data] : data
        ref = sum(dets) do d
            h = _xhm_strain(FREQSh, d.detector, p; geocentric = ndet > 1)
            -2 * (FREQSh[2] - FREQSh[1]) * sum(abs2.(h .- d.strain) ./ d.psd)
        end
        @test isapprox(_hloglike(data, _hcfg(), _hpositions([p]))[1], ref; rtol = 1e-10)
    end
end

@testset "IMRPhenomXHM refuses phase marginalization" begin
    data = _hmake_data(H1, 300.0, 1.0)
    @test_throws ArgumentError _hloglike(data, _hcfg(; marginalize_phase = true), _hpositions([TRUTHh]))
    @test_throws ArgumentError BNh.gw_logp_and_grad(data, _hcfg(; marginalize_phase = true), FREQSh; model = XHM)(
        zeros(11, 1), zeros(1), _hpositions([TRUTHh]), nothing)
end

@testset "IMRPhenomXHM distance marginalization == brute force" begin
    data = [_hmake_data(H1, 150.0, 1.0), _hmake_data(L1, 150.0, 2.0)]
    lo, hi, dpow = 0.05, 1.5, 2
    cfgd = _hcfg(; marginalize_distance = true, distance_lo = lo, distance_hi = hi, distance_power = dpow)
    pos = _hpositions([OTHERh])
    ld = _hloglike(data, cfgd, pos)[1]
    nD = 4000
    qs = repeat(pos, 1, nD)
    Ds = [lo + (hi - lo) * (k - 0.5) / nD for k in 1:nD]
    qs[5, :] .= Ds
    lls = _hloglike(data, _hcfg(), qs) .+ dpow .* log.(Ds)
    brute = _hlogsumexp(lls) + log((hi - lo) / nD) - log((hi^(dpow + 1) - lo^(dpow + 1)) / (dpow + 1))
    @test isapprox(ld, brute; atol = 1e-4)
    # the marginalized value does not depend on the walker's own distance
    q = copy(pos); q[5, 1] = 1.1
    @test isapprox(_hloglike(data, cfgd, q)[1], ld; atol = 1e-8)
end

@testset "IMRPhenomXHM time_reference and time_ref_frequency" begin
    data = [_hmake_data(H1, 150.0, 1.0), _hmake_data(L1, 150.0, 2.0)]
    g = gmst(GPSh)
    pos = _hpositions([TRUTHh])
    base = _hloglike(data, _hcfg(), pos)[1]
    for (k, det) in ((1, H1), (2, L1))
        q = copy(pos); q[6, 1] += time_delay_from_geocenter_gmst(det, TRUTHh[8], TRUTHh[9], g)
        @test isapprox(_hloglike(data, _hcfg(; time_reference = k), q)[1], base; rtol = 1e-12)
    end
    m1, m2, chi1, chi2 = TRUTHh[1:4]
    sys, = KFh._xhm_setup(m1, m2, chi1, chi2, 0.4, 0.7, 20.0)
    # (2,2) contribution to h+ alone: its stationary-phase time is t* = -psi'(f) / 2 pi
    function hp22(f)
        Mf = sys.coeffs22.M * sys.coeffs22.GMsun_over_c3 * f
        a = GravitationalWaveforms._xas_ampl_scalar(sys.amplcoeffs22, f)
        ph = GravitationalWaveforms._xhm22_phase_full(sys.coeffs22, sys.timeshift, sys.phiref22, Mf)
        h22 = GravitationalWaveforms._XP_HAS_MODE_NORMALIZATION * Complex(a * cos(ph), a * sin(ph))
        return GravitationalWaveforms._xhm_add_mode!(zero(ComplexF64), zero(ComplexF64), 2, 2, h22, 0.4, pi / 2)[1]
    end
    for (fref, off) in ((30.0, 0.0), (45.0, 0.37), (200.0, 0.0))
        tstar = time_at_frequency(XHM, m1, m2, chi1, chi2, fref)
        e = 1e-4
        @test isapprox(tstar, -angle(hp22(fref + e) / hp22(fref - e)) / (2e) / 2pi; atol = 1e-7)
        q = copy(pos); q[6, 1] += tstar + off
        @test isapprox(_hloglike(data, _hcfg(; time_ref_frequency = fref, time_ref_offset = off), q)[1], base; rtol = 1e-10)
    end
    @test_throws ArgumentError _hloglike(data, _hcfg(; fmax_taper = 2.0), pos)
end

@testset "IMRPhenomXHM gw_logp_and_grad vs finite differences (ndet=$ndet, opts=$opts)" for ndet in (1, 2), opts in (:plain, :all)
    data = ndet == 1 ? _hmake_data(H1, 300.0, 1.0) : [_hmake_data(H1, 150.0, 1.0), _hmake_data(L1, 150.0, 2.0)]
    kw = opts == :plain ? (;) :
         (; marginalize_distance = true, distance_lo = 0.05, distance_hi = 1.5, distance_power = 2,
            time_reference = ndet, time_ref_frequency = 35.0, time_ref_offset = 0.3)
    cfg = with_likelihood_options(PEConfig{Float64}(length(FREQSh), 2, 0.01, 1); kw...)
    positions = _hpositions([TRUTHh, OTHERh])
    lg! = BNh.gw_logp_and_grad(data, cfg, FREQSh; model = XHM)
    grad = zeros(11, 2); logp = zeros(2)
    lg!(grad, logp, positions, nothing)
    @test all(isfinite, logp)
    @test isapprox(logp, _hloglike(data, cfg, positions) .+ BNh._call_log_prior_vec(positions); rtol = 1e-10)
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

@testset "IMRPhenomXHM reconstructed distance follows the conditional posterior" begin
    data = [_hmake_data(H1, 150.0, 1.0), _hmake_data(L1, 150.0, 2.0)]
    lo, hi, dpow = 0.05, 1.5, 2
    cfgd = _hcfg(; marginalize_distance = true, distance_lo = lo, distance_hi = hi, distance_power = dpow)
    n = 2000
    pos = repeat(_hpositions([OTHERh]), 1, n)
    BNh.reconstruct_marginalized!(pos, data, cfgd, XHM; rng = Random.Xoshiro(7))
    draws = sort(pos[5, :])
    @test all(lo .<= draws .<= hi)
    # brute-force conditional CDF on a grid
    nD = 2000
    Ds = [lo + (hi - lo) * (k - 0.5) / nD for k in 1:nD]
    qs = repeat(_hpositions([OTHERh]), 1, nD); qs[5, :] .= Ds
    lw = _hloglike(data, _hcfg(), qs) .+ dpow .* log.(Ds)
    cdf = cumsum(exp.(lw .- maximum(lw))); cdf ./= cdf[end]
    ecdf(x) = searchsortedlast(draws, x) / n
    ks = maximum(abs(ecdf(Ds[k]) - cdf[k]) for k in 1:nD)
    @test ks < 0.05      # ~1.36 / sqrt(2000) = 0.030 at 95%
end

@testset "IMRPhenomXHM relative binning with the likelihood options" begin
    data = [_hmake_data(H1, 150.0, 1.0), _hmake_data(L1, 150.0, 2.0)]
    kw = (; marginalize_distance = true, distance_lo = 0.05, distance_hi = 1.5, distance_power = 2,
          time_reference = 1, time_ref_frequency = 35.0, time_ref_offset = 0.3)
    cfg = with_likelihood_options(PEConfig{Float64}(length(FREQSh), 4, 0.01, 1); kw...)
    ref = _hpositions([TRUTHh])[:, 1]
    rb = BNh.build_xhm_relative_binning(data, cfg, ref; nbins = 60)
    @test rb isa BNh.XHMRelBin
    # intrinsic + extrinsic offsets, and pure phase / inclination changes (per-mode ratios: exact for those)
    near = hcat(ref, ref .+ [0.05, 0.01, 0.02, -0.02, 0.01, 2e-4, 0.1, 0.01, 0.01, 0.02, 0.05],
                ref .+ [-0.08, -0.01, -0.03, 0.03, -0.02, -3e-4, -0.2, -0.02, 0.01, -0.03, -0.05],
                ref .+ [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pi, 0.0, 0.0, -0.6, 0.0])
    physical = similar(near); BNh._reparam_to_physical!(physical, near)
    llrb = zeros(4); lld = zeros(4)
    BNh._eval_ll!(llrb, physical, FREQSh, cfg, data, XHM, rb)
    BNh._eval_ll!(lld, physical, FREQSh, cfg, data, XHM, nothing)
    @test isapprox(llrb[1], lld[1]; atol = 1e-8)              # r = 1 at the fiducial point: exact
    @test isapprox(llrb[4], lld[4]; atol = 1e-8)              # phase / inclination only: each r_k constant, exact
    @test maximum(abs.(llrb .- lld)) < 2e-3
    # gradient of the relative-binning likelihood vs finite differences
    lg! = BNh.gw_logp_and_grad(data, cfg, FREQSh; model = XHM, relative_binning = rb)
    grad = zeros(11, 4); logp = zeros(4)
    lg!(grad, logp, near, nothing)
    @test isapprox(logp, llrb .+ BNh._call_log_prior_vec(near); rtol = 1e-10)
    steps = [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-8, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]
    tg = zeros(11, 4); tl = zeros(4)
    for r in 1:11
        pp = copy(near); pp[r, :] .+= steps[r]
        pm = copy(near); pm[r, :] .-= steps[r]
        lg!(tg, tl, pp, nothing); lp = copy(tl)
        lg!(tg, tl, pm, nothing)
        fd = (lp .- tl) ./ (2 * steps[r])
        ok = isapprox(grad[r, :], fd; rtol = 1e-4, atol = 1e-3)
        ok || @info "RB gradient mismatch" r grad[r, :] fd
        @test ok
    end
end

@testset "IMRPhenomXHM bin-parallel likelihood + gradient (CPU backend, setup = $setup)" for setup in (:host, :device)
    # `xhm_binparallel_logp_and_grad` (2026-10-04): the device path's factorization (per-mode c_k, flat setup with
    # forward tangents, pow-free modes, first-order surrogate) on KernelAbstractions.CPU() -- equal to the dense
    # likelihood and its finite differences.
    data = [_hmake_data(H1, 150.0, 1.0), _hmake_data(L1, 150.0, 2.0)]
    kw = (; marginalize_distance = true, distance_lo = 0.05, distance_hi = 1.5, distance_power = 2,
          time_reference = 1, time_ref_frequency = 35.0, time_ref_offset = 0.3)
    cfg = with_likelihood_options(PEConfig{Float64}(length(FREQSh), 2, 0.01, 1); kw...)
    positions = _hpositions([TRUTHh, OTHERh])
    f! = BNh.xhm_binparallel_logp_and_grad(data, cfg, FREQSh; backend = BNh.KernelAbstractions.CPU(), nwalkers = 2,
                                          setup)
    grad = zeros(11, 2); logp = zeros(2)
    f!(grad, logp, positions, nothing)
    @test isapprox(logp, _hloglike(data, cfg, positions) .+ BNh._call_log_prior_vec(positions); rtol = 1e-12)
    steps = [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-8, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]
    for r in 1:11
        pp = copy(positions); pp[r, :] .+= steps[r]
        pm = copy(positions); pm[r, :] .-= steps[r]
        fd = (_hloglike(data, cfg, pp) .- _hloglike(data, cfg, pm)) ./ (2 * steps[r])
        ok = isapprox(grad[r, :], fd; rtol = 1e-4, atol = 1e-3)
        ok || @info "bin-parallel gradient mismatch" r grad[r, :] fd
        @test ok
    end
end
