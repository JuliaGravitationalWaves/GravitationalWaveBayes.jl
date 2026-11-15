using Test
using Random
using GravitationalWaveBayes
using GravitationalWaveforms: IMRPhenomD
using GravitationalWaveData: H1, L1, gmst, geometry, time_delay_from_geocenter_gmst
import KernelAbstractions

const BNd = GravitationalWaveBayes.BatchedNUTS
const KFd = GravitationalWaveBayes.KernelFused

# Likelihood options on IMRPhenomD (2026-10-04): with no options the value equals the original fused PhenomD kernels;
# phase and distance marginalization equal brute-force integrals; time_reference shifts t_c by the reference
# detector's delay and time_ref_frequency by t*(f_ref) (checked against a finite difference of the waveform's phase);
# the per-walker Enzyme gradient matches finite differences.

const PD = IMRPhenomD()
const GPSd = 1126259462.4
const FREQSd = collect(20.0:0.5:300.0)
const TRUTHd = (36.0, 29.0, 0.3, -0.1, 0.4, 0.004, 1.3, 1.2, -0.4, 0.9, 0.6)
const OTHERd = (36.5, 28.5, 0.25, -0.05, 0.42, 0.0045, 0.55, 1.25, -0.35, 1.05, 0.65)

function _pd_strain(freqs, det, phys; geocentric = false)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = phys
    g = gmst(GPSd)
    t = geocentric ? t_c + time_delay_from_geocenter_gmst(det, ra, dec, g) : t_c
    mc, eta = KFd._chirp_mass_eta(m1, m2)
    pc = KFd._phi_coeffs(PD, mc, eta, chi1, chi2)
    ac = KFd._ampl_coeffs(PD, mc, eta, chi1, chi2, dL)
    _, combo = KFd._extrinsic_setup_phase_only(phase, ra, dec, theta_jn, psi, geometry(det), g)
    return [KFd._pd_h(pc, ac, combo, t, phase, f) for f in freqs]
end

function _dpositions(phys_list)
    positions = zeros(Float64, 11, length(phys_list))
    for (w, p) in enumerate(phys_list)
        m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = p
        mc, q = BNd._m1_m2_to_mc_q(m1, m2)
        positions[:, w] .= (mc, q, chi1, chi2, dL, t_c, phase, ra, sin(dec), cos(theta_jn), psi)
    end
    return positions
end

function _dloglike(data, cfg, positions)
    physical = similar(positions)
    BNd._reparam_to_physical!(physical, positions)
    ll = zeros(size(positions, 2))
    BNd._eval_ll!(ll, physical, data isa AbstractVector ? data[1].freqs : data.freqs, cfg, data, PD, nothing)
    return ll
end

_dlogsumexp(x) = (m = maximum(x); m + log(sum(exp.(x .- m))))
_dnoise(n, seed) = [complex(sin(1.7 * k + seed), cos(2.3 * k * seed + 0.1)) for k in 1:n]

function _dmake_data(det, snr2_target, seed; geocentric = false)
    h = _pd_strain(FREQSd, det, TRUTHd; geocentric)
    df = FREQSd[2] - FREQSd[1]
    s = 4 * df * sum(abs2, h) / snr2_target
    return LikelihoodData(FREQSd, h .+ sqrt(s / (4 * df)) .* _dnoise(length(FREQSd), seed), fill(s, length(FREQSd)), det, GPSd)
end

_dcfg(; kw...) = with_likelihood_options(PEConfig{Float64}(length(FREQSd), 1, 0.01, 1); kw...)

@testset "IMRPhenomD plain likelihood == the original fused kernels" begin
    cpu = KernelAbstractions.CPU()
    for p in (TRUTHd, OTHERd)
        physical = reshape(collect(Float64, p), 11, 1)
        # single detector
        data = _dmake_data(H1, 300.0, 1.0)
        ref = zeros(1)
        rf = KFd._geometry_array((data,), cpu)
        KFd._fused_waveform_likelihood_kernel_phenomd!(cpu)(ref, physical, data.freqs, data.strain, data.psd, rf,
            [gmst(GPSd)], PD; ndrange = 1, workgroupsize = 1)
        @test isapprox(_dloglike(data, _dcfg(), _dpositions([p]))[1], ref[1]; rtol = 1e-10)
        # two detectors (geocentric t_c)
        dd = [_dmake_data(H1, 150.0, 1.0; geocentric = true), _dmake_data(L1, 150.0, 2.0; geocentric = true)]
        refm = zeros(1)
        KFd._fused_waveform_likelihood_kernel_phenomd_multi!(cpu)(refm, physical, Tuple(d.freqs for d in dd),
            Tuple(d.strain for d in dd), Tuple(d.psd for d in dd), Tuple(geometry(d.detector) for d in dd), gmst(GPSd), PD; ndrange = 1, workgroupsize = 1)
        @test isapprox(_dloglike(dd, _dcfg(), _dpositions([p]))[1], refm[1]; rtol = 1e-10)
    end
end

@testset "IMRPhenomD phase and distance marginalization == brute force" begin
    data = [_dmake_data(H1, 150.0, 1.0), _dmake_data(L1, 150.0, 2.0)]
    pos = _dpositions([OTHERd])
    # phase
    nP = 2000
    qs = repeat(pos, 1, nP); qs[7, :] .= [2pi * (k - 0.5) / nP for k in 1:nP]
    brute = _dlogsumexp(_dloglike(data, _dcfg(), qs)) - log(nP)
    @test isapprox(_dloglike(data, _dcfg(; marginalize_phase = true), pos)[1], brute; atol = 1e-6)
    # distance (phase sampled)
    lo, hi, dpow = 0.05, 1.5, 2
    cfgd = _dcfg(; marginalize_distance = true, distance_lo = lo, distance_hi = hi, distance_power = dpow)
    nD = 4000
    Ds = [lo + (hi - lo) * (k - 0.5) / nD for k in 1:nD]
    qd = repeat(pos, 1, nD); qd[5, :] .= Ds
    bruted = _dlogsumexp(_dloglike(data, _dcfg(), qd) .+ dpow .* log.(Ds)) + log((hi - lo) / nD) -
             log((hi^(dpow + 1) - lo^(dpow + 1)) / (dpow + 1))
    @test isapprox(_dloglike(data, cfgd, pos)[1], bruted; atol = 1e-4)
end

@testset "IMRPhenomD time_reference and time_ref_frequency" begin
    data = [_dmake_data(H1, 150.0, 1.0), _dmake_data(L1, 150.0, 2.0)]
    g = gmst(GPSd)
    pos = _dpositions([TRUTHd])
    base = _dloglike(data, _dcfg(), pos)[1]
    for (k, det) in ((1, H1), (2, L1))
        q = copy(pos); q[6, 1] += time_delay_from_geocenter_gmst(det, TRUTHd[8], TRUTHd[9], g)
        @test isapprox(_dloglike(data, _dcfg(; time_reference = k), q)[1], base; rtol = 1e-12)
    end
    m1, m2, chi1, chi2 = TRUTHd[1:4]
    mc, eta = KFd._chirp_mass_eta(m1, m2)
    pc = KFd._phi_coeffs(PD, mc, eta, chi1, chi2)
    ac = KFd._ampl_coeffs(PD, mc, eta, chi1, chi2, 0.4)
    h0(f) = KFd._pd_h(pc, ac, 1.0 + 0.0im, 0.0, 0.0, f)
    for (fref, off) in ((30.0, 0.0), (45.0, 0.37), (200.0, 0.0))     # inspiral, inspiral, merger-ringdown regions
        tstar = time_at_frequency(PD, m1, m2, chi1, chi2, fref)
        e = 1e-4
        @test isapprox(tstar, -angle(h0(fref + e) / h0(fref - e)) / (2e) / 2pi; atol = 1e-7)
        q = copy(pos); q[6, 1] += tstar + off
        @test isapprox(_dloglike(data, _dcfg(; time_ref_frequency = fref, time_ref_offset = off), q)[1], base; rtol = 1e-10)
    end
end

@testset "IMRPhenomD gw_logp_and_grad vs finite differences (ndet=$ndet, opts=$opts)" for ndet in (1, 2), opts in (:plain, :all)
    data = ndet == 1 ? _dmake_data(H1, 300.0, 1.0) : [_dmake_data(H1, 150.0, 1.0), _dmake_data(L1, 150.0, 2.0)]
    kw = opts == :plain ? (;) :
         (; marginalize_phase = true, marginalize_distance = true, distance_lo = 0.05, distance_hi = 1.5,
            distance_power = 2, time_reference = ndet, time_ref_frequency = 35.0, time_ref_offset = 0.3)
    cfg = with_likelihood_options(PEConfig{Float64}(length(FREQSd), 2, 0.01, 1); kw...)
    positions = _dpositions([TRUTHd, OTHERd])
    lg! = BNd.gw_logp_and_grad(data, cfg, FREQSd; model = PD)
    grad = zeros(11, 2); logp = zeros(2)
    lg!(grad, logp, positions, nothing)
    @test isapprox(logp, _dloglike(data, cfg, positions) .+ BNd._call_log_prior_vec(positions); rtol = 1e-10)
    steps = [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-8, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]
    for r in 1:11
        pp = copy(positions); pp[r, :] .+= steps[r]
        pm = copy(positions); pm[r, :] .-= steps[r]
        fd = (_dloglike(data, cfg, pp) .- _dloglike(data, cfg, pm)) ./ (2 * steps[r])
        ok = isapprox(grad[r, :], fd; rtol = 1e-4, atol = 1e-3)
        ok || @info "PhenomD gradient mismatch" r grad[r, :] fd
        @test ok
    end
end
