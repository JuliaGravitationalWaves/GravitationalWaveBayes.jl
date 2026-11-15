using Test
using Random
using GravitationalWaveBayes
using GravitationalWaveforms: IMRPhenomXP, IMRPhenomXPHM, IMRPhenomXAS
using GravitationalWaveData: H1, L1, gmst, antenna_response_gmst, geometry, time_delay_from_geocenter_gmst
import KernelAbstractions

const BNp = GravitationalWaveBayes.BatchedNUTS
const KFp = GravitationalWaveBayes.KernelFused

# Likelihood options on the precessing models IMRPhenomXP / IMRPhenomXPHM: with no options the value
# equals the original fused kernels; phase marginalization is refused; distance marginalization equals a brute-force
# integral; time_reference shifts t_c by the reference detector's delay and time_ref_frequency by the co-precessing
# (2,2) mode's t*(f_ref); the per-walker Enzyme gradient (15 rows) matches finite differences.

const GPSp = 1126259462.4
const FREQSp = collect(20.0:1.0:300.0)
# (m1, m2, chi1z, chi2z, dL, t_c, phase, ra, dec, theta_jn, psi, chi1x, chi1y, chi2x, chi2y)
const TRUTHp = (36.0, 29.0, 0.3, -0.1, 0.4, 0.004, 1.3, 1.2, -0.4, 0.9, 0.6, 0.35, -0.2, 0.1, 0.25)
const OTHERp = (36.5, 28.5, 0.25, -0.05, 0.42, 0.0045, 0.55, 1.25, -0.35, 1.05, 0.65, 0.3, -0.25, 0.15, 0.2)

_pphys(p) = reshape(collect(Float64, p), 15, 1)
function _ppositions(phys_list)
    positions = zeros(Float64, 15, length(phys_list))
    for (w, p) in enumerate(phys_list)
        m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = p[1:11]
        mc, q = BNp._m1_m2_to_mc_q(m1, m2)
        positions[:, w] .= (mc, q, chi1, chi2, dL, t_c, phase, ra, sin(dec), cos(theta_jn), psi, p[12:15]...)
    end
    return positions
end
function _ploglike(model, data, cfg, positions)
    physical = similar(positions)
    BNp._reparam_to_physical!(physical, positions)
    ll = zeros(size(positions, 2))
    BNp._eval_ll!(ll, physical, data isa AbstractVector ? data[1].freqs : data.freqs, cfg, data, model, nothing)
    return ll
end
# Detector strain from the fused kernels' waveform: the per-detector sums code evaluated at one frequency each.
function _pstrain(model, det, p; geocentric = false)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi, c1x, c1y, c2x, c2y = p
    g = gmst(GPSp)
    lik = KFp._lik_opts(_pcfg())
    st = KFp._xprec_state(model, m1, m2, chi1, chi2, dL, phase, theta_jn, c1x, c1y, c2x, c2y, FREQSp[1], lik)
    t = geocentric ? t_c + time_delay_from_geocenter_gmst(det, ra, dec, g) : t_c
    Fp, Fc = antenna_response_gmst(det, ra, dec, psi, g)
    # h at f from the sums with unit PSD and zero data: hh = |h|^2, and <d,h> with d = 1 gives h itself
    return [begin
                _, _, zr, zi = KFp._xprec_det_sums(model, st, Fp, Fc, t, [f], [1.0 + 0.0im], [1.0])
                complex(zr, zi)
            end for f in FREQSp]
end
_pnoise(n, seed) = [complex(sin(1.7 * k + seed), cos(2.3 * k * seed + 0.1)) for k in 1:n]
function _pmake_data(model, det, snr2_target, seed; geocentric = false)
    h = _pstrain(model, det, TRUTHp; geocentric)
    df = FREQSp[2] - FREQSp[1]
    s = 4 * df * sum(abs2, h) / snr2_target
    return LikelihoodData(FREQSp, h .+ sqrt(s / (4 * df)) .* _pnoise(length(FREQSp), seed), fill(s, length(FREQSp)), det, GPSp)
end
_pcfg(; nw = 1, kw...) = with_likelihood_options(PEConfig{Float64}(length(FREQSp), nw, 0.01, 1); kw...)

@testset "$(typeof(model)) plain likelihood == the original fused kernels" for model in (IMRPhenomXP(), IMRPhenomXPHM())
    cpu = KernelAbstractions.CPU()
    single! = model isa IMRPhenomXP ? KFp._fused_waveform_likelihood_kernel_phenomxp!(cpu) :
                                      KFp._fused_waveform_likelihood_kernel_phenomxphm!(cpu)
    multi! = model isa IMRPhenomXP ? KFp._fused_waveform_likelihood_kernel_phenomxp_multi!(cpu) :
                                     KFp._fused_waveform_likelihood_kernel_phenomxphm_multi!(cpu)
    for p in (TRUTHp, OTHERp)
        data = _pmake_data(model, H1, 300.0, 1.0)
        ref = zeros(1)
        single!(ref, _pphys(p), data.freqs, data.strain, data.psd, KFp._geometry_array((data,), cpu),
                [gmst(GPSp)], model; ndrange = 1, workgroupsize = 1)
        @test isapprox(_ploglike(model, data, _pcfg(), _ppositions([p]))[1], ref[1]; rtol = 1e-10)
        dd = [_pmake_data(model, H1, 150.0, 1.0; geocentric = true), _pmake_data(model, L1, 150.0, 2.0; geocentric = true)]
        refm = zeros(1)
        multi!(refm, _pphys(p), Tuple(d.freqs for d in dd), Tuple(d.strain for d in dd), Tuple(d.psd for d in dd),
               Tuple(geometry(d.detector) for d in dd), gmst(GPSp), model;
               ndrange = 1, workgroupsize = 1)
        @test isapprox(_ploglike(model, dd, _pcfg(), _ppositions([p]))[1], refm[1]; rtol = 1e-10)
    end
end

@testset "$(typeof(model)) options: phase refused, distance marginalization, time frame" for model in (IMRPhenomXP(), IMRPhenomXPHM())
    data = [_pmake_data(model, H1, 150.0, 1.0), _pmake_data(model, L1, 150.0, 2.0)]
    pos = _ppositions([OTHERp])
    @test_throws ArgumentError _ploglike(model, data, _pcfg(; marginalize_phase = true), pos)
    lo, hi, dpow = 0.05, 1.5, 2
    cfgd = _pcfg(; marginalize_distance = true, distance_lo = lo, distance_hi = hi, distance_power = dpow)
    nD = 2000
    Ds = [lo + (hi - lo) * (k - 0.5) / nD for k in 1:nD]
    qd = repeat(pos, 1, nD); qd[5, :] .= Ds
    lls = _ploglike(model, data, _pcfg(; nw = nD), qd) .+ dpow .* log.(Ds)
    m = maximum(lls)
    brute = m + log(sum(exp.(lls .- m))) + log((hi - lo) / nD) - log((hi^(dpow + 1) - lo^(dpow + 1)) / (dpow + 1))
    @test isapprox(_ploglike(model, data, cfgd, pos)[1], brute; atol = 1e-4)
    # time frame
    g = gmst(GPSp)
    base = _ploglike(model, data, _pcfg(), pos)[1]
    for (k, det) in ((1, H1), (2, L1))
        q = copy(pos); q[6, 1] += time_delay_from_geocenter_gmst(det, OTHERp[8], OTHERp[9], g)
        @test isapprox(_ploglike(model, data, _pcfg(; time_reference = k), q)[1], base; rtol = 1e-12)
    end
    for (fref, off) in ((30.0, 0.0), (45.0, 0.37))
        p = OTHERp
        st = KFp._xprec_state(model, p[1], p[2], p[3], p[4], p[5], p[7], p[10], p[12], p[13], p[14], p[15], FREQSp[1],
                              (fref = fref, toff = 0.0))
        model isa IMRPhenomXP && @test st.tshift ≈ time_at_frequency(IMRPhenomXAS(), p[1], p[2], p[3], p[4], fref)
        q = copy(pos); q[6, 1] += st.tshift + off
        @test isapprox(_ploglike(model, data, _pcfg(; time_ref_frequency = fref, time_ref_offset = off), q)[1], base; rtol = 1e-10)
    end
end

@testset "$(typeof(model)) gw_logp_and_grad vs finite differences (opts=$opts)" for model in (IMRPhenomXP(), IMRPhenomXPHM()), opts in (:plain, :all)
    data = [_pmake_data(model, H1, 150.0, 1.0), _pmake_data(model, L1, 150.0, 2.0)]
    kw = opts == :plain ? (;) :
         (; marginalize_distance = true, distance_lo = 0.05, distance_hi = 1.5, distance_power = 2,
            time_reference = 2, time_ref_frequency = 35.0, time_ref_offset = 0.3)
    cfg = _pcfg(; nw = 2, kw...)
    positions = _ppositions([TRUTHp, OTHERp])
    lg! = BNp.gw_logp_and_grad(data, cfg, FREQSp; model = model)
    grad = zeros(15, 2); logp = zeros(2)
    lg!(grad, logp, positions, nothing)
    @test isapprox(logp, _ploglike(model, data, cfg, positions) .+ BNp._call_log_prior_vec(positions); rtol = 1e-10)
    for r in 1:15
        h = r == 6 ? 1e-8 : 1e-6
        pp = copy(positions); pp[r, :] .+= h
        pm = copy(positions); pm[r, :] .-= h
        fd = (_ploglike(model, data, cfg, pp) .- _ploglike(model, data, cfg, pm)) ./ (2h)
        ok = isapprox(grad[r, :], fd; rtol = 1e-4, atol = 1e-3)
        ok || @info "$(typeof(model)) gradient mismatch" r grad[r, :] fd
        @test ok
    end
end

@testset "$(typeof(model)) bin-parallel likelihood + gradient (CPU backend, setup = $setup)" for model in (IMRPhenomXP(), IMRPhenomXPHM(102), IMRPhenomXPHM()),
                                                                                          setup in (:host, :device)
    # `xp[hm]_binparallel_logp_and_grad`: the device path's factorization (XP: 5 twisted (2,2) components;
    # XPHM: hp, hc with hand-written thetaJN / alpha0 / epsilon0 derivatives; flat setup with 5 forward tangents incl.
    # chi_p; first-order surrogate) on KernelAbstractions.CPU() -- equal to the dense likelihood and its finite
    # differences; finite gradient at zero in-plane spin (chi_p = 0). `setup = :device`: every per-walker stage on the
    # backend (forward Jacobians + the surrogate kernel instead of the host Enzyme reverse).
    data = [_pmake_data(model, H1, 150.0, 1.0), _pmake_data(model, L1, 150.0, 2.0)]
    kw = (; marginalize_distance = true, distance_lo = 0.05, distance_hi = 1.5, distance_power = 2,
          time_reference = 1, time_ref_frequency = 35.0, time_ref_offset = 0.3)
    cfg = _pcfg(; nw = 3, kw...)
    positions = _ppositions([TRUTHp, OTHERp, (OTHERp[1:11]..., 0.0, 0.0, 0.0, 0.0)])
    bp = model isa IMRPhenomXP ? BNp.xp_binparallel_logp_and_grad : BNp.xphm_binparallel_logp_and_grad
    mkw = model isa IMRPhenomXPHM ? (; model) : (;)
    f! = bp(data, cfg, FREQSp; backend = KernelAbstractions.CPU(), nwalkers = 3, setup, mkw...)
    grad = zeros(15, 3); logp = zeros(3)
    f!(grad, logp, positions, nothing)
    # XPHM (223 MSA; 102 NNLO) interpolates its angles from a per-walker table (`XPHMBinParallel.jl`):
    # the value within ~1e-5 of the exact likelihood, and the gradient is checked against finite differences of the
    # closure itself (what NUTS needs)
    tab = model isa IMRPhenomXPHM
    lref = _ploglike(model, data, cfg, positions) .+ BNp._call_log_prior_vec(positions)
    tab && @info "$(typeof(model)) bin-parallel (angle table) vs dense logp" maxabs = maximum(abs.(logp .- lref))
    @test tab ? isapprox(logp, lref; atol = 1e-4) : isapprox(logp, lref; rtol = 1e-12)
    @test all(isfinite, grad)
    fself(p) = (gg = zeros(15, 3); ll = zeros(3); f!(gg, ll, p, nothing); ll .- BNp._call_log_prior_vec(p))
    for r in 1:15
        h = r == 6 ? 1e-8 : 1e-6
        ws = r >= 12 ? (1:2) : (1:3)                  # chi_p = |S_perp| has a kink at zero in-plane spin
        pp = copy(positions); pp[r, :] .+= h
        pm = copy(positions); pm[r, :] .-= h
        fd = tab ? (fself(pp) .- fself(pm)) ./ (2h) : (_ploglike(model, data, cfg, pp) .- _ploglike(model, data, cfg, pm)) ./ (2h)
        ok = isapprox(grad[r, ws], fd[ws]; rtol = 1e-4, atol = 1e-3)
        ok || @info "$(typeof(model)) bin-parallel gradient mismatch" r grad[r, :] fd
        @test ok
    end
end

@testset "IMRPhenomXPHM 223: LAL's MSA -> NNLO (102) fallback, dense and bin-parallel" begin
    # LAL (LALSimIMRPhenomX_precession.c, IMRPhenomX_Initialize_MSA_System) sets MSA_ERROR when |Omega_z5| > 1000;
    # IMRPhenomXGetAndSetPrecessionVariables then switches 223 to 102 for the whole waveform. A GW150914 posterior
    # walker of a production SMC run (~0.2-0.4% of those walkers fail) where it does at f_ref = 20 Hz
    # (LALSuite 6.2.0: XPHM 223 == 102 bit for bit there). The dense 223 likelihood must equal 102, the bin-parallel
    # one too (within its angle table's interpolation error); OTHERp (MSA ok) must not.
    m1, m2, _, _ = BNp._eobimr_physical(30.082095, 0.711455, 0.0, 0.0)
    pf = (m1, m2, -0.263881, -0.016319, TRUTHp[5:11]..., 0.35372, -0.475136, -0.586922, 0.61591)
    st = KFp._xp_precession_setup(m1, m2, pf[12], pf[13], pf[3], pf[14], pf[15], pf[4], FREQSp[1], pf[10], pf[7];
                                  prec_version = Val(223))
    @test !st.msa_ok
    data = [_pmake_data(IMRPhenomXPHM(), H1, 150.0, 1.0), _pmake_data(IMRPhenomXPHM(), L1, 150.0, 2.0)]
    cfg = _pcfg(; nw = 2, marginalize_distance = true, distance_lo = 0.05, distance_hi = 1.5, distance_power = 2,
                time_reference = 1, time_ref_frequency = 35.0, time_ref_offset = 0.3)
    positions = _ppositions([pf, OTHERp])
    l223 = _ploglike(IMRPhenomXPHM(), data, cfg, positions)
    l102 = _ploglike(IMRPhenomXPHM(102), data, cfg, positions)
    @test isapprox(l223[1], l102[1]; rtol = 1e-13)
    @test abs(l223[2] - l102[2]) > 1e-3
    f! = BNp.xphm_binparallel_logp_and_grad(data, cfg, FREQSp; backend = KernelAbstractions.CPU(), nwalkers = 2,
                                            setup = :device, model = IMRPhenomXPHM())
    grad = zeros(15, 2); logp = zeros(2)
    f!(grad, logp, positions, nothing)
    lp = BNp._call_log_prior_vec(positions)
    @test isapprox(logp .- lp, l223; atol = 1e-4)
    @test all(isfinite, grad)
    lv = zeros(2); gv = zeros(15, 2)
    f!(gv, lv, positions, nothing; value_only = true)  # (value-only setup: no tangents)
    @test isapprox(lv, logp; rtol = 1e-13)
end
