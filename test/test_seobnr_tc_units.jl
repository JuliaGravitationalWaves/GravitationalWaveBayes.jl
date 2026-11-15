using Test
using GravitationalWaveBayes
using GravitationalWaveBayes: LikelihoodData, PEConfig, batched_log_likelihood_kernel!
import GravitationalWaveforms
using GravitationalWaveforms: Ampl, Phi, SEOBNRv5AlignedSpinF2
using GravitationalWaveData
using Random

# ---------------------------------------------------------------------------------------------
# Regression test for a SEOBNRv5AlignedSpinF2 `t_c` unit bug.
#
# `_extrinsic_setup`'s `t0 = -2*pi*t_c/M_sec` is the Phenom/PN convention (multiplied by the
# DIMENSIONLESS Mf). The SEOBNR kernels applied `- t0 * f` with `f` in Hz, i.e. a time shift of
# `t_c/M_sec ~ 3000 * t_c` seconds (mass dependent). On real data that made logp a comb of ~10 us
# spikes in `t_c` (curvature ~1e11 s^-2 vs the physical ~1e7) and froze `t_c` in every walker.
# Every earlier test passed because the injection was built through the library's
# `Phi(...; t0=-2*pi*t_c/M_sec)` -- the SAME wrong convention as the kernel.
#
# So the checks here state the physics directly, WITHOUT any `t0` bookkeeping:
#   (1) a time shift of `t_c` seconds multiplies h(f) by exp(-2*pi*i*f*t_c) (physical convention;
#       the PA-F2 phase enters conjugated), compared against the kernel at several `t_c` and two very different total masses
#       (the bug's factor is mass dependent);
#   (2) smoothness: for a noise-free injection, moving `t_c` by 10 us changes ll by ~(2*pi*f*1e-5)^2
#       ~ 1e-6 of the SNR^2, not by O(SNR^2).
# ---------------------------------------------------------------------------------------------

const _TCU_GMSUN_OVER_C3 = GravitationalWaveforms.GMsun_over_c3

# Detector-frame waveform with the time shift/phase written out explicitly (no `t0`/`phiRef`).
function _tcu_h_det(model, freqs, physical::AbstractVector, detector, gps_time)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = physical
    mu = (m1 * m2) / (m1 + m2)
    mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
    eta = (m1 * m2) / (m1 + m2)^2
    amp = Ampl(model, freqs, mc, eta, chi1, chi2, dL)
    # Physical convention (KernelFused `_PHASE_CONVENTION`): the PA-F2 Phi enters conjugated,
    # and arrival at t_c is exp(-2*pi*i*f*t_c) -- i.e. the whole old phase negated.
    phi = .-(Phi(model, freqs, mc, eta, chi1, chi2) .+ 2pi .* freqs .* t_c .- phase)
    Fplus, Fcross = GravitationalWaveData.compute_antenna_response(detector, ra, dec, psi, gps_time)
    combo = Fplus * 0.5 * (1 + cos(theta_jn)^2) - im * Fcross * cos(theta_jn)
    return combo .* (amp .* exp.(im .* phi))
end

_tcu_ll(model, freqs, physical, data) =
    -2 * (freqs[2] - freqs[1]) * sum(abs2.(_tcu_h_det(model, freqs, physical, data.detector, data.gps_time) .- data.strain) ./ data.psd)

@testset "SEOBNRv5AlignedSpinF2: t_c is a physical time shift in SECONDS (not t_c/M_sec)" begin
    model = SEOBNRv5AlignedSpinF2()
    freqs = collect(20.5:0.25:30.5)
    ntime = length(freqs)
    psd = fill(1e-46, ntime)
    data0 = LikelihoodData(freqs, zeros(ComplexF64, ntime), psd)

    # two different total masses (M_sec ~ 3.2e-4 s and ~ 1.7e-4 s; a heavier system has no model support in this band): the bug's factor is 1/M_sec
    for (m1, m2) in ((36.0, 29.0), (20.0, 15.0))
        base = [m1, m2, 0.05, -0.03, 0.4, 0.0, 0.7, 1.2, -0.4, 1.0, 0.6]
        inj = _tcu_h_det(model, freqs, base, data0.detector, data0.gps_time)
        data = LikelihoodData(freqs, inj, psd)
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)

        kernel_ll(p) = (ll = zeros(1); batched_log_likelihood_kernel!(ll, reshape(p, :, 1), freqs, cfg, data; model=model); ll[1])

        @testset "kernel == explicit exp(-2*pi*i*f*t_c) reference, m=($m1,$m2)" begin
            for t_c in (0.0, 3e-3, -8e-3, 0.02)
                p = copy(base); p[6] = t_c
                @test isapprox(kernel_ll(p), _tcu_ll(model, freqs, p, data); rtol=1e-9, atol=1e-6)
            end
        end

        @testset "ll is smooth in t_c: a 10 us shift is a ~1e-6 effect, m=($m1,$m2)" begin
            snr2 = -kernel_ll(copy(base) .* [1, 1, 1, 1, 1e12, 1, 1, 1, 1, 1, 1])   # dL -> huge => h ~ 0 => ll ~ -<d,d> = -SNR^2
            @test snr2 > 10                       # the injection is loud enough for this to mean something
            ll0 = kernel_ll(base)
            @test isapprox(ll0, 0.0; atol=1e-6 * snr2)      # noise-free injection at truth
            for dt in (1e-5, -1e-5)
                p = copy(base); p[6] = dt
                @test abs(ll0 - kernel_ll(p)) < 1e-3 * snr2   # bug gave O(snr2)
            end
            # the exact analytic scale: (2*pi*f_max*dt)^2 * SNR^2 bounds the change from above
            p = copy(base); p[6] = 1e-4
            @test abs(ll0 - kernel_ll(p)) < (2pi * maximum(freqs) * 1e-4)^2 * snr2
        end
    end
end
