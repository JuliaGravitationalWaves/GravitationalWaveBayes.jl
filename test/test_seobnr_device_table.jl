using Test
using Random
using GravitationalWaveBayes
using GravitationalWaveData: H1, L1
using KernelAbstractions
import GravitationalWaveforms

const BNsdt = GravitationalWaveBayes.BatchedNUTS

# SEOBNRv5 IMR table and log-likelihood built on a KernelAbstractions device (`SEOBNRDeviceTable.jl`,
# `SEOBNRDeviceLikelihood.jl`, SEOBNR_GPU_PLAN.md M1-M3), here on the CPU backend: the device table equals
# `GravitationalWaveforms._eobimr_coeffs` to round-off (the device phase correction evaluates arg Gamma in real arithmetic; serial
# and workgroup FFT), the device log-likelihood equals the CPU per-walker likelihood (`_eobimr_walker_ll[_multi]`), and
# the forward-mode device gradient equals the CPU per-walker Enzyme gradient (`_eobimr_gradient!`), for one and two
# detectors, plain and with every likelihood option. The same code runs on CUDA (scripts/check_seobnr_device_table.jl,
# scripts/check_seobnr_device_logp.jl).

@testset "SEOBNRv5 IMR device table + likelihood (KA CPU backend)" begin
    model = GravitationalWaveforms.SEOBNRv5IMRAlignedSpinF2()
    be = KernelAbstractions.CPU()
    nw = 3
    tab_in = [30.0 14.0 45.0; 0.245 0.16 0.22; 0.2 -0.7 0.6; -0.3 0.5 0.1]     # (mc, eta, chi1, chi2) per walker
    @testset "table vs _eobimr_coeffs (serial and workgroup FFT)" begin
        T = BNsdt.SEOBNRDeviceTables(be, nw)
        T2 = BNsdt.SEOBNRDeviceTables(be, nw)
        BNsdt.seobnr_device_tables!(T, tab_in; serial_fft = true)
        BNsdt.seobnr_device_tables!(T2, tab_in)
        for w in 1:nw
            c = GravitationalWaveforms._eobimr_coeffs(model, tab_in[:, w]...)
            d = BNsdt.seobnr_device_table_host(T, w)
            d2 = BNsdt.seobnr_device_table_host(T2, w)
            @test c.success && d.success && d2.success
            for f in fieldnames(typeof(c))
                b = getfield(c, f)
                @test isapprox(getfield(d, f), b; rtol = 1e-9, atol = 1e-9 * maximum(abs, b))
                @test isapprox(getfield(d2, f), b; rtol = 1e-9, atol = 1e-9 * maximum(abs, b))
            end
        end
    end

    freqs = collect(20.0:0.25:400.0)
    gps = 1126259462.4
    function det_data(det, seed)
        n = length(freqs)
        strain = [1e-22 * complex(sin(1.3k + seed), cos(0.7k * seed)) for k in 1:n]
        GravitationalWaveBayes.LikelihoodData(freqs, strain, fill(1e-46, n), det, gps)
    end
    h1 = det_data(H1, 1); l1 = det_data(L1, 2)
    pos = [30.0, 0.8, 0.1, -0.1, 0.5, 0.01, 1.0, 1.9, -0.9, -0.6, 1.0] .+
          [0.5, 0.05, 0.1, 0.1, 0.05, 0.002, 0.3, 0.1, 0.05, 0.1, 0.3] .* randn(Random.Xoshiro(3), 11, nw)
    allopts = (; marginalize_phase = true, marginalize_distance = true, distance_lo = 0.05, distance_hi = 2.0,
               distance_power = 2, time_ref_frequency = 35.0, time_ref_offset = 0.3)
    for (data, label) in (([h1, l1], "H1+L1"), ([h1], "H1")), (opts, olabel) in (((;), "plain"), (allopts, "all options"))
        opts2 = length(data) > 1 && !isempty(opts) ? merge(opts, (; time_reference = 1)) : opts
        cfg = with_likelihood_options(PEConfig{Float64}(length(freqs), nw, 1e-3, 1); opts2...)
        lik = GravitationalWaveBayes.KernelFused._lik_opts(cfg)
        det = BNsdt._eobimr_det(length(data) == 1 ? data[1] : data, freqs, cfg)
        llf = BNsdt._eobimr_llfun(det)
        ref = [llf(pos[:, w]..., model, det, lik) for w in 1:nw]
        L = BNsdt.SEOBNRDeviceLikelihood(be, data, cfg; nwalkers = nw, chunk = 7)
        got = Array(BNsdt.seobnr_device_logp!(L, pos))
        gref = zeros(11, nw); lref = zeros(nw)
        BNsdt._eobimr_gradient!(gref, pos, freqs, cfg, length(data) == 1 ? data[1] : data, model; ll_out = lref)
        lg, gg = BNsdt.seobnr_device_logp_and_grad!(L, pos)
        lg = Array(lg); gg = Array(gg)
        @testset "$label $olabel" begin
            @test all(isfinite, ref)
            @test isapprox(got, ref; rtol = 1e-10)
            @test isapprox(lg, lref; rtol = 1e-10)
            # marginalized rows (phase, dL) are exactly flat: compare on each walker's gradient scale. ~3e-8: the device
            # differentiates one Newton step at the converged j0 root, the CPU the 20 FD-Jacobian Newton iterations
            for w in 1:nw
                @test maximum(abs.(gg[:, w] .- gref[:, w])) <= 1e-7 * maximum(abs.(gref[:, w]))
            end
        end
    end

    # The sampler closure (`seobnr_device_closures`, what `gw_logp_and_grad_gpu_staged` / the driver use) vs the CPU
    # `gw_logp_and_grad` (likelihood + box prior), with walker 3 outside the box (t_c): -Inf there, zero device gradient.
    @testset "sampler closures vs gw_logp_and_grad" begin
        data = [h1, l1]
        cfg = with_likelihood_options(PEConfig{Float64}(length(freqs), nw, 1e-3, 1);
            merge(allopts, (; time_reference = 1))...)
        p2 = copy(pos); p2[6, 3] = 0.2
        lpg!, ll! = BNsdt.seobnr_device_closures(data, cfg; backend = be, nwalkers = nw)
        g = zeros(11, nw); lp = zeros(nw)
        lpg!(g, lp, p2, trues(nw))
        gc = zeros(11, nw); lpc = zeros(nw)
        BNsdt.gw_logp_and_grad(data, cfg, freqs; model = model)(gc, lpc, p2, trues(nw))
        @test isapprox(lp[1:2], lpc[1:2]; rtol = 1e-10)
        @test lp[3] == -Inf && lpc[3] == -Inf && all(iszero, g[:, 3])
        for w in 1:2
            @test maximum(abs.(g[:, w] .- gc[:, w])) <= 1e-7 * maximum(abs.(gc[:, w]))
        end
        ll = zeros(nw)
        ll!(ll, p2)
        @test isapprox(ll[1:2], lpc[1:2]; rtol = 1e-10) && ll[3] == -Inf
    end
end
