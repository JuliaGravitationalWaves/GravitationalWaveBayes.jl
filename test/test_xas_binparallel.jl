using Test
using Random
using GravitationalWaveBayes
using GravitationalWaveforms: IMRPhenomXAS
using GravitationalWaveData: H1, L1
using KernelAbstractions
using Enzyme

const BNbp = GravitationalWaveBayes.BatchedNUTS

# Bin-parallel IMRPhenomXAS likelihood + gradient (`xas_binparallel_logp_and_grad`) on the
# KernelAbstractions CPU backend: logp and gradient equal the CPU per-walker path (`gw_logp_and_grad`) for one
# and two detectors, plain and with every likelihood option, shared and per-detector kernels, and walkers
# outside the prior box; a zero-PSD bin is handled. `resident = true` (per-walker stages as kernels, forward-mode
# coefficient Jacobian) gives the same numbers, with the width-8 or the split (width 5 + 3) Jacobian; so does the
# forward-tangent single pass (`variant = :fwd`, dense and relative binning). The same code runs on CUDA
# (scripts/check_xas_binparallel.jl, scripts/check_xas_variants.jl).

@testset "IMRPhenomXAS bin-parallel likelihood + gradient" begin
    freqs = collect(20.0:0.5:300.0)
    gps = 1126259462.4
    function det_data(det, seed; zero_bin=false)
        n = length(freqs)
        s = 1e-46
        psd = fill(s, n)
        zero_bin && (psd[17] = 0.0)
        strain = [1e-22 * complex(sin(1.3k + seed), cos(0.7k * seed)) for k in 1:n]
        LikelihoodData(freqs, strain, psd, det, gps)
    end
    h1 = det_data(H1, 1); l1 = det_data(L1, 2)
    nw = 5
    pos = [30.0, 0.8, 0.1, -0.1, 0.5, 0.01, 1.0, 1.9, -0.9, -0.6, 1.0] .+
          [0.3, 0.05, 0.1, 0.1, 0.05, 0.002, 0.3, 0.1, 0.05, 0.1, 0.3] .* randn(Random.Xoshiro(3), 11, nw)
    pos[1, nw] = 500.0          # outside the box prior: logp -Inf in both paths
    allopts = (; marginalize_phase=true, marginalize_distance=true, distance_lo=0.05, distance_hi=2.0,
               distance_power=2, time_ref_frequency=35.0, time_ref_offset=0.3)
    for (data, label) in (([h1, l1], "H1+L1"), (h1, "H1")), (opts, olabel) in (((;), "plain"), (allopts, "all options")),
        (shared, adj, res, var, jac) in ((true, :hand, false, :adjoint, :split), (true, :enzyme, false, :adjoint, :split),
                                         (false, :hand, false, :adjoint, :split), (true, :enzyme, true, :adjoint, :width8),
                                         (false, :hand, true, :adjoint, :split), (true, :enzyme, true, :fwd, :split5),
                                         (true, :enzyme, true, :fwd, :split3))
        opts2 = data isa AbstractVector && !isempty(opts) ? merge(opts, (; time_reference=1)) : opts
        cfg = with_likelihood_options(PEConfig{Float64}(length(freqs), nw, 1e-3, 1); opts2...)
        g1 = zeros(11, nw); l1v = zeros(nw); g2 = zeros(11, nw); l2v = zeros(nw)
        BNbp.gw_logp_and_grad(data, cfg, freqs; model=IMRPhenomXAS())(g1, l1v, pos, nothing)
        BNbp.xas_binparallel_logp_and_grad(data, cfg, freqs; backend=KernelAbstractions.CPU(), nwalkers=nw,
                                           chunk=7, shared=shared, adjoint=adj, resident=res, variant=var,
                                           jac=jac)(g2, l2v, pos, nothing)
        ok = isfinite.(l1v)
        @testset "$label $olabel shared=$shared adjoint=$adj resident=$res variant=$var jac=$jac" begin
            @test ok == isfinite.(l2v)
            @test count(ok) == nw - 1
            @test isapprox(l1v[ok], l2v[ok]; rtol=1e-10)
            # marginalized rows (phase, dL) are exactly flat: compare on the gradient's overall scale
            @test maximum(abs.(g1[:, ok] .- g2[:, ok])) <= 1e-9 * maximum(abs.(g1[:, ok]))
        end
    end
    # Relative binning on the device (`relative_binning = rb`, resident) == the CPU relative-binning path.
    for (data, label) in (([h1, l1], "H1+L1"), (h1, "H1")), (opts, olabel) in (((;), "plain"), (allopts, "all options")),
        var in (:adjoint, :fwd)
        opts2 = data isa AbstractVector && !isempty(opts) ? merge(opts, (; time_reference=1)) : opts
        cfg = with_likelihood_options(PEConfig{Float64}(length(freqs), nw, 1e-3, 1); opts2...)
        rb = BNbp.build_xas_relative_binning(data, cfg, pos[:, 1]; nbins=40)
        g1 = zeros(11, nw); l1v = zeros(nw); g2 = zeros(11, nw); l2v = zeros(nw)
        BNbp.gw_logp_and_grad(data, cfg, freqs; model=IMRPhenomXAS(), relative_binning=rb)(g1, l1v, pos, nothing)
        BNbp.xas_binparallel_logp_and_grad(data, cfg, freqs; backend=KernelAbstractions.CPU(), nwalkers=nw, chunk=3,
                                           relative_binning=rb, variant=var)(g2, l2v, pos, nothing)
        ok = isfinite.(l1v)
        @testset "relative binning $label $olabel variant=$var" begin
            @test ok == isfinite.(l2v)
            @test isapprox(l1v[ok], l2v[ok]; rtol=1e-10)
            @test maximum(abs.(g1[:, ok] .- g2[:, ok])) <= 1e-8 * maximum(abs.(g1[:, ok]))
        end
    end
    # Pow-free primal + tangent core (`_xas_pa_core`, used by `variant = :fwd`) == `_xas_phase_amp_flat` and the
    # hand VJP contracted with random tangents, at frequencies in every phase and amplitude segment, for several
    # binaries (including high spins and unequal masses).
    @testset "pow-free primal + tangent core" begin
        rng = Random.Xoshiro(5)
        model = IMRPhenomXAS()
        g = BNbp._xasbp_geometry([h1, l1], PEConfig{Float64}(length(freqs), nw, 1e-3, 1))
        lik = GravitationalWaveBayes.KernelFused._lik_opts(PEConfig{Float64}(length(freqs), nw, 1e-3, 1))
        worst_p = 0.0; worst_t = 0.0
        for (mc, q, c1, c2) in ((30.0, 0.8, 0.1, -0.1), (8.0, 0.25, 0.9, -0.5), (60.0, 0.95, -0.8, 0.7), (2.0, 0.6, 0.0, 0.3))
            K, _ = BNbp._xasbp_coeffs_tau(mc, q, c1, c2, 0.5, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, model, g, lik)
            t = ntuple(_ -> Tuple(randn(rng, BNbp._XASBP_NK)), 5)
            for f in exp.(range(log(5.0), log(4000.0); length=400))
                phi0, A0 = BNbp._xas_phase_amp_flat(K, f)
                phi, A, dphi, dA = BNbp._xas_phase_amp_flat_jvp(K, f, t)
                worst_p = max(worst_p, abs(phi - phi0) / max(abs(phi0), 1.0), abs(A - A0) / max(abs(A0), 1e-300))
                for m in 1:5
                    rp = BNbp._xas_phase_amp_flat_vjp(ntuple(_ -> 0.0, BNbp._XASBP_NK), K, f, 1.0, 0.0)
                    ra = BNbp._xas_phase_amp_flat_vjp(ntuple(_ -> 0.0, BNbp._XASBP_NK), K, f, 0.0, 1.0)
                    ep = sum(rp .* t[m]); ea = sum(ra .* t[m])
                    sp = sum(abs.(rp .* t[m])); sa = sum(abs.(ra .* t[m]))
                    worst_t = max(worst_t, abs(dphi[m] - ep) / max(sp, 1e-300), abs(dA[m] - ea) / max(sa, 1e-300))
                end
            end
        end
        @test worst_p < 1e-12
        @test worst_t < 1e-12
    end
    # Hand-written gradient of the marginalized likelihood combination vs Enzyme, every option combination, with
    # (ll_plain, hh, zr, zi, dL) drawn so that the quadrature's node-placement clamps are active in some draws.
    @testset "hand-written _marg_combine gradient" begin
        rng = Random.Xoshiro(11)
        for mp in (false, true), md in (false, true), dpow in (0, 2)
            lik = (margphase=mp, margdist=md, dlo=0.05, dhi=2.0, dpow=dpow, fref=0.0, toff=0.0, taper=0.0)
            worst = 0.0
            for _ in 1:200
                hh = exp(randn(rng) * 2 + 4)                   # <h,h> from ~1 to ~5000
                snr = sqrt(hh) * (0.5 + randn(rng))
                zr = snr * sqrt(hh) * cos(2pi * rand(rng)); zi = snr * sqrt(hh) * sin(2pi * rand(rng))
                llp = -8000.0 + randn(rng); dL = exp(randn(rng) - 0.5)
                v, g... = BNbp._marg_combine_grad(llp, hh, zr, zi, dL, lik)
                ref = Enzyme.autodiff(Enzyme.ReverseWithPrimal, GravitationalWaveBayes.KernelFused._marg_combine, Enzyme.Active,
                                      Enzyme.Active(llp), Enzyme.Active(hh), Enzyme.Active(zr), Enzyme.Active(zi),
                                      Enzyme.Active(dL), Enzyme.Const(lik))
                @test v == ref[2]
                rg = collect(ref[1][1:5])
                worst = max(worst, maximum(abs.(collect(g) .- rg) ./ max.(abs.(rg), 1e-8)))
            end
            @test worst < 1e-8
        end
    end
    # A zero-PSD bin contributes nothing. (The CPU per-walker path's gradient is NaN there: Enzyme sends a zero
    # cotangent through `x / 0` inside `KernelFused._safe_div`'s ifelse; the bin-parallel path precomputes w = 0.)
    @testset "zero-PSD bin" begin
        dz = [det_data(H1, 1; zero_bin=true), det_data(L1, 2; zero_bin=true)]
        cfg = with_likelihood_options(PEConfig{Float64}(length(freqs), nw, 1e-3, 1); allopts..., time_reference=1)
        g1 = zeros(11, nw); l1v = zeros(nw); g2 = zeros(11, nw); l2v = zeros(nw)
        BNbp.gw_logp_and_grad(dz, cfg, freqs; model=IMRPhenomXAS())(g1, l1v, pos, nothing)
        BNbp.xas_binparallel_logp_and_grad(dz, cfg, freqs; backend=KernelAbstractions.CPU(), nwalkers=nw)(g2, l2v, pos, nothing)
        ok = isfinite.(l1v)
        @test isapprox(l1v[ok], l2v[ok]; rtol=1e-10)
        @test all(isfinite, g2[:, ok])
    end
end
