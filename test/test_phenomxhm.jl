using Test
using GravitationalWaveBayes
using GravitationalWaveforms: IMRPhenomXHM
using GravitationalWaveforms
using GravitationalWaveData

# Manual, host-side reference implementation of the 11-parameter IMRPhenomXHM physical forward
# model, mirroring `KernelFused._fused_waveform_likelihood_kernel_phenomxhm!` line-for-line
# (antenna response, per-walker `_xhm_system_setup`/`_xhm_mode_setup`/`_xhm_mode32_setup`, the
# per-frequency `_xhm_hphc_scalar` core, the `t_c` post-multiplication, projection through the
# antenna response) -- same role as `test_phenomxp.jl`'s `_manual_h_det_phenomxp` for `IMRPhenomXP`,
# kept self-contained in this file per this codebase's existing per-test-file duplication style.
# This one additionally cross-checks against calling `GravitationalWaveforms.hphc` DIRECTLY (the public API),
# not just against the internal scalar cores -- the cheapest and most direct correctness check for
# this task (see DEVELOPMENT_PLAN.md Phase 3 / `KernelFused.jl`'s module docstring).
function _manual_h_det_phenomxhm_via_hphc(freqs::AbstractVector, physical::AbstractVector, detector, gps_time)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = physical
    f_ref = freqs[1]

    hp, hc = GravitationalWaveforms.hphc(IMRPhenomXHM(), freqs, m1, m2, chi1, chi2, dL, theta_jn, phase; f_ref = f_ref)
    shift = exp.(-im .* 2 .* pi .* freqs .* t_c)
    hp = hp .* shift
    hc = hc .* shift

    Fplus, Fcross = GravitationalWaveData.compute_antenna_response(detector, ra, dec, psi, gps_time)
    return Fplus .* hp .+ Fcross .* hc
end

# Independent reconstruction via the SAME internal scalar cores the kernel body itself calls
# (`_xhm_system_setup`/`_xhm_mode_setup`/`_xhm_mode32_setup`/`_xhm_hphc_scalar`), rather than the
# public `hphc` array entry point -- confirms the kernel's own per-frequency-bin loop (not just the
# high-level formula) matches.
function _manual_h_det_phenomxhm(freqs::AbstractVector, physical::AbstractVector, detector, gps_time)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = physical
    mu = (m1 * m2) / (m1 + m2)
    mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
    eta = (m1 * m2) / (m1 + m2)^2
    f_ref = freqs[1]

    sys = GravitationalWaveforms._xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, phase)
    ms21 = GravitationalWaveforms._xhm_mode_setup(2, 1, sys)
    ms33 = GravitationalWaveforms._xhm_mode_setup(3, 3, sys)
    ms44 = GravitationalWaveforms._xhm_mode_setup(4, 4, sys)
    m32 = GravitationalWaveforms._xhm_mode32_setup(sys)
    piHalf = pi / 2

    Fplus, Fcross = GravitationalWaveData.compute_antenna_response(detector, ra, dec, psi, gps_time)

    h = Vector{ComplexF64}(undef, length(freqs))
    for (i, f) in enumerate(freqs)
        hp, hc = GravitationalWaveforms._xhm_hphc_scalar(f, sys, ms21, ms33, ms44, m32, theta_jn, piHalf)
        shift = exp(-im * 2 * pi * f * t_c)
        hp *= shift; hc *= shift
        h[i] = Fplus * hp + Fcross * hc
    end
    return h
end

function _manual_ll_phenomxhm(freqs, physical::AbstractVector, data::LikelihoodData)
    h_det = _manual_h_det_phenomxhm(freqs, physical, data.detector, data.gps_time)
    df = freqs[2] - freqs[1]
    return -2 * df * sum(abs2.(h_det .- data.strain) ./ data.psd)
end

@testset "IMRPhenomXHM (Phase 3 follow-on, aligned-spin higher-order-mode) single-detector kernel" begin
    ntime = 48
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    gps = 1126259462.4
    detector = GravitationalWaveData.H1

    # Genuinely asymmetric mass ratio (GW190412/GW190814-like) so higher-order-mode content is
    # actually significant -- the whole point of this model, per DEVELOPMENT_PLAN.md.
    ref_physical = [32.0, 8.0, 0.3, -0.1, 0.5, 0.015, 0.4, 1.0, -0.3, 1.0, 0.6]

    @testset "kernel agrees with the manual internal-scalar-core reference" begin
        data0 = LikelihoodData(freqs, zeros(ComplexF64, ntime), ones(Float64, ntime), detector, gps)
        strain = _manual_h_det_phenomxhm(freqs, ref_physical, detector, gps) .+
                 1e-24 .* (randn(ntime) .+ im .* randn(ntime))
        psd = fill(1e-46, ntime)
        data = LikelihoodData(freqs, strain, psd, detector, gps)
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)

        ll_manual = _manual_ll_phenomxhm(freqs, ref_physical, data)
        ll_kernel = zeros(Float64, 1)
        batched_log_likelihood_kernel!(ll_kernel, reshape(ref_physical, 11, 1), freqs, cfg, data; model=IMRPhenomXHM())
        @test isapprox(ll_kernel[1], ll_manual; rtol=1e-8, atol=1e-8)
    end

    @testset "kernel's per-bin h(f) agrees with calling GravitationalWaveforms.hphc directly (public API cross-check)" begin
        # The cheapest, most direct correctness check for this task (see DEVELOPMENT_PLAN.md):
        # compare the kernel's own forward model against calling the public `hphc` entry point
        # directly, folding in t_c via the SAME post-hoc phase multiplication.
        h_via_hphc = _manual_h_det_phenomxhm_via_hphc(freqs, ref_physical, detector, gps)
        h_via_internal = _manual_h_det_phenomxhm(freqs, ref_physical, detector, gps)
        @test isapprox(h_via_hphc, h_via_internal; rtol=1e-10, atol=1e-12)

        psd = fill(1e-46, ntime)
        data = LikelihoodData(freqs, ComplexF64.(h_via_hphc), psd, detector, gps)
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)
        ll = zeros(Float64, 1)
        batched_log_likelihood_kernel!(ll, reshape(ref_physical, 11, 1), freqs, cfg, data; model=IMRPhenomXHM())
        @test isapprox(ll[1], 0.0; atol=1e-6)
    end

    @testset "noise-free injection recovery: peak at injection, single-parameter perturbations strictly decrease ll" begin
        strain = _manual_h_det_phenomxhm(freqs, ref_physical, detector, gps)
        psd = fill(1e-46, ntime)
        data = LikelihoodData(freqs, strain, psd, detector, gps)
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)

        ll0 = zeros(Float64, 1)
        batched_log_likelihood_kernel!(ll0, reshape(ref_physical, 11, 1), freqs, cfg, data; model=IMRPhenomXHM())
        @test isapprox(ll0[1], 0.0; atol=1e-6)

        row_names = ["m1", "m2", "chi1", "chi2", "dL", "t_c", "phase", "ra", "dec", "theta_jn", "psi"]
        perturbations = [0.5, 0.5, 0.02, 0.02, 0.05, 0.002, 0.05, 0.05, 0.05, 0.05, 0.05]
        for i in 1:11
            p2 = copy(ref_physical)
            p2[i] += perturbations[i]
            ll2 = zeros(Float64, 1)
            batched_log_likelihood_kernel!(ll2, reshape(p2, 11, 1), freqs, cfg, data; model=IMRPhenomXHM())
            @test ll2[1] < ll0[1]
        end
    end

    @testset "t_c convention: grid scan recovers the injected coalescence-time offset" begin
        # Direct empirical verification of this file's `t_c` post-multiplication convention (see
        # KernelFused.jl's module docstring for the derivation): a wide frequency band (so t_c is
        # well-resolved: bandwidth ~480 Hz here gives timing resolution well under a ms) and a
        # fine grid scan over candidate t_c values, holding every other parameter fixed at
        # injection, confirms the log-likelihood peak sits at (or extremely near) the injected
        # t_c -- not offset, not at its negative, not at some Mf-scaled multiple of it (the class
        # of bug this check is specifically designed to catch, per this task's brief).
        ntime_wide = 481
        freqs_wide = collect(20.0:1.0:(20.0 + ntime_wide - 1))
        t_c_true = 0.0137
        physical_true = [32.0, 8.0, 0.3, -0.1, 0.5, t_c_true, 0.4, 1.0, -0.3, 1.0, 0.6]

        strain = _manual_h_det_phenomxhm(freqs_wide, physical_true, detector, gps)
        psd = fill(1e-46, ntime_wide)
        data = LikelihoodData(freqs_wide, strain, psd, detector, gps)
        cfg = PEConfig{Float64}(ntime_wide, 1, 0.01, 1)

        t_c_grid = collect(-0.05:0.0005:0.05)
        ll_grid = zeros(Float64, length(t_c_grid))
        for (k, tc) in enumerate(t_c_grid)
            p2 = copy(physical_true)
            p2[6] = tc
            ll = zeros(Float64, 1)
            batched_log_likelihood_kernel!(ll, reshape(p2, 11, 1), freqs_wide, cfg, data; model=IMRPhenomXHM())
            ll_grid[k] = ll[1]
        end

        best_idx = argmax(ll_grid)
        t_c_recovered = t_c_grid[best_idx]
        @test isapprox(t_c_recovered, t_c_true; atol=1e-3)
        # Confirm it's a genuine, sharply-peaked maximum (not a plateau/saddle/aliased peak): the
        # true value itself (off-grid, since the grid step is 0.0005 and t_c_true=0.0137 doesn't
        # land exactly on a grid point) should score higher than every grid point.
        ll_at_truth = zeros(Float64, 1)
        batched_log_likelihood_kernel!(ll_at_truth, reshape(physical_true, 11, 1), freqs_wide, cfg, data; model=IMRPhenomXHM())
        @test isapprox(ll_at_truth[1], 0.0; atol=1e-6)
        @test all(ll_grid .< ll_at_truth[1] + 1e-9)
    end

    @testset "row-count assertion: IMRPhenomXHM uses the 11-row aligned-spin convention (not IMRPhenomXP's 15)" begin
        data = LikelihoodData(freqs, zeros(ComplexF64, ntime), ones(Float64, ntime), detector, gps)
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)
        ll = zeros(Float64, 1)
        params15 = reshape(vcat(ref_physical, [0.0, 0.0, 0.0, 0.0]), 15, 1)
        @test_throws AssertionError batched_log_likelihood_kernel!(ll, params15, freqs, cfg, data; model=IMRPhenomXHM())
    end

end

# Manual, host-side multi-detector IMRPhenomXHM reference: mirrors `_manual_h_det_phenomxhm` above,
# adding the per-detector `GravitationalWaveData.time_delay_from_geocenter`-shifted `t_c` -- same relationship
# `test_phenomxp.jl`'s `_manual_h_det_phenomxp_multi` bears to its own single-detector
# `_manual_h_det_phenomxp` (host-side, non-differentiated reference code, so calling the real
# `GravitationalWaveData` functions directly is fine here, unlike inside the kernel body itself).
function _manual_h_det_phenomxhm_multi(freqs::AbstractVector, physical::AbstractVector, detector, gps_time, f_ref)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = physical
    mu = (m1 * m2) / (m1 + m2)
    mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
    eta = (m1 * m2) / (m1 + m2)^2

    delay = GravitationalWaveData.time_delay_from_geocenter(detector, ra, dec, gps_time)
    Fplus, Fcross = GravitationalWaveData.compute_antenna_response(detector, ra, dec, psi, gps_time)

    # f_ref is taken from the FIRST detector's first frequency bin, matching the kernel's own
    # `freqs_tuple[1][1]` convention (sys is hoisted outside the per-detector loop, so it can only
    # anchor to one representative grid) -- see KernelFused.jl's
    # `_fused_waveform_likelihood_kernel_phenomxhm_multi!` comment block.
    sys = GravitationalWaveforms._xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, phase)
    ms21 = GravitationalWaveforms._xhm_mode_setup(2, 1, sys)
    ms33 = GravitationalWaveforms._xhm_mode_setup(3, 3, sys)
    ms44 = GravitationalWaveforms._xhm_mode_setup(4, 4, sys)
    m32 = GravitationalWaveforms._xhm_mode32_setup(sys)
    piHalf = pi / 2

    h = Vector{ComplexF64}(undef, length(freqs))
    for (i, f) in enumerate(freqs)
        hp, hc = GravitationalWaveforms._xhm_hphc_scalar(f, sys, ms21, ms33, ms44, m32, theta_jn, piHalf)
        # The multi-detector t_c generalization under test: t_c + dt_geo, dt_geo = this detector's
        # own light-travel-time offset from the geocenter.
        shift = exp(-im * 2 * pi * f * (t_c + delay))
        hp *= shift; hc *= shift
        h[i] = Fplus * hp + Fcross * hc
    end
    return h
end

@testset "IMRPhenomXHM (Phase 3 follow-on) multi-detector kernel" begin
    ntime = 40
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    gps = 1126259462.4
    dets = [GravitationalWaveData.H1, GravitationalWaveData.L1]
    f_ref = freqs[1]  # both detectors share this grid here, matching the kernel's own convention
    ref_physical = [32.0, 8.0, 0.3, -0.1, 0.5, 0.015, 0.4, 1.0, -0.3, 1.0, 0.6]

    @testset "agrees with a manual host-side multi-detector reference" begin
        strains = [_manual_h_det_phenomxhm_multi(freqs, ref_physical, d, gps, f_ref) .+
                   1e-24 .* (randn(ntime) .+ im .* randn(ntime)) for d in dets]
        psds = [fill(1e-46, ntime) for _ in dets]
        data = [LikelihoodData(freqs, strains[i], psds[i], dets[i], gps) for i in eachindex(dets)]
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)

        ll_manual = sum(begin
            h = _manual_h_det_phenomxhm_multi(freqs, ref_physical, dets[i], gps, f_ref)
            df = freqs[2] - freqs[1]
            -2 * df * sum(abs2.(h .- data[i].strain) ./ data[i].psd)
        end for i in eachindex(dets))

        ll_kernel = zeros(Float64, 1)
        batched_log_likelihood_kernel!(ll_kernel, reshape(ref_physical, 11, 1), cfg, data; model=IMRPhenomXHM())
        @test isapprox(ll_kernel[1], ll_manual; rtol=1e-8, atol=1e-8)
    end

    @testset "cheapest direct check: two IDENTICAL detectors give exactly 2x the one-detector evaluation" begin
        # Two detectors sharing IDENTICAL geometry (same `GravitationalWaveData.H1` object, hence identical
        # response/location -> identical Fplus/Fcross/dt_geo) and IDENTICAL freqs/strain/psd: the
        # multi-detector sum must equal exactly 2x a ONE-detector evaluation through the SAME
        # multi-detector kernel path. Deliberately NOT compared against the single-detector
        # kernel: that kernel never applies a `dt_geo` term at all (there is no second detector to
        # align against), so it is not an apples-to-apples comparison against the multi-detector
        # kernel's own per-detector `t_c + dt_geo` convention -- comparing two calls into the SAME
        # multi-detector kernel (`ndet=1` vs `ndet=2`) isolates exactly the `1:ndet`-summing
        # structure this check exists to validate. A real, meaningful check only because the two
        # "detectors" are genuinely identical (an actual H1-vs-L1 comparison would differ due to
        # antenna response and time-delay, which the dedicated testset below exercises instead) --
        # same technique as `test_phenomxp.jl`'s equivalent check. Evaluated away from exact
        # injection (`m1` perturbed) so `ll != 0` and doubling is a meaningful, non-degenerate
        # check (0 == 2*0 would pass vacuously).
        strain = _manual_h_det_phenomxhm_multi(freqs, ref_physical, GravitationalWaveData.H1, gps, f_ref)
        physical_offset = copy(ref_physical); physical_offset[1] += 0.3
        psd = fill(1e-46, ntime)
        data_one = [LikelihoodData(freqs, strain, psd, GravitationalWaveData.H1, gps)]
        data_two = [LikelihoodData(freqs, strain, psd, GravitationalWaveData.H1, gps),
                    LikelihoodData(freqs, strain, psd, GravitationalWaveData.H1, gps)]
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)

        ll_one = zeros(Float64, 1)
        batched_log_likelihood_kernel!(ll_one, reshape(physical_offset, 11, 1), cfg, data_one; model=IMRPhenomXHM())
        ll_two = zeros(Float64, 1)
        batched_log_likelihood_kernel!(ll_two, reshape(physical_offset, 11, 1), cfg, data_two; model=IMRPhenomXHM())
        @test ll_one[1] < 0.0  # sanity: genuinely away from injection, not accidentally exact
        @test isapprox(ll_two[1], 2 * ll_one[1]; rtol=1e-10, atol=1e-12)
    end

    @testset "noise-free H1+L1 injection recovery: peak at injection, single-parameter perturbations strictly decrease ll" begin
        strains = [_manual_h_det_phenomxhm_multi(freqs, ref_physical, d, gps, f_ref) for d in dets]
        psds = [fill(1e-46, ntime) for _ in dets]
        data = [LikelihoodData(freqs, strains[i], psds[i], dets[i], gps) for i in eachindex(dets)]
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)

        ll0 = zeros(Float64, 1)
        batched_log_likelihood_kernel!(ll0, reshape(ref_physical, 11, 1), cfg, data; model=IMRPhenomXHM())
        @test isapprox(ll0[1], 0.0; atol=1e-6)

        perturbations = [0.5, 0.5, 0.02, 0.02, 0.05, 0.002, 0.05, 0.05, 0.05, 0.05, 0.05]
        for i in 1:11
            p2 = copy(ref_physical)
            p2[i] += perturbations[i]
            ll2 = zeros(Float64, 1)
            batched_log_likelihood_kernel!(ll2, reshape(p2, 11, 1), cfg, data; model=IMRPhenomXHM())
            @test ll2[1] < ll0[1]
        end
    end

    @testset "dt_geo generalization: H1/L1 time delays genuinely differ at a non-overhead sky position, recovery still works" begin
        # Direct, targeted check of the `t_c -> t_c + dt_geo` generalization (see
        # KernelFused.jl's `_fused_waveform_likelihood_kernel_phenomxhm_multi!` comment block for
        # the derivation): pick a sky position where H1's and L1's own light-travel-time offsets
        # from the geocenter genuinely differ by a physically meaningful amount (comparable to
        # the ~10ms H1-L1 baseline light-travel time), confirm that difference is real (not
        # accidentally ~0, which would make this test vacuous), then confirm noise-free injection
        # recovery still succeeds -- if dt_geo were applied with the wrong sign, not applied per
        # detector, or dropped entirely, this specific test would catch it (test 1 above would
        # not, since it uses two identical detectors with identical dt_geo).
        ra_test, dec_test = 2.1, 0.4
        dt_H1 = GravitationalWaveData.time_delay_from_geocenter(GravitationalWaveData.H1, ra_test, dec_test, gps)
        dt_L1 = GravitationalWaveData.time_delay_from_geocenter(GravitationalWaveData.L1, ra_test, dec_test, gps)
        @test abs(dt_H1 - dt_L1) > 1e-3  # comfortably resolvable given this test's ~1Hz-spaced, 40-bin grid

        physical_sky = [32.0, 8.0, 0.3, -0.1, 0.5, 0.015, 0.4, ra_test, dec_test, 1.0, 0.6]
        strains = [_manual_h_det_phenomxhm_multi(freqs, physical_sky, d, gps, f_ref) for d in dets]
        psds = [fill(1e-46, ntime) for _ in dets]
        data = [LikelihoodData(freqs, strains[i], psds[i], dets[i], gps) for i in eachindex(dets)]
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)

        ll0 = zeros(Float64, 1)
        batched_log_likelihood_kernel!(ll0, reshape(physical_sky, 11, 1), cfg, data; model=IMRPhenomXHM())
        @test isapprox(ll0[1], 0.0; atol=1e-6)

        perturbations = [0.5, 0.5, 0.02, 0.02, 0.05, 0.002, 0.05, 0.05, 0.05, 0.05, 0.05]
        for i in 1:11
            p2 = copy(physical_sky)
            p2[i] += perturbations[i]
            ll2 = zeros(Float64, 1)
            batched_log_likelihood_kernel!(ll2, reshape(p2, 11, 1), cfg, data; model=IMRPhenomXHM())
            @test ll2[1] < ll0[1]
        end
    end

    @testset "mismatched gps_time across detectors is rejected" begin
        strain_small = zeros(ComplexF64, ntime)
        psd_small = ones(Float64, ntime)
        data_bad = [LikelihoodData(freqs, strain_small, psd_small, GravitationalWaveData.H1, gps),
                    LikelihoodData(freqs, strain_small, psd_small, GravitationalWaveData.L1, gps + 1.0)]
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)
        ll = zeros(Float64, 1)
        params = reshape(ref_physical, 11, 1)
        @test_throws ArgumentError batched_log_likelihood_kernel!(ll, params, cfg, data_bad; model=IMRPhenomXHM())
    end

    @testset "row-count assertion: multi-detector IMRPhenomXHM rejects IMRPhenomXP's 15-row params" begin
        strain_small = zeros(ComplexF64, ntime)
        psd_small = ones(Float64, ntime)
        data = [LikelihoodData(freqs, strain_small, psd_small, GravitationalWaveData.H1, gps),
                LikelihoodData(freqs, strain_small, psd_small, GravitationalWaveData.L1, gps)]
        cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)
        ll = zeros(Float64, 1)
        params15 = reshape(vcat(ref_physical, [0.0, 0.0, 0.0, 0.0]), 15, 1)
        @test_throws AssertionError batched_log_likelihood_kernel!(ll, params15, cfg, data; model=IMRPhenomXHM())
    end
end
