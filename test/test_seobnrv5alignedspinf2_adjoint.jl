using Test
using GravitationalWaveBayes
using GravitationalWaveforms

# GravitationalWaveBayes.SEOBNRv5AlignedSpinF2Adjoint: hand-derived forward-mode sensitivity (JVP) of
# GravitationalWaveforms._seobv5f2_coeffs's full return value w.r.t. its 4 intrinsic parameters
# (mc, eta, chi1, chi2), built to avoid ever calling Enzyme.autodiff on that pipeline's own
# loopy/fixed-buffer internals (see SEOBNRv5AlignedSpinF2Adjoint.jl's own module docstring for the
# full design rationale and the three upstream Enzyme/CUDA bugs this sidesteps). This test checks
# the resulting JVP against central finite differences on the real `_seobv5f2_coeffs` output --
# the same validation bar used everywhere else in this codebase, but at the level of this
# lower-level primitive rather than the full `gw_logp_and_grad` closure.
#
# The single LAST real grid node (index `nreal`) is excluded from the relative-error check on the
# ts/phis/omegas TABLES: the true `d(omega)/d(param)` etc. cross near zero there (it's the
# post-adiabatic table's own endpoint, closest to the plunge handoff), so relative error stops
# being a meaningful metric -- absolute error there is still tiny (~1e-8), confirmed during
# development, not just assumed here. `M_sec`/`eta`/`M`/`fmin`/`fmax` are single scalars with no
# such near-zero-crossing risk at this configuration, checked directly.

@testset "SEOBNRv5AlignedSpinF2Adjoint: seobv5f2_coeffs_jvp vs finite differences" begin
    model = GravitationalWaveforms.SEOBNRv5AlignedSpinF2()
    mc, eta, chi1, chi2 = 28.0961, 0.2479, 0.3, -0.2

    function fd_coeffs(dmc, deta, dchi1, dchi2, h)
        cp = GravitationalWaveforms._seobv5f2_coeffs(model, mc + h * dmc, eta + h * deta, chi1 + h * dchi1, chi2 + h * dchi2)
        cm = GravitationalWaveforms._seobv5f2_coeffs(model, mc - h * dmc, eta - h * deta, chi1 - h * dchi1, chi2 - h * dchi2)
        return (
            omegas = (cp.omegas .- cm.omegas) ./ (2h),
            ts = (cp.ts .- cm.ts) ./ (2h),
            phis = (cp.phis .- cm.phis) ./ (2h),
            M_sec = (cp.M_sec - cm.M_sec) / (2h),
            eta = (cp.eta - cm.eta) / (2h),
            M = (cp.M - cm.M) / (2h),
            fmin = (cp.fmin - cm.fmin) / (2h),
            fmax = (cp.fmax - cm.fmax) / (2h),
            nreal = cp.nreal,
        )
    end

    reference = GravitationalWaveforms._seobv5f2_coeffs(model, mc, eta, chi1, chi2)
    @test reference.success

    directions = [
        ("d/dmc", 1.0, 0.0, 0.0, 0.0),
        ("d/deta", 0.0, 1.0, 0.0, 0.0),
        ("d/dchi1", 0.0, 0.0, 1.0, 0.0),
        ("d/dchi2", 0.0, 0.0, 0.0, 1.0),
    ]

    for (name, dmc, deta, dchi1, dchi2) in directions
        dcoeffs = GravitationalWaveBayes.seobv5f2_coeffs_jvp(model, mc, eta, chi1, chi2, dmc, deta, dchi1, dchi2)
        fd = fd_coeffs(dmc, deta, dchi1, dchi2, 1e-6)
        # fmin / fmax (~20 Hz, d/dmc ~1e-4) change by ~1e-10 over a 1e-6 step: that difference is roundoff noise
        # (+-1% between neighbouring steps); at 1e-4 the central difference agrees with the JVP to ~1e-5.
        fd_f = fd_coeffs(dmc, deta, dchi1, dchi2, 1e-4)
        idx = 1:(fd.nreal - 1)

        @testset "$name" begin
            @test isapprox(dcoeffs.omegas[idx], fd.omegas[idx]; rtol = 5e-3, atol = 1e-8)
            @test isapprox(dcoeffs.ts[idx], fd.ts[idx]; rtol = 5e-3, atol = 1e-8)
            @test isapprox(dcoeffs.phis[idx], fd.phis[idx]; rtol = 5e-3, atol = 1e-8)
            @test isapprox(dcoeffs.M_sec, fd.M_sec; rtol = 1e-4, atol = 1e-8)
            @test isapprox(dcoeffs.eta, fd.eta; rtol = 1e-4, atol = 1e-8)
            @test isapprox(dcoeffs.M, fd.M; rtol = 1e-4, atol = 1e-8)
            @test isapprox(dcoeffs.fmin, fd_f.fmin; rtol = 5e-3, atol = 1e-8)
            @test isapprox(dcoeffs.fmax, fd_f.fmax; rtol = 5e-3, atol = 1e-8)
        end
    end
end
