using Test
using Random
using GravitationalWaveBayes

# Planck15_LAL distances against astropy 8.0.1 FlatLambdaCDM(H0 = lal.H0_SI, Om0 = lal.OMEGA_M) (LAL 7.7.0), i.e. bilby's
# get_cosmology("planck15_lal"); the source-frame prior against its closed form; reweighting reproduces GWTC-2.1's
# `_cosmo` GW150914 release from `_nocosmo` (checked 2026-10-19: d_L quantiles within 0.2 Mpc, KS D = 0.0009; not run
# here, the release files are 290 MB).
@testset "Cosmology: Planck15_LAL and the source-frame distance prior" begin
    for (z, d) in ((0.01, 44.4908572861), (0.09, 424.0225331976), (0.5, 2914.0041032710), (1.0, 6785.9445039778),
                   (3.0, 26043.4207975417))
        @test isapprox(luminosity_distance(z), d; rtol=1e-11)
        @test isapprox(redshift_at_luminosity_distance(d), z; rtol=1e-11)
    end
    # Low-z limit: p(d) / d^2 -> const, so the weights flatten towards d -> 0 and fall with distance.
    w = source_frame_prior_weights([10.0, 100.0, 1000.0, 5000.0])
    @test w[1] == 1.0 && issorted(w; rev=true) && 0.85 < w[2] < 0.95 && w[4] < 0.1   # 1, 0.906, 0.418, 0.065
    # Closed form at one point: w ∝ (dVc/dz / (1+z)) / (dd_L/dz) / d_L^2
    z = 0.5; DH = 299792458.0 / 2.200489137532724e-18 / 3.085677581491367e22; E = sqrt(0.3065 * 1.5^3 + 0.6935)
    Dc = luminosity_distance(z) / 1.5
    p = 4pi * Dc^2 * DH / E / 1.5 / (Dc + 1.5 * DH / E) / luminosity_distance(z)^2
    z0 = 0.001; E0 = sqrt(0.3065 * 1.001^3 + 0.6935); Dc0 = luminosity_distance(z0) / 1.001
    p0 = 4pi * Dc0^2 * DH / E0 / 1.001 / (Dc0 + 1.001 * DH / E0) / luminosity_distance(z0)^2
    ww = source_frame_prior_weights([luminosity_distance(z0), luminosity_distance(z)])
    @test isapprox(ww[2] / ww[1], p / p0; rtol=1e-10)
    keep = reweight_to_source_frame(Xoshiro(1), fill(100.0, 1000))
    @test length(keep) == 1000                                  # equal weights: nothing rejected
end
