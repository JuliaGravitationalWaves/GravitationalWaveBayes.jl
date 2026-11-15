# matched-gw150914 inputs (2026-10-11): bilby-convention window factor, PSD files, calibration spline.
using Test, GravitationalWaveBayes, GravitationalWaveData

@testset "DataIO: window factor and PSD files" begin
    # Tukey(alpha) mean square -> 1 - 5 alpha / 8 for long windows (0.875 at alpha 0.2, bilby's GW150914 value)
    wf = GravitationalWaveBayes.DataIO.window_factor(1 / 4096; duration=4.0, post_trigger=2.0, alpha=0.2)
    @test wf ≈ 1 - 5 * 0.2 / 8 rtol = 1e-3
    @test GravitationalWaveBayes.DataIO.window_factor(1 / 4096; duration=4.0, post_trigger=2.0, alpha=0.0) ≈ 1

    path = tempname()
    open(path, "w") do io
        println(io, "# test PSD")
        for f in 0:0.5:100; println(io, f, " ", 1e-46 * (1 + f)); end
    end
    S = GravitationalWaveBayes.DataIO.read_psd_file(path, [10.0, 10.25, 99.5])
    @test S ≈ 1e-46 .* (1 .+ [10.0, 10.25, 99.5])           # exact on nodes, linear between them
    @test_throws ArgumentError GravitationalWaveBayes.DataIO.read_psd_file(path, [10.0, 120.0])
end

@testset "Calibration: bilby CubicSpline (not-a-knot in log10 f)" begin
    nodes = calibration_nodes(20, 896, 10)
    @test nodes[1] ≈ 20 && nodes[end] ≈ 896
    f = collect(range(20, 896; step=0.25))
    W = calibration_weights(nodes, f)
    x = log10.(nodes); t = log10.(f)
    # interpolates the node values
    y = sin.(3 .* x)
    @test calibration_weights(nodes, nodes) * y ≈ y atol = 1e-12
    # a not-a-knot spline reproduces any cubic in log10 f exactly (a natural spline would not)
    p(u) = 0.3 - 0.2u + 0.15u^2 - 0.04u^3
    @test W * p.(x) ≈ p.(t) atol = 1e-12
    # zero outside the node range (scipy interp1d fill_value = 0); also checked against scipy itself to 1e-15
    @test all(iszero, calibration_weights(nodes, [19.9, 900.0]))
    # calibration factor: unity at zero, (1 + dA) e^{i dphi} to second order
    C0 = calibration_factor(W, zeros(10), zeros(10))
    @test all(==(1), C0)
    C = calibration_factor(W, fill(0.02, 10), fill(0.01, 10))
    @test abs.(C) ≈ fill(1.02, length(f)) rtol = 1e-12
    @test angle.(C) ≈ fill(2atan(0.005), length(f)) rtol = 1e-12
end
