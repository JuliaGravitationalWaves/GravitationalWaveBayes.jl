@testset "Renamed environment variables are refused" begin
    old = "GW" * "BAYES_KERNEL_CACHE"
    @test_throws "GRAVITATIONALWAVEBAYES_KERNEL_CACHE" GravitationalWaveBayes._check_renamed_env(Dict(old => "1"))
    @test GravitationalWaveBayes._check_renamed_env(Dict("GRAVITATIONALWAVEBAYES_KERNEL_CACHE" => "1", "PATH" => "x")) === nothing
end
