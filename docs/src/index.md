```@meta
CurrentModule = GravitationalWaveBayes
```

# GravitationalWaveBayes.jl

GPU-native, differentiable parameter estimation for gravitational-wave signals, with a custom
batched NUTS sampler and Phenom/EOB waveform models (via
[GravitationalWaveforms.jl](https://github.com/JuliaGravitationalWaves/GravitationalWaveforms.jl)) on top of
[GravitationalWaveData.jl](https://github.com/JuliaGravitationalWaves/GravitationalWaveData.jl)'s data containers and I/O.

See the [API Reference](api.md) for the exported interface.
