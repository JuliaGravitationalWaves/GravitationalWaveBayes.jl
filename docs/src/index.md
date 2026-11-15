```@meta
CurrentModule = GravitationalWaveBayes
```

# GravitationalWaveBayes.jl

A Julia library for GPU-native, differentiable gravitational-wave parameter estimation: batched
likelihoods written as KernelAbstractions kernels (CPU, CUDA), gradients by Enzyme, and custom
batched samplers (NUTS, parallel tempering, adaptive tempered SMC) that move thousands of walkers
at once. Waveforms come from
[GravitationalWaveforms.jl](https://github.com/JuliaGravitationalWaves/GravitationalWaveforms.jl),
detector data and I/O from
[GravitationalWaveData.jl](https://github.com/JuliaGravitationalWaves/GravitationalWaveData.jl).

## Models

IMRPhenomD, IMRPhenomXAS, IMRPhenomXHM, IMRPhenomXP, IMRPhenomXPHM, SEOBNRv5HM, SEOBNRv5PHM, the
post-adiabatic SPA family (SEOBNRv5AlignedSpinF2, SEOBNRv5IMRAlignedSpinF2) and AlignedSpinPNHamF2.

Likelihood options: phase and distance marginalization, calibration uncertainty (sampled spline
nodes, or marginalized for the precessing higher-mode models), detector-frame time reference,
multi-detector data and relative binning.

## Usage

Runs are driven by TOML configs:

```
julia --project=. scripts/run_from_config.jl scripts/configs/<name>.toml
```

[`scripts/configs/README.md`](https://github.com/JuliaGravitationalWaves/GravitationalWaveBayes.jl/blob/main/scripts/configs/README.md)
documents every config field, and `scripts/configs/` holds the configs used for the GW150914
analyses. GPU runs load `CUDA.jl` in the environment; the device paths activate through the package
extension. NVIDIA GPUs are the only tested and supported GPUs.

## GPU kernel cache

Compiling the CUDA kernels takes minutes in every fresh Julia process. Set
`GRAVITATIONALWAVEBAYES_KERNEL_CACHE=1` (or a directory) to store the compiled kernel images on disk
and reuse them in later processes; see the
[README](https://github.com/JuliaGravitationalWaves/GravitationalWaveBayes.jl#gpu-kernel-cache).

The API reference covers the exported interface: [data, likelihoods and priors](api/data.md),
[samplers](api/sampling.md) and [diagnostics and utilities](api/tools.md).
