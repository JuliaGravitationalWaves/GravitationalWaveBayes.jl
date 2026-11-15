# GravitationalWaveBayes.jl
<img src="docs/src/assets/logo.svg" alt="GravitationalWaveBayes.jl logo" width="160" align="right">

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliagravitationalwaves.github.io/GravitationalWaveBayes.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliagravitationalwaves.github.io/GravitationalWaveBayes.jl/dev/)
[![Build Status](https://github.com/JuliaGravitationalWaves/GravitationalWaveBayes.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaGravitationalWaves/GravitationalWaveBayes.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaGravitationalWaves/GravitationalWaveBayes.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaGravitationalWaves/GravitationalWaveBayes.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

A Julia library for GPU-native, differentiable gravitational-wave parameter estimation: batched
likelihoods written as KernelAbstractions kernels (CPU, CUDA), gradients by Enzyme, and
custom batched samplers (NUTS, parallel tempering, adaptive tempered SMC) that move thousands of
walkers at once. Waveforms come from [GravitationalWaveforms.jl](https://github.com/JuliaGravitationalWaves/GravitationalWaveforms.jl),
detector data and I/O from [GravitationalWaveData.jl](https://github.com/JuliaGravitationalWaves/GravitationalWaveData.jl).
Dependency direction is one-way: `GravitationalWaveData` <- `GravitationalWaveforms` <- `GravitationalWaveBayes`.

## Models

IMRPhenomD, IMRPhenomXAS, IMRPhenomXHM, IMRPhenomXP, IMRPhenomXPHM, SEOBNRv5HM, SEOBNRv5PHM, the
post-adiabatic SPA family (SEOBNRv5AlignedSpinF2, SEOBNRv5IMRAlignedSpinF2) and AlignedSpinPNHamF2.
Likelihood options:
phase and distance marginalization, calibration uncertainty (sampled spline nodes, or marginalized for the
precessing HM models), detector-frame time reference, multi-detector data, relative binning.

## Usage

Runs are driven by TOML configs:

```
julia --project=. scripts/run_from_config.jl scripts/configs/<name>.toml
```

[`scripts/configs/README.md`](scripts/configs/README.md) documents every config field. GPU runs load
`CUDA.jl` in the environment; the device paths activate through the package extension. NVIDIA GPUs
are the only tested and supported GPUs.

## Layout

| file | contents |
|---|---|
| `Types.jl` | `PEConfig` (grid size, walkers, likelihood options), `LikelihoodData`, `PEState` |
| `DataIO.jl`, `Calibration.jl`, `Injection.jl` | GWOSC strain to conditioned frequency-domain data; calibration splines; injections and SNRs |
| `KernelFused.jl` | dense fused waveform + likelihood kernels for every model, and the likelihood conventions |
| `RelativeBinning.jl` | relative-binning (heterodyned) likelihoods |
| `*BinParallel.jl`, `SEOBNR*Device*.jl` | GPU paths: one thread per (walker, frequency bin), device-built SEOBNR tables and their gradients |
| `BatchedNUTS.jl` | `gw_logp_and_grad` (likelihood + prior + reparametrization + gradient closures), run drivers |
| `NUTSCore.jl`, `SMCCore.jl`, `Flows.jl` | batched NUTS / parallel tempering, tempered SMC, normalizing-flow proposals |
| `Priors.jl`, `AnalyticInit.jl`, `Diagnostics.jl` | priors, Fisher-metric initialization, R-hat / ESS |
| `GPU.jl`, `KernelCache.jl` | device hooks set by the extensions; the on-disk compiled-kernel cache |

All under `src/GravitationalWaveBayes/`.

## GPU kernel cache

Compiling the CUDA kernels (in particular the Enzyme forward-mode gradient kernels) takes minutes in every
fresh Julia process. Setting `GRAVITATIONALWAVEBAYES_KERNEL_CACHE=1` (cache in `~/.cache/gravitationalwavebayes/kernels`, or
`$XDG_CACHE_HOME/gravitationalwavebayes/kernels`) or `GRAVITATIONALWAVEBAYES_KERNEL_CACHE=<dir>` stores the compiled kernel images on
disk and reuses them in later processes. Entries are keyed per kernel: the kernel signature, the compiler
configuration, the GPU, the toolchain/Julia versions, Enzyme's codegen switches and the active
Project/Manifest/LocalPreferences files, plus the kernel's own dependencies on `dev`'d/path packages and
`Main`, recorded at compile time from the compiled code instances and their inference edges: the lowered
code of every method it reaches (inlined or not, including Enzyme-differentiated code), the method lists of
the functions it calls (a new method that could change dispatch is a miss), referenced constants and the
definitions of the types involved. Editing one function therefore recompiles only the kernels that reach
it; comment, docstring and line-number changes recompile nothing. Up to six variants per kernel are kept,
so switching back and forth between two versions of a function hits. `GRAVITATIONALWAVEBAYES_KERNEL_CACHE_LOG=1` logs
every hit and miss with the changed dependencies; `GRAVITATIONALWAVEBAYES_KERNEL_CACHE_VERIFY=1` recompiles every hit and
reports (and replaces) any that differs from the cached entry beyond Julia's per-session function-name
counters. Clear the cache by deleting the directory. The hook (`ext/GravitationalWaveBayesCUDAExt/kernel_cache.jl`) targets CUDA.jl 6.4.1 / GPUCompiler
2.10.0 internals and disables itself with a warning on other versions. See `GravitationalWaveBayes.KernelCache`.

For development, `GRAVITATIONALWAVEBAYES_FAST_COMPILE=1` keeps a fixed set of large `@inline` waveform functions (the EOB
Hamiltonian gradient and RHS, post-adiabatic equations, HM attachment, ...) as calls in device code, so ptxas
compiles several moderate functions instead of one huge one per kernel: on a V100 the cold start of the SEOBNRv5
(2,2) + HM device likelihoods drops from ~640 s to ~380 s, at the price of ~10 % slower SEOBNR kernels at 4096
walkers (HM ~3 %); results agree with the default build to ~1e-12 (logL) / ~1e-8 (gradients, relative to the
largest component). It also affects Enzyme-differentiated CPU code. Leave it off for production runs. Fast-mode
kernels are cached separately (a temporary cache directory is used when `GRAVITATIONALWAVEBAYES_KERNEL_CACHE` is unset). See
`ext/GravitationalWaveBayesCUDAExt/fast_compile.jl`.
