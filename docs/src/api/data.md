# Data, likelihoods and priors

```@index
Pages = ["data.md"]
```

## Configuration and data types

```@autodocs
Modules = [GravitationalWaveBayes.Types]
Filter = t -> any(n -> getfield(GravitationalWaveBayes, n) === t, names(GravitationalWaveBayes))
```

## Data conditioning and I/O

```@autodocs
Modules = [GravitationalWaveBayes.DataIO]
Filter = t -> any(n -> getfield(GravitationalWaveBayes, n) === t, names(GravitationalWaveBayes))
```

## Injections and SNRs

```@autodocs
Modules = [GravitationalWaveBayes.Injection]
Filter = t -> any(n -> getfield(GravitationalWaveBayes, n) === t, names(GravitationalWaveBayes))
```

## Calibration uncertainty

```@autodocs
Modules = [GravitationalWaveBayes.Calibration]
Filter = t -> any(n -> getfield(GravitationalWaveBayes, n) === t, names(GravitationalWaveBayes))
```

## Priors

```@autodocs
Modules = [GravitationalWaveBayes.Priors]
Filter = t -> any(n -> getfield(GravitationalWaveBayes, n) === t, names(GravitationalWaveBayes))
```

## Likelihood kernels

```@autodocs
Modules = [GravitationalWaveBayes.KernelFused]
Filter = t -> any(n -> getfield(GravitationalWaveBayes, n) === t, names(GravitationalWaveBayes))
```

## Relative binning

```@autodocs
Modules = [GravitationalWaveBayes.RelativeBinning]
Filter = t -> any(n -> getfield(GravitationalWaveBayes, n) === t, names(GravitationalWaveBayes))
```
