# Samplers

```@index
Pages = ["sampling.md"]
```

## NUTS, parallel tempering and SMC

```@autodocs
Modules = [GravitationalWaveBayes.NUTSCore]
Filter = t -> any(n -> getfield(GravitationalWaveBayes, n) === t, names(GravitationalWaveBayes))
```

## Run drivers

```@autodocs
Modules = [GravitationalWaveBayes.BatchedNUTS]
Filter = t -> any(n -> getfield(GravitationalWaveBayes, n) === t, names(GravitationalWaveBayes))
```

## Normalizing-flow proposals

```@autodocs
Modules = [GravitationalWaveBayes.Flows]
Filter = t -> any(n -> getfield(GravitationalWaveBayes, n) === t, names(GravitationalWaveBayes))
```

## Analytic initialization

```@autodocs
Modules = [GravitationalWaveBayes.AnalyticInit]
Filter = t -> any(n -> getfield(GravitationalWaveBayes, n) === t, names(GravitationalWaveBayes))
```
