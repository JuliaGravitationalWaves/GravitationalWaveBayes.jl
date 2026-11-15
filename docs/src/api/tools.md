# Diagnostics and utilities

```@index
Pages = ["tools.md"]
```

```@autodocs
Modules = [GravitationalWaveBayes]
```

## Convergence diagnostics

```@autodocs
Modules = [GravitationalWaveBayes.Diagnostics]
Filter = t -> any(n -> getfield(GravitationalWaveBayes, n) === t, names(GravitationalWaveBayes))
```

## Cosmology and source-frame reweighting

```@autodocs
Modules = [GravitationalWaveBayes.Cosmology]
Filter = t -> any(n -> getfield(GravitationalWaveBayes, n) === t, names(GravitationalWaveBayes))
```

## Device hooks

```@autodocs
Modules = [GravitationalWaveBayes.GPU]
Filter = t -> any(n -> getfield(GravitationalWaveBayes, n) === t, names(GravitationalWaveBayes))
```
