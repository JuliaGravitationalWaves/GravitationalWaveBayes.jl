using Documenter
using GravitationalWaveBayes

DocMeta.setdocmeta!(GravitationalWaveBayes, :DocTestSetup, :(using GravitationalWaveBayes); recursive=true)

makedocs(;
    sitename="GravitationalWaveBayes.jl",
    modules=[GravitationalWaveBayes],
    authors="Paul Hopkins <paul.hopkins256@gmail.com>",
    format=Documenter.HTML(;
        canonical="https://juliagravitationalwaves.github.io/GravitationalWaveBayes.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => [
            "Data, likelihoods and priors" => "api/data.md",
            "Samplers" => "api/sampling.md",
            "Diagnostics and utilities" => "api/tools.md",
        ],
    ],
    checkdocs=:exports,
    warnonly=[:missing_docs, :cross_references],
)

deploydocs(;
    repo="github.com/JuliaGravitationalWaves/GravitationalWaveBayes.jl",
    devbranch="main",
)
