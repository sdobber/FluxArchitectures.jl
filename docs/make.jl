using FluxArchitectures
using Documenter

DocMeta.setdocmeta!(FluxArchitectures, :DocTestSetup, :(using FluxArchitectures); recursive=true)

makedocs(;
    modules=[FluxArchitectures],
    authors="Sören Dobberschütz and contributors",
    repo="https://github.com/sdobber/FluxArchitectures.jl/blob/{commit}{path}#{line}",
    sitename="FluxArchitectures.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sdobber.github.io/FluxArchitectures.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/sdobber/FluxArchitectures.jl",
)
