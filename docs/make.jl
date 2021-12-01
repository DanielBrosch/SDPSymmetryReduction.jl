using SDPSymmetryReduction
using Documenter

DocMeta.setdocmeta!(SDPSymmetryReduction, :DocTestSetup, :(using SDPSymmetryReduction); recursive=true)

makedocs(;
    modules=[SDPSymmetryReduction],
    authors="Daniel Brosch <daniel.brosch@outlook.com> and contributors",
    repo="https://github.com/DanielBrosch/SDPSymmetryReduction.jl/blob/{commit}{path}#{line}",
    sitename="SDPSymmetryReduction.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://DanielBrosch.github.io/SDPSymmetryReduction.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/DanielBrosch/SDPSymmetryReduction.jl",
    devbranch="main",
)
