using SDPSymmetryReduction
using Documenter
using Literate
using Test

DocMeta.setdocmeta!(SDPSymmetryReduction, :DocTestSetup, :(using SDPSymmetryReduction); recursive=true)

## Use Literate.jl to generate examples (functions modified from https://github.com/jump-dev/JuMP.jl/blob/master/docs/make.jl)

function _file_list(full_dir, relative_dir, extension)
    return map(
        file -> joinpath(relative_dir, file),
        filter(file -> endswith(file, extension), sort(readdir(full_dir))),
    )
end

"""
    _include_sandbox(filename)
Include the `filename` in a temporary module that acts as a sandbox. (Ensuring
no constants or functions leak into other files.)
"""
function _include_sandbox(filename)
    mod = @eval module $(gensym()) end
    return Base.include(mod, filename)
end

function _literate_directory(dir)
    rm.(_file_list(dir, dir, ".md"))
    for filename in _file_list(dir, dir, ".jl")
        # `include` the file to test it before `#src` lines are removed. It is
        # in a testset to isolate local variables between files.
        Test.@testset "$(filename)" begin
            _include_sandbox(filename)
        end
        Literate.markdown(
            filename,
            dir;
            documenter = true,
            credit = true,
        )
    end
    return nothing
end

_literate_directory.(joinpath(@__DIR__, "src", "examples"))

# Generate docs

makedocs(;
    modules=[SDPSymmetryReduction],
    authors="Daniel Brosch <daniel.brosch@outlook.com> and contributors",
    repo="https://github.com/DanielBrosch/SDPSymmetryReduction.jl/blob/{commit}{path}#{line}",
    sitename="SDPSymmetryReduction.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://DanielBrosch.com/SDPSymmetryReduction.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "examples/ErdosRenyiThetaFunction.md",
            "examples/QuadraticAssignmentProblems.md",
            "examples/ReduceAndSolveJuMP.md"
        ]
    ],
)
##
deploydocs(;
    repo="github.com/DanielBrosch/SDPSymmetryReduction.jl",
    devbranch="main",
)
