using Documenter, TransformerBlocks
makedocs(sitename="TransformerBlocks.jl")
deploydocs(
    dirname = "somedir",
    repo = "github.com/JuliaMLTools/TransformerBlocks.jl.git",
)