using Documenter, TransformerBlocks
makedocs(
    sitename="TransformerBlocks.jl",
    format=Documenter.HTML(),
    modules=[TransformerBlocks],
)
deploydocs(
    repo = "github.com/JuliaMLTools/TransformerBlocks.jl.git",
)