using Documenter, TransformerBlocks
makedocs(
    sitename="TransformerBlocks.jl",
    modules=[TransformerBlocks],
)
deploydocs(
    repo = "github.com/JuliaMLTools/TransformerBlocks.jl.git",
)