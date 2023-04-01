# TransformerBlocks.jl

[![][docs-stable-img]][docs-stable-url]
[![][docs-dev-img]][docs-dev-url]

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://juliamltools.github.io/TransformerBlocks.jl/dev/

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://juliamltools.github.io/TransformerBlocks.jl/stable/

This package aims to be a consise, performant implementation of the pseudocode found in 
[Formal Algorithms for Transformers](https://arxiv.org/pdf/2207.09238.pdf).

## Example usage

```julia
using TransformerBlocks

# C: input embedding dimension
# T: block size (sequence length)
# B: batch size
C, T, B = 10, 5, 3
x = rand(Float32, C, T, B)

# Example 1: Transformer block
block = Block(C)
@assert size(block(x)) == (C, T, B)

# Example 2: Block with mask
using LinearAlgebra
mask = tril(fill(-Inf, T, T), -1)
@assert size(block(x; mask=mask)) == (C, T, B)

# Example 3: Sequential blocks
num_layers = 3
blocks = BlockList([Block(C) for _ in 1:num_layers])
@assert size(blocks(x)) == (C, T, B)
```


## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add TransformerBlocks
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("TransformerBlocks")
```

## Project Status

The package is tested against, and being developed for, Julia `1.8` and above on Linux, macOS, and Windows.
