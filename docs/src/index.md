# TransformerBlocks.jl
### Simple, blazing fast, transformer components.


## Basic Usage

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
mask = (1 .- triu(ones(Float32, T, T))) .* (-1f9)
@assert size(block(x; mask=mask)) == (C, T, B)

# Example 3: Sequential blocks
num_layers = 3
blocks = BlockList([Block(C) for _ in 1:num_layers])
@assert size(blocks(x)) == (C, T, B)
```


## Components

```@docs
Block
```

```@docs
Block(input_dim; num_heads=1, head_size=(input_dim÷num_heads), dropout=0)
```

```@docs
func(x)
```

## API index

```@index
```