struct Block
    sa
    ffwd
    ln1
    ln2
end

Functors.@functor Block

"""
    Block(input_dim; num_heads=1, head_size=(input_dim÷num_heads), dropout=0)

Initializes an instance of the **`Block`** type, representing a transformer block.
"""
function Block(input_dim; num_heads=1, head_size=(input_dim÷num_heads), dropout=0)
    @assert num_heads > 0
    @assert head_size == (input_dim ÷ num_heads)
    @assert input_dim > 0
    Block(
        MultiheadAttention(input_dim, num_heads; dropout=dropout),        
        FeedForward(input_dim; dropout=dropout),
        LayerNorm(input_dim),
        LayerNorm(input_dim),
    )
end


"""
    (::Block)(x; mask=nothing)

A **`Block`** instance accepts an input array **`x`** of dimensions (C, T, B) and outputs an array of dimensions (HS, T, B). "C" is the channel size (embedding dimension). "T" is the block size (number of input tokens). "B" is the batch size.

The following keyword arguments are supported:
- `mask` (Defaults to nothing. Must be of dimensions (T, T).)

## Examples:

```julia
C,T,B = 8,3,4
block = Block(C)
@assert size(block(rand(Float32, C,T,B))) == (C,T,B)
```
"""
function (m::Block)(x; mask=nothing)
    x + m.sa(m.ln1(x); mask=mask) + m.ffwd(m.ln2(x))
end