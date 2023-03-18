"""
    BlockList(input_dim, num_heads; head_size=(input_dim ÷ num_heads), dropout=0)

Initializes an instance of the **`BlockList`** type, representing a sequence of transformer blocks composed together.

The following keyword arguments are supported:
- `head_size` (Defaults to **`input_dim`** / **`num_heads`**)
- `dropout` (Defaults to 0)

A **`BlockList`** instance accepts an input array **`x`** of dimensions (C, T, B) and outputs an array of dimensions (HS, T, B). "C" is the channel size (embedding dimension). "T" is the block size (number of input tokens). "B" is the batch size.

The following keyword arguments are supported:
- `mask` (Defaults to nothing. Must be of dimensions (T, T).)

## Examples:

```julia
C,T,B = 8,3,4
blocklist = BlockList([Block(C), Block(C)])
@assert size(blocklist(rand(Float32, C,T,B))) == (C,T,B)
```
"""
struct BlockList{T<:Union{Tuple, NamedTuple, AbstractVector}}
    list::T
end

Functors.@functor BlockList

function (m::BlockList)(x; mask=nothing)
    foldl((i,fn)->fn(i; mask=mask), m.list; init=x)
end