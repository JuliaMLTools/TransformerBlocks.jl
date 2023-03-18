struct MultiheadAttention{H<:Union{Tuple, NamedTuple, AbstractVector}, P, D}
    heads::H
    proj::P
    dropout::D
end

Functors.@functor MultiheadAttention

"""
    MultiheadAttention(input_dim, num_heads; head_size=(input_dim ÷ num_heads), dropout=0)

Initializes an instance of the **`MultiheadAttention`** type, representing multiple heads of parallel self-attention.

The following keyword arguments are supported:
- `head_size` (Defaults to **`input_dim`** / **`num_heads`**)
- `dropout` (Defaults to 0)

A **`MultiheadAttention`** instance accepts an input array **`x`** of dimensions (C, T, B) and outputs an array of dimensions (C, T, B). "C" is the channel size (embedding dimension). "T" is the block size (number of input tokens). "B" is the batch size.

The following keyword arguments are supported:
- `mask` (Defaults to nothing. Must be of dimensions (T, T).)

## Examples:

```julia
C,T,B = 8,3,4
NH = 4 # Num heads
multihead = MultiheadAttention(C,NH)
@assert size(multihead(rand(Float32, C, T, B))) == (C, T, B)
```
"""
function MultiheadAttention(input_dim, num_heads; head_size=(input_dim ÷ num_heads), dropout=0)
    @assert num_heads > 0
    @assert head_size == (input_dim ÷ num_heads)
    @assert input_dim > 0
    MultiheadAttention(
        [Head(input_dim, head_size; dropout=dropout) for _ in 1:num_heads],
        Dense(head_size * num_heads, input_dim),
        Dropout(dropout),
    )
end

function (m::MultiheadAttention)(x; mask=nothing)
    heads_out = mapreduce(head->head(x; mask=mask), vcat, m.heads)
    m.dropout(m.proj(heads_out))
end