struct Head{K,Q,V,D,DK}
    key::K
    query::Q
    value::V
    dropout::D
    inv_sqrt_dₖ::DK
end

Functors.@functor Head (key,query,value,dropout)

"""
    Head(input_dim, head_size; dropout=0)

Initializes an instance of the **`Head`** type, representing one head of self-attention.

A **`Head`** instance accepts an input array **`x`** of dimensions (C, T, B) and outputs an array of dimensions (HS, T, B). "C" is the channel size (embedding dimension). "T" is the block size (number of input tokens). "B" is the batch size. "HS" is the head size.

The following keyword arguments are supported:
- `mask` (Defaults to nothing. Must be of dimensions (T, T).)

## Examples:

```julia
C,T,B = 8,3,4
HS = 10
head = Head(C,HS)
@assert size(head(rand(Float32, C,T,B))) == (HS,T,B)
```
"""
function Head(input_dim, head_size; dropout=0)
    @assert input_dim > 0
    @assert head_size > 0
    Head(
        Dense(input_dim, head_size, bias=false),
        Dense(input_dim, head_size, bias=false),
        Dense(input_dim, head_size, bias=false),
        Dropout(dropout),
        Float32(1 / sqrt(head_size)),
    )
end

function (m::Head)(x; mask=nothing)
    C, T, B = size(x)
    k = m.key(x) # (hs,T,B)
    q = m.query(x) # (hs,T,B)
    v = m.value(x) # (hs,T,B)
    wei = transposebatchmul(q, k) .* m.inv_sqrt_dₖ
    if isnothing(mask)
        wei_masked = wei # (T, T, B)
    else
        wei_masked = wei .+ mask # (T, T, B)
    end
    probs_predrop = softmax(wei_masked) # (T, T, B)
    probs = m.dropout(probs_predrop) # (T, T, B)
    batched_mul(v, probs)
end