struct FeedForward{N}
    net::N
end

Functors.@functor FeedForward

"""
    FeedForward(input_dim::Integer; dropout=0)

Initializes an instance of the **`FeedForward`** type, representing a simple linear layer followed by a non-linearity.

The following keyword arguments are supported:
- `dropout` (Defaults to 0)

A **`FeedForward`** instance accepts an input array **`x`** of dimensions (C, T, B) and outputs an array of dimensions (C, T, B). "C" is the channel size (embedding dimension). "T" is the block size (number of input tokens). "B" is the batch size.

## Examples:

```julia
C,T,B = 8,3,4
ff = FeedForward(C)
@assert size(ff(rand(Float32, C, T, B))) == (C, T, B)
```
"""
function FeedForward(input_dim::Integer; dropout=0)
    @assert input_dim > 0
    FeedForward(
        Chain(
            Dense(input_dim => 4input_dim, relu),
            Dense(4input_dim => input_dim),
            Dropout(dropout),
        )
    )
end

(m::FeedForward)(x) = m.net(x)