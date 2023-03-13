# Transformer block: communication followed by computation

"""
    Block

The basic transformer block.
"""
struct Block
    sa
    ffwd
    ln1
    ln2
end

Functors.@functor Block

"""
    Block(input_dim; num_heads=1, head_size=(input_dim÷num_heads), dropout=0)

Creates a transformer block.
"""
function Block(input_dim; num_heads=1, head_size=(input_dim÷num_heads), dropout=0)
    Block(
        MultiheadAttention(input_dim, num_heads; dropout=dropout),        
        FeedForward(input_dim; dropout=dropout),
        LayerNorm(input_dim),
        LayerNorm(input_dim),
    )
end

function (m::Block)(x; mask=nothing)
    x + m.sa(m.ln1(x); mask=mask) + m.ffwd(m.ln2(x))
end