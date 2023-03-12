# multiple heads of self-attention in parallel
struct MultiheadAttention
    heads
    proj
    dropout
end

Functors.@functor MultiheadAttention

function MultiheadAttention(input_dim, num_heads; head_size=(input_dim ÷ num_heads), dropout=0)
    MultiheadAttention(
        [Head(input_dim, head_size; dropout=dropout) for _ in 1:num_heads],
        Dense(head_size * num_heads, input_dim),
        Dropout(dropout),
    )
end

function (m::MultiheadAttention)(x; mask=nothing)
    # TODO: replace with mapreduce
    heads_out = reduce(vcat, [head(x; mask=mask) for head in m.heads])
    m.dropout(m.proj(heads_out))
end