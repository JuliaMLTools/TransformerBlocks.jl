# one head of self-attention

struct Head
    key
    query
    value
    dropout
    inv_sqrt_dₖ
end

Functors.@functor Head (key,query,value,dropout)

function Head(input_dim, head_size; dropout=0)
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