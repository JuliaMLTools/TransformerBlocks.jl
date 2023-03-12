struct FeedForward
    net
end

Functors.@functor FeedForward

function FeedForward(input_dim::Integer; dropout=0)
    FeedForward(
        Chain(
            Dense(input_dim => 4input_dim, relu),
            Dense(4input_dim => input_dim),
            Dropout(dropout),
        )
    )
end

(m::FeedForward)(x) = m.net(x)