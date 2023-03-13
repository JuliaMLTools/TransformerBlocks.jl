"""
    BlockList

A sequence of transformer blocks composed together
"""
struct BlockList
    list
end

Functors.@functor BlockList

function (m::BlockList)(x; mask=nothing)
    foldl((i,fn)->fn(i; mask=mask), m.list; init=x)
end