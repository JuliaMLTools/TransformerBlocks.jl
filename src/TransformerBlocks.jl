module TransformerBlocks

export func

"""
    func(x)

Returns double the number a `x` plus `1`.
"""
func(x) = 2x + 1


include("imports.jl")
include("util.jl")

include("head.jl")
export Head

include("feed_forward.jl")
export FeedForward

include("multihead_attn.jl")
export MultiheadAttention

include("block.jl")
export Block

include("blocklist.jl")
export BlockList

import SnoopPrecompile
include("other/precompile.jl")

end