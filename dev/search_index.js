var documenterSearchIndex = {"docs":
[{"location":"#TransformerBlocks.jl","page":"TransformerBlocks.jl","title":"TransformerBlocks.jl","text":"","category":"section"},{"location":"","page":"TransformerBlocks.jl","title":"TransformerBlocks.jl","text":"CurrentModule = TransformerBlocks","category":"page"},{"location":"#Simple,-blazing-fast,-transformer-components.","page":"TransformerBlocks.jl","title":"Simple, blazing fast, transformer components.","text":"","category":"section"},{"location":"#Basic-Usage","page":"TransformerBlocks.jl","title":"Basic Usage","text":"","category":"section"},{"location":"","page":"TransformerBlocks.jl","title":"TransformerBlocks.jl","text":"using TransformerBlocks\n\n# C: channel size (embedding dimension)\n# T: block size (sequence length)\n# B: batch size\nC, T, B = 10, 5, 3\nx = rand(Float32, C, T, B)\n\n# Example 1: Transformer block\nblock = Block(C)\n@assert size(block(x)) == (C, T, B)\n\n# Example 2: Block with mask\nusing LinearAlgebra\nmask = (1 .- triu(ones(Float32, T, T))) .* (-1f9)\n@assert size(block(x; mask=mask)) == (C, T, B)\n\n# Example 3: Sequential blocks\nnum_layers = 3\nblocks = BlockList([Block(C) for _ in 1:num_layers])\n@assert size(blocks(x)) == (C, T, B)","category":"page"},{"location":"#API-index","page":"TransformerBlocks.jl","title":"API index","text":"","category":"section"},{"location":"","page":"TransformerBlocks.jl","title":"TransformerBlocks.jl","text":"","category":"page"},{"location":"#Components","page":"TransformerBlocks.jl","title":"Components","text":"","category":"section"},{"location":"","page":"TransformerBlocks.jl","title":"TransformerBlocks.jl","text":"TransformerBlocks.Head\nTransformerBlocks.MultiheadAttention\nTransformerBlocks.FeedForward\nTransformerBlocks.Block\nTransformerBlocks.BlockList","category":"page"},{"location":"#TransformerBlocks.Head","page":"TransformerBlocks.jl","title":"TransformerBlocks.Head","text":"Head(input_dim, head_size; dropout=0)\n\nInitializes an instance of the Head type, representing one head of self-attention.\n\nA Head instance accepts an input array x of dimensions (C, T, B) and outputs an array of dimensions (HS, T, B). \"C\" is the channel size (embedding dimension). \"T\" is the block size (number of input tokens). \"B\" is the batch size. \"HS\" is the head size.\n\nThe following keyword arguments are supported:\n\nmask (Defaults to nothing. Must be of dimensions (T, T).)\n\nExamples:\n\nC,T,B = 8,3,4\nHS = 10\nhead = Head(C,HS)\n@assert size(head(rand(Float32, C,T,B))) == (HS,T,B)\n\n\n\n\n\n","category":"type"},{"location":"#TransformerBlocks.MultiheadAttention","page":"TransformerBlocks.jl","title":"TransformerBlocks.MultiheadAttention","text":"MultiheadAttention(input_dim, num_heads; head_size=(input_dim ÷ num_heads), dropout=0)\n\nInitializes an instance of the MultiheadAttention type, representing multiple heads of parallel self-attention.\n\nThe following keyword arguments are supported:\n\nhead_size (Defaults to input_dim / num_heads)\ndropout (Defaults to 0)\n\nA MultiheadAttention instance accepts an input array x of dimensions (C, T, B) and outputs an array of dimensions (C, T, B). \"C\" is the channel size (embedding dimension). \"T\" is the block size (number of input tokens). \"B\" is the batch size.\n\nThe following keyword arguments are supported:\n\nmask (Defaults to nothing. Must be of dimensions (T, T).)\n\nExamples:\n\nC,T,B = 8,3,4\nNH = 4 # Num heads\nmultihead = MultiheadAttention(C,NH)\n@assert size(multihead(rand(Float32, C, T, B))) == (C, T, B)\n\n\n\n\n\n","category":"type"},{"location":"#TransformerBlocks.FeedForward","page":"TransformerBlocks.jl","title":"TransformerBlocks.FeedForward","text":"FeedForward(input_dim::Integer; dropout=0)\n\nInitializes an instance of the FeedForward type, representing a simple linear layer followed by a non-linearity.\n\nThe following keyword arguments are supported:\n\ndropout (Defaults to 0)\n\nA FeedForward instance accepts an input array x of dimensions (C, T, B) and outputs an array of dimensions (C, T, B). \"C\" is the channel size (embedding dimension). \"T\" is the block size (number of input tokens). \"B\" is the batch size.\n\nExamples:\n\nC,T,B = 8,3,4\nff = FeedForward(C)\n@assert size(ff(rand(Float32, C, T, B))) == (C, T, B)\n\n\n\n\n\n","category":"type"},{"location":"#TransformerBlocks.Block","page":"TransformerBlocks.jl","title":"TransformerBlocks.Block","text":"Block(input_dim; num_heads=1, head_size=(input_dim÷num_heads), dropout=0)\n\nInitializes an instance of the Block type, representing a transformer block.\n\nA Block instance accepts an input array x of dimensions (C, T, B) and outputs an array of dimensions (HS, T, B). \"C\" is the channel size (embedding dimension). \"T\" is the block size (number of input tokens). \"B\" is the batch size.\n\nThe following keyword arguments are supported:\n\nmask (Defaults to nothing. Must be of dimensions (T, T).)\n\nExamples:\n\nC,T,B = 8,3,4\nblock = Block(C)\n@assert size(block(rand(Float32, C,T,B))) == (C,T,B)\n\n\n\n\n\n","category":"type"},{"location":"#TransformerBlocks.BlockList","page":"TransformerBlocks.jl","title":"TransformerBlocks.BlockList","text":"BlockList(input_dim, num_heads; head_size=(input_dim ÷ num_heads), dropout=0)\n\nInitializes an instance of the BlockList type, representing a sequence of transformer blocks composed together.\n\nThe following keyword arguments are supported:\n\nhead_size (Defaults to input_dim / num_heads)\ndropout (Defaults to 0)\n\nA BlockList instance accepts an input array x of dimensions (C, T, B) and outputs an array of dimensions (HS, T, B). \"C\" is the channel size (embedding dimension). \"T\" is the block size (number of input tokens). \"B\" is the batch size.\n\nThe following keyword arguments are supported:\n\nmask (Defaults to nothing. Must be of dimensions (T, T).)\n\nExamples:\n\nC,T,B = 8,3,4\nblocklist = BlockList([Block(C), Block(C)])\n@assert size(blocklist(rand(Float32, C,T,B))) == (C,T,B)\n\n\n\n\n\n","category":"type"}]
}
