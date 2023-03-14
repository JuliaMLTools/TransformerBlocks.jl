import SnoopPrecompile
using LinearAlgebra

SnoopPrecompile.@precompile_all_calls begin
    C, T, B = 2, 2, 2
    block = Block(C)
    block(rand(Float32, C, T, B))
    mask = Float32.(tril(fill(-Inf, T, T), -1))
    block(rand(Float32, C, T, B); mask=mask)
    num_layers = 2
    blocks = BlockList([Block(C) for _ in 1:num_layers])
    blocks(rand(Float32, C, T, B))
end
