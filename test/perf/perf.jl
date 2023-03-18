cd(@__DIR__)
using TransformerBlocks, CUDA, Flux, BenchmarkTools

b = gpu(Block(512; num_heads = 4));
x = CUDA.randn(512, 128, 16);

@btime CUDA.@sync $b($x);
# 1.849 ms (2572 allocations: 136.44 KiB)

@btime CUDA.@sync gradient((m, inp)->sum(sin.(m(inp))), $b, $x);
# 6.449 ms (8929 allocations: 566.56 KiB)