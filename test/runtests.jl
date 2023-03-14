using TransformerBlocks
using Test

@testset "Test Head" begin
    C,T,B = 8,3,4
    HS = 10
    head = Head(C,HS)
    @test size(head(rand(Float32, C,T,B))) == (HS,T,B)
end

@testset "Test Block" begin
    C,T,B = 8,3,4
    block = Block(C)
    @test size(block(rand(Float32, C,T,B))) == (C,T,B)
end

@testset "Test BlockList" begin
    C,T,B = 8,3,4
    blocklist = BlockList([Block(C), Block(C)])
    @test size(blocklist(rand(Float32, C,T,B))) == (C,T,B)
end

@testset "Test FeedForward" begin
    C,T,B = 8,3,4
    ff = FeedForward(C)
    @test size(ff(rand(Float32, C, T, B))) == (C, T, B)
end

@testset "Test MultiheadAttention" begin
    C,T,B = 8,3,4
    NH = 4
    multihead = MultiheadAttention(C,NH)
    @test size(multihead(rand(Float32, C, T, B))) == (C, T, B)
end