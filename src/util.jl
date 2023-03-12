function transposebatchmul(a, b)
    batched_mul(batched_transpose(a),b)
end

function transposebatchmul(a::CuArray, b::CuArray)
    CUBLAS.gemm_strided_batched('T', 'N', a, b)
end

@adjoint transposebatchmul(a::CuArray, b::CuArray) = begin
    function rev(c̄)
        ā = CUBLAS.gemm_strided_batched('N', 'T', b, c̄)
        b̄ = CUBLAS.gemm_strided_batched('N', 'N', a, c̄)
        (ā, b̄)
    end
    transposebatchmul(a, b), rev
end