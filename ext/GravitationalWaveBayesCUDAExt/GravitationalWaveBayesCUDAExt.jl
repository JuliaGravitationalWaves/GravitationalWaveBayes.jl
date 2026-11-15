module GravitationalWaveBayesCUDAExt

using GravitationalWaveBayes
using CUDA

include("kernel_cache.jl")
include("fast_compile.jl")

function __init__()
    _init_kernel_cache()
    _init_fast_compile()
    GravitationalWaveBayes.GPU._to_device[] = x -> CUDA.functional() ? CuArray(x) : Array(x)
    GravitationalWaveBayes.GPU._array_ctor[] = (T, dims) -> CUDA.functional() ? CuArray{T}(undef, dims) : Array{T}(undef, dims)
    GravitationalWaveBayes.GPU._ensure_device_heap[] = _ensure_heap
    GravitationalWaveBayes.GPU._reclaim[] = () -> (CUDA.functional() && CUDA.reclaim(); nothing)
end

function GravitationalWaveBayes.GPU.batched_adj_mul_mid!(C::CuArray{ComplexF64,3}, A::CuArray{ComplexF64,3}, X::CuArray{ComplexF64,3})
    nb, m, nbat = size(A); n = size(X, 3)
    (size(X, 1) == nb && size(X, 2) == nbat && size(C) == (m, n, nbat)) || throw(DimensionMismatch("batched_adj_mul_mid!"))
    CUDA.CUBLAS.cublasZgemmStridedBatched(CUDA.CUBLAS.handle(), CUDA.CUBLAS.CUBLAS_OP_C, CUDA.CUBLAS.CUBLAS_OP_N, m, n, nb,
        one(ComplexF64), A, nb, nb * m, X, nb * nbat, nb, zero(ComplexF64), C, m, m * n, nbat)
    return C
end
GravitationalWaveBayes.GPU.block_alias(X::CuArray{T,N,M}, offset::Integer, dims::Dims{K}) where {T,N,M,K} =
    unsafe_wrap(CuArray{T,K,M}, pointer(X, offset + 1), dims)

# the smallest size a raise failed for: a later request of that size or more neither retries nor warns again (a
# failing `cuCtxSetLimit` plus the warning cost ~1 s per gradient call)
const _HEAP_FAILED = Ref(typemax(Int))
function _ensure_heap(bytes::Int)
    CUDA.functional() || return nothing
    cur = CUDA.limit(CUDA.LIMIT_MALLOC_HEAP_SIZE)
    cur >= bytes && return nothing
    bytes >= _HEAP_FAILED[] && return nothing
    try
        CUDA.limit!(CUDA.LIMIT_MALLOC_HEAP_SIZE, bytes)
    catch err
        _HEAP_FAILED[] = bytes
        @warn "Could not raise the CUDA device malloc heap to $(bytes >> 20) MB (currently $(cur >> 20) MB): it can only " *
              "be changed before the first kernel using device malloc has run. Set it at the start of the session: " *
              "CUDA.limit!(CUDA.LIMIT_MALLOC_HEAP_SIZE, $(bytes))" exception = err
    end
    return nothing
end

end
