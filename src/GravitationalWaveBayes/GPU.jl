module GPU

import LinearAlgebra

export to_device, compatible_array

# Device hooks. Both live behind mutable Refs that the GPU extension's `__init__` reassigns at load time
# (ext/GravitationalWaveBayesCUDAExt.jl).
const _to_device = Ref{Any}(x -> Array(x))

"""
    to_device(x::AbstractArray) -> AbstractArray

Move `x` onto whichever device this session is configured for. The base (CPU-only) behavior returns a
plain `Array` copy; loading `CUDA.jl` (and a functional GPU) makes it a `CuArray` (see the extension) -- callers never need
to branch on which backend is active.
"""
@inline to_device(x::AbstractArray) = _to_device[](x)

# The allocator behind `compatible_array` (same Ref mechanism as `to_device`); CPU default.
const _array_ctor = Ref{Any}((T, dims) -> Array{T}(undef, dims))

"""
    compatible_array(T::Type, dims::NTuple{N,Int}) -> AbstractArray{T,N}

Allocate an uninitialized `T`-element array of shape `dims` on whichever device this session is
configured for -- the allocating analogue of [`to_device`](@ref) for code that needs a fresh
array rather than a device copy of an existing one (e.g. a kernel's output buffer). The base
(CPU-only) behavior is `Array{T}(undef, dims)`; loading `CUDA.jl` reassigns the allocator this
function calls to `CUDA.functional() ? CuArray{T}(undef, dims) : Array{T}(undef, dims)` (see
`GravitationalWaveBayesCUDAExt`'s `__init__`) -- callers never need to branch on which backend is active.
"""
@inline function compatible_array(T::Type, dims::NTuple{N,Int}) where {N}
    return _array_ctor[](T, dims)
end

const _ensure_device_heap = Ref{Any}(bytes -> nothing)
"""
    ensure_device_heap(bytes::Integer)

Make sure the GPU's device-side `malloc` heap is at least `bytes` (no-op on the CPU). Kernels differentiated by Enzyme
inside the kernel launch allocate their reverse-pass caches there; CUDA's default heap is 8 MB, which the SEOBNRv5
PA-F2 multi-detector gradient with likelihood options overflows already at 16 walkers (CUDA error 700).
CUDA only allows raising the limit before the first kernel that uses `malloc` has run, so call this when a GPU
closure is built, before its first use (`gw_logp_and_grad_gpu_staged` does). Set by `GravitationalWaveBayesCUDAExt`.
"""
ensure_device_heap(bytes::Integer) = _ensure_device_heap[](Int(bytes))

const _reclaim = Ref{Any}(() -> nothing)
"""
    reclaim()

Return the device memory pool's cached blocks to the driver (CUDA: `CUDA.reclaim()`; no-op elsewhere). Before
reallocating large work buffers of a different size: the pool cannot reuse blocks of another size, and on a 16 GB
V100 the PHM reverse kernels' stack reservation (~58 KB x 163840 threads = 9.5 GB) leaves no room for both.
Set by `GravitationalWaveBayesCUDAExt`.
"""
reclaim() = _reclaim[]()

"""
    tuning(name::Symbol, default)

Hardware tuning value `name`: the environment variable `GRAVITATIONALWAVEBAYES_<name>` parsed as `typeof(default)` (a `Symbol` default
takes the variable's text), else `default`. For the launch shapes and crossover walker counts measured on one GPU
(V100) that may differ on another: override them without editing code, e.g. `GRAVITATIONALWAVEBAYES_PHM_CHAIN_PAR_PLUNGE=4096`.
Read on the host at each call (no caching), so a value must not change during a run.
Knobs: `PHM_WG`, `PHM_CHAIN_PAR_{PLUNGE,PN,STS}`, `SDT_WG`, `SDT_PLUNGE_DIRPAR_MAX`, `SDH_ATTACH_DIRPAR_MAX`,
`MBP_BIN_SUMS` (`kernel` | `gemm`).
"""
function tuning(name::Symbol, default::T) where {T}
    v = get(ENV, "GRAVITATIONALWAVEBAYES_" * String(name), "")
    isempty(v) && return default
    return T === Symbol ? Symbol(v) : parse(T, v)
end

"""
    batched_adj_mul_mid!(C, A, X)

`C[:, :, b] = A[:, :, b]' * X[:, b, :]` for every batch index `b` (the batch is X's MIDDLE dimension: column stride
size(X, 1) * size(X, 2), batch stride size(X, 1)). Generic loop; `GravitationalWaveBayesCUDAExt` adds one strided-batched cuBLAS call.
"""
function batched_adj_mul_mid!(C::AbstractArray{T,3}, A::AbstractArray{T,3}, X::AbstractArray{T,3}) where {T}
    for b in axes(C, 3)
        LinearAlgebra.mul!(view(C, :, :, b), adjoint(view(A, :, :, b)), view(X, :, b, :))
    end
    return C
end

"""
    block_alias(X, offset, dims)

A dense array of `X`'s array type with size `dims` sharing `X`'s memory from linear element `offset + 1` on (not a
view: e.g. separate same-typed Enzyme shadows inside one buffer). `X` must outlive it. CUDA method in `GravitationalWaveBayesCUDAExt`.
"""
block_alias(X::Array{T}, offset::Integer, dims::Dims) where {T} = unsafe_wrap(Array, pointer(X, offset + 1), dims)

end
