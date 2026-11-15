# Field update / take / drop on NamedTuples by generated constructors (Base `merge` / `structdiff` fall back to dynamic
# code for large records, which does not compile for GPUs). In a file of their own: the kernel cache keys a generated
# function by its whole defining file, so keeping them apart from frequently edited code avoids needless recompiles.
@generated function _nt_merge(nt::NamedTuple{K}, upd::NamedTuple{U}) where {K,U}
    args = [k in U ? :(getfield(upd, $(QuoteNode(k)))) : :(getfield(nt, $(QuoteNode(k)))) for k in K]
    return :(NamedTuple{$K}(($(args...),)))
end
@generated function _nt_drop(nt::NamedTuple{K}, ::Val{D}) where {K,D}
    keep = Tuple(k for k in K if !(k in D))
    args = [:(getfield(nt, $(QuoteNode(k)))) for k in keep]
    return :(NamedTuple{$keep}(($(args...),)))
end
@generated function _nt_take(nt::NamedTuple, ::Val{K}) where {K}
    args = [:(getfield(nt, $(QuoteNode(k)))) for k in K]
    return :(NamedTuple{$K}(($(args...),)))
end
