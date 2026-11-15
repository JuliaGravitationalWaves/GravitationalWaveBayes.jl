# Persistent on-disk cache of compiled CUDA kernels (see `GravitationalWaveBayes.KernelCache` for enabling/keying).
#
# Written against CUDA.jl 6.4.1 (CUDACore 6.4.1) and GPUCompiler 2.10.0. `cufunction` obtains the compiled
# artifacts of a job from `CUDACore.compile_or_lookup(job)`; on a miss that runs `CUDACore.compile(job)`, whose
# output `(image, entry, relocations)` is exactly what `CUDACore.link_kernel` turns into a `CuFunction`
# (exception buffers and `KernelState` are created per module after linking, the RNG-state initializer and
# libdevice/cudadevrt are already linked into the image). Host references are kept symbolic in
# `relocations` and patched in `link_kernel` (`relocation_lowering(job) == :patch`), so the artifacts are
# process-independent; CUDA.jl itself persists them in package images on that basis
# (`GPUCompiler.can_persist_results`). We add a more specific `compile_or_lookup` method that consults the
# disk cache before compiling. Any other CUDACore/GPUCompiler version leaves the method undefined and warns.
#
# Per-kernel dependencies (`KernelCache.dependencies`) need the code instances that went into the module, which
# GPUCompiler returns as `meta.compiled` of `compile(:asm, job)` (including Enzyme's deferred compilations) but
# `CUDACore.compile` drops. A more specific `GPUCompiler.compile_unhooked` method records them for the task.
# `CUDACore.compile` runs the compiler in the world captured at CUDACore's `__init__` (`invoke_frozen`), which
# predates this method, so a miss compiles with that world moved to the latest one (`_compile_latest`).

using GravitationalWaveBayes.KernelCache: KernelCache
import Enzyme

const CUDACore = CUDA.CUDACore
const GPUCompiler = CUDACore.GPUCompiler
const Libdl = Base.Libc.Libdl
const SUPPORTED = (CUDACore = v"6.4.1", GPUCompiler = v"2.10.0")
const COMPATIBLE = pkgversion(CUDACore) == SUPPORTED.CUDACore && pkgversion(GPUCompiler) == SUPPORTED.GPUCompiler

const _dir = Ref{Union{Nothing,String}}(nothing)
const _verify = Ref(false)
const _log = Ref(false)
const _stats = Dict{Symbol,Float64}(:hits => 0, :misses => 0, :stored => 0, :uncacheable => 0, :load_s => 0,
    :compile_s => 0)

function _init_kernel_cache()
    dir = KernelCache.cache_dir()
    dir === nothing && return nothing
    if !COMPATIBLE
        @warn "GravitationalWaveBayes kernel cache disabled: written for CUDACore $(SUPPORTED.CUDACore) / GPUCompiler " *
              "$(SUPPORTED.GPUCompiler), found $(pkgversion(CUDACore)) / $(pkgversion(GPUCompiler))"
        return nothing
    end
    mkpath(dir)
    _dir[] = dir
    _verify[] = KernelCache.verify_mode()
    _log[] = KernelCache.log_mode()
    @info "GravitationalWaveBayes kernel cache: $dir" * (_verify[] ? " (verify mode: every hit is recompiled and compared)" : "")
    return nothing
end

"""
    kernel_cache_stats() -> Dict

Counters of this process: `:hits`, `:misses` (compiled), `:stored`, `:uncacheable`, and seconds spent
loading and checking entries (`:load_s`), compiling misses (`:compile_s`) and collecting their dependencies
(`:deps_s`); `:deps_fallback` counts whole-package fallback dependencies.
"""
kernel_cache_stats() = copy(_stats)

## key

_try(f) = try
    string(f())
catch
    "?"
end

const _toolchain = Ref{String}("")
function _toolchain_key()
    isempty(_toolchain[]) || return _toolchain[]
    _toolchain[] = string("CUDACore=", pkgversion(CUDACore), " GPUCompiler=", pkgversion(GPUCompiler),
        " LLVM.jl=", pkgversion(CUDACore.LLVM), " Enzyme=", pkgversion(Enzyme),
        " nvptx_llvm=", _try(() -> CUDACore.nvptx_llvm_version), " ptxas=", _try(CUDACore.compiler_version),
        " driver=", _try(CUDACore.driver_version), " runtime=", _try(CUDACore.runtime_version),
        " ptxas_path=", _try(() -> CUDACore.CUDA_Compiler.ptxas().exec[1]), "\n")
    return _toolchain[]
end

# Enzyme's global codegen switches (LLVM cl::opts in libEnzyme and Julia-side Refs) are process state that
# scripts may change (e.g. `Enzyme.API.strictAliasing!(false)`, `maxtypeoffset!`).
const _ENZYME_BOOLS = (:EnzymeStrictAliasing, :looseTypeAnalysis, :EnzymeFastMath, :EnzymeInline, :EnzymeZeroCache,
    :EnzymeNonPower2Cache, :EnzymeJuliaAddrLoad)
const _ENZYME_INTS = (:MaxTypeOffset, :EnzymeMaxTypeDepth)
const _ENZYME_REFS = (:SYMBOLIC_PRIMAL, :CHECK_RULE_MEMO, :CheckNan, :RunAttributor)

function _enzyme_key()
    h = Libdl.dlopen(Enzyme.API.libEnzyme)
    getb = Libdl.dlsym(h, :EnzymeGetCLBool)
    geti = Libdl.dlsym(h, :EnzymeGetCLInteger)
    io = IOBuffer()
    print(io, "enzyme")
    for s in _ENZYME_BOOLS
        p = Libdl.dlsym(h, s; throw_error = false)
        print(io, ' ', s, '=', p === nothing ? "?" : ccall(getb, UInt8, (Ptr{Cvoid},), p))
    end
    for s in _ENZYME_INTS
        p = Libdl.dlsym(h, s; throw_error = false)
        print(io, ' ', s, '=', p === nothing ? "?" : ccall(geti, Int64, (Ptr{Cvoid},), p))
    end
    for s in _ENZYME_REFS
        print(io, ' ', s, '=', isdefined(Enzyme.Compiler, s) ? getfield(Enzyme.Compiler, s)[] : "?")
    end
    print(io, '\n')
    return String(take!(io))
end

function _cacheable(job)
    m = job.source.def
    return m isa Method && GPUCompiler.can_persist_results(job)
end

# the parts of a key that are fixed for a (method instance, config, device) are printed once per session
const _static_keys = IdDict{Any,Dict{Tuple{Any,Any},String}}()
const _static_lock = ReentrantLock()

function _static_key(job, dev)
    Base.@lock _static_lock begin
        d = get!(Dict{Tuple{Any,Any},String}, _static_keys, job.source)
        get!(d, (job.config, dev)) do
            m = job.source.def::Method
            string(KernelCache.process_key(), _toolchain_key(),
                "device=", CUDACore.name(dev), " capability=", CUDACore.capability(dev), '\n',
                "method=", KernelCache.qualified(m.module), '.', m.name, '\n',
                "spec=", KernelCache.qualified(job.source.specTypes), '\n',
                "config=", KernelCache.describe(job.config), '\n')
        end
    end
end

function _key_text(job)
    return string("GravitationalWaveBayes CUDA kernel cache format ", KernelCache.FORMAT, '\n', _enzyme_key(),
        _fast_key(),
        "env_state=", KernelCache.env_state(), '\n', _static_key(job, CUDACore.device()))
end

## compilation with dependency capture

const _COMPILED_KEY = :gwbayes_kernel_cache_compiled

@static if COMPATIBLE
    function GPUCompiler.compile_unhooked(output::Symbol,
            job::GPUCompiler.CompilerJob{GPUCompiler.PTXCompilerTarget,CUDACore.CUDACompilerParams}; kwargs...)
        res = invoke(GPUCompiler.compile_unhooked, Tuple{Symbol,GPUCompiler.CompilerJob}, output, job; kwargs...)
        if output === :asm && job.config.toplevel
            sink = get(task_local_storage(), _COMPILED_KEY, nothing)
            sink === nothing || push!(sink, res[2].compiled)
        end
        return res
    end
end

# `CUDACore.compile(job)` in the latest world; returns (artifacts, compiled code instances)
function _compile_latest(job)
    sink = Any[]
    w = CUDACore._initialization_world[]
    CUDACore._initialization_world[] = typemax(UInt)
    try
        compiled = task_local_storage(() -> CUDACore.compile(job), _COMPILED_KEY, sink)
        return compiled, sink
    finally
        CUDACore._initialization_world[] = w
    end
end

# dependency text of a compiled kernel, `nothing` if it could not be determined
function _dependencies(job, sink)
    length(sink) == 1 || return nothing
    cis = Core.CodeInstance[]
    for v in values(only(sink))
        ci = v.ci
        ci isa Core.CodeInstance && push!(cis, ci)
    end
    isempty(cis) && return nothing
    owners = unique!(Any[ci.owner for ci in cis])
    t0 = time()
    deps, fallback = KernelCache.dependencies(cis, owners)
    _stats[:deps_s] = get(_stats, :deps_s, 0.0) + time() - t0
    if !isempty(fallback)
        _stats[:deps_fallback] = get(_stats, :deps_fallback, 0.0) + length(fallback)
        _log[] && @info "GravitationalWaveBayes kernel cache: $(_kernel_name(job)) depends on whole packages through method instances without code instances" fallback
    end
    return deps
end

## artifacts

_portable_value(@nospecialize(v)) = v isa Union{Type,Module,Symbol,String,Core.TypeName} || isbits(v) ||
    (!ismutable(v) && all(i -> isdefined(v, i) && _portable_value(getfield(v, i)), 1:nfields(v)))
_portable(t::GPUCompiler.JuliaValueRef) = _portable_value(t.value)
_portable(::GPUCompiler.CGlobalRef) = true
_portable(r::GPUCompiler.Relocations) = all(rec -> _portable(rec.target), r.records) && all(_portable, r.table)

_same_target(a::GPUCompiler.JuliaValueRef, b::GPUCompiler.JuliaValueRef) = isequal(a.value, b.value)
_same_target(a::GPUCompiler.CGlobalRef, b::GPUCompiler.CGlobalRef) =
    a.symbol === b.symbol && a.library == b.library && a.offset == b.offset
_same_target(a, b) = false
function _same_relocations(a::GPUCompiler.Relocations, b::GPUCompiler.Relocations)
    length(a.records) == length(b.records) && length(a.table) == length(b.table) || return false
    for (x, y) in zip(a.records, b.records)
        (x.kind == y.kind && x.name == y.name && x.offset == y.offset && _same_target(x.target, y.target)) ||
            return false
    end
    return all(_same_target(x, y) for (x, y) in zip(a.table, b.table))
end
_same_artifacts(a, b) = a.image == b.image && a.entry == b.entry && _same_relocations(a.relocations, b.relocations)

# Julia appends a per-session counter to device function names (`julia_fldmod1_31967`), which ends up in the
# cubin's symbol and debug string tables. True when the images have equal length and every differing byte
# is a digit of a `_<digits>` run in both.
_isdigit(b::UInt8) = UInt8('0') <= b <= UInt8('9')
function _counter_digit(img, i)
    _isdigit(img[i]) || return false
    while i > 1 && _isdigit(img[i])
        i -= 1
    end
    return img[i] == UInt8('_')
end
_same_modulo_counters(a::Vector{UInt8}, b::Vector{UInt8}) = length(a) == length(b) &&
    all(i -> a[i] == b[i] || (_counter_digit(a, i) && _counter_digit(b, i)), eachindex(a))

## results for disk hits that have no CodeInstance in this session (`cached_results` returns `nothing`)
# Entries are found by world age on the launch path; after a world-age change the entry is reused (keeping its
# linked `CuFunction`) only if the base key, which is recomputed then, and the dependencies are unchanged.

mutable struct SessionEntry
    world::UInt
    version::Int                 # KernelCache.index_version() at the last full dependency check
    const config::GPUCompiler.CompilerConfig
    const key::String
    const deps::String
    const res::CUDACore.CUDACompilerResults
end

const _session_lock = ReentrantLock()
const _session = IdDict{Core.MethodInstance,Vector{SessionEntry}}()

function _session_get(job)
    Base.@lock _session_lock begin
        for e in get(_session, job.source, ())
            e.world == job.world && e.config === job.config && return e.res
        end
    end
    return nothing
end

function _session_revalidate!(job, key)
    Base.@lock _session_lock begin
        for e in get(_session, job.source, ())
            (e.config === job.config && e.key == key) || continue
            v = KernelCache.index_version()
            # same mutable-code methods as at the last check: only constants, types and fallback packages can differ
            ok = KernelCache.check_dependencies(e.deps; kinds = v == e.version ? "GTP" : "MFGTP") === nothing
            if ok
                e.world = job.world
                e.version = v
                return e.res
            end
        end
    end
    return nothing
end

function _session_new!(job, key, deps)
    r = CUDACore.CUDACompilerResults()
    Base.@lock _session_lock begin
        push!(get!(Vector{SessionEntry}, _session, job.source), SessionEntry(job.world, KernelCache.index_version(), job.config, key, deps, r))
    end
    return r
end

_kernel_name(job) = string(job.source.def.name)

function _verify_hit(job, art, dir, key)
    fresh = CUDACore.compile(job)
    if _same_artifacts(art, fresh)
        _stats[:verify_identical] = get(_stats, :verify_identical, 0.0) + 1
    elseif _same_modulo_counters(art.image, fresh.image) && art.entry == fresh.entry &&
           _same_relocations(art.relocations, fresh.relocations)
        _stats[:verify_names_only] = get(_stats, :verify_names_only, 0.0) + 1
    else
        _stats[:verify_mismatch] = get(_stats, :verify_mismatch, 0.0) + 1
        @error "GravitationalWaveBayes kernel cache: entry differs from a fresh compile; using the fresh artifacts" job.source base = KernelCache._base_dir(dir, key) image_same_modulo_name_counters = _same_modulo_counters(art.image, fresh.image) entry_same = art.entry == fresh.entry relocations_same = _same_relocations(art.relocations, fresh.relocations)
        return fresh
    end
    return art
end

@noinline function _compile_or_lookup(job, dir::String)
    R = CUDACore.CUDACompilerResults
    GPUCompiler.compile_hook[] === nothing ||
        return invoke(CUDACore.compile_or_lookup, Tuple{GPUCompiler.CompilerJob}, job)
    res = GPUCompiler.cached_results(R, job)
    res !== nothing && res.image !== nothing && return res
    if res === nothing
        s = _session_get(job)
        s === nothing || return s
    end
    t0 = time()
    key = _cacheable(job) ? _key_text(job) : nothing
    if key !== nothing && res === nothing
        s = _session_revalidate!(job, key)
        _stats[:revalidate_s] = get(_stats, :revalidate_s, 0.0) + time() - t0
        _stats[:revalidations] = get(_stats, :revalidations, 0.0) + 1
        s === nothing || return s
    end
    changed = String[]
    if key !== nothing
        t0 = time()
        art, deps = KernelCache.load_entry(dir, key)
        _stats[:load_s] += time() - t0
        if art !== nothing
            _verify[] && (art = _verify_hit(job, art, dir, key))
            target = res === nothing ? _session_new!(job, key, deps) : res
            target.image = art.image
            target.entry = art.entry
            target.relocations = art.relocations
            _stats[:hits] += 1
            _log[] && @info "GravitationalWaveBayes kernel cache hit: $(_kernel_name(job))"
            return target
        end
        changed = deps
    end
    t0 = time()
    compiled, sink = key === nothing ? (CUDACore.compile(job), Any[]) : _compile_latest(job)
    dt = time() - t0
    _stats[:compile_s] += dt
    _stats[:misses] += 1
    out = something(res, GPUCompiler.cached_results(R, job))
    out.image = compiled.image
    out.entry = compiled.entry
    out.relocations = compiled.relocations
    deps = key === nothing ? nothing : _dependencies(job, sink)
    if deps !== nothing && _portable(compiled.relocations)
        KernelCache.store_entry!(dir, key, deps, (; image = compiled.image, entry = compiled.entry,
            relocations = compiled.relocations))
        _stats[:stored] += 1
    else
        _stats[:uncacheable] += 1
    end
    if _log[]
        why = isempty(changed) ? "" : string("; changed: ", join(changed, ", "))
        @info string("GravitationalWaveBayes kernel cache miss: ", _kernel_name(job), " compiled in ", round(dt; digits = 1), " s",
            deps === nothing ? " (not cacheable)" : "", why)
    end
    return out
end

@static if COMPATIBLE
    function CUDACore.compile_or_lookup(job::GPUCompiler.CompilerJob{GPUCompiler.PTXCompilerTarget,CUDACore.CUDACompilerParams})
        dir = _dir[]
        dir === nothing && return invoke(CUDACore.compile_or_lookup, Tuple{GPUCompiler.CompilerJob}, job)
        return _compile_or_lookup(job, dir)::CUDACore.CUDACompilerResults
    end
end
