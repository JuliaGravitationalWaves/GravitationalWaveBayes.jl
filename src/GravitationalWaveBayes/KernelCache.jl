"""
    KernelCache

On-disk cache of compiled GPU kernels, shared by the GPU-backend extensions (the CUDA one
intercepts `CUDACore.compile_or_lookup`, see `ext/GravitationalWaveBayesCUDAExt/kernel_cache.jl`).

Opt-in through the environment variable `GRAVITATIONALWAVEBAYES_KERNEL_CACHE`:
  unset, empty or `0` -> off;  `1` -> `\$XDG_CACHE_HOME/gravitationalwavebayes/kernels` (default `~/.cache/gravitationalwavebayes/kernels`);
  any other value -> that directory.
`GRAVITATIONALWAVEBAYES_KERNEL_CACHE_VERIFY=1` additionally recompiles every hit and reports (`@error`) cached artifacts that differ
from the fresh ones beyond Julia's per-session function-name counters, then uses the fresh ones (determinism
check; costs the full compile time). `GRAVITATIONALWAVEBAYES_KERNEL_CACHE_LOG=1` logs every hit and miss, with the first changed
dependencies of a miss that has older entries.

Keying is per kernel, in two parts:

1. A *base key* text: the kernel signature, the full compiler configuration, the GPU, the toolchain versions, the
   Julia build and command-line codegen options, codegen-affecting Enzyme settings, and [`env_state`](@ref) (the
   active environments' Project/Manifest/LocalPreferences files and the Project/LocalPreferences files of every
   `dev`'d package). Code of registry/git installs is fixed by the Manifest.
2. The kernel's *dependencies* on mutable code (`dev`'d / path packages and `Main`), recorded when it is compiled
   ([`dependencies`](@ref)) from the code instances that went into the module (including Enzyme's deferred
   compilations) and, transitively, their inference edges (which include every inlined callee):
   - `M` every reached method of mutable code, by its lowered code (line numbers excluded; generated functions by
     the generator's lowered code plus the contents of the file defining it);
   - `F` for every function reached (in any module), the signatures of all its mutable-code methods, so that a
     new or deleted method that could change dispatch is seen;
   - `G` constant globals of mutable modules referenced by those methods (values are folded into the code);
   - `T` the definitions (field names and types, supertype) of mutable-code types in reached specializations;
   - `P` a whole package's sources, as a conservative fallback for a reached method instance whose code
     instance cannot be found;
   - every method of the `EnzymeCore.EnzymeRules` functions in mutable code (custom rules are looked up by
     Enzyme, not through inference edges).
   An entry is used only when every recorded dependency is unchanged ([`check_dependencies`](@ref)), so editing
   one function recompiles only the kernels that reach it.

Known blind spots: functions called by a `@generated` function's generator from another file; code evaluated
into immutable packages' modules (`@eval Base ...`) from mutable code. In-session `eval`/Revise edits are seen
(lowered code is read from the method table, not from disk).

Layout: `<dir>/<sha(base key)>/<sha(dependencies)>.{jls,deps}` plus `<dir>/<sha(base key)>.txt`; at most
`MAX_VARIANTS` dependency variants are kept per base key (oldest removed). Clear the cache by deleting the
directory.
"""
module KernelCache

using SHA: sha256
using Serialization: serialize, deserialize

const FORMAT = 3
const MAX_VARIANTS = 6

"""
    cache_dir() -> Union{Nothing,String}

The kernel-cache directory selected by `GRAVITATIONALWAVEBAYES_KERNEL_CACHE`, or `nothing` when the cache is off.
"""
function cache_dir()
    v = strip(get(ENV, "GRAVITATIONALWAVEBAYES_KERNEL_CACHE", ""))
    (isempty(v) || v == "0") && return nothing
    v == "1" || return abspath(expanduser(v))
    base = get(ENV, "XDG_CACHE_HOME", "")
    isempty(base) && (base = joinpath(homedir(), ".cache"))
    return joinpath(base, "gravitationalwavebayes", "kernels")
end

verify_mode() = get(ENV, "GRAVITATIONALWAVEBAYES_KERNEL_CACHE_VERIFY", "0") == "1"
log_mode() = get(ENV, "GRAVITATIONALWAVEBAYES_KERNEL_CACHE_LOG", "0") == "1"

_hex(x) = bytes2hex(sha256(x))

## mutable code

# packages under `<depot>/packages` are content-addressed by their Manifest tree hash
function _immutable_install(path::AbstractString)
    any(d -> startswith(path, joinpath(d, "packages") * Base.Filesystem.path_separator), DEPOT_PATH) ||
        startswith(path, Sys.STDLIB)
end

# directory holding the Project.toml of the package (or extension's parent package) that `file` belongs to
function _package_root(file::AbstractString)
    d = dirname(realpath(file))
    while true
        isfile(joinpath(d, "Project.toml")) && return d
        p = dirname(d)
        p == d && return nothing
        d = p
    end
end

const _mutable_lock = ReentrantLock()
const _mutable_n = Ref(-1)
const _mutable_roots = Ref(Base.IdSet{Module}())
const _mutable_dirs = Ref(Dict{Module,String}())

# root modules of loaded mutable (non registry/git) packages and extensions -> package directory; plus `Main`
function _mutable_modules()
    Base.@lock _mutable_lock begin
        n = length(Base.loaded_modules)
        if n != _mutable_n[]
            roots = Base.IdSet{Module}()
            dirs = Dict{Module,String}()
            push!(roots, Main)
            for mod in collect(values(Base.loaded_modules))
                f = pathof(mod)
                (f === nothing || _immutable_install(realpath(f))) && continue
                push!(roots, mod)
                root = _package_root(f)
                root === nothing || (dirs[mod] = root)
            end
            _mutable_roots[] = roots
            _mutable_dirs[] = dirs
            _mutable_n[] = n
        end
        return _mutable_roots[], _mutable_dirs[]
    end
end

_is_mutable(mod::Module) = Base.moduleroot(mod) in _mutable_modules()[1]
_is_mutable(m::Method) = _is_mutable(m.module)

## base key: environments

function _env_files()
    files = String[]
    for env in Base.load_path()
        isfile(env) || continue
        push!(files, env)
        man = Base.project_file_manifest_path(env)
        man === nothing || push!(files, man)
        for p in ("LocalPreferences.toml", "JuliaLocalPreferences.toml")
            f = joinpath(dirname(env), p)
            isfile(f) && push!(files, f)
        end
    end
    for root in sort!(unique(collect(values(_mutable_modules()[2]))))
        for p in ("Project.toml", "LocalPreferences.toml")
            f = joinpath(root, p)
            isfile(f) && push!(files, f)
        end
    end
    return files
end

function _hash_files(files)
    io = IOBuffer()
    for f in files
        print(io, f, '\n', _hex(read(f)), '\n')
    end
    return _hex(take!(io))
end

"""
    env_state() -> String

Hex SHA-256 over the paths and contents of the environment files listed in the module docstring (re-hashed only
when a file's size or modification time changes).
"""
const _env_lock = ReentrantLock()
const _env_sig = Ref("")
const _env_hash = Ref("")

function env_state()
    files = _env_files()
    sig = join((string(f, '\t', filesize(f), '\t', mtime(f)) for f in files), '\n')
    Base.@lock _env_lock begin
        sig == _env_sig[] && return _env_hash[]
        _env_hash[] = _hash_files(files)
        _env_sig[] = sig
        return _env_hash[]
    end
end

"""
    package_state(root) -> String

Hex SHA-256 over every `.jl` file under `root/src` and `root/ext` (fallback dependency `P`).
"""
function package_state(root::AbstractString)
    files = String[]
    for sub in ("src", "ext")
        d = joinpath(root, sub)
        isdir(d) || continue
        for (dir, _, fs) in walkdir(d), f in fs
            endswith(f, ".jl") && push!(files, joinpath(dir, f))
        end
    end
    return _hash_files(sort!(files))
end

## key text helpers

"""
    qualified(T) -> String

`T` printed with every name module-qualified (types are shown as seen from `Core`).
"""
qualified(@nospecialize(T)) = sprint(show, T; context = :module => Core)

"""
    describe(x) -> String

Structural dump of `x` (recursively every field of every struct; types module-qualified), for key texts.
"""
describe(@nospecialize(x)) = sprint(_describe, x, 0)

function _describe(io::IO, @nospecialize(x), depth::Int)
    depth > 12 && error("KernelCache.describe: nesting too deep")
    if x isa Type || x isa Module || (x isa Function && nfields(x) == 0)
        print(io, qualified(x))
    elseif x isa Union{Number,Symbol,AbstractString,Nothing,Char,VersionNumber}
        show(io, x)
    elseif x isa Tuple || x isa NamedTuple || x isa AbstractArray
        print(io, x isa NamedTuple ? "(;" : x isa AbstractArray ? string(qualified(typeof(x)), size(x), "(") : "(")
        for (k, v) in pairs(x)
            x isa NamedTuple && print(io, k, '=')
            _describe(io, v, depth + 1)
            print(io, ',')
        end
        print(io, ')')
    elseif isstructtype(typeof(x))
        print(io, qualified(typeof(x)), '(')
        for i in 1:nfields(x)
            print(io, fieldname(typeof(x), i), '=')
            isdefined(x, i) ? _describe(io, getfield(x, i), depth + 1) : print(io, "#undef")
            print(io, ',')
        end
        print(io, ')')
    else
        show(io, x)
    end
end

"""
    process_key() -> String

Key lines for the Julia build and the command-line options that change generated code.
"""
function process_key()
    o = Base.JLOptions()
    io = IOBuffer()
    print(io, "julia=", VERSION, " commit=", Base.GIT_VERSION_INFO.commit, " llvm=", Base.libllvm_version, "\njloptions")
    for f in (:check_bounds, :debug_level, :opt_level, :can_inline, :fast_math, :code_coverage, :malloc_log)
        hasfield(typeof(o), f) && print(io, ' ', f, '=', getfield(o, f))
    end
    print(io, '\n')
    return String(take!(io))
end

## names <-> objects

_modpath(mod::Module) = join(fullname(mod), '.')

function _resolve_module(path::AbstractString)
    parts = split(path, '.')
    first = Symbol(parts[1])
    mod = first === :Main ? Main : first === :Base ? Base : first === :Core ? Core : nothing
    if mod === nothing
        for m in values(Base.loaded_modules)
            nameof(m) === first && (mod = m; break)
        end
    end
    mod === nothing && return nothing
    for p in parts[2:end]
        s = Symbol(p)
        isdefined(mod, s) || return nothing
        mod = getglobal(mod, s)
        mod isa Module || return nothing
    end
    return mod
end

# "<module path>:<name>" -> (module, name) or nothing
function _resolve_global(key::AbstractString)
    i = findlast(':', key)
    i === nothing && return nothing
    mod = _resolve_module(key[1:i-1])
    mod === nothing && return nothing
    return mod, Symbol(key[i+1:end])
end

# function key of a method / call signature's function type: one key per generic function (type name for
# callable types, `Type{<wrapper>}` for constructors)
function _function_key(@nospecialize(ft))
    ft = Base.unwrap_unionall(ft)
    ft isa DataType || return "?"
    if ft.name === Type.body.name
        t = Base.unwrap_unionall(ft.parameters[1])
        return t isa DataType ? string("Type{", qualified(t.name.wrapper), "}") : "Type{?}"
    end
    return qualified(ft.name.wrapper)
end

function _sig_function_key(@nospecialize(sig))
    s = Base.unwrap_unionall(sig)
    (s isa DataType && s <: Tuple && !isempty(s.parameters)) || return nothing
    p = s.parameters
    # keyword methods `f(x; kw...)` are methods of `Core.kwcall` on `(kwargs, typeof(f), x...)`: key them on `f`
    # (otherwise any new keyword method anywhere in mutable code changes every kernel's `kwcall` method list)
    p[1] === typeof(Core.kwcall) && length(p) >= 3 && return string("kwcall:", _function_key(p[3]))
    return _function_key(p[1])
end

## method index of mutable code

# Rebuilt only when the set of live mutable-code methods changes; a world-age change without one (any method
# defined anywhere, e.g. a closure literal at a script's top level) costs one cheap scan of the method table and
# keeps the index, its `version` and the fingerprint memo.
struct MethodIndex
    world::UInt
    version::Int                             # bumped when the live mutable-code methods change
    nlive::Int
    sigs::Dict{String,Method}
    functions::Dict{String,Vector{String}}   # function key -> sorted method signatures
    fps::Dict{String,String}                 # dependency line -> fingerprint (M / F / T lines; G and P not memoized)
end

# replaced and deleted methods stay in the method table; live ones are `which(m.sig)` in the latest world
_live(m::Method) = !hasfield(Method, :dispatch_status) || (m.dispatch_status & 0x1) != 0

const _index_lock = ReentrantLock()
const _index = Ref{Union{Nothing,MethodIndex}}(nothing)

# number of live mutable-code methods and those of them newer than `world`
function _scan_mutable(roots, world)
    n = 0
    newer = Method[]
    Base.visit(Core.methodtable) do m
        (Base.moduleroot(m.module) in roots && _live(m)) || return
        n += 1
        m.primary_world > world && push!(newer, m)
    end
    return n, newer
end

function method_index()
    w = Base.get_world_counter()
    Base.@lock _index_lock begin
        ix = _index[]
        ix !== nothing && ix.world == w && return ix
        roots = _mutable_modules()[1]
        if ix !== nothing
            n, newer = _scan_mutable(roots, ix.world)
            if isempty(newer) && n == ix.nlive
                ix = MethodIndex(w, ix.version, ix.nlive, ix.sigs, ix.functions, ix.fps)
                _index[] = ix
                return ix
            end
            # only additions and redefinitions (e.g. a script's closures): update incrementally, dropping the memo
            # entries of the affected signatures and functions; deletions fall through to a full rebuild
            news = [(qualified(m.sig), m) for m in newer]
            if n == ix.nlive + count(((s, _),) -> !haskey(ix.sigs, s), news)
                sigs = copy(ix.sigs)
                functions = Dict(k => copy(v) for (k, v) in ix.functions)
                fps = copy(ix.fps)
                for (s, m) in news
                    k = _sig_function_key(m.sig)
                    if !haskey(sigs, s) && k !== nothing
                        v = get!(Vector{String}, functions, k)
                        insert!(v, searchsortedfirst(v, s), s)
                    end
                    sigs[s] = m
                    delete!(fps, string("M\t", s))
                    k === nothing || delete!(fps, string("F\t", k))
                end
                ix = MethodIndex(w, ix.version + 1, n, sigs, functions, fps)
                _index[] = ix
                return ix
            end
        end
        sigs = Dict{String,Method}()
        functions = Dict{String,Vector{String}}()
        nlive = 0
        Base.visit(Core.methodtable) do m
            (Base.moduleroot(m.module) in roots && _live(m)) || return
            nlive += 1
            s = qualified(m.sig)
            sigs[s] = m
            k = _sig_function_key(m.sig)
            k === nothing || push!(get!(Vector{String}, functions, k), s)
        end
        foreach(sort!, values(functions))
        ix = MethodIndex(w, ix === nothing ? 1 : ix.version + 1, nlive, sigs, functions, Dict{String,String}())
        _index[] = ix
        return ix
    end
end

## fingerprints

function _lowered(m::Method)
    try
        return Base.uncompressed_ast(m)
    catch
        return nothing
    end
end

function _print_code(io::IO, src)
    ctx = IOContext(io, :module => Core)
    for st in src.code
        show(ctx, st)
        print(io, '\n')
    end
    print(io, "slots ", src.slotnames, '\n')
end

function _method_fingerprint_text(m::Method)
    io = IOBuffer()
    print(io, qualified(m.sig), " nargs=", m.nargs, " isva=", m.isva)
    for f in (:nospecialize, :constprop, :purity, :max_varargs, :nkw)
        hasfield(Method, f) && isdefined(m, f) && print(io, ' ', f, '=', getfield(m, f))
    end
    print(io, '\n')
    if isdefined(m, :generator)
        gen = m.generator
        g = hasfield(typeof(gen), :gen) ? getfield(gen, :gen) : gen
        for gm in methods(g)
            src = _lowered(gm)
            src === nothing ? print(io, "generator ?\n") : _print_code(io, src)
        end
        f = string(m.file)
        print(io, "generator file ", isfile(f) ? _hex(read(f)) : "?", '\n')
    end
    if isdefined(m, :source) && m.source !== nothing
        src = _lowered(m)
        src === nothing ? print(io, "source ?\n") : _print_code(io, src)
    end
    return String(take!(io))
end

# Macro hygiene names locals `#<n>#name` (and `gensym` gives `##name#<n>`) from a process-wide counter, which shifts
# whenever any earlier macro expansion in the package changes. Renumber them by first appearance within one text:
# distinct names stay distinct, the counter drops out.
function _normalize_gensyms(txt::String)
    seen = Dict{String,Int}()
    io = IOBuffer()
    last = 1
    for m in eachmatch(r"#(\d+)(?=#)|(##[A-Za-z_][A-Za-z_0-9!]*#)(\d+)", txt)
        print(io, SubString(txt, last, prevind(txt, m.offset)))
        if m[1] !== nothing
            print(io, '#', 'g', get!(seen, m[1], length(seen) + 1))
        else
            print(io, m[2], 'g', get!(seen, m[3], length(seen) + 1))
        end
        last = m.offset + ncodeunits(m.match)
    end
    print(io, SubString(txt, last))
    return String(take!(io))
end

const _fp_lock = ReentrantLock()
const _method_fps = IdDict{Method,String}()

method_fingerprint(m::Method) =
    Base.@lock _fp_lock get!(() -> _hex(_normalize_gensyms(_method_fingerprint_text(m))), _method_fps, m)

# globals referenced (GlobalRef) by a method's lowered code (and its generator's)
function _globalrefs(m::Method)
    refs = GlobalRef[]
    srcs = Any[]
    isdefined(m, :source) && m.source !== nothing && push!(srcs, _lowered(m))
    if isdefined(m, :generator)
        gen = m.generator
        g = hasfield(typeof(gen), :gen) ? getfield(gen, :gen) : gen
        append!(srcs, (_lowered(gm) for gm in methods(g)))
    end
    walk(x) = x isa GlobalRef ? push!(refs, x) : x isa Expr ? foreach(walk, x.args) : nothing
    for src in srcs
        src === nothing || foreach(walk, src.code)
    end
    return refs
end

# constant global value -> fingerprint (`nothing`: not tracked: undefined, non-constant, function, module)
function _global_fingerprint(mod::Module, name::Symbol)
    (isdefined(mod, name) && isconst(mod, name)) || return nothing
    v = getglobal(mod, name)
    (v isa Module || (v isa Function && !(v isa Type))) && return nothing
    v isa Type && return _hex(string("type ", qualified(v)))
    txt = try
        describe(v)
    catch
        repr(v)
    end
    return _hex(txt)
end

# definition of a type's name: fields (names and declared types), supertype, kind
function _type_fingerprint(@nospecialize(T))
    T = Base.unwrap_unionall(T)
    T isa DataType || return nothing
    io = IOBuffer()
    print(io, qualified(T), " <: ", qualified(supertype(T)), " mutable=", ismutabletype(T), " primitive=",
        isprimitivetype(T))
    isprimitivetype(T) && print(io, " size=", sizeof(T))
    if isstructtype(T)
        for i in 1:fieldcount(T)
            print(io, '\n', fieldname(T, i), "::", qualified(fieldtype(T, i)))
        end
    end
    return _hex(_normalize_gensyms(String(take!(io))))
end

_type_key(@nospecialize(T)) = string(_modpath(T.name.module), ':', T.name.name)

function _resolve_type(key::AbstractString)
    r = _resolve_global(key)
    r === nothing && return nothing
    mod, name = r
    isdefined(mod, name) || return nothing
    t = getglobal(mod, name)
    t isa Type || return nothing
    return t
end

# mutable-code DataTypes inside a type (parameters, recursively)
function _collect_types!(out::Dict{String,Any}, @nospecialize(T), seen::Base.IdSet{Any})
    T in seen && return
    push!(seen, T)
    if T isa UnionAll
        _collect_types!(out, T.body, seen)
    elseif T isa DataType && T <: Function && Base.datatype_fieldcount(T) === 0
        # functions: tracked through their methods
    elseif T isa Union
        _collect_types!(out, T.a, seen)
        _collect_types!(out, T.b, seen)
    elseif T isa DataType
        if _is_mutable(T.name.module)
            k = _type_key(T)
            haskey(out, k) || (out[k] = T.name.wrapper)
        end
        for p in T.parameters
            _collect_types!(out, p, seen)
        end
    end
    return
end

## dependencies

# functions whose methods Enzyme looks up directly (custom rules, activity)
function _enzyme_rule_functions()
    keys = String[]
    for m in values(Base.loaded_modules)
        nameof(m) === :EnzymeCore || continue
        isdefined(m, :EnzymeRules) || continue
        R = getglobal(m, :EnzymeRules)
        for n in names(R; all = true)
            isdefined(R, n) || continue
            f = getglobal(R, n)
            f isa Function && !(f isa Type) && push!(keys, _function_key(typeof(f)))
        end
    end
    return keys
end

"""
    dependencies(roots, owners) -> (text, fallback)

Dependency text of a kernel (see the module docstring) from its code instances `roots` (e.g. the `ci`s of
GPUCompiler's `compiled` map). Edges are followed through method instances' code instances whose `owner` is in
`owners` (the inference caches the kernel was compiled with), falling back to any owner, and finally to the
method's package sources (`P`; the method instances that needed it are returned in `fallback`).
"""
function dependencies(roots, owners)
    lines = Set{String}()
    mis = Base.IdSet{Core.MethodInstance}()
    seen = Base.IdSet{Any}()
    stack = Any[roots...]
    fallback = Core.MethodInstance[]
    while !isempty(stack)
        x = pop!(stack)
        x in seen && continue
        push!(seen, x)
        if x isa Core.CodeInstance
            d = x.def
            d isa Core.ABIOverride && (d = d.def)
            push!(stack, d)
            isdefined(x, :edges) && append!(stack, x.edges)
        elseif x isa Core.MethodInstance
            push!(mis, x)
            cis = Core.CodeInstance[]
            if isdefined(x, :cache)
                ci = x.cache
                while ci isa Core.CodeInstance
                    push!(cis, ci)
                    ci = isdefined(ci, :next) ? ci.next : nothing
                end
            end
            own = filter(ci -> any(o -> o === ci.owner, owners), cis)
            if !isempty(own)
                append!(stack, own)
            elseif !isempty(cis)
                append!(stack, cis)
            elseif x.def isa Method
                push!(fallback, x)
            end
        elseif x isa Method
            k = _sig_function_key(x.sig)
            k === nothing || push!(lines, string("F\t", k))
            _is_mutable(x) && push!(lines, string("M\t", qualified(x.sig)))
        elseif x isa Type
            k = _sig_function_key(x)
            k === nothing || push!(lines, string("F\t", k))
        elseif x isa Core.Binding
            gr = x.globalref
            _is_mutable(gr.mod) && _global_fingerprint(gr.mod, gr.name) !== nothing &&
                push!(lines, string("G\t", _modpath(gr.mod), ':', gr.name))
        end
    end
    types = Dict{String,Any}()
    tseen = Base.IdSet{Any}()
    methods_ = Base.IdSet{Method}()
    for mi in mis
        _collect_types!(types, mi.specTypes, tseen)
        m = mi.def
        m isa Method && push!(methods_, m)
    end
    for m in methods_
        k = _sig_function_key(m.sig)
        k === nothing || push!(lines, string("F\t", k))
        _is_mutable(m) || continue
        push!(lines, string("M\t", qualified(m.sig)))
        for gr in _globalrefs(m)
            _is_mutable(gr.mod) || continue
            _global_fingerprint(gr.mod, gr.name) === nothing && continue
            push!(lines, string("G\t", _modpath(gr.mod), ':', gr.name))
            v = getglobal(gr.mod, gr.name)
            v isa Type && _collect_types!(types, v, tseen)
        end
    end
    # method instances without code instances: their callees are unknown. Mutable code -> its package's sources;
    # immutable code can only reach mutable code through mutable types in its signature -> those packages
    dirs = _mutable_modules()[2]
    used = Core.MethodInstance[]
    for mi in fallback
        m = mi.def::Method
        mods = Module[]
        if _is_mutable(m)
            push!(mods, Base.moduleroot(m.module))
        else
            ts = Dict{String,Any}()
            _collect_types!(ts, mi.specTypes, Base.IdSet{Any}())
            for T in values(ts)
                push!(mods, Base.moduleroot(Base.unwrap_unionall(T).name.module))
            end
        end
        isempty(mods) && continue
        push!(used, mi)
        for mod in unique!(mods)
            root = get(dirs, mod, nothing)
            push!(lines, root === nothing ? "P\t?" : string("P\t", root))
        end
    end
    ix = method_index()
    for k in _enzyme_rule_functions()
        push!(lines, string("F\t", k))
        for s in get(ix.functions, k, String[])
            push!(lines, string("M\t", s))
        end
    end
    for (k, T) in types
        _type_fingerprint(T) === nothing || push!(lines, string("T\t", k))
    end
    # fingerprints
    out = IOBuffer()
    for l in sort!(collect(lines))
        print(out, l, '\t', current_fingerprint(l, ix), '\n')
    end
    return String(take!(out)), used
end

"""
    current_fingerprint(line, index) -> String

Fingerprint of a dependency (`"<kind>\\t<name>"`) in the current session; `"absent"` if it no longer exists.
"""
function current_fingerprint(line::AbstractString, ix::MethodIndex = method_index())
    # M and F lines depend only on the index; G (constant values, redefinable without a method change) and P
    # (files) are recomputed; T lines are memoized per type object (a redefined type is a new object)
    (startswith(line, "M\t") || startswith(line, "F\t")) || return _current_fingerprint(line, ix)
    Base.@lock _index_lock begin
        fp = get(ix.fps, line, nothing)
        fp === nothing || return fp
    end
    fp = _current_fingerprint(line, ix)
    Base.@lock _index_lock ix.fps[String(line)] = fp
    return fp
end

const _type_fps = IdDict{Any,String}()
const _global_fps = Dict{Tuple{Module,Symbol,UInt},String}()

function _current_fingerprint(line::AbstractString, ix::MethodIndex)
    kind, name = split(line, '\t'; limit = 2)
    if kind == "M"
        m = get(ix.sigs, name, nothing)
        return m === nothing ? "absent" : method_fingerprint(m)
    elseif kind == "F"
        return _hex(join(get(ix.functions, name, String[]), '\n'))
    elseif kind == "G"
        r = _resolve_global(name)
        r === nothing && return "absent"
        mod, sym = r
        (isdefined(mod, sym) && isconst(mod, sym)) || return "absent"
        # memoized per value object (a redefined constant is a new value; isbits values are keyed by content)
        k = (mod, sym, objectid(getglobal(mod, sym)))
        fp = Base.@lock _index_lock get(_global_fps, k, nothing)
        if fp === nothing
            fp = something(_global_fingerprint(mod, sym), "absent")
            Base.@lock _index_lock _global_fps[k] = fp
        end
        return fp
    elseif kind == "T"
        T = _resolve_type(name)
        T === nothing && return "absent"
        fp = Base.@lock _index_lock get(_type_fps, T, nothing)
        if fp === nothing
            fp = something(_type_fingerprint(T), "absent")
            Base.@lock _index_lock _type_fps[T] = fp
        end
        return fp
    elseif kind == "P"
        return name == "?" || !isdir(name) ? "absent" : package_state(name)
    end
    return "absent"
end

"""
    check_dependencies(text; kinds = "MFGTP") -> Union{Nothing,Vector{String}}

`nothing` if every dependency in `text` whose kind is in `kinds` is unchanged, otherwise the (up to 5) changed
dependency lines.
"""
function check_dependencies(text::AbstractString; kinds::AbstractString = "MFGTP")
    ix = method_index()
    changed = String[]
    for l in eachsplit(text, '\n'; keepempty = false)
        first(l) in kinds || continue
        i = findlast('\t', l)
        i === nothing && return String["malformed"]
        dep = l[1:prevind(l, i)]
        current_fingerprint(dep, ix) == l[i+1:end] && continue
        push!(changed, dep)
        length(changed) >= 5 && break
    end
    return isempty(changed) ? nothing : changed
end

"""
    index_version() -> Int

Version of the mutable-code method index: unchanged versions mean `M` and `F` dependencies are unchanged, so a
dependency set already checked at this version only needs its `G`, `T` and `P` lines rechecked.
"""
index_version() = method_index().version

## storage

_base_dir(dir, basekey) = joinpath(dir, _hex(basekey))

"""
    load_entry(dir, basekey) -> (payload, deps) or (nothing, changed)

The most recent stored variant for `basekey` whose dependencies are unchanged. On a miss `changed` lists the
changed dependencies of the newest variant (empty if there is none).
"""
function load_entry(dir, basekey)
    bdir = _base_dir(dir, basekey)
    isdir(bdir) || return nothing, String[]
    variants = filter(f -> endswith(f, ".deps"), readdir(bdir; join = true))
    sort!(variants; by = mtime, rev = true)
    changed = String[]
    for (i, v) in enumerate(variants)
        deps = try
            read(v, String)
        catch
            continue
        end
        c = check_dependencies(deps)
        if c !== nothing
            i == 1 && (changed = c)
            continue
        end
        path = string(v[1:end-5], ".jls")
        try
            fmt, key, d, payload = open(deserialize, path)
            (fmt == FORMAT && key == basekey && d == deps) || continue
            return payload, deps
        catch err
            @debug "GravitationalWaveBayes kernel cache: unreadable entry treated as a miss" path exception = err
        end
    end
    return nothing, changed
end

"""
    store_entry!(dir, basekey, deps, payload)

Write atomically (temp file + rename): payload `.jls`, dependency text `.deps` (written last: it marks a complete
entry), and the base key text `<sha>.txt`. Keeps at most `MAX_VARIANTS` variants per base key. Errors are
reported, not thrown.
"""
function store_entry!(dir, basekey, deps, payload)
    bdir = _base_dir(dir, basekey)
    stem = joinpath(bdir, _hex(deps))
    tag = string(".tmp.", getpid(), ".", rand(UInt32))
    try
        mkpath(bdir)
        open(io -> serialize(io, (FORMAT, basekey, deps, payload)), stem * ".jls" * tag, "w")
        mv(stem * ".jls" * tag, stem * ".jls"; force = true)
        write(stem * ".deps" * tag, deps)
        mv(stem * ".deps" * tag, stem * ".deps"; force = true)
        txt = string(bdir, ".txt")
        isfile(txt) || (write(txt * tag, basekey); mv(txt * tag, txt; force = true))
        variants = filter(f -> endswith(f, ".deps"), readdir(bdir; join = true))
        if length(variants) > MAX_VARIANTS
            sort!(variants; by = mtime, rev = true)
            for v in variants[MAX_VARIANTS+1:end]
                rm(v; force = true)
                rm(string(v[1:end-5], ".jls"); force = true)
            end
        end
    catch err
        @warn "GravitationalWaveBayes kernel cache: could not write $stem" exception = err
        for f in (stem * ".jls" * tag, stem * ".deps" * tag)
            isfile(f) && rm(f; force = true)
        end
    end
    return nothing
end

end
