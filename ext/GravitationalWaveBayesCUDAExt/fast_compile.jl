# Fast-compile mode (`GRAVITATIONALWAVEBAYES_FAST_COMPILE=1`): device code keeps a fixed set of large `@inline`-declared waveform
# functions as calls instead of inlining them, so the kernels' PTX consists of several functions of moderate size
# instead of one huge entry function. ptxas time grows much faster than linearly with function size, and it is
# about half of a cold compile.
#
# Measured on a V100 (SEOBNRv5 (2,2) device likelihood + SEOBNRv5HM bin-parallel likelihood, 79 kernels,
# empty kernel cache): cold start 642 -> ~382 s, steady-state logL+grad +2 / +10 % (SEOBNR at 1024 / 4096 walkers),
# -1 / +3 % (HM at 256 / 1024 walkers). Meant for development and tests; production runs leave it off. Results are
# not bit-identical to the default build (logL ~1e-12 relative, gradients ~1e-8 of max |g|).
#
# Julia honours `@inline` regardless of size, and neither GPUCompiler's nor Enzyme's LLVM pipeline has a cost-based
# inliner, so the switch acts on Julia's inlining decision: `inline_cost_model` of the GPU and Enzyme interpreters is
# replaced by one that declines the listed functions. It is evaluated at `__init__` because it replaces
# GPUCompiler's own method for `GPUInterpreter` (not allowed during precompilation), and kernels must then be
# compiled in the latest world, which the kernel cache's miss path does (`_compile_latest`); without
# `GRAVITATIONALWAVEBAYES_KERNEL_CACHE` a per-process temporary cache directory is used. Only specializations whose arguments and
# result are bits values (or `nothing`) stay calls: a call returning, say, a NamedTuple with a `Type` field
# references a host object.
# Functions taking model records inside Enzyme reverse kernels must not be listed (Enzyme then emits 1-byte-aligned
# atomic shadow adds: "unsupported atomicrmw fadd: alignment 1" in ptxas, or CUDA_ERROR_MISALIGNED_ADDRESS); this is
# why `_seobv5hm_tup6`, `_seobv5_rr_force` and `_seobv5_flux_coeffs` (used by the SEOBNRv5PHM reverses) are not.

const FAST_COMPILE_FUNCTIONS = (
    :GravitationalWaveforms => (:_seobv5_rhs, :_seobv5_sym_gradient, :_seobv5_sym_auxderivs, :_seobv5_pr_eqn, :_seobv5_pphi_eqn_fixed,
        :_imr_amp, :_eobimr_R, :_seobv5hm_attach, :_seobv5hm_state, :_seobv5hm_R, :_seobv5hm_mix_k),
    :BatchedNUTS => (:_sdt_j0_gd, :_sdt_base_ift, :_sdt_diss_residual),
)

const _fast = Ref(false)

fast_compile_mode() = get(ENV, "GRAVITATIONALWAVEBAYES_FAST_COMPILE", "0") == "1"

# kernel-cache key line (empty in the default mode, so existing entries stay valid)
_fast_key() = _fast[] ? string("fast_compile=", FAST_COMPILE_FUNCTIONS, '\n') : ""

function _fast_listed(m::Method)
    mod = nameof(m.module)
    for (M, names) in FAST_COMPILE_FUNCTIONS
        M === mod && m.name in names && return true
    end
    return false
end

_fast_bitsish(@nospecialize(t)) = t === Nothing || (t isa DataType && isbitstype(t))

function _fast_decline(result::Core.Compiler.InferenceResult, inline_flag::UInt8)
    CC = Core.Compiler
    inline_flag === CC.SRC_FLAG_DECLARED_INLINE || return false
    m = result.linfo.def
    (m isa Method && _fast_listed(m)) || return false
    _fast_bitsish(CC.widenconst(CC.widenslotwrapper(result.result))) || return false
    sig = Base.unwrap_unionall(result.linfo.specTypes)
    return sig isa DataType && all(_fast_bitsish, sig.parameters)
end

function _init_fast_compile()
    fast_compile_mode() || return nothing
    if !COMPATIBLE
        @warn "GRAVITATIONALWAVEBAYES_FAST_COMPILE ignored: written for CUDACore $(SUPPORTED.CUDACore) / GPUCompiler " *
              "$(SUPPORTED.GPUCompiler), found $(pkgversion(CUDACore)) / $(pkgversion(GPUCompiler))"
        return nothing
    end
    ccall(:jl_generating_output, Cint, ()) == 0 || return nothing
    for I in (GPUCompiler.GPUInterpreter, Enzyme.Compiler.Interpreter.EnzymeInterpreter)
        @eval function Core.Compiler.inline_cost_model(interp::$I, result::Core.Compiler.InferenceResult,
                inline_flag::UInt8, ir::Core.Compiler.IRCode)
            CC = Core.Compiler
            _fast_decline(result, inline_flag) && return CC.MAX_INLINE_COST
            # what the replaced GPUCompiler 2.10.0 method does (`always_inline` jobs)
            if interp isa GPUCompiler.GPUInterpreter && GPUCompiler.force_inlineable(interp, result.linfo,
                    inline_flag === CC.SRC_FLAG_DECLARED_INLINE, inline_flag === CC.SRC_FLAG_DECLARED_NOINLINE,
                    CC.widenslotwrapper(result.result))
                return CC.MIN_INLINE_COST
            end
            return @invoke Core.Compiler.inline_cost_model(interp::Core.Compiler.AbstractInterpreter,
                result::Core.Compiler.InferenceResult, inline_flag::UInt8, ir::Core.Compiler.IRCode)
        end
    end
    _fast[] = true
    if _dir[] === nothing
        _dir[] = mktempdir()
        _verify[] = KernelCache.verify_mode()
        _log[] = KernelCache.log_mode()
    end
    @info "GravitationalWaveBayes fast-compile mode: large waveform functions stay calls in device code (slower kernels)"
    return nothing
end
