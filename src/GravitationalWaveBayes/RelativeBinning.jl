module RelativeBinning

export RelativeBinningSetup, build_relative_binning, binned_log_likelihood_kernel!

using KernelAbstractions
using ..Types: LikelihoodData
using ..GPU: to_device
using GravitationalWaveforms: Ampl, Phi, IMRPhenomD, IMRPhenomXAS, IMRPhenomXP, IMRPhenomXHM, IMRPhenomXPHM, AlignedSpinPNHamF2, SEOBNRv5AlignedSpinF2
using ..Types: PAF2Model
using GravitationalWaveforms: _phi_coeffs, _phi_scalar, _ampl_coeffs, _ampl_scalar, GMsun_over_c3, GMsun_over_c2_Gpc
using GravitationalWaveforms: _xas_phi_coeffs, _xas_phi_scalar, _xas_ampl_coeffs, _xas_ampl_scalar, _xas_precessing_final_spin
using GravitationalWaveforms: _asphf2_phi_coeffs, _asphf2_phi_scalar, _asphf2_ampl_coeffs, _asphf2_ampl_scalar
using GravitationalWaveforms: _seobv5f2_coeffs, _seobv5f2_phi_scalar, _seobv5f2_ampl_scalar
using GravitationalWaveforms: _xp_precession_setup, _xp_twist_up22, _xp_apply_zeta_rotation, _XP_HAS_MODE_NORMALIZATION
using GravitationalWaveforms: _xhm_system_setup, _xhm_mode_setup, _xhm_mode32_setup, _xhm_hphc_scalar, _xphm_hphc_scalar
using GravitationalWaveData: gmst, time_delay_from_geocenter, compute_antenna_response, geometry, antenna_response_gmst,
    time_delay_from_geocenter_gmst
using ..KernelFused: _xphm_prec_setup, _chirp_mass_eta, _phi_t0, _extrinsic_setup, _safe_div,
    _SEOBV5F2_DEADZONE_LOGLIKELIHOOD, _seobv5f2_apply_deadzone_reject

# ---------------------------------------------------------------------------------------------
# Relative binning (heterodyning): standard GW-PE technique to avoid evaluating the waveform on
# the full dense frequency grid on every sampler step. References for what's implemented below:
#
#   - N. Cornish, "Fast Fisher Matrices and Lazy Likelihoods" (2010), arXiv:1007.4820 -- the
#     original heterodyning idea (factor the trial waveform against a fixed reference waveform,
#     amplitude/phase ratio is smooth and can be handled on a coarse grid).
#   - B. Zackay, L. Dai, T. Venumadhav, "Relative Binning and Fast Likelihood Evaluation for
#     Gravitational Wave Inference" (2018), arXiv:1806.08792 -- the specific A0/A1/B0/B1 per-bin
#     summary-statistic formulation implemented here (their eqs. ~13-19): within each frequency
#     bin, the ratio r(f) = h(f;theta)/h(f;theta_ref) is well-approximated by a LINEAR function of
#     f, r(f) ~= r0_b + r1_b*(f - fm_b) (fm_b = bin's central frequency), so <d|h> and <h|h> reduce
#     from an O(n_freq) sum to an O(n_bins) sum against precomputed summary data that only depends
#     on the fixed reference waveform, the data, and the PSD (not on the trial theta).
#
# Summary data (this module's `A0`/`A1`/`B0`/`B1`, matching Zackay et al.'s notation), using this
# codebase's own inner-product convention (see `KernelFused.batched_log_likelihood_kernel!`'s docstring):
#     <a|b> = 4 * Re[ sum_k a_k * conj(b_k) / Sn_k ] * df
#
#     A0_b = 4*df * sum_{k in bin b} d_k * conj(href_k) / Sn_k
#     A1_b = 4*df * sum_{k in bin b} d_k * conj(href_k) / Sn_k * (f_k - fm_b)
#     B0_b = 4*df * sum_{k in bin b} |href_k|^2 / Sn_k                        (real)
#     B1_b = 4*df * sum_{k in bin b} |href_k|^2 / Sn_k * (f_k - fm_b)        (real)
#
# Then, for a trial theta with per-bin-edge ratio r0_b/r1_b (linear fit from the two bin-edge
# values of r(f) = h(f;theta)/href(f)):
#     Re<d|h>  ~= sum_b Re[A0_b * conj(r0_b) + A1_b * conj(r1_b)]
#     <h|h>    ~= sum_b [B0_b * |r0_b|^2 + 2*B1_b * Re(r0_b * conj(r1_b))]     (dropping the
#                 second-order-small (f-fm_b)^2*|r1_b|^2 term within each narrow bin, exactly as
#                 Zackay et al. do)
#
# and the full log-likelihood (this codebase's convention, ll = -0.5*<d-h|d-h>, see
# `KernelFused.batched_log_likelihood_kernel!`) is reconstructed as `dd_term + Re<d|h> - 0.5*<h|h>`, where `dd_term =
# -0.5*<d|d> = -2*df*sum(|d_k|^2/Sn_k)` is a theta-independent constant precomputed once.
#
# Bin placement: bins are chosen to be NARROWER where the reference phase changes fastest with
# frequency (the inspiral, at low f) -- rather than hardcoding a fixed power-law frequency
# spacing (a common simplification in some implementations, e.g. spacing uniform in f^(-5/3) to
# mimic the leading-order PN chirp phase), this module places edges at points of EQUAL
# accumulated |phase(theta_ref)| along the actual reference-waveform phase evaluated on the dense
# grid -- data-driven, requires no assumption about which PN order dominates, and directly gives
# the desired "finer at low frequency" behavior since inspiral phase accumulates fastest there
# for both IMRPhenomD and IMRPhenomXAS. This is a pragmatic choice consistent with the spirit of Zackay
# et al.'s bin-selection criterion (bound the phase/amplitude variation within a bin) rather than
# a reimplementation of their specific analytic bound.
# ---------------------------------------------------------------------------------------------

# `_chirp_mass_eta`, `_extrinsic_setup`, `_phi_t0`, `_safe_div` and the SEOBNR dead-zone reject are
# imported from `KernelFused` (one definition each; `@inline` functions inline across modules).
_rb_xas_family(model) = model isa IMRPhenomXAS || model isa IMRPhenomXP || model isa IMRPhenomXHM || model isa IMRPhenomXPHM

# Every kernel/helper below writes `Complex(cos(x), sin(x))` instead of the algebraically
# identical `exp(im * x)`. Reason: `Enzyme.autodiff(Enzyme.Reverse, ...)` on `CUDABackend()`
# hard-crashes the process (SIGABRT, an assert in Enzyme's TypeAnalysis.cpp) when differentiating
# `exp(im*x)`/`cis(x)` of an argument that depends on the differentiated parameter .
# This `cos`/`sin` split avoids the crash entirely, while the
# combined `sincos(x)` intrinsic does NOT (a second, different Enzyme assert). Workaround pending
# an upstream Enzyme fix; applied here across every model -- see the fuller
# note in `KernelFused.jl` above `_fused_waveform_likelihood_kernel!`.

# IMRPhenomXP per-frequency detector-frame strain: the exact
# per-frequency chain `KernelFused._fused_waveform_likelihood_kernel_phenomxp!`/`_multi!` use
# inline (co-precessing-frame aligned-(2,2)-mode scalar core -> `_XP_HAS_MODE_NORMALIZATION` ->
# `_xp_twist_up22` -> `_xp_apply_zeta_rotation` -> antenna projection), factored into one `@inline`
# helper here (unlike KernelFused.jl, which only ever needs it once per frequency in a plain
# accumulator loop) since relative binning needs to evaluate it TWICE per bin (`r_lo`/`r_hi`, see
# the kernel bodies below) -- avoids duplicating the twisting-up chain inline at each call site.
# `setup`/`ampl_coeffs` are detector-independent (passed in already hoisted, matching
# `KernelFused.jl`'s own hoisting split); `phi_coeffs`/`Fplus`/`Fcross` are detector-specific (built
# from that detector's own time-delay-shifted `t0` and antenna response) and passed in fresh by the
# caller. Non-allocating scalar arithmetic throughout -- same Enzyme-vs-allocation discipline as
# every other `@inline` helper in this file.
@inline function _xp_h_det_scalar(f, M_sec, ampl_coeffs, phi_coeffs, setup, Fplus, Fcross)
    T = typeof(Fplus)
    amp = _xas_ampl_scalar(ampl_coeffs, f)
    phase_val = _xas_phi_scalar(phi_coeffs, f)
    hAS = (amp * Complex(cos(phase_val), sin(phase_val))) * T(_XP_HAS_MODE_NORMALIZATION)

    Mf = M_sec * f
    v = cbrt(T(pi) * Mf)
    hp, hc = _xp_twist_up22(v, setup, hAS)
    hp, hc = _xp_apply_zeta_rotation(hp, hc, setup.zeta_polarization)

    return Fplus * hp + Fcross * hc
end

# IMRPhenomXHM per-frequency detector-frame strain: mirrors
# `_xp_h_det_scalar` above exactly in role (the per-frequency chain
# `KernelFused._fused_waveform_likelihood_kernel_phenomxhm!` uses inline, factored out here since
# relative binning needs it evaluated TWICE per bin, at `edges[b]`/`edges[b+1]`, by both this
# file's host-side setup code and the sparse kernel bodies below).
#
# `sys`/`ms21`/`ms33`/`ms44`/`m32` are the frequency-independent per-walker setup
# (`_xhm_system_setup`/`_xhm_mode_setup`(x3)/`_xhm_mode32_setup`), computed ONCE by the caller and
# passed in already-hoisted -- exactly like `_xp_h_det_scalar`'s `setup` argument.
# `theta_jn`/`piHalf` are `_xhm_hphc_scalar`'s `inclination`/`phi` arguments (see
# `KernelFused.jl`'s IMRPhenomXHM kernel-body comment block for why `phi` is a FIXED `pi/2`, not
# `phase`/`phiRef`). `t_c` has no native input to `_xhm_system_setup`/`hphc` (unlike `Phi`'s `t0`
# kwarg) -- applied here as the SAME direct post-multiplication in physical frequency `f`,
# `(hp,hc) *= exp(im*2*pi*f*t_c)`, that `KernelFused.jl`'s dense kernel body derives and empirically
# verifies (see that kernel's own comment block / `test/test_phenomxhm.jl`'s "t_c convention" grid
# scan) -- reused here unchanged, not re-derived.
@inline function _xhm_h_det_scalar(f, sys, ms21, ms33, ms44, m32, theta_jn, piHalf, t_c, Fplus, Fcross)
    T = sys.T
    hp, hc = _xhm_hphc_scalar(f, sys, ms21, ms33, ms44, m32, theta_jn, piHalf)

    tc_phase = -2 * T(pi) * f * t_c   # time sign: see KernelFused _PHASE_CONVENTION
    shift = Complex{T}(cos(tc_phase), sin(tc_phase))
    hp *= shift
    hc *= shift

    return Fplus * hp + Fcross * hc
end

# IMRPhenomXPHM per-frequency detector-frame strain: combines `_xp_h_det_scalar`'s
# role (precession setup, `_xp_apply_zeta_rotation`) with `_xhm_h_det_scalar`'s (higher-mode
# assembly via a non-allocating scalar core, `t_c` as a direct post-multiplication since
# `hphc(::IMRPhenomXPHM, ...)` has no native `t_c` input either) -- exactly the same two-way
# combination `KernelFused._fused_waveform_likelihood_kernel_phenomxphm!` makes of its own two
# single-model ancestors. `setup` is `_xp_precession_setup`'s output; `sys`/`ms21`/`ms33`/`ms44`/
# `m32` are `_xhm_system_setup` (called with `final_spin_override`, see that kernel's own
# comment block for why omitting it leaves a ~7% amplitude gap) and its four per-mode setups, all computed ONCE by the caller and passed in
# already-hoisted, exactly like `_xp_h_det_scalar`'s `setup` argument.
@inline function _xphm_h_det_scalar(f, setup, sys, ms21, ms33, ms44, m32, t_c, Fplus, Fcross)
    T = sys.T
    hp, hc = _xphm_hphc_scalar(f, setup, sys, ms21, ms33, ms44, m32)
    hp, hc = _xp_apply_zeta_rotation(hp, hc, setup.zeta_polarization)

    tc_phase = -2 * T(pi) * f * t_c   # time sign: see KernelFused _PHASE_CONVENTION
    shift = Complex{T}(cos(tc_phase), sin(tc_phase))
    hp *= shift
    hc *= shift

    return Fplus * hp + Fcross * hc
end

# Guards `numer / href_edges[b]` against a literal `x/0` at a bin edge where the REFERENCE
# waveform's amplitude has underflowed to exact IEEE `0.0` -- a real, previously-unhit failure
# mode found running relative binning on the full-resolution (20-1024 Hz, 4017-point) real
# GW150914 analysis band for the first time (every prior validation used small synthetic dense
# grids, 16-129 points, that never reached this regime). IMRPhenomD/XAS's amplitude decays
# exponentially past ringdown and genuinely underflows to `0.0` well before 1024 Hz at
# GW150914-scale masses (confirmed empirically: ~938 Hz for m1=36/m2=29 Msun) -- any bin edge
# landing past that point makes `href_edges[b] == 0`, and the resulting NaN/Inf then gets
# multiplied into `dh_sum`/`hh_sum` via `A0`/`B0`/`B1`, which are ALREADY ~0 for that bin (they're
# themselves dense sums weighted by `|href|^2`/`href*conj(strain)` over the same
# negligible-amplitude region) -- so the mathematically correct contribution is ~0, but IEEE
# `0 * NaN = NaN` poisons the ENTIRE reduction (every walker, including at theta_ref itself)
# instead of vanishing. Returning `zero(numer)` exactly at `denom == 0` (not a wider
# epsilon-band -- no other bin in any existing validated test ever hits this) fixes precisely this
# failure mode without perturbing any well-conditioned bin. Branch-free (`ifelse`, not `if`), same
# Enzyme/KernelAbstractions-safe pattern as `_clamp_unit`'s pole guard in `BatchedNUTS.jl` and the
# region-selection `ifelse`s throughout `GravitationalWaveforms.jl`'s scalar cores.
# Multiply by a guarded inverse: see `KernelFused._safe_div` for why (Enzyme adjoint 0/0 = NaN in the
# selected-away lane; a guarded denominator is folded away by LLVM).
@inline function _safe_ratio(numer::Complex{T}, denom::Complex{T}) where {T}
    return numer * ifelse(denom == zero(denom), zero(denom), inv(denom))
end

"""
    RelativeBinningSetup{T,E,C}

Precomputed relative-binning summary data for a fixed reference parameter set `theta_ref` and a
fixed `(data, psd)` pair. Built once per analysis by [`build_relative_binning`](@ref) (whose OWN
internal computation always runs host-side, reading `data.freqs`/`.strain`/`.psd` scalar-by-scalar
-- that part is unchanged), then reused for every sampler step via
[`binned_log_likelihood_kernel!`](@ref) -- none of these fields depend on the trial parameters
being evaluated.

**GPU-portability**: `E`/`C` are free type parameters (mirroring `PEState`'s own `{T,A,V}` pattern
in `Types.jl`) rather than hardcoded `Vector{T}`/`Vector{Complex{T}}` -- every `build_relative_binning`
method wraps its freshly-built host arrays in [`to_device`](@ref GravitationalWaveBayes.GPU.to_device) immediately
before constructing the returned struct, so `E`/`C` become `CuVector`/`CuVector{Complex{T}}` under
the CUDA extension (same global device-selection convention `compatible_array`/`to_device` already
use everywhere else in this codebase) and plain `Vector`/`Vector{Complex{T}}` otherwise -- callers
never branch on which backend is active. This is ONLY the "does the array live on the right device
for `binned_log_likelihood_kernel!`'s own kernel launch" concern; `build_relative_binning`'s own
setup-time computation (reading `data.freqs`/`.strain`/`.psd`) is still ALWAYS host-side regardless
of `E`/`C` (`data` itself keeps its own array type, generally host `Vector`s even in a GPU-backed
NUTS run, since this setup cost is paid once per analysis, not once per leapfrog step -- unlike
`positions`/`momenta`/`logp`/`energy`, which genuinely need `compatible_array` inside the sampling
hot loop).

Fields:
- `edges::E`: sparse bin-edge frequencies (length `nbins+1`), a subset of the original
  dense frequency grid.
- `fm::E`: per-bin central frequency (length `nbins`), the expansion point for the
  linear ratio fit `r(f) ~= r0_b + r1_b*(f - fm_b)`. NOT itself passed into
  `binned_log_likelihood_kernel!`'s kernel launch (kept for external diagnostics only) -- still
  wrapped in `to_device` for type-parameter consistency with `edges`/`B0`/`B1`, not because the
  kernel needs it there.
- `href_edges::C`: reference waveform `h(theta_ref; f)` evaluated at each bin
  edge (length `nbins+1`) -- needed at kernel-eval time to convert a trial waveform's bin-edge
  values into the ratio `r`.
- `A0`, `A1::C`, `B0`, `B1::E`: the per-bin relative-binning summary
  coefficients (length `nbins` each), see this file's header comment for their definition.
- `dd_term::T`: the theta-independent `-0.5*<d|d>` constant, precomputed once so
  `binned_log_likelihood_kernel!` reproduces the same likelihood normalization as
  `KernelFused.batched_log_likelihood_kernel!`.
"""
struct RelativeBinningSetup{T, E<:AbstractVector{T}, C<:AbstractVector{Complex{T}}}
    edges::E
    fm::E
    href_edges::C
    A0::C
    A1::C
    B0::E
    B1::E
    dd_term::T
end

"""
    _phase_equal_bin_edges(phase, nbins)

Choose `nbins+1` indices into the dense grid `phase` (the reference waveform's phase, evaluated
on the full dense frequency grid) such that each of the `nbins` resulting bins spans an
approximately equal increment of ACCUMULATED |phase difference| -- since phase accumulates
fastest at low frequency for an inspiralling CBC, this concentrates bins there and leaves the
slowly-evolving merger/ringdown region coarsely sampled. Falls back to uniform index spacing if
the reference phase is degenerate (e.g. flat/zero, as it never is for a real IMRPhenomD/XAS
waveform, but which some tiny synthetic test fixtures could in principle hit).

Returns a sorted `Vector{Int}` of unique dense-grid indices, `edges_idx[1] == 1` and
`edges_idx[end] == length(phase)` always; `nbins` is silently clipped to `length(phase) - 1` if
the dense grid is too short to support the requested bin count, and duplicate target indices
(possible when phase accumulates very unevenly) are deduplicated, so the returned vector may
contain fewer than `nbins+1` entries -- callers must use `length(edges_idx) - 1` as the actual
bin count, not the requested `nbins`.
"""
function _phase_equal_bin_edges(phase::AbstractVector{T}, nbins::Int) where {T}
    n = length(phase)
    nbins = clamp(nbins, 1, n - 1)

    cum = Vector{T}(undef, n)
    cum[1] = zero(T)
    @inbounds for i in 2:n
        cum[i] = cum[i-1] + abs(phase[i] - phase[i-1])
    end
    total = cum[end]

    edges_idx = Vector{Int}(undef, nbins + 1)
    edges_idx[1] = 1
    edges_idx[end] = n

    if !(total > 0)
        # Degenerate reference phase (constant/flat): fall back to uniform index spacing.
        @inbounds for b in 2:nbins
            edges_idx[b] = 1 + round(Int, (b - 1) * (n - 1) / nbins)
        end
    else
        j = 1
        @inbounds for b in 1:(nbins-1)
            target = total * b / nbins
            while j < n && cum[j] < target
                j += 1
            end
            edges_idx[b+1] = j
        end
    end

    return sort(unique(edges_idx))
end

"""
    _cap_bin_width_for_extrinsic_ramp(edges_idx, freqs, M_sec, max_tc_deviation, max_extrinsic_phase)

POST-PROCESSES [`_phase_equal_bin_edges`](@ref)'s output by inserting extra dense-grid indices into
any bin whose frequency span would let the model-independent extrinsic time-shift ramp (see below)
accumulate more than `max_extrinsic_phase` radians across it -- and otherwise leaves `edges_idx`
completely unchanged. This is deliberately an ADDITIVE, edges-only-get-denser transformation: it
never removes or relocates an edge `_phase_equal_bin_edges` already placed, so for any model whose
own intrinsic-phase-driven bins are already narrower than the cap everywhere (every wideband,
many-cycle model this file already supports), this is a no-op -- zero regression risk by
construction, unlike an earlier version of this fix (see git history) that instead added a
synthetic ramp directly into the phase array `_phase_equal_bin_edges` sees, which measurably
regressed `IMRPhenomD`/`IMRPhenomXAS`'s own existing dense-vs-binned agreement (confirmed via
`test/test_relative_binning.jl`) by letting the added ramp compete
with and sometimes dominate the model's own placement heuristic even where it didn't need to.

Why this is needed at all: `_extrinsic_setup`'s `t0 = -2*pi*t_c/M_sec` (or
`-2*pi*(t_c+dt_geo)/M_sec` per detector) enters the full phase as a purely linear,
frequency-independent ramp of slope `-t0` rad/Hz -- completely unrelated to the model's own
intrinsic phase gradient `_phase_equal_bin_edges` places bins against. At `theta_ref` itself this
ramp is irrelevant (trial `t_c` == reference `t_c`, no residual ramp), but a NUTS trial point's
`t_c` need not match `params_ref`'s exactly, and the RESIDUAL ramp (slope proportional to the
trial/reference `t_c` MISMATCH, up to `max_tc_deviation` seconds) must still be resolved by the
same fixed bin edges chosen once at setup time. For a wideband, many-cycle model the model's own
accumulation already forces bins finer than this cap almost everywhere; for a narrow,
low-cycle-count band (e.g. `SEOBNRv5AlignedSpinF2`'s post-adiabatic table) it does not, which is
exactly the aliasing bug this fixes (see
`test/test_relative_binning.jl`).

**Defaults to a no-op** (`max_tc_deviation=0`/`max_extrinsic_phase=0` in every
[`build_relative_binning`](@ref) method) -- this is deliberately OPT-IN, not a new default
behavior for existing callers/models. Reason:
any nonzero setting strong enough to meaningfully help `SEOBNRv5AlignedSpinF2`'s own narrow band
ALSO forces every existing wideband model (`IMRPhenomD`/`IMRPhenomXAS`/etc.) down to
near-native-grid-density bins almost everywhere -- e.g. `max_tc_deviation=1e-4`s (already the
SMALLEST of this model's own documented failing `t_c` values) inflated `IMRPhenomD`'s own
20-284Hz/`df=0.5`Hz testset from 62 bins to 527 (of 529 possible) -- because the ramp's rad/Hz rate
depends only on the system's total mass (`M_sec`) and the requested `max_tc_deviation`, NOT on the
model, so it applies just as forcefully to a model whose own placement was already adequate. An
unconditional default would silently destroy relative binning's whole performance benefit for
every existing user. Callers that need this robustness (e.g. a future `SEOBNRv5AlignedSpinF2`-
specific analysis) should pass explicit, small values sized to their OWN dense grid's `df` (see
next paragraph) -- not a blanket "safe" constant, since none exists across grids of different
coarseness.

**A hard ceiling this fix cannot cross, discovered while validating it (not assumed, not present
in the original diagnosis)**: `binned_log_likelihood_kernel!`'s per-bin ratio model is a two-point
SECANT fit, `r(f) ~= r0 + r1*(f-fm)`, built from the trial/reference ratio evaluated ONLY at each
bin's two edge frequencies (see that function's body). For a `t_c`-only trial mismatch, the TRUE
ratio is `r(f) = exp(i*(-t0*f + const))` -- a complex EXPONENTIAL of a linear phase, not a linear
function itself -- so the secant fit is only accurate when the ratio's phase changes little between
the two sample points, i.e. `|t0|*Δf << 1` for a bin of width `Δf`. Since no bin can be narrower
than the dense grid's own spacing `df`, this method CANNOT resolve a `t_c` mismatch once
`|t0|*df >~ 1` regardless of which points become edges -- confirmed directly: forcing
`SEOBNRv5AlignedSpinF2`'s `20.5-30.5Hz`/`df=0.05`Hz test grid to FULL native density (every one of
its 201 points an edge) still left `~18-57` log-likelihood absolute disagreement for
`t_c_trial in (1e-3, 2e-3, 5e-3)`s (this system's `|t0|*df` there is `~1-5`), while `t_c_trial=1e-4`s
(`|t0|*df~0.98`, right at the edge) dropped from `~29` (this model's existing coarse placement) to
`~0.2` once bins were pushed to native density. Below that (`|t0|*df` a few times smaller,
comfortably resolved by the grid), agreement is excellent. This means `max_tc_deviation` should be
sized as roughly `phase_tol*M_sec/(2*pi*df)` for the target grid (small multiple of `df`'s own
resolving power), not the sampler's full prior width -- a Δt_c mismatch beyond that needs either a
finer dense grid or an analytically de-ramped relative-binning formulation (factor the known linear
`t_c` term out of the ratio before interpolating, re-apply it exactly at evaluation time), neither
attempted here.

`max_tc_deviation<=0` or `max_extrinsic_phase<=0` is a no-op (returns `edges_idx` unchanged). When
the requested cap can't be met even at the dense grid's own native resolution (`freqs`' spacing),
this degrades gracefully to using every available dense-grid point in that stretch -- slower there
(lost relative-binning speedup, ONLY in the affected stretch), never silently wrong or unbounded
(the split count is capped at the number of native grid steps actually available, so an
unreasonably small `max_extrinsic_phase`/large `max_tc_deviation` can't blow up into a
near-infinite loop -- without the cap the loop ran for more than 600 s).
"""
function _cap_bin_width_for_extrinsic_ramp(edges_idx::Vector{Int}, freqs::AbstractVector{T},
                                            M_sec::T, max_tc_deviation::T, max_extrinsic_phase::T) where {T}
    (max_tc_deviation <= 0 || max_extrinsic_phase <= 0) && return edges_idx
    extra_rate = 2 * T(pi) * max_tc_deviation / M_sec
    extra_rate <= 0 && return edges_idx
    max_df = max_extrinsic_phase / extra_rate

    out = Int[edges_idx[1]]
    @inbounds for k in 1:(length(edges_idx)-1)
        lo, hi = edges_idx[k], edges_idx[k+1]
        fspan = freqs[hi] - freqs[lo]
        # Capped at `hi-lo` (the number of native dense-grid steps actually available in this
        # bin) -- without this cap, a tiny `max_extrinsic_phase`/large `max_tc_deviation` can
        # demand a `ceil(fspan/max_df)` many orders of magnitude larger than the grid could ever
        # provide, turning the loop below into a near-infinite one (without this cap it ran for
        # more than 600 s). Since a bin narrower than one grid
        # step doesn't exist, `hi-lo` IS the finest this bin can ever be split.
        nsplit = fspan > max_df ? min(ceil(Int, fspan / max_df), hi - lo) : 1
        if nsplit <= 1
            push!(out, hi)
        else
            for s in 1:nsplit
                idx = clamp(lo + round(Int, s * (hi - lo) / nsplit), lo + 1, hi)
                idx != out[end] && push!(out, idx)
            end
        end
    end
    return sort(unique(out))
end

# Bin-PLACEMENT heuristic model, distinct from the actual waveform model `build_relative_binning`
# is building a reference for. `_phase_equal_bin_edges` only needs a cheap proxy for where the
# (2,2)-mode intrinsic phase varies fastest -- for `IMRPhenomD`/`IMRPhenomXAS`/`AlignedSpinPNHamF2`,
# the model's own `Phi` already IS that cheap proxy, so this is the identity. `SEOBNRv5AlignedSpinF2`
# is the one model this file's GENERIC method (below) handles whose own `Phi` is expensive (a
# PA-table-backed EOB evaluation, not a closed-form fit) -- for it, use `IMRPhenomXAS()` instead,
# exactly the same proxy the file's IMRPhenomXP/XHM/XPHM-specific methods already use for their own
# placement heuristic (see e.g. `build_relative_binning(model::IMRPhenomXHM, ...)`'s docstring).
# `SEOBNRv5AlignedSpinF2`'s own output is dominant-`(2,2)`-mode-only (same physical content class as
# `IMRPhenomXAS`), so this is not a loss of accuracy for the heuristic, only a cost cut -- the actual
# reference-waveform values (`href_dense`/`A0`/`A1`/`B0`/`B1`) still come from the real `model`,
# unaffected by this choice, exactly as for the XP/XHM/XPHM methods' own use of the same proxy.
_bin_placement_model(model) = model
_bin_placement_model(::PAF2Model) = IMRPhenomXAS()

"""
    build_relative_binning(model, params_ref, data::LikelihoodData; nbins=150)

Host-side, once-per-analysis setup for relative binning. `params_ref` is an 11-element vector
`[m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi]`, the same PHYSICAL parametrization
`batched_log_likelihood_kernel!`/`KernelFused.jl` consume (see that file's module docstring for
the full row convention) -- NOT the sampler's `(mc, q, chi1, chi2, dL, t_c, phase, ra, sin_dec,
cos_theta_jn, psi)` layout; convert first if `params_ref` comes from sampler-space (e.g. via
`BatchedNUTS._mc_q_to_m1_m2` plus `asin`/`acos` for the two reparametrized angles).

The reference waveform `href_dense` is the FULL detector-frame projection at `params_ref` --
distance-scaled, `t_c`/`phase`-shifted, plus/cross-decomposed by `theta_jn`, and antenna-response-
projected via `data.detector`/`data.gps_time` -- computed with the exact same `_extrinsic_setup`
helper (mirrored in this file, see its docstring) the dense kernel bodies use, so relative binning
approximates the SAME likelihood surface `KernelFused.jl` evaluates on the full grid, not a
face-on/no-extrinsic approximation of it.

Bin-edge PLACEMENT (via [`_phase_equal_bin_edges`](@ref), narrower at low frequency) is primarily
driven by the bare INTRINSIC phase (`Phi(_bin_placement_model(model), freqs, mc, eta, chi1, chi2)`,
no `t0`/`phiRef`/antenna terms) rather than the full extrinsic phase: this is a heuristic for where
the waveform's SHAPE varies fastest (justifying finer bins in the inspiral), and per
[`_bin_placement_model`](@ref) uses a cheap `IMRPhenomXAS()` proxy rather than `model`'s own `Phi`
for `SEOBNRv5AlignedSpinF2` (whose own `Phi` is a PA-table-backed EOB evaluation, not cheap to run
across the full dense grid just for a placement heuristic) -- identity for every other model this
method handles. Folding in `params_ref`'s OWN actual
`t_c`-driven linear-in-frequency term directly would add a roughly uniform-in-f phase-accumulation
rate that scales with however large `params_ref`'s `t_c` itself happens to be (skewing the heuristic
toward uniform spacing for large `|t_c|`, defeating its purpose) without changing where the
intrinsic waveform shape itself varies fastest -- so that is NOT what's added.

Instead, a fixed, `params_ref`-independent SAFETY FLOOR on bin WIDTH (not on placement priority) is
applied afterward via [`_cap_bin_width_for_extrinsic_ramp`](@ref) (see its docstring for the full
mechanism and rationale, and for why this is a strictly additive, regression-proof-by-construction
post-processing step rather than a change folded into the placement heuristic itself -- an earlier
version of this fix tried the latter and measurably regressed existing wideband models). It
guarantees no bin is wider than the frequency span over which a residual `t_c` (or per-detector
arrival-time) mismatch of up to `max_tc_deviation` seconds between a NUTS trial point and
`params_ref` would accumulate more than `max_extrinsic_phase` radians of extra, model-independent
phase -- which only matters (widens bin count above what the model's own accumulation alone would
pick) for a narrow, low-cycle-count band like `SEOBNRv5AlignedSpinF2`'s; for wideband, many-cycle
models the model's own gradient already forces bins narrower than this cap almost everywhere, so
placement is unchanged for those. The per-bin `A0`/`A1`/`B0`/`B1` summary coefficients and
`href_edges` below still use the FULL extrinsic `href_dense`, so accuracy of the actual
relative-binning approximation is unaffected by this choice -- only which frequencies get edges.

`max_tc_deviation`/`max_extrinsic_phase` (keywords below, BOTH DEFAULT TO `0` = exact pre-fix
behavior, opt-in only): see [`_cap_bin_width_for_extrinsic_ramp`](@ref)'s docstring for the exact
roles, why the default must stay `0` (a nonzero default would silently collapse relative binning's
speedup for every existing model, confirmed not assumed), and the hard native-grid-resolution
ceiling on what any setting can actually fix.

Computes the reference waveform on `data`'s full dense frequency grid (`data.freqs`, one
`Ampl`/`Phi` call each -- this is the only place this module evaluates the array-oriented,
allocating waveform API; it runs once, on the host, well outside any hot loop), picks `nbins`
non-uniform bin edges via [`_phase_equal_bin_edges`](@ref) (narrower at low frequency), and
reduces the dense data/reference-waveform/PSD arrays into the `A0`/`A1`/`B0`/`B1` per-bin summary
coefficients described in this file's header comment.

`nbins` is a target; see [`_phase_equal_bin_edges`](@ref)'s docstring for when the actual bin
count returned (`length(result.edges) - 1`) can be smaller.
"""
function build_relative_binning(model, params_ref::AbstractVector{T}, data::LikelihoodData;
                                 nbins::Int=150, max_tc_deviation::Real=0,
                                 max_extrinsic_phase::Real=0) where {T}
    # Every `w = 4*df/psd[k]` weight below (this method and its
    # IMRPhenomXHM/IMRPhenomXPHM siblings, all 4 single-detector `build_relative_binning` methods --
    # multi-detector reuses these per-detector, so no separate site there) was a genuinely missed
    # division site from the earlier "targeted" PSD audit -- that pass fixed `dd_term`'s OWN
    # separate `abs2(strain[k])/psd[k]` division (`_safe_div`'s docstring above) but not this one.
    # Unlike a dense-kernel PSD division (re-evaluated every leapfrog step, affecting only that one
    # step if triggered), a zero-valued `psd[k]` here would make `w == Inf`, poisoning
    # `A0`/`A1`/`B0`/`B1` with `Inf`/`NaN` ONCE at `build_relative_binning` setup time -- silently
    # corrupting every walker's log-likelihood for the ENTIRE subsequent NUTS run, not just one
    # evaluation. Regression coverage
    # in `test/test_relative_binning.jl`.
    @assert length(params_ref) == 11
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = params_ref
    gmst_val = gmst(data.gps_time)
    mc, eta, t0, phiRef, combo = _extrinsic_setup(m1, m2, dL, t_c, phase, ra, dec, theta_jn, psi, geometry(data.detector), gmst_val)

    freqs = data.freqs
    strain = data.strain
    psd = data.psd
    n = length(freqs)
    @assert length(strain) == n && length(psd) == n

    amp_ref = Ampl(model, freqs, mc, eta, chi1, chi2, dL)
    phase_ref_bare = Phi(_bin_placement_model(model), freqs, mc, eta, chi1, chi2)  # bin-PLACEMENT heuristic only, see docstring above and `_bin_placement_model`
    M_sec = (mc / eta^T(0.6)) * T(GMsun_over_c3)
    # `_phi_t0`: the library's SEOBNR `Phi` takes `t0 = 2*pi*seconds`, the Phenom/PN ones an Mf-based `t0`.
    # Same convention as every kernel (KernelFused `_PHASE_CONVENTION`): XAS family flips the time sign,
    # the PhenomD/PN/PA-F2 family is conjugated as a whole.
    phase_ref_full = _rb_xas_family(model) ?
        Phi(model, freqs, mc, eta, chi1, chi2; t0=-_phi_t0(model, t0, M_sec), phiRef=phiRef) :
        .-Phi(model, freqs, mc, eta, chi1, chi2; t0=_phi_t0(model, t0, M_sec), phiRef=phiRef)
    href_dense = combo .* (amp_ref .* exp.(im .* phase_ref_full))

    edges_idx = _phase_equal_bin_edges(phase_ref_bare, nbins)
    edges_idx = _cap_bin_width_for_extrinsic_ramp(edges_idx, freqs, M_sec, T(max_tc_deviation), T(max_extrinsic_phase))
    nb = length(edges_idx) - 1
    edges = Vector{T}(undef, nb + 1)
    @inbounds for i in eachindex(edges_idx)
        edges[i] = freqs[edges_idx[i]]
    end
    href_edges = href_dense[edges_idx]

    df = freqs[2] - freqs[1]
    fm = Vector{T}(undef, nb)
    A0 = Vector{Complex{T}}(undef, nb)
    A1 = Vector{Complex{T}}(undef, nb)
    B0 = Vector{T}(undef, nb)
    B1 = Vector{T}(undef, nb)

    @inbounds for b in 1:nb
        lo = edges_idx[b]
        hi = b == nb ? edges_idx[b+1] : edges_idx[b+1] - 1   # half-open [lo,hi) except last bin
        fmb = T(0.5) * (edges[b] + edges[b+1])
        fm[b] = fmb

        a0 = zero(Complex{T})
        a1 = zero(Complex{T})
        b0 = zero(T)
        b1 = zero(T)
        for k in lo:hi
            w = _safe_div(T(4) * df, psd[k])
            dhc = strain[k] * conj(href_dense[k])
            hh = abs2(href_dense[k])
            dfk = freqs[k] - fmb
            a0 += w * dhc
            a1 += w * dhc * dfk
            b0 += w * hh
            b1 += w * hh * dfk
        end
        A0[b] = a0
        A1[b] = a1
        B0[b] = b0
        B1[b] = b1
    end

    dd_term = zero(T)
    @inbounds for k in 1:n
        dd_term -= 2 * df * _safe_div(abs2(strain[k]), psd[k])
    end

    return RelativeBinningSetup(to_device(edges), to_device(fm), to_device(href_edges),
                                 to_device(A0), to_device(A1), to_device(B0), to_device(B1), dd_term)
end

"""
    build_relative_binning(model, params_ref, data::AbstractVector{<:LikelihoodData}; nbins=150)
        -> Vector{RelativeBinningSetup}

Multi-detector entry point: builds one [`RelativeBinningSetup`](@ref) per
detector in `data`, by reusing the single-detector method above once per detector — NOT a
reimplementation. All entries of `data` must share the same `gps_time` (same physical event);
asserted, not assumed.

Physical reasoning: `params_ref`'s row 6 (`t_c`) is the coalescence-time offset at the GEOCENTER
(shared across detectors, same as the kernel bodies below), but each detector's own reference
waveform needs its OWN arrival-time shift, `t_c + GravitationalWaveData.time_delay_from_geocenter(detector, ra,
dec, gps_time)`. This function is host-side, one-time setup code — NOT inside a differentiated
`@kernel` body — so it takes the named `Detector` directly; the kernel bodies below use the isbits
`geometry(detector)` instead (the `Detector`'s `String` prefix trips Enzyme's activity analysis). For each detector `i`, a COPY of `params_ref` with row 6 replaced by `t_c +
delay_i` is passed through the existing, unchanged single-detector `build_relative_binning` —
reusing its already-validated bin-placement/summary-coefficient logic exactly, rather than
duplicating it.
"""
function build_relative_binning(model, params_ref::AbstractVector{T}, data::AbstractVector{<:LikelihoodData};
                                 nbins::Int=150, max_tc_deviation::Real=0,
                                 max_extrinsic_phase::Real=0) where {T}
    @assert length(params_ref) == 11
    ndet = length(data)
    @assert ndet >= 1 "multi-detector build_relative_binning needs at least one detector"
    ra, dec = params_ref[8], params_ref[9]

    gps_time = data[1].gps_time
    setups = Vector{RelativeBinningSetup{T}}(undef, ndet)
    @inbounds for i in 1:ndet
        data[i].gps_time == gps_time || throw(ArgumentError(
            "all detectors must share the same event gps_time (got $(data[i].gps_time) at index $i, expected $gps_time)"))
        delay = time_delay_from_geocenter(data[i].detector, ra, dec, gps_time)
        params_ref_i = copy(params_ref)
        params_ref_i[6] += T(delay)
        setups[i] = build_relative_binning(model, params_ref_i, data[i]; nbins=nbins,
                                            max_tc_deviation=max_tc_deviation, max_extrinsic_phase=max_extrinsic_phase)
    end
    return setups
end

"""
    build_relative_binning(model::IMRPhenomXP, params_ref, data::LikelihoodData; nbins=150)

Single-detector `IMRPhenomXP` specialization, dispatched on
`model::IMRPhenomXP` rather than on `length(params_ref)` (the aligned-spin methods above stay generic
in `model` since `IMRPhenomD`/`IMRPhenomXAS` share one code path; `IMRPhenomXP` genuinely needs a different
one, so a dedicated method reads more cleanly than an `if model isa IMRPhenomXP` branch threaded
through the aligned-spin body). `params_ref` is a 15-element vector `[m1, m2, chi1, chi2, dL, t_c,
phase, ra, dec, theta_jn, psi, chi1x, chi1y, chi2x, chi2y]`, the PHYSICAL row convention documented
at the top of `KernelFused.jl` (rows 12-15 extending the 11-row aligned-spin convention with the
in-plane spin components) — NOT the sampler's reparametrized layout, same caveat as the aligned-spin
method's docstring above.

The reference waveform `href_dense` is built by replicating, per dense-grid frequency,
EXACTLY the same per-frequency chain `KernelFused._fused_waveform_likelihood_kernel_phenomxp!`
evaluates inside its kernel body (`_xp_h_det_scalar` above, factored out so both this host-side
setup code and the sparse per-bin-edge kernels below share one implementation): `_xp_precession_setup`
computed once (`f_ref = freqs[1]`, `inclination = theta_jn`, `phiRef = phase`, same deliberate
scope simplifications as the dense kernel, see that kernel's own comment block), then
IMRPhenomXAS's non-allocating scalar core (`_xas_phi_coeffs`/`_xas_ampl_coeffs`, WITH `t0`/`phiRef`
folding in the `t_c`/`phase` shift) for the co-precessing-frame aligned-(2,2) mode, twisted up via
`_xp_twist_up22`, zeta-rotated via `_xp_apply_zeta_rotation`, and projected through the antenna
response. Antenna response uses the REAL `GravitationalWaveData.compute_antenna_response(data.detector, ra, dec,
psi, data.gps_time)` directly since this is host-side, one-time, non-differentiated setup code -- same reasoning as the aligned-spin method's
own `_extrinsic_setup`/`GravitationalWaveData.gmst` calls above.

Bin-edge PLACEMENT reuses the same heuristic as the aligned-spin method: `Phi(IMRPhenomXAS(), freqs,
mc, eta, chi1, chi2)` (the CO-PRECESSING-FRAME aligned phase alone, no `t0`/`phiRef`/twisting-up/
antenna terms) -- see the aligned-spin `build_relative_binning`'s docstring above for why this is a
deliberate heuristic (finer bins where the underlying waveform's intrinsic shape varies fastest),
not an exact accounting of the full precessing phase; the per-bin `A0`/`A1`/`B0`/`B1` coefficients
and `href_edges` still use the FULL twisted-up/projected `href_dense`, so approximation accuracy is
unaffected by this choice.
"""
function build_relative_binning(model::IMRPhenomXP, params_ref::AbstractVector{T}, data::LikelihoodData;
                                 nbins::Int=150, max_tc_deviation::Real=0,
                                 max_extrinsic_phase::Real=0) where {T}
    @assert length(params_ref) == 15
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi, chi1x, chi1y, chi2x, chi2y = params_ref

    freqs = data.freqs
    strain = data.strain
    psd = data.psd
    n = length(freqs)
    @assert length(strain) == n && length(psd) == n

    mc, eta = _chirp_mass_eta(m1, m2)
    M_sec = (mc / eta^T(0.6)) * T(GMsun_over_c3)
    t0 = -2 * T(pi) * t_c / M_sec
    phiRef = phase

    Fplus, Fcross = compute_antenna_response(data.detector, ra, dec, psi, data.gps_time)

    f_ref = freqs[1]
    setup = _xp_precession_setup(m1, m2, chi1x, chi1y, chi1, chi2x, chi2y, chi2,
                                  f_ref, theta_jn, phase; GMsun_over_c3=T(GMsun_over_c3))

    phi_coeffs = _xas_phi_coeffs(IMRPhenomXAS(), mc, eta, chi1, chi2; t0=-t0, phiRef=phiRef)
    ampl_coeffs = _xas_ampl_coeffs(IMRPhenomXAS(), mc, eta, chi1, chi2, dL)

    href_dense = Vector{Complex{T}}(undef, n)
    @inbounds for i in 1:n
        href_dense[i] = _xp_h_det_scalar(freqs[i], M_sec, ampl_coeffs, phi_coeffs, setup, Fplus, Fcross)
    end

    phase_ref_bare = Phi(IMRPhenomXAS(), freqs, mc, eta, chi1, chi2)  # bin-PLACEMENT heuristic only, see docstring above

    edges_idx = _phase_equal_bin_edges(phase_ref_bare, nbins)
    edges_idx = _cap_bin_width_for_extrinsic_ramp(edges_idx, freqs, M_sec, T(max_tc_deviation), T(max_extrinsic_phase))
    nb = length(edges_idx) - 1
    edges = Vector{T}(undef, nb + 1)
    @inbounds for i in eachindex(edges_idx)
        edges[i] = freqs[edges_idx[i]]
    end
    href_edges = href_dense[edges_idx]

    df = freqs[2] - freqs[1]
    fm = Vector{T}(undef, nb)
    A0 = Vector{Complex{T}}(undef, nb)
    A1 = Vector{Complex{T}}(undef, nb)
    B0 = Vector{T}(undef, nb)
    B1 = Vector{T}(undef, nb)

    @inbounds for b in 1:nb
        lo = edges_idx[b]
        hi = b == nb ? edges_idx[b+1] : edges_idx[b+1] - 1   # half-open [lo,hi) except last bin
        fmb = T(0.5) * (edges[b] + edges[b+1])
        fm[b] = fmb

        a0 = zero(Complex{T})
        a1 = zero(Complex{T})
        b0 = zero(T)
        b1 = zero(T)
        for k in lo:hi
            w = _safe_div(T(4) * df, psd[k])
            dhc = strain[k] * conj(href_dense[k])
            hh = abs2(href_dense[k])
            dfk = freqs[k] - fmb
            a0 += w * dhc
            a1 += w * dhc * dfk
            b0 += w * hh
            b1 += w * hh * dfk
        end
        A0[b] = a0
        A1[b] = a1
        B0[b] = b0
        B1[b] = b1
    end

    dd_term = zero(T)
    @inbounds for k in 1:n
        dd_term -= 2 * df * _safe_div(abs2(strain[k]), psd[k])
    end

    return RelativeBinningSetup(to_device(edges), to_device(fm), to_device(href_edges),
                                 to_device(A0), to_device(A1), to_device(B0), to_device(B1), dd_term)
end

"""
    build_relative_binning(model::IMRPhenomXP, params_ref, data::AbstractVector{<:LikelihoodData}; nbins=150)
        -> Vector{RelativeBinningSetup}

Multi-detector `IMRPhenomXP` entry point: mirrors the aligned-spin multi-detector method above EXACTLY
(same per-detector `t_c + GravitationalWaveData.time_delay_from_geocenter(...)` geocenter-to-detector shift, same
"reuse the single-detector method once per detector, not a reimplementation" structure) — the only
difference is `params_ref` is 15-element and dispatches to the `IMRPhenomXP`-specialized single-detector
method above instead of the aligned-spin one.

**One documented simplification, inherited from reusing the single-detector method unchanged**: each
detector's own `_xp_precession_setup` is anchored to ITS OWN `f_ref = data[i].freqs[1]` (since the
single-detector method always uses its own `freqs[1]`), whereas the multi-detector DENSE kernel
(`KernelFused._fused_waveform_likelihood_kernel_phenomxp_multi!`) anchors `setup` to the FIRST
detector's `f_ref` only and reuses that SAME `setup` for every detector (`setup` is hoisted outside
the `1:ndet` loop there, since it can only be built once). When every detector shares the same
frequency grid (the common case — e.g. this module's own tests, and typical real-data multi-detector
analyses fetching one shared frequency array) `f_ref` coincides across detectors and this is a
non-issue; if detectors ever have genuinely different starting frequencies, this reference-waveform
construction would use a per-detector `f_ref` while the dense kernel it's approximating uses a
single shared one, a small additional source of at-`theta_ref` mismatch beyond relative binning's
usual linear-interpolation error. Not fixed here (would require threading a shared `f_ref` through
a reimplementation rather than reusing the single-detector method) — flagged rather than silently accepted.
"""
function build_relative_binning(model::IMRPhenomXP, params_ref::AbstractVector{T}, data::AbstractVector{<:LikelihoodData};
                                 nbins::Int=150, max_tc_deviation::Real=0,
                                 max_extrinsic_phase::Real=0) where {T}
    @assert length(params_ref) == 15
    ndet = length(data)
    @assert ndet >= 1 "multi-detector build_relative_binning needs at least one detector"
    ra, dec = params_ref[8], params_ref[9]

    gps_time = data[1].gps_time
    setups = Vector{RelativeBinningSetup{T}}(undef, ndet)
    @inbounds for i in 1:ndet
        data[i].gps_time == gps_time || throw(ArgumentError(
            "all detectors must share the same event gps_time (got $(data[i].gps_time) at index $i, expected $gps_time)"))
        delay = time_delay_from_geocenter(data[i].detector, ra, dec, gps_time)
        params_ref_i = copy(params_ref)
        params_ref_i[6] += T(delay)
        setups[i] = build_relative_binning(model, params_ref_i, data[i]; nbins=nbins,
                                            max_tc_deviation=max_tc_deviation, max_extrinsic_phase=max_extrinsic_phase)
    end
    return setups
end

"""
    build_relative_binning(model::IMRPhenomXHM, params_ref, data::LikelihoodData; nbins=150)

Single-detector `IMRPhenomXHM` specialization, dispatched on
`model::IMRPhenomXHM` for the same reason `IMRPhenomXP` gets its own dedicated method above rather
than threading an `if` through the generic aligned-spin body: `IMRPhenomXHM` doesn't share the
`Ampl`/`Phi` scalar contract at all (see `KernelFused.jl`'s module docstring / IMRPhenomXHM kernel-
body comment block) -- it's a genuine sum over 5 independent complex mode functions returning
fully-formed `(hp,hc)`, not a single `(Ampl,Phi)` pair. `params_ref` is the EXISTING 11-element
aligned-spin vector `[m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi]` (NOT `IMRPhenomXP`'s
15-row extension -- `IMRPhenomXHM` has no in-plane spin/precession at all, same row convention as
the generic aligned-spin method above).

The reference waveform `href_dense` is built by replicating, per dense-grid frequency, EXACTLY the
same per-frequency chain `KernelFused._fused_waveform_likelihood_kernel_phenomxhm!` evaluates
inside its kernel body (`_xhm_h_det_scalar` above, factored out so both this host-side setup code
and the sparse per-bin-edge kernels below share one implementation): `_xhm_system_setup`/
`_xhm_mode_setup`(for `(2,1),(3,3),(4,4)`)/`_xhm_mode32_setup`(for the mode-mixing `(3,2)`) computed
ONCE (`f_ref = freqs[1]`, `phiRef = phase`, same deliberate scope simplifications as the dense
kernel -- see that kernel's own comment block), then `_xhm_hphc_scalar` at each dense-grid
frequency (`inclination = theta_jn` directly, `phi` FIXED at `pi/2`), the `t_c` post-multiplication
convention, and antenna-response projection. Antenna response uses the REAL
`GravitationalWaveData.compute_antenna_response(data.detector, ra, dec, psi, data.gps_time)` directly since this is
host-side, one-time, non-differentiated
setup code -- same reasoning as `IMRPhenomXP`'s own method above.

Bin-edge PLACEMENT reuses the SAME heuristic as the generic aligned-spin and `IMRPhenomXP` methods:
`Phi(IMRPhenomXAS(), freqs, mc, eta, chi1, chi2)` (the plain dominant-mode phase alone, no
higher-order-mode/`t0`/`phiRef`/antenna terms) -- an explicit, deliberate simplification already
established for exactly this kind of model (see `IMRPhenomXP`'s own docstring above for the full
reasoning: bin placement just needs "a reasonable proxy for where the phase changes fast," not
exact physics), NOT improved upon here. The per-bin `A0`/`A1`/`B0`/`B1` coefficients and
`href_edges` still use the FULL higher-order-mode `href_dense`, so approximation accuracy is
unaffected by this choice -- only which frequencies get finer bins.
"""
function build_relative_binning(model::IMRPhenomXHM, params_ref::AbstractVector{T}, data::LikelihoodData;
                                 nbins::Int=150, max_tc_deviation::Real=0,
                                 max_extrinsic_phase::Real=0) where {T}
    @assert length(params_ref) == 11
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = params_ref

    freqs = data.freqs
    strain = data.strain
    psd = data.psd
    n = length(freqs)
    @assert length(strain) == n && length(psd) == n

    Fplus, Fcross = compute_antenna_response(data.detector, ra, dec, psi, data.gps_time)

    mc, eta = _chirp_mass_eta(m1, m2)
    M_sec = (mc / eta^T(0.6)) * T(GMsun_over_c3)
    f_ref = freqs[1]
    sys = _xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, phase;
                             GMsun_over_c3=T(GMsun_over_c3), GMsun_over_c2_Gpc=T(GMsun_over_c2_Gpc))
    ms21 = _xhm_mode_setup(2, 1, sys)
    ms33 = _xhm_mode_setup(3, 3, sys)
    ms44 = _xhm_mode_setup(4, 4, sys)
    m32 = _xhm_mode32_setup(sys)
    piHalf = T(pi) / T(2)

    href_dense = Vector{Complex{T}}(undef, n)
    @inbounds for i in 1:n
        href_dense[i] = _xhm_h_det_scalar(freqs[i], sys, ms21, ms33, ms44, m32, theta_jn, piHalf, t_c, Fplus, Fcross)
    end

    phase_ref_bare = Phi(IMRPhenomXAS(), freqs, mc, eta, chi1, chi2)  # bin-PLACEMENT heuristic only, see docstring above

    edges_idx = _phase_equal_bin_edges(phase_ref_bare, nbins)
    edges_idx = _cap_bin_width_for_extrinsic_ramp(edges_idx, freqs, M_sec, T(max_tc_deviation), T(max_extrinsic_phase))
    nb = length(edges_idx) - 1
    edges = Vector{T}(undef, nb + 1)
    @inbounds for i in eachindex(edges_idx)
        edges[i] = freqs[edges_idx[i]]
    end
    href_edges = href_dense[edges_idx]

    df = freqs[2] - freqs[1]
    fm = Vector{T}(undef, nb)
    A0 = Vector{Complex{T}}(undef, nb)
    A1 = Vector{Complex{T}}(undef, nb)
    B0 = Vector{T}(undef, nb)
    B1 = Vector{T}(undef, nb)

    @inbounds for b in 1:nb
        lo = edges_idx[b]
        hi = b == nb ? edges_idx[b+1] : edges_idx[b+1] - 1   # half-open [lo,hi) except last bin
        fmb = T(0.5) * (edges[b] + edges[b+1])
        fm[b] = fmb

        a0 = zero(Complex{T})
        a1 = zero(Complex{T})
        b0 = zero(T)
        b1 = zero(T)
        for k in lo:hi
            w = _safe_div(T(4) * df, psd[k])
            dhc = strain[k] * conj(href_dense[k])
            hh = abs2(href_dense[k])
            dfk = freqs[k] - fmb
            a0 += w * dhc
            a1 += w * dhc * dfk
            b0 += w * hh
            b1 += w * hh * dfk
        end
        A0[b] = a0
        A1[b] = a1
        B0[b] = b0
        B1[b] = b1
    end

    dd_term = zero(T)
    @inbounds for k in 1:n
        dd_term -= 2 * df * _safe_div(abs2(strain[k]), psd[k])
    end

    return RelativeBinningSetup(to_device(edges), to_device(fm), to_device(href_edges),
                                 to_device(A0), to_device(A1), to_device(B0), to_device(B1), dd_term)
end

"""
    build_relative_binning(model::IMRPhenomXHM, params_ref, data::AbstractVector{<:LikelihoodData}; nbins=150)
        -> Vector{RelativeBinningSetup}

Multi-detector `IMRPhenomXHM` entry point: mirrors the aligned-spin/`IMRPhenomXP` multi-detector
methods above EXACTLY (same per-detector `t_c + GravitationalWaveData.time_delay_from_geocenter(...)`
geocenter-to-detector shift, same "reuse the single-detector method once per detector, not a
reimplementation" structure, same shared-`gps_time` assertion) -- this is the SAME physical event,
one geocenter `t_c`, so each detector only needs its own arrival-time shift and independent
`nbins`-bin summary; nothing about `IMRPhenomXHM`'s higher-order-mode content changes that
reasoning. The only difference from the generic aligned-spin multi-detector method is dispatching
to the `IMRPhenomXHM`-specialized single-detector method above.

Note this is `RelativeBinning.jl`'s OWN self-contained multi-detector support (independent
`RelativeBinningSetup`s built and combined by `binned_log_likelihood_kernel!`'s multi-detector
method below) -- it does NOT require or touch `KernelFused.jl`'s multi-detector method, which still
explicitly rejects `IMRPhenomXHM` (multi-detector `IMRPhenomXHM` support for the DENSE kernel is a
separate, out-of-scope follow-on, see `KernelFused.jl`'s module docstring).
"""
function build_relative_binning(model::IMRPhenomXHM, params_ref::AbstractVector{T}, data::AbstractVector{<:LikelihoodData};
                                 nbins::Int=150, max_tc_deviation::Real=0,
                                 max_extrinsic_phase::Real=0) where {T}
    @assert length(params_ref) == 11
    ndet = length(data)
    @assert ndet >= 1 "multi-detector build_relative_binning needs at least one detector"
    ra, dec = params_ref[8], params_ref[9]

    gps_time = data[1].gps_time
    setups = Vector{RelativeBinningSetup{T}}(undef, ndet)
    @inbounds for i in 1:ndet
        data[i].gps_time == gps_time || throw(ArgumentError(
            "all detectors must share the same event gps_time (got $(data[i].gps_time) at index $i, expected $gps_time)"))
        delay = time_delay_from_geocenter(data[i].detector, ra, dec, gps_time)
        params_ref_i = copy(params_ref)
        params_ref_i[6] += T(delay)
        setups[i] = build_relative_binning(model, params_ref_i, data[i]; nbins=nbins,
                                            max_tc_deviation=max_tc_deviation, max_extrinsic_phase=max_extrinsic_phase)
    end
    return setups
end

"""
    build_relative_binning(model::IMRPhenomXPHM, params_ref, data::LikelihoodData; nbins=150)

Single-detector `IMRPhenomXPHM` specialization, combining `IMRPhenomXP`'s own
method above (precession setup, `params_ref` is `IMRPhenomXP`'s 15-element vector) with
`IMRPhenomXHM`'s (higher-order-mode `_xhm_system_setup`/per-mode setups, `_xhm_h_det_scalar`-style
`t_c` post-multiplication) -- exactly the same two-way combination
`KernelFused._fused_waveform_likelihood_kernel_phenomxphm!` makes of its own two ancestors, and the
same one `_xphm_h_det_scalar` above makes at the single-frequency level.

**Uses `final_spin_override`** (`_xas_precessing_final_spin(eta, chi1, chi2,
setup.chi_p)` fed to `_xhm_system_setup`), matching `hphc(::IMRPhenomXPHM, ...)`'s and the dense
kernel's own (fixed) behavior — omitting it here would silently build a reference waveform (and
hence every relative-binning summary statistic derived from it) using the WRONG, pre-Task-5e
physics, even though the dense dispatch used for injection/other comparisons would be correct.

Bin-edge PLACEMENT reuses the SAME heuristic as every other method in this file:
`Phi(IMRPhenomXAS(), freqs, mc, eta, chi1, chi2)` (the plain dominant-mode phase alone) — the
per-bin `A0`/`A1`/`B0`/`B1` coefficients and `href_edges` still use the FULL precessing+
higher-order-mode `href_dense`, so approximation accuracy is unaffected by this choice.
"""
function build_relative_binning(model::IMRPhenomXPHM, params_ref::AbstractVector{T}, data::LikelihoodData;
                                 nbins::Int=150, max_tc_deviation::Real=0,
                                 max_extrinsic_phase::Real=0) where {T}
    @assert length(params_ref) == 15
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi, chi1x, chi1y, chi2x, chi2y = params_ref

    freqs = data.freqs
    strain = data.strain
    psd = data.psd
    n = length(freqs)
    @assert length(strain) == n && length(psd) == n

    mc, eta = _chirp_mass_eta(m1, m2)
    M_sec = (mc / eta^T(0.6)) * T(GMsun_over_c3)

    Fplus, Fcross = compute_antenna_response(data.detector, ra, dec, psi, data.gps_time)

    f_ref = freqs[1]
    setup = _xphm_prec_setup(model, m1, m2, chi1x, chi1y, chi1, chi2x, chi2y, chi2,
                                  f_ref, theta_jn, phase; GMsun_over_c3=T(GMsun_over_c3))

    final_spin = setup.final_spin   # version-dependent (102: chi_p, 223: MSA)
    sys = _xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, zero(phase);
                             GMsun_over_c3=T(GMsun_over_c3), GMsun_over_c2_Gpc=T(GMsun_over_c2_Gpc),
                             final_spin_override=final_spin)
    ms21 = _xhm_mode_setup(2, 1, sys)
    ms33 = _xhm_mode_setup(3, 3, sys)
    ms44 = _xhm_mode_setup(4, 4, sys)
    m32 = _xhm_mode32_setup(sys)

    href_dense = Vector{Complex{T}}(undef, n)
    @inbounds for i in 1:n
        href_dense[i] = _xphm_h_det_scalar(freqs[i], setup, sys, ms21, ms33, ms44, m32, t_c, Fplus, Fcross)
    end

    phase_ref_bare = Phi(IMRPhenomXAS(), freqs, mc, eta, chi1, chi2)  # bin-PLACEMENT heuristic only, see docstring above

    edges_idx = _phase_equal_bin_edges(phase_ref_bare, nbins)
    edges_idx = _cap_bin_width_for_extrinsic_ramp(edges_idx, freqs, M_sec, T(max_tc_deviation), T(max_extrinsic_phase))
    nb = length(edges_idx) - 1
    edges = Vector{T}(undef, nb + 1)
    @inbounds for i in eachindex(edges_idx)
        edges[i] = freqs[edges_idx[i]]
    end
    href_edges = href_dense[edges_idx]

    df = freqs[2] - freqs[1]
    fm = Vector{T}(undef, nb)
    A0 = Vector{Complex{T}}(undef, nb)
    A1 = Vector{Complex{T}}(undef, nb)
    B0 = Vector{T}(undef, nb)
    B1 = Vector{T}(undef, nb)

    @inbounds for b in 1:nb
        lo = edges_idx[b]
        hi = b == nb ? edges_idx[b+1] : edges_idx[b+1] - 1   # half-open [lo,hi) except last bin
        fmb = T(0.5) * (edges[b] + edges[b+1])
        fm[b] = fmb

        a0 = zero(Complex{T})
        a1 = zero(Complex{T})
        b0 = zero(T)
        b1 = zero(T)
        for k in lo:hi
            w = _safe_div(T(4) * df, psd[k])
            dhc = strain[k] * conj(href_dense[k])
            hh = abs2(href_dense[k])
            dfk = freqs[k] - fmb
            a0 += w * dhc
            a1 += w * dhc * dfk
            b0 += w * hh
            b1 += w * hh * dfk
        end
        A0[b] = a0
        A1[b] = a1
        B0[b] = b0
        B1[b] = b1
    end

    dd_term = zero(T)
    @inbounds for k in 1:n
        dd_term -= 2 * df * _safe_div(abs2(strain[k]), psd[k])
    end

    return RelativeBinningSetup(to_device(edges), to_device(fm), to_device(href_edges),
                                 to_device(A0), to_device(A1), to_device(B0), to_device(B1), dd_term)
end

"""
    build_relative_binning(model::IMRPhenomXPHM, params_ref, data::AbstractVector{<:LikelihoodData}; nbins=150)
        -> Vector{RelativeBinningSetup}

Multi-detector `IMRPhenomXPHM` entry point: mirrors `IMRPhenomXP`'s/`IMRPhenomXHM`'s own
multi-detector methods above EXACTLY (same per-detector `t_c + GravitationalWaveData.time_delay_from_geocenter(...)`
geocenter-to-detector shift, same "reuse the single-detector method once per detector" structure,
same shared-`gps_time` assertion, and the SAME single-detector-method `f_ref`-anchoring
simplification `IMRPhenomXP`'s own multi-detector docstring documents and flags).
"""
function build_relative_binning(model::IMRPhenomXPHM, params_ref::AbstractVector{T}, data::AbstractVector{<:LikelihoodData};
                                 nbins::Int=150, max_tc_deviation::Real=0,
                                 max_extrinsic_phase::Real=0) where {T}
    @assert length(params_ref) == 15
    ndet = length(data)
    @assert ndet >= 1 "multi-detector build_relative_binning needs at least one detector"
    ra, dec = params_ref[8], params_ref[9]

    gps_time = data[1].gps_time
    setups = Vector{RelativeBinningSetup{T}}(undef, ndet)
    @inbounds for i in 1:ndet
        data[i].gps_time == gps_time || throw(ArgumentError(
            "all detectors must share the same event gps_time (got $(data[i].gps_time) at index $i, expected $gps_time)"))
        delay = time_delay_from_geocenter(data[i].detector, ra, dec, gps_time)
        params_ref_i = copy(params_ref)
        params_ref_i[6] += T(delay)
        setups[i] = build_relative_binning(model, params_ref_i, data[i]; nbins=nbins,
                                            max_tc_deviation=max_tc_deviation, max_extrinsic_phase=max_extrinsic_phase)
    end
    return setups
end

# ---------------------------------------------------------------------------------------------
# Sparse, per-bin-edge evaluation kernels. Mirror KernelFused.jl's IMRPhenomD/IMRPhenomXAS-specialized
# kernel bodies exactly: `_phi_coeffs`/`_ampl_coeffs` (or the `_xas_` equivalents) computed once
# per walker, then a non-allocating scalar-core loop -- here over the sparse `edges` vector
# (~100-200 points) instead of the full dense frequency grid (thousands of points). Zero array
# allocation inside the kernel body, for the same Enzyme-through-KernelAbstractions reason
# documented in KernelFused.jl: the array-in/array-out Ampl/Phi hang Enzyme.Reverse compilation indefinitely, the
# scalar core does not.
# ---------------------------------------------------------------------------------------------

@kernel function _relative_binning_kernel_phenomd!(ll, @Const(params), @Const(edges), @Const(href_edges),
                                                     @Const(A0), @Const(A1), @Const(B0), @Const(B1),
                                                     dd_term, det, gmst_val, model)
    j = @index(Global, Linear)
    T = eltype(params)

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]

    mc, eta, t0, phiRef, combo = _extrinsic_setup(m1, m2, dL, t_c, phase, ra, dec, theta_jn, psi, det, gmst_val)

    phi_coeffs = _phi_coeffs(model, mc, eta, chi1, chi2; t0=t0, phiRef=phiRef)
    ampl_coeffs = _ampl_coeffs(model, mc, eta, chi1, chi2, dL)

    nb = length(A0)
    amp_lo = _ampl_scalar(ampl_coeffs, edges[1])
    phase_lo = -_phi_scalar(phi_coeffs, edges[1])
    r_lo = _safe_ratio(combo * (amp_lo * Complex(cos(phase_lo), sin(phase_lo))), href_edges[1])

    dh_sum = zero(Complex{T})
    hh_sum = zero(T)
    @inbounds for b in 1:nb
        amp_hi = _ampl_scalar(ampl_coeffs, edges[b+1])
        phase_hi = -_phi_scalar(phi_coeffs, edges[b+1])
        r_hi = _safe_ratio(combo * (amp_hi * Complex(cos(phase_hi), sin(phase_hi))), href_edges[b+1])

        r0 = T(0.5) * (r_lo + r_hi)
        r1 = (r_hi - r_lo) / (edges[b+1] - edges[b])

        dh_sum += A0[b] * conj(r0) + A1[b] * conj(r1)
        hh_sum += B0[b] * abs2(r0) + 2 * B1[b] * real(r0 * conj(r1))

        r_lo = r_hi
    end

    ll[j] = dd_term + real(dh_sum) - T(0.5) * hh_sum
end

@kernel function _relative_binning_kernel_phenomxas!(ll, @Const(params), @Const(edges), @Const(href_edges),
                                                       @Const(A0), @Const(A1), @Const(B0), @Const(B1),
                                                       dd_term, det, gmst_val, model)
    j = @index(Global, Linear)
    T = eltype(params)

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]

    mc, eta, t0, phiRef, combo = _extrinsic_setup(m1, m2, dL, t_c, phase, ra, dec, theta_jn, psi, det, gmst_val)

    phi_coeffs = _xas_phi_coeffs(model, mc, eta, chi1, chi2; t0=-t0, phiRef=phiRef)
    ampl_coeffs = _xas_ampl_coeffs(model, mc, eta, chi1, chi2, dL)

    nb = length(A0)
    amp_lo = _xas_ampl_scalar(ampl_coeffs, edges[1])
    phase_lo = _xas_phi_scalar(phi_coeffs, edges[1])
    r_lo = _safe_ratio(combo * (amp_lo * Complex(cos(phase_lo), sin(phase_lo))), href_edges[1])

    dh_sum = zero(Complex{T})
    hh_sum = zero(T)
    @inbounds for b in 1:nb
        amp_hi = _xas_ampl_scalar(ampl_coeffs, edges[b+1])
        phase_hi = _xas_phi_scalar(phi_coeffs, edges[b+1])
        r_hi = _safe_ratio(combo * (amp_hi * Complex(cos(phase_hi), sin(phase_hi))), href_edges[b+1])

        r0 = T(0.5) * (r_lo + r_hi)
        r1 = (r_hi - r_lo) / (edges[b+1] - edges[b])

        dh_sum += A0[b] * conj(r0) + A1[b] * conj(r1)
        hh_sum += B0[b] * abs2(r0) + 2 * B1[b] * real(r0 * conj(r1))

        r_lo = r_hi
    end

    ll[j] = dd_term + real(dh_sum) - T(0.5) * hh_sum
end

# AlignedSpinPNHamF2 single-detector relative-binning kernel:
# mirrors `_relative_binning_kernel_phenomxas!` exactly, using the `_asphf2_*` scalar core instead
# of the `_xas_*` one -- same 11-row aligned-spin `Phi`/`Ampl(model, f, mc, eta, chi1, chi2[, dL])`
# contract, so no new row convention or bin-placement heuristic is needed (`build_relative_binning`
# above already works unchanged for this model, since it dispatches generically on `model` via the
# array-oriented `Ampl`/`Phi` calls).
@kernel function _relative_binning_kernel_alignedspinpnhamf2!(ll, @Const(params), @Const(edges), @Const(href_edges),
                                                                 @Const(A0), @Const(A1), @Const(B0), @Const(B1),
                                                                 dd_term, det, gmst_val, model)
    j = @index(Global, Linear)
    T = eltype(params)

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]

    mc, eta, t0, phiRef, combo = _extrinsic_setup(m1, m2, dL, t_c, phase, ra, dec, theta_jn, psi, det, gmst_val)

    phi_coeffs = _asphf2_phi_coeffs(model, mc, eta, chi1, chi2; t0=t0, phiRef=phiRef)
    ampl_coeffs = _asphf2_ampl_coeffs(model, mc, eta, chi1, chi2, dL)

    nb = length(A0)
    amp_lo = _asphf2_ampl_scalar(ampl_coeffs, edges[1])
    phase_lo = -_asphf2_phi_scalar(phi_coeffs, edges[1])
    r_lo = _safe_ratio(combo * (amp_lo * Complex(cos(phase_lo), sin(phase_lo))), href_edges[1])

    dh_sum = zero(Complex{T})
    hh_sum = zero(T)
    @inbounds for b in 1:nb
        amp_hi = _asphf2_ampl_scalar(ampl_coeffs, edges[b+1])
        phase_hi = -_asphf2_phi_scalar(phi_coeffs, edges[b+1])
        r_hi = _safe_ratio(combo * (amp_hi * Complex(cos(phase_hi), sin(phase_hi))), href_edges[b+1])

        r0 = T(0.5) * (r_lo + r_hi)
        r1 = (r_hi - r_lo) / (edges[b+1] - edges[b])

        dh_sum += A0[b] * conj(r0) + A1[b] * conj(r1)
        hh_sum += B0[b] * abs2(r0) + 2 * B1[b] * real(r0 * conj(r1))

        r_lo = r_hi
    end

    ll[j] = dd_term + real(dh_sum) - T(0.5) * hh_sum
end

# SEOBNRv5AlignedSpinF2 single-detector relative-binning kernel: mirrors
# `_relative_binning_kernel_alignedspinpnhamf2!` structurally, but `coeffs` is `SEOBNRv5AlignedSpinF2`'s
# own dense post-adiabatic table (`_seobv5f2_coeffs`), built ONCE per walker since it depends only on
# `mc`/`eta`/`chi1`/`chi2` -- not on `t0`/`phiRef`/`dL`, applied per-edge in the scalar calls exactly
# as `KernelFused.jl`'s sibling kernel does. Deliberately NOT passing an explicit `f_min` (an earlier
# draft tried `edges[1]`, mirroring `IMRPhenomXP`'s own `f_ref = edges[1]`) -- see
# `KernelFused.jl`'s sibling single-detector kernel's own comment for why: unlike `IMRPhenomXP`'s
# `f_ref`, this model's `f_min` sets its `phi=0`/`t=0` phase anchor itself, and must match whatever
# `build_relative_binning`'s generic reference-waveform construction implicitly used (this model's
# own default) or the two computations' phase origins silently misalign -- confirmed empirically via
# a failing exact-at-theta_ref agreement test before this fix. `build_relative_binning` needs no
# changes for this model otherwise -- it already dispatches generically via the array-oriented
# `Ampl`/`Phi` calls this model implements.
#
# With GravitationalWaveforms' fixed-buffer (SVector) table and the array-type-generic
# `RelativeBinningSetup`, this kernel compiles and runs on CUDA (GTX 1650 Ti), matching the CPU to
# `rtol=1e-8`.
@kernel function _relative_binning_kernel_seobnrv5alignedspinf2!(ll, @Const(params), @Const(edges), @Const(href_edges),
                                                                     @Const(A0), @Const(A1), @Const(B0), @Const(B1),
                                                                     dd_term, det, gmst_val, model)
    j = @index(Global, Linear)
    T = eltype(params)

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]

    mc, eta, t0, phiRef, combo = _extrinsic_setup(m1, m2, dL, t_c, phase, ra, dec, theta_jn, psi, det, gmst_val)
    tshift = -2 * T(pi) * t_c      # seconds -> rad/Hz; NOT the Mf-based `t0` (see `KernelFused._phi_t0`)

    coeffs = _seobv5f2_coeffs(model, mc, eta, chi1, chi2)

    nb = length(A0)
    amp_lo = _seobv5f2_ampl_scalar(coeffs, edges[1], dL, GMsun_over_c2_Gpc)
    phase_lo = -(_seobv5f2_phi_scalar(coeffs, edges[1]) - tshift * edges[1] - phiRef)
    r_lo = _safe_ratio(combo * (amp_lo * Complex(cos(phase_lo), sin(phase_lo))), href_edges[1])

    dh_sum = zero(Complex{T})
    hh_sum = zero(T)
    @inbounds for b in 1:nb
        amp_hi = _seobv5f2_ampl_scalar(coeffs, edges[b+1], dL, GMsun_over_c2_Gpc)
        phase_hi = -(_seobv5f2_phi_scalar(coeffs, edges[b+1]) - tshift * edges[b+1] - phiRef)
        r_hi = _safe_ratio(combo * (amp_hi * Complex(cos(phase_hi), sin(phase_hi))), href_edges[b+1])

        r0 = T(0.5) * (r_lo + r_hi)
        r1 = (r_hi - r_lo) / (edges[b+1] - edges[b])

        dh_sum += A0[b] * conj(r0) + A1[b] * conj(r1)
        hh_sum += B0[b] * abs2(r0) + 2 * B1[b] * real(r0 * conj(r1))

        r_lo = r_hi
    end

    ll[j] = _seobv5f2_apply_deadzone_reject(dd_term + real(dh_sum) - T(0.5) * hh_sum, coeffs, edges[1])
end

# IMRPhenomXP single-detector relative-binning kernel: mirrors
# `KernelFused._fused_waveform_likelihood_kernel_phenomxp!` EXACTLY (same `_xp_precession_setup`
# hoisted once per walker outside the frequency loop, `f_ref = <first grid point>` -- here `edges[1]`
# rather than `freqs[1]`, but they're equal: `edges[1] == freqs[edges_idx[1]] == freqs[1]` always,
# see `_phase_equal_bin_edges`'s docstring), except the per-frequency twisting-up chain
# (`_xp_h_det_scalar` above) is evaluated only at the sparse `edges` (bin edges), not the dense
# `freqs` grid, and combined into the `r0`/`r1` linear-ratio-interpolation summary-statistic
# reduction exactly like `_relative_binning_kernel_phenomxas!` above (just with the twisting-up
# chain standing in for `combo*(amp*exp(i*phase))`).
@kernel function _relative_binning_kernel_phenomxp!(ll, @Const(params), @Const(edges), @Const(href_edges),
                                                      @Const(A0), @Const(A1), @Const(B0), @Const(B1),
                                                      dd_term, det, gmst_val, model)
    j = @index(Global, Linear)
    T = eltype(params)

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]
    chi1x = params[12, j]; chi1y = params[13, j]; chi2x = params[14, j]; chi2y = params[15, j]

    mc, eta = _chirp_mass_eta(m1, m2)
    M_sec = (mc / eta^T(0.6)) * T(GMsun_over_c3)
    t0 = -2 * T(pi) * t_c / M_sec
    phiRef = phase

    Fplus, Fcross = antenna_response_gmst(det, ra, dec, psi, gmst_val)

    f_ref = edges[1]
    setup = _xp_precession_setup(m1, m2, chi1x, chi1y, chi1, chi2x, chi2y, chi2,
                                  f_ref, theta_jn, phase; GMsun_over_c3=T(GMsun_over_c3))

    phi_coeffs = _xas_phi_coeffs(IMRPhenomXAS(), mc, eta, chi1, chi2; t0=-t0, phiRef=phiRef)
    ampl_coeffs = _xas_ampl_coeffs(IMRPhenomXAS(), mc, eta, chi1, chi2, dL)

    nb = length(A0)
    h_lo = _xp_h_det_scalar(edges[1], M_sec, ampl_coeffs, phi_coeffs, setup, Fplus, Fcross)
    r_lo = _safe_ratio(h_lo, href_edges[1])

    dh_sum = zero(Complex{T})
    hh_sum = zero(T)
    @inbounds for b in 1:nb
        h_hi = _xp_h_det_scalar(edges[b+1], M_sec, ampl_coeffs, phi_coeffs, setup, Fplus, Fcross)
        r_hi = _safe_ratio(h_hi, href_edges[b+1])

        r0 = T(0.5) * (r_lo + r_hi)
        r1 = (r_hi - r_lo) / (edges[b+1] - edges[b])

        dh_sum += A0[b] * conj(r0) + A1[b] * conj(r1)
        hh_sum += B0[b] * abs2(r0) + 2 * B1[b] * real(r0 * conj(r1))

        r_lo = r_hi
    end

    ll[j] = dd_term + real(dh_sum) - T(0.5) * hh_sum
end

# IMRPhenomXHM single-detector relative-binning kernel: mirrors
# `KernelFused._fused_waveform_likelihood_kernel_phenomxhm!` EXACTLY (same `_xhm_system_setup`/
# `_xhm_mode_setup`(x3)/`_xhm_mode32_setup` hoisted once per walker outside the frequency loop,
# `f_ref = <first grid point>` -- here `edges[1]` rather than `freqs[1]`, but they're equal:
# `edges[1] == freqs[edges_idx[1]] == freqs[1]` always, same reasoning as the `IMRPhenomXP` kernel
# above), except the per-frequency `(hp,hc)`/`t_c`-shift/antenna-projection chain
# (`_xhm_h_det_scalar` above) is evaluated only at the sparse `edges` (bin edges), not the dense
# frequency grid, and combined into the SAME `r0`/`r1` linear-ratio-interpolation summary-statistic
# reduction every other kernel body in this file uses (copied verbatim -- that reduction is
# model-independent, only the per-frequency `h(f)` evaluation above it differs).
@kernel function _relative_binning_kernel_phenomxhm!(ll, @Const(params), @Const(edges), @Const(href_edges),
                                                       @Const(A0), @Const(A1), @Const(B0), @Const(B1),
                                                       dd_term, det, gmst_val, model)
    j = @index(Global, Linear)
    T = eltype(params)

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]

    Fplus, Fcross = antenna_response_gmst(det, ra, dec, psi, gmst_val)

    mc, eta = _chirp_mass_eta(m1, m2)
    f_ref = edges[1]
    sys = _xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, phase;
                             GMsun_over_c3=T(GMsun_over_c3), GMsun_over_c2_Gpc=T(GMsun_over_c2_Gpc))
    ms21 = _xhm_mode_setup(2, 1, sys)
    ms33 = _xhm_mode_setup(3, 3, sys)
    ms44 = _xhm_mode_setup(4, 4, sys)
    m32 = _xhm_mode32_setup(sys)
    piHalf = T(pi) / T(2)

    nb = length(A0)
    h_lo = _xhm_h_det_scalar(edges[1], sys, ms21, ms33, ms44, m32, theta_jn, piHalf, t_c, Fplus, Fcross)
    r_lo = _safe_ratio(h_lo, href_edges[1])

    dh_sum = zero(Complex{T})
    hh_sum = zero(T)
    @inbounds for b in 1:nb
        h_hi = _xhm_h_det_scalar(edges[b+1], sys, ms21, ms33, ms44, m32, theta_jn, piHalf, t_c, Fplus, Fcross)
        r_hi = _safe_ratio(h_hi, href_edges[b+1])

        r0 = T(0.5) * (r_lo + r_hi)
        r1 = (r_hi - r_lo) / (edges[b+1] - edges[b])

        dh_sum += A0[b] * conj(r0) + A1[b] * conj(r1)
        hh_sum += B0[b] * abs2(r0) + 2 * B1[b] * real(r0 * conj(r1))

        r_lo = r_hi
    end

    ll[j] = dd_term + real(dh_sum) - T(0.5) * hh_sum
end

# IMRPhenomXPHM single-detector relative-binning kernel: combines
# `_relative_binning_kernel_phenomxp!`'s precession setup (+ `final_spin_override`) with
# `_relative_binning_kernel_phenomxhm!`'s higher-mode setup, evaluating the sparse `_xphm_h_det_scalar`
# at bin edges instead of `KernelFused._fused_waveform_likelihood_kernel_phenomxphm!`'s dense
# frequency loop -- exactly the same combination `_xphm_h_det_scalar` itself makes at the
# single-frequency level. `f_ref = edges[1]` (equal to `freqs[1]`, same reasoning as every other
# kernel in this file).
@kernel function _relative_binning_kernel_phenomxphm!(ll, @Const(params), @Const(edges), @Const(href_edges),
                                                        @Const(A0), @Const(A1), @Const(B0), @Const(B1),
                                                        dd_term, det, gmst_val, model)
    j = @index(Global, Linear)
    T = eltype(params)

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]
    chi1x = params[12, j]; chi1y = params[13, j]; chi2x = params[14, j]; chi2y = params[15, j]

    mc, eta = _chirp_mass_eta(m1, m2)

    Fplus, Fcross = antenna_response_gmst(det, ra, dec, psi, gmst_val)

    f_ref = edges[1]
    setup = _xphm_prec_setup(model, m1, m2, chi1x, chi1y, chi1, chi2x, chi2y, chi2,
                                  f_ref, theta_jn, phase; GMsun_over_c3=T(GMsun_over_c3))

    final_spin = setup.final_spin   # version-dependent (102: chi_p, 223: MSA)
    sys = _xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, zero(phase);
                             GMsun_over_c3=T(GMsun_over_c3), GMsun_over_c2_Gpc=T(GMsun_over_c2_Gpc),
                             final_spin_override=final_spin)
    ms21 = _xhm_mode_setup(2, 1, sys)
    ms33 = _xhm_mode_setup(3, 3, sys)
    ms44 = _xhm_mode_setup(4, 4, sys)
    m32 = _xhm_mode32_setup(sys)

    nb = length(A0)
    h_lo = _xphm_h_det_scalar(edges[1], setup, sys, ms21, ms33, ms44, m32, t_c, Fplus, Fcross)
    r_lo = _safe_ratio(h_lo, href_edges[1])

    dh_sum = zero(Complex{T})
    hh_sum = zero(T)
    @inbounds for b in 1:nb
        h_hi = _xphm_h_det_scalar(edges[b+1], setup, sys, ms21, ms33, ms44, m32, t_c, Fplus, Fcross)
        r_hi = _safe_ratio(h_hi, href_edges[b+1])

        r0 = T(0.5) * (r_lo + r_hi)
        r1 = (r_hi - r_lo) / (edges[b+1] - edges[b])

        dh_sum += A0[b] * conj(r0) + A1[b] * conj(r1)
        hh_sum += B0[b] * abs2(r0) + 2 * B1[b] * real(r0 * conj(r1))

        r_lo = r_hi
    end

    ll[j] = dd_term + real(dh_sum) - T(0.5) * hh_sum
end

# ---------------------------------------------------------------------------------------------
# Multi-detector kernel bodies. Mirror `KernelFused.jl`'s multi-detector kernel
# bodies exactly (same hoist-detector-independent-quantities-outside-the-loop structure, same
# `time_delay_from_geocenter_gmst`-based per-detector `t0` shift), applied to the sparse per-detector
# bin-edge evaluation instead of the full dense frequency grid. Each detector's own
# `RelativeBinningSetup` (`edges`/`href_edges`/`A0`/`A1`/`B0`/`B1`/`dd_term`) is looked up from a
# per-detector tuple, same as `KernelFused.jl`'s per-detector `freqs`/`strain`/`psd` tuples.
# ---------------------------------------------------------------------------------------------

@kernel function _relative_binning_kernel_phenomd_multi!(ll, @Const(params), edges_tuple, href_edges_tuple,
                                                           A0_tuple, A1_tuple, B0_tuple, B1_tuple, dd_term_tuple,
                                                           det_tuple, gmst_val, model)
    j = @index(Global, Linear)
    T = eltype(params)

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]

    mc, eta = _chirp_mass_eta(m1, m2)
    M_sec = (mc / eta^T(0.6)) * T(GMsun_over_c3)
    phiRef = phase
    pfac = T(0.5) * (1 + cos(theta_jn) * cos(theta_jn))
    cfac = cos(theta_jn)
    ampl_coeffs = _ampl_coeffs(model, mc, eta, chi1, chi2, dL)

    acc = zero(T)
    ndet = length(edges_tuple)
    @inbounds for d in 1:ndet
        edges = edges_tuple[d]; href_edges = href_edges_tuple[d]
        A0 = A0_tuple[d]; A1 = A1_tuple[d]; B0 = B0_tuple[d]; B1 = B1_tuple[d]
        dd_term = dd_term_tuple[d]
        det_d = det_tuple[d]

        dt_geo = time_delay_from_geocenter_gmst(det_d, ra, dec, gmst_val)
        t0 = -2 * T(pi) * (t_c + dt_geo) / M_sec
        Fplus, Fcross = antenna_response_gmst(det_d, ra, dec, psi, gmst_val)
        combo = Fplus * pfac - im * Fcross * cfac
        phi_coeffs = _phi_coeffs(model, mc, eta, chi1, chi2; t0=t0, phiRef=phiRef)

        nb = length(A0)
        amp_lo = _ampl_scalar(ampl_coeffs, edges[1])
        phase_lo = -_phi_scalar(phi_coeffs, edges[1])
        r_lo = _safe_ratio(combo * (amp_lo * Complex(cos(phase_lo), sin(phase_lo))), href_edges[1])

        dh_sum = zero(Complex{T})
        hh_sum = zero(T)
        for b in 1:nb
            amp_hi = _ampl_scalar(ampl_coeffs, edges[b+1])
            phase_hi = -_phi_scalar(phi_coeffs, edges[b+1])
            r_hi = _safe_ratio(combo * (amp_hi * Complex(cos(phase_hi), sin(phase_hi))), href_edges[b+1])

            r0 = T(0.5) * (r_lo + r_hi)
            r1 = (r_hi - r_lo) / (edges[b+1] - edges[b])

            dh_sum += A0[b] * conj(r0) + A1[b] * conj(r1)
            hh_sum += B0[b] * abs2(r0) + 2 * B1[b] * real(r0 * conj(r1))

            r_lo = r_hi
        end

        acc += dd_term + real(dh_sum) - T(0.5) * hh_sum
    end
    ll[j] = acc
end

@kernel function _relative_binning_kernel_phenomxas_multi!(ll, @Const(params), edges_tuple, href_edges_tuple,
                                                             A0_tuple, A1_tuple, B0_tuple, B1_tuple, dd_term_tuple,
                                                             det_tuple, gmst_val, model)
    j = @index(Global, Linear)
    T = eltype(params)

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]

    mc, eta = _chirp_mass_eta(m1, m2)
    M_sec = (mc / eta^T(0.6)) * T(GMsun_over_c3)
    phiRef = phase
    pfac = T(0.5) * (1 + cos(theta_jn) * cos(theta_jn))
    cfac = cos(theta_jn)
    ampl_coeffs = _xas_ampl_coeffs(model, mc, eta, chi1, chi2, dL)

    acc = zero(T)
    ndet = length(edges_tuple)
    @inbounds for d in 1:ndet
        edges = edges_tuple[d]; href_edges = href_edges_tuple[d]
        A0 = A0_tuple[d]; A1 = A1_tuple[d]; B0 = B0_tuple[d]; B1 = B1_tuple[d]
        dd_term = dd_term_tuple[d]
        det_d = det_tuple[d]

        dt_geo = time_delay_from_geocenter_gmst(det_d, ra, dec, gmst_val)
        t0 = -2 * T(pi) * (t_c + dt_geo) / M_sec
        Fplus, Fcross = antenna_response_gmst(det_d, ra, dec, psi, gmst_val)
        combo = Fplus * pfac - im * Fcross * cfac
        phi_coeffs = _xas_phi_coeffs(model, mc, eta, chi1, chi2; t0=-t0, phiRef=phiRef)

        nb = length(A0)
        amp_lo = _xas_ampl_scalar(ampl_coeffs, edges[1])
        phase_lo = _xas_phi_scalar(phi_coeffs, edges[1])
        r_lo = _safe_ratio(combo * (amp_lo * Complex(cos(phase_lo), sin(phase_lo))), href_edges[1])

        dh_sum = zero(Complex{T})
        hh_sum = zero(T)
        for b in 1:nb
            amp_hi = _xas_ampl_scalar(ampl_coeffs, edges[b+1])
            phase_hi = _xas_phi_scalar(phi_coeffs, edges[b+1])
            r_hi = _safe_ratio(combo * (amp_hi * Complex(cos(phase_hi), sin(phase_hi))), href_edges[b+1])

            r0 = T(0.5) * (r_lo + r_hi)
            r1 = (r_hi - r_lo) / (edges[b+1] - edges[b])

            dh_sum += A0[b] * conj(r0) + A1[b] * conj(r1)
            hh_sum += B0[b] * abs2(r0) + 2 * B1[b] * real(r0 * conj(r1))

            r_lo = r_hi
        end

        acc += dd_term + real(dh_sum) - T(0.5) * hh_sum
    end
    ll[j] = acc
end

# AlignedSpinPNHamF2 multi-detector relative-binning kernel:
# mirrors `_relative_binning_kernel_phenomxas_multi!` exactly, using the `_asphf2_*` scalar core
# instead of the `_xas_*` one.
@kernel function _relative_binning_kernel_alignedspinpnhamf2_multi!(ll, @Const(params), edges_tuple, href_edges_tuple,
                                                                       A0_tuple, A1_tuple, B0_tuple, B1_tuple, dd_term_tuple,
                                                                       det_tuple, gmst_val, model)
    j = @index(Global, Linear)
    T = eltype(params)

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]

    mc, eta = _chirp_mass_eta(m1, m2)
    M_sec = (mc / eta^T(0.6)) * T(GMsun_over_c3)
    phiRef = phase
    pfac = T(0.5) * (1 + cos(theta_jn) * cos(theta_jn))
    cfac = cos(theta_jn)
    ampl_coeffs = _asphf2_ampl_coeffs(model, mc, eta, chi1, chi2, dL)

    acc = zero(T)
    ndet = length(edges_tuple)
    @inbounds for d in 1:ndet
        edges = edges_tuple[d]; href_edges = href_edges_tuple[d]
        A0 = A0_tuple[d]; A1 = A1_tuple[d]; B0 = B0_tuple[d]; B1 = B1_tuple[d]
        dd_term = dd_term_tuple[d]
        det_d = det_tuple[d]

        dt_geo = time_delay_from_geocenter_gmst(det_d, ra, dec, gmst_val)
        t0 = -2 * T(pi) * (t_c + dt_geo) / M_sec
        Fplus, Fcross = antenna_response_gmst(det_d, ra, dec, psi, gmst_val)
        combo = Fplus * pfac - im * Fcross * cfac
        phi_coeffs = _asphf2_phi_coeffs(model, mc, eta, chi1, chi2; t0=t0, phiRef=phiRef)

        nb = length(A0)
        amp_lo = _asphf2_ampl_scalar(ampl_coeffs, edges[1])
        phase_lo = -_asphf2_phi_scalar(phi_coeffs, edges[1])
        r_lo = _safe_ratio(combo * (amp_lo * Complex(cos(phase_lo), sin(phase_lo))), href_edges[1])

        dh_sum = zero(Complex{T})
        hh_sum = zero(T)
        for b in 1:nb
            amp_hi = _asphf2_ampl_scalar(ampl_coeffs, edges[b+1])
            phase_hi = -_asphf2_phi_scalar(phi_coeffs, edges[b+1])
            r_hi = _safe_ratio(combo * (amp_hi * Complex(cos(phase_hi), sin(phase_hi))), href_edges[b+1])

            r0 = T(0.5) * (r_lo + r_hi)
            r1 = (r_hi - r_lo) / (edges[b+1] - edges[b])

            dh_sum += A0[b] * conj(r0) + A1[b] * conj(r1)
            hh_sum += B0[b] * abs2(r0) + 2 * B1[b] * real(r0 * conj(r1))

            r_lo = r_hi
        end

        acc += dd_term + real(dh_sum) - T(0.5) * hh_sum
    end
    ll[j] = acc
end

# SEOBNRv5AlignedSpinF2 multi-detector relative-binning kernel: mirrors
# `_relative_binning_kernel_alignedspinpnhamf2_multi!` exactly, using `_seobv5f2_*` instead of
# `_asphf2_*` -- `coeffs` (detector-independent) is hoisted outside the `1:ndet` loop. No explicit
# `f_min` is passed (this model's own default) -- see the single-detector sibling kernel above's
# comment for why.
@kernel function _relative_binning_kernel_seobnrv5alignedspinf2_multi!(ll, @Const(params), edges_tuple, href_edges_tuple,
                                                                           A0_tuple, A1_tuple, B0_tuple, B1_tuple, dd_term_tuple,
                                                                           det_tuple, gmst_val, model)
    j = @index(Global, Linear)
    T = eltype(params)

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]

    mc, eta = _chirp_mass_eta(m1, m2)
    M_sec = (mc / eta^T(0.6)) * T(GMsun_over_c3)
    phiRef = phase
    pfac = T(0.5) * (1 + cos(theta_jn) * cos(theta_jn))
    cfac = cos(theta_jn)
    coeffs = _seobv5f2_coeffs(model, mc, eta, chi1, chi2)

    acc = zero(T)
    ndet = length(edges_tuple)
    @inbounds for d in 1:ndet
        edges = edges_tuple[d]; href_edges = href_edges_tuple[d]
        A0 = A0_tuple[d]; A1 = A1_tuple[d]; B0 = B0_tuple[d]; B1 = B1_tuple[d]
        dd_term = dd_term_tuple[d]
        det_d = det_tuple[d]

        dt_geo = time_delay_from_geocenter_gmst(det_d, ra, dec, gmst_val)
        tshift = -2 * T(pi) * (t_c + dt_geo)      # seconds -> rad/Hz; NOT the Mf-based t0 (see KernelFused._phi_t0)
        Fplus, Fcross = antenna_response_gmst(det_d, ra, dec, psi, gmst_val)
        combo = Fplus * pfac - im * Fcross * cfac

        nb = length(A0)
        amp_lo = _seobv5f2_ampl_scalar(coeffs, edges[1], dL, GMsun_over_c2_Gpc)
        phase_lo = -(_seobv5f2_phi_scalar(coeffs, edges[1]) - tshift * edges[1] - phiRef)
        r_lo = _safe_ratio(combo * (amp_lo * Complex(cos(phase_lo), sin(phase_lo))), href_edges[1])

        dh_sum = zero(Complex{T})
        hh_sum = zero(T)
        for b in 1:nb
            amp_hi = _seobv5f2_ampl_scalar(coeffs, edges[b+1], dL, GMsun_over_c2_Gpc)
            phase_hi = -(_seobv5f2_phi_scalar(coeffs, edges[b+1]) - tshift * edges[b+1] - phiRef)
            r_hi = _safe_ratio(combo * (amp_hi * Complex(cos(phase_hi), sin(phase_hi))), href_edges[b+1])

            r0 = T(0.5) * (r_lo + r_hi)
            r1 = (r_hi - r_lo) / (edges[b+1] - edges[b])

            dh_sum += A0[b] * conj(r0) + A1[b] * conj(r1)
            hh_sum += B0[b] * abs2(r0) + 2 * B1[b] * real(r0 * conj(r1))

            r_lo = r_hi
        end

        acc += dd_term + real(dh_sum) - T(0.5) * hh_sum
    end
    ll[j] = _seobv5f2_apply_deadzone_reject(acc, coeffs, edges_tuple[1][1])
end

# IMRPhenomXP multi-detector relative-binning kernel: mirrors
# `KernelFused._fused_waveform_likelihood_kernel_phenomxp_multi!` EXACTLY (the ONE structural
# difference from the aligned-spin multi-detector kernel above: `_xp_precession_setup`'s `setup`
# AND `ampl_coeffs` are genuinely detector-independent, so hoisted OUTSIDE the `1:ndet` loop,
# anchored to the FIRST detector's own first bin edge `edges_tuple[1][1]` -- equal to that
# detector's dense `freqs[1]`, same reasoning as the single-detector kernel above -- since `setup`
# can only be built once; `t0`/`phi_coeffs`/`Fplus`/`Fcross` are recomputed per detector since they
# depend on that detector's own time-delay-shifted `t_c` and antenna-response geometry), applied to
# the sparse per-detector bin-edge evaluation (`_xp_h_det_scalar`) instead of the dense per-detector
# frequency loop.
@kernel function _relative_binning_kernel_phenomxp_multi!(ll, @Const(params), edges_tuple, href_edges_tuple,
                                                            A0_tuple, A1_tuple, B0_tuple, B1_tuple, dd_term_tuple,
                                                            det_tuple, gmst_val, model)
    j = @index(Global, Linear)
    T = eltype(params)

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]
    chi1x = params[12, j]; chi1y = params[13, j]; chi2x = params[14, j]; chi2y = params[15, j]

    mc, eta = _chirp_mass_eta(m1, m2)
    M_sec = (mc / eta^T(0.6)) * T(GMsun_over_c3)
    phiRef = phase

    # Detector-independent (hoisted outside the 1:ndet loop below): setup/ampl_coeffs.
    f_ref = edges_tuple[1][1]
    setup = _xp_precession_setup(m1, m2, chi1x, chi1y, chi1, chi2x, chi2y, chi2,
                                  f_ref, theta_jn, phase; GMsun_over_c3=T(GMsun_over_c3))
    ampl_coeffs = _xas_ampl_coeffs(IMRPhenomXAS(), mc, eta, chi1, chi2, dL)

    acc = zero(T)
    ndet = length(edges_tuple)
    @inbounds for d in 1:ndet
        edges = edges_tuple[d]; href_edges = href_edges_tuple[d]
        A0 = A0_tuple[d]; A1 = A1_tuple[d]; B0 = B0_tuple[d]; B1 = B1_tuple[d]
        dd_term = dd_term_tuple[d]
        det_d = det_tuple[d]

        # Detector-specific (recomputed every iteration): t0/phi_coeffs (time-delay-shifted t_c)
        # and Fplus/Fcross (detector geometry).
        dt_geo = time_delay_from_geocenter_gmst(det_d, ra, dec, gmst_val)
        t0 = -2 * T(pi) * (t_c + dt_geo) / M_sec
        Fplus, Fcross = antenna_response_gmst(det_d, ra, dec, psi, gmst_val)
        phi_coeffs = _xas_phi_coeffs(IMRPhenomXAS(), mc, eta, chi1, chi2; t0=-t0, phiRef=phiRef)

        nb = length(A0)
        h_lo = _xp_h_det_scalar(edges[1], M_sec, ampl_coeffs, phi_coeffs, setup, Fplus, Fcross)
        r_lo = _safe_ratio(h_lo, href_edges[1])

        dh_sum = zero(Complex{T})
        hh_sum = zero(T)
        for b in 1:nb
            h_hi = _xp_h_det_scalar(edges[b+1], M_sec, ampl_coeffs, phi_coeffs, setup, Fplus, Fcross)
            r_hi = _safe_ratio(h_hi, href_edges[b+1])

            r0 = T(0.5) * (r_lo + r_hi)
            r1 = (r_hi - r_lo) / (edges[b+1] - edges[b])

            dh_sum += A0[b] * conj(r0) + A1[b] * conj(r1)
            hh_sum += B0[b] * abs2(r0) + 2 * B1[b] * real(r0 * conj(r1))

            r_lo = r_hi
        end

        acc += dd_term + real(dh_sum) - T(0.5) * hh_sum
    end
    ll[j] = acc
end

# IMRPhenomXHM multi-detector relative-binning kernel: mirrors
# `_relative_binning_kernel_phenomxp_multi!` above EXACTLY (the one structural difference from the
# aligned-spin multi-detector kernel: `_xhm_system_setup`/`_xhm_mode_setup`(x3)/`_xhm_mode32_setup`
# are genuinely detector-independent -- none of `mc`/`eta`/`chi1`/`chi2`/`dL`/`f_ref`/`phase`
# depends on detector geometry or the per-detector time-delay shift -- so hoisted OUTSIDE the
# `1:ndet` loop, anchored to the FIRST detector's own first bin edge `edges_tuple[1][1]`, same
# reasoning as the single-detector kernel above; `Fplus`/`Fcross` AND the `t_c` used in the
# post-multiplication are recomputed per detector, since they depend on that detector's own
# time-delay-shifted `t_c` and antenna-response geometry), applied to the sparse per-detector
# bin-edge evaluation (`_xhm_h_det_scalar`) instead of a dense per-detector frequency loop.
@kernel function _relative_binning_kernel_phenomxhm_multi!(ll, @Const(params), edges_tuple, href_edges_tuple,
                                                             A0_tuple, A1_tuple, B0_tuple, B1_tuple, dd_term_tuple,
                                                             det_tuple, gmst_val, model)
    j = @index(Global, Linear)
    T = eltype(params)

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]

    mc, eta = _chirp_mass_eta(m1, m2)

    # Detector-independent (hoisted outside the 1:ndet loop below): sys/ms21/ms33/ms44/m32.
    f_ref = edges_tuple[1][1]
    sys = _xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, phase;
                             GMsun_over_c3=T(GMsun_over_c3), GMsun_over_c2_Gpc=T(GMsun_over_c2_Gpc))
    ms21 = _xhm_mode_setup(2, 1, sys)
    ms33 = _xhm_mode_setup(3, 3, sys)
    ms44 = _xhm_mode_setup(4, 4, sys)
    m32 = _xhm_mode32_setup(sys)
    piHalf = T(pi) / T(2)

    acc = zero(T)
    ndet = length(edges_tuple)
    @inbounds for d in 1:ndet
        edges = edges_tuple[d]; href_edges = href_edges_tuple[d]
        A0 = A0_tuple[d]; A1 = A1_tuple[d]; B0 = B0_tuple[d]; B1 = B1_tuple[d]
        dd_term = dd_term_tuple[d]
        det_d = det_tuple[d]

        # Detector-specific (recomputed every iteration): the time-delay-shifted t_c used in the
        # post-multiplication, and Fplus/Fcross (detector geometry).
        dt_geo = time_delay_from_geocenter_gmst(det_d, ra, dec, gmst_val)
        t_c_d = t_c + dt_geo
        Fplus, Fcross = antenna_response_gmst(det_d, ra, dec, psi, gmst_val)

        nb = length(A0)
        h_lo = _xhm_h_det_scalar(edges[1], sys, ms21, ms33, ms44, m32, theta_jn, piHalf, t_c_d, Fplus, Fcross)
        r_lo = _safe_ratio(h_lo, href_edges[1])

        dh_sum = zero(Complex{T})
        hh_sum = zero(T)
        for b in 1:nb
            h_hi = _xhm_h_det_scalar(edges[b+1], sys, ms21, ms33, ms44, m32, theta_jn, piHalf, t_c_d, Fplus, Fcross)
            r_hi = _safe_ratio(h_hi, href_edges[b+1])

            r0 = T(0.5) * (r_lo + r_hi)
            r1 = (r_hi - r_lo) / (edges[b+1] - edges[b])

            dh_sum += A0[b] * conj(r0) + A1[b] * conj(r1)
            hh_sum += B0[b] * abs2(r0) + 2 * B1[b] * real(r0 * conj(r1))

            r_lo = r_hi
        end

        acc += dd_term + real(dh_sum) - T(0.5) * hh_sum
    end
    ll[j] = acc
end

# IMRPhenomXPHM multi-detector relative-binning kernel: combines
# `_relative_binning_kernel_phenomxp_multi!`'s detector-independent precession setup (+
# `final_spin_override`) with `_relative_binning_kernel_phenomxhm_multi!`'s detector-independent
# higher-mode setup and per-detector `t_c + dt_geo` post-multiplication -- exactly the same
# combination the single-detector `_relative_binning_kernel_phenomxphm!` above makes, generalized
# to multiple detectors the same way every other multi-detector kernel in this file is.
@kernel function _relative_binning_kernel_phenomxphm_multi!(ll, @Const(params), edges_tuple, href_edges_tuple,
                                                               A0_tuple, A1_tuple, B0_tuple, B1_tuple, dd_term_tuple,
                                                               det_tuple, gmst_val, model)
    j = @index(Global, Linear)
    T = eltype(params)

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]
    chi1x = params[12, j]; chi1y = params[13, j]; chi2x = params[14, j]; chi2y = params[15, j]

    mc, eta = _chirp_mass_eta(m1, m2)

    # Detector-independent (hoisted outside the 1:ndet loop below): precession setup,
    # final_spin_override, and the per-walker mode setups.
    f_ref = edges_tuple[1][1]
    setup = _xphm_prec_setup(model, m1, m2, chi1x, chi1y, chi1, chi2x, chi2y, chi2,
                                  f_ref, theta_jn, phase; GMsun_over_c3=T(GMsun_over_c3))

    final_spin = setup.final_spin   # version-dependent (102: chi_p, 223: MSA)
    sys = _xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, zero(phase);
                             GMsun_over_c3=T(GMsun_over_c3), GMsun_over_c2_Gpc=T(GMsun_over_c2_Gpc),
                             final_spin_override=final_spin)
    ms21 = _xhm_mode_setup(2, 1, sys)
    ms33 = _xhm_mode_setup(3, 3, sys)
    ms44 = _xhm_mode_setup(4, 4, sys)
    m32 = _xhm_mode32_setup(sys)

    acc = zero(T)
    ndet = length(edges_tuple)
    @inbounds for d in 1:ndet
        edges = edges_tuple[d]; href_edges = href_edges_tuple[d]
        A0 = A0_tuple[d]; A1 = A1_tuple[d]; B0 = B0_tuple[d]; B1 = B1_tuple[d]
        dd_term = dd_term_tuple[d]
        det_d = det_tuple[d]

        # Detector-specific (recomputed every iteration): the time-delay-shifted t_c used in the
        # post-multiplication, and Fplus/Fcross (detector geometry).
        dt_geo = time_delay_from_geocenter_gmst(det_d, ra, dec, gmst_val)
        t_c_d = t_c + dt_geo
        Fplus, Fcross = antenna_response_gmst(det_d, ra, dec, psi, gmst_val)

        nb = length(A0)
        h_lo = _xphm_h_det_scalar(edges[1], setup, sys, ms21, ms33, ms44, m32, t_c_d, Fplus, Fcross)
        r_lo = _safe_ratio(h_lo, href_edges[1])

        dh_sum = zero(Complex{T})
        hh_sum = zero(T)
        for b in 1:nb
            h_hi = _xphm_h_det_scalar(edges[b+1], setup, sys, ms21, ms33, ms44, m32, t_c_d, Fplus, Fcross)
            r_hi = _safe_ratio(h_hi, href_edges[b+1])

            r0 = T(0.5) * (r_lo + r_hi)
            r1 = (r_hi - r_lo) / (edges[b+1] - edges[b])

            dh_sum += A0[b] * conj(r0) + A1[b] * conj(r1)
            hh_sum += B0[b] * abs2(r0) + 2 * B1[b] * real(r0 * conj(r1))

            r_lo = r_hi
        end

        acc += dd_term + real(dh_sum) - T(0.5) * hh_sum
    end
    ll[j] = acc
end

"""
    binned_log_likelihood_kernel!(ll, params, setup::RelativeBinningSetup{T};
                                   backend=KernelAbstractions.CPU(), model=IMRPhenomD())

Relative-binning approximation to [`KernelFused.batched_log_likelihood_kernel!`](@ref): for each
walker `j`, evaluates the trial waveform only at `setup`'s sparse bin-edge frequencies (via the
same non-allocating `_phi_coeffs`/`_phi_scalar`/`_ampl_coeffs`/`_ampl_scalar` scalar core used by
the full-grid fused kernel), linearly interpolates the amplitude/phase ratio relative to the
reference waveform within each bin, and combines with `setup`'s precomputed `A0`/`A1`/`B0`/`B1`
summary coefficients -- see this file's header comment for the exact formulas (Cornish 2010;
Zackay, Dai & Venumadhav 2018).

`model isa IMRPhenomD`/`IMRPhenomXAS`/`AlignedSpinPNHamF2`/`IMRPhenomXP`/`IMRPhenomXHM`/`IMRPhenomXPHM`
dispatch to specialized kernel bodies exactly like `batched_log_likelihood_kernel!`; no other model
is currently supported. `params` is
`(11, nwalkers)` for `IMRPhenomD`/`IMRPhenomXAS`/`IMRPhenomXHM`, `(15, nwalkers)` for `IMRPhenomXP` (same
PHYSICAL row convention as `KernelFused.jl`, asserted here exactly like `batched_log_likelihood_kernel!`
does).

`data::LikelihoodData` supplies `detector`/`gps_time` for the antenna-response projection (its
`freqs`/`strain`/`psd` are NOT reused here -- `setup` already carries everything derived from
those, per its docstring). Same `gmst_val = GravitationalWaveData.gmst(data.gps_time)`-computed-once-outside-the-
kernel pattern as `batched_log_likelihood_kernel!`; see that function's docstring / this file's
module docstring for why `ra`/`dec`/`psi` themselves must still flow in via `params` and be used
INSIDE the kernel body (Enzyme correctness).

Written generically against `KernelAbstractions.Backend`, same portability caveat as
`batched_log_likelihood_kernel!`: only `KernelAbstractions.CPU()` is exercised/tested here.
"""
function binned_log_likelihood_kernel!(
    ll::AbstractVector{T},
    params::AbstractMatrix{T},
    setup::RelativeBinningSetup{T},
    data::LikelihoodData;
    backend::KernelAbstractions.Backend=KernelAbstractions.CPU(),
    model=IMRPhenomD(),
) where {T<:Real}
    @assert size(params, 1) == (model isa IMRPhenomXP || model isa IMRPhenomXPHM ? 15 : 11)
    @assert length(ll) == size(params, 2)

    nwalkers = size(params, 2)
    gmst_val = gmst(data.gps_time)
    kernel! = if model isa IMRPhenomD
        _relative_binning_kernel_phenomd!(backend)
    elseif model isa IMRPhenomXAS
        _relative_binning_kernel_phenomxas!(backend)
    elseif model isa AlignedSpinPNHamF2
        _relative_binning_kernel_alignedspinpnhamf2!(backend)
    elseif model isa PAF2Model
        _relative_binning_kernel_seobnrv5alignedspinf2!(backend)
    elseif model isa IMRPhenomXP
        _relative_binning_kernel_phenomxp!(backend)
    elseif model isa IMRPhenomXHM
        _relative_binning_kernel_phenomxhm!(backend)
    elseif model isa IMRPhenomXPHM
        _relative_binning_kernel_phenomxphm!(backend)
    else
        error("binned_log_likelihood_kernel! only supports IMRPhenomD/IMRPhenomXAS/AlignedSpinPNHamF2/SEOBNRv5AlignedSpinF2/SEOBNRv5IMRAlignedSpinF2/IMRPhenomXP/IMRPhenomXHM/IMRPhenomXPHM models (got $(typeof(model)))")
    end
    # `workgroupsize=1` on CPU only: KernelAbstractions.CPU() distributes work across
    # `Threads.nthreads()` PER WORKGROUP, not per work item -- the default workgroupsize heuristic
    # collapses everything into ONE workgroup whenever `prod(ndrange) <= 1024` (always true here),
    # silently forcing single-threaded execution regardless of `-t`/`Threads.nthreads()` (confirmed
    # empirically: identical wall-clock at -t1 vs -t6 without this, ~2.7x speedup at -t4 with it).
    # `nothing` (KA's own default) on other backends -- GPU backends want large workgroups.
    kernel!(ll, params, setup.edges, setup.href_edges, setup.A0, setup.A1, setup.B0, setup.B1,
            setup.dd_term, geometry(data.detector), gmst_val, model;
            ndrange=nwalkers, workgroupsize=(backend isa KernelAbstractions.CPU ? 1 : nothing))
    KernelAbstractions.synchronize(backend)
    return ll
end

"""
    binned_log_likelihood_kernel!(ll, params, setups::AbstractVector{<:RelativeBinningSetup{T}},
                                   data::AbstractVector{<:LikelihoodData};
                                   backend=KernelAbstractions.CPU(), model=IMRPhenomD())

Multi-detector method: sums the relative-binning-approximated log-likelihood
across every detector, mirroring [`KernelFused.batched_log_likelihood_kernel!`](@ref)'s own
multi-detector method exactly (one kernel thread per walker, an inner `for d in 1:ndet` loop over
detectors, per-detector `t0` via `time_delay_from_geocenter` and antenna response via that
detector's own `response`/`location`). `setups[i]` must correspond to `data[i]` (same detector,
same order) — both must have the same length. All `data` entries must share the same `gps_time`.

Build `setups` via the multi-detector [`build_relative_binning`](@ref) above (one call, all
detectors at once) rather than assembling the vector by hand.

`model isa IMRPhenomD`/`IMRPhenomXAS`/`AlignedSpinPNHamF2`/`IMRPhenomXP`/`IMRPhenomXHM`/`IMRPhenomXPHM`
dispatch exactly like the single-detector method above; `params` is `(11, nwalkers)` for
`IMRPhenomD`/`IMRPhenomXAS`/`AlignedSpinPNHamF2`/`IMRPhenomXHM`,
`(15, nwalkers)` for `IMRPhenomXP`. Note `IMRPhenomXHM` is the ONE case here that does NOT mirror an
existing `KernelFused.jl` multi-detector dense kernel -- `KernelFused.jl`'s multi-detector
`batched_log_likelihood_kernel!` still explicitly rejects `IMRPhenomXHM` (multi-detector support for
the DENSE kernel remains a separate, out-of-scope follow-on, see that file's module docstring) --
this relative-binning multi-detector `IMRPhenomXHM` path is self-contained (its own
`RelativeBinningSetup`s built directly from each detector's `LikelihoodData`, summed here exactly
like every other model), not a sparse re-expression of a dense multi-detector kernel that exists
elsewhere in this codebase yet.
"""
function binned_log_likelihood_kernel!(
    ll::AbstractVector{T},
    params::AbstractMatrix{T},
    setups::AbstractVector{<:RelativeBinningSetup{T}},
    data::AbstractVector{<:LikelihoodData};
    backend::KernelAbstractions.Backend=KernelAbstractions.CPU(),
    model=IMRPhenomD(),
) where {T<:Real}
    @assert size(params, 1) == (model isa IMRPhenomXP || model isa IMRPhenomXPHM ? 15 : 11)
    @assert length(ll) == size(params, 2)
    ndet = length(setups)
    @assert ndet == length(data) "setups and data must have the same number of detectors, in the same order"
    @assert ndet >= 1 "multi-detector binned_log_likelihood_kernel! needs at least one detector"

    gps_time = data[1].gps_time
    @inbounds for i in 2:ndet
        data[i].gps_time == gps_time || throw(ArgumentError(
            "all detectors must share the same event gps_time (got $(data[i].gps_time) at index $i, expected $gps_time)"))
    end
    gmst_val = gmst(gps_time)

    edges_tuple = ntuple(i -> setups[i].edges, ndet)
    href_edges_tuple = ntuple(i -> setups[i].href_edges, ndet)
    A0_tuple = ntuple(i -> setups[i].A0, ndet)
    A1_tuple = ntuple(i -> setups[i].A1, ndet)
    B0_tuple = ntuple(i -> setups[i].B0, ndet)
    B1_tuple = ntuple(i -> setups[i].B1, ndet)
    dd_term_tuple = ntuple(i -> setups[i].dd_term, ndet)
    det_tuple = ntuple(i -> geometry(data[i].detector), ndet)

    nwalkers = size(params, 2)
    kernel! = if model isa IMRPhenomD
        _relative_binning_kernel_phenomd_multi!(backend)
    elseif model isa IMRPhenomXAS
        _relative_binning_kernel_phenomxas_multi!(backend)
    elseif model isa AlignedSpinPNHamF2
        _relative_binning_kernel_alignedspinpnhamf2_multi!(backend)
    elseif model isa PAF2Model
        _relative_binning_kernel_seobnrv5alignedspinf2_multi!(backend)
    elseif model isa IMRPhenomXP
        _relative_binning_kernel_phenomxp_multi!(backend)
    elseif model isa IMRPhenomXHM
        _relative_binning_kernel_phenomxhm_multi!(backend)
    elseif model isa IMRPhenomXPHM
        _relative_binning_kernel_phenomxphm_multi!(backend)
    else
        error("binned_log_likelihood_kernel! only supports IMRPhenomD/IMRPhenomXAS/AlignedSpinPNHamF2/SEOBNRv5AlignedSpinF2/SEOBNRv5IMRAlignedSpinF2/IMRPhenomXP/IMRPhenomXHM/IMRPhenomXPHM models (got $(typeof(model)))")
    end
    # See the single-detector method above for why `workgroupsize=1` on CPU is needed for real
    # multithreading.
    kernel!(ll, params, edges_tuple, href_edges_tuple, A0_tuple, A1_tuple, B0_tuple, B1_tuple, dd_term_tuple,
            det_tuple, gmst_val, model;
            ndrange=nwalkers, workgroupsize=(backend isa KernelAbstractions.CPU ? 1 : nothing))
    KernelAbstractions.synchronize(backend)
    return ll
end

end
