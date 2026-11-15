module KernelFused

export batched_log_likelihood_kernel!

using KernelAbstractions
import LinearAlgebra
using ..Types: PEConfig, LikelihoodData
import ..GPU
using GravitationalWaveforms: Ampl, Phi, IMRPhenomD, IMRPhenomXAS, IMRPhenomXP, IMRPhenomXHM, IMRPhenomXPHM, AlignedSpinPNHamF2, SEOBNRv5AlignedSpinF2, SEOBNRv5PHM
using ..Types: PAF2Model
import GravitationalWaveforms

# Models sampled with the 15-row precessing-spin convention (rows 12-15: chi1x, chi1y, chi2x, chi2y in the L_N frame at
# f_ref; row 10 the L_N inclination; row 7 the reference phase).
const _Prec15 = Union{IMRPhenomXP,IMRPhenomXPHM,SEOBNRv5PHM}
using GravitationalWaveforms: _phi_coeffs, _phi_scalar, _ampl_coeffs, _ampl_scalar, GMsun_over_c3, GMsun_over_c2_Gpc
using GravitationalWaveforms: _xas_phi_coeffs, _xas_phi_scalar, _xas_ampl_coeffs, _xas_ampl_scalar, _xas_precessing_final_spin
using GravitationalWaveforms: _xas_phi_ins_deriv, _xas_phi_int_deriv, _xas_phi_mrd_deriv
using GravitationalWaveforms: _xas_phi_ins_val, _xas_phi_int_val, _xas_phi_mrd_val, _xas_amp_ins_val, _xas_amp_rd_val
using GravitationalWaveforms: _asphf2_phi_coeffs, _asphf2_phi_scalar, _asphf2_ampl_coeffs, _asphf2_ampl_scalar
using GravitationalWaveforms: _seobv5f2_coeffs, _seobv5f2_phi_scalar, _seobv5f2_ampl_scalar, _seobv5f2_dense_to_freq, EOBIMRTable,
    _seobv5f2_amp_phi
using GravitationalWaveforms: _xp_precession_setup, _xp_twist_up22, _xp_apply_zeta_rotation, _XP_HAS_MODE_NORMALIZATION
using GravitationalWaveforms: _xhm_system_setup, _xhm_mode_setup, _xhm_mode32_setup, _xhm_hphc_scalar, _xphm_hphc_scalar,
                   _xhm22_phase_full, _xhm_hlm_nonmixing, _xhm_hlm_32, _xhm_add_mode!
using GravitationalWaveforms: SEOBNRv5HM
using GravitationalWaveData: gmst, DetectorGeometry, geometry, antenna_response_gmst, time_delay_from_geocenter_gmst
import FFTW
using StaticArrays: SVector

# =================================================================================================
# Physical-parameter row convention (single-detector, aligned-spin, 11-parameter extrinsic set).
# `params`/`physical` matrices consumed by every kernel body in this file (and
# by `RelativeBinning.jl`'s kernel bodies, which mirror this exactly) are `(11, nwalkers)`, rows:
#
#   1. m1        -- primary mass, solar masses
#   2. m2        -- secondary mass, solar masses
#   3. chi1      -- primary aligned dimensionless spin, [-1, 1]
#   4. chi2      -- secondary aligned dimensionless spin, [-1, 1]
#   5. dL        -- luminosity distance, Gpc (matches `Ampl`'s own `dL` unit convention)
#   6. t_c       -- coalescence-time OFFSET from `LikelihoodData.gps_time`, seconds (small window,
#                   e.g. +/-0.1s -- NOT the absolute GPS trigger time itself, which is fixed data,
#                   not sampled; see `LikelihoodData`'s docstring in Types.jl)
#   7. phase     -- reference/coalescence phase, radians
#   8. ra        -- right ascension, radians, [0, 2*pi)
#   9. dec       -- declination, radians, [-pi/2, pi/2]
#  10. theta_jn  -- inclination angle between the line of sight and orbital angular momentum,
#                   radians, [0, pi] -- called `theta_jn` (matching the standard aligned-spin/
#                   non-precessing PE convention) since it coincides with `iota` when there is no
#                   precession, which is always true here (aligned spins only)
#  11. psi       -- polarization angle, radians, [0, pi)
#
# Rows 1-4 are UNCHANGED from the pre-Task-4 4-parameter-only convention. This is the PHYSICAL
# parametrization the kernels consume; `BatchedNUTS.jl`'s `_reparam_to_physical!` documents the
# separate SAMPLED-space row convention (`(mc, q, chi1, chi2, dL, t_c, phase, ra, sin_dec,
# cos_theta_jn, psi)`) and how it maps onto this one.
#
# The precessing-spin models (`IMRPhenomXP`, see `_fused_waveform_likelihood_kernel_phenomxp!`
# below for single-detector and `_fused_waveform_likelihood_kernel_phenomxp_multi!` for
# multi-detector) appends 4 more rows, WITHOUT renumbering/renaming rows 1-11 above (`chi1`/`chi2`
# keep their existing names/meanings, now understood as the L-aligned/z-component of each body's
# spin rather than "the whole spin", since in-plane components now exist too):
#
#  12. chi1x  -- primary in-plane spin x-component, [-1, 1] (source-frame, L-aligned-z convention,
#                same frame `_xp_precession_setup` in GravitationalWaveforms.jl expects)
#  13. chi1y  -- primary in-plane spin y-component, [-1, 1]
#  14. chi2x  -- secondary in-plane spin x-component, [-1, 1]
#  15. chi2y  -- secondary in-plane spin y-component, [-1, 1]
#
# Sampled DIRECTLY, no reparametrization (matching rows 3-4/5-8/11's existing pass-through style).
# `params`/`physical` is `(11, nwalkers)` for `IMRPhenomD`/`IMRPhenomXAS`, `(15, nwalkers)` for `IMRPhenomXP`
# -- both the single- and multi-detector `batched_log_likelihood_kernel!` methods below assert the
# row count matches `model`. Multi-detector `IMRPhenomXP` now dispatches to
# `_fused_waveform_likelihood_kernel_phenomxp_multi!` (a Phase-3 follow-on task); `RelativeBinning.jl`
# now also dispatches `model isa IMRPhenomXP` (single- and multi-detector), a later Phase-3 follow-on
# task -- see that module's own kernel bodies/docstrings for the twisting-up-vs-sparse-bin-edges
# adaptation.
#
# `IMRPhenomXHM` (single-detector -- see
# `_fused_waveform_likelihood_kernel_phenomxhm!` below) is ALIGNED-SPIN, so it uses the EXISTING
# 11-row convention above, NOT the 15-row `IMRPhenomXP` extension -- `theta_jn` (row 10) is passed
# straight through as `GravitationalWaveforms.hphc`'s `inclination` argument (the angle between the fixed
# orbital angular momentum and the line of sight; there is no precession here to complicate that
# identification, unlike `IMRPhenomXP`'s own `inclination`-at-`f_ref` caveat). `GravitationalWaveforms.IMRPhenomXHM`
# doesn't share the `Ampl`/`Phi` scalar contract at all (it's a genuine sum over 5 complex modes,
# see that package's own `IMRPhenomXHM.jl` module docstring) -- it instead already returns
# fully-formed `(hp, hc)` per frequency bin via its own non-allocating `_xhm_hphc_scalar` scalar
# core, so this kernel projects via `h = Fplus*hp + Fcross*hc` exactly like `IMRPhenomXP`'s kernel
# does (NOT `_extrinsic_setup`'s `combo`-shortcut, which only works for a single-scalar-`h` model).
# `_xhm_system_setup`/`hphc` has NO `t_c` input at all -- unlike `_xas_phi_coeffs`'s `t0` kwarg or
# `_xp_precession_setup`'s implicit handling -- so the standard frequency-domain time-shift is
# applied directly as a post-multiplication in PHYSICAL frequency `f`: `(hp, hc) -> (hp, hc) *
# exp(im*2*pi*f*t_c)`. This reproduces the SAME net convention this file's other kernels get from
# `t0 = -2*pi*t_c/M_sec` (see above: `_phi_scalar` adds `-t0*fgrid = +2*pi*f*t_c` to the phase,
# i.e. `h(f) -> h(f)*exp(im*2*pi*f*t_c)`) -- empirically verified here too (not just derived) via a
# synthetic injection-recovery check, see `test/test_phenomxhm.jl`. Multi-detector `IMRPhenomXHM`
# (`_fused_waveform_likelihood_kernel_phenomxhm_multi!` below) generalizes the
# `t_c` post-multiplication to `exp(im*2*pi*f*(t_c + dt_geo))` per detector, `dt_geo` from
# `GravitationalWaveData.time_delay_from_geocenter_gmst` -- see that kernel body's own comment block for the derivation.
# `RelativeBinning.jl` supports `IMRPhenomXHM` (single- and multi-detector), see that file's
# `build_relative_binning(::IMRPhenomXHM, ...)` docstrings; its multi-detector path is self-contained
# and does not touch this file's multi-detector method.
#
# Physics applied per walker (see this file's kernel bodies below for the exact code):
#   - `dL` is threaded straight into `Ampl`'s existing `dL` argument (confirmed by direct REPL
#     test against this codebase's `GravitationalWaveforms.jl` dependency: `Ampl(...,dL=2) ==
#     0.5*Ampl(...,dL=1)` to full floating-point precision) -- NOT a separate post-hoc 1/dL
#     rescaling. No redundant distance-scaling code exists elsewhere in this file.
#   - `t_c`/`phase` are threaded into `Phi`'s existing `t0`/`phiRef` keyword arguments, NOT a
#     redundant post-hoc `exp(-2*pi*im*f*t_c + im*phase)` multiplication (which would double-apply
#     the shift). Their exact unit convention was determined empirically (not assumed from the
#     kwarg names alone): `_phi_scalar` adds `-t0*fgrid - phiRef` to the phase, where
#     `fgrid = M_sec*f` is the DIMENSIONLESS geometric frequency (`M_sec = M_total_solar_masses *
#     GMsun_over_c3`), NOT `f` itself. So `t0` must be pre-scaled by that walker's own `M_sec` to
#     represent a physical time in seconds: `t0 = -2*pi*t_c/M_sec` (derived in `_extrinsic_setup`
#     below) gives `phase(f) += 2*pi*f*t_c`, the standard SPA linear-in-frequency time-shift term.
#     `phiRef = phase` (the sampled reference phase, pass-through, matching `Phi`'s own `-phiRef`
#     convention). This sign/scale choice is internally self-consistent (validated by the
#     synthetic-injection NUTS-recovery test, which constructs the injection with this exact
#     forward model) but was NOT independently cross-checked against LALSuite's own `t_c`/`phase`
#     sign convention -- only the plus/cross decomposition and `dL` scaling were LALSuite-validated.
#   - Plus/cross decomposition of the dominant (2,2)-mode-only strain `h = Ampl*exp(im*Phi)` (now
#     with `dL`/`t_c`/`phase` folded in via the two points above) by `theta_jn`, PORTED FROM
#     LALSUITE'S ACTUAL C SOURCE (not derived from memory -- see
#     `lalsimulation/lib/LALSimInspiral.c:6287-6294,6551-6553`):
#         pfac  = 0.5*(1 + cos(theta_jn)^2)
#         cfac  = cos(theta_jn)
#         hplus  = pfac * h
#         hcross = -im * cfac * h
#     At `theta_jn = 0` (face-on), `pfac = cfac = 1`, so `hplus == h` exactly. Validated
#     numerically against LALSuite's FD `hplus`/`hcross` output at several inclinations.
#   - Antenna response `(Fplus, Fcross)` projects `hplus`/`hcross` onto the single detector:
#     `h_det = Fplus*hplus + Fcross*hcross`. Computed as `combo = Fplus*pfac - im*Fcross*cfac` once
#     per walker (frequency-independent, like `_phi_coeffs`/`_ampl_coeffs`), so `h_det(f) = combo *
#     amp(f) * exp(im*phase(f))` needs only one extra complex multiply per frequency bin.
#
#     CORRECTNESS-CRITICAL: `Fplus`/`Fcross` are computed as plain scalar arithmetic INSIDE this
#     kernel body, as a function of that walker's own sampled `ra`/`dec`/`psi` (rows 8/9/11 of
#     `params`, which Enzyme differentiates through via `Duplicated(positions, grad_out)`) -- NOT
#     precomputed host-side into an `Enzyme.Const` array before the kernel launch. A host-
#     precomputed `Const` antenna response would be invisible to Enzyme's reverse pass, silently
#     zeroing `d(logL)/d(ra)`, `d(logL)/d(dec)`, `d(logL)/d(psi)` -- a correctness bug in exactly
#     the 3 parameters most worth getting right. Only `gps_time` (the event
#     trigger time, fixed data -- NOT `t_c`) and the detector's geometry are treated as compile-
#     time/host constants: `gmst_val = GravitationalWaveData.gmst(data.gps_time)` computed ONCE outside the kernel
#     launch (see `batched_log_likelihood_kernel!` below) and passed in as a plain scalar kernel
#     argument (not wrapped in `@Const`, matching this file's existing convention for scalar/
#     struct arguments like `model`).
#
#     The projection is `GravitationalWaveData.antenna_response_gmst` (port of `XLALComputeDetAMResponse`,
#     `lal/lib/tools/DetResponse.c`, Anderson et al. PRD 63 042003 (2001) eqs. B4/B5/B7), called on
#     the detector's isbits `GravitationalWaveData.DetectorGeometry`, never on the named `GravitationalWaveData.Detector`: the
#     latter's `String` `prefix` makes Enzyme fail with `EnzymeRuntimeActivityError: ... mismatched
#     activity between its rooted and non-rooted fields` (confirmed empirically). Geometry crosses
#     into GPU kernels Enzyme differentiates as a small `@Const` device array (`_geometry_array`,
#     rebuilt by `_geometry_from_array`), because a bare isbits struct argument is classified
#     Active there (see `_lik_array`).

# All of the above is plain non-allocating scalar arithmetic (dot products, trig, complex
# multiplies on scalars/tuples) -- consistent with this file's pre-existing Enzyme-vs-allocation
# discipline (see the `_phi_coeffs`/`_phi_scalar` split's docstring below): no array is allocated
# inside any `@kernel` body in this file.
# =================================================================================================

@inline function _chirp_mass_eta(m1, m2)
    mu = (m1 * m2) / (m1 + m2)
    mc = (mu / (m1 + m2))^(3 / 5) * (m1 + m2)
    eta = min((m1 * m2) / (m1 + m2)^2, one(m1) / 4)   # <= 0.25: near q = 1 rounding gives 0.25 + ulp, and sqrt(1 - 4 eta) throws on the device (R4)
    return mc, eta
end

# Guards every kernel body's `abs2(residual) / psd[i]` below against a literal `x/0` if a PSD
# estimate ever has an exact-zero bin -- the same style of guard as `RelativeBinning.jl`'s
# `_safe_ratio` (which handles `href_edges[end] == 0` from a reference-waveform amplitude underflow),
# applied to the full-grid likelihood kernels' own PSD-division sites. A real, non-pathological PSD (a Welch/median estimate off real detector noise) should never
# be EXACTLY `0.0`, but a frequency bin that has been notched/zeroed upstream, or an all-zero
# synthetic/placeholder PSD used in a test, both make `psd[i] == 0` a real possibility this codebase
# should not respond to with silent NaN/Inf poisoning of the summed likelihood (mirroring
# `_safe_ratio`'s reasoning: a bin with zero noise-weight should contribute exactly zero to the
# sum, which is also the physically correct infinite-noise/zero-information limit for that bin, not
# an undefined one). Branch-free (`ifelse`, not `if`) for the same Enzyme/KernelAbstractions-safety
# reason as `_clamp_unit`'s pole guard in `BatchedNUTS.jl` and `RelativeBinning.jl`'s `_safe_ratio`.
# Multiply by a GUARDED INVERSE: with `ifelse(denom == 0, 0, numer / denom)` Enzyme's adjoint of the
# division is `0 / 0 = NaN` in the selected-away lane, poisoning the whole gradient (seen for IMRPhenomXAS and
# IMRPhenomD with an exactly-zero PSD bin). Guarding the denominator instead (`numer / ifelse(z, 1, denom)`) does NOT
# work: LLVM legally folds it back to `numer / denom` before Enzyme runs. With `numer * w`, d/dnumer is the finite `w`
# and the `inv(0) = Inf` only lives in the inactive (data-only) weight. Can differ from `numer / denom` by 1 ulp.
@inline function _safe_div(numer::T, denom::T) where {T<:Real}
    return numer * ifelse(denom == zero(T), zero(T), inv(denom))
end

# Dead-zone hard-reject: every
# `SEOBNRv5AlignedSpinF2` likelihood kernel below builds its walker's post-adiabatic table once
# (`coeffs = _seobv5f2_coeffs(...)`) and then calls `_seobv5f2_ampl_scalar`/`_seobv5f2_phi_scalar`,
# which themselves already return an exact `0.0` for any queried frequency outside
# `[coeffs.fmin, coeffs.fmax]` (a deliberate, documented hard-cutoff convention -- see those
# functions' own docstrings in `GravitationalWaveforms.jl`). That convention is silently catastrophic when the
# table's `coeffs.fmax` never reaches the analysis band's own lowest frequency at all (confirmed by
# direct code execution against real GW150914 H1 run data: ~11/16 walkers in one run had
# `coeffs.fmax` around 1-3 Hz against a `20`Hz floor) -- `h` is then IDENTICALLY ZERO across the
# WHOLE frequency/bin grid, so the likelihood collapses to a FIXED noise-only constant
# (`dd_term`/`-2*df*sum(|strain|^2/psd)`) with ZERO gradient w.r.t. every parameter. NUTS sees this
# as a perfectly flat, valid-looking target and lets the walker random-walk freely with no pull back
# toward the signal region -- worse, `_seobv5f2_deadzone_boundary_map` shows this
# "dead zone" covers a LARGE, not narrow, fraction of this sampler's own `(mc, q)` prior at every
# spin tried. Fix: once `coeffs` is built, `_seobv5f2_apply_deadzone_reject` turns that flat,
# zero-gradient constant into a genuinely bad (but still finite) log-likelihood, so NUTS's own
# `max_delta_energy`-based divergence machinery (`NUTSCore.jl`) treats it as a hard rejection
# instead of a silently plausible flat region. Deliberately a large FINITE constant, not `-Inf`
# (unlike `BatchedNUTS.jl`'s `_log_prior` out-of-bounds convention) -- this can be reached mid-
# trajectory from an otherwise-valid point, and a finite penalty keeps every downstream delta-energy/
# Metropolis-style arithmetic ordinary IEEE-finite math (no `Inf - Inf -> NaN` risk to reason about),
# while still overwhelming `NUTSConfig`'s default `max_delta_energy=1000` by many orders of
# magnitude so it is unconditionally treated as a divergence. Branch-free (`ifelse` and bitwise `|`,
# not `if`/`||`) for the same Enzyme/KernelAbstractions-safety reason as `_safe_div` above --
# `coeffs.success`/`coeffs.fmax` are read, never assigned a placeholder tangent, so `ifelse`'s
# untaken branch contributes no gradient, exactly like `_log_prior`'s own bounds-indicator pattern.
const _SEOBV5F2_DEADZONE_LOGLIKELIHOOD = -1.0e6

@inline function _seobv5f2_apply_deadzone_reject(ll_val::T, coeffs, f_min_requested::T) where {T<:Real}
    dead = !coeffs.success | (coeffs.fmax < f_min_requested)
    return ifelse(dead, T(_SEOBV5F2_DEADZONE_LOGLIKELIHOOD), ll_val)
end

# -------------------------------------------------------------------------------------------------
# Analytic phase marginalization (`PEConfig.marginalize_phase`). For a waveform whose reference phase
# enters only as a common factor `h = h0 * exp(-i*phase)` (true of every dominant-mode-only kernel
# here: `phiRef` is subtracted from the phase of every bin of every detector), the likelihood
# averaged over a uniform phase prior is
#     (1/2pi) * int exp(-1/2 <d-h,d-h>) dphase = exp(-1/2(<h,h> + <d,d>)) * I0(|<d,h>|),
# with `<a,b> = 4 df sum conj(a) b / S`. `|<d,h>|` does not depend on `phase`, so kernels accumulate
# the usual residual sum AND `z = <d,h>` at whatever `phase` the walker holds, then combine here.
# `ll_plain = -1/2<d-h,d-h> = -1/2(<h,h>+<d,d>) + Re z`, so the marginal is `ll_plain - Re z + log I0(|z|)`.
# `ifelse` (not `if`) and the `max`/`min` guards inside `_log_besseli0` keep the untaken branch
# finite so no NaN leaks into the selected branch's gradient.
# -------------------------------------------------------------------------------------------------

# log(I0(x)) for x >= 0: Abramowitz & Stegun 9.8.1 (x < 3.75) and 9.8.2 (x >= 3.75), relative error
# on I0 below 2e-7 -- i.e. below 5e-7 absolute in the log-likelihood. Plain polynomials, so it runs
# (and differentiates) inside GPU kernels. The large-x form is written in log space: I0 itself
# overflows Float64 near x = 700, and real-signal |<d,h>| (~SNR^2) can exceed that.
@inline function _log_besseli0(x::T) where {T<:Real}
    xs = min(x, T(3.75))
    t = (xs / T(3.75))^2
    small = log(1 + t * (T(3.5156229) + t * (T(3.0899424) + t * (T(1.2067492) +
                t * (T(0.2659732) + t * (T(0.0360768) + t * T(0.0045813)))))))
    xl = max(x, T(3.75))
    u = T(3.75) / xl
    large = xl - T(0.5) * log(xl) + log(T(0.39894228) + u * (T(0.01328592) + u * (T(0.00225319) +
                u * (T(-0.00157565) + u * (T(0.00916281) + u * (T(-0.02057706) + u * (T(0.02635537) +
                u * (T(-0.01647633) + u * T(0.00392377)))))))))
    return ifelse(x < T(3.75), small, large)
end

# Likelihood-variant options as an isbits NamedTuple (kernel argument). `_LIK_PLAIN` = no marginalization.
_lik_opts(cfg::PEConfig) = (margphase=cfg.marginalize_phase, margdist=cfg.marginalize_distance,
                            dlo=Float64(cfg.distance_lo), dhi=Float64(cfg.distance_hi), dpow=cfg.distance_power,
                            fref=Float64(cfg.time_ref_frequency), toff=Float64(cfg.time_ref_offset),
                            taper=Float64(cfg.fmax_taper))
const _LIK_PLAIN = (margphase=false, margdist=false, dlo=0.05, dhi=4.0, dpow=0, fref=0.0, toff=0.0, taper=0.0)

# C1 end-of-table amplitude taper (`PEConfig.fmax_taper`): 1 below `fmax - width`, sin^2 down to 0 at
# `fmax`; identically 1 when `width == 0`. Continuous with continuous derivative in both `f` and `fmax`.
@inline function _end_taper(f::T, fmax, width) where {T}
    xc = clamp((fmax - f) / max(width, T(1e-12)), zero(T), one(T))
    return ifelse(width > 0, sin(T(pi) / 2 * xc)^2, one(T))
end

# Kernels receive the options as a 7-element Float64 array on the kernel's backend (built by
# `_lik_array`) and rebuild the NamedTuple inside: a bare isbits struct with Float64 fields passed to a
# kernel that Enzyme differentiates on CUDA is classified Active and rejected ("Active kernel
# arguments not supported on GPU") -- the same reason response/gmst are passed as small Const arrays.
function _lik_array(cfg::PEConfig, backend)
    o = _lik_opts(cfg)
    a = KernelAbstractions.allocate(backend, Float64, 8)
    copyto!(a, Float64[o.margphase, o.margdist, o.dlo, o.dhi, o.dpow, o.fref, o.toff, o.taper])
    return a
end
# Per-detector geometry (`DetectorGeometry`: 3 location + 9 response entries) flattened into one small
# device array, 12 Float64 per detector: a bare isbits struct or tuple of Float64 is classified Active
# by Enzyme's CUDA rule, like `_lik_array`'s options. Rebuilt inside the kernel by `_geometry_from_array`
# (one detector) or `_geometry_tuple` (detector count from the compile-time length of `freqs_tuple`).
_geometry_array(data, backend) = _geometry_array(Tuple(geometry(d.detector) for d in data), backend)
function _geometry_array(dets::Tuple{Vararg{DetectorGeometry}}, backend)
    a = KernelAbstractions.allocate(backend, Float64, 12 * length(dets))
    copyto!(a, collect(Float64, Iterators.flatten((g.location..., Iterators.flatten(g.response)...) for g in dets)))
    return a
end
@inline function _geometry_from_array(a, d::Int)
    o = 12 * (d - 1)
    return DetectorGeometry((a[o + 1], a[o + 2], a[o + 3]),
                            ((a[o + 4], a[o + 5], a[o + 6]), (a[o + 7], a[o + 8], a[o + 9]), (a[o + 10], a[o + 11], a[o + 12])))
end
@inline _geometry_tuple(a, ::NTuple{N,Any}) where {N} = ntuple(d -> _geometry_from_array(a, d), Val(N))

@inline _lik_from_array(a) = (margphase=a[1] != 0, margdist=a[2] != 0, dlo=a[3], dhi=a[4],
                              dpow=unsafe_trunc(Int, a[5]), fref=a[6], toff=a[7], taper=a[8])

# Top of the (t, phase, omega) SPA table: the table's `fmax` for the inspiral-only PA-F2 models; the
# IMR table's SPA part ends at the blend band (`fspa_max`), above which it is a Fourier table.
@inline _spa_fmax(c) = c.fmax
@inline _spa_fmax(c::EOBIMRTable) = c.fspa_max
# Time origin of the table's phase: the start of the PA table for the inspiral-only models, t_attach
# (~ the (2,2) peak) for the IMR table.
@inline _spa_t0(c) = 0.0
@inline _spa_t0(c::EOBIMRTable) = c.t_a

"""
    _seobv5f2_time_shift(coeffs, lik) -> Float64

`t*(f_ref) + offset` when `PEConfig.time_ref_frequency > 0`, else `0`: the amount subtracted from the
sampled time row to recover the coalescence time. `t*` is the table's own stationary-phase time at
which the GW frequency equals `f_ref` (clamped into the table's band), a smooth function of the table
entries -- so Enzyme differentiates it into the coeffs cotangents and the hand VJP carries it to
(mc, eta, chi1, chi2) like every other coeffs dependence.
"""
@inline function _seobv5f2_time_shift(coeffs, lik)
    (lik.fref > 0 && coeffs.success) || return 0.0
    fr = clamp(lik.fref, coeffs.fmin, _spa_fmax(coeffs))
    r = _seobv5f2_dense_to_freq(coeffs.ts, coeffs.phis, coeffs.omegas, coeffs.nreal, coeffs.M_sec, fr)
    return (r.t_star - _spa_t0(coeffs)) * coeffs.M_sec + lik.toff
end

"""
    _xas_time_shift(pc, lik) -> Float64

IMRPhenomXAS analogue of [`_seobv5f2_time_shift`](@ref): `t*(f_ref) + offset` when
`PEConfig.time_ref_frequency > 0`, else `0`. `pc` are the INTRINSIC phase coefficients
(`_xas_phi_coeffs` with `t0 = phiRef = 0`). In the kernel convention `h = A exp(i (Phi(f) - 2 pi f t_c))`
the stationary-phase time at frequency `f` is `t_c - Phi'(f) / 2 pi`, so `t* = -Phi'(f_ref) / 2 pi`,
evaluated from XAS's own analytic phase derivatives (`_xas_phi_*_deriv`, the same pieces its C1 joins
use) -- a smooth function of (mc, eta, chi1, chi2) that Enzyme differentiates.
"""
@inline function _xas_time_shift(pc, lik)
    lik.fref > 0 || return 0.0
    msec = pc.M * pc.GMsun_over_c3
    x = msec * lik.fref
    p = pc.phi_ins; b = pc.phi_int; c = pc.phi_mrd
    dins = _xas_phi_ins_deriv(x, p.phi0, p.phi1, p.phi2, p.phi3, p.phi4, p.phi5, p.phi5L, p.phi6, p.phi6L,
                              p.phi7, p.phi8, p.phi8L, p.sigma1, p.sigma2, p.sigma3, p.sigma4, p.phiN)
    dint = _xas_phi_int_deriv(x, b.b0, b.b1, b.b2, b.b3, b.b4, c.cL, pc.fdamp, pc.fRD) + b.alpha1
    dmrd = _xas_phi_mrd_deriv(x, c.c0, c.c1, c.c2, c.c4, c.cL, pc.fdamp, pc.fRD) + c.beta1
    dphi_dx = ifelse(x < pc.f1, dins, ifelse(x < pc.f2, dint, dmrd)) / pc.eta
    return -msec * dphi_dx / (2 * pi) + lik.toff
end

# Intrinsic XAS phase and amplitude at one frequency: `_xas_phi_scalar(pc, f)` (t0 = phiRef = 0) and
# `_xas_ampl_scalar(ac, f)` bit for bit, but with `if` branches instead of nested `ifelse`, so only the
# segment containing `f` is evaluated (`ifelse` evaluates the inspiral, intermediate and merger-ringdown
# forms at every bin; the inspiral one, the most expensive, is needed for only ~10-20% of a 20-512 Hz grid).
@inline function _xas_phase_amp(pc, ac, f)
    fs = max(f, eps(typeof(f)))
    x = pc.M * pc.GMsun_over_c3 * fs
    p = pc.phi_ins; b = pc.phi_int; c = pc.phi_mrd
    phis = if x < pc.f1
        _xas_phi_ins_val(x, p.phi0, p.phi1, p.phi2, p.phi3, p.phi4, p.phi5, p.phi5L, p.phi6, p.phi6L,
                         p.phi7, p.phi8, p.phi8L, p.sigma1, p.sigma2, p.sigma3, p.sigma4, p.phiN)
    elseif x < pc.f2
        _xas_phi_int_val(x, b.b0, b.b1, b.b2, b.b3, b.b4, c.cL, pc.fdamp, pc.fRD) + b.alpha1 * x + b.alpha0
    elseif x < pc.fcutPar
        _xas_phi_mrd_val(x, c.c0, c.c1, c.c2, c.c4, c.cL, pc.fdamp, pc.fRD) + c.beta0 + c.beta1 * x
    else
        zero(x)
    end / pc.eta
    xa = ac.M * ac.GMsun_over_c3 * fs
    ai = ac.amp_ins; an = ac.amp_int; ar = ac.amp_rd
    aimr = if xa < ac.fA1
        _xas_amp_ins_val(xa, ai.A2, ai.A3, ai.A4, ai.A5, ai.A6, ai.rho1, ai.rho2, ai.rho3)
    elseif xa < ac.fA4
        xa^(7.0 / 6.0) / (an.delta0 + xa * (an.delta1 + xa * (an.delta2 + xa * (an.delta3 + xa * an.delta4))))
    elseif xa < ac.fcutPar
        _xas_amp_rd_val(xa, ar.gamma1, ar.gamma2, ar.gamma3, ac.fdamp, ac.fRD)
    else
        zero(xa)
    end
    return phis, ac.Overallamp * ac.amp0 * (xa^(-7.0 / 6.0)) * aimr
end

# One detector's IMRPhenomXAS sums for the likelihood options: `sum |h - d|^2 / S`, `sum |h|^2 / S` and
# `sum conj(d) h / S` (real, imaginary), with `h = combo * A(f) * exp(i (Phi(f) - 2 pi f tsec - phiRef))`
# built from the intrinsic coefficients `pc`/`ac` (the time/phase term applied below the XAS cutoff,
# exactly as `_xas_phi_scalar` applies its own `t0`/`phiRef`; the amplitude is zero above it).
@inline function _xas_det_sums(pc, ac, combo, tsec::T, phiRef::T, freqs, strain, psd) where {T}
    msec = pc.M * pc.GMsun_over_c3
    acc = zero(T); hh = zero(T); zr = zero(T); zi = zero(T)
    @inbounds for i in eachindex(freqs)
        f = freqs[i]
        phi0, amp = _xas_phase_amp(pc, ac, f)
        ph = phi0 + ifelse(msec * f < pc.fcutPar, -2 * T(pi) * tsec * f - phiRef, zero(T))
        h = combo * (amp * Complex(cos(ph), sin(ph)))
        acc += _safe_div(abs2(h - strain[i]), psd[i])
        hh += _safe_div(abs2(h), psd[i])
        sh = conj(strain[i]) * h
        zr += _safe_div(real(sh), psd[i])
        zi += _safe_div(imag(sh), psd[i])
    end
    return acc, hh, zr, zi
end

# Whole-walker IMRPhenomXAS log-likelihood with the likelihood options, from PHYSICAL parameters: the
# body of the XAS kernels below, and the scalar function `BatchedNUTS` differentiates per walker on the
# CPU (much cheaper than Enzyme through the KernelAbstractions launch).
@inline function _xas_ll_given_physical(model, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi,
                                        det, gmst_val, freqs, strain, psd, lik)
    T = typeof(m1)
    mc, eta = _chirp_mass_eta(m1, m2)
    phiRef, combo = _extrinsic_setup_phase_only(phase, ra, dec, theta_jn, psi, det, gmst_val)
    pc = _xas_phi_coeffs(model, mc, eta, chi1, chi2)
    ac = _xas_ampl_coeffs(model, mc, eta, chi1, chi2, dL)
    tsec = t_c - _xas_time_shift(pc, lik)
    acc, hh, zr, zi = _xas_det_sums(pc, ac, combo, tsec, phiRef, freqs, strain, psd)
    df = freqs[2] - freqs[1]
    return _marg_combine(T(-2) * df * acc, T(4) * df * hh, T(4) * df * zr, T(4) * df * zi, dL, lik)
end

# One XAS detector-frame strain value, the dense kernels' convention (`_xas_det_sums`).
@inline function _xas_h(pc, ac, combo, tsec::T, phiRef::T, f) where {T}
    phi0, amp = _xas_phase_amp(pc, ac, f)
    ph = phi0 + ifelse(pc.M * pc.GMsun_over_c3 * f < pc.fcutPar, -2 * T(pi) * tsec * f - phiRef, zero(T))
    return combo * (amp * Complex(cos(ph), sin(ph)))
end

@inline function _rb_ratio(num::Complex{T}, den::Complex{S}) where {T,S}  # guarded inverse: see `_safe_div`
    return num * ifelse(abs2(den) == zero(S), zero(Complex{S}), inv(den))
end

"""
    _xas_ll_rb(model, m1, ..., psi, rb, lik) -> Float64

Relative-binning IMRPhenomXAS log-likelihood WITH the likelihood options: the same waveform,
time frame (`rb.geo`, `rb.tref`, f_ref timing) and `_marg_combine` as the dense `_xas_ll_given_physical[_multi]`,
but `h` is evaluated only at each detector's bin edges and `<d,h>`, `<h,h>` come from the reference's
per-bin summary data (`rb`, built by `BatchedNUTS.build_xas_relative_binning`): with `r = h/h_ref` linear
in each bin (`r0` at the bin centre, slope `r1`), `sum conj(h) d w = sum A0 conj(r0) + A1 conj(r1)` and
`sum |h|^2 w = sum B0 |r0|^2 + 2 B1 Re(r0 conj(r1))`.
"""
@inline function _xas_ll_rb(model, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi, rb, lik)
    T = typeof(m1)
    mc, eta = _chirp_mass_eta(m1, m2)
    pfac = T(0.5) * (1 + cos(theta_jn) * cos(theta_jn))
    cfac = cos(theta_jn)
    pc = _xas_phi_coeffs(model, mc, eta, chi1, chi2)
    ac = _xas_ampl_coeffs(model, mc, eta, chi1, chi2, dL)
    g = rb.gmst
    dt_ref = ifelse(rb.geo & (rb.tref > 0), time_delay_from_geocenter_gmst(rb.dets[max(rb.tref, 1)], ra, dec, g), zero(T))
    t_c = t_c - _xas_time_shift(pc, lik)
    ll_plain = zero(T); hh = zero(T); zr = zero(T); zi = zero(T)
    ndet = length(rb.edges)
    @inbounds for d in 1:ndet
        dt_geo = ifelse(rb.geo, time_delay_from_geocenter_gmst(rb.dets[d], ra, dec, g) - dt_ref, zero(T))
        Fplus, Fcross = antenna_response_gmst(rb.dets[d], ra, dec, psi, g)
        combo = Fplus * pfac - im * Fcross * cfac
        tsec = t_c + dt_geo
        edges = rb.edges[d]; href = rb.href[d]; A0 = rb.A0[d]; A1 = rb.A1[d]; B0 = rb.B0[d]; B1 = rb.B1[d]
        r_lo = _rb_ratio(_xas_h(pc, ac, combo, tsec, phase, edges[1]), href[1])
        dh = zero(Complex{T}); hs = zero(T)
        for b in eachindex(A0)
            r_hi = _rb_ratio(_xas_h(pc, ac, combo, tsec, phase, edges[b+1]), href[b+1])
            r0 = T(0.5) * (r_lo + r_hi)
            r1 = (r_hi - r_lo) / (edges[b+1] - edges[b])
            dh += A0[b] * conj(r0) + A1[b] * conj(r1)
            hs += B0[b] * abs2(r0) + 2 * B1[b] * real(r0 * conj(r1))
            r_lo = r_hi
        end
        ll_plain += rb.dd[d] + real(dh) - T(0.5) * hs
        hh += hs
        zr += real(dh)
        zi -= imag(dh)
    end
    return _marg_combine(ll_plain, hh, zr, zi, dL, lik)
end

# Several detectors. `tref == 0`: t_c is geocentric; `tref == k`: t_c is the arrival time at detector k.
# The intrinsic coefficients are built once per walker, not per detector.
@inline function _xas_ll_given_physical_multi(model, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi,
                                              freqs_tuple, strain_tuple, psd_tuple, det_tuple,
                                              gmst_val, lik, tref::Int)
    T = typeof(m1)
    mc, eta = _chirp_mass_eta(m1, m2)
    phiRef = phase
    pfac = T(0.5) * (1 + cos(theta_jn) * cos(theta_jn))
    cfac = cos(theta_jn)
    pc = _xas_phi_coeffs(model, mc, eta, chi1, chi2)
    ac = _xas_ampl_coeffs(model, mc, eta, chi1, chi2, dL)
    dt_ref = ifelse(tref == 0, zero(T), time_delay_from_geocenter_gmst(det_tuple[max(tref, 1)], ra, dec, gmst_val))
    t_c = t_c - _xas_time_shift(pc, lik)
    ll_plain = zero(T); hh = zero(T); zr = zero(T); zi = zero(T)
    ndet = length(freqs_tuple)
    @inbounds for d in 1:ndet
        dt_geo = time_delay_from_geocenter_gmst(det_tuple[d], ra, dec, gmst_val) - dt_ref
        Fplus, Fcross = antenna_response_gmst(det_tuple[d], ra, dec, psi, gmst_val)
        combo = Fplus * pfac - im * Fcross * cfac
        a, h2, r, i = _xas_det_sums(pc, ac, combo, t_c + dt_geo, phiRef, freqs_tuple[d], strain_tuple[d], psd_tuple[d])
        df_d = freqs_tuple[d][2] - freqs_tuple[d][1]
        ll_plain += T(-2) * df_d * a
        hh += T(4) * df_d * h2
        zr += T(4) * df_d * r
        zi += T(4) * df_d * i
    end
    return _marg_combine(ll_plain, hh, zr, zi, dL, lik)
end

# IMRPhenomXPHM precession setup for the model's LAL precession version (`IMRPhenomXPHM{V}`: 102 NNLO, 223 MSA):
# frame, Euler angles (`GravitationalWaveforms._xp_euler_angles`) and the co-precessing final spin (`setup.final_spin`).
@inline _xphm_prec_setup(::IMRPhenomXPHM{V}, args...; GMsun_over_c3) where {V} =
    _xp_precession_setup(args...; GMsun_over_c3 = GMsun_over_c3, prec_version = Val(V))

# ---- IMRPhenomXP / IMRPhenomXPHM with the likelihood options ------------------------------------------
# Distance marginalization, f_ref timing and the detector time reference (NOT phase marginalization: the phase enters
# the precession setup, it is not an overall phase). Same waveforms and time handling as the dense XP / XPHM kernels
# (`_fused_waveform_likelihood_kernel_phenomxp[hm][_multi]!`), with t_c replaced by t_c - t_shift (+ the detector delay
# relative to the reference detector), so with no options the values equal those kernels. t_shift(f_ref) is the
# co-precessing (2,2) mode's: XAS's `_xas_time_shift` (XP) / `_xhm_time_shift` (XPHM) -- the same time frame as an
# XAS / XHM pass 1.
@inline function _xprec_state(::IMRPhenomXP, m1, m2, chi1, chi2, dL, phase, theta_jn, chi1x, chi1y, chi2x, chi2y, f_ref, lik,
                              freqs = nothing)
    T = typeof(m1)
    mc, eta = _chirp_mass_eta(m1, m2)
    M_sec = (mc / eta^T(0.6)) * T(GMsun_over_c3)
    setup = _xp_precession_setup(m1, m2, chi1x, chi1y, chi1, chi2x, chi2y, chi2,
                                  f_ref, theta_jn, phase; GMsun_over_c3 = T(GMsun_over_c3))
    ac = _xas_ampl_coeffs(IMRPhenomXAS(), mc, eta, chi1, chi2, dL)
    tshift = _xas_time_shift(_xas_phi_coeffs(IMRPhenomXAS(), mc, eta, chi1, chi2), lik)
    return (; mc, eta, chi1, chi2, M_sec, phase, setup, ac, tshift)
end
@inline function _xprec_state(model::IMRPhenomXPHM, m1, m2, chi1, chi2, dL, phase, theta_jn, chi1x, chi1y, chi2x, chi2y, f_ref, lik,
                              freqs = nothing)
    T = typeof(m1)
    mc, eta = _chirp_mass_eta(m1, m2)
    setup = _xphm_prec_setup(model, m1, m2, chi1x, chi1y, chi1, chi2x, chi2y, chi2,
                                  f_ref, theta_jn, phase; GMsun_over_c3 = T(GMsun_over_c3))
    final_spin = setup.final_spin   # version-dependent (102: chi_p, 223: MSA)
    sys = _xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, zero(phase);
                             GMsun_over_c3 = T(GMsun_over_c3), GMsun_over_c2_Gpc = T(GMsun_over_c2_Gpc),
                             final_spin_override = final_spin)
    ms21 = _xhm_mode_setup(2, 1, sys); ms33 = _xhm_mode_setup(3, 3, sys); ms44 = _xhm_mode_setup(4, 4, sys)
    m32 = _xhm_mode32_setup(sys)
    return (; setup, sys, ms21, ms33, ms44, m32, tshift = _xhm_time_shift(sys, lik))
end
# SEOBNRv5PHM (CPU reference path): the polarizations on the (shared) data frequency grid, once per walker
# (`GravitationalWaveforms.hphc`: FFT of the time-domain model, t = 0 at the frame-invariant amplitude peak; waveform from
# 0.9 f_min, phase and frame referred to f_ref = f_min, the grid's first frequency, as the XP / XPHM rows).
function _xprec_state(::SEOBNRv5PHM, m1, m2, chi1, chi2, dL, phase, theta_jn, chi1x, chi1y, chi2x, chi2y, f_ref, lik, freqs)
    lik.fref > 0 && throw(ArgumentError("SEOBNRv5PHM CPU reference path: time_ref_frequency is device-only " *
                                        "(`seobnr_phm_device_loglikelihood!`)"))
    hp, hc = GravitationalWaveforms.hphc(SEOBNRv5PHM(), freqs, m1, m2, (chi1x, chi1y, chi1), (chi2x, chi2y, chi2), dL, theta_jn, phase;
        f_min = 0.9 * f_ref, f_ref = f_ref)
    return (; hp, hc, tshift = zero(m1))
end
function _xprec_det_sums(::SEOBNRv5PHM, st, Fplus, Fcross, tsec::T, freqs, strain, psd) where {T}
    length(freqs) == length(st.hp) || throw(ArgumentError("SEOBNRv5PHM: all detectors must share one frequency grid"))
    acc = zero(T); hh = zero(T); zr = zero(T); zi = zero(T)
    @inbounds for i in eachindex(freqs)
        tc_phase = -2 * T(pi) * freqs[i] * tsec   # time sign: see _PHASE_CONVENTION
        h = (Fplus * st.hp[i] + Fcross * st.hc[i]) * Complex{T}(cos(tc_phase), sin(tc_phase))
        acc += _safe_div(abs2(h - strain[i]), psd[i])
        hh += _safe_div(abs2(h), psd[i])
        sh = conj(strain[i]) * h
        zr += _safe_div(real(sh), psd[i])
        zi += _safe_div(imag(sh), psd[i])
    end
    return acc, hh, zr, zi
end

# The detector template at frequency f for arrival time tsec (`pre = _xprec_det_pre(model, st, tsec)`, per detector),
# and one detector's sums (`sum |h - d|^2 / S`, `sum |h|^2 / S`, `sum conj(d) h / S`); `xprec_strain` evaluates the
# same template for injections.
@inline _xprec_det_pre(::IMRPhenomXP, st, tsec::T) where {T} =
    _xas_phi_coeffs(IMRPhenomXAS(), st.mc, st.eta, st.chi1, st.chi2; t0 = 2 * T(pi) * tsec / st.M_sec,
                    phiRef = st.phase)    # time sign: see _PHASE_CONVENTION (the dense XP kernels' t0 = -t0)
@inline _xprec_det_pre(::IMRPhenomXPHM, st, tsec) = nothing
@inline function _xprec_h_bin(::IMRPhenomXP, st, pc, Fplus, Fcross, tsec::T, f) where {T}
    amp = _xas_ampl_scalar(st.ac, f)
    ph = _xas_phi_scalar(pc, f)
    hAS = (amp * Complex(cos(ph), sin(ph))) * T(_XP_HAS_MODE_NORMALIZATION)
    hp, hc = _xp_twist_up22(cbrt(T(pi) * st.M_sec * f), st.setup, hAS)
    hp, hc = _xp_apply_zeta_rotation(hp, hc, st.setup.zeta_polarization)
    return Fplus * hp + Fcross * hc
end
@inline function _xprec_h_bin(::IMRPhenomXPHM, st, _, Fplus, Fcross, tsec::T, f) where {T}
    hp, hc = _xphm_hphc_scalar(f, st.setup, st.sys, st.ms21, st.ms33, st.ms44, st.m32)
    hp, hc = _xp_apply_zeta_rotation(hp, hc, st.setup.zeta_polarization)
    tc_phase = -2 * T(pi) * f * tsec   # time sign: see _PHASE_CONVENTION
    return (Fplus * hp + Fcross * hc) * Complex{T}(cos(tc_phase), sin(tc_phase))
end
@inline function _xprec_det_sums(model::Union{IMRPhenomXP,IMRPhenomXPHM}, st, Fplus, Fcross, tsec::T, freqs, strain,
                                 psd) where {T}
    pre = _xprec_det_pre(model, st, tsec)
    acc = zero(T); hh = zero(T); zr = zero(T); zi = zero(T)
    @inbounds for i in eachindex(freqs)
        h = _xprec_h_bin(model, st, pre, Fplus, Fcross, tsec, freqs[i])
        acc += _safe_div(abs2(h - strain[i]), psd[i])
        hh += _safe_div(abs2(h), psd[i])
        sh = conj(strain[i]) * h
        zr += _safe_div(real(sh), psd[i])
        zi += _safe_div(imag(sh), psd[i])
    end
    return acc, hh, zr, zi
end
# Whole-walker log-likelihood from PHYSICAL parameters. `geo = false`: one detector, t_c is its arrival time (the
# single-detector kernels); `geo = true`: t_c geocentric (`tref = 0`) or at detector `tref`.
@inline function _xprec_ll_given_physical(model, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi,
                                          chi1x, chi1y, chi2x, chi2y, freqs_tuple, strain_tuple, psd_tuple,
                                          det_tuple, gmst_val, lik, tref::Int, geo::Bool)
    T = typeof(m1)
    st = _xprec_state(model, m1, m2, chi1, chi2, dL, phase, theta_jn, chi1x, chi1y, chi2x, chi2y, freqs_tuple[1][1], lik,
                      freqs_tuple[1])
    dt_ref = ifelse(geo & (tref > 0), time_delay_from_geocenter_gmst(det_tuple[max(tref, 1)], ra, dec, gmst_val), zero(T))
    t_c = t_c - st.tshift
    ll_plain = zero(T); hh = zero(T); zr = zero(T); zi = zero(T)
    @inbounds for d in 1:length(freqs_tuple)
        dt_geo = ifelse(geo, time_delay_from_geocenter_gmst(det_tuple[d], ra, dec, gmst_val) - dt_ref, zero(T))
        Fplus, Fcross = antenna_response_gmst(det_tuple[d], ra, dec, psi, gmst_val)
        a, h2, r, i = _xprec_det_sums(model, st, Fplus, Fcross, t_c + dt_geo, freqs_tuple[d], strain_tuple[d], psd_tuple[d])
        df_d = freqs_tuple[d][2] - freqs_tuple[d][1]
        ll_plain += T(-2) * df_d * a
        hh += T(4) * df_d * h2
        zr += T(4) * df_d * r
        zi += T(4) * df_d * i
    end
    return _marg_combine(ll_plain, hh, zr, zi, dL, lik)
end
"""
    xprec_strain(model::Union{IMRPhenomXP,IMRPhenomXPHM}, params, cfg, data) -> Vector{Vector{ComplexF64}}

The detector templates the IMRPhenomXP / XPHM likelihood (`_xprec_ll_given_physical`) compares with the data, for ONE
parameter vector `params` (15 physical rows, the time row in the likelihood's frame: `cfg.time_reference` /
`time_ref_frequency`; phase and spins at the grid's first frequency) on each detector of `data` (`LikelihoodData`s; their
strain is not used): a zero-noise injection of the result has logL = 0 at `params`.
"""
function xprec_strain(model::Union{IMRPhenomXP,IMRPhenomXPHM}, params::AbstractVector, cfg::PEConfig, data::AbstractVector)
    p = Float64.(collect(params))
    freqs1 = Array(data[1].freqs)
    lik = _lik_opts(cfg)
    st = _xprec_state(model, p[1], p[2], p[3], p[4], p[5], p[7], p[10], p[12], p[13], p[14], p[15], freqs1[1], lik, freqs1)
    gmst_val = gmst(data[1].gps_time)
    geo = length(data) > 1
    tref = cfg.time_reference
    ra, dec, psi = p[8], p[9], p[11]
    dt_ref = geo && tref > 0 ? time_delay_from_geocenter_gmst(data[tref].detector, ra, dec, gmst_val) : 0.0
    t_c = p[6] - st.tshift
    return map(data) do d
        dt_geo = geo ? time_delay_from_geocenter_gmst(d.detector, ra, dec, gmst_val) - dt_ref : 0.0
        Fplus, Fcross = antenna_response_gmst(d.detector, ra, dec, psi, gmst_val)
        tsec = t_c + dt_geo
        pre = _xprec_det_pre(model, st, tsec)
        ComplexF64[_xprec_h_bin(model, st, pre, Fplus, Fcross, tsec, f) for f in Array(d.freqs)]
    end |> collect
end
@kernel function _xprec_opts_kernel!(ll, @Const(params), @Const(freqs), @Const(strain), @Const(psd), @Const(geom_flat),
                                     @Const(gmst_arr), model, @Const(lik_arr))
    j = @index(Global, Linear)
    lik = _lik_from_array(lik_arr)
    det = _geometry_from_array(geom_flat, 1)
    ll[j] = _xprec_ll_given_physical(model, params[1, j], params[2, j], params[3, j], params[4, j], params[5, j],
                                     params[6, j], params[7, j], params[8, j], params[9, j], params[10, j], params[11, j],
                                     params[12, j], params[13, j], params[14, j], params[15, j],
                                     (freqs,), (strain,), (psd,), (det,), gmst_arr[1], lik, 0, false)
end
@kernel function _xprec_opts_multi_kernel!(ll, @Const(params), freqs_tuple, strain_tuple, psd_tuple, det_tuple, gmst_val, model, @Const(lik_arr), tref)
    j = @index(Global, Linear)
    lik = _lik_from_array(lik_arr)
    ll[j] = _xprec_ll_given_physical(model, params[1, j], params[2, j], params[3, j], params[4, j], params[5, j],
                                     params[6, j], params[7, j], params[8, j], params[9, j], params[10, j], params[11, j],
                                     params[12, j], params[13, j], params[14, j], params[15, j],
                                     freqs_tuple, strain_tuple, psd_tuple, det_tuple, gmst_val,
                                     lik, tref, true)
end

# ---- IMRPhenomD with the likelihood options ------------------------------------------------------
# Distance and phase marginalization, f_ref timing and the detector time reference, as for XAS. The waveform is the
# dense PhenomD kernels': h = combo * A(f) * exp(-i Phi(f)), Phi = Phi0(f) + 2 pi f t_c - phase below the cutoff
# (`_phi_scalar` with t0 = -2 pi t_c / M_sec, phiRef = phase; the conjugate: see _PHASE_CONVENTION), so with no
# options the value equals the plain PhenomD kernels.

# d Phi0 / d(M f) of `_phi_scalar` (t0 = phiRef = 0), region by region.
@inline function _pd_dphi_dx(pc, x)
    T = typeof(x)
    if x < pc.fInsJoin
        c = pc.PhiInspcoeffs
        x13 = cbrt(x); x23 = x13 * x13; lx = log(T(pi) * x)
        return c.two_thirds * (2 / T(3)) / x13 + c.third / (3 * x23) +
               c.third_log * (lx / (9 * x23) + 1 / (3 * x23)) + c.log / (3 * x) -
               c.min_third / (3 * x * x13) - c.min_two_thirds * (2 / T(3)) / (x * x23) - c.min_one / (x * x) -
               c.min_four_thirds * (4 / T(3)) / (x * x * x13) - c.min_five_thirds * (5 / T(3)) / (x * x * x23) +
               (c.one + c.four_thirds * (4 / T(3)) * x13 + c.five_thirds * (5 / T(3)) * x23 + 2 * c.two * x) * pc.etaInv
    elseif x < pc.fMRDJoin
        return (pc.beta1 + pc.beta3 / (x * x * x * x) + pc.beta2 / x) * pc.etaInv + pc.C2Int
    else
        u = (x - pc.alpha5 * pc.fring) / pc.fdamp
        return (pc.alpha2 / (x * x) + pc.alpha3 / sqrt(sqrt(x)) + pc.alpha1 + pc.alpha4 / (pc.fdamp * (1 + u * u))) *
               pc.etaInv + pc.C2MRD
    end
end
# t_c -> t_c - shift: with f_ref timing, t_c is the time at which frequency f_ref arrives, plus toff (PhenomD's phase
# enters as exp(-i Phi0): the opposite sign of `_xas_time_shift`; checked against a finite difference of the phase).
@inline function _pd_time_shift(pc, lik)
    lik.fref > 0 || return 0.0
    msec = pc.M * pc.GMsun_over_c3
    return msec * _pd_dphi_dx(pc, msec * lik.fref) / (2 * pi) + lik.toff
end
@inline function _pd_h(pc, ac, combo, tsec::T, phase::T, f) where {T}
    msec = pc.M * pc.GMsun_over_c3
    ph = _phi_scalar(pc, f) + ifelse(msec * f < pc.fcutPar, 2 * T(pi) * f * tsec - phase, zero(T))
    return combo * (_ampl_scalar(ac, f) * Complex(cos(ph), -sin(ph)))
end
@inline function _pd_det_sums(pc, ac, combo, tsec::T, phase::T, freqs, strain, psd) where {T}
    acc = zero(T); hh = zero(T); zr = zero(T); zi = zero(T)
    @inbounds for i in eachindex(freqs)
        h = _pd_h(pc, ac, combo, tsec, phase, freqs[i])
        acc += _safe_div(abs2(h - strain[i]), psd[i])
        hh += _safe_div(abs2(h), psd[i])
        sh = conj(strain[i]) * h
        zr += _safe_div(real(sh), psd[i])
        zi += _safe_div(imag(sh), psd[i])
    end
    return acc, hh, zr, zi
end
@inline function _pd_ll_given_physical(model, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi,
                                       det, gmst_val, freqs, strain, psd, lik)
    T = typeof(m1)
    mc, eta = _chirp_mass_eta(m1, m2)
    _, combo = _extrinsic_setup_phase_only(phase, ra, dec, theta_jn, psi, det, gmst_val)
    pc = _phi_coeffs(model, mc, eta, chi1, chi2)
    ac = _ampl_coeffs(model, mc, eta, chi1, chi2, dL)
    tsec = t_c - _pd_time_shift(pc, lik)
    acc, hh, zr, zi = _pd_det_sums(pc, ac, combo, tsec, phase, freqs, strain, psd)
    df = freqs[2] - freqs[1]
    return _marg_combine(T(-2) * df * acc, T(4) * df * hh, T(4) * df * zr, T(4) * df * zi, dL, lik)
end
@inline function _pd_ll_given_physical_multi(model, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi,
                                             freqs_tuple, strain_tuple, psd_tuple, det_tuple,
                                             gmst_val, lik, tref::Int)
    T = typeof(m1)
    mc, eta = _chirp_mass_eta(m1, m2)
    pc = _phi_coeffs(model, mc, eta, chi1, chi2)
    ac = _ampl_coeffs(model, mc, eta, chi1, chi2, dL)
    dt_ref = ifelse(tref == 0, zero(T), time_delay_from_geocenter_gmst(det_tuple[max(tref, 1)], ra, dec, gmst_val))
    t_c = t_c - _pd_time_shift(pc, lik)
    ll_plain = zero(T); hh = zero(T); zr = zero(T); zi = zero(T)
    @inbounds for d in 1:length(freqs_tuple)
        dt_geo = time_delay_from_geocenter_gmst(det_tuple[d], ra, dec, gmst_val) - dt_ref
        _, combo = _extrinsic_setup_phase_only(phase, ra, dec, theta_jn, psi, det_tuple[d], gmst_val)
        a, h2, r, i = _pd_det_sums(pc, ac, combo, t_c + dt_geo, phase, freqs_tuple[d], strain_tuple[d], psd_tuple[d])
        df_d = freqs_tuple[d][2] - freqs_tuple[d][1]
        ll_plain += T(-2) * df_d * a
        hh += T(4) * df_d * h2
        zr += T(4) * df_d * r
        zi += T(4) * df_d * i
    end
    return _marg_combine(ll_plain, hh, zr, zi, dL, lik)
end
@kernel function _pd_opts_kernel!(ll, @Const(params), @Const(freqs), @Const(strain), @Const(psd), @Const(geom_flat),
                                  @Const(gmst_arr), model, @Const(lik_arr))
    j = @index(Global, Linear)
    lik = _lik_from_array(lik_arr)
    det = _geometry_from_array(geom_flat, 1)
    ll[j] = _pd_ll_given_physical(model, params[1, j], params[2, j], params[3, j], params[4, j], params[5, j],
                                  params[6, j], params[7, j], params[8, j], params[9, j], params[10, j], params[11, j],
                                  det, gmst_arr[1], freqs, strain, psd, lik)
end
@kernel function _pd_opts_multi_kernel!(ll, @Const(params), freqs_tuple, strain_tuple, psd_tuple, det_tuple, gmst_val, model, @Const(lik_arr), tref)
    j = @index(Global, Linear)
    lik = _lik_from_array(lik_arr)
    ll[j] = _pd_ll_given_physical_multi(model, params[1, j], params[2, j], params[3, j], params[4, j], params[5, j],
                                        params[6, j], params[7, j], params[8, j], params[9, j], params[10, j],
                                        params[11, j], freqs_tuple, strain_tuple, psd_tuple, det_tuple, gmst_val, lik, tref)
end

# ---- IMRPhenomXHM with the likelihood options ------------------------------------------------------
# Distance marginalization, f_ref timing and the detector time reference, as for XAS. NOT phase marginalization:
# with higher modes the orbital phase enters each (l, m) mode as e^{i m phi}, so <d,h> is not a single overall phase
# rotation and the Bessel-function marginal does not apply (phase stays sampled). The waveform is the existing XHM
# kernels' (f_ref = first bin for phiRef, theta_jn as inclination, t_c as exp(-2 pi i f t)), so with no options the
# value equals `batched_log_likelihood_kernel!` for IMRPhenomXHM.
#
# t*(f_ref): XHM's (2,2) phase is Phase22(Mf)/eta + timeshift*Mf + phiref22 (`_xhm22_phase_full`), i.e. XAS's phase
# plus the linear `timeshift` term, so t* = XAS's t* minus msec*timeshift/(2 pi) in the kernel convention
# (sign checked against a finite difference of the XHM (2,2) contribution in test/test_xhm_likelihood_options.jl).
const _XHM_TSIGN = -1.0

@inline function _xhm_time_shift(sys, lik)
    lik.fref > 0 || return 0.0
    msec = sys.coeffs22.M * sys.coeffs22.GMsun_over_c3
    return _xas_time_shift(sys.coeffs22, lik) + _XHM_TSIGN * msec * sys.timeshift / (2 * pi)
end

# One detector's XHM sums (`sum |h - d|^2 / S`, `sum |h|^2 / S`, `sum conj(d) h / S`), h = Fp hp + Fc hc shifted by tsec.
@inline function _xhm_det_sums(sys, ms21, ms33, ms44, m32, theta_jn, Fplus, Fcross, tsec::T, freqs, strain, psd) where {T}
    piHalf = T(pi) / T(2)
    acc = zero(T); hh = zero(T); zr = zero(T); zi = zero(T)
    @inbounds for i in eachindex(freqs)
        f = freqs[i]
        hp, hc = _xhm_hphc_scalar(f, sys, ms21, ms33, ms44, m32, theta_jn, piHalf)
        tc_phase = -2 * T(pi) * f * tsec   # time sign: see _PHASE_CONVENTION
        h = (Fplus * hp + Fcross * hc) * Complex{T}(cos(tc_phase), sin(tc_phase))
        acc += _safe_div(abs2(h - strain[i]), psd[i])
        hh += _safe_div(abs2(h), psd[i])
        sh = conj(strain[i]) * h
        zr += _safe_div(real(sh), psd[i])
        zi += _safe_div(imag(sh), psd[i])
    end
    return acc, hh, zr, zi
end

@inline function _xhm_setup(m1, m2, chi1, chi2, dL, phase, f_ref)
    T = typeof(m1)
    mc, eta = _chirp_mass_eta(m1, m2)
    sys = _xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, phase;
                            GMsun_over_c3 = T(GMsun_over_c3), GMsun_over_c2_Gpc = T(GMsun_over_c2_Gpc))
    return sys, _xhm_mode_setup(2, 1, sys), _xhm_mode_setup(3, 3, sys), _xhm_mode_setup(4, 4, sys), _xhm_mode32_setup(sys)
end

# Whole-walker XHM log-likelihood with the likelihood options from PHYSICAL parameters, one detector.
@inline function _xhm_ll_given_physical(model, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi,
                                        det, gmst_val, freqs, strain, psd, lik)
    T = typeof(m1)
    sys, ms21, ms33, ms44, m32 = _xhm_setup(m1, m2, chi1, chi2, dL, phase, freqs[1])
    Fplus, Fcross = antenna_response_gmst(det, ra, dec, psi, gmst_val)
    tsec = t_c - _xhm_time_shift(sys, lik)
    acc, hh, zr, zi = _xhm_det_sums(sys, ms21, ms33, ms44, m32, theta_jn, Fplus, Fcross, tsec, freqs, strain, psd)
    df = freqs[2] - freqs[1]
    return _marg_combine(T(-2) * df * acc, T(4) * df * hh, T(4) * df * zr, T(4) * df * zi, dL, lik)
end

# The five XHM modes h_lm at f, in `_xhm_hphc_scalar`'s order ((2,2), (2,1), (3,3), (3,2), (4,4)), and each mode's
# polarization factors (`_xhm_add_mode!`: hp += fp h_lm, hc += fc h_lm), so a detector sees sum_k (Fp fp_k + Fc fc_k) h_k.
@inline function _xhm_modes(f, sys, ms21, ms33, ms44, m32)
    T = sys.T
    Mf = sys.coeffs22.M * sys.coeffs22.GMsun_over_c3 * T(f)
    amp22 = _xas_ampl_scalar(sys.amplcoeffs22, T(f))
    phi22 = _xhm22_phase_full(sys.coeffs22, sys.timeshift, sys.phiref22, Mf)
    h22 = T(_XP_HAS_MODE_NORMALIZATION) * Complex{T}(amp22 * cos(phi22), amp22 * sin(phi22))
    return (h22, _xhm_hlm_nonmixing(2, T(f), sys, ms21), _xhm_hlm_nonmixing(3, T(f), sys, ms33),
            _xhm_hlm_32(T(f), sys, m32), _xhm_hlm_nonmixing(4, T(f), sys, ms44))
end
# No closure: an `ntuple` closure over `T` captured a pointer, which Enzyme's type analysis rejects on Julia 1.13.1.
@inline _xhm_mode_factor(theta::T, l, m) where {T} =
    _xhm_add_mode!(zero(Complex{T}), zero(Complex{T}), l, m, one(Complex{T}), theta, T(pi) / T(2))
@inline _xhm_mode_factors(theta) = (_xhm_mode_factor(theta, 2, 2), _xhm_mode_factor(theta, 2, 1),
    _xhm_mode_factor(theta, 3, 3), _xhm_mode_factor(theta, 3, 2), _xhm_mode_factor(theta, 4, 4))
# Per-mode detector-frame strain u_k = (Fp fp_k + Fc fc_k) h_k e^{-2 pi i f tsec}.
@inline function _xhm_u(hs, fac, Fplus, Fcross, tsec::T, f) where {T}
    tc_phase = -2 * T(pi) * f * tsec
    sh = Complex{T}(cos(tc_phase), sin(tc_phase))
    return ntuple(k -> (Fplus * fac[k][1] + Fcross * fac[k][2]) * hs[k] * sh, Val(5))
end

# Bin b's <d,h> and <h,h> terms, unrolled over the 5 modes / 15 mode pairs (compile-time indices: runtime-indexed
# tuples made Enzyme's reverse pass need runtime activity).
const _XHM_PAIRS = Tuple((k, kp) for k in 1:5 for kp in k:5)
@generated function _xhm_rb_dh(A0, A1, r0, r1, b)
    Expr(:call, :+, (:(@inbounds(A0[$k, b]) * conj(r0[$k]) + @inbounds(A1[$k, b]) * conj(r1[$k])) for k in 1:5)...)
end
@generated function _xhm_rb_hh(C0, C1, r0, r1, b)
    terms = map(enumerate(_XHM_PAIRS)) do (p, (k, kp))
        t = :(real(@inbounds(C0[$p, b]) * r0[$k] * conj(r0[$kp]) + @inbounds(C1[$p, b]) * (r1[$k] * conj(r0[$kp]) + r0[$k] * conj(r1[$kp]))))
        k == kp ? t : :(2 * $t)
    end
    Expr(:call, :+, terms...)
end

"""
    _xhm_ll_rb(model, m1, ..., psi, rb, lik) -> Float64

Per-mode relative-binning IMRPhenomXHM log-likelihood with the likelihood options. Each mode's
detector-frame strain u_k is a complex constant times h_lm (`_xhm_add_mode!` is linear), so r_k = u_k / u_k,ref is as
smooth as XAS's ratio -- including under phase changes, which rotate each mode differently and are NOT smooth in the
total h / h_ref. <d,h> = sum_k sum_b A0_kb conj(r0_k) + A1_kb conj(r1_k); <h,h> = sum_{k,k'} sum_b C0 r0_k conj(r0_k') +
C1 (r1_k conj(r0_k') + r0_k conj(r1_k')) with the mode-pair summary data C (15 pairs; the (f - fm)^2 r1 r1 term is
dropped as in `_xas_ll_rb`). `rb` from `BatchedNUTS.build_xhm_relative_binning`.
"""
@inline function _xhm_ll_rb(model, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi, rb, lik)
    T = typeof(m1)
    sys, ms21, ms33, ms44, m32 = _xhm_setup(m1, m2, chi1, chi2, dL, phase, rb.fref0)
    fac = _xhm_mode_factors(theta_jn)
    g = rb.gmst
    dt_ref = ifelse(rb.geo & (rb.tref > 0), time_delay_from_geocenter_gmst(rb.dets[max(rb.tref, 1)], ra, dec, g), zero(T))
    t_c = t_c - _xhm_time_shift(sys, lik)
    ll_plain, hh, zr, zi = _xhm_rb_dets(sys, ms21, ms33, ms44, m32, fac, ra, dec, psi, g, t_c, dt_ref, rb)
    return _marg_combine(ll_plain, hh, zr, zi, dL, lik)
end

# All detectors, unrolled with literal indices (`rb.edges[1]`, ...): a runtime-indexed tuple of arrays, and arrays passed
# through a `map` closure, both made Enzyme's reverse pass demand runtime activity here (not in `_xas_ll_rb`).
@generated function _xhm_rb_dets(sys, ms21, ms33, ms44, m32, fac, ra, dec, psi, g, t_c::T, dt_ref, rb) where {T}
    N = fieldcount(fieldtype(rb, :edges))
    body = Expr[:(ll_plain = zero(T)), :(hh = zero(T)), :(zr = zero(T)), :(zi = zero(T))]
    for d in 1:N
        push!(body, quote
            dt_geo = ifelse(rb.geo, time_delay_from_geocenter_gmst(rb.dets[$d], ra, dec, g) - dt_ref, zero(T))
            Fplus, Fcross = antenna_response_gmst(rb.dets[$d], ra, dec, psi, g)
            dh, hs = _xhm_rb_det(sys, ms21, ms33, ms44, m32, fac, Fplus, Fcross, t_c + dt_geo, rb.edges[$d], rb.href[$d],
                                 rb.A0[$d], rb.A1[$d], rb.C0[$d], rb.C1[$d])
            ll_plain += rb.dd[$d] + real(dh) - T(0.5) * hs
            hh += hs
            zr += real(dh)
            zi -= imag(dh)
        end)
    end
    push!(body, :(return ll_plain, hh, zr, zi))
    return Expr(:block, Expr(:meta, :inline), body...)
end

# r_k = u_k / u_k,ref at edge e (no closure over the constant `uref`: see `_xhm_rb_dets`).
@generated _xhm_rb_ratios(u, uref, e) =
    Expr(:block, Expr(:meta, :inline), Expr(:tuple, (:(_rb_ratio(u[$k], @inbounds(uref[$k, e]))) for k in 1:5)...))

# One detector's binned <d,h> and <h,h> (`_xhm_ll_rb`).
@inline function _xhm_rb_det(sys, ms21, ms33, ms44, m32, fac, Fplus, Fcross, tsec::T, edges, uref, A0, A1, C0, C1) where {T}
    u = _xhm_u(_xhm_modes(edges[1], sys, ms21, ms33, ms44, m32), fac, Fplus, Fcross, tsec, edges[1])
    r_lo = _xhm_rb_ratios(u, uref, 1)
    dh = zero(Complex{T}); hs = zero(T)
    @inbounds for b in 1:length(edges) - 1
        f = edges[b+1]
        u = _xhm_u(_xhm_modes(f, sys, ms21, ms33, ms44, m32), fac, Fplus, Fcross, tsec, f)
        r_hi = _xhm_rb_ratios(u, uref, b + 1)
        idf = 1 / (f - edges[b])
        r0 = ntuple(k -> T(0.5) * (r_lo[k] + r_hi[k]), Val(5))
        r1 = ntuple(k -> (r_hi[k] - r_lo[k]) * idf, Val(5))
        dh += _xhm_rb_dh(A0, A1, r0, r1, b)
        hs += _xhm_rb_hh(C0, C1, r0, r1, b)
        r_lo = r_hi
    end
    return dh, hs
end

# Several detectors (`tref` as in `_xas_ll_given_physical_multi`; phiRef's f_ref = the first detector's first bin).
@inline function _xhm_ll_given_physical_multi(model, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi,
                                              freqs_tuple, strain_tuple, psd_tuple, det_tuple,
                                              gmst_val, lik, tref::Int)
    T = typeof(m1)
    sys, ms21, ms33, ms44, m32 = _xhm_setup(m1, m2, chi1, chi2, dL, phase, freqs_tuple[1][1])
    dt_ref = ifelse(tref == 0, zero(T), time_delay_from_geocenter_gmst(det_tuple[max(tref, 1)], ra, dec, gmst_val))
    t_c = t_c - _xhm_time_shift(sys, lik)
    ll_plain = zero(T); hh = zero(T); zr = zero(T); zi = zero(T)
    @inbounds for d in 1:length(freqs_tuple)
        dt_geo = time_delay_from_geocenter_gmst(det_tuple[d], ra, dec, gmst_val) - dt_ref
        Fplus, Fcross = antenna_response_gmst(det_tuple[d], ra, dec, psi, gmst_val)
        a, h2, r, i = _xhm_det_sums(sys, ms21, ms33, ms44, m32, theta_jn, Fplus, Fcross, t_c + dt_geo,
                                    freqs_tuple[d], strain_tuple[d], psd_tuple[d])
        df_d = freqs_tuple[d][2] - freqs_tuple[d][1]
        ll_plain += T(-2) * df_d * a
        hh += T(4) * df_d * h2
        zr += T(4) * df_d * r
        zi += T(4) * df_d * i
    end
    return _marg_combine(ll_plain, hh, zr, zi, dL, lik)
end

# CPU kernels for IMRPhenomXHM with the likelihood options (one walker per work item; the GPU XHM path is the
# structure-of-arrays kernels, which have no options).
@kernel function _xhm_opts_kernel!(ll, @Const(params), @Const(freqs), @Const(strain), @Const(psd), @Const(geom_flat),
                                   @Const(gmst_arr), model, @Const(lik_arr))
    j = @index(Global, Linear)
    lik = _lik_from_array(lik_arr)
    det = _geometry_from_array(geom_flat, 1)
    ll[j] = _xhm_ll_given_physical(model, params[1, j], params[2, j], params[3, j], params[4, j], params[5, j],
                                   params[6, j], params[7, j], params[8, j], params[9, j], params[10, j], params[11, j],
                                   det, gmst_arr[1], freqs, strain, psd, lik)
end

@kernel function _xhm_opts_multi_kernel!(ll, @Const(params), freqs_tuple, strain_tuple, psd_tuple, det_tuple, gmst_val, model, @Const(lik_arr), tref)
    j = @index(Global, Linear)
    lik = _lik_from_array(lik_arr)
    ll[j] = _xhm_ll_given_physical_multi(model, params[1, j], params[2, j], params[3, j], params[4, j], params[5, j],
                                         params[6, j], params[7, j], params[8, j], params[9, j], params[10, j],
                                         params[11, j], freqs_tuple, strain_tuple, psd_tuple, det_tuple, gmst_val, lik, tref)
end

# n-point Gauss-Legendre nodes/weights on [-1, 1] (Newton on P_n; computed once at load time).
function _gauss_legendre(n::Int)
    x = zeros(n); w = zeros(n)
    for i in 1:n
        z = cos(pi * (i - 0.25) / (n + 0.5))
        for _ in 1:100
            p0, p1 = 1.0, z
            for k in 2:n
                p0, p1 = p1, ((2k - 1) * z * p1 - (k - 1) * p0) / k
            end
            dp = n * (z * p1 - p0) / (z^2 - 1)
            dz = p1 / dp
            z -= dz
            abs(dz) < 1e-15 && break
        end
        p0, p1 = 1.0, z
        for k in 2:n
            p0, p1 = p1, ((2k - 1) * z * p1 - (k - 1) * p0) / k
        end
        dp = n * (z * p1 - p0) / (z^2 - 1)
        x[i] = z
        w[i] = 2 / ((1 - z^2) * dp^2)
    end
    return x, w
end
const _NDIST = 64
const _GL_X, _GL_W = let (x, w) = _gauss_legendre(_NDIST); (NTuple{_NDIST,Float64}(x), NTuple{_NDIST,Float64}(w)) end

# Distance-marginalized log-likelihood from the walker's own inner products. At the walker's distance
# `dL` the kernels have `hh = <h,h>`, `z = <d,h>` (scaled by 4df), and `ll_plain`. At distance D these
# scale as hh (dL/D)^2 and z (dL/D), so with u = dL/D
#     L(D) = exp(-dd/2 - hh u^2/2 + [Re z | log I0(|z|)] u)          (plain | phase-marginalized)
# and the marginal is log( int p(D) L(D) dD ), p(D) = D^k / N on [dlo, dhi]. The integrand is ~Gaussian
# in u (mean u* = z/hh, sd 1/sqrt(hh)), so the quadrature window is u* +- 12 sd (mapped to D, clipped to
# the prior range, kept >= 24 sd wide when the peak sits outside it); on that window a 64-point
# Gauss-Legendre rule in D is accurate at any SNR (tested against brute-force integration).
@inline function _distance_marg(ll_plain::T, hh::T, zr::T, absz::T, dL::T, lik) where {T<:Real}
    zeff = ifelse(lik.margphase, absz, zr)
    minus_dd_half = ll_plain + hh / 2 - zr
    hhs = max(hh, T(1e-30))
    ustar = zeff / hhs
    sig = 1 / sqrt(hhs)
    umin = dL / T(lik.dhi)
    umax = dL / T(lik.dlo)
    a0 = clamp(ustar - 12 * sig, umin, umax)
    b0 = clamp(ustar + 12 * sig, umin, umax)
    ua = min(a0, max(umin, b0 - 24 * sig))
    ub = max(b0, min(umax, a0 + 24 * sig))
    Da = dL / ub          # u and D are inversely related
    Db = dL / ua
    half = (Db - Da) / 2
    mid = (Db + Da) / 2
    k = lik.dpow
    # log-sum-exp over the nodes (two passes: max, then sum)
    m = T(-Inf)
    @inbounds for i in 1:_NDIST
        D = mid + half * T(_GL_X[i])
        u = dL / D
        e = k * log(D) - hhs * u * u / 2 + ifelse(lik.margphase, _log_besseli0(absz * u), zr * u)
        m = max(m, e)
    end
    acc = zero(T)
    @inbounds for i in 1:_NDIST
        D = mid + half * T(_GL_X[i])
        u = dL / D
        e = k * log(D) - hhs * u * u / 2 + ifelse(lik.margphase, _log_besseli0(absz * u), zr * u)
        acc += T(_GL_W[i]) * exp(e - m)
    end
    lognorm = log((T(lik.dhi)^(k + 1) - T(lik.dlo)^(k + 1)) / (k + 1))
    return minus_dd_half + m + log(acc * half) - lognorm
end

# `_distance_marg` (no phase marginalization) with its partials from the same quadrature, for calibration
# marginalization (one value per response curve and walker: an in-kernel reverse per curve would cost too much).
# With u = dL / D the integrand is exp(e(D)), e = k log D - hh u^2/2 + zr u, so (exact integral, the window fixed)
#   d val / d ll_plain = 1,  d val / d hh = 1/2 - E[u^2]/2,  d val / d zr = -1 + E[u],
#   d val / d dL (explicit) = E[(zr - hh u) u] / dL,
# E over the normalized integrand. Returns (val, E[u], E[u^2], E[(zr - hh u) u] / dL).
@inline function _distance_marg_moments(ll_plain::T, hh::T, zr::T, dL::T, lik) where {T<:Real}
    minus_dd_half = ll_plain + hh / 2 - zr
    hhs = max(hh, T(1e-30))
    ustar = zr / hhs
    sig = 1 / sqrt(hhs)
    umin = dL / T(lik.dhi)
    umax = dL / T(lik.dlo)
    a0 = clamp(ustar - 12 * sig, umin, umax)
    b0 = clamp(ustar + 12 * sig, umin, umax)
    ua = min(a0, max(umin, b0 - 24 * sig))
    ub = max(b0, min(umax, a0 + 24 * sig))
    Da = dL / ub
    Db = dL / ua
    half = (Db - Da) / 2
    mid = (Db + Da) / 2
    k = lik.dpow
    m = T(-Inf)
    @inbounds for i in 1:_NDIST
        D = mid + half * T(_GL_X[i])
        u = dL / D
        m = max(m, k * log(D) - hhs * u * u / 2 + zr * u)
    end
    acc = zero(T); au = zero(T); au2 = zero(T); ad = zero(T)
    @inbounds for i in 1:_NDIST
        D = mid + half * T(_GL_X[i])
        u = dL / D
        wgt = T(_GL_W[i]) * exp(k * log(D) - hhs * u * u / 2 + zr * u - m)
        acc += wgt; au += wgt * u; au2 += wgt * u * u; ad += wgt * (zr - hhs * u) * u
    end
    lognorm = log((T(lik.dhi)^(k + 1) - T(lik.dlo)^(k + 1)) / (k + 1))
    return minus_dd_half + m + log(acc * half) - lognorm, au / acc, au2 / acc, ad / (acc * dL)
end

# `hh`, `zr`, `zi`: `<h,h>` and real/imag parts of `<d,h>` (all already scaled by `4 df`).
@inline function _marg_combine(ll_plain::T, hh::T, zr::T, zi::T, dL::T, lik) where {T<:Real}
    absz = sqrt(zr * zr + zi * zi + T(1e-300))
    ll_phase = ifelse(lik.margphase, ll_plain - zr + _log_besseli0(absz), ll_plain)
    return ifelse(lik.margdist, _distance_marg(ll_plain, hh, zr, absz, dL, lik), ll_phase)
end

"""
    _chirp_mass_eta_kernel!(mc, eta, m1, m2)

`@kernel`-based (not broadcast) computation of `mc`/`eta` from `m1`/`m2`, used ONLY by the
IMRPhenomXHM GPU structure-of-arrays dispatch path (see
[`_fused_waveform_likelihood_kernel_phenomxhm_soa!`](@ref)'s docstring). A plain broadcast
expression (`mc = ((m1.*m2)./(m1.+m2)./(m1.+m2)).^T(3/5) .* (m1.+m2)`) computes the right forward
values but Enzyme's CUDA broadcast-differentiation rule (`EnzymeCore08Ext`'s
`_create_tape_kernel`) throws a `MethodError` constructing `MixedDuplicated` for this specific
nested-power broadcast expression when reverse-mode-differentiated -- confirmed directly (not
guessed): the forward pass alone (no `Enzyme.autodiff`) with the broadcast form works and matches
CPU exactly, and switching ONLY this piece from broadcast to a `@kernel` (Enzyme-through-
KernelAbstractions is the path this whole file's Enzyme validation already exercises everywhere
else) resolves it with no other change. A real Enzyme/GPUArrays integration bug, not a bug in this
package -- worth reporting upstream, but this sidesteps it either way.
"""
@kernel function _chirp_mass_eta_kernel!(mc, eta, @Const(m1), @Const(m2))
    i = @index(Global, Linear)
    mc[i], eta[i] = _chirp_mass_eta(m1[i], m2[i])
end

# =================================================================================================
# Multi-detector extension. Design summary (see `Types.DetectorData`'s docstring
# for the data-type decision):
#
#   - `data` becomes `AbstractVector{<:LikelihoodData}`, one entry per detector, all sharing the
#     same physical `gps_time` (asserted at the top of `batched_log_likelihood_kernel!`'s
#     multi-detector method below).
#   - Each detector contributes its own antenna-response projection AND its own coalescence-time
#     shift: `t_c` (row 6 of `params`, the SAME sampled/geocenter-relative offset for every
#     detector -- there is one physical event, one `t_c`) is adjusted per detector by
#     `GravitationalWaveData.time_delay_from_geocenter(detector, ra, dec, gps_time)`, the light-travel-time
#     offset between the geocenter and that detector for a wave from sky position `(ra, dec)`.
#   - The time delay is `GravitationalWaveData.time_delay_from_geocenter_gmst` on each detector's isbits
#     `GravitationalWaveData.DetectorGeometry` (see the module docstring for why kernels never see the named
#     `Detector`). `ra`/`dec` are read fresh from `params` INSIDE the kernel body on every call, so
#     Enzyme differentiates `d(logL)/d(ra)`/`d(logL)/d(dec)` through the time-delay term, not just
#     through the antenna response term.
#   - Per-detector geometries (`DetectorGeometry`, plain bits) and `freqs`/`strain`/
#     `psd` (all `Vector`s, homogeneously typed across detectors since every `LikelihoodData` in
#     `data` shares the same element type `T`) are packed host-side into `NTuple{ndet, ...}`s and
#     passed as plain (not Enzyme-active) kernel arguments -- `ndet` is small (2-3: H1/L1[/V1]), so
#     each kernel thread loops over `1:ndet` (not a second kernel dimension), matching this file's
#     existing per-walker-thread style. `NTuple{ndet, Vector{T}}` etc. are all HOMOGENEOUS tuples
#     (every detector's data shares the same Julia type), so runtime-`Int`-indexed `freqs_tuple[d]`
#     inside the kernel loop is type-stable -- deliberately not `Vector{Vector{T}}`, which would
#     also work but loses this compile-time homogeneity.
#   - `gmst_val` (from the shared `gps_time`) is still computed ONCE, host-side, outside the kernel
#     launch and reused for every detector's antenna-response AND
#     time-delay calculation inside the per-detector loop (both are legitimately functions of the
#     SAME Greenwich Mean Sidereal Time, since GMST depends only on `gps_time`, not on detector or
#     sampled parameters).
#   - Per-walker frequency-independent quantities that do NOT depend on the detector (`mc`, `eta`,
#     `M_sec`, `phiRef`, `pfac`, `cfac`, and `ampl_coeffs` since amplitude has no detector- or
#     time-delay-dependence) are computed ONCE per walker, outside the `1:ndet` loop; only
#     detector-specific `t0` (via the time-delay-shifted `t_c`), `combo` (via that detector's own
#     `Fplus`/`Fcross`), and `phi_coeffs` (which depends on `t0`) are recomputed per detector.
#   - single-detector kernel bodies/`_extrinsic_setup` above are LEFT UNCHANGED (no
#     time-delay term) -- single-detector `t_c` has no second detector to align against, so adding
#     a geocenter-vs-detector time-delay term there would be a physically-meaningless (if harmless)
#     change to already-validated code.
# =================================================================================================

# Frequency-independent, per-walker "extrinsic" setup shared by every kernel body below:
#   - `t0`/`phiRef` kwargs for `_phi_coeffs`/`Phi` implementing the `t_c`/`phase` shift (see this
#     file's module docstring for the derivation of `t0`'s exact scale/sign).
#   - `combo`, the single complex scalar folding together the plus/cross decomposition
#     (`pfac`/`cfac`, LALSuite-ported, see module docstring) and the antenna response
#     (`GravitationalWaveData.compute_antenna_response`, called with the precomputed-GMST fast path so the
#     kernel doesn't repeat the leap-second/Julian-day GMST computation once per walker): the
#     detector-frame waveform is exactly `combo(theta) * amp(f;theta) * exp(im*phase(f;theta))`.
# `mc`, `eta`, `M_sec` are returned too since callers need `mc`/`eta` again for the intrinsic
# coefficient calls (`_phi_coeffs`/`_ampl_coeffs`/`_xas_phi_coeffs`/`_xas_ampl_coeffs`).
# **`t0` UNIT TRAP for SEOBNRv5AlignedSpinF2.** `_extrinsic_setup`'s `t0 =
# -2*pi*t_c/M_sec` is the convention of the Phenom/PN phases, which multiply `t0` by the DIMENSIONLESS
# `Mf = M_sec*f` (so the net phase shift is `2*pi*f*t_c`). `_seobv5f2_phi_scalar`/`Phi(::SEOBNRv5AlignedSpinF2)`
# instead apply `- t0 * f` with `f` in Hz, so feeding them that same `t0` shifts the phase by
# `2*pi*f*t_c/M_sec` -- a mass-dependent factor `~1/M_sec ~ 3000` too large. Symptom: on real data the
# sampled `t_c` posterior was a comb of `~10 us`-wide spikes (logp curvature ~1e11 s^-2 vs the physical
# ~1e7), every walker froze in `t_c` (Rhat 228), and injection/recovery tests still passed because
# injection and inference shared the same wrong convention. Every SEOBNR phase in this package now uses
# the time shift in SECONDS directly (`-2*pi*t_c` [+ dt_geo]) with the physical `f`; anything that calls
# the library's `Phi(::SEOBNRv5AlignedSpinF2; t0=...)` must go through `_phi_t0`.
_phi_t0(model, t0, M_sec) = t0                                  # Phenom / PN: `t0` is Mf-based already
_phi_t0(::PAF2Model, t0, M_sec) = t0 * M_sec        # SEOBNR `Phi` takes `t0 = 2*pi*seconds`

# _PHASE_CONVENTION: the data are FFTW
# transforms, X(f) = sum x(t) e^{-2 pi i f t}, referenced to the trigger time. A signal arriving at time
# t (relative to the trigger) is therefore h(f) e^{-2 pi i f t}. Measured on real GW150914 data, the
# IMRPhenomXAS-family `Phi` (XAS, XP, XHM, XPHM) has the data's sign, while IMRPhenomD, TaylorF2,
# AlignedSpinPNHamF2 and every PA-F2 / IMR-table model return the conjugate phase. Every kernel builds
#   h = combo * A * exp(i * (s * Phi_intrinsic(f) - 2 pi f (t_c + dt_geo) - s * phiRef))
# with s = +1 for the XAS family and s = -1 for the others: the XAS-family kernels flip the time term
# (`t0 = -t0`, `tc_phase = -2 pi f t`), the others negate the whole assembled phase (which conjugates the
# intrinsic phase and the time term together). With exp(+2 pi i f t_c) instead, t_c would be minus
# the arrival time and multi-detector time delays would enter with the wrong relative sign.
@inline function _extrinsic_setup(m1, m2, dL, t_c, phase, ra, dec, theta_jn, psi, det, gmst_val)
    T = typeof(m1)
    mc, eta = _chirp_mass_eta(m1, m2)
    M_sec = (mc / eta^T(0.6)) * T(GMsun_over_c3)
    t0 = -2 * T(pi) * t_c / M_sec
    phiRef = phase

    pfac = T(0.5) * (1 + cos(theta_jn) * cos(theta_jn))
    cfac = cos(theta_jn)
    Fplus, Fcross = antenna_response_gmst(det, ra, dec, psi, gmst_val)
    combo = Fplus * pfac - im * Fcross * cfac

    return mc, eta, t0, phiRef, combo
end

# Trimmed variant of `_extrinsic_setup` returning only `phiRef`/`combo`, for call sites that don't
# use `mc`/`eta`/`t0` (unlike every OTHER caller of `_extrinsic_setup` in this file, which needs
# `mc`/`eta`/`t0` to build a waveform via `Ampl`/`Phi`/`_*_coeffs` -- see e.g. lines 399-402,
# 437-439, 483-486, 522-524, 562-564, 630-633 -- do not redirect those to this function).
# `_seobv5f2_ll_given_coeffs` is the one exception: its `coeffs` argument is already built
# (`_seobv5f2_build_coeffs_soa_kernel!`, which separately produces `M_sec`/`eta` into its own SoA
# buffers), so `_extrinsic_setup`'s internal `mc`/`eta`/`M_sec`/`t0` recomputation there was always
# dead code -- confirmed by reading `_seobv5f2_ll_given_coeffs`'s body, which only reads `phiRef`/
# `combo` from the 5-tuple. This isn't just an optimization: it also works around a confirmed
# upstream Enzyme/KernelAbstractions bug
# where a discarded power-law expression co-occurring with a later `Complex`-via-`im` construction,
# in a kernel differentiated via `Enzyme.Reverse` on CUDA, makes the "peek" tape-type compile
# (`EnzymeCore.tape_type`) disagree with the real kernel compile and throw
# `AssertionError: expectedTapeType === TapeType`. Dropping the dead `mc`/`eta` computation removes
# the trigger.
@inline function _extrinsic_setup_phase_only(phase, ra, dec, theta_jn, psi, det, gmst_val)
    T = typeof(phase)
    phiRef = phase

    pfac = T(0.5) * (1 + cos(theta_jn) * cos(theta_jn))
    cfac = cos(theta_jn)
    Fplus, Fcross = antenna_response_gmst(det, ra, dec, psi, gmst_val)
    combo = Fplus * pfac - im * Fcross * cfac

    return phiRef, combo
end

# Every kernel below writes `Complex(cos(x), sin(x))` instead of the algebraically identical
# `exp(im * x)`. Reason: `Enzyme.autodiff(Enzyme.Reverse, ...)` on `CUDABackend()` hard-crashes
# the process (SIGABRT, an assert in Enzyme's TypeAnalysis.cpp) when differentiating `exp(im*x)`/
# `cis(x)` of an argument that depends on the differentiated parameter .
# This `cos`/`sin` split avoids the crash entirely, while the
# combined `sincos(x)` intrinsic does NOT (it hits a second, different Enzyme assert in
# `GradientUtils.cpp`'s `invertPointerM`). This is a workaround pending an upstream Enzyme fix
# and is applied here across every model, not just SEOBNR -- distinct from
# the narrower SEOBNR-only forward-mode workaround in `BatchedNUTS.jl`, which sidesteps a
# different bug (no forward-mode rule for CUDA's fused `__nv_sincosf`).

# Generic fallback: one kernel thread per walker, generating that walker's waveform via the
# array-in/array-out `Ampl`/`Phi` calls (which internally `@.`-broadcast and allocate a fresh
# output array per call) and reducing it against the data into a single log-likelihood scalar.
# This works for any model sharing the `Ampl`/`Phi` contract (e.g. IMRPhenomXAS), but is NOT used
# for IMRPhenomD -- see `_fused_waveform_likelihood_kernel_phenomd!` below and its rationale.
#
# Enzyme note: `Enzyme.autodiff(Enzyme.Reverse, ...)` over a function that calls this kernel did
# NOT complete within 5 minutes (confirmed twice, at a tiny problem size: ntime=16, nwalkers=2)
# -- reverse-mode compilation appears to hang, not merely run slowly, when the kernel body
# allocates via the vector-in/vector-out `Ampl`/`Phi`. This fallback is kept for models that
# don't (yet) have a non-allocating scalar core; do not rely on Enzyme differentiating through
# it until/unless that's independently verified for the model in question.
@kernel function _fused_waveform_likelihood_kernel!(ll, @Const(params), @Const(freqs), @Const(strain), @Const(psd), @Const(geom_flat), @Const(gmst_arr), model)
    j = @index(Global, Linear)
    T = eltype(params)
    det = _geometry_from_array(geom_flat, 1)
    gmst_val = gmst_arr[1]

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]

    mc, eta, t0, phiRef, combo = _extrinsic_setup(m1, m2, dL, t_c, phase, ra, dec, theta_jn, psi, det, gmst_val)

    amp = Ampl(model, freqs, mc, eta, chi1, chi2, dL)
    phi = Phi(model, freqs, mc, eta, chi1, chi2; t0=t0, phiRef=phiRef)

    acc = zero(T)
    @inbounds for i in eachindex(freqs)
        h = combo * (amp[i] * Complex(cos(phi[i]), -sin(phi[i])))   # conj: see _PHASE_CONVENTION
        d = h - strain[i]
        acc += _safe_div(abs2(d), psd[i])
    end
    df = freqs[2] - freqs[1]
    ll[j] = T(-2) * df * acc
end

# IMRPhenomD-specialized fast path: one kernel thread per walker, computing the frequency-
# independent phase/amplitude coefficients ONCE per walker (`_phi_coeffs`/`_ampl_coeffs`), then
# looping over frequency bins calling the non-allocating scalar cores (`_phi_scalar`/
# `_ampl_scalar`) directly -- no array is ever allocated inside the kernel body.
#
# This exists because the generic fallback above (calling the array-in/array-out `Ampl`/`Phi`)
# made `Enzyme.autodiff` hang indefinitely when reverse-mode-compiling this kernel; the
# non-allocating scalar-core version differentiates successfully (see `test_enzyme_ka.jl`). It
# is also strictly faster on real waveforms: the fallback would otherwise recompute ~100 lines
# of PN/phenomenological-coefficient arithmetic for every one of possibly thousands of frequency
# bins instead of once per walker.
@kernel function _fused_waveform_likelihood_kernel_phenomd!(ll, @Const(params), @Const(freqs), @Const(strain), @Const(psd), @Const(geom_flat), @Const(gmst_arr), model)
    j = @index(Global, Linear)
    T = eltype(params)
    det = _geometry_from_array(geom_flat, 1)
    gmst_val = gmst_arr[1]

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]

    mc, eta, t0, phiRef, combo = _extrinsic_setup(m1, m2, dL, t_c, phase, ra, dec, theta_jn, psi, det, gmst_val)

    phi_coeffs = _phi_coeffs(model, mc, eta, chi1, chi2; t0=t0, phiRef=phiRef)
    ampl_coeffs = _ampl_coeffs(model, mc, eta, chi1, chi2, dL)

    acc = zero(T)
    @inbounds for i in eachindex(freqs)
        amp = _ampl_scalar(ampl_coeffs, freqs[i])
        phase_val = -_phi_scalar(phi_coeffs, freqs[i])   # conj: see _PHASE_CONVENTION
        h = combo * (amp * Complex(cos(phase_val), sin(phase_val)))
        d = h - strain[i]
        acc += _safe_div(abs2(d), psd[i])
    end
    df = freqs[2] - freqs[1]
    ll[j] = T(-2) * df * acc
end

"""
    _fused_waveform_likelihood_kernel_phenomd_soa!(ll, m1, m2, chi1, chi2, dL, t_c, phase, ra,
                                                     dec, theta_jn, psi, freqs, strain, psd,
                                                     geom_flat, gmst_arr, model)

GPU-only structure-of-arrays variant of [`_fused_waveform_likelihood_kernel_phenomd!`](@ref) --
same relationship, same reason, as [`_fused_waveform_likelihood_kernel_phenomxhm_soa!`](@ref) has
to its own matrix-based counterpart (see that docstring for the general pattern this follows).
The matrix-plus-in-kernel-`_extrinsic_setup`(which itself calls `_chirp_mass_eta`) path hits the
same unresolved GPUCompiler.jl/Julia inference issue once the DOWNSTREAM `_phi_coeffs`/
`_ampl_coeffs` calls follow it -- confirmed directly: `Base.return_types`/`code_typed` report
`_phi_coeffs` fully concrete in isolation, yet `Enzyme.autodiff` through the real matrix-based
kernel fails with `IllegalTypeAnalysisException` localized to that exact call, unaffected by
`Enzyme.API.strictAliasing!(false)` (Enzyme's own suggested fix). Separate 1-D `CuArray`s per
field sidesteps it, matching the XHM fix exactly. Selected automatically by
`batched_log_likelihood_kernel!` for `model isa IMRPhenomD` on non-CPU backends; CPU path
untouched.
"""
@kernel function _fused_waveform_likelihood_kernel_phenomd_soa!(ll, @Const(m1), @Const(m2),
        @Const(chi1), @Const(chi2), @Const(dL), @Const(t_c), @Const(phase), @Const(ra),
        @Const(dec), @Const(theta_jn), @Const(psi), @Const(freqs), @Const(strain), @Const(psd),
        @Const(geom_flat), @Const(gmst_arr), model)
    j = @index(Global, Linear)
    T = eltype(m1)
    det = _geometry_from_array(geom_flat, 1)
    gmst_val = gmst_arr[1]

    mc, eta, t0, phiRef, combo = _extrinsic_setup(m1[j], m2[j], dL[j], t_c[j], phase[j], ra[j],
                                                   dec[j], theta_jn[j], psi[j], det, gmst_val)

    phi_coeffs = _phi_coeffs(model, mc, eta, chi1[j], chi2[j]; t0=t0, phiRef=phiRef)
    ampl_coeffs = _ampl_coeffs(model, mc, eta, chi1[j], chi2[j], dL[j])

    acc = zero(T)
    @inbounds for i in eachindex(freqs)
        amp = _ampl_scalar(ampl_coeffs, freqs[i])
        phase_val = -_phi_scalar(phi_coeffs, freqs[i])   # conj: see _PHASE_CONVENTION
        h = combo * (amp * Complex(cos(phase_val), sin(phase_val)))
        d = h - strain[i]
        acc += _safe_div(abs2(d), psd[i])
    end
    df = freqs[2] - freqs[1]
    ll[j] = T(-2) * df * acc
end

# IMRPhenomXAS-specialized fast path: mirrors `_fused_waveform_likelihood_kernel_phenomd!` above,
# using IMRPhenomXAS's own non-allocating scalar core (`_xas_phi_coeffs`/`_xas_phi_scalar`/
# `_xas_ampl_coeffs`/`_xas_ampl_scalar`, see GravitationalWaveforms.jl's IMRPhenomXAS.jl) instead of IMRPhenomD's. Exists
# for the same reason: the generic fallback's array-in/array-out `Ampl`/`Phi` calls made
# `Enzyme.autodiff` hang indefinitely when reverse-mode-compiling this kernel (see
# `test/test_enzyme_ka.jl`'s IMRPhenomXAS testset and the notes above), and recomputing XAS's collocation-point/linear-solve coefficient block
# per frequency bin instead of once per walker would be a severe performance regression on real
# waveforms.
@kernel function _fused_waveform_likelihood_kernel_phenomxas!(ll, @Const(params), @Const(freqs), @Const(strain), @Const(psd), @Const(geom_flat), @Const(gmst_arr), model, @Const(lik_arr))
    j = @index(Global, Linear)
    T = eltype(params)
    lik = _lik_from_array(lik_arr)
    det = _geometry_from_array(geom_flat, 1)
    gmst_val = gmst_arr[1]

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]

    ll[j] = _xas_ll_given_physical(model, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi,
                                   det, gmst_val, freqs, strain, psd, lik)
end

# AlignedSpinPNHamF2-specialized fast path: mirrors
# `_fused_waveform_likelihood_kernel_phenomxas!` exactly, using AlignedSpinPNHamF2's own
# non-allocating scalar core (`_asphf2_phi_coeffs`/`_asphf2_phi_scalar`/`_asphf2_ampl_coeffs`/
# `_asphf2_ampl_scalar`, see GravitationalWaveforms.jl's `AlignedSpinPNHamF2.jl`) instead of IMRPhenomXAS's --
# same aligned-spin, 11-row physical convention (`Phi`/`Ampl(model, f, mc, eta, chi1, chi2[, dL])`,
# no in-plane spin, no lambda1/lambda2), so no new row convention or `_extrinsic_setup` variant is
# needed. Exists for the same Enzyme-through-KernelAbstractions reason as every other specialized
# fast path in this file: the generic array-in/array-out fallback above hangs `Enzyme.autodiff`
# reverse-mode compilation; this non-allocating scalar-core version is expected to differentiate
# the same way IMRPhenomD/XAS already do (see `test/test_enzyme_ka_alignedspinpnhamf2.jl`).
@kernel function _fused_waveform_likelihood_kernel_alignedspinpnhamf2!(ll, @Const(params), @Const(freqs), @Const(strain), @Const(psd), @Const(geom_flat), @Const(gmst_arr), model)
    j = @index(Global, Linear)
    T = eltype(params)
    det = _geometry_from_array(geom_flat, 1)
    gmst_val = gmst_arr[1]

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]

    mc, eta, t0, phiRef, combo = _extrinsic_setup(m1, m2, dL, t_c, phase, ra, dec, theta_jn, psi, det, gmst_val)

    phi_coeffs = _asphf2_phi_coeffs(model, mc, eta, chi1, chi2; t0=t0, phiRef=phiRef)
    ampl_coeffs = _asphf2_ampl_coeffs(model, mc, eta, chi1, chi2, dL)

    acc = zero(T)
    @inbounds for i in eachindex(freqs)
        amp = _asphf2_ampl_scalar(ampl_coeffs, freqs[i])
        phase_val = -_asphf2_phi_scalar(phi_coeffs, freqs[i])   # conj: see _PHASE_CONVENTION
        h = combo * (amp * Complex(cos(phase_val), sin(phase_val)))
        d = h - strain[i]
        acc += _safe_div(abs2(d), psd[i])
    end
    df = freqs[2] - freqs[1]
    ll[j] = T(-2) * df * acc
end

# SEOBNRv5AlignedSpinF2-specialized kernel body: mirrors
# `_fused_waveform_likelihood_kernel_alignedspinpnhamf2!` structurally (same 11-row aligned-spin
# convention, `_extrinsic_setup` unchanged), but `coeffs` here is NOT closed-form -- it is
# `SEOBNRv5AlignedSpinF2`'s own dense post-adiabatic table (`_seobv5f2_coeffs`, GravitationalWaveforms.jl),
# built ONCE per walker (like `ampl_coeffs` elsewhere in this file) since it depends only on
# `mc`/`eta`/`chi1`/`chi2`/`f_min`, not on `t0`/`phiRef`/`dL` -- those are applied per-frequency in
# the scalar calls exactly as `SEOBNRv5AlignedSpinF2.Phi`/`.Ampl` do. Deliberately NOT passing an
# explicit `f_min` here (unlike an earlier draft of this kernel, which tried `f_min = freqs[1]`,
# mirroring `IMRPhenomXP`'s own `f_ref = freqs[1]` choice) -- `build_relative_binning`'s generic,
# array-oriented reference-waveform construction (used to build this same model's relative-binning
# summary statistics) has no way to pass a model-specific `f_min` either, so it always uses this
# model's own DEFAULT (`f_min=20.0`). Unlike `IMRPhenomXP`'s `f_ref` (a pure orientation/reference
# parameter that doesn't change the aligned-(2,2)-mode phase itself), THIS model's `f_min` literally
# defines where its `phi=0`/`t=0` anchor sits along the physical trajectory -- using a different
# `f_min` here than `build_relative_binning` used would silently misalign the two computations'
# phase origin by the accumulated phase between the two anchors. Confirmed empirically: the
# `freqs[1]`-based version failed this exact model's own dense-vs-binned agreement test at
# `theta_ref` (should be exact to numerical precision, was off by `~0.4` in log-likelihood) until
# this fix. Always using the shared default is only correct as long as the actual data band's own
# lowest frequency is `>= 20.0`Hz (this ecosystem's usual convention); a genuinely different-`f_min`
# analysis would need `build_relative_binning` itself extended to pass a model-specific `f_min`
# through consistently, not attempted here.
#
# `_seobv5f2_coeffs` works on fixed-size (`SVector`) buffers, so this kernel compiles and runs on
# CUDA (matching the CPU to `rtol=1e-8`). The gradients on the GPU go through the dedicated device
# paths (`BatchedNUTS`), not through `Enzyme.Reverse` on this kernel.
@kernel function _fused_waveform_likelihood_kernel_seobnrv5alignedspinf2!(ll, @Const(params), @Const(freqs), @Const(strain), @Const(psd), @Const(geom_flat), @Const(gmst_arr), model, @Const(lik_arr))
    j = @index(Global, Linear)
    lik = _lik_from_array(lik_arr)
    det = _geometry_from_array(geom_flat, 1)
    gmst_val = gmst_arr[1]

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]

    mc, eta = _chirp_mass_eta(m1, m2)
    coeffs = _seobv5f2_coeffs(model, mc, eta, chi1, chi2)
    ll[j] = _seobv5f2_ll_given_coeffs(coeffs, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi,
        det, gmst_val, freqs, strain, psd, lik)
end

# =================================================================================================
# Precomputed-coeffs variant of the kernel above: `coeffs` is NOT built
# internally via `_seobv5f2_coeffs(model, mc, eta, chi1, chi2)` -- it's read from caller-supplied
# struct-of-arrays buffers instead. This exists so `BatchedNUTS.jl`'s forward-mode gradient for
# `SEOBNRv5AlignedSpinF2` can differentiate this kernel via `Enzyme.Forward` treating the SoA
# buffers as `Duplicated` (their tangent computed by `SEOBNRv5AlignedSpinF2Adjoint.seobv5f2_coeffs_jvp`
# on the HOST, never inside a differentiated kernel launch), WITHOUT ever asking Enzyme to trace
# `_seobv5f2_coeffs`'s own construction -- the fixed-iteration Newton/PA-driver internals that
# trigger the upstream Enzyme/CUDA bugs described at `BatchedNUTS.jl`'s
# `_forward_mode_gradient!`. Everything this kernel body still
# does directly (`_extrinsic_setup`, the bracket-search reconstruction, the residual sum) is exactly
# the code the ORIGINAL kernel already differentiates fine -- unchanged here, just no longer bundled
# with the coeffs construction in the same differentiated call.
#
# `success`/`nreal` are `@Const` (never `Duplicated`) -- they're Bool/Int, never differentiable, and
# passing them as ordinary Const arguments (rather than a placeholder-tangent `Duplicated`, which
# was ALSO confirmed to work on CPU via `Enzyme.Duplicated` on a whole mixed-field `NamedTuple` in
# this feature's own development, see `SEOBNRv5AlignedSpinF2Adjoint.jl`) is the more standard,
# lower-risk Enzyme usage -- avoids relying on that NamedTuple-with-mixed-activity-fields behavior
# composing safely on `CUDABackend()`, which is unverified.

"""
    _seobv5f2_ll_given_coeffs(coeffs, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi,
        det, gmst_val, freqs, strain, psd) -> Float64

Identical body to `_fused_waveform_likelihood_kernel_seobnrv5alignedspinf2!`'s per-thread
computation, minus the `coeffs = _seobv5f2_coeffs(model, mc, eta, chi1, chi2)` line -- `coeffs` is
an explicit argument here instead.
"""
@inline function _seobv5f2_ll_given_coeffs(coeffs, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi,
        det, gmst_val, freqs, strain, psd, lik=_LIK_PLAIN)
    T = typeof(m1)
    phiRef, combo = _extrinsic_setup_phase_only(phase, ra, dec, theta_jn, psi, det, gmst_val)
    t_c = t_c - _seobv5f2_time_shift(coeffs, lik)
    tshift = -2 * T(pi) * t_c      # seconds -> rad/Hz; NOT `t0` (see `_phi_t0`)
    acc = zero(T)
    hh = zero(T)
    zr = zero(T)
    zi = zero(T)
    @inbounds for i in eachindex(freqs)
        # Outside the table's band (or for a failed table) amplitude and phase are exactly 0, so h == 0:
        # only the noise term |d|^2/S remains. Skipping the interpolation there is bit-identical and is
        # most of the work when fmax is far below the data's f_max (~31 Hz vs 128 Hz for GW150914 masses).
        if coeffs.success && coeffs.fmin <= freqs[i] <= coeffs.fmax
            a0, p0 = _seobv5f2_amp_phi(coeffs, freqs[i], dL, GMsun_over_c2_Gpc)
            amp = a0 * _end_taper(freqs[i], coeffs.fmax, lik.taper)
            phase_val = -(p0 - tshift * freqs[i] - phiRef)   # conj: see _PHASE_CONVENTION
            h = combo * (amp * Complex(cos(phase_val), sin(phase_val)))
            d = h - strain[i]
            acc += _safe_div(abs2(d), psd[i])
            hh += _safe_div(abs2(h), psd[i])
            sh = conj(strain[i]) * h
            zr += _safe_div(real(sh), psd[i])
            zi += _safe_div(imag(sh), psd[i])
        else
            acc += _safe_div(abs2(strain[i]), psd[i])
        end
    end
    df = freqs[2] - freqs[1]
    ll = _marg_combine(T(-2) * df * acc, T(4) * df * hh, T(4) * df * zr, T(4) * df * zi, dL, lik)
    return _seobv5f2_apply_deadzone_reject(ll, coeffs, freqs[1])
end

"""
    _seobv5f2_ll_given_coeffs_multi(coeffs, m1, ..., psi, freqs_tuple, strain_tuple, psd_tuple,
        det_tuple, gmst_val, lik, tref)

Multi-detector sum of `_seobv5f2_ll_given_coeffs`'s per-detector terms, with each detector's arrival
time `t_c + dt_geo_d - dt_geo_tref` (`tref == 0`: `t_c` is geocentric; `tref == k`: `t_c` is the
arrival time at detector `k`). With `lik.margphase`, `<d,h>` is summed over detectors before the Bessel
term (the phase is common to all detectors).
"""
@inline function _seobv5f2_ll_given_coeffs_multi(coeffs, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi,
        freqs_tuple, strain_tuple, psd_tuple, det_tuple, gmst_val,
        lik, tref::Int)
    T = typeof(m1)
    phiRef = phase
    pfac = T(0.5) * (1 + cos(theta_jn) * cos(theta_jn))
    cfac = cos(theta_jn)
    dt_ref = ifelse(tref == 0, zero(T),
                    time_delay_from_geocenter_gmst(det_tuple[max(tref, 1)], ra, dec, gmst_val))
    t_c = t_c - _seobv5f2_time_shift(coeffs, lik)

    ll_plain = zero(T)
    hh = zero(T)
    zr = zero(T)
    zi = zero(T)
    ndet = length(freqs_tuple)
    @inbounds for d in 1:ndet
        freqs_d = freqs_tuple[d]; strain_d = strain_tuple[d]; psd_d = psd_tuple[d]
        det_d = det_tuple[d]

        dt_geo = time_delay_from_geocenter_gmst(det_d, ra, dec, gmst_val) - dt_ref
        tshift = -2 * T(pi) * (t_c + dt_geo)      # seconds -> rad/Hz; NOT the Mf-based `t0` (see `_phi_t0`)
        Fplus, Fcross = antenna_response_gmst(det_d, ra, dec, psi, gmst_val)
        combo = Fplus * pfac - im * Fcross * cfac

        det_acc = zero(T)
        det_hh = zero(T)
        det_zr = zero(T)
        det_zi = zero(T)
        for i in eachindex(freqs_d)
            # Same exact out-of-band skip as `_seobv5f2_ll_given_coeffs`.
            if coeffs.success && coeffs.fmin <= freqs_d[i] <= coeffs.fmax
                a0, p0 = _seobv5f2_amp_phi(coeffs, freqs_d[i], dL, GMsun_over_c2_Gpc)
                amp = a0 * _end_taper(freqs_d[i], coeffs.fmax, lik.taper)
                phase_val = -(p0 - tshift * freqs_d[i] - phiRef)   # conj: see _PHASE_CONVENTION
                h = combo * (amp * Complex(cos(phase_val), sin(phase_val)))
                resid = h - strain_d[i]
                det_acc += _safe_div(abs2(resid), psd_d[i])
                det_hh += _safe_div(abs2(h), psd_d[i])
                sh = conj(strain_d[i]) * h
                det_zr += _safe_div(real(sh), psd_d[i])
                det_zi += _safe_div(imag(sh), psd_d[i])
            else
                det_acc += _safe_div(abs2(strain_d[i]), psd_d[i])
            end
        end
        df_d = freqs_d[2] - freqs_d[1]
        ll_plain += T(-2) * df_d * det_acc
        hh += T(4) * df_d * det_hh
        zr += T(4) * df_d * det_zr
        zi += T(4) * df_d * det_zi
    end
    ll = _marg_combine(ll_plain, hh, zr, zi, dL, lik)
    return _seobv5f2_apply_deadzone_reject(ll, coeffs, freqs_tuple[1][1])
end

"""
    _seobv5f2_build_coeffs_soa_kernel!(ts_soa, phis_soa, omegas_soa, Msec_soa, eta_soa, M_soa,
        fmin_soa, fmax_soa, nreal_soa, success_soa, params, model)

Plain (non-differentiated) forward pass building `_seobv5f2_coeffs`'s per-walker output into
struct-of-arrays buffers (`ts_soa`/`phis_soa`/`omegas_soa` are `(256, nwalkers)`, the rest
`(nwalkers,)`) for `_fused_ll_kernel_seobnrv5alignedspinf2_precoeffs!` to consume. Ordinary forward
GPU execution of this model works -- this is that same execution, just with
its output laid out for the precoeffs kernel instead of consumed immediately.
"""
@kernel function _seobv5f2_build_coeffs_soa_kernel!(ts_soa, phis_soa, omegas_soa, Msec_soa, eta_soa, M_soa,
        fmin_soa, fmax_soa, nreal_soa, success_soa, @Const(params), model)
    j = @index(Global, Linear)
    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    mc, eta = _chirp_mass_eta(m1, m2)
    coeffs = _seobv5f2_coeffs(model, mc, eta, chi1, chi2)
    @inbounds for k in 1:256
        ts_soa[k, j] = coeffs.ts[k]
        phis_soa[k, j] = coeffs.phis[k]
        omegas_soa[k, j] = coeffs.omegas[k]
    end
    Msec_soa[j] = coeffs.M_sec
    eta_soa[j] = coeffs.eta
    M_soa[j] = coeffs.M
    fmin_soa[j] = coeffs.fmin
    fmax_soa[j] = coeffs.fmax
    nreal_soa[j] = coeffs.nreal
    success_soa[j] = coeffs.success
end

@kernel function _fused_ll_kernel_seobnrv5alignedspinf2_precoeffs!(ll, @Const(params),
        @Const(ts_soa), @Const(phis_soa), @Const(omegas_soa), @Const(Msec_soa), @Const(eta_soa), @Const(M_soa),
        @Const(fmin_soa), @Const(fmax_soa), @Const(nreal_soa), @Const(success_soa),
        @Const(freqs), @Const(strain), @Const(psd), @Const(geom_flat), @Const(gmst_arr), model, @Const(lik_arr))
    j = @index(Global, Linear)
    lik = _lik_from_array(lik_arr)
    det = _geometry_from_array(geom_flat, 1)
    gmst_val = gmst_arr[1]

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]

    # `Val(256)` (not the plain `Int` literal `256`) is load-bearing here: `ntuple(f, n::Integer)`
    # falls back to Base's generic, non-generated implementation for large `n` (internally
    # `collect`/`Vector`-based, per `_ntuple`), which compiles fine standalone (LLVM optimizes the
    # dynamism away for plain forward CUDA execution) but resulted in invalid LLVM IR
    # (`jl_f_tuple`/`jl_f_apply_type`/`ijl_new_structv`/"unsupported dynamic function invocation")
    # under `Enzyme.autodiff(Enzyme.Forward, ...)` on `CUDABackend()` (real-CUDA
    # gradient validation) -- Enzyme's own compilation pass runs before that dynamism would have
    # been optimized out. `ntuple(f, ::Val{N})` uses Base's `@generated`, fully-unrolled
    # implementation instead: no allocation, no dynamic dispatch, at any `N`.
    coeffs = (success = success_soa[j], M_sec = Msec_soa[j], eta = eta_soa[j], M = M_soa[j],
              nreal = nreal_soa[j],
              ts = SVector{256,Float64}(ntuple(k -> ts_soa[k, j], Val(256))),
              phis = SVector{256,Float64}(ntuple(k -> phis_soa[k, j], Val(256))),
              omegas = SVector{256,Float64}(ntuple(k -> omegas_soa[k, j], Val(256))),
              fmin = fmin_soa[j], fmax = fmax_soa[j])

    ll[j] = _seobv5f2_ll_given_coeffs(coeffs, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi,
        det, gmst_val, freqs, strain, psd, lik)
end

# Multi-detector sibling of the precoeffs kernel above (same SoA coeffs inputs; the detector data
# arrives as tuples like `_fused_waveform_likelihood_kernel_seobnrv5alignedspinf2_multi!`).
@kernel function _fused_ll_kernel_seobnrv5alignedspinf2_precoeffs_multi!(ll, @Const(params),
        @Const(ts_soa), @Const(phis_soa), @Const(omegas_soa), @Const(Msec_soa), @Const(eta_soa), @Const(M_soa),
        @Const(fmin_soa), @Const(fmax_soa), @Const(nreal_soa), @Const(success_soa),
        freqs_tuple, strain_tuple, psd_tuple, @Const(geom_arr), @Const(gmst_arr), model,
        @Const(lik_arr), tref)
    j = @index(Global, Linear)
    lik = _lik_from_array(lik_arr)
    det_tuple = _geometry_tuple(geom_arr, freqs_tuple)
    gmst_val = gmst_arr[1]

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]

    coeffs = (success = success_soa[j], M_sec = Msec_soa[j], eta = eta_soa[j], M = M_soa[j],
              nreal = nreal_soa[j],
              ts = SVector{256,Float64}(ntuple(k -> ts_soa[k, j], Val(256))),
              phis = SVector{256,Float64}(ntuple(k -> phis_soa[k, j], Val(256))),
              omegas = SVector{256,Float64}(ntuple(k -> omegas_soa[k, j], Val(256))),
              fmin = fmin_soa[j], fmax = fmax_soa[j])

    ll[j] = _seobv5f2_ll_given_coeffs_multi(coeffs, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi,
        freqs_tuple, strain_tuple, psd_tuple, det_tuple, gmst_val, lik, tref)
end

# -------------------------------------------------------------------------------------------
# IMRPhenomXP-specialized kernel body: single-detector, dominant-mode precessing-spin.
# `params` is `(15, nwalkers)` (see this file's module docstring, rows 12-15). Mirrors the
# IMRPhenomD/IMRPhenomXAS fast paths' overall shape (frequency-independent setup hoisted out of the
# frequency loop, non-allocating scalar core inside it) but does NOT go through `_extrinsic_setup`
# (that helper's `combo = Fplus*pfac - im*Fcross*cfac` bakes in the ALIGNED-spin plus/cross
# decomposition by `theta_jn`, which is simply wrong for a precessing waveform -- twisting-up
# already produces genuinely frequency-dependent `(hp, hc)` that mixes amplitude and phase between
# polarizations, see `GravitationalWaveforms.IMRPhenomXP`'s own docstring). Antenna response is still computed
# fresh, once per walker (frequency-independent, unlike hp/hc), via the SAME
# `antenna_response_gmst` every other kernel body in this file uses -- not recomputed
# differently, per this file's own Enzyme-correctness discipline (see module docstring).
#
# Parametrization choices (deliberate scope simplifications, not silent physics errors):
#   - `theta_jn` (row 10) is reused directly as `_xp_precession_setup`'s `inclination` argument
#     (angle between L and the line of sight AT f_ref). The full standard PE convention's
#     `theta_jn` is actually the angle between the TOTAL angular momentum J and the line of sight,
#     needing an extra coordinate transform to recover L's own inclination (the
#     `(a1,a2,tilt_1,tilt_2,phi_12,phi_jl)` parametrization is not used here).
#     This choice is self-consistent (the injection-recovery test below validates it end-to-end)
#     and reduces EXACTLY to the existing aligned-spin convention at zero in-plane spin.
#   - `phase` (row 7) is reused as BOTH `_xp_precession_setup`'s `phiRef` (feeds the line-of-sight
#     vector construction, hence `thetaJN`/`zeta_polarization`) AND `_xas_phi_coeffs`'s `phiRef`
#     kwarg (the aligned-spin (2,2) mode's own external phase offset) -- these coincide in
#     LALSuite's own construction too (see `twist_up22`'s docstring in GravitationalWaveforms.jl for the
#     `phiRef` vs `phiRefOrbital` distinction this collapses).
#   - `f_ref = freqs[1]`: the first bin of that walker's own frequency grid, a natural
#     parameter-free reference frequency (no new sampled parameter needed).
@kernel function _fused_waveform_likelihood_kernel_phenomxp!(ll, @Const(params), @Const(freqs), @Const(strain), @Const(psd), @Const(geom_flat), @Const(gmst_arr), model)
    j = @index(Global, Linear)
    T = eltype(params)
    det = _geometry_from_array(geom_flat, 1)
    gmst_val = gmst_arr[1]

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]
    chi1x = params[12, j]; chi1y = params[13, j]; chi2x = params[14, j]; chi2y = params[15, j]

    mc, eta = _chirp_mass_eta(m1, m2)
    M_sec = (mc / eta^T(0.6)) * T(GMsun_over_c3)
    t0 = -2 * T(pi) * t_c / M_sec
    phiRef = phase

    Fplus, Fcross = antenna_response_gmst(det, ra, dec, psi, gmst_val)

    f_ref = freqs[1]
    setup = _xp_precession_setup(m1, m2, chi1x, chi1y, chi1, chi2x, chi2y, chi2,
                                  f_ref, theta_jn, phase; GMsun_over_c3 = T(GMsun_over_c3))

    phi_coeffs = _xas_phi_coeffs(IMRPhenomXAS(), mc, eta, chi1, chi2; t0=-t0, phiRef=phiRef)   # time sign: see _PHASE_CONVENTION
    ampl_coeffs = _xas_ampl_coeffs(IMRPhenomXAS(), mc, eta, chi1, chi2, dL)

    acc = zero(T)
    @inbounds for i in eachindex(freqs)
        amp = _xas_ampl_scalar(ampl_coeffs, freqs[i])
        phase_val = _xas_phi_scalar(phi_coeffs, freqs[i])
        hAS = (amp * Complex(cos(phase_val), sin(phase_val))) * T(_XP_HAS_MODE_NORMALIZATION)

        Mf = M_sec * freqs[i]
        v = cbrt(T(pi) * Mf)
        hp, hc = _xp_twist_up22(v, setup, hAS)
        hp, hc = _xp_apply_zeta_rotation(hp, hc, setup.zeta_polarization)

        h = Fplus * hp + Fcross * hc
        d = h - strain[i]
        acc += _safe_div(abs2(d), psd[i])
    end
    df = freqs[2] - freqs[1]
    ll[j] = T(-2) * df * acc
end

# -------------------------------------------------------------------------------------------
# IMRPhenomXHM-specialized kernel body: single-detector, ALIGNED-SPIN,
# higher-order-mode. `params` is `(11, nwalkers)` -- the EXISTING aligned-spin convention (this
# file's module docstring), not the 15-row `IMRPhenomXP` extension, since `IMRPhenomXHM` has no
# in-plane spin/precession at all. Mirrors `_fused_waveform_likelihood_kernel_phenomxp!`'s overall
# shape (frequency-independent per-walker setup hoisted out of the frequency loop, projection via
# `h = Fplus*hp + Fcross*hc` since `hphc`/`_xhm_hphc_scalar` already returns fully-formed
# polarizations rather than a single scalar `Ampl*exp(im*Phi)`) -- NOT `_fused_waveform_likelihood_
# kernel_phenomxas!`'s pattern, because `IMRPhenomXHM` doesn't share the `Ampl`/`Phi` scalar contract
# (see this file's module docstring, and `GravitationalWaveforms.IMRPhenomXHM`'s own module docstring for why).
#
# Per-walker setup (`_xhm_system_setup`, `_xhm_mode_setup` for `(2,1)`/`(3,3)`/`(4,4)`,
# `_xhm_mode32_setup` for the mode-mixing `(3,2)` mode) is genuinely frequency-independent, exactly
# like `_xp_precession_setup`, and is computed ONCE per walker; the frequency loop then calls
# `_xhm_hphc_scalar` (GravitationalWaveforms.jl's own already-non-allocating per-frequency-bin `(hp,hc)` core --
# `hphc(model, f, ...)`'s public array-based entry point literally just loops calling this) directly,
# with no array allocation anywhere in this kernel body -- the same Enzyme-vs-allocation discipline
# every other kernel in this file follows (see module docstring / `test/test_phenomxhm.jl`'s
# dedicated Enzyme-through-KernelAbstractions testset).
#
# Parametrization choices (deliberate scope simplifications, matching `IMRPhenomXP`'s own precedent):
#   - `theta_jn` (row 10) is passed directly as `_xhm_hphc_scalar`'s `inclination` argument -- exactly
#     the angle `GravitationalWaveforms.hphc`'s own docstring documents ("the angle between the orbital angular
#     momentum and the line of sight"), with NO precession to complicate that identification (unlike
#     `IMRPhenomXP`'s `inclination`-at-`f_ref` caveat) -- this is the SAME identification this file's
#     module docstring already makes for `IMRPhenomD`/`IMRPhenomXAS`'s `theta_jn`.
#   - `phase` (row 7) is passed as `_xhm_system_setup`'s `phiRef` argument (the reference orbital
#     phase at `f_ref`) -- this is the ONLY place `phase` enters; `_xhm_hphc_scalar`'s own `phi`
#     (azimuthal angle) argument is a FIXED `pi/2` (matching `GravitationalWaveforms.hphc`'s own convention,
#     `LAL_PI_2` in `IMRPhenomXHM_MultiMode`'s C source -- NOT `phiRef`, which instead enters through
#     each mode's own `phi0`-dependent phase construction, see that package's module docstring).
#   - `f_ref = freqs[1]`: the first bin of that walker's own frequency grid, same natural
#     parameter-free reference-frequency choice `IMRPhenomXP`'s kernel makes.
#   - **`t_c` has NO native input to `_xhm_system_setup`/`hphc`** (unlike `Phi`'s `t0` kwarg or
#     `_xp_precession_setup`'s implicit handling) -- applied here as a direct post-multiplication in
#     PHYSICAL frequency `f`: `(hp, hc) *= exp(im*2*pi*f*t_c)`. This reproduces the SAME net
#     `phase(f) += 2*pi*f*t_c` convention this file's other kernels get via `t0 = -2*pi*t_c/M_sec`
#     (see module docstring) -- the standard frequency-domain time-translation theorem applied
#     directly to a `(hp,hc)` pair rather than through the dimensionless-`Mf`/`t0` detour `Phi`
#     needs, since `hphc` has no such detour to hook into. Empirically verified (not just derived)
#     via a synthetic injection-recovery check, see `test/test_phenomxhm.jl`.
@kernel function _fused_waveform_likelihood_kernel_phenomxhm!(ll, @Const(params), @Const(freqs), @Const(strain), @Const(psd), @Const(geom_flat), @Const(gmst_arr), model)
    j = @index(Global, Linear)
    T = eltype(params)
    det = _geometry_from_array(geom_flat, 1)
    gmst_val = gmst_arr[1]

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]

    Fplus, Fcross = antenna_response_gmst(det, ra, dec, psi, gmst_val)

    mc, eta = _chirp_mass_eta(m1, m2)
    f_ref = freqs[1]
    sys = _xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, phase;
                             GMsun_over_c3 = T(GMsun_over_c3), GMsun_over_c2_Gpc = T(GMsun_over_c2_Gpc))
    ms21 = _xhm_mode_setup(2, 1, sys)
    ms33 = _xhm_mode_setup(3, 3, sys)
    ms44 = _xhm_mode_setup(4, 4, sys)
    m32 = _xhm_mode32_setup(sys)
    piHalf = T(pi) / T(2)

    acc = zero(T)
    @inbounds for i in eachindex(freqs)
        f = freqs[i]
        hp, hc = _xhm_hphc_scalar(f, sys, ms21, ms33, ms44, m32, theta_jn, piHalf)

        tc_phase = -2 * T(pi) * f * t_c   # time sign: see _PHASE_CONVENTION
        shift = Complex{T}(cos(tc_phase), sin(tc_phase))
        hp *= shift
        hc *= shift

        h = Fplus * hp + Fcross * hc
        d = h - strain[i]
        acc += _safe_div(abs2(d), psd[i])
    end
    df = freqs[2] - freqs[1]
    ll[j] = T(-2) * df * acc
end

"""
    _fused_waveform_likelihood_kernel_phenomxhm_soa!(ll, mc, eta, chi1, chi2, dL, t_c, phase,
                                                       ra, dec, theta_jn, psi, freqs, strain, psd,
                                                       det, gmst_val)

GPU-only structure-of-arrays variant of [`_fused_waveform_likelihood_kernel_phenomxhm!`](@ref) --
identical physics, only the parameter-passing convention differs: separate contiguous 1-D
`CuArray`s per physical field instead of one `(11, nwalkers)` matrix, and `mc`/`eta` precomputed
by the caller (host-side broadcast, before kernel launch) instead of `_chirp_mass_eta` called
inside the kernel body.

Exists because the matrix-plus-in-kernel-`_chirp_mass_eta` path hits an unresolved
`GPUCompiler.jl`/Julia type-inference issue specific to this model's 5-mode call graph --
confirmed by isolation testing on real GPU hardware: every individual piece of this kernel
(`_xhm_system_setup`, each `_xhm_mode_setup` call including the `(3,2)` mixing mode via
`_xhm_mode32_setup`, `_xhm_hphc_scalar`, the per-frequency accumulation loop) compiles and runs
correctly on `CUDABackend()` in isolation, and the combination of ALL of them together compiles
fine too -- PROVIDED every value flowing into `_xhm_system_setup` is a direct, single-index read
from a dedicated 1-D `@Const` array. Reading from a 2-D `params` matrix row, OR computing `mc`/
`eta` via `_chirp_mass_eta` inside the kernel (local OR imported, cross-module OR same-module --
both tested), reproducibly breaks compilation once ANY `_xhm_mode_setup` call is also present,
with the identical symptom (`_xhm_system_setup`'s own `NamedTuple` construction reported as
requiring a dynamic/runtime dispatch it cannot resolve, cascading into every downstream call).
This SoA layout sidesteps both known triggers simultaneously.

Selected automatically by `batched_log_likelihood_kernel!` for `model isa IMRPhenomXHM` on
non-CPU backends -- the CPU path is untouched, still uses the matrix kernel above (which remains
the only one ever exercised by `KernelAbstractions.CPU()`, and is the one the existing
Enzyme-vs-finite-differences validation covers).

The geometry/`gmst_val` are passed as length-12/length-1 `@Const` `CuArray`s (`geom_flat`,
`gmst_arr`), NOT a bare `NTuple`/`Float64` kernel argument -- confirmed necessary, not cosmetic:
Enzyme's GPU kernel-launch activity inference correctly infers `Const` for every `CuArray`
argument derived from an outer `Const(data)` (all of `freqs`/`strain`/`psd`/`mc`/`eta`/...), but
misclassifies a bare non-`CuArray` value (a plain `Tuple` or `Float64`) derived the same way as
`Active` instead, and Enzyme's GPU rule hard-rejects `Active` kernel arguments outright
("Active kernel arguments not supported on GPU") -- confirmed directly by reproducing the error
with the bare form and having it disappear with this one, no other change.
"""
@kernel function _fused_waveform_likelihood_kernel_phenomxhm_soa!(ll, @Const(mc), @Const(eta),
        @Const(chi1), @Const(chi2), @Const(dL), @Const(t_c), @Const(phase), @Const(ra),
        @Const(dec), @Const(theta_jn), @Const(psi), @Const(freqs), @Const(strain), @Const(psd),
        @Const(geom_flat), @Const(gmst_arr))
    j = @index(Global, Linear)
    T = eltype(mc)

    det = _geometry_from_array(geom_flat, 1)
    gmst_val = gmst_arr[1]
    Fplus, Fcross = antenna_response_gmst(det, ra[j], dec[j], psi[j], gmst_val)

    f_ref = freqs[1]
    sys = _xhm_system_setup(mc[j], eta[j], chi1[j], chi2[j], dL[j], f_ref, phase[j];
                             GMsun_over_c3 = T(GMsun_over_c3), GMsun_over_c2_Gpc = T(GMsun_over_c2_Gpc))
    ms21 = _xhm_mode_setup(2, 1, sys)
    ms33 = _xhm_mode_setup(3, 3, sys)
    ms44 = _xhm_mode_setup(4, 4, sys)
    m32 = _xhm_mode32_setup(sys)
    piHalf = T(pi) / T(2)

    acc = zero(T)
    @inbounds for i in eachindex(freqs)
        f = freqs[i]
        hp, hc = _xhm_hphc_scalar(f, sys, ms21, ms33, ms44, m32, theta_jn[j], piHalf)

        tc_phase = -2 * T(pi) * f * t_c[j]   # time sign: see _PHASE_CONVENTION
        shift = Complex{T}(cos(tc_phase), sin(tc_phase))
        hp *= shift
        hc *= shift

        h = Fplus * hp + Fcross * hc
        d = h - strain[i]
        acc += _safe_div(abs2(d), psd[i])
    end
    df = freqs[2] - freqs[1]
    ll[j] = T(-2) * df * acc
end

# -------------------------------------------------------------------------------------------
# IMRPhenomXPHM-specialized kernel body: single-detector, PRECESSING +
# higher-order-mode. `params` is `(15, nwalkers)` -- `IMRPhenomXP`'s own precessing-spin row
# convention (this file's module docstring, rows 12-15), since `IMRPhenomXPHM` needs exactly the
# same physical inputs `IMRPhenomXP` does (no new sampled parameters). Combines
# `_fused_waveform_likelihood_kernel_phenomxp!`'s precession setup (`_xp_precession_setup`) with
# `_fused_waveform_likelihood_kernel_phenomxhm!`'s higher-mode/no-native-`t_c` handling: this model
# shares neither the single-scalar `Ampl`/`Phi` contract (`IMRPhenomXP`'s own escape hatch) nor a
# `t_c`/`t0` input (`IMRPhenomXHM`'s own escape hatch) -- it needs BOTH workarounds simultaneously.
#
# Per-walker setup, all frequency-independent and computed ONCE per walker (same discipline as
# every other kernel here): `_xp_precession_setup` (Euler-angle/`zeta_polarization`/`chi_p`
# geometry) and `_xhm_system_setup` (aligned-spin `(2,2)`-anchored coefficient bundle) plus the
# three non-mixing per-mode setups and the `(3,2)`-mixing setup -- exactly
# `hphc(::IMRPhenomXPHM, ...)`'s own top-level assembly (`IMRPhenomXPHM.jl`), just with the
# frequency loop inlined here instead of going through that array-based public entry point.
#
# **`final_spin_override`: `_xhm_system_setup`
# is called with the PRECESSING final spin (`_xas_precessing_final_spin(eta, chi1, chi2,
# setup.chi_p)`), matching LALSuite's default `PhenomXPFinalSpinMod = 0` and
# `hphc(::IMRPhenomXPHM, ...)`'s own (fixed) behavior exactly** -- getting this kernel's own final
# spin wrong (e.g. the aligned-only `_xhm_system_setup` call with no
# override) would leave a ~7% amplitude gap, invisible to any test
# that doesn't compare against real LALSuite output.
#
# Parametrization choices, identical to `_fused_waveform_likelihood_kernel_phenomxp!`'s own (this
# model's extrinsic/spin parameter set IS `IMRPhenomXP`'s, unchanged):
#   - `theta_jn` (row 10) is `_xp_precession_setup`'s `inclination` argument (angle between `L` and
#     the line of sight AT `f_ref`, not the full-PE-convention `J`-based `theta_jn` -- same
#     documented simplification `IMRPhenomXP`'s own kernel makes).
#   - `phase` (row 7) is `_xp_precession_setup`'s `phiRef` -- and, matching
#     `hphc(::IMRPhenomXPHM, ...)`'s own documented convention (NOT `IMRPhenomXP`'s `twist_up22`
#     default), is deliberately NOT ALSO passed to `_xhm_system_setup`'s own `phiRef` (`zero(phase)`
#     instead): LALSuite's Convention 1 overwrites `pWF->phi0 = 0` before generating any
#     co-precessing mode for a MULTI-harmonic model, so passing the real `phase` there too would be
#     a genuine `mprime`-dependent misalignment bug (see `IMRPhenomXPHM.jl`'s own module docstring),
#     not the harmless global-phase gauge freedom it is for `IMRPhenomXP`'s single dominant mode.
#   - `f_ref = freqs[1]`, same natural parameter-free choice every precessing/HM kernel here makes.
#   - `_xphm_hphc_scalar` does NOT apply the `zeta_polarization` rotation internally (see its own
#     docstring) -- applied here explicitly via `_xp_apply_zeta_rotation`, exactly once, to the
#     already-five-mode-summed output, matching `hphc(::IMRPhenomXPHM, ...)`'s own placement.
#   - `t_c`: same direct `exp(im*2*pi*f*t_c)` post-multiplication
#     `_fused_waveform_likelihood_kernel_phenomxhm!` uses (`hphc(::IMRPhenomXPHM, ...)` has no
#     native `t_c`/`t0` input either), applied AFTER the zeta rotation (order does not matter --
#     both are just multiplicative phase factors on `(hp, hc)` -- but matching the aligned-`XHM`
#     kernel's own ordering for consistency).
#
# Tests: `test/test_phenomxphm.jl` and `test/test_enzyme_ka_phenomxphm.jl` (Enzyme gradient against
# finite differences).
@kernel function _fused_waveform_likelihood_kernel_phenomxphm!(ll, @Const(params), @Const(freqs), @Const(strain), @Const(psd), @Const(geom_flat), @Const(gmst_arr), model)
    j = @index(Global, Linear)
    T = eltype(params)
    det = _geometry_from_array(geom_flat, 1)
    gmst_val = gmst_arr[1]

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]
    chi1x = params[12, j]; chi1y = params[13, j]; chi2x = params[14, j]; chi2y = params[15, j]

    mc, eta = _chirp_mass_eta(m1, m2)

    Fplus, Fcross = antenna_response_gmst(det, ra, dec, psi, gmst_val)

    f_ref = freqs[1]
    setup = _xphm_prec_setup(model, m1, m2, chi1x, chi1y, chi1, chi2x, chi2y, chi2,
                                  f_ref, theta_jn, phase; GMsun_over_c3 = T(GMsun_over_c3))

    final_spin = setup.final_spin   # version-dependent (102: chi_p, 223: MSA)
    sys = _xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, zero(phase);
                             GMsun_over_c3 = T(GMsun_over_c3), GMsun_over_c2_Gpc = T(GMsun_over_c2_Gpc),
                             final_spin_override = final_spin)
    ms21 = _xhm_mode_setup(2, 1, sys)
    ms33 = _xhm_mode_setup(3, 3, sys)
    ms44 = _xhm_mode_setup(4, 4, sys)
    m32 = _xhm_mode32_setup(sys)

    acc = zero(T)
    @inbounds for i in eachindex(freqs)
        f = freqs[i]
        hp, hc = _xphm_hphc_scalar(f, setup, sys, ms21, ms33, ms44, m32)
        hp, hc = _xp_apply_zeta_rotation(hp, hc, setup.zeta_polarization)

        tc_phase = -2 * T(pi) * f * t_c   # time sign: see _PHASE_CONVENTION
        shift = Complex{T}(cos(tc_phase), sin(tc_phase))
        hp *= shift
        hc *= shift

        h = Fplus * hp + Fcross * hc
        d = h - strain[i]
        acc += _safe_div(abs2(d), psd[i])
    end
    df = freqs[2] - freqs[1]
    ll[j] = T(-2) * df * acc
end

# -------------------------------------------------------------------------------------------
# Multi-detector kernel bodies. Mirror the single-detector IMRPhenomD/IMRPhenomXAS
# fast paths above exactly, except: (a) detector-independent per-walker quantities (mc, eta,
# M_sec, phiRef, pfac, cfac, ampl_coeffs) are hoisted OUTSIDE the `1:ndet` loop (computed once,
# not once per detector), and (b) `t0`/`combo`/`phi_coeffs` are recomputed inside the `1:ndet`
# loop since they depend on that detector's own time-delay-shifted `t_c` and antenna response.
# See the multi-detector design note above `_extrinsic_setup` for the full reasoning.
# -------------------------------------------------------------------------------------------

@kernel function _fused_waveform_likelihood_kernel_phenomd_multi!(ll, @Const(params), freqs_tuple, strain_tuple,
                                                                    psd_tuple, det_tuple,
                                                                    gmst_val, model)
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
    ndet = length(freqs_tuple)
    @inbounds for d in 1:ndet
        freqs_d = freqs_tuple[d]; strain_d = strain_tuple[d]; psd_d = psd_tuple[d]
        det_d = det_tuple[d]

        dt_geo = time_delay_from_geocenter_gmst(det_d, ra, dec, gmst_val)
        t0 = -2 * T(pi) * (t_c + dt_geo) / M_sec
        Fplus, Fcross = antenna_response_gmst(det_d, ra, dec, psi, gmst_val)
        combo = Fplus * pfac - im * Fcross * cfac
        phi_coeffs = _phi_coeffs(model, mc, eta, chi1, chi2; t0=t0, phiRef=phiRef)

        det_acc = zero(T)
        for i in eachindex(freqs_d)
            amp = _ampl_scalar(ampl_coeffs, freqs_d[i])
            phase_val = -_phi_scalar(phi_coeffs, freqs_d[i])   # conj: see _PHASE_CONVENTION
            h = combo * (amp * Complex(cos(phase_val), sin(phase_val)))
            resid = h - strain_d[i]
            det_acc += _safe_div(abs2(resid), psd_d[i])
        end
        df_d = freqs_d[2] - freqs_d[1]
        acc += T(-2) * df_d * det_acc
    end
    ll[j] = acc
end

@kernel function _fused_waveform_likelihood_kernel_phenomxas_multi!(ll, @Const(params), freqs_tuple, strain_tuple,
                                                                      psd_tuple, det_tuple,
                                                                      gmst_val, model, @Const(lik_arr), tref)
    j = @index(Global, Linear)
    T = eltype(params)
    lik = _lik_from_array(lik_arr)

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]

    ll[j] = _xas_ll_given_physical_multi(model, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi,
                                         freqs_tuple, strain_tuple, psd_tuple, det_tuple,
                                         gmst_val, lik, tref)
end

# AlignedSpinPNHamF2 multi-detector kernel body: mirrors
# `_fused_waveform_likelihood_kernel_phenomxas_multi!` exactly, using the `_asphf2_*` scalar core
# instead of the `_xas_*` one -- same 11-row aligned-spin convention, so the per-detector
# `t0`/antenna-response structure is identical.
@kernel function _fused_waveform_likelihood_kernel_alignedspinpnhamf2_multi!(ll, @Const(params), freqs_tuple, strain_tuple,
                                                                                psd_tuple, det_tuple,
                                                                                gmst_val, model)
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
    ndet = length(freqs_tuple)
    @inbounds for d in 1:ndet
        freqs_d = freqs_tuple[d]; strain_d = strain_tuple[d]; psd_d = psd_tuple[d]
        det_d = det_tuple[d]

        dt_geo = time_delay_from_geocenter_gmst(det_d, ra, dec, gmst_val)
        t0 = -2 * T(pi) * (t_c + dt_geo) / M_sec
        Fplus, Fcross = antenna_response_gmst(det_d, ra, dec, psi, gmst_val)
        combo = Fplus * pfac - im * Fcross * cfac
        phi_coeffs = _asphf2_phi_coeffs(model, mc, eta, chi1, chi2; t0=t0, phiRef=phiRef)

        det_acc = zero(T)
        for i in eachindex(freqs_d)
            amp = _asphf2_ampl_scalar(ampl_coeffs, freqs_d[i])
            phase_val = -_asphf2_phi_scalar(phi_coeffs, freqs_d[i])   # conj: see _PHASE_CONVENTION
            h = combo * (amp * Complex(cos(phase_val), sin(phase_val)))
            resid = h - strain_d[i]
            det_acc += _safe_div(abs2(resid), psd_d[i])
        end
        df_d = freqs_d[2] - freqs_d[1]
        acc += T(-2) * df_d * det_acc
    end
    ll[j] = acc
end

# SEOBNRv5AlignedSpinF2 multi-detector kernel body: mirrors
# `_fused_waveform_likelihood_kernel_alignedspinpnhamf2_multi!` exactly, using `_seobv5f2_*` instead
# of `_asphf2_*` -- `coeffs` (the dense post-adiabatic table) is detector-independent, so it is
# hoisted OUTSIDE the `1:ndet` loop exactly like `ampl_coeffs` in the sibling kernel. No explicit
# `f_min` is passed (uses this model's own default) -- see the single-detector sibling kernel
# above's comment for why that, not a per-walker/per-detector frequency, is the correct choice here.
@kernel function _fused_waveform_likelihood_kernel_seobnrv5alignedspinf2_multi!(ll, @Const(params), freqs_tuple, strain_tuple,
                                                                                    psd_tuple, det_tuple,
                                                                                    gmst_val, model, @Const(lik_arr), tref)
    j = @index(Global, Linear)
    lik = _lik_from_array(lik_arr)

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]

    mc, eta = _chirp_mass_eta(m1, m2)
    coeffs = _seobv5f2_coeffs(model, mc, eta, chi1, chi2)
    ll[j] = _seobv5f2_ll_given_coeffs_multi(coeffs, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi,
        freqs_tuple, strain_tuple, psd_tuple, det_tuple, gmst_val, lik, tref)
end

# -------------------------------------------------------------------------------------------
# IMRPhenomXP multi-detector kernel body (follows the single-detector `IMRPhenomXP`
# kernel above, mirroring the structural pattern of `_fused_waveform_likelihood_kernel_
# {phenomd,phenomxas}_multi!` above). `params` is `(15, nwalkers)` (this file's module docstring, rows 12-15).
#
# The ONE structural difference from the aligned-spin multi-detector kernels above:
#   - `setup` (from `_xp_precession_setup`) IS detector-independent (depends only on physical
#     params: masses, spins, `f_ref`, `inclination`=`theta_jn`, `phiRef`=`phase` -- no detector
#     geometry), so it is hoisted OUTSIDE the `1:ndet` loop, exactly like `ampl_coeffs`.
#     `f_ref` itself is taken from the FIRST detector's first frequency bin
#     (`freqs_tuple[1][1]`) -- same "natural parameter-free reference frequency" choice the
#     single-detector kernel makes with its own single grid, just anchored to a representative
#     detector since `setup` can only be computed once.
#   - Everything that depends on that detector's own time-delay-shifted `t_c` (via `_time_delay_
#     scalar`, exactly as in the aligned-spin multi-detector kernels) MUST be recomputed per
#     detector, not hoisted: `t0`/`phi_coeffs` (the co-precessing-frame aligned-(2,2)-mode phase,
#     `_xas_phi_coeffs`), plus `Fplus`/`Fcross` (antenna response, detector-specific geometry).
#     The frequency loop inside each detector iteration then reproduces the single-detector
#     kernel's own per-frequency body exactly (`hAS` -> twisting-up -> zeta rotation -> antenna
#     projection -> residual).
# -------------------------------------------------------------------------------------------

@kernel function _fused_waveform_likelihood_kernel_phenomxp_multi!(ll, @Const(params), freqs_tuple, strain_tuple,
                                                                      psd_tuple, det_tuple,
                                                                      gmst_val, model)
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
    f_ref = freqs_tuple[1][1]
    setup = _xp_precession_setup(m1, m2, chi1x, chi1y, chi1, chi2x, chi2y, chi2,
                                  f_ref, theta_jn, phase; GMsun_over_c3 = T(GMsun_over_c3))
    ampl_coeffs = _xas_ampl_coeffs(IMRPhenomXAS(), mc, eta, chi1, chi2, dL)

    acc = zero(T)
    ndet = length(freqs_tuple)
    @inbounds for d in 1:ndet
        freqs_d = freqs_tuple[d]; strain_d = strain_tuple[d]; psd_d = psd_tuple[d]
        det_d = det_tuple[d]

        # Detector-specific (recomputed every iteration): t0/phi_coeffs (time-delay-shifted t_c)
        # and Fplus/Fcross (detector geometry).
        dt_geo = time_delay_from_geocenter_gmst(det_d, ra, dec, gmst_val)
        t0 = -2 * T(pi) * (t_c + dt_geo) / M_sec
        Fplus, Fcross = antenna_response_gmst(det_d, ra, dec, psi, gmst_val)
        phi_coeffs = _xas_phi_coeffs(IMRPhenomXAS(), mc, eta, chi1, chi2; t0=-t0, phiRef=phiRef)   # time sign: see _PHASE_CONVENTION

        det_acc = zero(T)
        for i in eachindex(freqs_d)
            amp = _xas_ampl_scalar(ampl_coeffs, freqs_d[i])
            phase_val = _xas_phi_scalar(phi_coeffs, freqs_d[i])
            hAS = (amp * Complex(cos(phase_val), sin(phase_val))) * T(_XP_HAS_MODE_NORMALIZATION)

            Mf = M_sec * freqs_d[i]
            v = cbrt(T(pi) * Mf)
            hp, hc = _xp_twist_up22(v, setup, hAS)
            hp, hc = _xp_apply_zeta_rotation(hp, hc, setup.zeta_polarization)

            h = Fplus * hp + Fcross * hc
            resid = h - strain_d[i]
            det_acc += _safe_div(abs2(resid), psd_d[i])
        end
        df_d = freqs_d[2] - freqs_d[1]
        acc += T(-2) * df_d * det_acc
    end
    ll[j] = acc
end

# -------------------------------------------------------------------------------------------
# IMRPhenomXHM multi-detector kernel body (follows the single-detector `IMRPhenomXHM`
# kernel above, mirroring the structural pattern of `_fused_waveform_likelihood_kernel_phenomxp_multi!`
# above). `params` is
# `(11, nwalkers)`, the EXISTING aligned-spin convention -- `IMRPhenomXHM` has no in-plane spin at
# all, so it does NOT get `IMRPhenomXP`'s 15-row extension here either.
#
# Detector-INDEPENDENT per-walker setup (hoisted OUTSIDE the `1:ndet` loop, exactly like
# `_xp_precession_setup`'s `setup`/`ampl_coeffs` in the `IMRPhenomXP` multi-detector kernel):
# `mc`/`eta`, `_xhm_system_setup`/`_xhm_mode_setup` (`(2,1),(3,3),(4,4)`)/`_xhm_mode32_setup`
# (`(3,2)`) -- genuinely frequency- and detector-independent physics. `f_ref` is taken from the
# FIRST detector's first frequency bin (`freqs_tuple[1][1]`), the same "anchored to a
# representative detector" choice the `IMRPhenomXP` multi-detector kernel already makes, since
# `sys` can only be built once and needs one reference grid.
#
# Detector-SPECIFIC quantities (recomputed INSIDE the `1:ndet` loop): `Fplus`/`Fcross` (antenna
# response depends on that detector's own geometry) and the coalescence-time phase shift. The
# single-detector `IMRPhenomXHM` kernel applies `t_c` as a direct `exp(im*2*pi*f*t_c)`
# post-multiplication on `(hp,hc)` (see that kernel body's own comment block -- `_xhm_system_setup`
# has no native `t_c` input at all, unlike `Phi`'s `t0` kwarg). The physically correct multi-
# detector generalization, matching how every OTHER multi-detector kernel in this file adds
# `time_delay_from_geocenter_gmst`'s per-detector light-travel-time offset on top of the shared `t_c` (via
# their own `t0` mechanism): each detector's own effective time offset is `t_c + dt_geo`, so the
# phase shift becomes `exp(im*2*pi*f*(t_c + dt_geo))`, with `dt_geo = time_delay_from_geocenter_gmst(det_d,
# ra, dec, gmst_val)` recomputed per detector since it depends on that detector's own geometry.
# This correctly reduces to the single-detector formula when `dt_geo = 0` (e.g. a source directly
# overhead the geocenter, or trivially when there is only one detector to align against) and
# applies the SAME time-translation term the aligned-spin/`IMRPhenomXP` multi-detector kernels
# apply via `t0 = -2*pi*(t_c+dt_geo)/M_sec` (their `-t0*fgrid` term inside `_phi_scalar` expands to
# exactly `2*pi*f*(t_c+dt_geo)`, the identical phase this kernel adds directly here) -- same
# principle, just applied without the dimensionless-`Mf`/`t0` detour since `hphc` has no such
# detour to hook into (see the single-detector kernel's own derivation). Verified empirically (not
# just derived), see `test/test_phenomxhm.jl`'s dedicated multi-detector time-delay testset
# (nonzero sky position so H1/L1's `dt_geo` differ meaningfully -- this is the test that actually
# exercises this generalization, unlike the "two identical detectors" equivalence check).
#
# Antenna projection per detector: `h = Fplus*hp + Fcross*hc` (after the per-detector time-shift),
# exactly like the single-detector kernel and the `IMRPhenomXP` multi-detector kernel.
# -------------------------------------------------------------------------------------------

@kernel function _fused_waveform_likelihood_kernel_phenomxhm_multi!(ll, @Const(params), freqs_tuple, strain_tuple,
                                                                       psd_tuple, det_tuple,
                                                                       gmst_val, model)
    j = @index(Global, Linear)
    T = eltype(params)

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]

    # Detector-independent (hoisted outside the 1:ndet loop below): mc/eta and the per-walker
    # mode setups, anchored to the FIRST detector's first frequency bin.
    mc, eta = _chirp_mass_eta(m1, m2)
    f_ref = freqs_tuple[1][1]
    sys = _xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, phase;
                             GMsun_over_c3 = T(GMsun_over_c3), GMsun_over_c2_Gpc = T(GMsun_over_c2_Gpc))
    ms21 = _xhm_mode_setup(2, 1, sys)
    ms33 = _xhm_mode_setup(3, 3, sys)
    ms44 = _xhm_mode_setup(4, 4, sys)
    m32 = _xhm_mode32_setup(sys)
    piHalf = T(pi) / T(2)

    acc = zero(T)
    ndet = length(freqs_tuple)
    @inbounds for d in 1:ndet
        freqs_d = freqs_tuple[d]; strain_d = strain_tuple[d]; psd_d = psd_tuple[d]
        det_d = det_tuple[d]

        # Detector-specific (recomputed every iteration): Fplus/Fcross (detector geometry) and
        # dt_geo (that detector's own light-travel-time offset from the geocenter), folded into
        # the t_c phase shift as t_c + dt_geo.
        dt_geo = time_delay_from_geocenter_gmst(det_d, ra, dec, gmst_val)
        Fplus, Fcross = antenna_response_gmst(det_d, ra, dec, psi, gmst_val)

        det_acc = zero(T)
        for i in eachindex(freqs_d)
            f = freqs_d[i]
            hp, hc = _xhm_hphc_scalar(f, sys, ms21, ms33, ms44, m32, theta_jn, piHalf)

            tc_phase = -2 * T(pi) * f * (t_c + dt_geo)   # time sign: see _PHASE_CONVENTION
            shift = Complex{T}(cos(tc_phase), sin(tc_phase))
            hp *= shift
            hc *= shift

            h = Fplus * hp + Fcross * hc
            resid = h - strain_d[i]
            det_acc += _safe_div(abs2(resid), psd_d[i])
        end
        df_d = freqs_d[2] - freqs_d[1]
        acc += T(-2) * df_d * det_acc
    end
    ll[j] = acc
end

"""
    _fused_waveform_likelihood_kernel_phenomxhm_multi_soa!(ll, mc, eta, chi1, chi2, dL, t_c,
        phase, ra, dec, theta_jn, psi, freqs_tuple, strain_tuple, psd_tuple, det_tuple, gmst_val)

GPU-only structure-of-arrays variant of [`_fused_waveform_likelihood_kernel_phenomxhm_multi!`](@ref) --
same relationship, same reason, as [`_fused_waveform_likelihood_kernel_phenomxhm_soa!`](@ref) has
to its own single-detector non-SoA counterpart (see that docstring for the full GPU-compile-failure
investigation). Only the parameter-passing convention differs from the matrix-based kernel above;
the per-detector loop body is unchanged.
"""
@kernel function _fused_waveform_likelihood_kernel_phenomxhm_multi_soa!(ll, @Const(mc), @Const(eta),
        @Const(chi1), @Const(chi2), @Const(dL), @Const(t_c), @Const(phase), @Const(ra),
        @Const(dec), @Const(theta_jn), @Const(psi), freqs_tuple, strain_tuple, psd_tuple,
        det_tuple, gmst_val)
    j = @index(Global, Linear)
    T = eltype(mc)

    f_ref = freqs_tuple[1][1]
    sys = _xhm_system_setup(mc[j], eta[j], chi1[j], chi2[j], dL[j], f_ref, phase[j];
                             GMsun_over_c3 = T(GMsun_over_c3), GMsun_over_c2_Gpc = T(GMsun_over_c2_Gpc))
    ms21 = _xhm_mode_setup(2, 1, sys)
    ms33 = _xhm_mode_setup(3, 3, sys)
    ms44 = _xhm_mode_setup(4, 4, sys)
    m32 = _xhm_mode32_setup(sys)
    piHalf = T(pi) / T(2)

    acc = zero(T)
    ndet = length(freqs_tuple)
    @inbounds for d in 1:ndet
        freqs_d = freqs_tuple[d]; strain_d = strain_tuple[d]; psd_d = psd_tuple[d]
        det_d = det_tuple[d]

        dt_geo = time_delay_from_geocenter_gmst(det_d, ra[j], dec[j], gmst_val)
        Fplus, Fcross = antenna_response_gmst(det_d, ra[j], dec[j], psi[j], gmst_val)

        det_acc = zero(T)
        for i in eachindex(freqs_d)
            f = freqs_d[i]
            hp, hc = _xhm_hphc_scalar(f, sys, ms21, ms33, ms44, m32, theta_jn[j], piHalf)

            tc_phase = -2 * T(pi) * f * (t_c[j] + dt_geo)   # time sign: see _PHASE_CONVENTION
            shift = Complex{T}(cos(tc_phase), sin(tc_phase))
            hp *= shift
            hc *= shift

            h = Fplus * hp + Fcross * hc
            resid = h - strain_d[i]
            det_acc += _safe_div(abs2(resid), psd_d[i])
        end
        df_d = freqs_d[2] - freqs_d[1]
        acc += T(-2) * df_d * det_acc
    end
    ll[j] = acc
end

# -------------------------------------------------------------------------------------------
# IMRPhenomXPHM multi-detector kernel body (follow-on to the single-detector
# `IMRPhenomXPHM` kernel above). Combines `_fused_waveform_likelihood_kernel_phenomxp_multi!`'s
# detector-independent precession setup with `_fused_waveform_likelihood_kernel_phenomxhm_multi!`'s
# per-detector `t_c + dt_geo` post-multiplication -- the same two-way combination the
# single-detector `IMRPhenomXPHM` kernel makes of its own two single-detector ancestors, just with
# the "hoist detector-independent physics outside the `1:ndet` loop" pattern every OTHER
# multi-detector kernel in this file already uses. `params` is `(15, nwalkers)`, `IMRPhenomXP`'s
# own precessing-spin row convention (unchanged from the single-detector `IMRPhenomXPHM` kernel).
#
# Detector-INDEPENDENT (hoisted outside the `1:ndet` loop, computed once per walker): `mc`/`eta`,
# `setup` (`_xp_precession_setup`, anchored to `freqs_tuple[1][1]` as `f_ref` -- the same
# "representative detector" choice `_fused_waveform_likelihood_kernel_phenomxp_multi!` makes),
# `final_spin` (`_xas_precessing_final_spin(eta, chi1, chi2, setup.chi_p)` -- **omitting
# this would silently reintroduce ~7% amplitude gap here too**, exactly as for the
# single-detector kernel), `sys`/`ms21`/`ms33`/`ms44`/`m32` (`_xhm_system_setup` with
# `final_spin_override = final_spin`, then the four per-mode setups).
#
# Detector-SPECIFIC (recomputed inside the `1:ndet` loop): `Fplus`/`Fcross` (antenna response) and
# `dt_geo` (`time_delay_from_geocenter_gmst`, that detector's own light-travel-time offset from the geocenter).
# Per-frequency body: `_xphm_hphc_scalar` -> `_xp_apply_zeta_rotation` (once, to the five-mode sum,
# exactly like the single-detector kernel) -> `exp(im*2*pi*f*(t_c+dt_geo))` post-multiplication
# (the SAME per-detector time generalization `_fused_waveform_likelihood_kernel_phenomxhm_multi!`
# already uses and its own comment block derives/verifies, since `hphc(::IMRPhenomXPHM, ...)` has
# no native `t_c` input to hook a `t0`-style per-detector recompute into either) -> antenna
# projection -> residual.
# -------------------------------------------------------------------------------------------

@kernel function _fused_waveform_likelihood_kernel_phenomxphm_multi!(ll, @Const(params), freqs_tuple, strain_tuple,
                                                                        psd_tuple, det_tuple,
                                                                        gmst_val, model)
    j = @index(Global, Linear)
    T = eltype(params)

    m1 = params[1, j]; m2 = params[2, j]; chi1 = params[3, j]; chi2 = params[4, j]
    dL = params[5, j]; t_c = params[6, j]; phase = params[7, j]
    ra = params[8, j]; dec = params[9, j]; theta_jn = params[10, j]; psi = params[11, j]
    chi1x = params[12, j]; chi1y = params[13, j]; chi2x = params[14, j]; chi2y = params[15, j]

    # Detector-independent (hoisted outside the 1:ndet loop below): mc/eta, precession setup,
    # final_spin_override, and the per-walker mode setups, all anchored to the FIRST
    # detector's first frequency bin.
    mc, eta = _chirp_mass_eta(m1, m2)
    f_ref = freqs_tuple[1][1]
    setup = _xphm_prec_setup(model, m1, m2, chi1x, chi1y, chi1, chi2x, chi2y, chi2,
                                  f_ref, theta_jn, phase; GMsun_over_c3 = T(GMsun_over_c3))

    final_spin = setup.final_spin   # version-dependent (102: chi_p, 223: MSA)
    sys = _xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, zero(phase);
                             GMsun_over_c3 = T(GMsun_over_c3), GMsun_over_c2_Gpc = T(GMsun_over_c2_Gpc),
                             final_spin_override = final_spin)
    ms21 = _xhm_mode_setup(2, 1, sys)
    ms33 = _xhm_mode_setup(3, 3, sys)
    ms44 = _xhm_mode_setup(4, 4, sys)
    m32 = _xhm_mode32_setup(sys)

    acc = zero(T)
    ndet = length(freqs_tuple)
    @inbounds for d in 1:ndet
        freqs_d = freqs_tuple[d]; strain_d = strain_tuple[d]; psd_d = psd_tuple[d]
        det_d = det_tuple[d]

        # Detector-specific (recomputed every iteration): Fplus/Fcross (detector geometry) and
        # dt_geo (that detector's own light-travel-time offset from the geocenter), folded into
        # the t_c phase shift as t_c + dt_geo.
        dt_geo = time_delay_from_geocenter_gmst(det_d, ra, dec, gmst_val)
        Fplus, Fcross = antenna_response_gmst(det_d, ra, dec, psi, gmst_val)

        det_acc = zero(T)
        for i in eachindex(freqs_d)
            f = freqs_d[i]
            hp, hc = _xphm_hphc_scalar(f, setup, sys, ms21, ms33, ms44, m32)
            hp, hc = _xp_apply_zeta_rotation(hp, hc, setup.zeta_polarization)

            tc_phase = -2 * T(pi) * f * (t_c + dt_geo)   # time sign: see _PHASE_CONVENTION
            shift = Complex{T}(cos(tc_phase), sin(tc_phase))
            hp *= shift
            hc *= shift

            h = Fplus * hp + Fcross * hc
            resid = h - strain_d[i]
            det_acc += _safe_div(abs2(resid), psd_d[i])
        end
        df_d = freqs_d[2] - freqs_d[1]
        acc += T(-2) * df_d * det_acc
    end
    ll[j] = acc
end

"""
    batched_log_likelihood_kernel!(ll, params, freqs, cfg, data;
                                    backend = KernelAbstractions.CPU(), model = IMRPhenomD())

KernelAbstractions-based fused waveform-generation + Gaussian-likelihood evaluation.

Computes, for each walker `j`, the standard frequency-domain Gaussian matched-filter
log-likelihood (the convention of bilby, LALInference, PyCBC, gwpy)

    ll[j] = -0.5 * <d-h_j|d-h_j> = -2 * df * sum(abs2.(h_j .- data.strain) ./ data.psd)

(`<a|b> = 4 * Re ∫ a(f) b*(f) / Sn(f) df`, discretized with bin width `df = freqs[2] - freqs[1]`, a
uniform FFT-derived grid; `Sn` the one-sided PSD in `data.psd`, in `GravitationalWaveData.psd`'s units, NOT
prescaled) — as a single kernel launch with one thread per walker, fusing waveform generation and
the likelihood reduction instead of materializing an intermediate `(ntime, nwalkers)` waveform
matrix and reducing it in a second pass.

`params` is `(11, nwalkers)`, the PHYSICAL parametrization documented at the top of this file
(`m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi`) — `waveform_j` now includes
distance scaling, coalescence-time/phase alignment, the plus/cross inclination decomposition, and
single-detector antenna-response projection (`data.detector`/`data.gps_time`), not just the bare
face-on dominant-mode strain the pre-Task-4 4-parameter version used.

Written generically against `KernelAbstractions.Backend` (not hardcoded to `CPU`) so the same
kernel body can later target `CUDABackend`/`MetalBackend`/`ROCBackend` without a rewrite; only
`KernelAbstractions.CPU()` is exercised/tested here since no GPU hardware is available in this
environment. `model` defaults to `IMRPhenomD()` but is a normal dispatch argument, so `IMRPhenomXAS()`
works too since both share the same `Ampl`/`Phi` contract — this is not itself wired into the
sampler's default call path, which is out of scope for this change.

`model isa IMRPhenomD` dispatches to a specialized kernel body (`_fused_waveform_likelihood_kernel_phenomd!`)
that uses IMRPhenomD's non-allocating scalar coefficient/evaluation core instead of the array-in/
array-out `Ampl`/`Phi`; `model isa IMRPhenomXAS` likewise dispatches to
`_fused_waveform_likelihood_kernel_phenomxas!`, using IMRPhenomXAS's own scalar core
(`_xas_phi_coeffs`/`_xas_phi_scalar`/`_xas_ampl_coeffs`/`_xas_ampl_scalar`). `model isa IMRPhenomXP`
(single-detector only, `params` is `(15, nwalkers)`) dispatches to
`_fused_waveform_likelihood_kernel_phenomxp!`, which twists up that same IMRPhenomXAS scalar core via
`GravitationalWaveforms._xp_precession_setup`/`_xp_twist_up22`/`_xp_apply_zeta_rotation` into precessing
`(hp, hc)` before projecting through the antenna response — see that kernel body's own comment
block for the parametrization choices. `model isa IMRPhenomXHM` (single-detector
method, `params` is back to `(11, nwalkers)` since this model is aligned-spin/non-precessing)
dispatches to `_fused_waveform_likelihood_kernel_phenomxhm!`, which calls `GravitationalWaveforms._xhm_hphc_scalar`
(the non-allocating per-frequency-bin core behind the public `hphc(::IMRPhenomXHM, ...)`) directly,
projects via the SAME `h = Fplus*hp + Fcross*hc` pattern `IMRPhenomXP` uses (since this model also
doesn't share the single-scalar `Ampl`/`Phi` contract), and applies the `t_c` frequency-domain
time-shift as a direct `exp(im*2*pi*f*t_c)` post-multiplication (see this file's module docstring
for the derivation/verification of that convention, since `hphc` has no native `t_c` input to hook
into the way `Phi`'s `t0` kwarg does). `model isa IMRPhenomXPHM` (single-detector
only, `params` back to `(15, nwalkers)` since this model IS precessing) dispatches to
`_fused_waveform_likelihood_kernel_phenomxphm!`, combining `IMRPhenomXP`'s precession setup with
`IMRPhenomXHM`'s higher-mode/no-native-`t_c` handling (see that kernel body's own extensive comment
block) — it depends on GravitationalWaveforms' precessing final spin. Any other model still goes through the original array-based
kernel body unchanged. This split exists because `Enzyme.autodiff` hangs indefinitely on the
array-based kernel body but completes on the scalar-core ones (see `test/test_enzyme_ka.jl`, which
covers both IMRPhenomD and IMRPhenomXAS, and the notes in this file).

`gmst_val = GravitationalWaveData.gmst(data.gps_time)` is computed once here, outside the kernel launch (cheap
relative to the per-walker work, but there's no reason to repeat the leap-second/Julian-day
lookup `nwalkers` times), and passed as a plain scalar kernel argument — see this file's module
docstring for why `ra`/`dec`/`psi` themselves must still be read from `params` and used to compute
the antenna response fresh INSIDE the kernel body (Enzyme correctness), not folded into this
precomputed value.

Numerical note: floating-point summation order inside the kernel's sequential accumulator loop
is not guaranteed to bit-for-bit match `sum(abs2.(...) ./ ...)` from the old path (which may use
pairwise summation internally for longer arrays), so agreement between the two paths should be
checked with a tight `isapprox` tolerance rather than exact equality.
"""
# `PEConfig.marginalize_phase`/`time_reference` are implemented for the `PAF2Model` kernels only;
# fail loudly rather than silently evaluating the plain likelihood for any other model.
# Whether the batched kernel selected for `model` on `backend` takes the likelihood-options array (XP / XPHM: only their
# CPU options kernels do; on a GPU they keep the original fused kernels, which refuse options).
_opts_kernel(model, backend) = _has_likelihood_options(model) &&
    !(model isa _Prec15 && !(backend isa KernelAbstractions.CPU))
_any_lik_options(cfg::PEConfig) = cfg.marginalize_phase || cfg.marginalize_distance || cfg.time_ref_frequency > 0 ||
                                  cfg.fmax_taper > 0 || cfg.time_reference != 0
_has_likelihood_options(model) = model isa PAF2Model || model isa IMRPhenomXAS || model isa IMRPhenomXHM || model isa IMRPhenomD ||
                                  model isa _Prec15 || model isa SEOBNRv5HM

function _check_likelihood_options(cfg::PEConfig, model, ndet::Int)
    model isa IMRPhenomXAS && cfg.fmax_taper > 0 && throw(ArgumentError("fmax_taper is not implemented for IMRPhenomXAS"))
    if model isa Union{IMRPhenomD,_Prec15}
        cfg.fmax_taper > 0 && throw(ArgumentError("fmax_taper is not implemented for $(nameof(typeof(model)))"))
    end
    if model isa _Prec15
        cfg.marginalize_phase && throw(ArgumentError(
            "marginalize_phase is not valid for $(typeof(model)): the phase enters the precession setup, it is not " *
            "an overall phase of the waveform (sample `phase` instead; distance marginalization and f_ref timing are supported)"))
    end
    if model isa Union{IMRPhenomXHM,SEOBNRv5HM}
        cfg.fmax_taper > 0 && throw(ArgumentError("fmax_taper is not implemented for $(nameof(typeof(model)))"))
        cfg.marginalize_phase && throw(ArgumentError(
            "marginalize_phase is not valid for $(nameof(typeof(model))): with higher modes the orbital phase is not an overall " *
            "phase of the waveform (sample `phase` instead; distance marginalization and f_ref timing are supported)"))
    end
    if !_has_likelihood_options(model)
        (cfg.marginalize_phase || cfg.marginalize_distance || cfg.time_ref_frequency > 0 || cfg.fmax_taper > 0) && throw(ArgumentError(
            "marginalize_phase/marginalize_distance/time_ref_frequency are only implemented for the PA-F2/IMR-table models and IMRPhenomXAS (got $(typeof(model)))"))
        ndet > 1 && cfg.time_reference != 0 && throw(ArgumentError(
            "time_reference is only implemented for the PA-F2/IMR-table models and IMRPhenomXAS (got $(typeof(model)))"))
    end
    cfg.time_reference <= ndet || throw(ArgumentError(
        "time_reference=$(cfg.time_reference) but only $ndet detector(s) were given"))
    return nothing
end

function batched_log_likelihood_kernel!(
    ll::AbstractVector{T},
    params::AbstractMatrix{T},
    freqs::AbstractVector{T},
    cfg::PEConfig{T},
    data::LikelihoodData;
    backend::KernelAbstractions.Backend = KernelAbstractions.CPU(),
    model = IMRPhenomD(),
) where {T<:Real}
    # IMRPhenomXP/IMRPhenomXPHM are the models needing the 15-row
    # precessing-spin convention (rows 12-15, this file's module docstring); every other model
    # still uses the 11-row convention.
    @assert size(params, 1) == (model isa _Prec15 ? 15 : 11)
    @assert length(ll) == size(params, 2)
    @assert cfg.ntime == length(freqs)

    nwalkers = size(params, 2)
    gmst_val = gmst(data.gps_time)
    _check_likelihood_options(cfg, model, 1)

    # IMRPhenomD/IMRPhenomXHM on a non-CPU backend take a separate structure-of-arrays dispatch
    # path -- see `_fused_waveform_likelihood_kernel_phenomxhm_soa!`'s own docstring for the full
    # GPU-compile-failure investigation this works around. `params[row, :]` (no `@view`) below
    # materializes fresh, contiguous 1-D arrays -- required, not cosmetic: a strided view into
    # the matrix reintroduces the same failure this path exists to avoid.
    if model isa IMRPhenomD && !(backend isa KernelAbstractions.CPU)
        _any_lik_options(cfg) && throw(ArgumentError("IMRPhenomD likelihood options are CPU-only for now"))
        m1 = params[1, :]; m2 = params[2, :]; chi1 = params[3, :]; chi2 = params[4, :]
        dL = params[5, :]; t_c = params[6, :]; phase = params[7, :]
        ra = params[8, :]; dec = params[9, :]; theta_jn = params[10, :]; psi = params[11, :]
        geom_flat = _geometry_array((data,), backend)
        gmst_arr = KernelAbstractions.allocate(backend, Float64, 1)
        copyto!(gmst_arr, [gmst_val])
        kernel! = _fused_waveform_likelihood_kernel_phenomd_soa!(backend)
        kernel!(ll, m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi,
                freqs, data.strain, data.psd, geom_flat, gmst_arr, model; ndrange = nwalkers)
        KernelAbstractions.synchronize(backend)
        return ll
    end

    if model isa IMRPhenomXHM && !(backend isa KernelAbstractions.CPU)
        _any_lik_options(cfg) && throw(ArgumentError("IMRPhenomXHM likelihood options are CPU-only for now"))
        m1 = params[1, :]; m2 = params[2, :]
        mc = similar(m1); eta = similar(m1)
        chirp_kernel! = _chirp_mass_eta_kernel!(backend)
        chirp_kernel!(mc, eta, m1, m2; ndrange = nwalkers)
        KernelAbstractions.synchronize(backend)
        chi1 = params[3, :]; chi2 = params[4, :]; dL = params[5, :]
        t_c = params[6, :]; phase = params[7, :]
        ra = params[8, :]; dec = params[9, :]; theta_jn = params[10, :]; psi = params[11, :]
        # See `_fused_waveform_likelihood_kernel_phenomxhm_soa!`'s own docstring for why
        # response/gmst_val must be `Const` CuArrays here, not bare values.
        geom_flat = _geometry_array((data,), backend)
        gmst_arr = KernelAbstractions.allocate(backend, Float64, 1)
        copyto!(gmst_arr, [gmst_val])
        kernel! = _fused_waveform_likelihood_kernel_phenomxhm_soa!(backend)
        kernel!(ll, mc, eta, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi,
                freqs, data.strain, data.psd, geom_flat, gmst_arr; ndrange = nwalkers)
        KernelAbstractions.synchronize(backend)
        return ll
    end

    # time_ref_frequency / phase_ref_frequency: the device stages also on the CPU (the reference `hphc` path has neither)
    if model isa SEOBNRv5PHM && (!(backend isa KernelAbstractions.CPU) || cfg.time_ref_frequency > 0 || cfg.phase_ref_frequency > 0)
        return seobnr_phm_device_loglikelihood!(ll, params, cfg, data; backend = backend)
    end

    kernel! = if model isa IMRPhenomD
        _pd_opts_kernel!(backend)           # with no options == the fused kernel
    elseif model isa IMRPhenomXAS
        _fused_waveform_likelihood_kernel_phenomxas!(backend)
    elseif model isa AlignedSpinPNHamF2
        _fused_waveform_likelihood_kernel_alignedspinpnhamf2!(backend)
    elseif model isa PAF2Model
        _fused_waveform_likelihood_kernel_seobnrv5alignedspinf2!(backend)
    elseif model isa _Prec15
        if backend isa KernelAbstractions.CPU
            _xprec_opts_kernel!(backend)    # with no options == the fused kernels
        else
            _any_lik_options(cfg) && throw(ArgumentError("$(typeof(model)) likelihood options are CPU-only for now"))
            model isa IMRPhenomXP ? _fused_waveform_likelihood_kernel_phenomxp!(backend) :
                                    _fused_waveform_likelihood_kernel_phenomxphm!(backend)
        end
    elseif model isa IMRPhenomXHM
        _xhm_opts_kernel!(backend)          # CPU (the GPU path returned above); with no options == the fused kernel
    else
        _fused_waveform_likelihood_kernel!(backend)
    end
    # `workgroupsize=1` on the CPU backend only: KernelAbstractions.CPU() distributes work across
    # `Threads.nthreads()` PER WORKGROUP, not per work item -- its default workgroupsize heuristic
    # (`default_cpu_workgroupsize`) sets workgroupsize == ndrange whenever `prod(ndrange) <= 1024`
    # (always true here, ndrange=nwalkers), which packs the entire launch into ONE workgroup and
    # silently collapses execution to a single thread regardless of `-t`/`Threads.nthreads()` --
    # confirmed empirically (identical wall-clock at -t1 vs -t6 without this fix; ~2.7x speedup
    # with it, at -t4, on a toy kernel). `workgroupsize=1` makes every walker its own workgroup, so
    # `Threads.@threads`/`Threads.@spawn` in `cpu.jl`'s `__run` actually has multiple blocks to
    # distribute. `nothing` (KA's own default behavior) on other backends -- GPU backends want
    # LARGE workgroups for occupancy, so this must not apply there.
    #
    # response/gmst_val as small `Const` arrays (built via `KernelAbstractions.allocate`, backend-
    # generic -- a plain `Array` on `CPU()`, a `CuArray` on `CUDABackend()`) rather than bare
    # values: see `_fused_waveform_likelihood_kernel_phenomxhm_soa!`'s docstring for why this
    # matters for Enzyme's GPU activity inference (a bare non-array value derived from an outer
    # `Const` argument gets misclassified `Active`, which Enzyme's GPU rule rejects outright).
    # Harmless on CPU -- `Const`-wrapped arrays already work identically for every other input here.
    geom_flat = _geometry_array((data,), backend)
    gmst_arr = KernelAbstractions.allocate(backend, Float64, 1)
    copyto!(gmst_arr, [gmst_val])
    if _opts_kernel(model, backend)
        kernel!(ll, params, freqs, data.strain, data.psd, geom_flat, gmst_arr, model, _lik_array(cfg, backend);
                ndrange = nwalkers, workgroupsize = (backend isa KernelAbstractions.CPU ? 1 : nothing))
    else
        kernel!(ll, params, freqs, data.strain, data.psd, geom_flat, gmst_arr, model;
                ndrange = nwalkers, workgroupsize = (backend isa KernelAbstractions.CPU ? 1 : nothing))
    end
    KernelAbstractions.synchronize(backend)
    return ll
end

"""
    batched_log_likelihood_kernel!(ll, params, cfg, data::AbstractVector{<:LikelihoodData};
                                    backend = KernelAbstractions.CPU(), model = IMRPhenomD())

Multi-detector method: sums
the frequency-domain matched-filter log-likelihood across every detector in `data` (one
`LikelihoodData` entry per detector, e.g. `[h1_data, l1_data]`) inside a single kernel launch —
still one kernel thread per walker, with an inner `for d in 1:ndet` loop over detectors (see the
multi-detector design note above `_extrinsic_setup` for the full reasoning). `params` is
`(11, nwalkers)` for `IMRPhenomD`/`IMRPhenomXAS`, `(15, nwalkers)` for `IMRPhenomXP` (same PHYSICAL row
convention documented at the top of this file, asserted against `model` here exactly like the
single-detector method) — there is one physical event (one `m1`, `m2`, ..., one `t_c`), only the
per-detector projection (antenna response) and per-detector arrival-time offset (`t_c` plus that
detector's own `GravitationalWaveData.time_delay_from_geocenter`) differ.

All entries of `data` must share the same `gps_time` (same physical event); this is asserted, not
assumed. Each detector's own `freqs`/`strain`/`psd` may have a different length — the inner
per-detector frequency loop uses that detector's own grid, not a shared shape.

Dispatches to `model isa IMRPhenomD`/`IMRPhenomXAS`/`IMRPhenomXP` specialized kernel bodies exactly like the
single-detector method above (same Enzyme-through-KernelAbstractions reasoning: the scalar-core
kernel bodies differentiate, the generic array-based fallback does not); no generic fallback is
provided for the multi-detector path since every supported waveform family already has a
specialized scalar core and there is no tested Enzyme-safe generic multi-detector kernel body. For
`IMRPhenomXP`, `_xp_precession_setup`'s frequency-independent `setup` and the aligned-(2,2)-mode
`ampl_coeffs` are hoisted outside the `1:ndet` loop (genuinely detector-independent, using the
FIRST detector's first frequency bin as `f_ref`); `t0`/`phi_coeffs`/`Fplus`/`Fcross` are recomputed
per detector since they depend on that detector's own time-delay-shifted `t_c` and antenna-response
geometry — see `_fused_waveform_likelihood_kernel_phenomxp_multi!`'s own comment block. Relative
binning for `IMRPhenomXP` (single- and multi-detector) is now also supported, see `RelativeBinning.jl`.

`model isa IMRPhenomXHM` (`params` is `(11, nwalkers)` since this model is
aligned-spin/non-precessing, same convention as its single-detector method) dispatches to
`_fused_waveform_likelihood_kernel_phenomxhm_multi!`: `_xhm_system_setup`/`_xhm_mode_setup`/
`_xhm_mode32_setup` (genuinely detector-independent, anchored to the FIRST detector's first
frequency bin as `f_ref`) are hoisted outside the `1:ndet` loop, exactly like `IMRPhenomXP`'s
`setup`; `Fplus`/`Fcross` and the coalescence-time phase shift are recomputed per detector, since
the single-detector kernel's `exp(im*2*pi*f*t_c)` post-multiplication generalizes to
`exp(im*2*pi*f*(t_c + dt_geo))` per detector (`dt_geo = time_delay_from_geocenter_gmst(det_d, ra, dec,
gmst_val)`, the same per-detector light-travel-time offset every other multi-detector kernel here
folds into its own `t0` — see `_fused_waveform_likelihood_kernel_phenomxhm_multi!`'s own comment
block for the full derivation/verification).
"""
function batched_log_likelihood_kernel!(
    ll::AbstractVector{T},
    params::AbstractMatrix{T},
    cfg::PEConfig{T},
    data::AbstractVector{<:LikelihoodData};
    backend::KernelAbstractions.Backend = KernelAbstractions.CPU(),
    model = IMRPhenomD(),
) where {T<:Real}
    # Multi-detector IMRPhenomXP/IMRPhenomXPHM use the 15-row
    # precessing-spin convention, exactly like the single-detector method above -- every other
    # model still uses the 11-row convention.
    _needs15 = model isa _Prec15
    @assert size(params, 1) == (_needs15 ? 15 : 11) "multi-detector batched_log_likelihood_kernel! expects $(_needs15 ? 15 : 11) rows for $(typeof(model)) (got $(size(params, 1)) rows)"
    @assert length(ll) == size(params, 2)
    ndet = length(data)
    @assert ndet >= 1 "multi-detector batched_log_likelihood_kernel! needs at least one detector"

    gps_time = data[1].gps_time
    @inbounds for i in 2:ndet
        data[i].gps_time == gps_time || throw(ArgumentError(
            "all detectors must share the same event gps_time (got $(data[i].gps_time) at index $i, expected $gps_time)"))
    end
    gmst_val = gmst(gps_time)

    freqs_tuple = ntuple(i -> data[i].freqs, ndet)
    strain_tuple = ntuple(i -> data[i].strain, ndet)
    psd_tuple = ntuple(i -> data[i].psd, ndet)
    det_tuple = ntuple(i -> geometry(data[i].detector), ndet)

    nwalkers = size(params, 2)
    _check_likelihood_options(cfg, model, ndet)

    # See the single-detector method's own comment for why: IMRPhenomXHM on a non-CPU backend
    # takes a separate structure-of-arrays dispatch path.
    if model isa IMRPhenomXHM && !(backend isa KernelAbstractions.CPU)
        _any_lik_options(cfg) && throw(ArgumentError("IMRPhenomXHM likelihood options are CPU-only for now"))
        m1 = params[1, :]; m2 = params[2, :]
        mc = similar(m1); eta = similar(m1)
        chirp_kernel! = _chirp_mass_eta_kernel!(backend)
        chirp_kernel!(mc, eta, m1, m2; ndrange = nwalkers)
        KernelAbstractions.synchronize(backend)
        chi1 = params[3, :]; chi2 = params[4, :]; dL = params[5, :]
        t_c = params[6, :]; phase = params[7, :]
        ra = params[8, :]; dec = params[9, :]; theta_jn = params[10, :]; psi = params[11, :]
        kernel! = _fused_waveform_likelihood_kernel_phenomxhm_multi_soa!(backend)
        kernel!(ll, mc, eta, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi,
                freqs_tuple, strain_tuple, psd_tuple, det_tuple, gmst_val;
                ndrange = nwalkers)
        KernelAbstractions.synchronize(backend)
        return ll
    end

    # time_ref_frequency / phase_ref_frequency: the device stages also on the CPU (the reference `hphc` path has neither)
    if model isa SEOBNRv5PHM && (!(backend isa KernelAbstractions.CPU) || cfg.time_ref_frequency > 0 || cfg.phase_ref_frequency > 0)
        return seobnr_phm_device_loglikelihood!(ll, params, cfg, data; backend = backend)
    end

    kernel! = if model isa IMRPhenomD
        _pd_opts_multi_kernel!(backend)     # with no options == the fused kernel
    elseif model isa IMRPhenomXAS
        _fused_waveform_likelihood_kernel_phenomxas_multi!(backend)
    elseif model isa AlignedSpinPNHamF2
        _fused_waveform_likelihood_kernel_alignedspinpnhamf2_multi!(backend)
    elseif model isa PAF2Model
        _fused_waveform_likelihood_kernel_seobnrv5alignedspinf2_multi!(backend)
    elseif model isa _Prec15 && backend isa KernelAbstractions.CPU
        _xprec_opts_multi_kernel!(backend)  # with no options == the fused kernels
    elseif model isa IMRPhenomXP
        _any_lik_options(cfg) && throw(ArgumentError("IMRPhenomXP likelihood options are CPU-only for now"))
        _fused_waveform_likelihood_kernel_phenomxp_multi!(backend)
    elseif model isa IMRPhenomXHM
        _xhm_opts_multi_kernel!(backend)    # CPU (the GPU path returned above)
    elseif model isa IMRPhenomXPHM
        _any_lik_options(cfg) && throw(ArgumentError("IMRPhenomXPHM likelihood options are CPU-only for now"))
        _fused_waveform_likelihood_kernel_phenomxphm_multi!(backend)
    else
        error("multi-detector batched_log_likelihood_kernel! only supports IMRPhenomD/IMRPhenomXAS/AlignedSpinPNHamF2/SEOBNRv5AlignedSpinF2/SEOBNRv5IMRAlignedSpinF2/IMRPhenomXP/IMRPhenomXHM/IMRPhenomXPHM models (got $(typeof(model)))")
    end
    # See the single-detector method above for why `workgroupsize=1` on CPU is needed for real
    # multithreading (KernelAbstractions.CPU() distributes across workgroups, not work items).
    if _opts_kernel(model, backend)
        kernel!(ll, params, freqs_tuple, strain_tuple, psd_tuple, det_tuple, gmst_val, model,
                _lik_array(cfg, backend), cfg.time_reference;
                ndrange = nwalkers, workgroupsize = (backend isa KernelAbstractions.CPU ? 1 : nothing))
    else
        kernel!(ll, params, freqs_tuple, strain_tuple, psd_tuple, det_tuple, gmst_val, model;
                ndrange = nwalkers, workgroupsize = (backend isa KernelAbstractions.CPU ? 1 : nothing))
    end
    KernelAbstractions.synchronize(backend)
    return ll
end


include("SEOBNRPHMDevice.jl")

end
