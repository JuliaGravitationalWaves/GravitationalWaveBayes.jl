module SEOBNRv5AlignedSpinF2Adjoint

export seobv5f2_coeffs_jvp, seobv5f2_coeffs_vjp

using Enzyme
using StaticArrays: SVector, MVector
using LinearAlgebra: dot
using GravitationalWaveforms: SEOBNRv5AlignedSpinF2, _SEOBV5_PA_NMAX, SEOBNRv5HamParams,
    _seobv5_sym_gradient, _seobv5_sym_tphi, _seobv5_potentials,
    _seobv5_pr_eqn, _seobv5_pphi_eqn_fixed, _seobv5_findiff_derivative9_fixed,
    _seobv5_cumulative_integral7_fixed, _seobv5_adiabatic_j0_fixed, _seobv5_postadiabatic_solution_fixed,
    _seobv5_circular_ic, _seobv5_pa_grid_fixed, _seobv5_kerr_isco, _seobv5_a6_NS, _seobv5_dSO,
    _seobv5_multipole_prefix_abs, _seobv5_flux_coeffs, GMsun_over_c3, _SEOBV5_FINDIFF9_COEFFS, _SEOBV5_CUMINT7_COEFFS

"""
    SEOBNRv5AlignedSpinF2Adjoint

Hand-derived forward-mode sensitivity (JVP) of `GravitationalWaveforms._seobv5f2_coeffs`'s three
differentiable table fields (`omegas`, `ts`, `phis`) with respect to its 4 intrinsic parameters
(`mc`, `eta`, `chi1`, `chi2`), built to avoid ever calling `Enzyme.autodiff` on that pipeline's own
loopy/fixed-buffer internals -- the two-stage initial-condition Newton solve
(`_seobv5_circular_ic`), the per-node adiabatic Newton solve (`_seobv5_adiabatic_j0_fixed`), and the
8-pass alternating post-adiabatic sweep (`_seobv5_postadiabatic_solution_fixed`), all of which
operate on `NMAX`-length `SVector`/`MVector` buffers on `CUDABackend()`. Differentiating that
combination directly with `Enzyme.Forward` is what produces three upstream Enzyme/CUDA
compiler failures: an activity-
ambiguity error, `set_runtime_activity` breaking `_seobv5f2_coeffs`'s own static type analysis, and
an internal LLVM shadow-allocation assertion.

**Every `Enzyme.autodiff` call in this file targets a small, non-allocating, loop-free SCALAR
function** (`_seobv5_gradient`/`_seobv5_ham_xi`/`_seobv5_potentials` composed together, or the
closed-form `mc,eta,chi1,chi2 -> nu,X1,X2,a6,dSO,...` parameter chain) -- never a kernel, an
`MVector`, or anything CUDA-shaped. Wherever the primal involves a Newton/root-find
(`_seobv5_circular_ic`'s 2D solve, `_seobv5_adiabatic_j0_fixed`'s per-node 1D solve), sensitivity is
obtained via the implicit function theorem evaluated AT the already-converged solution -- one or
two extra small JVP calls per solve, not by differentiating through the solver's own iterations.
Wherever the primal uses a fixed-coefficient linear stencil (`_seobv5_findiff_derivative9_fixed`,
`_seobv5_cumulative_integral7_fixed`), the JVP reuses the SAME stencil on the tangent array (it's
linear in its array argument) plus, since both stencils also depend on the grid spacing `dr_step`
which itself has a nonzero tangent once `r0` depends on the intrinsic parameters, an explicit
correction term for that dependence (see `_seobv5_postadiabatic_solution_fixed_jvp`'s `hscale` and
`_seobv5_cumulative_integral7_fixed_jvp` below) -- omitting this term was a real bug caught during
development (a smoothly-growing error away from `r0`, ruled out as finite-difference noise by a
step-size scan) before the fix below.

**Scope**: only the pipeline actually wired into `GravitationalWaveBayes.jl`'s NUTS sampler is covered. The other
half of `GravitationalWaveforms.jl`'s two-stage initial-condition solve, `_seobv5_dissipative_ic` (which sets
`coeffs.pr0`), was prototyped and validated during development but is deliberately NOT included
here: `seobnrv5alignedspin_postadiabatic_dynamics_fixed` never reads `coeffs.pr0`/`coeffs.pphi0`
(confirmed by inspection) -- only `coeffs.r0`, from `_seobv5_circular_ic` alone, feeds this
pipeline. A future non-adiabatic (full RK4) track wired into NUTS would need that piece too.

**Validated** against central finite differences on `_seobv5f2_coeffs`'s real output, all 4
intrinsic-parameter directions, at `mc=28.0961, eta=0.2479, chi1=0.3, chi2=-0.2, f_min=20`: max
relative error across the real grid region (excluding the single last node, where the true
`domega/dparam` etc. cross near zero and relative error stops being a meaningful metric -- absolute
error there is `~1e-8`-`3e-8`) is `1.4e-4`/`5.4e-5`/`1.6e-3`/`1.8e-3` for `domegas` and similarly
`1e-4`-`5e-5` for `dts`/`dphis`, w.r.t. `mc`/`eta`/`chi1`/`chi2` respectively.

**Wired into `BatchedNUTS.jl`**: `_forward_mode_gradient!`'s
`SEOBNRv5AlignedSpinF2` method calls `seobv5f2_coeffs_jvp` directly instead of letting
`Enzyme.Forward` trace `_seobv5f2_coeffs`'s own construction (see `KernelFused.jl`'s
`_fused_ll_kernel_seobnrv5alignedspinf2_precoeffs!`/`_seobv5f2_build_coeffs_soa_kernel!` and
`BatchedNUTS.jl`'s own comments for the full integration). Checked by the full
`gw_logp_and_grad`-vs-finite-differences test and an end-to-end NUTS smoke test.
"""
SEOBNRv5AlignedSpinF2Adjoint

# --- Layer B: JVP of the Hamiltonian's own analytic first partials, via Enzyme.Forward on the
# small composed (dHdr, dHdpr, dHdpphi) function. This is the ONLY place `Enzyme.autodiff` sees the
# Hamiltonian at all -- a ~100-line pure scalar function, no loops, no allocation. Differentiating
# it a second time this way gives exactly the second-partials-of-H the implicit function theorem
# needs below, without ever writing them out by hand. ---

@inline _seobv5_gradient_full(r, pr, pphi, nu, X1, X2, chi1, chi2, a6, dSO) =
    _seobv5_sym_gradient(r, pr, pphi, SEOBNRv5HamParams(nu, X1, X2, chi1, chi2, a6, dSO))

function _seobv5_gradient_full_jvp(r, pr, pphi, nu, X1, X2, chi1, chi2, a6, dSO,
                                    dr, dpr, dpphi, dnu, dX1, dX2, dchi1, dchi2, da6, ddSO)
    (res,) = Enzyme.autodiff(Enzyme.Forward, _seobv5_gradient_full,
        Enzyme.Duplicated(r, dr), Enzyme.Duplicated(pr, dpr), Enzyme.Duplicated(pphi, dpphi),
        Enzyme.Duplicated(nu, dnu), Enzyme.Duplicated(X1, dX1), Enzyme.Duplicated(X2, dX2),
        Enzyme.Duplicated(chi1, dchi1), Enzyme.Duplicated(chi2, dchi2),
        Enzyme.Duplicated(a6, da6), Enzyme.Duplicated(dSO, ddSO))
    return res  # (d(dHdr), d(dHdpr), d(dHdpphi))
end

# Extended with xi (from _seobv5_potentials), needed for dt_dr = 1/(dHdpr*xi) below.
@inline function _seobv5_gradient_full2(r, pr, pphi, nu, X1, X2, chi1, chi2, a6, dSO)
    hp = SEOBNRv5HamParams(nu, X1, X2, chi1, chi2, a6, dSO)
    dHdr, dHdpr, dHdpphi = _seobv5_sym_gradient(r, pr, pphi, hp)
    return (dHdr, dHdpr, dHdpphi, _seobv5_potentials(r, pr, pphi, hp)[2])
end

function _seobv5_gradient_full2_jvp(r, pr, pphi, nu, X1, X2, chi1, chi2, a6, dSO,
                                     dr, dpr, dpphi, dnu, dX1, dX2, dchi1, dchi2, da6, ddSO)
    (res,) = Enzyme.autodiff(Enzyme.Forward, _seobv5_gradient_full2,
        Enzyme.Duplicated(r, dr), Enzyme.Duplicated(pr, dpr), Enzyme.Duplicated(pphi, dpphi),
        Enzyme.Duplicated(nu, dnu), Enzyme.Duplicated(X1, dX1), Enzyme.Duplicated(X2, dX2),
        Enzyme.Duplicated(chi1, dchi1), Enzyme.Duplicated(chi2, dchi2),
        Enzyme.Duplicated(a6, da6), Enzyme.Duplicated(dSO, ddSO))
    return res  # (d(dHdr), d(dHdpr), d(dHdpphi), dxi)
end

# The two closed-form, per-node post-adiabatic correction formulas (`_seobv5_pr_eqn`/`_seobv5_pphi_eqn_fixed`)
# with the flux's per-binary inputs (`delta, chiS, chiA, atest`, the six `absN`) as scalar arguments: the
# GravitationalWaveforms functions take the flux coefficients `fc` built from them (`_seobv5_flux_coeffs`), built here
# inside, so the tangents / cotangents below are with respect to these 21 scalars (`nu` enters `fc` too).
@inline _pr_eqn21(pr, r, pphi, d, nu, X1, X2, chi1, chi2, a6, dSO, delta, chiS, chiA, atest, n33, n21, n44, n32, n55, n43) =
    _seobv5_pr_eqn(pr, r, pphi, d, SEOBNRv5HamParams(nu, X1, X2, chi1, chi2, a6, dSO),
        _seobv5_flux_coeffs(nu, delta, chiS, chiA, atest, n33, n21, n44, n32, n55, n43))
@inline _pphi_eqn21(pphi, r, pr, d, nu, X1, X2, chi1, chi2, a6, dSO, delta, chiS, chiA, atest, n33, n21, n44, n32, n55, n43) =
    _seobv5_pphi_eqn_fixed(pphi, r, pr, d, SEOBNRv5HamParams(nu, X1, X2, chi1, chi2, a6, dSO),
        _seobv5_flux_coeffs(nu, delta, chiS, chiA, atest, n33, n21, n44, n32, n55, n43))

# JVP wrappers for the two closed-form, per-node post-adiabatic correction formulas
# (`_seobv5_pr_eqn`/`_seobv5_pphi_eqn_fixed`, both quadratic-formula evaluations, NOT root-finds --
# see `SEOBNRv5AlignedSpinPA.jl`'s own docstring). 21 positional Float64 args each.
function _seobv5_pr_eqn_jvp(args::NTuple{21,Float64}, dargs::NTuple{21,Float64})
    (res,) = Enzyme.autodiff(Enzyme.Forward, _pr_eqn21,
        ntuple(i -> Enzyme.Duplicated(args[i], dargs[i]), 21)...)
    return res
end
function _seobv5_pphi_eqn_fixed_jvp(args::NTuple{21,Float64}, dargs::NTuple{21,Float64})
    (res,) = Enzyme.autodiff(Enzyme.Forward, _pphi_eqn21,
        ntuple(i -> Enzyme.Duplicated(args[i], dargs[i]), 21)...)
    return res
end

"""
    _seobv5_adiabatic_j0_ift(r, pphi0, nu, X1, X2, chi1, chi2, a6, dSO, dr, dnu, dX1, dX2, dchi1, dchi2, da6, ddSO)

Implicit-function-theorem sensitivity of `_seobv5_adiabatic_j0_fixed`'s per-node circular-orbit
Newton solve (`F(p) = dHdr(r_i, 0, p, params) = 0`), evaluated AT the already-converged `pphi0`
(the primal's own 20-iteration solve is never differentiated through): one extra
`_seobv5_gradient_full_jvp` call per grid node beyond the primal's own iterations.
"""
function _seobv5_adiabatic_j0_ift(r::SVector{NMAX,Float64}, pphi0::SVector{NMAX,Float64},
        nu, X1, X2, chi1, chi2, a6, dSO,
        dr::SVector{NMAX,Float64}, dnu, dX1, dX2, dchi1, dchi2, da6, ddSO, nreal::Int = NMAX) where {NMAX}
    dpphi0 = zero(MVector{NMAX,Float64})
    @inbounds for i in 1:nreal  # padding lanes never reach the real region
        p = pphi0[i]
        dFdp, _, _ = _seobv5_gradient_full_jvp(r[i], 0.0, p, nu, X1, X2, chi1, chi2, a6, dSO,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        dFdparam, _, _ = _seobv5_gradient_full_jvp(r[i], 0.0, p, nu, X1, X2, chi1, chi2, a6, dSO,
            dr[i], 0.0, 0.0, dnu, dX1, dX2, dchi1, dchi2, da6, ddSO)
        dpphi0[i] = -dFdparam / dFdp
    end
    return SVector(dpphi0)
end

"""
    _seobv5_postadiabatic_solution_fixed_jvp(r, nreal, pphi_in, dpphi_in, dr_grid, nu, X1, X2, chi1,
        chi2, a6, dSO, delta, chiS, chiA, atest, absN33, absN21, absN44, absN32, absN55, absN43,
        dr_step; d<name>=0.0 for every parameter, ddr_step=0.0, order=8)

Hand-propagated JVP through `_seobv5_postadiabatic_solution_fixed`'s exact 8-pass alternating-sweep
control flow. `_seobv5_findiff_derivative9_fixed` is LINEAR in its array argument, so the same
stencil computes its own JVP w.r.t. that argument directly -- but it also divides by the grid
spacing `dr_step`, which has a nonzero tangent `ddr_step` once `r0` depends on the intrinsic
parameters (via the grid's own `r0`/`r_final`), so the correction term
`-primal_stencil_result * (ddr_step/dr_step)` (the `hscale` term below) is required, not optional --
see this file's own module docstring for how this was caught (a smoothly-growing discrepancy vs.
finite differences away from `r0`, confirmed real by a step-size scan, not FD noise).
"""
function _seobv5_postadiabatic_solution_fixed_jvp(r::SVector{NMAX,Float64}, nreal::Int,
        pphi_in::SVector{NMAX,Float64}, dpphi_in::SVector{NMAX,Float64}, dr_grid::SVector{NMAX,Float64},
        nu, X1, X2, chi1, chi2, a6, dSO, delta, chiS, chiA, atest,
        absN33, absN21, absN44, absN32, absN55, absN43, dr_step::Float64;
        dnu = 0.0, dX1 = 0.0, dX2 = 0.0, dchi1 = 0.0, dchi2 = 0.0, da6 = 0.0, ddSO = 0.0,
        ddelta = 0.0, dchiS = 0.0, dchiA = 0.0, datest = 0.0,
        dabsN33 = 0.0, dabsN21 = 0.0, dabsN44 = 0.0, dabsN32 = 0.0, dabsN55 = 0.0, dabsN43 = 0.0,
        ddr_step::Float64 = 0.0,
        order::Int = 8) where {NMAX}
    pr = zero(MVector{NMAX,Float64})
    pphi = MVector(pphi_in)
    dpr = zero(MVector{NMAX,Float64})
    dpphi = MVector(dpphi_in)
    dparams_tail = (dnu, dX1, dX2, dchi1, dchi2, da6, ddSO, ddelta, dchiS, dchiA, datest,
                     dabsN33, dabsN21, dabsN44, dabsN32, dabsN55, dabsN43)
    hscale = ddr_step / dr_step
    for n in 1:order
        if isodd(n)
            pphi_dr_arr = _seobv5_findiff_derivative9_fixed(SVector(pphi), nreal, dr_step)
            dpphi_dr_arr_raw = _seobv5_findiff_derivative9_fixed(SVector(dpphi), nreal, dr_step)
            @inbounds for i in 1:nreal  # padding lanes never reach the real region
                dpphi_dr_i = i <= nreal ? -(dpphi_dr_arr_raw[i] - pphi_dr_arr[i] * hscale) : 0.0
                pphi_dr_i = i <= nreal ? -pphi_dr_arr[i] : 1.0
                args = (pr[i], r[i], pphi[i], pphi_dr_i,
                        nu, X1, X2, chi1, chi2, a6, dSO, delta, chiS, chiA, atest,
                        absN33, absN21, absN44, absN32, absN55, absN43)
                dargs = (dpr[i], dr_grid[i], dpphi[i], dpphi_dr_i, dparams_tail...)
                pr[i] = _pr_eqn21(args...)
                dpr[i] = _seobv5_pr_eqn_jvp(args, dargs)
            end
        else
            pr_dr_arr = _seobv5_findiff_derivative9_fixed(SVector(pr), nreal, dr_step)
            dpr_dr_arr_raw = _seobv5_findiff_derivative9_fixed(SVector(dpr), nreal, dr_step)
            @inbounds for i in 1:nreal
                dpr_dr_i = i <= nreal ? -(dpr_dr_arr_raw[i] - pr_dr_arr[i] * hscale) : 0.0
                pr_dr_i = i <= nreal ? -pr_dr_arr[i] : 1.0
                args = (pphi[i], r[i], pr[i], pr_dr_i,
                        nu, X1, X2, chi1, chi2, a6, dSO, delta, chiS, chiA, atest,
                        absN33, absN21, absN44, absN32, absN55, absN43)
                dargs = (dpphi[i], dr_grid[i], dpr[i], dpr_dr_i, dparams_tail...)
                pphi[i] = _pphi_eqn21(args...)
                dpphi[i] = _seobv5_pphi_eqn_fixed_jvp(args, dargs)
            end
        end
    end
    return SVector(pr), SVector(pphi), SVector(dpr), SVector(dpphi)
end

"""
    _seobv5_circular_ic_ift(rhat, pphi, nu, X1, X2, chi1, chi2, a6, dSO, domega0, dnu, dX1, dX2,
        dchi1, dchi2, da6, ddSO) -> (drhat, dpphi)

Implicit-function-theorem sensitivity of `_seobv5_circular_ic`'s 2D Newton solve
(`F1 = dHdpphi - omega0 = 0`, `F2 = dHdr = 0`), evaluated at the converged `(rhat, pphi)`: builds
the 2x2 Jacobian from 2 `_seobv5_gradient_full_jvp` calls, then solves the same linear system the
primal's own Newton update uses, with `(F1, F2)` replaced by their total derivative from the
incoming tangents.
"""
function _seobv5_circular_ic_ift(rhat, pphi, nu, X1, X2, chi1, chi2, a6, dSO,
        domega0, dnu, dX1, dX2, dchi1, dchi2, da6, ddSO)
    dg1_r, _, dg3_r = _seobv5_gradient_full_jvp(rhat, 0.0, pphi, nu, X1, X2, chi1, chi2, a6, dSO,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    J11, J21 = dg3_r, dg1_r
    dg1_p, _, dg3_p = _seobv5_gradient_full_jvp(rhat, 0.0, pphi, nu, X1, X2, chi1, chi2, a6, dSO,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    J12, J22 = dg3_p, dg1_p
    dg1_param, _, dg3_param = _seobv5_gradient_full_jvp(rhat, 0.0, pphi, nu, X1, X2, chi1, chi2, a6, dSO,
        0.0, 0.0, 0.0, dnu, dX1, dX2, dchi1, dchi2, da6, ddSO)
    dF1 = dg3_param - domega0
    dF2 = dg1_param
    det = J11 * J22 - J12 * J21
    drhat = (-dF1 * J22 + dF2 * J12) / det
    dpphi = (-J11 * dF2 + J21 * dF1) / det
    return drhat, dpphi
end

# Closed-form parameter chain (mc,eta,chi1,chi2) -> everything _seobv5_coeffs derives from them
# (minus the IC solve, handled separately above since it's implicit). Small, allocation-free, safe
# for direct Enzyme.Forward -- mirrors `GravitationalWaveforms._seobv5_coeffs`'s own body exactly.
@inline function _seobv5_params_from_mc_eta_chi(mc, eta, chi1, chi2)
    delta = sqrt(max(1.0 - 4.0 * eta, 0.0))
    X1 = 0.5 * (1.0 + delta)
    X2 = 0.5 * (1.0 - delta)
    ap = X1 * chi1 + X2 * chi2
    am = X1 * chi1 - X2 * chi2
    chiS = 0.5 * (chi1 + chi2)
    chiA = 0.5 * (chi1 - chi2)
    atest = (1.0 - 2.0 * eta) * chiS + delta * chiA
    a6 = _seobv5_a6_NS(eta)
    dSO = _seobv5_dSO(eta, ap, am)
    absN33 = _seobv5_multipole_prefix_abs(eta, X1, X2, 3, 3)
    absN21 = _seobv5_multipole_prefix_abs(eta, X1, X2, 2, 1)
    absN44 = _seobv5_multipole_prefix_abs(eta, X1, X2, 4, 4)
    absN32 = _seobv5_multipole_prefix_abs(eta, X1, X2, 3, 2)
    absN55 = _seobv5_multipole_prefix_abs(eta, X1, X2, 5, 5)
    absN43 = _seobv5_multipole_prefix_abs(eta, X1, X2, 4, 3)
    return (eta, X1, X2, delta, chiS, chiA, atest, a6, dSO, absN33, absN21, absN44, absN32, absN55, absN43)
end
function _seobv5_params_from_mc_eta_chi_jvp(mc, eta, chi1, chi2, dmc, deta, dchi1, dchi2)
    (res,) = Enzyme.autodiff(Enzyme.Forward, _seobv5_params_from_mc_eta_chi,
        Enzyme.Duplicated(mc, dmc), Enzyme.Duplicated(eta, deta),
        Enzyme.Duplicated(chi1, dchi1), Enzyme.Duplicated(chi2, dchi2))
    return res
end

@inline function _seobv5_omega0_from_mc_eta(mc, eta, f_min, GMsun_over_c3_)
    M_sec = (mc / eta^0.6) * GMsun_over_c3_
    v0 = cbrt(pi * M_sec * f_min)
    return v0^3
end
function _seobv5_omega0_jvp(mc, eta, f_min, GMsun_over_c3_, dmc, deta)
    (domega0,) = Enzyme.autodiff(Enzyme.Forward, _seobv5_omega0_from_mc_eta,
        Enzyme.Duplicated(mc, dmc), Enzyme.Duplicated(eta, deta),
        Enzyme.Const(f_min), Enzyme.Const(GMsun_over_c3_))
    return domega0
end

@inline function _seobv5_M_and_Msec_from_mc_eta(mc, eta, GMsun_over_c3_)
    M = mc / eta^0.6
    return (M, M * GMsun_over_c3_)
end
function _seobv5_M_and_Msec_jvp(mc, eta, GMsun_over_c3_, dmc, deta)
    (res,) = Enzyme.autodiff(Enzyme.Forward, _seobv5_M_and_Msec_from_mc_eta,
        Enzyme.Duplicated(mc, dmc), Enzyme.Duplicated(eta, deta), Enzyme.Const(GMsun_over_c3_))
    return res  # (dM, dM_sec)
end

@inline function _seobv5_r_final_from_params(nu, X1, X2, chi1, chi2)
    chi_eff = X1 * chi1 + X2 * chi2
    r_final_prefactor = 2.7 + chi_eff * (1.0 - 4.0 * nu)
    r_isco, _ = _seobv5_kerr_isco(chi1, chi2, X1, X2)
    return r_final_prefactor * r_isco  # the uncapped value; caller applies max(10.0, .)
end
function _seobv5_r_final_jvp(nu, X1, X2, chi1, chi2, dnu, dX1, dX2, dchi1, dchi2)
    (dval,) = Enzyme.autodiff(Enzyme.Forward, _seobv5_r_final_from_params,
        Enzyme.Duplicated(nu, dnu), Enzyme.Duplicated(X1, dX1), Enzyme.Duplicated(X2, dX2),
        Enzyme.Duplicated(chi1, dchi1), Enzyme.Duplicated(chi2, dchi2))
    return dval
end

"""
    _seobv5_cumulative_integral7_fixed_jvp(y, dy, nreal, h, dh)

JVP of `_seobv5_cumulative_integral7_fixed(y, nreal, h) = h * cumsum(val(y))` (bilinear in `(y,h)`,
`val` linear in `y`): `d(result) = cumint7(dy, nreal, h) + dh * cumint7(y, nreal, 1.0)`.
"""
function _seobv5_cumulative_integral7_fixed_jvp(y::SVector{NMAX,Float64}, dy::SVector{NMAX,Float64},
        nreal::Int, h::Float64, dh::Float64) where {NMAX}
    term1 = _seobv5_cumulative_integral7_fixed(dy, nreal, h)
    term2 = _seobv5_cumulative_integral7_fixed(y, nreal, 1.0)
    return term1 .+ dh .* term2
end

"""
    seobv5f2_coeffs_jvp(model::SEOBNRv5AlignedSpinF2, mc, eta, chi1, chi2, dmc, deta, dchi1, dchi2;
        f_min=20.0, GMsun_over_c3=GMsun_over_c3, order=8, dr0_step=0.2) -> dcoeffs::NamedTuple

The directional derivative (JVP) of `GravitationalWaveforms._seobv5f2_coeffs(model, mc, eta, chi1, chi2;
f_min, GMsun_over_c3)`'s full return value, in the direction `(dmc, deta, dchi1, dchi2)` -- entirely
via this file's hand-derived sensitivities, with `Enzyme.autodiff` only ever called on small scalar
leaf functions (see this file's module docstring). The returned `dcoeffs` has the SAME field names
as `_seobv5f2_coeffs`'s own return (`success, M_sec, eta, M, nreal, ts, phis, omegas, fmin, fmax`),
with `success`/`nreal` set to inert placeholders (`false`/`0`, never meaningful -- these two fields
are not floating-point and have no tangent) so `dcoeffs` can be passed directly as the shadow half
of `Enzyme.Duplicated(coeffs, dcoeffs)` wherever a caller differentiates a function of `coeffs`
(e.g. `_seobv5f2_phi_scalar`/`_seobv5f2_ampl_scalar`) w.r.t. these 4 intrinsic parameters.

Recomputes the same (cheap, non-Enzyme) primal pipeline `_seobv5f2_coeffs` itself computes --
intermediate values (`r`, `pr`, `pphi`, `r0`, ...) aren't exposed by that function's own return
type, so this is not yet a single shared-primal call; calling this 4 times (once per intrinsic
parameter direction, matching `BatchedNUTS.jl`'s existing per-parameter forward-mode loop
convention) still recomputes the primal 4 times, a real but bounded inefficiency (strictly better
than today's design, which pays for a full `Enzyme.Forward`-traced forward pass once per ALL 11
sampled parameters, not just the 4 that reach this pipeline).

`GMsun_over_c3` keyword collides with the module-level constant of the same name imported from
`GravitationalWaveforms` -- kept as a keyword (default `GravitationalWaveforms.GMsun_over_c3`) to mirror
`_seobv5f2_coeffs`'s own signature exactly.
"""
function seobv5f2_coeffs_jvp(::SEOBNRv5AlignedSpinF2, mc, eta, chi1, chi2, dmc, deta, dchi1, dchi2;
        f_min = 20.0, GMsun_over_c3 = GMsun_over_c3, order::Int = 8, dr0_step::Float64 = 0.2)
    NMAX = _SEOBV5_PA_NMAX

    nu, X1, X2, delta, chiS, chiA, atest, a6, dSO, absN33, absN21, absN44, absN32, absN55, absN43 =
        _seobv5_params_from_mc_eta_chi(mc, eta, chi1, chi2)
    dnu, dX1, dX2, ddelta, dchiS, dchiA, datest, da6, ddSO, dabsN33, dabsN21, dabsN44, dabsN32, dabsN55, dabsN43 =
        _seobv5_params_from_mc_eta_chi_jvp(mc, eta, chi1, chi2, dmc, deta, dchi1, dchi2)

    M_sec = (mc / eta^0.6) * GMsun_over_c3
    v0 = cbrt(pi * M_sec * f_min)
    omega0 = v0^3
    domega0 = _seobv5_omega0_jvp(mc, eta, f_min, GMsun_over_c3, dmc, deta)

    hp = SEOBNRv5HamParams(nu, X1, X2, chi1, chi2, a6, dSO)
    r0, pphi0_ic = _seobv5_circular_ic(omega0, hp)
    dr0, _ = _seobv5_circular_ic_ift(r0, pphi0_ic, nu, X1, X2, chi1, chi2, a6, dSO,
        domega0, dnu, dX1, dX2, dchi1, dchi2, da6, ddSO)

    chi_eff = X1 * chi1 + X2 * chi2
    r_final_prefactor = 2.7 + chi_eff * (1.0 - 4.0 * nu)
    r_isco, _ = _seobv5_kerr_isco(chi1, chi2, X1, X2)
    r_final_raw = r_final_prefactor * r_isco
    r_final = max(10.0, r_final_raw)
    floor_active = r_final_raw < 10.0
    dr_final_raw = _seobv5_r_final_jvp(nu, X1, X2, chi1, chi2, dnu, dX1, dX2, dchi1, dchi2)
    dr_final = floor_active ? 0.0 : dr_final_raw

    nreal_raw = ceil(Int, (r0 - r_final) / dr0_step)
    success = !(nreal_raw <= 4 || r0 <= 11.5)
    nreal_bumped = nreal_raw < 10 ? 10 : nreal_raw
    nreal = success ? min(nreal_bumped, NMAX) : 10
    # nreal is discrete (a grid POINT COUNT) -> treated as locally constant, standard AD practice;
    # stable under the parameter perturbations this was validated against (confirmed).

    r0_grid = success ? r0 : 50.0
    r_final_grid = success ? r_final : 40.0
    dr0_grid = success ? dr0 : 0.0
    dr_final_grid = success ? dr_final : 0.0
    dr_step = (r0_grid - r_final_grid) / (nreal - 1)
    ddr_step = (dr0_grid - dr_final_grid) / (nreal - 1)

    r = _seobv5_pa_grid_fixed(r0_grid, dr_step, nreal, Val(NMAX))
    dr_grid_mv = zero(MVector{NMAX,Float64})
    @inbounds for i in 1:NMAX
        dr_grid_mv[i] = i <= nreal ? dr0_grid - (i - 1) * ddr_step : 0.0
    end
    dr_grid = SVector(dr_grid_mv)

    # Padding lanes (i > nreal) are skipped throughout: they never reach the real region or the
    # returned tangents (same argument as the GravitationalWaveforms driver's `skip_padding`).
    pphi0_pa = _seobv5_adiabatic_j0_fixed(r, hp, nreal)
    dpphi0_pa = _seobv5_adiabatic_j0_ift(r, pphi0_pa, nu, X1, X2, chi1, chi2, a6, dSO,
        dr_grid, dnu, dX1, dX2, dchi1, dchi2, da6, ddSO, nreal)

    fc = _seobv5_flux_coeffs(nu, delta, chiS, chiA, atest, absN33, absN21, absN44, absN32, absN55, absN43)
    pr, pphi = _seobv5_postadiabatic_solution_fixed(r, nreal, pphi0_pa, hp, fc, dr_step; order = order, skip_padding = true)
    _, _, dpr, dpphi = _seobv5_postadiabatic_solution_fixed_jvp(r, nreal, pphi0_pa, dpphi0_pa, dr_grid,
        nu, X1, X2, chi1, chi2, a6, dSO, delta, chiS, chiA, atest,
        absN33, absN21, absN44, absN32, absN55, absN43, dr_step;
        dnu = dnu, dX1 = dX1, dX2 = dX2, dchi1 = dchi1, dchi2 = dchi2, da6 = da6, ddSO = ddSO,
        ddelta = ddelta, dchiS = dchiS, dchiA = dchiA, datest = datest,
        dabsN33 = dabsN33, dabsN21 = dabsN21, dabsN44 = dabsN44, dabsN32 = dabsN32, dabsN55 = dabsN55, dabsN43 = dabsN43,
        ddr_step = ddr_step, order = order)

    omegas_raw = zero(MVector{NMAX,Float64})
    domegas_raw = zero(MVector{NMAX,Float64})
    dt_dr = zero(MVector{NMAX,Float64})
    ddt_dr = zero(MVector{NMAX,Float64})
    dphi_dr = zero(MVector{NMAX,Float64})
    ddphi_dr = zero(MVector{NMAX,Float64})
    @inbounds for i in 1:nreal
        xi, dHdpr, dHdpphi = _seobv5_sym_tphi(r[i], pr[i], pphi[i], hp)
        omegas_raw[i] = dHdpphi
        _, ddHdpr, ddHdpphi, dxi = _seobv5_gradient_full2_jvp(r[i], pr[i], pphi[i], nu, X1, X2, chi1, chi2, a6, dSO,
            dr_grid[i], dpr[i], dpphi[i], dnu, dX1, dX2, dchi1, dchi2, da6, ddSO)
        domegas_raw[i] = ddHdpphi

        u = dHdpr * xi
        du = ddHdpr * xi + dHdpr * dxi
        dt_dr_i = 1.0 / u
        dt_dr[i] = dt_dr_i
        ddt_dr[i] = -du * dt_dr_i^2
        dphi_dr[i] = dHdpphi * dt_dr_i
        ddphi_dr[i] = ddHdpphi * dt_dr_i + dHdpphi * ddt_dr[i]
    end
    om_n = omegas_raw[clamp(nreal, 1, NMAX)]
    dom_n = domegas_raw[clamp(nreal, 1, NMAX)]
    domegas = zero(MVector{NMAX,Float64})
    @inbounds for i in 1:NMAX
        domegas[i] = i <= nreal ? domegas_raw[i] : dom_n
    end

    dts = _seobv5_cumulative_integral7_fixed_jvp(SVector(dt_dr), SVector(ddt_dr), nreal, -dr_step, -ddr_step)
    dphis = _seobv5_cumulative_integral7_fixed_jvp(SVector(dphi_dr), SVector(ddphi_dr), nreal, -dr_step, -ddr_step)

    # `coeffs.M_sec`/`coeffs.M`/`coeffs.fmin`/`coeffs.fmax`/`coeffs.eta` also feed
    # `_seobv5f2_phi_scalar`/`_seobv5f2_ampl_scalar` directly (not just the ts/phis/omegas tables),
    # so their own tangents are needed too for a caller differentiating THOSE functions w.r.t. a
    # `Duplicated(coeffs, dcoeffs)` built from this return value (`BatchedNUTS.jl`'s own consumer).
    dM, dM_sec = _seobv5_M_and_Msec_jvp(mc, eta, GMsun_over_c3, dmc, deta)
    dfmin = (domegas_raw[1] - omegas_raw[1] * dM_sec / M_sec) / (pi * M_sec)
    dfmax = (domegas_raw[clamp(nreal, 1, NMAX)] - omegas_raw[clamp(nreal, 1, NMAX)] * dM_sec / M_sec) / (pi * M_sec)

    return (success = false, M_sec = dM_sec, eta = deta, M = dM, nreal = 0,
            ts = dts, phis = dphis, omegas = SVector(domegas), fmin = dfmin, fmax = dfmax)
end

# ============================================================================
# Reverse-mode (VJP) counterpart of everything above: replaces 11
# per-sampled-parameter `seobv5f2_coeffs_jvp` calls (one Enzyme.Forward pass
# each) with ONE reverse pass over `_seobv5f2_coeffs`'s construction, taking a
# cotangent on its own output (`ts`,`phis`,`omegas`,`M_sec`,`eta`,`M`,`fmin`,
# `fmax`) and returning `(bar_mc, bar_eta, bar_chi1, bar_chi2)` directly --
# collapsing what would otherwise be 11 launches down to close to 1, PROVIDED
# it is wired to run as a single Enzyme.Reverse call on the small
# "given coeffs" kernel plus one call here (this file supplies
# only the coeffs-level VJP).
#
# **Same "small scalar leaf function" discipline as the JVP above, just mirrored
# into reverse mode**: every `Enzyme.autodiff` call in this section targets a
# small, non-allocating, loop-free SCALAR function (never the loopy/fixed-buffer
# PA driver itself) -- confirmed safe for `Enzyme.Reverse` specifically (not just
# `Enzyme.Forward`) by a three-way finite-difference/Forward/Reverse cross-check
# on the "given coeffs" kernel. A multi-output scalar leaf (e.g. `_seobv5_gradient_full`,
# which returns `(dHdr,dHdpr,dHdpphi)`) is differentiated via the standard
# Enzyme reverse-mode technique for that case: a `_seeded` wrapper returning
# `b1*out1+b2*out2+...` is reverse-differentiated instead, and by linearity that
# yields the VJP contraction against seed `(b1,b2,...)` directly.
#
# **Where reverse mode genuinely helps beyond "same math, opposite direction"**:
# the two implicit-function-theorem solves (`_seobv5_circular_ic`'s 2D Newton
# solve, `_seobv5_adiabatic_j0_fixed`'s per-node 1D solve) need the SAME 2x2 (or
# scalar) Jacobian either way, but forward mode needs one extra JVP call per
# INPUT direction to get the "parameter" side of the residual's total derivative,
# while reverse mode gets the gradient w.r.t. ALL parameters in ONE call (seed
# the residual's own output cotangent, not each input in turn) -- see
# `_seobv5_circular_ic_reverse`/`_seobv5_adiabatic_j0_reverse` below.
#
# **Where it's genuinely harder than "just transpose"**: the 8-pass alternating
# post-adiabatic sweep (`_seobv5_postadiabatic_solution_fixed`'s exact control
# flow) is a real sequential computation -- reversing it means walking the 8
# passes backward, maintaining running cotangents for `pr`/`pphi` that get
# REPLACED (not accumulated) at the pass that actually computed that array's
# new value and ACCUMULATED at every other pass that merely READ it (directly,
# or via `_seobv5_findiff_derivative9_fixed`'s 9-point stencil over the whole
# array) -- see `seobv5f2_coeffs_vjp`'s own pass-reversal loop below for the
# exact bookkeeping. The two array-level linear maps this pipeline uses
# (`_seobv5_findiff_derivative9_fixed`, `_seobv5_cumulative_integral7_fixed`)
# are reversed by mechanical transpose-by-scatter (same index/weight structure
# as the primal's own gather loop, just scattering into the input instead of
# gathering from it) -- exact by construction (validated against the defining
# adjoint identity `dot(bar_v, J*dy) == dot(J^T*bar_v, dy)` to machine precision
# for random `dy`/`bar_v`, not just approximately via finite differences).
#
# **Validated**: (1) every leaf/composite reverse
# wrapper against central finite differences; (2) `_findiff9_adjoint`/
# `_cumint7_adjoint` against the exact adjoint identity (relative error
# ~1e-16, machine precision); (3) `seobv5f2_coeffs_vjp`'s FULL pipeline against
# `seobv5f2_coeffs_jvp` via the JVP-VJP consistency identity
# `dot(bar_out, JVP(dparams)) == dot(VJP(bar_out), dparams)` for random
# `dparams`/`bar_out`, isolated per output channel AND combined (relative error
# ~1e-10 to 1e-13 -- this identity is EXACT regardless of the pipeline's own
# Newton-solve/discretization approximations, unlike a finite-difference check,
# so it is the primary correctness tool here); (4) one direct finite-difference
# cross-check against `_seobv5f2_coeffs`'s real output at
# `mc=28.0961,eta=0.2479,chi1=0.3,chi2=-0.2,f_min=20` (relative error
# ~1e-5 to 1e-6, matching the JVP's own documented tolerance class).
#
# A real bug was found and fixed during this derivation, the same "trust but
# verify" discipline the rest of this codebase follows: an early version of
# `_seobv5_circular_ic_reverse`'s adjoint 2x2 linear solve was missing an
# overall minus sign (solving `(J^T)^{-1}*(bar_rhat,bar_pphi)` instead of
# `-(J^T)^{-1}*(bar_rhat,bar_pphi)`) -- every affected output was flipped in
# sign by exactly this factor, caught by isolating the two IFT-reverse
# functions and finite-differencing them independently rather than trusting
# the algebra by inspection.
#
# **Not yet done**: wiring this into `BatchedNUTS.jl`/`KernelFused.jl` as an
# actual `_uses_forward_mode_gradient`-style alternative gradient path, and
# measuring real wall-clock speedup (GPU or CPU) before trusting it's actually
# faster in production -- this file supplies a validated building block, not
# yet a production gradient path. Not validated on CUDA.
# ============================================================================

@inline function _seobv5_gradient_full_seeded(r, pr, pphi, nu, X1, X2, chi1, chi2, a6, dSO, b1, b2, b3)
    d1, d2, d3 = _seobv5_gradient_full(r, pr, pphi, nu, X1, X2, chi1, chi2, a6, dSO)
    return b1 * d1 + b2 * d2 + b3 * d3
end
function _seobv5_gradient_full_vjp(r, pr, pphi, nu, X1, X2, chi1, chi2, a6, dSO, b1, b2, b3)
    (g,) = Enzyme.autodiff(Enzyme.Reverse, _seobv5_gradient_full_seeded,
        Enzyme.Active(r), Enzyme.Active(pr), Enzyme.Active(pphi),
        Enzyme.Active(nu), Enzyme.Active(X1), Enzyme.Active(X2),
        Enzyme.Active(chi1), Enzyme.Active(chi2), Enzyme.Active(a6), Enzyme.Active(dSO),
        Enzyme.Const(b1), Enzyme.Const(b2), Enzyme.Const(b3))
    return g[1:10]  # (bar_r, bar_pr, bar_pphi, bar_nu, bar_X1, bar_X2, bar_chi1, bar_chi2, bar_a6, bar_dSO)
end

@inline function _seobv5_gradient_full2_seeded(r, pr, pphi, nu, X1, X2, chi1, chi2, a6, dSO, b1, b2, b3, b4)
    d1, d2, d3, d4 = _seobv5_gradient_full2(r, pr, pphi, nu, X1, X2, chi1, chi2, a6, dSO)
    return b1 * d1 + b2 * d2 + b3 * d3 + b4 * d4
end
function _seobv5_gradient_full2_vjp(r, pr, pphi, nu, X1, X2, chi1, chi2, a6, dSO, b1, b2, b3, b4)
    (g,) = Enzyme.autodiff(Enzyme.Reverse, _seobv5_gradient_full2_seeded,
        Enzyme.Active(r), Enzyme.Active(pr), Enzyme.Active(pphi),
        Enzyme.Active(nu), Enzyme.Active(X1), Enzyme.Active(X2),
        Enzyme.Active(chi1), Enzyme.Active(chi2), Enzyme.Active(a6), Enzyme.Active(dSO),
        Enzyme.Const(b1), Enzyme.Const(b2), Enzyme.Const(b3), Enzyme.Const(b4))
    return g[1:10]
end

# 21-arg leaf closed-form corrections. Single scalar output -> plain Active args, seed via
# post-multiplying the raw (seed=1) gradient by `b` (linear, so this is exact, not approximate).
function _seobv5_pr_eqn_vjp(args::NTuple{21,Float64}, b::Float64)
    (g,) = Enzyme.autodiff(Enzyme.Reverse, _pr_eqn21, ntuple(i -> Enzyme.Active(args[i]), 21)...)
    return ntuple(i -> b * g[i], 21)
end
function _seobv5_pphi_eqn_fixed_vjp(args::NTuple{21,Float64}, b::Float64)
    (g,) = Enzyme.autodiff(Enzyme.Reverse, _pphi_eqn21, ntuple(i -> Enzyme.Active(args[i]), 21)...)
    return ntuple(i -> b * g[i], 21)
end

@inline function _seobv5_params_from_mc_eta_chi_seeded(mc, eta, chi1, chi2, b::NTuple{15,Float64})
    r = _seobv5_params_from_mc_eta_chi(mc, eta, chi1, chi2)
    s = 0.0
    for i in 1:15
        s += b[i] * r[i]
    end
    return s
end
function _seobv5_params_from_mc_eta_chi_vjp(mc, eta, chi1, chi2, b::NTuple{15,Float64})
    (g,) = Enzyme.autodiff(Enzyme.Reverse, _seobv5_params_from_mc_eta_chi_seeded,
        Enzyme.Active(mc), Enzyme.Active(eta), Enzyme.Active(chi1), Enzyme.Active(chi2), Enzyme.Const(b))
    return g[1:4]  # (bar_mc, bar_eta, bar_chi1, bar_chi2)
end

function _seobv5_omega0_vjp(mc, eta, f_min, GMsun_over_c3_, b::Float64)
    (g,) = Enzyme.autodiff(Enzyme.Reverse, _seobv5_omega0_from_mc_eta,
        Enzyme.Active(mc), Enzyme.Active(eta), Enzyme.Const(f_min), Enzyme.Const(GMsun_over_c3_))
    return (b * g[1], b * g[2])  # (bar_mc, bar_eta)
end

@inline function _seobv5_M_and_Msec_seeded(mc, eta, GMsun_over_c3_, b1, b2)
    M, M_sec = _seobv5_M_and_Msec_from_mc_eta(mc, eta, GMsun_over_c3_)
    return b1 * M + b2 * M_sec
end
function _seobv5_M_and_Msec_vjp(mc, eta, GMsun_over_c3_, b1::Float64, b2::Float64)
    (g,) = Enzyme.autodiff(Enzyme.Reverse, _seobv5_M_and_Msec_seeded,
        Enzyme.Active(mc), Enzyme.Active(eta), Enzyme.Const(GMsun_over_c3_), Enzyme.Const(b1), Enzyme.Const(b2))
    return (g[1], g[2])  # (bar_mc, bar_eta)
end

function _seobv5_r_final_vjp(nu, X1, X2, chi1, chi2, b::Float64)
    (g,) = Enzyme.autodiff(Enzyme.Reverse, _seobv5_r_final_from_params,
        Enzyme.Active(nu), Enzyme.Active(X1), Enzyme.Active(X2), Enzyme.Active(chi1), Enzyme.Active(chi2))
    return ntuple(i -> b * g[i], 5)  # (bar_nu, bar_X1, bar_X2, bar_chi1, bar_chi2)
end

@inline function _pa_clampidx(j::Int, ::Val{NMAX}) where {NMAX}
    return min(max(j, 1), NMAX)
end

"""
    _findiff9_adjoint(bar_v, y, nreal, h) -> (bar_y, bar_h)

Adjoint (VJP) of `_seobv5_findiff_derivative9_fixed(y, nreal, h)` w.r.t. its array argument `y` and
its scalar step `h`, for a given output cotangent `bar_v`. `dydx = stencil(y)/h` is linear in `y`
(a fixed sparse gather per output index, exactly mirroring the primal's own three-region row
selection) -- its adjoint is the TRANSPOSE of that same sparse map, computed by mechanically
inverting the primal's gather loop into a scatter loop with the identical index/weight structure,
scaled by `1/h`. `bar_h` follows from `d(dydx[i])/dh = -dydx[i]/h`. Validated against the exact
adjoint identity `dot(bar_v, stencil(dy)) == dot(bar_y, dy)` to machine precision, not just by
finite differences (see this section's own header comment).
"""
function _findiff9_adjoint(bar_v::SVector{NMAX,Float64}, y::SVector{NMAX,Float64}, nreal::Int, h::Float64) where {NMAX}
    C = _SEOBV5_FINDIFF9_COEFFS
    v = _seobv5_findiff_derivative9_fixed(y, nreal, h)
    bar_y = zero(MVector{NMAX,Float64})
    @inbounds for i in 1:nreal
        bar_val_i = bar_v[i] / h
        is_left = i <= 4
        is_right = (i >= nreal - 3) && (i <= nreal)
        if is_left
            row = C[clamp(i, 1, 4)]
            for k in 1:9
                bar_y[_pa_clampidx(k, Val(NMAX))] += row[k] * bar_val_i
            end
        elseif is_right
            row = C[clamp(i - nreal + 9, 6, 9)]
            for k in 1:9
                bar_y[_pa_clampidx(nreal - 9 + k, Val(NMAX))] += row[k] * bar_val_i
            end
        else
            row = C[5]
            for k in 1:9
                bar_y[_pa_clampidx(i - 5 + k, Val(NMAX))] += row[k] * bar_val_i
            end
        end
    end
    bar_h = -dot(bar_v, v) / h
    return SVector(bar_y), bar_h
end

"""
    _cumint7_adjoint(bar_v, y, nreal, h) -> (bar_y, bar_h)

Adjoint of `_seobv5_cumulative_integral7_fixed(y, nreal, h) = h*cumsum(val(y))` (bilinear in
`(y,h)`, mirroring `_seobv5_cumulative_integral7_fixed_jvp`'s own JVP decomposition). The `cumsum`
layer's adjoint is a suffix sum (transpose of a lower-triangular all-ones matrix); the per-index
`val` gather's adjoint is the same mechanical scatter-transpose `_findiff9_adjoint` uses. `bar_h`
follows from `v` being exactly linear in `h` at fixed `y` (`v(h) = h*v(1)`, so `dv/dh = v/h`).
"""
function _cumint7_adjoint(bar_v::SVector{NMAX,Float64}, y::SVector{NMAX,Float64}, nreal::Int, h::Float64) where {NMAX}
    ROW = _SEOBV5_CUMINT7_COEFFS
    v = _seobv5_cumulative_integral7_fixed(y, nreal, h)
    bar_contrib = zero(MVector{NMAX,Float64})
    acc = 0.0
    @inbounds for i in NMAX:-1:1
        acc += bar_v[i]
        bar_contrib[i] = acc
    end
    bar_y = zero(MVector{NMAX,Float64})
    @inbounds for i in 1:nreal
        bar_val_i = bar_contrib[i] * h
        is_near_left = (i >= 2) && (i <= 4)
        is_near_right = (i >= nreal - 2) && (i <= nreal)
        is_interior = (i >= 5) && (i <= nreal - 3)
        if is_near_left
            row = ROW[clamp(i - 1, 1, 3)]
            for k in 1:8
                bar_y[_pa_clampidx(k, Val(NMAX))] += row[k] * bar_val_i
            end
        elseif is_near_right
            row = ROW[clamp(i - nreal + 7, 5, 7)]
            for k in 1:8
                bar_y[_pa_clampidx(nreal - 8 + k, Val(NMAX))] += row[k] * bar_val_i
            end
        elseif is_interior
            row = ROW[4]
            for k in 1:8
                bar_y[_pa_clampidx(i - 5 + k, Val(NMAX))] += row[k] * bar_val_i
            end
        end
    end
    bar_h = dot(bar_v, v) / h
    return SVector(bar_y), bar_h
end

"""
    _seobv5_adiabatic_j0_reverse(r, pphi0, nu, X1, X2, chi1, chi2, a6, dSO, bar_pphi0)
        -> (bar_r, bar_nu, bar_X1, bar_X2, bar_chi1, bar_chi2, bar_a6, bar_dSO)

Reverse-mode (IFT) sensitivity of `_seobv5_adiabatic_j0_fixed`'s per-node circular-orbit Newton
solve (`F(p) = dHdr(r_i,0,p,params) = 0`), evaluated AT the already-converged `pphi0` -- one
`_seobv5_gradient_full_vjp` call per grid node (vs. the JVP's own two calls per node), since reverse
mode gets `dF/dp` and the full `dF/dparam` vector from the SAME seeded call.
"""
function _seobv5_adiabatic_j0_reverse(r::SVector{NMAX,Float64}, pphi0::SVector{NMAX,Float64},
        nu, X1, X2, chi1, chi2, a6, dSO, bar_pphi0::SVector{NMAX,Float64}) where {NMAX}
    bar_r = zero(MVector{NMAX,Float64})
    bar_nu = 0.0; bar_X1 = 0.0; bar_X2 = 0.0; bar_chi1 = 0.0; bar_chi2 = 0.0; bar_a6 = 0.0; bar_dSO = 0.0
    @inbounds for i in 1:NMAX
        bpp = bar_pphi0[i]
        bpp == 0.0 && continue
        p = pphi0[i]
        g = _seobv5_gradient_full_vjp(r[i], 0.0, p, nu, X1, X2, chi1, chi2, a6, dSO, 1.0, 0.0, 0.0)
        dFdp = g[3]  # d(dHdr)/dpphi, at (pr=0, dHdr seeded)
        lambda = bpp / dFdp
        bar_r[i] += -lambda * g[1]
        bar_nu += -lambda * g[4]; bar_X1 += -lambda * g[5]; bar_X2 += -lambda * g[6]
        bar_chi1 += -lambda * g[7]; bar_chi2 += -lambda * g[8]
        bar_a6 += -lambda * g[9]; bar_dSO += -lambda * g[10]
    end
    return SVector(bar_r), bar_nu, bar_X1, bar_X2, bar_chi1, bar_chi2, bar_a6, bar_dSO
end

"""
    _seobv5_circular_ic_reverse(rhat, pphi, nu, X1, X2, chi1, chi2, a6, dSO, bar_rhat, bar_pphi)
        -> (bar_omega0, bar_nu, bar_X1, bar_X2, bar_chi1, bar_chi2, bar_a6, bar_dSO)

Reverse-mode (IFT) sensitivity of `_seobv5_circular_ic`'s 2D Newton solve (`F1 = dHdpphi-omega0 = 0`,
`F2 = dHdr = 0`), evaluated at the converged `(rhat,pphi)`, given a cotangent on the solution.
Builds the SAME 2x2 Jacobian `_seobv5_circular_ic_ift` (the JVP counterpart) does, from 2
`_seobv5_gradient_full_jvp` calls, then solves the ADJOINT system `J^T*lambda = -(bar_rhat,bar_pphi)`
(the sign follows from `[drhat;dpphi] = -J^{-1}[dF1;dF2] + J^{-1}[domega0;0]`: contracting with
`(bar_rhat,bar_pphi)` and defining `lambda = -(J^T)^{-1}[bar_rhat;bar_pphi]` gives
`bar_omega0 = -lambda1` and `bar_param = lambda1*dF1/dparam + lambda2*dF2/dparam`) -- an initial
version omitted this minus sign, flipping every output's sign; caught and fixed by finite-
differencing this function in isolation (see this section's own header comment). The full
parameter gradient then comes from ONE seeded `_seobv5_gradient_full_vjp` call (`b1=lambda2` on the
`dHdr` channel, `b3=lambda1` on the `dHdpphi` channel), vs. the JVP's own 3 calls to assemble the
same information (2 for the Jacobian, 1 for the parameter direction).
"""
function _seobv5_circular_ic_reverse(rhat, pphi, nu, X1, X2, chi1, chi2, a6, dSO, bar_rhat, bar_pphi)
    dg1_r, _, dg3_r = _seobv5_gradient_full_jvp(rhat, 0.0, pphi, nu, X1, X2, chi1, chi2, a6, dSO,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    J11, J21 = dg3_r, dg1_r
    dg1_p, _, dg3_p = _seobv5_gradient_full_jvp(rhat, 0.0, pphi, nu, X1, X2, chi1, chi2, a6, dSO,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    J12, J22 = dg3_p, dg1_p
    det = J11 * J22 - J12 * J21
    lambda1 = -(J22 * bar_rhat - J21 * bar_pphi) / det
    lambda2 = -(-J12 * bar_rhat + J11 * bar_pphi) / det
    bar_omega0 = -lambda1
    g = _seobv5_gradient_full_vjp(rhat, 0.0, pphi, nu, X1, X2, chi1, chi2, a6, dSO, lambda2, 0.0, lambda1)
    bar_nu, bar_X1, bar_X2, bar_chi1, bar_chi2, bar_a6, bar_dSO = g[4], g[5], g[6], g[7], g[8], g[9], g[10]
    return bar_omega0, bar_nu, bar_X1, bar_X2, bar_chi1, bar_chi2, bar_a6, bar_dSO
end

"""
    seobv5f2_coeffs_vjp(model::SEOBNRv5AlignedSpinF2, mc, eta, chi1, chi2,
        bar_ts, bar_phis, bar_omegas, bar_M_sec, bar_eta, bar_M, bar_fmin, bar_fmax;
        f_min=20.0, GMsun_over_c3_=GMsun_over_c3, order=8, dr0_step=0.2)
        -> (bar_mc, bar_eta, bar_chi1, bar_chi2)

Reverse-mode (VJP) counterpart of [`seobv5f2_coeffs_jvp`](@ref): given a cotangent on EVERY
differentiable field of `GravitationalWaveforms._seobv5f2_coeffs`'s return value (the three `NMAX`-length
tables `ts`/`phis`/`omegas` plus the scalars `M_sec`/`eta`/`M`/`fmin`/`fmax`), returns the gradient
w.r.t. the 4 intrinsic parameters `(mc,eta,chi1,chi2)` in ONE pass -- vs. `seobv5f2_coeffs_jvp`
needing one call per intrinsic direction. Recomputes the same (cheap, non-Enzyme) primal pipeline
`_seobv5f2_coeffs` itself computes, additionally storing the post-adiabatic sweep's full 8-pass
history (`pr_hist`/`pphi_hist`, `order+1` `SVector{NMAX,Float64}` snapshots each -- cheap at
`NMAX=256`) since the reverse pass needs every intermediate state, not just the final one.

**Caller contract for `bar_ts`/`bar_phis`/`bar_omegas`**: exactly mirrors what a real downstream
consumer (differentiating `_seobv5f2_phi_scalar`/`_seobv5f2_ampl_scalar`'s own Hermite
reconstruction, or `_forward_mode_gradient!`'s existing per-node kernel) actually produces --
nonzero cotangent only in the real region `[1,nreal]`; entries `i>nreal` may be nonzero too (handled
correctly, since the primal's own padding is a genuine, differentiable function of the real-region
values via `_seobv5_cumulative_integral7_fixed`'s flat-tail convention and `omegas`' own
`om_n+(i-n)` continuation -- both correctly absorbed back into the real region here, exactly as
their own forward tangent construction does in `seobv5f2_coeffs_jvp`).

See this file's own "Reverse-mode (VJP) counterpart" section header comment above for the full
derivation discipline, validation methodology, and the one real bug this derivation found and fixed.
"""
function seobv5f2_coeffs_vjp(::SEOBNRv5AlignedSpinF2, mc, eta, chi1, chi2,
        bar_ts::SVector{NMAX,Float64}, bar_phis::SVector{NMAX,Float64}, bar_omegas::SVector{NMAX,Float64},
        bar_M_sec, bar_eta, bar_M, bar_fmin, bar_fmax;
        f_min = 20.0, GMsun_over_c3_ = GMsun_over_c3, order::Int = 8, dr0_step::Float64 = 0.2) where {NMAX}
    NMAX == _SEOBV5_PA_NMAX || throw(ArgumentError("bar_ts/bar_phis/bar_omegas must be SVector{$(_SEOBV5_PA_NMAX)}"))

    nu, X1, X2, delta, chiS, chiA, atest, a6, dSO, absN33, absN21, absN44, absN32, absN55, absN43 =
        _seobv5_params_from_mc_eta_chi(mc, eta, chi1, chi2)

    M_sec = (mc / eta^0.6) * GMsun_over_c3_
    v0 = cbrt(pi * M_sec * f_min)
    omega0 = v0^3

    hp = SEOBNRv5HamParams(nu, X1, X2, chi1, chi2, a6, dSO)
    r0, pphi0_ic = _seobv5_circular_ic(omega0, hp)

    chi_eff = X1 * chi1 + X2 * chi2
    r_final_prefactor = 2.7 + chi_eff * (1.0 - 4.0 * nu)
    r_isco, _ = _seobv5_kerr_isco(chi1, chi2, X1, X2)
    r_final_raw = r_final_prefactor * r_isco
    r_final = max(10.0, r_final_raw)
    floor_active = r_final_raw < 10.0

    nreal_raw = ceil(Int, (r0 - r_final) / dr0_step)
    success = !(nreal_raw <= 4 || r0 <= 11.5)
    nreal_bumped = nreal_raw < 10 ? 10 : nreal_raw
    nreal = success ? min(nreal_bumped, NMAX) : 10

    r0_grid = success ? r0 : 50.0
    r_final_grid = success ? r_final : 40.0
    dr_step = (r0_grid - r_final_grid) / (nreal - 1)

    r = _seobv5_pa_grid_fixed(r0_grid, dr_step, nreal, Val(NMAX))
    pphi0_pa = _seobv5_adiabatic_j0_fixed(r, hp, nreal)  # padding lanes skipped throughout, as in the JVP

    # --- forward recompute of the PA sweep, storing the full pass history (the reverse pass
    # below needs every intermediate (pr,pphi) snapshot, not just the final one) ---
    pr_hist = Vector{SVector{NMAX,Float64}}(undef, order + 1)
    pphi_hist = Vector{SVector{NMAX,Float64}}(undef, order + 1)
    pr_cur = zero(MVector{NMAX,Float64})
    pphi_cur = MVector(pphi0_pa)
    pr_hist[1] = SVector(pr_cur)
    pphi_hist[1] = SVector(pphi_cur)
    for n in 1:order
        if isodd(n)
            dpphi_dr = -_seobv5_findiff_derivative9_fixed(SVector(pphi_cur), nreal, dr_step)
            @inbounds for i in 1:nreal
                dpphi_dr_i = i <= nreal ? dpphi_dr[i] : 1.0
                pr_cur[i] = _pr_eqn21(pr_cur[i], r[i], pphi_cur[i], dpphi_dr_i, nu, X1, X2, chi1, chi2, a6, dSO,
                    delta, chiS, chiA, atest, absN33, absN21, absN44, absN32, absN55, absN43)
            end
        else
            dpr_dr = -_seobv5_findiff_derivative9_fixed(SVector(pr_cur), nreal, dr_step)
            @inbounds for i in 1:nreal
                dpr_dr_i = i <= nreal ? dpr_dr[i] : 1.0
                pphi_cur[i] = _pphi_eqn21(pphi_cur[i], r[i], pr_cur[i], dpr_dr_i, nu, X1, X2, chi1, chi2, a6, dSO,
                    delta, chiS, chiA, atest, absN33, absN21, absN44, absN32, absN55, absN43)
            end
        end
        pr_hist[n+1] = SVector(pr_cur)
        pphi_hist[n+1] = SVector(pphi_cur)
    end
    pr, pphi = pr_hist[order+1], pphi_hist[order+1]

    # --- post-PA per-node Hamiltonian loop (omegas_raw, dt_dr, dphi_dr): forward values needed
    # by the reverse pass below, mirrors _seobv5f2_coeffs's own omegas_raw loop exactly ---
    omegas_raw = zero(MVector{NMAX,Float64})
    dt_dr = zero(MVector{NMAX,Float64})
    dphi_dr = zero(MVector{NMAX,Float64})
    dHdpr_arr = zero(MVector{NMAX,Float64})
    xi_arr = zero(MVector{NMAX,Float64})
    @inbounds for i in 1:nreal
        xi, dHdpr, dHdpphi = _seobv5_sym_tphi(r[i], pr[i], pphi[i], hp)
        omegas_raw[i] = dHdpphi
        xi_arr[i] = xi
        dHdpr_arr[i] = dHdpr
        u = dHdpr * xi
        dt_dr[i] = 1.0 / u
        dphi_dr[i] = dHdpphi * dt_dr[i]
    end

    # --- bar_omegas (external) -> bar_omegas_raw: the padding tail omegas[i]=om_n+(i-n) for i>nreal
    # is an additive-constant-offset copy of omegas_raw[nreal], so its ENTIRE cotangent folds into
    # bar_omegas_raw[nreal] -- mirrors seobv5f2_coeffs_jvp's own om_n construction in reverse. Also
    # folds in fmin/fmax's OWN separate read of omegas_raw[1]/omegas_raw[om_n_idx] (fmin=omegas[1]/
    # (pi*M_sec), fmax=omegas[nreal]/(pi*M_sec)) -- this MUST happen before the main per-node
    # reverse loop below, since bar_pr/bar_pphi get fully consumed and repurposed by the PA-sweep/
    # adiabatic reverse afterward; adding a contribution later would silently drop it (caught during
    # validation by isolating the bar_fmin/bar_fmax JVP-VJP consistency check).
    om_n_idx = clamp(nreal, 1, NMAX)
    bar_omegas_raw = MVector(bar_omegas)
    @inbounds for i in 1:NMAX
        if i > nreal
            bar_omegas_raw[om_n_idx] += bar_omegas[i]
            bar_omegas_raw[i] = 0.0
        end
    end
    bar_omegas_raw[1] += bar_fmin / (pi * M_sec)
    bar_omegas_raw[om_n_idx] += bar_fmax / (pi * M_sec)
    bar_M_sec_extra = -bar_fmin * omegas_raw[1] / (pi * M_sec^2) - bar_fmax * omegas_raw[om_n_idx] / (pi * M_sec^2)

    # --- bar_ts, bar_phis -> bar_dt_dr, bar_dphi_dr, bar_dr_step (via cumint7 adjoint; h=-dr_step
    # matches the primal's own ts=cumint7(dt_dr,nreal,-dr_step) call) ---
    bar_dt_dr_arr, bar_h1 = _cumint7_adjoint(bar_ts, SVector(dt_dr), nreal, -dr_step)
    bar_dphi_dr_arr, bar_h2 = _cumint7_adjoint(bar_phis, SVector(dphi_dr), nreal, -dr_step)
    bar_dr_step = -bar_h1 - bar_h2  # h = -dr_step, chain rule through the negation

    # --- per-node reverse of the Hamiltonian loop -> bar_r, bar_pr, bar_pphi, bar_params ---
    bar_r = zero(MVector{NMAX,Float64})
    bar_pr = zero(MVector{NMAX,Float64})
    bar_pphi = zero(MVector{NMAX,Float64})
    bar_nu = 0.0; bar_X1 = 0.0; bar_X2 = 0.0; bar_chi1 = 0.0; bar_chi2 = 0.0; bar_a6 = 0.0; bar_dSO = 0.0
    @inbounds for i in 1:nreal
        bdHdpphi = bar_omegas_raw[i] + bar_dphi_dr_arr[i] * dt_dr[i]
        bdt_dr = bar_dt_dr_arr[i] + bar_dphi_dr_arr[i] * omegas_raw[i]
        bu = -bdt_dr * dt_dr[i]^2
        bdHdpr = bu * xi_arr[i]
        bxi = bu * dHdpr_arr[i]
        g = _seobv5_gradient_full2_vjp(r[i], pr[i], pphi[i], nu, X1, X2, chi1, chi2, a6, dSO,
            0.0, bdHdpr, bdHdpphi, bxi)
        bar_r[i] += g[1]; bar_pr[i] += g[2]; bar_pphi[i] += g[3]
        bar_nu += g[4]; bar_X1 += g[5]; bar_X2 += g[6]; bar_chi1 += g[7]; bar_chi2 += g[8]
        bar_a6 += g[9]; bar_dSO += g[10]
    end

    # --- reverse through the 8-pass alternating sweep: walk passes backward, maintaining running
    # bar_pr/bar_pphi that get REPLACED at the pass that computed that array's new value (the
    # cotangent of its PREVIOUS/"trial" value becomes the next iteration's seed) and ACCUMULATED at
    # every pass that merely read it (directly as an argument, or via the stencil's 9-point window
    # over the whole array). Only i<=nreal is ever touched here -- padding-region (i>nreal) cotangents
    # of pr/pphi are ALWAYS exactly zero by construction (bar_omegas_raw/bar_dt_dr_arr/bar_dphi_dr_arr
    # are all zero there, per the per-node loop above), so restricting these loops to 1:nreal is
    # correct, not a truncation (confirmed during validation: seeding padding-region cotangents that
    # never legitimately arise here reproduces a mismatch, seeding only what the real pipeline
    # actually produces does not). ---
    bar_delta = 0.0; bar_chiS = 0.0; bar_chiA = 0.0; bar_atest = 0.0
    bar_absN33 = 0.0; bar_absN21 = 0.0; bar_absN44 = 0.0; bar_absN32 = 0.0; bar_absN55 = 0.0; bar_absN43 = 0.0

    for n in order:-1:1
        if isodd(n)
            pphi_n = pphi_hist[n]
            pr_n = pr_hist[n]
            dpphi_dr_full = -_seobv5_findiff_derivative9_fixed(pphi_n, nreal, dr_step)
            bar_dpphi_dr_local = zero(MVector{NMAX,Float64})
            new_bar_pr = zero(MVector{NMAX,Float64})
            @inbounds for i in 1:nreal
                dpphi_dr_i = dpphi_dr_full[i]
                args = (pr_n[i], r[i], pphi_n[i], dpphi_dr_i,
                        nu, X1, X2, chi1, chi2, a6, dSO, delta, chiS, chiA, atest,
                        absN33, absN21, absN44, absN32, absN55, absN43)
                gg = _seobv5_pr_eqn_vjp(args, bar_pr[i])
                new_bar_pr[i] = gg[1]
                bar_r[i] += gg[2]
                bar_pphi[i] += gg[3]
                bar_dpphi_dr_local[i] = gg[4]
                bar_nu += gg[5]; bar_X1 += gg[6]; bar_X2 += gg[7]; bar_chi1 += gg[8]; bar_chi2 += gg[9]
                bar_a6 += gg[10]; bar_dSO += gg[11]; bar_delta += gg[12]; bar_chiS += gg[13]; bar_chiA += gg[14]
                bar_atest += gg[15]; bar_absN33 += gg[16]; bar_absN21 += gg[17]; bar_absN44 += gg[18]
                bar_absN32 += gg[19]; bar_absN55 += gg[20]; bar_absN43 += gg[21]
            end
            bar_pphi_from_stencil, bar_h_c = _findiff9_adjoint(SVector(-bar_dpphi_dr_local), pphi_n, nreal, dr_step)
            bar_pphi .+= bar_pphi_from_stencil
            bar_dr_step += bar_h_c
            bar_pr = new_bar_pr
        else
            pr_n = pr_hist[n]
            pphi_n = pphi_hist[n]
            dpr_dr_full = -_seobv5_findiff_derivative9_fixed(pr_n, nreal, dr_step)
            bar_dpr_dr_local = zero(MVector{NMAX,Float64})
            new_bar_pphi = zero(MVector{NMAX,Float64})
            @inbounds for i in 1:nreal
                dpr_dr_i = dpr_dr_full[i]
                args = (pphi_n[i], r[i], pr_n[i], dpr_dr_i,
                        nu, X1, X2, chi1, chi2, a6, dSO, delta, chiS, chiA, atest,
                        absN33, absN21, absN44, absN32, absN55, absN43)
                gg = _seobv5_pphi_eqn_fixed_vjp(args, bar_pphi[i])
                new_bar_pphi[i] = gg[1]
                bar_r[i] += gg[2]
                bar_pr[i] += gg[3]
                bar_dpr_dr_local[i] = gg[4]
                bar_nu += gg[5]; bar_X1 += gg[6]; bar_X2 += gg[7]; bar_chi1 += gg[8]; bar_chi2 += gg[9]
                bar_a6 += gg[10]; bar_dSO += gg[11]; bar_delta += gg[12]; bar_chiS += gg[13]; bar_chiA += gg[14]
                bar_atest += gg[15]; bar_absN33 += gg[16]; bar_absN21 += gg[17]; bar_absN44 += gg[18]
                bar_absN32 += gg[19]; bar_absN55 += gg[20]; bar_absN43 += gg[21]
            end
            bar_pr_from_stencil, bar_h_c = _findiff9_adjoint(SVector(-bar_dpr_dr_local), pr_n, nreal, dr_step)
            bar_pr .+= bar_pr_from_stencil
            bar_dr_step += bar_h_c
            bar_pphi = new_bar_pphi
        end
    end
    # bar_pr now == cotangent of pr_hist[1] == 0 (constant) -> discard.
    # bar_pphi now == cotangent of pphi_hist[1] == pphi0_pa -> feed into the adiabatic IFT reverse.
    bar_pphi0_pa = SVector(bar_pphi)

    ir, inu, iX1, iX2, ichi1, ichi2, ia6, idSO =
        _seobv5_adiabatic_j0_reverse(r, pphi0_pa, nu, X1, X2, chi1, chi2, a6, dSO, bar_pphi0_pa)
    bar_r .+= ir
    bar_nu += inu; bar_X1 += iX1; bar_X2 += iX2; bar_chi1 += ichi1; bar_chi2 += ichi2; bar_a6 += ia6; bar_dSO += idSO

    # --- bar_r (NMAX) -> bar_r0_grid, bar_dr_step (via _seobv5_pa_grid_fixed's linear adjoint:
    # r[i] = r0_grid - (i-1)*dr_step for i<=nreal) ---
    bar_r0_grid = 0.0
    @inbounds for i in 1:nreal
        bar_r0_grid += bar_r[i]
        bar_dr_step += bar_r[i] * (-(i - 1))
    end
    bar_r0 = success ? bar_r0_grid : 0.0

    # dr_step = (r0_grid - r_final_grid)/(nreal-1)
    bar_r0_grid_from_drstep = bar_dr_step / (nreal - 1)
    bar_r_final_grid_from_drstep = -bar_dr_step / (nreal - 1)
    bar_r0 += success ? bar_r0_grid_from_drstep : 0.0
    bar_r_final = success ? bar_r_final_grid_from_drstep : 0.0

    # r_final = max(10.0, r_final_raw)
    bar_r_final_raw = floor_active ? 0.0 : bar_r_final
    nu_b, X1_b, X2_b, chi1_b, chi2_b = _seobv5_r_final_vjp(nu, X1, X2, chi1, chi2, bar_r_final_raw)
    bar_nu += nu_b; bar_X1 += X1_b; bar_X2 += X2_b; bar_chi1 += chi1_b; bar_chi2 += chi2_b

    # r0 = _seobv5_circular_ic(omega0, ...)[1] -- pphi0_ic's own cotangent is 0 (discarded, matches
    # seobv5f2_coeffs_jvp's own "dr0, _ = ..." -- only r0 feeds this pipeline downstream).
    bo, nu_c, X1_c, X2_c, chi1_c, chi2_c, a6_c, dSO_c =
        _seobv5_circular_ic_reverse(r0, pphi0_ic, nu, X1, X2, chi1, chi2, a6, dSO, bar_r0, 0.0)
    bar_omega0 = bo
    bar_nu += nu_c; bar_X1 += X1_c; bar_X2 += X2_c; bar_chi1 += chi1_c; bar_chi2 += chi2_c
    bar_a6 += a6_c; bar_dSO += dSO_c

    mc_o, eta_o = _seobv5_omega0_vjp(mc, eta, f_min, GMsun_over_c3_, bar_omega0)
    mc_m, eta_m = _seobv5_M_and_Msec_vjp(mc, eta, GMsun_over_c3_, bar_M, bar_M_sec)
    mc_f, eta_f = _seobv5_M_and_Msec_vjp(mc, eta, GMsun_over_c3_, 0.0, bar_M_sec_extra)

    # --- closed-form params chain reverse: (nu=eta identity, X1,X2,delta,chiS,chiA,atest,a6,dSO,
    # absN33,absN21,absN44,absN32,absN55,absN43) -> (mc,eta,chi1,chi2). NOTE: chi1_c/chi2_c
    # (circular_ic_reverse's own direct-chi1/chi2 contribution) were already folded into the running
    # bar_chi1/bar_chi2 accumulators above (`bar_chi1 += chi1_c`) -- do not add them again below,
    # only the closed-form params-chain reverse's own contribution (chi1_p/chi2_p). ---
    bparams15 = (bar_nu, bar_X1, bar_X2, bar_delta, bar_chiS, bar_chiA, bar_atest, bar_a6, bar_dSO,
                 bar_absN33, bar_absN21, bar_absN44, bar_absN32, bar_absN55, bar_absN43)
    mc_p, eta_p, chi1_p, chi2_p = _seobv5_params_from_mc_eta_chi_vjp(mc, eta, chi1, chi2, bparams15)

    out_mc = mc_o + mc_m + mc_f + mc_p
    out_eta = eta_o + eta_m + eta_f + eta_p + bar_eta
    out_chi1 = chi1_p + bar_chi1
    out_chi2 = chi2_p + bar_chi2

    return out_mc, out_eta, out_chi1, out_chi2
end

end # module SEOBNRv5AlignedSpinF2Adjoint
