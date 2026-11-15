using Test
using Enzyme
using GravitationalWaveBayes
using GravitationalWaveBayes.BatchedNUTS: gw_logp_and_grad
using GravitationalWaveforms: IMRPhenomXPHM  # not re-exported at GravitationalWaveBayes top level; XAS/XP/XHM already are

# ---------------------------------------------------------------------------------------------
# Validate gradients (Enzyme vs. finite differences) across the full parameter space, including
# near boundaries.
#
# What existing coverage already checks (NOT repeated here):
#   - test_enzyme_ka.jl's main testset: Enzyme-vs-FD agreement across the full PHYSICAL 11-row
#     parameter space at generic (interior) points.
#   - test_enzyme_ka.jl's "near the dec/theta_jn coordinate boundaries" testset: the PHYSICAL
#     antenna-response trig at its own poles.
#   - test_batched_nuts_gw_smoke.jl's "sin_dec/cos_theta_jn near the reparametrization boundary"
#     testset: the asin/acos SAMPLED-space singularity specifically.
#
# What none of those cover, and this file adds: Enzyme-vs-FD agreement for `gw_logp_and_grad`
# (the SAMPLED-space closure `NUTSCore.nuts_sample!` actually calls every leapfrog step, i.e.
# `_reparam_to_physical!` + the likelihood kernel + `_log_prior` all composed together) evaluated
# NEAR each of the plain BOX-bounded sampled parameters' own prior edges (`mc, q, chi1, chi2, dL,
# t_c` -- `_MC_PRIOR_BOUNDS` etc. in BatchedNUTS.jl) -- exactly the parameters the
# `NUTSBounds` reflection (see NUTSCore.jl) now actively pushes trajectories back towards/along.
# If Enzyme's gradient through `_reparam_to_physical!`/`_log_prior` were ever wrong specifically
# near one of these edges (e.g. a sign error that only shows up once a term is small), NUTS could
# silently mis-navigate exactly the region reflection now relies on being navigated correctly.
#
# Points are placed strictly inside each bound by a small margin (not exactly AT the bound) so the
# central finite-difference stencil (`x +/- h`) never crosses into the `-Inf` prior region itself
# (where `_log_prior` is genuinely discontinuous and a finite-difference derivative is meaningless
# by construction, not a gradient bug) -- this validates "near the boundary", which is what Phase
# 5's actual leapfrog-time reflection acts on, not "exactly at the measure-zero boundary point"
# (already covered, for the asin/acos-singular rows specifically, by the smoke test referenced
# above).
# ---------------------------------------------------------------------------------------------

@testset "Enzyme gradient vs finite differences near box-bounded sampled-space prior edges" begin
    ntime = 16
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    strain = zeros(ComplexF64, ntime)
    psd = ones(Float64, ntime)
    data = LikelihoodData(freqs, strain, psd)
    cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)

    logp_and_grad! = gw_logp_and_grad(data, cfg, freqs)

    # A safe, comfortably-interior baseline point, one row perturbed to sit near its own bound at
    # a time -- (mc, q, chi1, chi2, dL, t_c, phase, ra, sin_dec, cos_theta_jn, psi).
    base = [30.0, 0.5, 0.0, 0.0, 0.41, 0.0, 0.3, 1.2, -0.5, 0.2, 0.8]

    eps_margin = 1e-3
    near_bound_cases = [
        ("mc lo", 1, 10.0 + eps_margin), ("mc hi", 1, 80.0 - eps_margin),
        ("q lo", 2, 0.125 + eps_margin), ("q hi", 2, 1.0 - eps_margin),
        ("chi1 lo", 3, -1.0 + eps_margin), ("chi1 hi", 3, 1.0 - eps_margin),
        ("chi2 lo", 4, -1.0 + eps_margin), ("chi2 hi", 4, 1.0 - eps_margin),
        ("dL lo", 5, 0.05 + eps_margin), ("dL hi", 5, 4.0 - eps_margin),
        ("t_c lo", 6, -0.1 + eps_margin), ("t_c hi", 6, 0.1 - eps_margin),
    ]

    h = 1e-6
    for (label, row, value) in near_bound_cases
        positions = reshape(copy(base), 11, 1)
        positions[row, 1] = value

        grad_out = zeros(Float64, 11, 1)
        logp_out = zeros(Float64, 1)
        logp_and_grad!(grad_out, logp_out, positions, trues(1))
        @test all(isfinite, grad_out)
        @test isfinite(logp_out[1])

        # Finite-difference derivative of the SAME (sampled-space, likelihood+prior) forward pass,
        # only in the one perturbed row -- central difference, margin chosen so x+/-h never
        # crosses this row's own bound (eps_margin=1e-3 >> h=1e-6).
        function fwd(x)
            p = copy(positions)
            p[row, 1] = x
            lp = zeros(Float64, 1)
            g = zeros(Float64, 11, 1)
            logp_and_grad!(g, lp, p, trues(1))
            return lp[1]
        end
        fd = (fwd(value + h) - fwd(value - h)) / (2h)

        @test isapprox(grad_out[row, 1], fd; rtol=1e-3, atol=1e-5)
    end
end

# ---------------------------------------------------------------------------------------------
# Extends the IMRPhenomD-only sweep above to every OTHER model
# this package's likelihood/relative-binning kernels actually wire in. Confirmed by reading
# KernelFused.jl's/RelativeBinning.jl's own `using GravitationalWaveforms: ...` lines and their `model isa
# ...` dispatch branches (not assumed): the full set is IMRPhenomD (above), IMRPhenomXAS,
# IMRPhenomXP, IMRPhenomXHM, IMRPhenomXPHM. IMRPhenomXAS/IMRPhenomXHM are aligned-spin and share
# IMRPhenomD's exact 11-row sampled space (`_reparam_to_physical!`'s row convention dispatches on
# `size(positions,1)`, not `model`), so the SAME 12 near-bound cases (mc/q/chi1/chi2/dL/t_c x
# lo/hi) apply verbatim. IMRPhenomXP/IMRPhenomXPHM are precessing (15-row): 4 more box-bounded
# rows (chi1x, chi1y, chi2x, chi2y) -- 20 near-bound cases total -- PLUS a joint nonlinear
# constraint (chi1x^2+chi1y^2+chi1z^2 <= 1, and the chi2 analogue) that is NOT a per-coordinate
# box and has no `NUTSCore.NUTSBounds` reflection (`BatchedNUTS._sampled_space_bounds`'s own
# comment calls this a "known, accepted gap"): a dedicated testset below checks the gradient near
# THAT boundary specifically, since boundary reflection leaves it uncovered.
# ---------------------------------------------------------------------------------------------

"""
    _check_near_bound_gradients(logp_and_grad!, base, cases, nrows; h=1e-6)

Shared helper (factored out of the IMRPhenomD testset above, which this extension deliberately
does not duplicate): for each `(label, row, value)` in `cases`, sets `base[row] = value`, checks
`logp_and_grad!`'s forward+gradient output is finite, and compares its `row`-th gradient entry
against a central finite difference in that same row (step `h`, chosen -- as in the IMRPhenomD
testset above -- much smaller than `cases`' own margin from the true bound so the FD stencil never
crosses into the `-Inf` prior region).
"""
function _check_near_bound_gradients(logp_and_grad!, base::Vector{Float64}, cases, nrows::Int; h::Float64=1e-6)
    for (label, row, value) in cases
        positions = reshape(copy(base), nrows, 1)
        positions[row, 1] = value

        grad_out = zeros(Float64, nrows, 1)
        logp_out = zeros(Float64, 1)
        logp_and_grad!(grad_out, logp_out, positions, trues(1))
        @test all(isfinite, grad_out)
        @test isfinite(logp_out[1])

        function fwd(x)
            p = copy(positions)
            p[row, 1] = x
            lp = zeros(Float64, 1)
            g = zeros(Float64, nrows, 1)
            logp_and_grad!(g, lp, p, trues(1))
            return lp[1]
        end
        fd = (fwd(value + h) - fwd(value - h)) / (2h)

        @test isapprox(grad_out[row, 1], fd; rtol=1e-3, atol=1e-5)
    end
end

const _EPS_MARGIN = 1e-3
const _BOX11_CASES = [
    ("mc lo", 1, 10.0 + _EPS_MARGIN), ("mc hi", 1, 80.0 - _EPS_MARGIN),
    ("q lo", 2, 0.125 + _EPS_MARGIN), ("q hi", 2, 1.0 - _EPS_MARGIN),
    ("chi1 lo", 3, -1.0 + _EPS_MARGIN), ("chi1 hi", 3, 1.0 - _EPS_MARGIN),
    ("chi2 lo", 4, -1.0 + _EPS_MARGIN), ("chi2 hi", 4, 1.0 - _EPS_MARGIN),
    ("dL lo", 5, 0.05 + _EPS_MARGIN), ("dL hi", 5, 4.0 - _EPS_MARGIN),
    ("t_c lo", 6, -0.1 + _EPS_MARGIN), ("t_c hi", 6, 0.1 - _EPS_MARGIN),
]
const _BOX15_EXTRA_CASES = [
    ("chi1x lo", 12, -1.0 + _EPS_MARGIN), ("chi1x hi", 12, 1.0 - _EPS_MARGIN),
    ("chi1y lo", 13, -1.0 + _EPS_MARGIN), ("chi1y hi", 13, 1.0 - _EPS_MARGIN),
    ("chi2x lo", 14, -1.0 + _EPS_MARGIN), ("chi2x hi", 14, 1.0 - _EPS_MARGIN),
    ("chi2y lo", 15, -1.0 + _EPS_MARGIN), ("chi2y hi", 15, 1.0 - _EPS_MARGIN),
]
# Same 11-row baseline the IMRPhenomD testset above uses -- (mc, q, chi1, chi2, dL, t_c, phase, ra,
# sin_dec, cos_theta_jn, psi), comfortably interior everywhere. `chi1 = chi2 = 0` matters for the
# 15-row extension below: it's what keeps the box-edge chi1x/chi1y/chi2x/chi2y sweeps (chi1=chi2=0,
# the OTHER in-plane component = 0) from also crossing the joint spin-norm constraint -- that
# constraint gets its own dedicated testset instead.
const _BASE11 = [30.0, 0.5, 0.0, 0.0, 0.41, 0.0, 0.3, 1.2, -0.5, 0.2, 0.8]
# NOT all-zero, unlike an earlier version of this fixture.
# `chi1x=chi1y=chi2x=chi2y=0` (both spins purely aligned, zero TOTAL in-plane spin) sits exactly on
# a genuine Enzyme-gradient singularity in `GravitationalWaveforms.jl`'s precession setup, found and root-
# caused empirically while running this exact testset: `IMRPhenomXPrecession.jl`'s
# `_xp_precession_setup` computes `J0x = m1_2*chi1x + m2_2*chi2x`, `J0y = m1_2*chi1y + m2_2*chi2y`
# (lines 345-346) and `phiJ = atan(J0y, J0x)` (line 352). The FORWARD value `atan(0,0) == 0.0` is
# perfectly well-defined, but `d(atan2)/dy = x/(x^2+y^2)` and `d(atan2)/dx = -y/(x^2+y^2)` are
# LITERALLY `0/0` at the origin -- a genuine derivative singularity (not a forward-pass bug, and
# not an Enzyme bug either: this is exactly like `d(sqrt(x))/dx` at `x=0`, a real feature of the
# underlying math, not a defect). Confirmed directly (not assumed): with the all-zero version of
# this fixture, all 12 of `_BOX11_CASES` (mc/q/chi1/chi2/dL/t_c) produced a NaN gradient for
# `IMRPhenomXP`/`IMRPhenomXPHM` (`chi1`/`chi2`'s own tested row plus every OTHER row touching
# `m1,m2,chi1{x,y,z},chi2{x,y,z}` -- i.e. rows 1,2,3,4,12,13,14,15 -- all came back NaN
# simultaneously); substituting a small nonzero value for the OTHER in-plane spin rows (as done
# here) made the NaN disappear in every case, confirming this exact mechanism (not merely a
# plausible story). Using `1e-3` on ALL FOUR in-plane rows (rather than a single axis) both (a)
# reproduces the exact substitution already empirically validated, and (b) avoids relying on any
# one single component being the "safe" one. This is a genuine, LIVE reachability concern, not just
# a test-fixture artifact: `BatchedNUTS._gw150914_initial_positions` initializes EVERY walker at
# exactly this singular point (`chi1x=chi1y=chi2x=chi2y=0`) for `IMRPhenomXP`/`IMRPhenomXPHM`.
const _BASE15 = vcat(_BASE11, [1e-3, 1e-3, 1e-3, 1e-3])

function _make_gw_fixture(model)
    ntime = 16
    freqs = collect(20.0:1.0:(20.0 + ntime - 1))
    strain = zeros(ComplexF64, ntime)
    psd = ones(Float64, ntime)
    data = LikelihoodData(freqs, strain, psd)
    cfg = PEConfig{Float64}(ntime, 1, 0.01, 1)
    return gw_logp_and_grad(data, cfg, freqs; model=model)
end

@testset "Enzyme gradient vs finite differences near box-bounded sampled-space prior edges (IMRPhenomXAS)" begin
    _check_near_bound_gradients(_make_gw_fixture(IMRPhenomXAS()), _BASE11, _BOX11_CASES, 11)
end

@testset "Enzyme gradient vs finite differences near box-bounded sampled-space prior edges (IMRPhenomXHM)" begin
    _check_near_bound_gradients(_make_gw_fixture(IMRPhenomXHM()), _BASE11, _BOX11_CASES, 11)
end

@testset "Enzyme gradient vs finite differences near box-bounded sampled-space prior edges (IMRPhenomXP)" begin
    _check_near_bound_gradients(_make_gw_fixture(IMRPhenomXP()), _BASE15, vcat(_BOX11_CASES, _BOX15_EXTRA_CASES), 15)
end

@testset "Enzyme gradient vs finite differences near box-bounded sampled-space prior edges (IMRPhenomXPHM)" begin
    _check_near_bound_gradients(_make_gw_fixture(IMRPhenomXPHM()), _BASE15, vcat(_BOX11_CASES, _BOX15_EXTRA_CASES), 15)
end

"""
    _check_joint_spin_norm_boundary(model)

The one gap `BatchedNUTS._sampled_space_bounds`'s own docstring names explicitly: `IMRPhenomXP`'s
(and `IMRPhenomXPHM`'s) `chi1x^2+chi1y^2+chi1z^2 <= 1` constraint (and the `chi2` analogue) is not
a per-coordinate box, has no `NUTSBounds` reflection, and stays enforced only via `_log_prior`'s
`-Inf` fallback. Two checks:

1. Just INSIDE the constraint surface (`chi1_norm2 = 1 - eps_margin`), Enzyme's gradient in the 3
   rows entering that constraint (`chi1` z-component, `chi1x`, `chi1y`) must still agree with
   central finite differences -- same rigor as every other near-bound case above, just for a
   nonlinear rather than a per-coordinate boundary.
2. Just OUTSIDE it (`chi1_norm2 = 1 + eps_margin`), `logp_out` must be exactly `-Inf` (confirms
   `_log_prior`'s 15-argument method's fallback actually fires here, not just reasoned about), and
   `Enzyme.autodiff` must complete without throwing/crashing -- `NUTSCore.nuts_draw!` only needs
   `isfinite(Hw)` to discard this walker's subtree (see that module's own comment on
   `subtree_valid`), not a numerically meaningful gradient at a point the prior already rejects.
"""
function _check_joint_spin_norm_boundary(model)
    logp_and_grad! = _make_gw_fixture(model)
    h = 1e-6

    # chi1 (z, row 3) = 0.3, chi1y (row 13) = 0.2 fixed; chi1x (row 12) chosen so
    # chi1x^2 + chi1y^2 + chi1^2 == 1 - eps_margin -- just inside the constraint.
    chi1z, chi1y_fixed = 0.3, 0.2
    chi1x_near = sqrt(1.0 - _EPS_MARGIN - chi1z^2 - chi1y_fixed^2)
    base = copy(_BASE15)
    base[3] = chi1z
    base[12] = chi1x_near
    base[13] = chi1y_fixed
    positions = reshape(base, 15, 1)

    grad_out = zeros(Float64, 15, 1)
    logp_out = zeros(Float64, 1)
    logp_and_grad!(grad_out, logp_out, positions, trues(1))
    @test all(isfinite, grad_out)
    @test isfinite(logp_out[1])

    for row in (3, 12, 13)
        function fwd(x)
            p = copy(positions)
            p[row, 1] = x
            lp = zeros(Float64, 1)
            g = zeros(Float64, 15, 1)
            logp_and_grad!(g, lp, p, trues(1))
            return lp[1]
        end
        fd = (fwd(positions[row, 1] + h) - fwd(positions[row, 1] - h)) / (2h)
        @test isapprox(grad_out[row, 1], fd; rtol=1e-3, atol=1e-5)
    end

    # Negative control: chi1_norm2 == 1 + eps_margin, just OUTSIDE the constraint.
    positions_violate = copy(positions)
    positions_violate[12, 1] = sqrt(1.0 + _EPS_MARGIN - chi1z^2 - chi1y_fixed^2)
    grad_violate = zeros(Float64, 15, 1)
    logp_violate = zeros(Float64, 1)
    logp_and_grad!(grad_violate, logp_violate, positions_violate, trues(1))
    @test logp_violate[1] == -Inf
    return grad_violate[[3, 12, 13], 1]
end

@testset "Enzyme gradient vs finite differences near the joint spin-norm constraint (IMRPhenomXP)" begin
    grad_violate = _check_joint_spin_norm_boundary(IMRPhenomXP())
    println("IMRPhenomXP: gradient at a joint-spin-norm-VIOLATING point (chi1,chi1x,chi1y rows; ",
            "informational -- _log_prior's -Inf fallback only needs this to not crash, not to be ",
            "numerically meaningful): ", grad_violate)
end

@testset "Enzyme gradient vs finite differences near the joint spin-norm constraint (IMRPhenomXPHM)" begin
    grad_violate = _check_joint_spin_norm_boundary(IMRPhenomXPHM())
    println("IMRPhenomXPHM: gradient at a joint-spin-norm-VIOLATING point (chi1,chi1x,chi1y rows; ",
            "informational -- _log_prior's -Inf fallback only needs this to not crash, not to be ",
            "numerically meaningful): ", grad_violate)
end
