module Priors

export isotropic_spins!, isotropic_spins_pullback!, prior_support_vec
export GWPrior, prior_logpdf!, prior_logpdf_and_grad!, prior_bounds, sample_prior, prior_log_normalizer,
       GaussianPriorMixture, KDEPriorMixture, local_kde_covariances, sample_reference, reference_logpdf_and_grad!

using Random
using LinearAlgebra: cholesky, Symmetric, LowerTriangular

# ------------------------------------------------------------------------------------------------
# Configurable priors over the 11-row aligned-spin SAMPLED space
# `(mc, q, chi1, chi2, dL, t_c, phase, ra, sin_dec, cos_theta_jn, psi)` (see `BatchedNUTS.jl`'s
# row-convention comment above `_reparam_to_physical!`).
#
# The default package prior (`BatchedNUTS._log_prior`) is a flat box in exactly those coordinates.
# `GWPrior` keeps the same box (with every bound overridable) and adds the two standard
# astrophysical shapes production PE codes use:
#   - `distance = :volume`: p(dL) ∝ dL^2 (uniform in Euclidean volume) instead of uniform in dL.
#   - `masses = :components`: uniform in DETECTOR-FRAME component masses (m1, m2) instead of uniform
#     in (mc, q). The Jacobian |d(m1,m2)/d(mc,q)| = m1^2/mc = mc (1+q)^(2/5) q^(-6/5) (bilby's
#     `chirp_mass`/`mass_ratio` prior uses the same factor).
# The sky/orientation rows stay isotropic (flat in sin_dec / cos_theta_jn, see `BatchedNUTS.jl`).
# Precessing models (`nrows = 15`) add the in-plane components (chi1x, chi1y, chi2x, chi2y): each spin
# VECTOR is uniform in the ball |chi_i| <= a_i, a_i = max(|lo|, |hi|) of the chi_i (z) bounds, cut to the z box --
# flat in the Cartesian components (NOT the LVK uniform-magnitude isotropic prior, whose density ~ 1/|chi|^2).
#
# Both extra factors have closed-form gradients, so `prior_logpdf_and_grad!` is added to the
# likelihood's gradient OUTSIDE any Enzyme call -- no kernel or AD path changes.
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# LVK spin prior (matched-gw150914): spin magnitude uniform on [0, amax], direction isotropic, as a smooth
# map of the sampled unit ball. The sampler sees y uniform in |y| <= 1 (the 15-row ball prior with spin bounds +-1);
# the likelihood gets chi = amax |y|^2 y: P(|y| < R) = R^3, so |chi| = amax |y|^3 is uniform, and directions stay
# isotropic. No 1/|chi|^2 density spike at the origin (the map's derivative vanishes there instead). Spin rows:
# spin 1 (x, y, z) = rows (12, 13, 3), spin 2 = (14, 15, 4).
# ------------------------------------------------------------------------------------------------
const SPIN_ROWS = ((12, 13, 3), (14, 15, 4))

"""
    prior_support_vec(prior, positions) -> Vector (0 inside the prior's support, -Inf outside)

The support of a configured prior per walker column -- for device likelihood closures that otherwise impose the
package's default box (`BatchedNUTS._log_prior`: d_L <= 4 Gpc, q >= 0.125, ...), which a wider configured prior (e.g.
the LVK d_L in [0.01, 10] Gpc, q >= 0.05) must replace.
"""
prior_support_vec(prior, positions::AbstractMatrix{T}) where {T} =
    [ifelse(isfinite(_col_logpdf(prior, positions, w)), zero(T), T(-Inf)) for w in axes(positions, 2)]

# Aligned-spin models (11 rows): the LVK prior's z-projection, p(chi_z) = log(a / |chi_z|) / (2 a) on
# [-a, a] (bilby `AlignedSpin` with a uniform magnitude and uniform cos tilt), as the inverse-CDF map of y uniform on
# [-1, 1]: chi = a sign(y) x, x (1 - log x) = |y| (P(|chi| < a x) = x (1 - log x)). In s = -log x:
# e^{-s} (1 + s) = |y|, dchi / dy = a / s -- 0 at the origin (no density spike to sample), a sqrt singularity at
# |y| = 1 where the density vanishes. Rows 3, 4.
# `amax`: one cap for both bodies or a per-body (a1, a2) (e.g. a neutron star's a2 <= 0.05).
@inline _amax(amax::Real, i) = amax
@inline _amax(amax::Tuple, i) = amax[i]

# s = -log x with e^{-s} (1 + s) = u, u in [0, 1] (Newton on log(1 + s) - s - log u, monotone in s)
function _aligned_s(u::T) where {T<:Real}
    u >= 1 && return zero(T)
    u <= 0 && return T(Inf)
    lu = log(u)
    s = u > T(0.5) ? sqrt(2 * (1 - u)) : -lu + log1p(-lu)
    for _ in 1:30
        h = log1p(s) - s - lu
        ds = h / (-s / (1 + s))
        s = max(s - ds, s / 2)
        abs(ds) <= 4 * eps(T) * max(s, one(T)) && break
    end
    return s
end

"""
    isotropic_spins!(out, pos, amax) -> out

`out = pos` with the sampled spin rows mapped to the LVK isotropic prior's spins: 15 rows, each spin vector
y -> amax |y|^2 y (see the comment above `SPIN_ROWS`); 11 rows (aligned spins), rows 3 / 4 y -> the z-projection
(see `_aligned_s`). `amax` is a number or a per-body tuple.
"""
function isotropic_spins!(out::AbstractMatrix, pos::AbstractMatrix, amax)
    out .= pos
    if size(pos, 1) >= 15
        @inbounds for w in axes(pos, 2), (b, r) in enumerate(SPIN_ROWS)
            y2 = pos[r[1], w]^2 + pos[r[2], w]^2 + pos[r[3], w]^2
            for i in r
                out[i, w] = _amax(amax, b) * y2 * pos[i, w]
            end
        end
    else
        @inbounds for w in axes(pos, 2), (b, r) in enumerate((3, 4))
            y = pos[r, w]
            out[r, w] = _amax(amax, b) * sign(y) * exp(-_aligned_s(abs(y)))
        end
    end
    return out
end

"""
    isotropic_spins_pullback!(g, pos, amax) -> g

In place: a gradient with respect to the mapped spins chi -> the gradient with respect to the sampled y. 15 rows:
the symmetric Jacobian d chi_i / d y_j = amax (|y|^2 delta_ij + 2 y_i y_j); 11 rows: d chi / d y = amax / s.
"""
function isotropic_spins_pullback!(g::AbstractMatrix, pos::AbstractMatrix, amax)
    if size(pos, 1) >= 15
        @inbounds for w in axes(pos, 2), (b, r) in enumerate(SPIN_ROWS)
            a = _amax(amax, b)
            y1, y2, y3 = pos[r[1], w], pos[r[2], w], pos[r[3], w]
            g1, g2, g3 = g[r[1], w], g[r[2], w], g[r[3], w]
            yy = y1 * y1 + y2 * y2 + y3 * y3
            yg = y1 * g1 + y2 * g2 + y3 * g3
            g[r[1], w] = a * (yy * g1 + 2 * y1 * yg)
            g[r[2], w] = a * (yy * g2 + 2 * y2 * yg)
            g[r[3], w] = a * (yy * g3 + 2 * y3 * yg)
        end
    else
        @inbounds for w in axes(pos, 2), (b, r) in enumerate((3, 4))
            s = _aligned_s(abs(pos[r, w]))
            # (finite at |y| = 1, where the map's slope diverges)
            g[r, w] *= _amax(amax, b) / max(s, sqrt(eps(typeof(s))))
        end
    end
    return g
end

const ROW_NAMES = ("mc", "q", "chi1", "chi2", "dL", "t_c", "phase", "ra", "sin_dec", "cos_theta_jn", "psi",
                   "chi1x", "chi1y", "chi2x", "chi2y")
const PERIODIC_ROWS = (7, 8, 11)

"""
    GWPrior{T}(; bounds=Dict(), distance=:uniform, masses=:uniform_mc_q, nrows=11)

Prior over the 11-row (or, `nrows = 15`, precessing: + in-plane spins, see the header) sampled space. `bounds` maps row names (`"mc"`, `"q"`, `"chi1"`, `"chi2"`,
`"dL"`, `"t_c"`) to `(lo, hi)` overrides of the package defaults; the angular rows keep their natural
domains. `distance` is `:uniform` or `:volume` (∝ dL^2); `masses` is `:uniform_mc_q` or `:components`
(uniform in detector-frame m1, m2). `m1`, `m2`: `(lo, hi)` detector-frame component-mass constraints (support
cuts, as bilby's `Constraint` priors: the prior is renormalized). `GWPrior{T}()` reproduces `BatchedNUTS._log_prior`
exactly.
"""
struct GWPrior{T}
    lo::Vector{T}             # per row (length nrows)
    hi::Vector{T}
    distance::Symbol
    masses::Symbol
    spin_max::NTuple{2,T}     # nrows = 15: |chi_i| <= spin_max[i]
    m1::NTuple{2,T}           # detector-frame component-mass constraints (bilby `Constraint` on mass_1 / mass_2);
    m2::NTuple{2,T}           # (0, Inf) = none
end
nrows(prior::GWPrior) = length(prior.lo)

const _DEFAULT_BOUNDS = Dict(
    "mc" => (10.0, 80.0), "q" => (0.125, 1.0), "chi1" => (-1.0, 1.0), "chi2" => (-1.0, 1.0),
    "dL" => (0.05, 4.0), "t_c" => (-0.1, 0.1), "phase" => (0.0, 2pi), "ra" => (0.0, 2pi),
    "sin_dec" => (-1.0, 1.0), "cos_theta_jn" => (-1.0, 1.0), "psi" => (0.0, 1.0 * pi))

const _OVERRIDABLE = ("mc", "q", "chi1", "chi2", "dL", "t_c")

# Calibration rows (matched-gw150914): `ncal` rows after row 15, each a standard normal z (the spline node
# value is mu + sigma z, mu / sigma from the analysis' calibration prior), boxed at +-_CAL_BOX (mass outside ~1e-15).
const _CAL_BOX = 8.0

function GWPrior{T}(; bounds=Dict{String,Any}(), distance::Symbol=:uniform, masses::Symbol=:uniform_mc_q,
                    nrows::Int=11, ncal::Int=0, m1=(0, Inf), m2=(0, Inf)) where {T}
    nrows in (11, 15) || throw(ArgumentError("nrows must be 11 or 15, got $nrows"))
    (ncal == 0 || nrows == 15) || throw(ArgumentError("calibration rows (ncal) need the 15-row precessing space"))
    distance in (:uniform, :volume) || throw(ArgumentError("distance must be :uniform or :volume, got :$distance"))
    masses in (:uniform_mc_q, :components) ||
        throw(ArgumentError("masses must be :uniform_mc_q or :components, got :$masses"))
    lo = Vector{T}(undef, nrows)
    hi = Vector{T}(undef, nrows)
    for i in 1:11
        lo[i], hi[i] = _DEFAULT_BOUNDS[ROW_NAMES[i]]
    end
    for (k, v) in pairs(bounds)
        name = String(k)
        name in _OVERRIDABLE || throw(ArgumentError(
            "prior bounds can be overridden for $(join(_OVERRIDABLE, ", ")); got \"$name\""))
        i = findfirst(==(name), ROW_NAMES)
        l, h = T(v[1]), T(v[2])
        l < h || throw(ArgumentError("prior bounds for $name must satisfy lo < hi, got ($l, $h)"))
        lo[i], hi[i] = l, h
    end
    lo[2] > 0 || throw(ArgumentError("q lower bound must be > 0"))
    hi[2] <= 1 || throw(ArgumentError("q upper bound must be <= 1 (q = m2/m1)"))
    lo[5] > 0 || throw(ArgumentError("dL lower bound must be > 0"))
    (lo[3] >= -1 && hi[3] <= 1 && lo[4] >= -1 && hi[4] <= 1) ||
        throw(ArgumentError("aligned-spin bounds must lie within [-1, 1]"))
    a = (max(abs(lo[3]), abs(hi[3])), max(abs(lo[4]), abs(hi[4])))
    if nrows == 15
        lo[12] = lo[13] = -a[1]; hi[12] = hi[13] = a[1]
        lo[14] = lo[15] = -a[2]; hi[14] = hi[15] = a[2]
    end
    (0 <= m1[1] < m1[2] && 0 <= m2[1] < m2[2]) || throw(ArgumentError("component-mass constraints need 0 <= lo < hi"))
    append!(lo, fill(T(-_CAL_BOX), ncal)); append!(hi, fill(T(_CAL_BOX), ncal))
    pr = GWPrior{T}(lo, hi, distance, masses, a, (T(m1[1]), T(m1[2])), (T(m2[1]), T(m2[2])))
    _has_mass_cut(pr) && _log_mass_normalizer(pr) == -Inf &&
        throw(ArgumentError("the component-mass constraints exclude the whole (mc, q) box"))
    return pr
end

"""
    prior_bounds(prior) -> (lo, hi, periodic)

Per-row box bounds and periodicity flags, for building `NUTSCore.NUTSBounds` (reflect/wrap during
leapfrog) consistent with the prior's own support.
"""
function prior_bounds(prior::GWPrior)
    periodic = falses(nrows(prior))
    for r in PERIODIC_ROWS
        periodic[r] = true
    end
    return copy(prior.lo), copy(prior.hi), periodic
end

# Component masses from (mc, q): m1 = mc (1 + q)^(1/5) q^(-3/5), m2 = q m1.
@inline _m1_factor(q) = (1 + q)^(1 / 5) * q^(-3 / 5)
_has_mass_cut(prior::GWPrior) = prior.m1 != (0, Inf) || prior.m2 != (0, Inf)
@inline function _masses_ok(prior::GWPrior, mc, q)
    m1 = mc * _m1_factor(q); m2 = q * m1
    return prior.m1[1] <= m1 <= prior.m1[2] && prior.m2[1] <= m2 <= prior.m2[2]
end

@inline function _col_logpdf(prior::GWPrior{T}, positions::AbstractMatrix{T}, w::Int) where {T}
    @inbounds for r in 1:nrows(prior)
        x = positions[r, w]
        (prior.lo[r] <= x <= prior.hi[r]) || return T(-Inf)
    end
    @inbounds _masses_ok(prior, positions[1, w], positions[2, w]) || return T(-Inf)
    @inbounds if nrows(prior) >= 15
        (positions[3, w]^2 + positions[12, w]^2 + positions[13, w]^2 <= prior.spin_max[1]^2 &&
         positions[4, w]^2 + positions[14, w]^2 + positions[15, w]^2 <= prior.spin_max[2]^2) || return T(-Inf)
    end
    lp = zero(T)
    @inbounds if prior.distance === :volume
        lp += 2 * log(positions[5, w])
    end
    @inbounds if prior.masses === :components
        mc = positions[1, w]; q = positions[2, w]
        lp += log(mc) + T(0.4) * log1p(q) - T(1.2) * log(q)
    end
    @inbounds for r in 16:nrows(prior)
        lp -= positions[r, w]^2 / 2
    end
    return lp
end

"""
    prior_logpdf!(lp, prior, positions)

Unnormalized log prior density per walker column of the sampled `positions` (host arrays).
"""
function prior_logpdf!(lp::AbstractVector{T}, prior::GWPrior{T}, positions::AbstractMatrix{T}) where {T}
    size(positions, 1) == nrows(prior) ||
        throw(ArgumentError("this GWPrior expects $(nrows(prior)) sampled rows, got $(size(positions, 1))"))
    for w in axes(positions, 2)
        lp[w] = _col_logpdf(prior, positions, w)
    end
    return lp
end

"""
    prior_logpdf_and_grad!(lp, grad, prior, positions)

`lp[w] = log p(positions[:, w])`, and ADDS `d log p / d positions` into `grad` (in-bounds walkers only;
out-of-bounds walkers get `lp = -Inf` and no gradient contribution).
"""
function prior_logpdf_and_grad!(lp::AbstractVector{T}, grad::AbstractMatrix{T}, prior::GWPrior{T},
                                positions::AbstractMatrix{T}) where {T}
    prior_logpdf!(lp, prior, positions)
    for w in axes(positions, 2)
        isfinite(lp[w]) || continue
        @inbounds if prior.distance === :volume
            grad[5, w] += 2 / positions[5, w]
        end
        @inbounds if prior.masses === :components
            mc = positions[1, w]; q = positions[2, w]
            grad[1, w] += 1 / mc
            grad[2, w] += T(0.4) / (1 + q) - T(1.2) / q
        end
        @inbounds for r in 16:nrows(prior)
            grad[r, w] -= positions[r, w]
        end
    end
    return lp
end

"""
    sample_prior(rng, prior, n) -> Matrix (nrows, n)

Exact draws from `prior` (rejection sampling for the non-uniform mass/distance factors and the spin balls).
"""
function sample_prior(rng::AbstractRNG, prior::GWPrior{T}, n::Int) where {T}
    nr = nrows(prior)
    out = Matrix{T}(undef, nr, n)
    lo, hi = prior.lo, prior.hi
    for w in 1:n
        for r in 1:nr
            out[r, w] = lo[r] + (hi[r] - lo[r]) * rand(rng, T)
        end
        for r in 16:nr                               # calibration rows: standard normal inside the box
            z = randn(rng, T)
            while abs(z) > hi[r]; z = randn(rng, T); end
            out[r, w] = z
        end
        if nr >= 15
            for (rz, rx, ry, a) in ((3, 12, 13, prior.spin_max[1]), (4, 14, 15, prior.spin_max[2]))
                while out[rz, w]^2 + out[rx, w]^2 + out[ry, w]^2 > a^2
                    for r in (rz, rx, ry)
                        out[r, w] = lo[r] + (hi[r] - lo[r]) * rand(rng, T)
                    end
                end
            end
        end
        if prior.distance === :volume
            # inverse CDF of p ∝ dL^2 on [lo, hi]
            u = rand(rng, T)
            out[5, w] = cbrt(lo[5]^3 + u * (hi[5]^3 - lo[5]^3))
        end
        if prior.masses === :components
            # rejection against the (mc, q) box: density ∝ mc (1+q)^0.4 q^-1.2, maximized at
            # mc = hi, q = lo (both factors are monotone on the box); and the component-mass cut
            logmax = log(hi[1]) + T(0.4) * log1p(lo[2]) - T(1.2) * log(lo[2])
            while true
                mc = lo[1] + (hi[1] - lo[1]) * rand(rng, T)
                q = lo[2] + (hi[2] - lo[2]) * rand(rng, T)
                if log(rand(rng, T)) <= log(mc) + T(0.4) * log1p(q) - T(1.2) * log(q) - logmax && _masses_ok(prior, mc, q)
                    out[1, w] = mc; out[2, w] = q
                    break
                end
            end
        else
            while !_masses_ok(prior, out[1, w], out[2, w])
                out[1, w] = lo[1] + (hi[1] - lo[1]) * rand(rng, T)
                out[2, w] = lo[2] + (hi[2] - lo[2]) * rand(rng, T)
            end
        end
    end
    return out
end


"""
    prior_log_normalizer(prior) -> log C

`log ∫ exp(prior_logpdf)` over the box, so `prior_logpdf - log C` is the NORMALIZED prior density
(needed when the prior is compared with another normalized density, e.g. an SMC reference).
"""
prior_log_normalizer(prior::GWPrior) = _log_marginal_normalizer(prior, 1:nrows(prior))

# log of the (mc, q) factor's integral over the box and the component-mass cut: ∫ mc (1+q)^0.4 q^-1.2 (components)
# or ∫ 1 (uniform in mc, q). At fixed q the cut is an mc interval (m1, m2 are linear in mc), integrated exactly;
# Simpson in log q (smooth without a cut; with one, kinks where the interval's ends switch -- hence more nodes).
function _log_mass_normalizer(prior::GWPrior{T}) where {T}
    lo, hi = prior.lo, prior.hi
    comp = prior.masses === :components
    cut = _has_mass_cut(prior)
    n = cut ? 20_000 : 2000
    a, b = log(lo[2]), log(hi[2])
    h = (b - a) / n
    function f(u)
        q = exp(u); f1 = _m1_factor(q)
        l = max(lo[1], prior.m1[1] / f1, prior.m2[1] / (q * f1))
        r = min(hi[1], prior.m1[2] / f1, prior.m2[2] / (q * f1))
        r > l || return zero(T)
        return comp ? (r^2 - l^2) / 2 * (1 + q)^T(0.4) * q^T(-1.2) * q : (r - l) * q
    end
    sq = f(a) + f(b)
    for k in 1:n-1
        sq += (isodd(k) ? 4 : 2) * f(a + k * h)
    end
    return log(sq * h / 3)
end

# The spin-ball row groups (chi_iz, chi_ix, chi_iy, i) of a 15-row prior: uniform in |chi_i| <= spin_max[i].
const _SPIN_BALLS = ((3, 12, 13, 1), (4, 14, 15, 2))

# Log normalizer of the prior marginal on `rows`. The prior factorizes as p(mc, q) times one
# independent factor per other row, except a 15-row prior's spin balls, one factor per ball group (`rows` must hold
# both or neither of mc/q when masses = :components, and all or none of each ball group).
function _log_marginal_normalizer(prior::GWPrior{T}, rows) where {T}
    lo, hi = prior.lo, prior.hi
    c = zero(T)
    if nrows(prior) >= 15
        rest = collect(Int, rows)
        for (rz, rx, ry, i) in _SPIN_BALLS
            m = count(in(rest), (rz, rx, ry))
            m == 0 && continue
            m == 3 || throw(ArgumentError("a 15-row GWPrior marginal needs all or none of the spin rows ($rz, $rx, $ry)"))
            a = prior.spin_max[i]
            vol(z) = pi * (a^2 * z - z^3 / 3)          # ball volume below height z (|z| <= a)
            c += log(vol(hi[rz]) - vol(lo[rz]))
            setdiff!(rest, (rz, rx, ry))
        end
        rows = rest
    end
    mjoint = prior.masses === :components || _has_mass_cut(prior)
    (mjoint && (1 in rows) != (2 in rows)) &&
        throw(ArgumentError("a GWPrior marginal needs both or neither of mc / q (masses = :components or a mass cut)"))
    for r in rows
        (r == 1 || r == 2) && mjoint && continue
        r > 15 && (c += log(sqrt(2 * T(pi))); continue)    # calibration rows: unit normal (box mass ~1)
        c += (r == 5 && prior.distance === :volume) ? log((hi[5]^3 - lo[5]^3) / 3) : log(hi[r] - lo[r])
    end
    (mjoint && 1 in rows) && (c += _log_mass_normalizer(prior))
    return c
end

"""
    GaussianPriorMixture(prior, rows, mu, cov; alpha=0.1, rng, nmc=200_000)

Normalized SMC reference density on the 11-row sampled space:
`q0 = alpha * prior + (1 - alpha) * N_box(x[rows]; mu, cov) * prior(x[other rows])`,
the Gaussian truncated to the prior box (its in-box mass estimated once by Monte Carlo). The prior
component keeps q0's support equal to the prior's, so annealing from q0 stays evidence-exact; the
Gaussian concentrates the particles where a cheaper fit (e.g. another model's posterior) put them.
`rows` should be the well-localized, non-periodic rows (intrinsic + t_c); the sky/orientation and
marginalized rows keep the prior. Sample with [`sample_reference`](@ref), evaluate with
[`reference_logpdf_and_grad!`](@ref).
"""
struct GaussianPriorMixture{T}
    prior::GWPrior{T}
    rows::Vector{Int}
    mu::Vector{T}
    L::Matrix{T}              # cov = L L'
    alpha::T
    log_prior_norm::T         # log C of the full prior
    log_rows_norm::T          # log C of the prior marginal on `rows`
    log_gauss_norm::T         # log of the Gaussian normalizer, including the in-box mass
end

function GaussianPriorMixture(prior::GWPrior{T}, rows::AbstractVector{<:Integer}, mu::AbstractVector,
                              cov::AbstractMatrix; alpha=0.1, rng::AbstractRNG=Random.Xoshiro(1),
                              nmc::Int=200_000) where {T}
    rows = collect(Int, rows)
    ((prior.masses !== :components && !_has_mass_cut(prior)) || (1 in rows) == (2 in rows)) ||
        throw(ArgumentError("rows must contain both mc and q, or neither, with masses = :components or a mass cut"))
    any(r -> r in PERIODIC_ROWS, rows) && throw(ArgumentError("GaussianPriorMixture rows must be non-periodic"))
    0 < alpha < 1 || throw(ArgumentError("alpha must lie in (0, 1)"))
    k = length(rows)
    (length(mu) == k && size(cov) == (k, k)) || throw(ArgumentError("mu/cov must match length(rows) = $k"))
    L = Matrix{T}(cholesky(Symmetric(Matrix{T}(cov))).L)
    inbox = 0
    z = Vector{T}(undef, k)
    balls = _ball_index(prior, rows)
    nop = zeros(T, k)
    for _ in 1:nmc
        randn!(rng, z)
        x = T.(mu) .+ L * z
        inbox += _rows_in_support(prior, rows, x, nop, balls)
    end
    inbox > 0 || throw(ArgumentError("the Gaussian reference has no mass inside the prior box"))
    lg = T(k / 2) * log(2 * T(pi)) + sum(log(L[i, i]) for i in 1:k) + log(T(inbox) / nmc)
    return GaussianPriorMixture{T}(prior, rows, Vector{T}(mu), L, T(alpha), prior_log_normalizer(prior),
                                   _log_marginal_normalizer(prior, rows), lg)
end

"""
    sample_reference(rng, ref::GaussianPriorMixture, n) -> Matrix (11, n)

Exact draws from `ref`: prior draws, of which a `1 - alpha` fraction have their `rows` replaced by
in-box Gaussian draws (rejection on the box).
"""
function sample_reference(rng::AbstractRNG, ref::GaussianPriorMixture{T}, n::Int) where {T}
    out = sample_prior(rng, ref.prior, n)
    k = length(ref.rows)
    z = Vector{T}(undef, k)
    balls = _ball_index(ref.prior, ref.rows)
    nop = zeros(T, k)
    for w in 1:n
        rand(rng, T) < ref.alpha && continue
        for tries in Iterators.countfrom(1)
            _reject_cap(tries, ref.mu, ref.rows)
            randn!(rng, z)
            x = ref.mu .+ ref.L * z
            if _rows_in_support(ref.prior, ref.rows, x, nop, balls)
                out[ref.rows, w] .= x
                break
            end
        end
    end
    return out
end

# Unnormalized log of the prior factors living on `rows` (only the mass factor is non-constant) and
# its gradient contribution.
@inline function _rows_prior_un(ref::GaussianPriorMixture{T}, positions, w) where {T}
    (ref.prior.masses === :components && 1 in ref.rows) || return (zero(T), zero(T), zero(T))
    mc = positions[1, w]; q = positions[2, w]
    return (log(mc) + T(0.4) * log1p(q) - T(1.2) * log(q), 1 / mc, T(0.4) / (1 + q) - T(1.2) / q)
end

"""
    reference_logpdf_and_grad!(lq, grad, ref, positions)

Normalized `log q0` per column, and its gradient (OVERWRITES `grad`). Outside the prior box `lq = -Inf`.
"""
function reference_logpdf_and_grad!(lq::AbstractVector{T}, grad::AbstractMatrix{T}, ref::GaussianPriorMixture{T},
                                    positions::AbstractMatrix{T}) where {T}
    lp = Vector{T}(undef, size(positions, 2))
    fill!(grad, zero(T))
    prior_logpdf_and_grad!(lp, grad, ref.prior, positions)       # grad now holds d log prior
    Lt = LowerTriangular(ref.L)
    for w in axes(positions, 2)
        if !isfinite(lp[w])
            lq[w] = T(-Inf)
            continue
        end
        u = Lt \ (positions[ref.rows, w] .- ref.mu)
        lgauss = -sum(abs2, u) / 2 - ref.log_gauss_norm
        lprior = lp[w] - ref.log_prior_norm
        lrows_un, dmc, dq = _rows_prior_un(ref, positions, w)
        # Gaussian component: N(rows) * prior(other rows) = N * prior / prior_rows.
        a = log(ref.alpha) + lprior
        b = log1p(-ref.alpha) + lgauss + lprior - (lrows_un - ref.log_rows_norm)
        m = max(a, b)
        lq[w] = m + log(exp(a - m) + exp(b - m))
        rb = exp(b - lq[w])
        # d log q = d log prior + rb * (d log N - d log prior_rows)
        g = -(transpose(Lt) \ u)
        for (i, r) in enumerate(ref.rows)
            grad[r, w] += rb * g[i]
        end
        if ref.prior.masses === :components && 1 in ref.rows
            grad[1, w] -= rb * dmc
            grad[2, w] -= rb * dq
        end
    end
    return lq
end


"""
    KDEPriorMixture(prior, rows, centers, cov; alpha=0.1, rng, nmc=4000)

`cov` is one shared bandwidth matrix or a `(k, k, K)` array of per-kernel ones (see
[`local_kde_covariances`](@ref)).

Normalized SMC reference built from posterior samples of a cheaper analysis:
`q0 = alpha * prior + (1 - alpha) * (1/K) sum_k N_box(x[rows]; centers[:, k], cov) * prior(x[other rows])`.
A Gaussian kernel density estimate (shared bandwidth matrix `cov`) on `rows`, so unlike a single
Gaussian it follows multimodal / curved structure such as a two-detector sky ring. Periodic rows
(ra, psi) use the nearest-image difference (accurate while the bandwidth is well below the period);
each kernel is truncated to the prior box on the non-periodic rows with its own in-box mass
(Monte Carlo, `nmc` draws per kernel). Evidence-exact as a reference, like `GaussianPriorMixture`.
"""
struct KDEPriorMixture{T}
    prior::GWPrior{T}
    rows::Vector{Int}
    centers::Matrix{T}        # (length(rows), K)
    L::Array{T,3}             # per-kernel bandwidth Cholesky factors: cov_j = L[:,:,j] L[:,:,j]'
    alpha::T
    log_prior_norm::T
    log_rows_norm::T
    log_kernel_norm::Vector{T}    # per kernel: Gaussian normalizer + log in-box mass
    period::Vector{T}             # per row in `rows`: period (0 = not periodic)
end

function KDEPriorMixture(prior::GWPrior{T}, rows::AbstractVector{<:Integer}, centers::AbstractMatrix,
                         cov::AbstractArray; alpha=0.1, rng::AbstractRNG=Random.Xoshiro(1),
                         nmc::Int=4000) where {T}
    rows = collect(Int, rows)
    ((prior.masses !== :components && !_has_mass_cut(prior)) || (1 in rows) == (2 in rows)) ||
        throw(ArgumentError("rows must contain both mc and q, or neither, with masses = :components or a mass cut"))
    0 < alpha < 1 || throw(ArgumentError("alpha must lie in (0, 1)"))
    k, K = size(centers)
    # `cov`: one shared (k, k) bandwidth matrix, or a (k, k, K) array of per-kernel ones (local KDE).
    covs = ndims(cov) == 2 ? repeat(reshape(Matrix{T}(cov), k, k, 1), 1, 1, K) : Array{T,3}(cov)
    size(covs) == (k, k, K) || throw(ArgumentError("cov must be ($k, $k) or ($k, $k, $K)"))
    L = similar(covs)
    for j in 1:K
        L[:, :, j] .= cholesky(Symmetric(covs[:, :, j])).L
    end
    period = T[r in PERIODIC_ROWS ? prior.hi[r] - prior.lo[r] : zero(T) for r in rows]
    for j in 1:K, i in 1:k
        # sd of kernel j along row i = norm of row i of L_j
        period[i] > 0 && sqrt(sum(abs2, view(L, i, :, j))) > period[i] / 6 && throw(ArgumentError(
            "KDE bandwidth on periodic row $(rows[i]) must be below period/6 (kernel $j)"))
    end
    C = Matrix{T}(centers)
    for j in 1:K, i in 1:k
        period[i] > 0 && (C[i, j] = _wrap_to(C[i, j], prior.lo[rows[i]], period[i]))
    end
    lkn = Vector{T}(undef, K)
    z = Vector{T}(undef, k)
    balls = _ball_index(prior, rows)
    for j in 1:K
        base = T(k / 2) * log(2 * T(pi)) + sum(log(L[i, i, j]) for i in 1:k)
        Lj = view(L, :, :, j)
        inbox = 0
        for _ in 1:nmc
            randn!(rng, z)
            x = view(C, :, j) .+ Lj * z
            inbox += _rows_in_support(prior, rows, x, period, balls)
        end
        lkn[j] = base + log(max(inbox, 1) / T(nmc))
    end
    return KDEPriorMixture{T}(prior, rows, C, L, T(alpha), prior_log_normalizer(prior),
                              _log_marginal_normalizer(prior, rows), lkn, period)
end

# Rejection sampling of a reference kernel inside the prior support: a kernel (nearly) outside it, e.g. a pass-1
# posterior whose time row lies beyond this prior's window, would loop forever
_reject_cap(tries, center, rows) = tries > 100_000 && throw(ArgumentError(
    "reference kernel at $(collect(center)) (rows $(collect(rows))) has no mass inside the prior support; " *
    "is the reference outside the prior box (e.g. the time row: smc_reference_time_shift, time_ref_offset, t_c)?"))

@inline _wrap_to(x, lo, p) = lo + mod(x - lo, p)
@inline _nearest_image(d, p) = p > 0 ? d - p * round(d / p) : d

# Positions of each complete spin-ball group of a 15-row prior within `rows` (none for 11 rows).
function _ball_index(prior::GWPrior, rows)
    nrows(prior) >= 15 || return NTuple{4,Int}[]
    out = NTuple{4,Int}[]
    for (rz, rx, ry, i) in _SPIN_BALLS
        iz, ix, iy = findfirst(==(rz), rows), findfirst(==(rx), rows), findfirst(==(ry), rows)
        iz === nothing || ix === nothing || iy === nothing || push!(out, (iz, ix, iy, i))
    end
    return out
end
# x (values on `rows`) inside the prior support: the box on the non-periodic rows and the spin balls
function _rows_in_support(prior::GWPrior, rows, x, period, balls)
    for i in eachindex(rows)
        (period[i] > 0 || prior.lo[rows[i]] <= x[i] <= prior.hi[rows[i]]) || return false
    end
    for (iz, ix, iy, i) in balls
        x[iz]^2 + x[ix]^2 + x[iy]^2 <= prior.spin_max[i]^2 || return false
    end
    imc, iq = findfirst(==(1), rows), findfirst(==(2), rows)
    (imc === nothing || iq === nothing || _masses_ok(prior, x[imc], x[iq])) || return false
    return true
end

function sample_reference(rng::AbstractRNG, ref::KDEPriorMixture{T}, n::Int) where {T}
    out = sample_prior(rng, ref.prior, n)
    k, K = size(ref.centers)
    z = Vector{T}(undef, k)
    balls = _ball_index(ref.prior, ref.rows)
    for w in 1:n
        rand(rng, T) < ref.alpha && continue
        j = rand(rng, 1:K)
        for tries in Iterators.countfrom(1)
            _reject_cap(tries, view(ref.centers, :, j), ref.rows)
            randn!(rng, z)
            x = view(ref.centers, :, j) .+ view(ref.L, :, :, j) * z
            ok = true
            for i in 1:k
                r = ref.rows[i]
                if ref.period[i] > 0
                    x[i] = _wrap_to(x[i], ref.prior.lo[r], ref.period[i])
                elseif !(ref.prior.lo[r] <= x[i] <= ref.prior.hi[r])
                    ok = false
                end
            end
            for (iz, ix, iy, b) in balls
                ok &= x[iz]^2 + x[ix]^2 + x[iy]^2 <= ref.prior.spin_max[b]^2
            end
            ok && 1 in ref.rows && (ok = _masses_ok(ref.prior, x[findfirst(==(1), ref.rows)], x[findfirst(==(2), ref.rows)]))
            if ok
                out[ref.rows, w] .= x
                break
            end
        end
    end
    return out
end

function reference_logpdf_and_grad!(lq::AbstractVector{T}, grad::AbstractMatrix{T}, ref::KDEPriorMixture{T},
                                    positions::AbstractMatrix{T}) where {T}
    k, K = size(ref.centers)
    nw = size(positions, 2)
    lp = Vector{T}(undef, nw)
    fill!(grad, zero(T))
    prior_logpdf_and_grad!(lp, grad, ref.prior, positions)
    # Threaded over walkers (independent; per-task buffers), triangular solves as plain loops: 2 K small solves per walker
    # are the SMC's per-gradient-call host cost with thousands of particles (0.6 s per call at 4096 x 400
    # kernels with BLAS trsv per kernel, which also contends across threads).
    ntask = max(1, min(Threads.nthreads(), nw))
    @sync for c in Iterators.partition(1:nw, cld(nw, ntask))
        Threads.@spawn begin
            lk = Vector{T}(undef, K)
            U = Matrix{T}(undef, k, K)
            d = Vector{T}(undef, k)
            gk = zeros(T, k)
            tmp = Vector{T}(undef, k)
            for w in c
                _kde_walker_logpdf_and_grad!(lq, grad, ref, positions, lp, w, lk, U, d, gk, tmp)
            end
        end
    end
    return lq
end

function _kde_walker_logpdf_and_grad!(lq, grad, ref::KDEPriorMixture{T}, positions, lp, w, lk, U, d, gk, tmp) where {T}
    k, K = size(ref.centers)
    if !isfinite(lp[w])
        lq[w] = T(-Inf)
        return nothing
    end
    for j in 1:K
        for i in 1:k
            d[i] = _nearest_image(positions[ref.rows[i], w] - ref.centers[i, j], ref.period[i])
        end
        # u = L_j \ d (forward substitution; k is small, so plain loops, not BLAS trsv per kernel)
        ss = zero(T)
        @inbounds for i in 1:k
            acc = d[i]
            for l in 1:i-1
                acc -= ref.L[i, l, j] * U[l, j]
            end
            ui = acc / ref.L[i, i, j]
            U[i, j] = ui
            ss += ui * ui
        end
        lk[j] = -ss / 2 - ref.log_kernel_norm[j]
    end
    m = maximum(lk)
    s = zero(T)
    for j in 1:K
        s += exp(lk[j] - m)
    end
    lkde = m + log(s) - log(T(K))
    lprior = lp[w] - ref.log_prior_norm
    lrows_un = zero(T); dmc = zero(T); dq = zero(T)
    if ref.prior.masses === :components && 1 in ref.rows
        mc = positions[1, w]; q = positions[2, w]
        lrows_un = log(mc) + T(0.4) * log1p(q) - T(1.2) * log(q)
        dmc = 1 / mc; dq = T(0.4) / (1 + q) - T(1.2) / q
    end
    a = log(ref.alpha) + lprior
    b = log1p(-ref.alpha) + lkde + lprior - (lrows_un - ref.log_rows_norm)
    mm = max(a, b)
    lq[w] = mm + log(exp(a - mm) + exp(b - mm))
    rb = exp(b - lq[w])
    # d log KDE = sum_j resp_j * (-(L_j L_j')^-1 d_j) = -sum_j resp_j L_j'^-1 u_j
    fill!(gk, zero(T))
    for j in 1:K
        rj = exp(lk[j] - m) / s
        rj < eps(T) && continue
        # tmp = L_j' \ u_j (back substitution)
        @inbounds for i in k:-1:1
            acc = U[i, j]
            for l in i+1:k
                acc -= ref.L[l, i, j] * tmp[l]
            end
            tmp[i] = acc / ref.L[i, i, j]
        end
        gk .+= rj .* tmp
    end
    for (i, r) in enumerate(ref.rows)
        grad[r, w] += rb * (-gk[i])
    end
    if ref.prior.masses === :components && 1 in ref.rows
        grad[1, w] -= rb * dmc
        grad[2, w] -= rb * dq
    end
    return nothing
end


"""
    local_kde_covariances(X, centers_idx, period; k=24, h=0.8, floor=0.02) -> Array (d, d, K)

Per-kernel bandwidth matrices for a LOCAL Gaussian KDE over samples `X` (d x N, rows in sampled
coordinates; `period[i] > 0` marks periodic rows): kernel j sits at `X[:, centers_idx[j]]` and gets
`h^2` times the covariance of its `k` nearest samples (distance scaled by the global sd per row,
nearest image on periodic rows), plus `floor^2` times the global covariance so no kernel is
degenerate. Unlike one global bandwidth, local covariances follow narrow curved ridges, so draws
from the mixture stay near the posterior's support.
"""
function local_kde_covariances(X::AbstractMatrix{T}, centers_idx::AbstractVector{<:Integer},
                               period::AbstractVector; k::Int=24, h=0.8, floor=0.02) where {T}
    d, N = size(X)
    k = min(k, N - 1)
    dev(a, b, i) = (Δ = a - b; period[i] > 0 ? Δ - period[i] * round(Δ / period[i]) : Δ)
    mu = [period[i] > 0 ? X[i, 1] : sum(view(X, i, :)) / N for i in 1:d]
    D = [dev(X[i, n], mu[i], i) for i in 1:d, n in 1:N]
    D .-= sum(D; dims=2) ./ N
    G = D * D' ./ (N - 1)
    sd = sqrt.([G[i, i] for i in 1:d])
    out = Array{T,3}(undef, d, d, length(centers_idx))
    dist = Vector{T}(undef, N)
    for (j, c) in enumerate(centers_idx)
        for n in 1:N
            dist[n] = sum(i -> abs2(dev(X[i, n], X[i, c], i) / sd[i]), 1:d)
        end
        dist[c] = T(Inf)
        nn = partialsortperm(dist, 1:k)
        E = [dev(X[i, n], X[i, c], i) for i in 1:d, n in nn]
        E .-= sum(E; dims=2) ./ k
        out[:, :, j] .= T(h)^2 .* (E * E') ./ (k - 1) .+ T(floor)^2 .* G
    end
    return out
end

end
