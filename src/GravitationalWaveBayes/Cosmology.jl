module Cosmology

# Cosmological distance prior, for reweighting posteriors sampled with the Euclidean d_L^2 prior (what the samplers use:
# the distance marginalization assumes a power law) to LVK's "cosmo" prior: a uniform merger rate in the source's
# comoving frame, p(z) ∝ dV_c/dz / (1 + z) (bilby `UniformSourceFrame`; GWTC-2 / GWTC-2.1 `_cosmo` releases), with
# Planck15_LAL (bilby `get_cosmology("planck15_lal")`: astropy FlatLambdaCDM(H0 = lal.H0_SI, Om0 = lal.OMEGA_M), no
# radiation; LAL 7.7.0 values below; LIGO-T2000185). This is how LVK makes `_cosmo` from `_nocosmo` samples.

using Random

export FlatLambdaCDM, PLANCK15_LAL, luminosity_distance, redshift_at_luminosity_distance, source_frame_prior_weights,
       source_frame_log_weights, reweight_to_source_frame

const _C_SI = 299792458.0
const _MPC_SI = 3.085677581491367e22          # 1e6 lal.PC_SI

"""
    FlatLambdaCDM(H0_SI, Om0)

Flat ΛCDM without radiation (astropy's `FlatLambdaCDM` default `Tcmb0 = 0`): `E(z) = sqrt(Om0 (1 + z)^3 + 1 - Om0)`.
`H0_SI` in 1/s.
"""
struct FlatLambdaCDM
    H0_SI::Float64
    Om0::Float64
end

"`Planck15_LAL`: lal.H0_SI = 2.200489137532724e-18 1/s (67.90 km/s/Mpc), lal.OMEGA_M = 0.3065 (LAL 7.7.0)."
const PLANCK15_LAL = FlatLambdaCDM(2.200489137532724e-18, 0.3065)

_E(c::FlatLambdaCDM, z) = sqrt(c.Om0 * (1 + z)^3 + 1 - c.Om0)
_hubble_distance_mpc(c::FlatLambdaCDM) = _C_SI / c.H0_SI / _MPC_SI

# Comoving distance (Mpc): 16-point Gauss-Legendre on [0, z] per unit-length piece (the integrand is smooth and
# slowly varying; relative error < 1e-12 for z <= 10).
const _GL16 = let
    x = [-0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.7554044083550030, -0.6178762444026438,
         -0.4580167776572274, -0.2816035507792589, -0.0950125098376374]
    w = [0.0271524594117541, 0.0622535239386479, 0.0951585116824928, 0.1246289712555339, 0.1495959888165767,
         0.1691565193950025, 0.1826034150449236, 0.1894506104550685]
    (vcat(x, -reverse(x)), vcat(w, reverse(w)))
end
function _comoving_distance_mpc(c::FlatLambdaCDM, z::Real)
    z <= 0 && return 0.0
    n = max(1, ceil(Int, z))
    h = z / n
    s = 0.0
    for k in 0:(n - 1), (x, w) in zip(_GL16...)
        s += w * h / 2 / _E(c, h * (k + (x + 1) / 2))
    end
    return _hubble_distance_mpc(c) * s
end

"""
    luminosity_distance(z; cosmology = PLANCK15_LAL) -> Mpc
"""
luminosity_distance(z::Real; cosmology::FlatLambdaCDM=PLANCK15_LAL) = (1 + z) * _comoving_distance_mpc(cosmology, z)

"""
    redshift_at_luminosity_distance(dL_mpc; cosmology = PLANCK15_LAL) -> z

Inverse of [`luminosity_distance`](@ref) (bisection then Newton, to ~1e-14 relative).
"""
function redshift_at_luminosity_distance(dL::Real; cosmology::FlatLambdaCDM=PLANCK15_LAL)
    dL > 0 || throw(ArgumentError("luminosity distance must be positive, got $dL"))
    lo, hi = 0.0, 1.0
    while luminosity_distance(hi; cosmology) < dL
        hi *= 2
    end
    for _ in 1:60
        mid = (lo + hi) / 2
        luminosity_distance(mid; cosmology) < dL ? (lo = mid) : (hi = mid)
    end
    z = (lo + hi) / 2
    for _ in 1:3
        z -= (luminosity_distance(z; cosmology) - dL) / _ddl_dz(cosmology, z)
    end
    return z
end

_ddl_dz(c::FlatLambdaCDM, z) = _comoving_distance_mpc(c, z) + (1 + z) * _hubble_distance_mpc(c) / _E(c, z)

"""
    source_frame_prior_weights(dL_mpc; cosmology = PLANCK15_LAL) -> Vector{Float64}

Importance weights (normalized to max 1) taking samples drawn under a `d_L^2` prior to the uniform-rate-in-source-frame
prior p(d_L) ∝ [dV_c/dz / (1 + z)] / (dd_L/dz), dV_c/dz = 4π D_c^2 D_H / E(z): w = p(d_L) / d_L^2. Both priors share
the same d_L bounds (the reweighting cannot extend the support).
"""
function source_frame_prior_weights(dL::AbstractVector{<:Real}; cosmology::FlatLambdaCDM=PLANCK15_LAL)
    w = map(d -> _source_frame_density(cosmology, d) / d^2, dL)
    return w ./ maximum(w)
end

# dV_c/dz / (1 + z) at redshift z (Mpc^3), and the unnormalized p(d_L) = that / (dd_L/dz)
function _dvc_dz_over_1pz(c::FlatLambdaCDM, z)
    dc = _comoving_distance_mpc(c, z)
    return 4pi * dc^2 * _hubble_distance_mpc(c) / _E(c, z) / (1 + z)
end
function _source_frame_density(c::FlatLambdaCDM, d)
    z = redshift_at_luminosity_distance(d; cosmology=c)
    return _dvc_dz_over_1pz(c, z) / _ddl_dz(c, z)
end

"""
    source_frame_log_weights(dL_mpc, lo_mpc, hi_mpc; cosmology = PLANCK15_LAL) -> Vector{Float64}

`log p_sf(d_L) - log p_vol(d_L)` with BOTH priors normalized on `[lo, hi]` (p_vol = 3 d_L^2 / (hi^3 - lo^3)): importance
weights from a run under the d_L^2 prior to bilby's `UniformSourceFrame(lo, hi)`. Their posterior mean is
Z_sf / Z_vol, so `logZ_sf = logZ_vol + log(mean(exp.(logw)))`.
"""
function source_frame_log_weights(dL::AbstractVector{<:Real}, lo::Real, hi::Real; cosmology::FlatLambdaCDM=PLANCK15_LAL)
    0 < lo < hi || throw(ArgumentError("need 0 < lo < hi"))
    za, zb = redshift_at_luminosity_distance(lo; cosmology), redshift_at_luminosity_distance(hi; cosmology)
    # normalizer: ∫ dV_c/dz / (1 + z) dz over [za, zb], Gauss-Legendre on pieces of length <= 0.05
    n = max(1, ceil(Int, (zb - za) / 0.05)); h = (zb - za) / n
    norm = 0.0
    for k in 0:(n - 1), (x, w) in zip(_GL16...)
        norm += w * h / 2 * _dvc_dz_over_1pz(cosmology, za + h * (k + (x + 1) / 2))
    end
    lvol = log(3 / (hi^3 - lo^3))
    return [log(_source_frame_density(cosmology, d) / norm) - (lvol + 2 * log(d)) for d in dL]
end

"""
    reweight_to_source_frame(rng, dL_mpc; cosmology = PLANCK15_LAL) -> Vector{Int}

Rejection sampling with [`source_frame_prior_weights`](@ref): the indices of the samples kept (each kept with
probability w / max w), i.e. an unweighted posterior sample under the cosmological prior.
"""
function reweight_to_source_frame(rng::AbstractRNG, dL::AbstractVector{<:Real}; cosmology::FlatLambdaCDM=PLANCK15_LAL)
    w = source_frame_prior_weights(dL; cosmology)
    return findall(i -> rand(rng) < w[i], eachindex(w))
end

end
