module Injection

export detector_strain, simulate_injection, optimal_snr, matched_filter_snr, time_at_frequency

using Random
using ..Types: LikelihoodData
using GravitationalWaveforms: IMRPhenomXAS, IMRPhenomXHM, IMRPhenomD
using ..Types: PAF2Model
import ..KernelFused
using GravitationalWaveData: gmst, geometry, time_delay_from_geocenter_gmst
using GravitationalWaveforms: _seobv5f2_coeffs, _seobv5f2_ampl_scalar, _seobv5f2_phi_scalar, GMsun_over_c2_Gpc, _xas_phi_coeffs

# ------------------------------------------------------------------------------------------------
# Simulated-data injections: a model signal plus stationary Gaussian noise drawn from a detector's
# PSD, on that detector's own frequency grid. This is the controlled setting (known truth, exactly
# Gaussian noise, the same waveform model for injection and recovery) needed for sampler
# convergence studies and probability-probability (P-P) calibration tests.
#
# Noise convention matches the likelihood in `KernelFused.jl`: `logL = -2 df sum |d-h|^2 / S`
# (= -1/2 <d-h,d-h>, <a,b> = 4 df Re sum conj(a) b / S). Unit-variance-per-inner-product noise needs
# independent real and imaginary parts, each N(0, S / (4 df)) -- then <n,h> ~ N(0, <h,h>) and
# <n,n> has mean 2 per bin.
# ------------------------------------------------------------------------------------------------

"""
    detector_strain(model::PAF2Model, data::LikelihoodData, physical; geocentric=true) -> Vector{ComplexF64}

Frequency-domain strain the likelihood kernels would predict in `data`'s detector for the PHYSICAL
parameters `physical = (m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi)` (see
`KernelFused.jl`'s row convention). Uses the kernels' own scalar pieces, so injection and recovery
share one forward model. `geocentric=true` treats `t_c` as the geocentric arrival time (the
multi-detector convention) and adds the detector's light-travel delay; `geocentric=false` treats it
as the arrival time at this detector (the single-detector likelihood's convention).
"""
function detector_strain(model::PAF2Model, data::LikelihoodData, physical; geocentric::Bool=true,
                         fmax_taper::Real=0.0)
    m1, m2, chi1, chi2, dL, t_c, phase, ra, dec, theta_jn, psi = physical
    mc, eta = KernelFused._chirp_mass_eta(m1, m2)
    coeffs = _seobv5f2_coeffs(model, mc, eta, chi1, chi2)
    coeffs.success || throw(ArgumentError("waveform table construction failed for $(physical)"))
    g = gmst(data.gps_time)
    phiRef, combo = KernelFused._extrinsic_setup_phase_only(phase, ra, dec, theta_jn, psi, geometry(data.detector), g)
    t_det = geocentric ? t_c + time_delay_from_geocenter_gmst(geometry(data.detector), ra, dec, g) : t_c
    return [combo * _seobv5f2_ampl_scalar(coeffs, f, dL, GMsun_over_c2_Gpc) *
            KernelFused._end_taper(f, coeffs.fmax, Float64(fmax_taper)) *
            cis(-(_seobv5f2_phi_scalar(coeffs, f) + 2pi * t_det * f - phiRef)) for f in data.freqs]   # KernelFused _PHASE_CONVENTION
end

"""
    time_at_frequency(model::PAF2Model, m1, m2, chi1, chi2, f) -> Float64

The model's stationary-phase time (s, table time origin) at which the GW frequency equals `f` -- the
`t*` that `PEConfig.time_ref_frequency` adds to the coalescence time (`KernelFused._seobv5f2_time_shift`).
"""
function time_at_frequency(model::PAF2Model, m1, m2, chi1, chi2, f)
    mc, eta = KernelFused._chirp_mass_eta(m1, m2)
    coeffs = _seobv5f2_coeffs(model, mc, eta, chi1, chi2)
    coeffs.success || throw(ArgumentError("waveform table construction failed"))
    return KernelFused._seobv5f2_time_shift(coeffs, (fref=Float64(f), toff=0.0))
end

# IMRPhenomXAS: `-Phi'(f) / 2 pi` of the intrinsic phase (`KernelFused._xas_time_shift`). XAS's phase has no
# merger-time alignment, so this time origin is NOT the merger: with `time_ref_frequency` use a numeric
# `time_ref_offset` shared with the other model(s) when comparing time rows across models.
function time_at_frequency(model::IMRPhenomXAS, m1, m2, chi1, chi2, f)
    mc, eta = KernelFused._chirp_mass_eta(m1, m2)
    return KernelFused._xas_time_shift(_xas_phi_coeffs(model, mc, eta, chi1, chi2), (fref=Float64(f), toff=0.0))
end

# IMRPhenomXHM: the (2,2) mode's -Phi'(f) / 2 pi (`KernelFused._xhm_time_shift`: XAS's plus the XHM timeshift term).
function time_at_frequency(model::IMRPhenomXHM, m1, m2, chi1, chi2, f)
    sys, = KernelFused._xhm_setup(m1, m2, chi1, chi2, 1.0, 0.0, 20.0)
    return KernelFused._xhm_time_shift(sys, (fref=Float64(f), toff=0.0))
end

# IMRPhenomD: its phase enters as exp(-i Phi), so t* = +Phi'(f) / 2 pi (`KernelFused._pd_time_shift`).
function time_at_frequency(model::IMRPhenomD, m1, m2, chi1, chi2, f)
    mc, eta = KernelFused._chirp_mass_eta(m1, m2)
    return KernelFused._pd_time_shift(KernelFused._phi_coeffs(model, mc, eta, chi1, chi2), (fref=Float64(f), toff=0.0))
end

_inner(a, b, psd, df) = 4 * df * real(sum(conj(a[i]) * b[i] / psd[i] for i in eachindex(a) if psd[i] > 0))

"""
    optimal_snr(h, data) -> Float64

`sqrt(<h,h>)` on `data`'s grid and PSD.
"""
optimal_snr(h::AbstractVector, data::LikelihoodData) =
    sqrt(_inner(h, h, data.psd, data.freqs[2] - data.freqs[1]))

"""
    matched_filter_snr(h, data) -> Float64

`<d,h>/sqrt(<h,h>)` at the template's own phase and time.
"""
matched_filter_snr(h::AbstractVector, data::LikelihoodData) =
    _inner(data.strain, h, data.psd, data.freqs[2] - data.freqs[1]) / optimal_snr(h, data)

"""
    simulate_injection(model, physical, templates; rng=Random.default_rng(), noise=true, geocentric=nothing)
        -> (data::Vector{LikelihoodData}, info::NamedTuple)

Replace each template's strain by the injected signal plus (if `noise`) Gaussian noise drawn from
that template's PSD. Frequency grids, PSDs, detectors and the event GPS time are taken from
`templates` (e.g. real-data `LikelihoodData`s from `build_gwdata`, so the noise curve is realistic).
`fmax_taper` must match the analysis's `PEConfig.fmax_taper`. `geocentric` defaults to `length(templates) > 1`, matching how the likelihood interprets `t_c`
(`PEConfig.time_reference` then selects the sampled time's frame). `info` holds per-detector optimal
and matched-filter SNRs and their network totals.
"""
function simulate_injection(model::PAF2Model, physical, templates::AbstractVector{<:LikelihoodData};
                            rng::AbstractRNG=Random.default_rng(), noise::Bool=true,
                            geocentric::Union{Nothing,Bool}=nothing, fmax_taper::Real=0.0)
    geo = geocentric === nothing ? length(templates) > 1 : geocentric
    h = [detector_strain(model, tmpl, physical; geocentric=geo, fmax_taper=fmax_taper) for tmpl in templates]
    return simulate_injection(h, templates; rng=rng, noise=noise)
end

"""
    simulate_injection(h::AbstractVector{<:AbstractVector}, templates; rng=Random.default_rng(), noise=true)
        -> (data::Vector{LikelihoodData}, info::NamedTuple)

As the model method, for detector templates `h[k]` already evaluated on `templates[k]`'s frequency grid (e.g.
`KernelFused.xprec_strain` / `seobnr_phm_device_strain`, the likelihood's own templates).
"""
function simulate_injection(h::AbstractVector{<:AbstractVector}, templates::AbstractVector{<:LikelihoodData};
                            rng::AbstractRNG=Random.default_rng(), noise::Bool=true)
    length(h) == length(templates) || throw(ArgumentError("one template per detector: got $(length(h)) for $(length(templates))"))
    out = LikelihoodData[]
    opt = Float64[]
    mf = Float64[]
    for (hk, tmpl) in zip(h, templates)
        df = tmpl.freqs[2] - tmpl.freqs[1]
        n = zeros(ComplexF64, length(hk))
        if noise
            for i in eachindex(n)
                sigma = tmpl.psd[i] > 0 ? sqrt(tmpl.psd[i] / (4 * df)) : 0.0
                n[i] = complex(sigma * randn(rng), sigma * randn(rng))
            end
        end
        d = LikelihoodData(tmpl.freqs, hk .+ n, tmpl.psd, tmpl.detector, tmpl.gps_time)
        push!(out, d)
        push!(opt, optimal_snr(hk, d))
        push!(mf, matched_filter_snr(hk, d))
    end
    info = (optimal_snr=opt, matched_filter_snr=mf,
            network_optimal_snr=sqrt(sum(abs2, opt)), network_matched_filter_snr=sum(mf .* opt) / sqrt(sum(abs2, opt)))
    return out, info
end

end
