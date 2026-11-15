module GravitationalWaveBayes

using LinearAlgebra

include("GravitationalWaveBayes/Types.jl")
using .Types: PEConfig, with_likelihood_options, PEState, LikelihoodData, DetectorData, PAF2Model, IMRTableModel

include("GravitationalWaveBayes/GPU.jl")
using .GPU: to_device, compatible_array

include("GravitationalWaveBayes/KernelCache.jl")

using GravitationalWaveforms: IMRPhenomD, IMRPhenomXAS, IMRPhenomXP, IMRPhenomXHM

include("GravitationalWaveBayes/SEOBNRv5AlignedSpinF2Adjoint.jl")
using .SEOBNRv5AlignedSpinF2Adjoint: seobv5f2_coeffs_jvp

include("GravitationalWaveBayes/KernelFused.jl")
using .KernelFused: batched_log_likelihood_kernel!

include("GravitationalWaveBayes/Injection.jl")
using .Injection: detector_strain, simulate_injection, optimal_snr, matched_filter_snr, time_at_frequency

include("GravitationalWaveBayes/RelativeBinning.jl")
using .RelativeBinning: RelativeBinningSetup, build_relative_binning, binned_log_likelihood_kernel!

include("GravitationalWaveBayes/Priors.jl")
using .Priors: isotropic_spins!, isotropic_spins_pullback!, prior_support_vec, GWPrior, prior_logpdf!, prior_logpdf_and_grad!, prior_bounds, sample_prior, prior_log_normalizer, GaussianPriorMixture, KDEPriorMixture, local_kde_covariances, sample_reference, reference_logpdf_and_grad!

include("GravitationalWaveBayes/Flows.jl")
using .Flows: FlowDensity, fit_flow, flow_logpdf!, flow_sample, flow_to_latent, flow_from_latent, flow_latent_pull

include("GravitationalWaveBayes/NUTSCore.jl")
using .NUTSCore: smc_sample, nuts_sample!, NUTSConfig, NUTSBounds, PTLadderState, pt_swap!, DenseMetric

include("GravitationalWaveBayes/BatchedNUTS.jl")
using .BatchedNUTS: batched_nuts!, run_gw150914_estimation, batched_pt_nuts!, run_gw150914_pt_estimation

include("GravitationalWaveBayes/AnalyticInit.jl")
using .AnalyticInit: analytic_reference_metric, informed_initial_positions, analytic_snr2, analytic_beta_min

include("GravitationalWaveBayes/Calibration.jl")
using .Calibration: calibration_weights, calibration_factor, calibration_nodes, calibration_prior, calibration_curves

include("GravitationalWaveBayes/DataIO.jl")
using .DataIO: load_gwosc_strain, condition_segment, to_frequency_domain, build_gwdata, build_gwdata_multi, synthetic_gwdata

include("GravitationalWaveBayes/Diagnostics.jl")
using .Diagnostics: split_rhat, ess_geyer, ess_multichain, ParamDiagnostic, chain_diagnostics, worst_rhat, min_ess, print_diagnostics

include("GravitationalWaveBayes/Cosmology.jl")
using .Cosmology: FlatLambdaCDM, PLANCK15_LAL, luminosity_distance, redshift_at_luminosity_distance,
    source_frame_prior_weights, source_frame_log_weights, reweight_to_source_frame

export FlatLambdaCDM, PLANCK15_LAL, luminosity_distance, redshift_at_luminosity_distance, source_frame_prior_weights,
    source_frame_log_weights, reweight_to_source_frame

export calibration_weights, calibration_factor, calibration_nodes, calibration_prior, calibration_curves
export isotropic_spins!, isotropic_spins_pullback!, prior_support_vec
export FlowDensity, fit_flow, flow_logpdf!, flow_sample, flow_to_latent, flow_from_latent, flow_latent_pull
export detector_strain, simulate_injection, optimal_snr, matched_filter_snr, time_at_frequency, GWPrior, prior_logpdf!, prior_logpdf_and_grad!, prior_bounds, sample_prior, prior_log_normalizer, GaussianPriorMixture, KDEPriorMixture, local_kde_covariances, sample_reference, reference_logpdf_and_grad!, PEConfig, with_likelihood_options, PEState, LikelihoodData, DetectorData, PAF2Model, IMRTableModel, IMRPhenomD, IMRPhenomXAS, IMRPhenomXP, IMRPhenomXHM, batched_log_likelihood_kernel!, batched_nuts!, run_gw150914_estimation, to_device, compatible_array, nuts_sample!, smc_sample, NUTSConfig, NUTSBounds, DenseMetric, PTLadderState, pt_swap!, batched_pt_nuts!, run_gw150914_pt_estimation, load_gwosc_strain, condition_segment, to_frequency_domain, build_gwdata, build_gwdata_multi, synthetic_gwdata, RelativeBinningSetup, build_relative_binning, binned_log_likelihood_kernel!, split_rhat, ess_geyer, ess_multichain, ParamDiagnostic, chain_diagnostics, worst_rhat, min_ess, print_diagnostics, analytic_reference_metric, informed_initial_positions, analytic_snr2, analytic_beta_min

# Variables renamed with the package: an old one would otherwise be ignored without any error, silently changing a run's
# tuning.
function _check_renamed_env(env=ENV)
    old = "GW" * "BAYES_"
    for k in keys(env)
        startswith(k, old) && error("environment variable $k was renamed to GRAVITATIONALWAVEBAYES_$(k[ncodeunits(old)+1:end])")
    end
    return nothing
end
__init__() = _check_renamed_env()

end
