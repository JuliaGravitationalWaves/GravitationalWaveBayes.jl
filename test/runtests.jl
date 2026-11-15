using Test
using GravitationalWaveBayes
using Aqua

# Test selection: `Pkg.test(test_args = ["priors", "test_smc_core.jl"])` runs only the files whose names contain an arg
# (an arg ending in .jl must match exactly; "aqua" = the Aqua checks); a slow file named this way runs too. Each file
# prints `[test-time] <file> <s>`. Parallel full runs: `julia +1.13.2 test/run_parallel.jl -j 3` (see there).
const SELECT = ARGS
selected(file) = isempty(SELECT) || any(a -> endswith(a, ".jl") ? a == file : occursin(a, file), SELECT)

function test_include(file)
    selected(file) || return
    t = @elapsed include(file)
    println("[test-time] $file $(round(t; digits=1))")
end

if selected("aqua")
    @testset "Code quality (Aqua.jl)" begin
        # DelimitedFiles (like Serialization/TOML, which Aqua can't flag because other deps load
        # them) is a dep only for scripts/run_from_config.jl, which runs in the package env.
        Aqua.test_all(GravitationalWaveBayes; stale_deps=(ignore=[:DelimitedFiles],))
    end
end

test_include("test_env_names.jl")

# Enzyme reverse-mode compilation is slow (minutes per testset, dominated by compile time, not
# test logic) and, once a kernel's differentiability has been established, re-verifying it on
# every `Pkg.test()` run adds little value -- Julia's AD either works structurally or it doesn't;
# it does not regress silently the way a numeric formula can. These heavy gradient-vs-finite-
# difference / gradient-through-kernel testsets are skipped by default; set
# GRAVITATIONALWAVEBAYES_RUN_SLOW_TESTS=true to run them (e.g. after touching kernel/Enzyme-sensitive code).
const RUN_SLOW_ENZYME_TESTS = get(ENV, "GRAVITATIONALWAVEBAYES_RUN_SLOW_TESTS", "false") == "true"

# `test_include(file)` when the slow tests are on or the file was selected by name; otherwise a skipped placeholder
# testset `name`.
function slow_include(file, name, why="slow, dominated by Enzyme compile time")
    selected(file) || return
    if RUN_SLOW_ENZYME_TESTS || !isempty(SELECT)
        test_include(file)
    else
        @testset "$name" begin
            @test_skip "skipped by default ($why); set GRAVITATIONALWAVEBAYES_RUN_SLOW_TESTS=true to run"
        end
    end
end

test_include("test_kernel_fused.jl")

test_include("test_seobnr_tc_units.jl")

slow_include("test_enzyme_ka.jl", "Enzyme gradient through KernelAbstractions kernels (IMRPhenomD, single- and multi-detector)")

slow_include("test_enzyme_ka_xas.jl", "Enzyme gradient through the fused KernelAbstractions kernel (IMRPhenomXAS)")

slow_include("test_enzyme_ka_alignedspinpnhamf2.jl", "Enzyme gradient through the fused KernelAbstractions kernel (AlignedSpinPNHamF2)")

slow_include("test_seobnr_forward_mode_gradient.jl", "SEOBNRv5AlignedSpinF2 forward-mode Enzyme gradient (gw_logp_and_grad) vs finite differences")

test_include("test_priors.jl")
test_include("test_jump_moves.jl")
test_include("test_real_data_conventions.jl")
test_include("test_xas_likelihood_options.jl")
test_include("test_xhm_likelihood_options.jl")
test_include("test_phenomd_likelihood_options.jl")
test_include("test_calmarg_dl_recon.jl")
test_include("test_xprec_likelihood_options.jl")
test_include("test_xas_relative_binning.jl")
test_include("test_zero_psd_gradient.jl")
test_include("test_xas_binparallel.jl")
test_include("test_seobnr_device_table.jl")

slow_include("test_phase_marg_time_reference.jl", "phase/distance marginalization, time reference, multi-detector reverse gradient")

slow_include("test_seobnr_enzyme_e2e_vs_adjoint.jl", "SEOBNRv5AlignedSpinF2: Enzyme end-to-end gradient matches the hand-derived adjoint")

slow_include("test_seobnrv5alignedspinf2_adjoint.jl", "SEOBNRv5AlignedSpinF2Adjoint: seobv5f2_coeffs_jvp vs finite differences")

slow_include("test_seobnr_reverse_mode_gradient_cuda_repeated.jl", "SEOBNRv5AlignedSpinF2 reverse-mode CUDA: repeated calls do not leak device heap", "slow, needs CUDA")

test_include("test_phenomxp.jl")

test_include("test_phenomxhm.jl")

test_include("test_phenomxphm.jl")

test_include("test_batched_nuts_gaussian.jl")

test_include("test_batched_nuts_gw_smoke.jl")

test_include("test_pt_nuts_core.jl")

test_include("test_smc_core.jl")
test_include("test_smc_mode_jumps.jl")

test_include("test_smc_reference.jl")

test_include("test_batched_nuts_pt_gw_smoke.jl")

test_include("test_relative_binning.jl")

test_include("test_seobnr_deadzone_reject.jl")

slow_include("test_seobnr_deadzone_reject_gradient.jl", "SEOBNRv5AlignedSpinF2 reverse-mode Enzyme gradient: dead-zone reject stays finite, good walker unaffected")

test_include("test_dataio.jl")
test_include("test_cosmology.jl")
slow_include("test_injection_templates.jl", "precessing injection templates: logL = 0 at the truth", "slow, SEOBNRv5PHM device pipeline compile")
test_include("test_calibration_psdfile.jl")

# Diagnostic tooling and NUTS boundary-reflection tests -- both are pure-Julia (Random/Statistics/Test only), no Enzyme/GPU involved, so they run
# unconditionally rather than being gated behind GRAVITATIONALWAVEBAYES_RUN_SLOW_TESTS.
test_include("test_diagnostics.jl")

test_include("test_nuts_core_bounds.jl")

# Diagonal mass-matrix adaptation (NUTSCore): pure-Julia analytic-Gaussian checks, no Enzyme/GPU.
test_include("test_nuts_core_mass_matrix.jl")
test_include("test_nuts_core_dense_mass.jl")
test_include("test_nuts_core_block_mass.jl")
test_include("test_nuts_core_quorum.jl")
test_include("test_analytic_init.jl")
test_include("test_flows.jl")
test_include("test_compact_active.jl")

slow_include("test_gradient_validation_boundaries.jl", "Enzyme gradient vs finite differences near box-bounded sampled-space prior edges")

# 2-8 min each.
slow_include("test_seobnr_reverse_mode_gradient.jl", "SEOBNRv5AlignedSpinF2 reverse-mode Enzyme gradient (gw_logp_and_grad) vs finite differences and forward-mode")
slow_include("test_calibration_binparallel.jl", "IMRPhenomXPHM bin-parallel with calibration (CPU backend)")
slow_include("test_batched_nuts_seobnrv5alignedspinf2_smoke.jl", "SEOBNRv5AlignedSpinF2 dense-kernel NUTS smoke test")
slow_include("test_enzyme_ka_seobnrv5alignedspinf2.jl", "Enzyme gradient through the fused KernelAbstractions kernel (SEOBNRv5AlignedSpinF2)")
slow_include("test_nuts_recovery_seobnrv5alignedspinf2.jl", "SEOBNRv5AlignedSpinF2: dense-kernel NUTS statistical recovery (real injection)")
slow_include("test_enzyme_ka_phenomxphm.jl", "Enzyme gradient through the fused KernelAbstractions kernel (IMRPhenomXPHM)")
slow_include("test_enzyme_ka_phenomxhm.jl", "Enzyme gradient through the fused KernelAbstractions kernel (IMRPhenomXHM)")
slow_include("test_enzyme_ka_phenomxhm_multi.jl", "Enzyme gradient through the fused KernelAbstractions kernel (IMRPhenomXHM, multi-detector H1+L1)")
slow_include("test_enzyme_ka_relative_binning_phenomxhm.jl", "Enzyme gradient through the binned relative-binning kernel (IMRPhenomXHM)")
