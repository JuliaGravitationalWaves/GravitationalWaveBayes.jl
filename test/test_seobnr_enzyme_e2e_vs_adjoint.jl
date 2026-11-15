using Test
using GravitationalWaveBayes
using GravitationalWaveforms
using Enzyme

# Independent cross-check of the hand-derived SEOBNRv5AlignedSpinF2Adjoint: Enzyme differentiating
# the WHOLE SEOBNR table build end-to-end (IC Newton solve, fixed PA driver, omega table, F2 phase/
# amplitude). That used to stop at EnzymeNoTypeError, which is why the hand adjoint exists; with
# Enzyme's maxtypeoffset raised to >= 4096 (`BatchedNUTS.__init__` sets 32768 whenever GravitationalWaveBayes is
# loaded) it works in both modes. Measured on the CPU: agrees with the hand adjoint
# to 1.4e-11 relative. Central finite differences only agree to 1e-5..2e-4 depending on the step,
# which is FD noise from the solver, not a disagreement. The hand adjoint stays in production:
# 13.5 ms per walker vs 38 ms for the end-to-end reverse pass.

const _E2E_FS = (21.0, 23.5, 26.0, 29.0)

function _e2e_lossc(c)
    s = 0.0
    for f in _E2E_FS
        s += GravitationalWaveforms._seobv5f2_phi_scalar(c, f) +
             1e21 * GravitationalWaveforms._seobv5f2_ampl_scalar(c, f, 0.4, GravitationalWaveforms.GMsun_over_c2_Gpc)
    end
    return s
end

_e2e_seob_loss(mc, eta, chi1, chi2) =
    _e2e_lossc(GravitationalWaveforms._seobv5f2_coeffs(SEOBNRv5AlignedSpinF2(), mc, eta, chi1, chi2))

@testset "SEOBNRv5AlignedSpinF2: Enzyme end-to-end gradient matches the hand-derived adjoint" begin
    model = SEOBNRv5AlignedSpinF2()
    x = (28.1, 0.247, 0.3, -0.2)
    c = GravitationalWaveforms._seobv5f2_coeffs(model, x...)
    hand = ntuple(4) do k
        e = ntuple(j -> j == k ? 1.0 : 0.0, 4)
        dc = GravitationalWaveBayes.SEOBNRv5AlignedSpinF2Adjoint.seobv5f2_coeffs_jvp(model, x..., e...)
        Enzyme.autodiff(Enzyme.Forward, _e2e_lossc, Duplicated(c, dc))[1]
    end
    rev = Enzyme.autodiff(Enzyme.Reverse, _e2e_seob_loss, Active, Active.(x)...)[1]
    fwd = ntuple(k -> Enzyme.autodiff(Enzyme.Forward, _e2e_seob_loss,
        ntuple(j -> Duplicated(x[j], j == k ? 1.0 : 0.0), 4)...)[1], 4)
    for k in 1:4
        @test rev[k] ≈ hand[k] rtol = 1e-9
        @test fwd[k] ≈ hand[k] rtol = 1e-9
    end
end
