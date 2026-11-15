using Test
using Random
using GravitationalWaveBayes
using GravitationalWaveforms: SEOBNRv5AlignedSpinF2

const BN = GravitationalWaveBayes.BatchedNUTS

# Regression test for the CUDA reverse-mode device-heap LEAK.
#
# Enzyme's reverse pass caches a loop with an unknown trip count via its "exponential allocation"
# scheme; on GPU (no `realloc`) the grow step `malloc`s a bigger buffer, `memcpy`s and never frees the
# old one. `searchsortedlast` inside `GravitationalWaveforms._seobv5f2_dense_to_freq` was such a loop, leaking
# ~0.5MB of CUDA device heap per 16-walker gradient call; the 8MB default heap was exhausted after
# exactly ~11 calls with `CUDA error: an illegal memory access (code 700)`. Fixed by a constant-trip-
# count search (`GravitationalWaveforms._seobv5f2_searchsortedlast`). A single-call test cannot see this -- only
# REPEATED calls at real-ish scale (16 walkers, ~433 frequency bins) do -- so this loops 30 calls
# (~3x the old crash point) and then checks the reverse gradient against forward mode.
#
# Needs a functional CUDA device and `CUDA` in the test environment (it is only a weakdep of GravitationalWaveBayes);
# skipped otherwise. Slow (dominated by Enzyme's GPU compile time), so runtests.jl only includes it
# under GRAVITATIONALWAVEBAYES_RUN_SLOW_TESTS.

const _HAVE_CUDA_FOR_REVERSE_LEAK_TEST = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

if !_HAVE_CUDA_FOR_REVERSE_LEAK_TEST
    @testset "SEOBNRv5AlignedSpinF2 reverse-mode CUDA: repeated calls do not leak device heap" begin
        @test_skip "no functional CUDA device / CUDA.jl not in the test environment"
    end
else
    @testset "SEOBNRv5AlignedSpinF2 reverse-mode CUDA: repeated calls do not leak device heap" begin
        nwalkers = 16
        ntime = 433
        ncalls = 30

        freqs = collect(range(20.0, 128.0; length = ntime))
        # Signal-bearing fixture (strain=0/psd=1 gives ~1e-44 gradients that underflow and make the
        # forward-vs-reverse comparison meaningless on some rows).
        rng = MersenneTwister(7)
        strain = 1e-22 .* (randn(rng, ntime) .+ im .* randn(rng, ntime))
        psd = fill(1e-46, ntime)
        data_cpu = LikelihoodData(freqs, strain, psd)
        cfg = PEConfig{Float64}(ntime, nwalkers, 0.01, 1)
        data_gpu = LikelihoodData(CuArray(data_cpu.freqs), CuArray(data_cpu.strain), CuArray(data_cpu.psd),
                                  data_cpu.detector, data_cpu.gps_time)

        positions = zeros(Float64, 11, nwalkers)
        for w in 1:nwalkers
            m1 = 36.0 + 0.5 * mod(w, 3); m2 = 29.0 + 0.2 * mod(w, 4)
            mc, q = BN._m1_m2_to_mc_q(m1, m2)
            positions[1, w] = mc; positions[2, w] = q
            positions[3, w] = 0.02 * mod(w, 3) - 0.02; positions[4, w] = -0.01 * mod(w, 2)
            positions[5, w] = 0.4 + 0.02 * mod(w, 3); positions[6, w] = 0.01 * (mod(w, 5) - 2)
            positions[7, w] = 0.5 + 0.05 * mod(w, 4); positions[8, w] = 1.2 + 0.05 * mod(w, 3)
            positions[9, w] = -0.4 + 0.05 * mod(w, 4); positions[10, w] = 0.98 + 0.005 * mod(w, 3)
            positions[11, w] = 0.6 + 0.05 * mod(w, 4)
        end

        f! = BN.gw_logp_and_grad(data_gpu, cfg, CuArray(freqs); model = SEOBNRv5AlignedSpinF2())
        mask = trues(nwalkers)
        grad = CUDA.zeros(Float64, 11, nwalkers)
        logp = CUDA.zeros(Float64, nwalkers)

        nok = 0
        old_reverse_flag = BN._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[]
        BN._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[] = true
        try
            for _ in 1:ncalls
                f!(grad, logp, CuArray(positions .+ 1e-7 .* randn(rng, size(positions))), mask)
                CUDA.synchronize()
                nok += 1
            end

            pos_gpu = CuArray(positions)
            g_rev = CUDA.zeros(Float64, 11, nwalkers); lp_rev = CUDA.zeros(Float64, nwalkers)
            f!(g_rev, lp_rev, pos_gpu, mask); CUDA.synchronize()
            BN._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[] = false
            g_fwd = CUDA.zeros(Float64, 11, nwalkers); lp_fwd = CUDA.zeros(Float64, nwalkers)
            f!(g_fwd, lp_fwd, pos_gpu, mask); CUDA.synchronize()

            gr = Array(g_rev); gf = Array(g_fwd)
            @test all(isfinite, gr)
            @test Array(lp_rev) == Array(lp_fwd)
            for p in 1:11
                scale = maximum(abs.(gf[p, :]))
                @test maximum(abs.(gr[p, :] .- gf[p, :])) <= 1e-9 * scale
            end
        catch e
            @test false   # a CUDA illegal-memory-access here means the device-heap leak is back
            @info "reverse-mode CUDA repeated-call regression" completed_calls = nok exception = e
        finally
            BN._EXPERIMENTAL_REVERSE_MODE_SEOBNR_GRADIENT[] = old_reverse_flag
        end
        @test nok == ncalls
    end
end
