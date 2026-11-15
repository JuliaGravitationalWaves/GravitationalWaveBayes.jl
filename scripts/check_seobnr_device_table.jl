# SEOBNR_GPU_PLAN.md M1: the device-built SEOBNRv5 IMR table (`seobnr_device_tables!`) vs the CPU
# `GravitationalWaveforms._eobimr_coeffs`, field by field, for many random in-prior binaries, plus timing.
#   julia --project=<env with CUDA> scripts/check_seobnr_device_table.jl [cuda|cpu] [nwalkers...]
using GravitationalWaveBayes, KernelAbstractions, Random, Printf, StaticArrays
import GravitationalWaveforms
const BN = GravitationalWaveBayes.BatchedNUTS
const W = GravitationalWaveforms
const M = W.SEOBNRv5IMRAlignedSpinF2()

which = isempty(ARGS) ? "cpu" : ARGS[1]
sizes = length(ARGS) > 1 ? parse.(Int, ARGS[2:end]) : [64]
if which == "cuda"
    using CUDA
    backend = CUDA.CUDABackend()
    println("device: ", CUDA.name(CUDA.device()))
else
    backend = KernelAbstractions.CPU()
    println("KernelAbstractions CPU backend, ", Threads.nthreads(), " threads")
end
flush(stdout)

# Broad prior box around GW150914 plus a wider-mass set (heavier/lighter, larger spins).
function positions(nw; seed = 1, narrow = false)
    rng = Random.Xoshiro(seed)
    P = zeros(4, nw)
    for w in 1:nw
        wide = isodd(w) && !narrow
        P[1, w] = wide ? 12.0 + 40.0 * rand(rng) : 26.0 + 10.0 * rand(rng)
        P[2, w] = wide ? 0.12 + 0.1299 * rand(rng) : 0.20 + 0.0499 * rand(rng)
        P[3, w] = wide ? -0.9 + 1.8 * rand(rng) : -0.5 + rand(rng)
        P[4, w] = wide ? -0.9 + 1.8 * rand(rng) : -0.5 + rand(rng)
    end
    P
end

relerr(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), 1e-300)
const VFIELDS = (:ts, :phis, :omegas, :Qs, :dQs, :ft_f, :ft_amp, :ft_damp, :ft_arg, :ft_darg)
const SFIELDS = (:M_sec, :eta, :M, :fmin, :fmax, :fspa_max, :f_b1, :f_b2, :t_a, :ft_shift)

for nw in sizes
    P = positions(nw)
    T = BN.SEOBNRDeviceTables(backend, nw)
    t0 = time(); BN.seobnr_device_tables!(T, P); el = time() - t0
    @printf("\n=== %d walkers: first call %.1f s (incl. compile)\n", nw, el); flush(stdout)
    ncheck = min(nw, 200)
    worst = Dict{Symbol,Float64}(); nbad = 0; nsucc = 0; nnreal = 0
    for w in 1:ncheck
        c = W._eobimr_coeffs(M, P[1, w], P[2, w], P[3, w], P[4, w])
        d = BN.seobnr_device_table_host(T, w)
        nsucc += c.success
        if c.success != d.success || c.nreal != d.nreal
            nbad += 1
            nbad <= 5 && @printf("  walker %d: success %s/%s nreal %d/%d  pos %s\n", w, c.success, d.success, c.nreal,
                d.nreal, string(round.(P[:, w]; digits = 3)))
            continue
        end
        c.success || continue
        for f in VFIELDS
            worst[f] = max(get(worst, f, 0.0), relerr(getfield(d, f), getfield(c, f)))
        end
        for f in SFIELDS
            worst[f] = max(get(worst, f, 0.0), abs(getfield(d, f) - getfield(c, f)) / max(abs(getfield(c, f)), 1e-300))
        end
    end
    @printf("  checked %d walkers (%d successful on CPU): success/nreal mismatches %d\n", ncheck, nsucc, nbad)
    for f in (VFIELDS..., SFIELDS...)
        @printf("    %-9s max rel err %.2e\n", f, get(worst, f, NaN))
    end
    flush(stdout)
    # serial (bitwise) FFT vs workgroup FFT on the same walkers
    Ts = BN.SEOBNRDeviceTables(backend, nw); BN.seobnr_device_tables!(Ts, P; serial_fft = true)
    fe = 0.0
    for w in 1:ncheck
        a = BN.seobnr_device_table_host(T, w); b = BN.seobnr_device_table_host(Ts, w)
        a.success || continue
        for f in (:ft_amp, :ft_arg, :ft_darg)
            fe = max(fe, relerr(getfield(a, f), getfield(b, f)))
        end
    end
    @printf("  workgroup FFT vs serial FFT: max rel err (ft_amp/arg/darg) %.2e
", fe)
    # timing on GW150914-like walkers only (the pass-2 regime)
    P = positions(nw; seed = 2, narrow = true)
    BN.seobnr_device_tables!(T, P)
    tm = Dict{Symbol,Float64}()
    ncall = 5
    tt = @elapsed for _ in 1:ncall
        BN.seobnr_device_tables!(T, P; timers = tm)
    end
    @printf("  [GW150914-like walkers] %d calls: %.2f ms per call = %.5f ms per walker (stage-timed, synchronizing)\n", ncall, 1000tt / ncall,
        1000tt / ncall / nw)
    for k in (:base, :pa, :plunge, :nodes, :build, :spa, :segment, :fft, :fourier)
        @printf("    %-8s %.3f ms per call\n", k, 1000 * get(tm, k, 0.0) / ncall)
    end
    tt2 = @elapsed for _ in 1:ncall
        BN.seobnr_device_tables!(T, P)
    end
    @printf("  untimed: %.2f ms per call = %.5f ms per walker\n", 1000tt2 / ncall, 1000tt2 / ncall / nw)
    flush(stdout)
end
println("\nDONE")
