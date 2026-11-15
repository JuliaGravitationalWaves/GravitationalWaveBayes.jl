module Calibration

export calibration_weights, calibration_factor, calibration_nodes, calibration_prior, calibration_curves

# ------------------------------------------------------------------------------------------------
# Detector calibration uncertainty, as the LVK bilby analyses model it (`bilby.gw.detector.calibration.CubicSpline`,
# bilby 1.0.4, the version behind the GWTC-2.1 release):
#
#   nodes x_k = linspace(log10 f_min, log10 f_max, n), k = 1..n (n = 10 in GWTC-2.1)
#   delta_A(f)   = interp1d(x, a,   kind = "cubic", bounds_error = False, fill_value = 0)(log10 f)
#   delta_phi(f) = interp1d(x, phi, kind = "cubic", bounds_error = False, fill_value = 0)(log10 f)
#   h_det(f) -> h_det(f) * (1 + delta_A) * (2 + i delta_phi) / (2 - i delta_phi)
#
# scipy's `interp1d(kind = "cubic")` is `make_interp_spline(k = 3)` with the default not-a-knot end conditions (NOT a
# natural spline), and is zero outside [x_1, x_n]. The spline is linear in the node values, so per detector we
# precompute W (nbins x n) once: delta_A = W a, delta_phi = W phi. The device likelihood and its gradient only need W.
# ------------------------------------------------------------------------------------------------

"""
    calibration_nodes(f_min, f_max, n) -> Vector{Float64}

The `n` spline node frequencies bilby uses: log10-spaced from `f_min` to `f_max` inclusive.
"""
calibration_nodes(f_min::Real, f_max::Real, n::Integer) = exp10.(range(log10(f_min), log10(f_max); length=n))

# Second derivatives M (n x n map from node values y: M = S y) of the not-a-knot cubic spline through nodes x.
function _notaknot_second_derivative_map(x::AbstractVector{Float64})
    n = length(x)
    n >= 4 || throw(ArgumentError("a not-a-knot cubic spline needs at least 4 nodes, got $n"))
    h = diff(x)
    A = zeros(n, n); B = zeros(n, n)          # A M = B y
    # not-a-knot at x_2: the third derivative is continuous, (M2 - M1)/h1 = (M3 - M2)/h2
    A[1, 1] = -1 / h[1]; A[1, 2] = 1 / h[1] + 1 / h[2]; A[1, 3] = -1 / h[2]
    for i in 2:n-1                             # C2 continuity at interior nodes
        A[i, i-1] = h[i-1]; A[i, i] = 2 * (h[i-1] + h[i]); A[i, i+1] = h[i]
        B[i, i-1] = 6 / h[i-1]; B[i, i] = -6 / h[i-1] - 6 / h[i]; B[i, i+1] = 6 / h[i]
    end
    # not-a-knot at x_{n-1}
    A[n, n-2] = -1 / h[n-2]; A[n, n-1] = 1 / h[n-2] + 1 / h[n-1]; A[n, n] = -1 / h[n-1]
    return A \ B
end

"""
    calibration_weights(node_freqs, freqs) -> Matrix{Float64}  (length(freqs) x length(node_freqs))

Weights W with `delta(f_j) = sum_k W[j, k] y_k` for bilby's calibration spline through the values `y_k` at
`node_freqs` (log10 abscissa, not-a-knot, zero outside the node range).
"""
function calibration_weights(node_freqs::AbstractVector{<:Real}, freqs::AbstractVector{<:Real})
    x = log10.(Float64.(node_freqs))
    issorted(x) || throw(ArgumentError("calibration node frequencies must be increasing"))
    S = _notaknot_second_derivative_map(x)
    n = length(x)
    W = zeros(length(freqs), n)
    for (j, f) in enumerate(freqs)
        t = log10(Float64(f))
        # scipy interp1d's bounds test (fill_value = 0 outside); a tiny tolerance keeps the end nodes themselves in
        (x[1] - 1e-12 <= t <= x[end] + 1e-12) || continue
        i = clamp(searchsortedlast(x, t), 1, n - 1)
        hi = x[i+1] - x[i]
        a = (x[i+1] - t) / hi; b = (t - x[i]) / hi
        # y(t) = a y_i + b y_{i+1} + ((a^3 - a) M_i + (b^3 - b) M_{i+1}) h^2 / 6
        W[j, i] += a; W[j, i+1] += b
        ca = (a^3 - a) * hi^2 / 6; cb = (b^3 - b) * hi^2 / 6
        for k in 1:n
            W[j, k] += ca * S[i, k] + cb * S[i+1, k]
        end
    end
    return W
end

"""
    calibration_factor(W, amp, phase) -> Vector{ComplexF64}

bilby's multiplicative calibration factor `(1 + delta_A) (2 + i delta_phi) / (2 - i delta_phi)` on the grid of `W`
(from [`calibration_weights`](@ref)) for spline node values `amp`, `phase`.
"""
function calibration_factor(W::AbstractMatrix, amp::AbstractVector, phase::AbstractVector)
    dA = W * amp; dphi = W * phase
    return @. (1 + dA) * (2 + im * dphi) / (2 - im * dphi)
end

"""
    calibration_prior(priors_path, detectors, freqs) -> (; nn, W, mu, sigma, nodes)

The calibration model of a bilby analysis from its prior strings (one `key = "value"` per line, as extracted
from a PE release): per detector name in `detectors` (data order), `nn` node frequencies
(`recalib_<det>_frequency_k = "DeltaFunction(peak=...)"`) and Gaussian priors (`recalib_<det>_amplitude_k`,
`..._phase_k = "Gaussian(mu=..., sigma=...)"`). `W` = [`calibration_weights`](@ref) on `freqs` (all detectors must share
the node frequencies); `mu`, `sigma` in the sampled-row order: per detector, nn amplitude then nn phase nodes.
"""
function calibration_prior(priors_path::AbstractString, detectors, freqs::AbstractVector{<:Real})
    kv = Dict{String,String}()
    for l in eachline(priors_path)
        m = match(r"^\s*(recalib_\w+)\s*=\s*\"(.*)\"\s*$", l)
        m === nothing || (kv[m.captures[1]] = m.captures[2])
    end
    num(str, key) = (m = match(Regex(key * raw"=([-+0-9.eE]+)"), str); m === nothing ?
                     throw(ArgumentError("no $key in \"$str\"")) : parse(Float64, m.captures[1]))
    nn = count(k -> startswith(k, "recalib_$(first(detectors))_frequency_"), keys(kv))
    nn >= 4 || throw(ArgumentError("calibration prior $priors_path: $nn nodes for $(first(detectors))"))
    nodes = [num(kv["recalib_$(first(detectors))_frequency_$k"], "peak") for k in 0:nn-1]
    mu = Float64[]; sigma = Float64[]
    for d in detectors
        fd = [num(kv["recalib_$(d)_frequency_$k"], "peak") for k in 0:nn-1]
        isapprox(fd, nodes; rtol=1e-12) || throw(ArgumentError("calibration nodes differ between detectors"))
        for part in ("amplitude", "phase"), k in 0:nn-1
            v = kv["recalib_$(d)_$(part)_$k"]
            startswith(v, "Gaussian(") || throw(ArgumentError("recalib_$(d)_$(part)_$k: Gaussian prior expected, got $v"))
            push!(mu, num(v, "mu")); push!(sigma, num(v, "sigma"))
        end
    end
    return (; nn, W=calibration_weights(nodes, freqs), mu, sigma, nodes)
end

"""
    calibration_curves(cal, ncurves; rng, ndet = length(cal.mu) ÷ (2 cal.nn)) -> (; C, Z)

`ncurves` calibration response curves drawn from the prior `cal` ([`calibration_prior`](@ref)), for calibration
marginalization (Payne et al. 2020; bilby's `calibration_marginalization`, `number_of_response_curves` = 1000): node
values `mu + sigma z` with `z ~ N(0, 1)` (`Z`, `length(mu) x ncurves`, kept to reconstruct the calibration posterior),
and `C[k, (d - 1) nbins + i]` the factor of curve `k` at bin `i` of detector `d` (`ncurves x ndet nbins`, complex).
"""
function calibration_curves(cal, ncurves::Integer; rng, ndet::Integer = length(cal.mu) ÷ (2 * cal.nn))
    nn = cal.nn; nb = size(cal.W, 1)
    length(cal.mu) == 2 * nn * ndet || throw(ArgumentError("calibration prior has $(length(cal.mu)) rows, expected $(2nn * ndet)"))
    Z = randn(rng, length(cal.mu), ncurves)
    C = Matrix{ComplexF64}(undef, ncurves, ndet * nb)
    for k in 1:ncurves, d in 1:ndet
        o = (d - 1) * 2 * nn
        node = cal.mu[o+1:o+2nn] .+ cal.sigma[o+1:o+2nn] .* Z[o+1:o+2nn, k]
        C[k, (d-1)*nb+1:d*nb] .= calibration_factor(cal.W, node[1:nn], node[nn+1:2nn])
    end
    return (; C, Z)
end

end
