"""
    Flows

A small normalizing flow (RealNVP affine couplings) fitted to weighted samples, for SMC independence proposals
(`smc_sample`'s `online_proposal`; the training-free local KDE's role, flowMC's "global jumps"): a density that can be
sampled and evaluated, trained between stages on the previous stage's states weighted to the new beta. Every jump is
an exact MH step against the true target, so a poor fit costs acceptance, never correctness.

Pipeline (sample space x -> latent z):
- box -> R: each row to the unit interval -- periodic rows cut at the largest circular gap of the training samples --
  then logit;
- standardization by the weighted mean / sd;
- `nlayers` affine couplings z_B = y_B exp(s(y_A)) + t(y_A) (random half masks), s, t from one MLP (2 tanh hidden
  layers) per layer, s soft-clipped to +-`smax`; the last MLP layer starts at zero (identity flow);
- N(0, I) base.
Training: Adam on the mean negative log density of mini-batches resampled by the weights; hand-written backprop
(no AD dependency; checked against finite differences in the tests).

Arrays: training, sampling and the density are array code (no scalar loops) and run on any array type:
`fit_flow(...; device)` trains on a device (e.g. `GravitationalWaveBayes.GPU.to_device`) and returns a host flow;
`flow_to_device(F, device)` gives a copy whose `flow_sample` / `flow_logpdf!` take and return device arrays. The
latent-space functions are host only.
"""
module Flows

using Random
using LinearAlgebra

export FlowDensity, fit_flow, flow_logpdf!, flow_sample, flow_to_latent, flow_from_latent, flow_latent_pull,
    flow_to_device

struct Coupling{B<:AbstractVector{Bool},M<:AbstractMatrix{Float64},V<:AbstractVector{Float64}}
    mask::B                     # true = conditioning (A) rows, passed through
    W1::M; b1::V
    W2::M; b2::V
    W3::M; b3::V                # out = [s_raw; t] (2d)
end

struct FlowDensity{V<:AbstractVector{Float64},B<:AbstractVector{Bool},C<:Coupling}
    d::Int
    lo::V; hi::V
    periodic::B
    cut::V                      # periodic rows: the angle where the period is cut (in [lo, hi))
    mu::V; sd::V
    layers::Vector{C}
    smax::Float64
end
const _HostCoupling = Coupling{Vector{Bool},Matrix{Float64},Vector{Float64}}

# copies of a layer / the whole flow with every array mapped by `f` (a device transfer, `Array`, `copy`)
_map_layer(f, L::Coupling) = Coupling(f(L.mask), f(L.W1), f(L.b1), f(L.W2), f(L.b2), f(L.W3), f(L.b3))
_map_flow(f, F::FlowDensity) = FlowDensity(F.d, f(F.lo), f(F.hi), f(F.periodic), f(F.cut), f(F.mu), f(F.sd),
                                           [_map_layer(f, L) for L in F.layers], F.smax)
"""
    flow_to_device(F, device) -> FlowDensity

A copy of `F` with its arrays on `device` (`device(::Array)`, e.g. `GravitationalWaveBayes.GPU.to_device`): `flow_sample` and
`flow_logpdf!` on it take and return device arrays. `flow_to_device(F, Array)` brings a flow back to the host.
"""
flow_to_device(F::FlowDensity, device) = _map_flow(x -> device(copy(Array(x))), F)
# an array of the device type of `v` holding the host array `A`
_like(v::AbstractVector, A::AbstractArray) = copyto!(similar(v, size(A)...), A)

const _EPS = 1e-9

# ---- box <-> R ----
# u = (x - lo) / (hi - lo) in (0, 1) (periodic rows: x wrapped to [cut, cut + P) and measured from cut), y = (logit(u) -
# mu) / sd and log |dy/dx| (row-wise sum) for every column of X; broadcasts, so any array type
function _box_to_y(F::FlowDensity, X::AbstractMatrix)
    P = F.hi .- F.lo
    U = clamp.((ifelse.(F.periodic, F.cut .+ mod.(X .- F.cut, P), X) .- ifelse.(F.periodic, F.cut, F.lo)) ./ P,
               _EPS, 1 - _EPS)
    Y = (log.(U) .- log1p.(.-U) .- F.mu) ./ F.sd
    lj = vec(sum(.-log.(U) .- log1p.(.-U) .- log.(P) .- log.(F.sd); dims=1))
    return Y, lj
end
function _y_to_box(F::FlowDensity, Y::AbstractMatrix)
    P = F.hi .- F.lo
    U = clamp.(1 ./ (1 .+ exp.(.-(Y .* F.sd .+ F.mu))), _EPS, 1 - _EPS)
    return ifelse.(F.periodic, F.lo .+ mod.(F.cut .+ U .* P .- F.lo, P), F.lo .+ U .* P)
end

# ---- couplings (columns = samples) ----
# forward y -> z for one layer; returns z, s (masked: 0 on A rows) and the cache for backprop
function _cforward(L::Coupling, Y::AbstractMatrix, smax)
    d = size(Y, 1)
    m = L.mask
    A = Y .* m
    H1 = tanh.(L.W1 * A .+ L.b1)
    H2 = tanh.(L.W2 * H1 .+ L.b2)
    O = L.W3 * H2 .+ L.b3
    Sr = O[1:d, :]; Tt = O[d+1:2d, :]
    S = smax .* tanh.(Sr ./ smax) .* .!m
    Tt = Tt .* .!m
    Z = Y .* m .+ (.!m) .* (Y .* exp.(S) .+ Tt)
    return Z, S, (A, H1, H2, S)
end
# inverse z -> y for one layer
function _cinverse(L::Coupling, Z::AbstractMatrix, smax)
    d = size(Z, 1)
    m = L.mask
    A = Z .* m                         # A rows are unchanged by the layer
    H1 = tanh.(L.W1 * A .+ L.b1)
    H2 = tanh.(L.W2 * H1 .+ L.b2)
    O = L.W3 * H2 .+ L.b3
    S = smax .* tanh.(O[1:d, :] ./ smax) .* .!m
    Tt = O[d+1:2d, :] .* .!m
    return Z .* m .+ (.!m) .* ((Z .- Tt) .* exp.(-S))
end

# log density in y space (base + log det of the couplings), per column
function _logq_y(F::FlowDensity, Y::AbstractMatrix)
    Z = Y; ld = fill!(similar(Y, size(Y, 2)), 0.0)
    for L in F.layers
        Z, S, _ = _cforward(L, Z, F.smax)
        ld .+= vec(sum(S; dims=1))
    end
    return vec(-0.5 .* sum(abs2, Z; dims=1)) .- 0.5 * F.d * log(2pi) .+ ld
end

"""
    flow_logpdf!(lq, F, X) -> lq

Log density of the flow at the columns of `X` (sample space; periodic rows any representative).
"""
function flow_logpdf!(lq::AbstractVector, F::FlowDensity, X::AbstractMatrix)
    Y, lj = _box_to_y(F, X)
    lq .= _logq_y(F, Y) .+ lj
    return lq
end

"""
    flow_sample(rng, F, n) -> Matrix (d, n)
"""
function flow_sample(rng::AbstractRNG, F::FlowDensity, n::Int)
    Z = _like(F.mu, randn(rng, F.d, n))            # (host draws: the same stream for every array type)
    for L in Iterators.reverse(F.layers)
        Z = _cinverse(L, Z, F.smax)
    end
    return _y_to_box(F, Z)
end

# ---- latent space (NeuTra: samplers run on z, x = T(z)) ----
"""
    flow_to_latent(F, X) -> Z

The latent coordinates z = T^-1(x) of the columns of `X` (the base-space point the flow maps to x).
"""
function flow_to_latent(F::FlowDensity, X::AbstractMatrix)
    Z, _ = _box_to_y(F, X)
    for L in F.layers
        Z, _, _ = _cforward(L, Z, F.smax)
    end
    return Z
end

# softplus, stable for both signs
@inline _softplus(v) = v > 0 ? v + log1p(exp(-v)) : log1p(exp(v))

"""
    flow_from_latent(F, Z) -> (X, ldj, cache)

x = T(z) for the columns of `Z`, `ldj` = log |det dx/dz| per column, and the cache `flow_latent_pull` needs. A density
pi(x) pulled back to the latent space is pi(T(z)) exp(ldj(z)); for pi = the flow's own density it is N(0, I).
"""
function flow_from_latent(F::FlowDensity, Z::AbstractMatrix)
    n = size(Z, 2)
    ld = zeros(n)
    ins = Vector{Matrix{Float64}}(undef, length(F.layers))
    Y = Matrix{Float64}(Z)
    for k in length(F.layers):-1:1
        ins[k] = Y                                  # this layer's input on the z side
        L = F.layers[k]
        Ynew = _cinverse(L, Y, F.smax)
        # log det of the inverse coupling: -sum(s(z_A)) (recomputed here; cheap next to the likelihood)
        A = Y .* L.mask
        H1 = tanh.(L.W1 * A .+ L.b1); H2 = tanh.(L.W2 * H1 .+ L.b2); O = L.W3 * H2 .+ L.b3
        S = F.smax .* tanh.(O[1:F.d, :] ./ F.smax) .* .!L.mask
        ld .-= vec(sum(S; dims=1))
        Y = Ynew
    end
    X = _y_to_box(F, Y)
    @inbounds for j in 1:n, r in 1:F.d
        v = Y[r, j] * F.sd[r] + F.mu[r]
        ld[j] += log(F.hi[r] - F.lo[r]) + log(F.sd[r]) - _softplus(v) - _softplus(-v)
    end
    return X, ld, (ins, Y)
end

"""
    flow_latent_pull(F, cache, GX) -> GZ

The latent gradient of f(T(z)) + ldj(z), given GX = df/dx at x = T(z) (the cache from `flow_from_latent`):
GZ = J' GX + d ldj / dz. Columns where GX is not finite get a zero gradient.
"""
function flow_latent_pull(F::FlowDensity, cache, GX::AbstractMatrix)
    ins, Y = cache
    d, n = size(Y)
    GY = Matrix{Float64}(undef, d, n)
    @inbounds for j in 1:n, r in 1:d
        v = Y[r, j] * F.sd[r] + F.mu[r]
        u = 1 / (1 + exp(-v))
        g = GX[r, j]
        # x = lo + P u(v), v = y sd + mu: dx/dy = P u (1 - u) sd; d/dy of log(u (1 - u)) = (1 - 2u) sd
        GY[r, j] = (isfinite(g) ? g * (F.hi[r] - F.lo[r]) * u * (1 - u) * F.sd[r] : 0.0) + (1 - 2u) * F.sd[r]
    end
    for k in 1:length(F.layers)                     # backward through the inverse couplings, x side to z side
        L = F.layers[k]; Zk = ins[k]
        m = L.mask; nm = .!m
        A = Zk .* m
        H1 = tanh.(L.W1 * A .+ L.b1); H2 = tanh.(L.W2 * H1 .+ L.b2); O = L.W3 * H2 .+ L.b3
        S = F.smax .* tanh.(O[1:d, :] ./ F.smax) .* nm
        Tt = O[d+1:2d, :] .* nm
        E = exp.(-S)
        YB = (Zk .- Tt) .* E                        # the layer's output on B rows
        # y_B = (z_B - t) e^-s, y_A = z_A; the layer adds -sum(s) to ldj
        GZ = GY .* m .+ GY .* nm .* E
        GS = (-GY .* YB .- 1) .* nm
        GT = -GY .* E .* nm
        GSr = GS .* (1 .- (S ./ F.smax) .^ 2)
        GO = vcat(GSr, GT)
        GH2 = (L.W3' * GO) .* (1 .- H2 .^ 2)
        GH1 = (L.W2' * GH2) .* (1 .- H1 .^ 2)
        GZ .+= (L.W1' * GH1) .* m
        GY = GZ
    end
    for j in 1:n
        all(isfinite, view(GX, :, j)) || (GY[:, j] .= 0)
    end
    return GY
end

# ---- training ----
# mean negative log density of the batch (y space, without the fixed box Jacobian) and its gradients w.r.t. every
# layer's parameters (same layout as the layers: a Vector of NamedTuples)
function _loss_and_grad(F::FlowDensity, Y::AbstractMatrix)
    n = size(Y, 2)
    caches = Vector{Any}(undef, length(F.layers)); ins = Vector{Any}(undef, length(F.layers))
    Z = Y; ldsum = 0.0
    for (k, L) in enumerate(F.layers)
        ins[k] = Z
        Z, S, c = _cforward(L, Z, F.smax)
        caches[k] = c
        ldsum += sum(S)
    end
    loss = (0.5 * sum(abs2, Z) - ldsum) / n + 0.5 * F.d * log(2pi)
    GZ = Z ./ n                                          # d loss / d z
    grads = Vector{Any}(undef, length(F.layers))
    for k in length(F.layers):-1:1
        L = F.layers[k]; Yk = ins[k]; (A, H1, H2, S) = caches[k]
        m = L.mask; nm = .!m
        d = size(Yk, 1)
        eS = exp.(S)
        # z_B = y_B e^s + t, loss also has -sum(s)/n
        GY = GZ .* m .+ GZ .* nm .* eS
        GS = (GZ .* Yk .* eS .- 1 / n) .* nm
        GT = GZ .* nm
        GSr = GS .* (1 .- (S ./ F.smax) .^ 2)            # s = smax tanh(sr / smax)
        GO = vcat(GSr, GT)
        gW3 = GO * H2'; gb3 = vec(sum(GO; dims=2))
        GH2 = (L.W3' * GO) .* (1 .- H2 .^ 2)
        gW2 = GH2 * H1'; gb2 = vec(sum(GH2; dims=2))
        GH1 = (L.W2' * GH2) .* (1 .- H1 .^ 2)
        gW1 = GH1 * A'; gb1 = vec(sum(GH1; dims=2))
        GA = L.W1' * GH1
        GY .+= GA .* m
        grads[k] = (; W1=gW1, b1=gb1, W2=gW2, b2=gb2, W3=gW3, b3=gb3)
        GZ = GY
    end
    return loss, grads
end

const _PFIELDS = (:W1, :b1, :W2, :b2, :W3, :b3)

"""
    fit_flow(X, w; lo, hi, periodic, nlayers=6, hidden=32, steps=2000, batch=512, lr=2e-3, smax=3.0, holdout=0.2,
             patience=5, init=nothing, device=nothing, rng) -> FlowDensity

Fit the flow to the columns of `X` (sample space, inside `[lo, hi]`; `periodic` rows wrap) with weights `w`
(normalized internally) by maximum likelihood. A random `holdout` fraction of the columns is kept out of training:
every 100 steps its weighted mean negative log density is evaluated, the best parameters are kept, and training stops
after `patience` checks without improvement (15-d fits to ~1000 samples otherwise memorize them: held-out log density
-317 vs +20 in training). `holdout = 0`: no early stopping. `init`: a fitted `FlowDensity` of the same shape whose
coupling parameters (and masks) start the training instead of the random / identity initialization (a warm start from
the previous SMC stage's flow; the box cut and the standardization are refitted to `X` as usual). `device`: train on
that device (`device(::Array)`, e.g. `GravitationalWaveBayes.GPU.to_device`; the mini-batch draws stay on the host); the returned
flow is on the host.
"""
function fit_flow(X::AbstractMatrix, w::AbstractVector; lo, hi, periodic=falses(size(X, 1)), nlayers::Int=6,
                  hidden::Int=32, steps::Int=2000, batch::Int=512, lr=2e-3, smax=3.0, holdout=0.2, patience::Int=5,
                  init::Union{Nothing,FlowDensity}=nothing, device=nothing, rng::AbstractRNG=Random.default_rng())
    d, n = size(X)
    W = Float64.(w) ./ sum(w)
    lo = Float64.(collect(lo)); hi = Float64.(collect(hi)); periodic = Bool.(collect(periodic))
    # periodic rows: cut at the middle of the largest circular gap of the samples
    cut = copy(lo)
    for r in 1:d
        periodic[r] || continue
        P = hi[r] - lo[r]
        a = sort(lo[r] .+ mod.(X[r, :] .- lo[r], P))
        gaps = vcat(diff(a), a[1] + P - a[end])
        k = argmax(gaps)
        cut[r] = lo[r] + mod(a[k] + gaps[k] / 2 - lo[r], P)
    end
    F0 = FlowDensity(d, lo, hi, periodic, cut, zeros(d), ones(d), _HostCoupling[], Float64(smax))
    Y0, _ = _box_to_y(F0, X)
    mu = Y0 * W
    sd = sqrt.(max.((Y0 .- mu) .^ 2 * W, 1e-12))
    layers = _HostCoupling[]
    warm = init !== nothing && init.d == d && length(init.layers) == nlayers && size(init.layers[1].W1) == (hidden, d)
    warm && (layers = [Coupling(copy(L.mask), copy(L.W1), copy(L.b1), copy(L.W2), copy(L.b2), copy(L.W3), copy(L.b3))
                       for L in init.layers])
    for k in 1:(warm ? 0 : nlayers)
        m = falses(d)
        m[randperm(rng, d)[1:d÷2]] .= true
        isodd(k) || (m = .!layers[end].mask)       # alternate: every row is transformed every two layers
        s1 = 1 / sqrt(d); s2 = 1 / sqrt(hidden)
        push!(layers, Coupling(collect(m), s1 .* randn(rng, hidden, d), zeros(hidden), s2 .* randn(rng, hidden, hidden),
                               zeros(hidden), zeros(2d, hidden), zeros(2d)))
    end
    Fh = FlowDensity(d, lo, hi, periodic, cut, vec(mu), vec(sd), layers, Float64(smax))
    Yall, _ = _box_to_y(Fh, X)
    dev = device === nothing ? identity : (x -> device(copy(Array(x))))
    F = device === nothing ? Fh : flow_to_device(Fh, device)
    # training / held-out split
    perm = randperm(rng, n)
    nh = holdout > 0 ? clamp(round(Int, holdout * n), 1, n - 1) : 0
    hid_idx = perm[1:nh]; tr_idx = perm[nh+1:end]
    Y = dev(Yall[:, tr_idx]); Wt = W[tr_idx] ./ sum(W[tr_idx])
    Yh = dev(Yall[:, hid_idx]); Wh = dev(nh > 0 ? W[hid_idx] ./ max(sum(W[hid_idx]), eps(Float64)) : Float64[])
    n = length(tr_idx)
    W = Wt
    # Adam
    mom = [(; (f => zero(getfield(L, f)) for f in _PFIELDS)...) for L in F.layers]   # (F: on the training device)
    vel = [(; (f => zero(getfield(L, f)) for f in _PFIELDS)...) for L in F.layers]   # (F: on the training device)
    b1, b2, aeps = 0.9, 0.999, 1e-8
    cw = cumsum(W)
    heldout() = -sum(Wh .* _logq_y(F, Yh))
    best = nh > 0 ? heldout() : Inf
    bestp = [_map_layer(copy, L) for L in F.layers]; bad = 0
    for it in 1:steps
        if nh > 0 && it % 100 == 0
            h = heldout()
            if h < best
                best = h; bestp = [_map_layer(copy, L) for L in F.layers]; bad = 0
            else
                (bad += 1) >= patience && break
            end
        end
        idx = [min(searchsortedfirst(cw, rand(rng)), n) for _ in 1:batch]
        _, g = _loss_and_grad(F, Y[:, dev(idx)])
        a = lr * sqrt(1 - b2^it) / (1 - b1^it)
        for (k, L) in enumerate(F.layers), f in _PFIELDS
            P = getfield(L, f); G = getfield(g[k], f); M = getfield(mom[k], f); V = getfield(vel[k], f)
            @. M = b1 * M + (1 - b1) * G
            @. V = b2 * V + (1 - b2) * G^2
            @. P -= a * M / (sqrt(V) + aeps)
        end
    end
    if nh > 0 && heldout() > best                     # restore the best held-out parameters
        for (k, L) in enumerate(F.layers), f in _PFIELDS
            getfield(L, f) .= getfield(bestp[k], f)
        end
    end
    return device === nothing ? F : flow_to_device(F, Array)
end

end # module
