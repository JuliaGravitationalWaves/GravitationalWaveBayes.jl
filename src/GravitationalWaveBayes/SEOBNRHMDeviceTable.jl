# ------------------------------------------------------------------------------------------------
# SEOBNRv5HM frequency-domain tables on a KernelAbstractions device, walker x node parallel.
# Included inside `BatchedNUTS`, after `SEOBNRDeviceTable.jl`.
#
# The (2,2) device stages up to the attachment build (`seobnr_device_tables!(...; build_only = true)`: base, PA,
# plunge, node amplitudes, t_ISCO / t_attach / (2,2) NQC + MR), then, per walker, exactly what
# `GravitationalWaveforms._seobv5hm_coeffs!` computes, with GravitationalWaveforms' own scalar code on views of the device arrays:
#   attachment states, special coefficients, per-mode NQC / MR / mixing (`_seobv5hm_attach`, walker)
#   -> reduced modes at the dynamics nodes (node) -> their time derivatives (node)
#   -> per-mode SPA tables (node) + padding (node) -> band edges and Fourier windows (walker)
#   -> per mode: segment samples split even / odd (sample) -> two 2048-point workgroup FFTs each (the (2,2) kernel)
#      -> radix-2 combination into the mode's 4096-point transform on a window of `_SDH_NFH` bins (bin).
# Forward mode: as the (2,2) table, every stage with 4 tangent directions (`TT`).
# ------------------------------------------------------------------------------------------------

const _SDH_NM = GravitationalWaveforms._SEOBV5HM_NMODES
const _SDH_M = GravitationalWaveforms._SEOBV5HM_M
const _SDH_NSEG = GravitationalWaveforms._SEOBV5HM_NSEG          # 4096 segment samples per mode (= 2 x _SDT_NS)
const _SDH_NFH = 1024                                  # Fourier bins kept per mode, from the window start j0
const _SDH_TBLEND1 = GravitationalWaveforms._EOBIMR_TBLEND[1]
@assert _SDH_NSEG == 2 * _SDT_NS

# The dense-output NamedTuple of the reduced modes `_seobv5hm_at` reads, as (node, mode) views of walker w.
@inline _sdh_modes(HD, w) = (A = view(HD.A, :, :, w), dA = view(HD.dA, :, :, w), th = view(HD.th, :, :, w),
    dth = view(HD.dth, :, :, w))

# Attachment states, special coefficients, per-mode NQC, merger-ringdown and mixing.
@inline function _sdh_b_attach!(w, HA, D, DM, BLD, B)
    @inbounds begin
        Bw = (base = B[w], dyn = _sdt_dyn(D, w, DM[w]), t_a = BLD[w].t_a)
        HA[w] = _SDT_W._seobv5hm_attach(Bw)
    end
    return nothing
end

# Signed amplitude and smooth phase of the six reduced modes at node i (`_seobv5hm_node_modes!`); zero beyond n.
@inline function _sdh_b_nodes!(i, w, HD, D, DM, B, HA)
    @inbounds begin
        n = DM[w].n
        if i <= n
            b = B[w]; a = HA[w]
            pp = D.pphi[i, w]; om = D.dphi[i, w]
            H, Heff, om_circ = _SDT_W._seobv5hm_state(D.r[i, w], D.pr[i, w], pp, b)
            Base.Cartesian.@nexprs 6 k -> begin
                A_k, th_k = _SDT_W._seobv5hm_mode_ap(k, om, om_circ, H, Heff, pp, b, a.pref[k], a.c21, a.c43)
                HD.A[i, k, w] = A_k; HD.th[i, k, w] = th_k
            end
        else
            Base.Cartesian.@nexprs 6 k -> (HD.A[i, k, w] = 0.0; HD.th[i, k, w] = 0.0)
        end
    end
    return nothing
end

# Their time derivatives (3-point non-uniform FD, `_seobv5hm_node_deriv`).
@inline function _sdh_b_dnodes!(i, w, HD, D, DM)
    @inbounds begin
        n = DM[w].n
        if i <= n
            t = view(D.t, :, w)
            Base.Cartesian.@nexprs 6 k -> begin
                HD.dA[i, k, w] = _SDT_W._seobv5hm_node_deriv(t, view(HD.A, :, k, w), i, n)
                HD.dth[i, k, w] = _SDT_W._seobv5hm_node_deriv(t, view(HD.th, :, k, w), i, n)
            end
        else
            Base.Cartesian.@nexprs 6 k -> (HD.dA[i, k, w] = 0.0; HD.dth[i, k, w] = 0.0)
        end
    end
    return nothing
end

# SPA node times (the (2,2) layout of `BLD`: PA nodes up to npa_use, then uniform up to t_top) and, per mode, the
# phase and frequency over m, R = A / (varphi'/m)^((l+eps)/3) and its centred difference (`_seobv5hm_coeffs!`).
@inline function _sdh_b_spa!(i, w, HT, HD, D, DM, BLD, HA)
    @inbounds begin
        bl = BLD[w]
        nspa = bl.nspa
        if i <= nspa
            dyn = _sdt_dyn(D, w, DM[w]); Dm = _sdh_modes(HD, w); nqc = HA[w].nqc
            npa_use = bl.npa_use
            tq = i <= npa_use ? dyn.t[i] :
                 bl.t_seg0 + (bl.t_top - bl.t_seg0) * Float64(i - npa_use) / Float64(nspa - npa_use)
            HT.ts[i, w] = tq
            Base.Cartesian.@nexprs 6 k -> begin
                m_k = Float64(_SDH_M[k])
                _, vp_k = _SDT_W._seobv5hm_insp_ap(dyn, Dm, nqc, k, tq)
                HT.phis[i, k, w] = vp_k / m_k
                HT.oms[i, k, w] = _SDT_W._seobv5hm_insp_freq(dyn, Dm, nqc, k, tq) / m_k
                HT.Qs[i, k, w] = _SDT_W._seobv5hm_R(dyn, Dm, nqc, k, tq)     # R and dR/dt; Q in `_sdh_b_spa_finish!`
                HT.dQs[i, k, w] = _SDT_W._seobv5hm_R(dyn, Dm, nqc, k, tq + 0.5) - _SDT_W._seobv5hm_R(dyn, Dm, nqc, k, tq - 0.5)
            end
        end
    end
    return nothing
end

# Padding nodes, and R -> Q on the real ones, per mode (as `_sdt_b_spa_finish!` / `GravitationalWaveforms._seobv5hm_coeffs!`).
@inline function _sdh_b_spa_finish!(i, w, HT, BLD)
    @inbounds begin
        nspa = BLD[w].nspa
        if i > nspa
            HT.ts[i, w] = HT.ts[nspa, w] + Float64(i - nspa)
            Base.Cartesian.@nexprs 6 k -> begin
                HT.oms[i, k, w] = HT.oms[nspa, k, w] + Float64(i - nspa)
                HT.phis[i, k, w] = 0.0; HT.Qs[i, k, w] = 0.0; HT.dQs[i, k, w] = 0.0
            end
        else
            ts = view(HT.ts, :, w)
            Base.Cartesian.@nexprs 6 k -> begin
                HT.Qs[i, k, w], HT.dQs[i, k, w] = _SDT_W._spa_node_q(ts, view(HT.oms, :, k, w), HT.Qs[i, k, w],
                    HT.dQs[i, k, w], i, nspa)
            end
        end
    end
    return nothing
end


# Band edges per mode (`_seobv5hm_coeffs!`), the Fourier segment layout and each mode's bin window: bins
# j0 + 1 .. j0 + _SDH_NFH of the 4096-point transform (0-based bin index j0 + j - 1), starting at 0.8 f_b1.
@inline function _sdh_b_band!(w, HS, HT, D, DM, BLD, B, HA, HD)
    @inbounds begin
        bl = BLD[w]; b = B[w]; a = HA[w]
        Ms = b.M_sec
        nspa = bl.nspa
        dyn = _sdt_dyn(D, w, DM[w]); Dm = _sdh_modes(HD, w)
        t_a = bl.t_a
        t_w = max(bl.t_start, t_a - _SDT_TSEG_PRE)
        Lt = min(_SDT_TTAPER, 0.4 * (t_a - t_w))
        t_e = t_a + _SDT_NTAU * a.mr[1].tau
        dts = (t_e - t_w) / Float64(_SDH_NSEG - 1)
        df = 1.0 / (Float64(_SDH_NSEG) * dts * Ms)
        fmin = ntuple(k -> _SDH_M[k] * HT.oms[1, k, w] / (2pi * Ms), Val(6))
        fspa_max = ntuple(k -> _SDH_M[k] * HT.oms[nspa, k, w] / (2pi * Ms), Val(6))
        f_b1 = bl.use_spa ? ntuple(k -> _SDT_W._seobv5hm_insp_freq(dyn, Dm, a.nqc, k, t_a - _SDH_TBLEND1) / (2pi * Ms), Val(6)) : fmin
        f_b2 = bl.use_spa ? fspa_max : fmin
        fmax = ntuple(k -> _SDT_FHI_QNM * a.mr[k].wr / (2pi * Ms), Val(6))
        # = clamp(floor(Int, x), 0, NSEG/2 - NFH) for finite x, without the throwing conversion (see `_sdt_pa_meta`)
        j0 = ntuple(k -> clamp(unsafe_trunc(Int, floor(clamp(0.8 * f_b1[k] / df, 0.0, Float64(div(_SDH_NSEG, 2) - _SDH_NFH)))), 0,
            div(_SDH_NSEG, 2) - _SDH_NFH), Val(6))
        covered = true
        for k in 1:6
            covered &= (fmax[k] < 0.45 / (dts * Ms)) & (Float64(j0[k] + _SDH_NFH - 1) * df >= fmax[k])
        end
        HS[w] = (t_a = t_a, t_w = t_w, Lt = Lt, dts = dts, df = df, fmin = fmin, fspa_max = fspa_max, f_b1 = f_b1, f_b2 = f_b2,
            fmax = fmax, j0 = j0, success = bl.success & covered)
    end
    return nothing
end

# Mode K's tapered segment samples z_n = taper sg Ms dts h(t_n) (`_seobv5hm_coeffs!`, h = `_seobv5hm_full`, internal
# convention) and (t_n - t_a) z_n, n = 2(q - 1) (even, column 2w - 1) and 2(q - 1) + 1 (odd, column 2w).
@inline function _sdh_b_seg!(q, w, SG, HS, D, DM, B, HA, HD, ::Val{K}) where {K}
    @inbounds begin
        hs = HS[w]; a = HA[w]; t_a = hs.t_a; Ms = B[w].M_sec
        S = (t_a = t_a, dyn = _sdt_dyn(D, w, DM[w]), D = _sdh_modes(HD, w), nqc = a.nqc, mr = a.mr, mix = a.mix)
        sg = _SDT_W._seobv5hm_fdsign(K)
        for par in 0:1
            n = 2 * (q - 1) + par
            tn = hs.t_w + hs.dts * Float64(n)
            h = _SDT_W._seobv5hm_full(S, K, tn)
            wt = _sdt_turn_on((tn - hs.t_w) / hs.Lt) * sg * Ms * hs.dts
            col = 2 * (w - 1) + 1 + par
            zr = wt * real(h); zi = wt * imag(h)
            SG.sre[q, col] = zr; SG.sim[q, col] = zi
            SG.tre[q, col] = (tn - t_a) * zr; SG.tim[q, col] = (tn - t_a) * zi
        end
    end
    return nothing
end

# The same samples one per thread (sample n1 - 1), and only those on one side of t_attach (`MR`: the merger-ringdown
# samples, dt > 0, else the inspiral ones, as `_seobv5hm_full` splits them), so that each kernel holds the registers of
# one branch (`seg_split`).
@inline function _sdh_b_seg1!(n1, w, SG, HS, D, DM, B, HA, HD, ::Val{K}, ::Val{MR}) where {K,MR}
    @inbounds begin
        hs = HS[w]; t_a = hs.t_a
        n = n1 - 1
        tn = hs.t_w + hs.dts * Float64(n)
        if (tn - t_a > 0.0) == MR
            a = HA[w]; Ms = B[w].M_sec
            h = if MR
                _SDT_W._seobv5hm_full_mr((mr = a.mr, mix = a.mix), K, tn - t_a)
            else
                _SDT_W._seobv5hm_insp(_sdt_dyn(D, w, DM[w]), _sdh_modes(HD, w), a.nqc, K, tn)
            end
            wt = _sdt_turn_on((tn - hs.t_w) / hs.Lt) * _SDT_W._seobv5hm_fdsign(K) * Ms * hs.dts
            q = (n >> 1) + 1
            col = 2 * (w - 1) + 1 + (n & 1)
            zr = wt * real(h); zi = wt * imag(h)
            SG.sre[q, col] = zr; SG.sim[q, col] = zi
            SG.tre[q, col] = (tn - t_a) * zr; SG.tim[q, col] = (tn - t_a) * zi
        end
    end
    return nothing
end

# Bin j of mode K's window: the 4096-point +i transform from its even / odd halves, X = E + e^{2 pi i jj / 4096} O,
# then the table entries of `_seobv5hm_coeffs!` (conjugate, phase reference t_a, d/df by the (t - t_a) transform).
@inline function _sdh_b_bins!(j, w, FT, SG, HS, B, ::Val{K}) where {K}
    @inbounds begin
        hs = HS[w]; Ms = B[w].M_sec
        jj = hs.j0[K] + j - 1                         # 0-based bin index
        ce = 2 * (w - 1) + 1; co = ce + 1
        ang = 2pi * Float64(jj) / Float64(_SDH_NSEG)
        cw, sw = cos(ang), sin(ang)
        Er, Ei = SG.sre[jj + 1, ce], SG.sim[jj + 1, ce]; Or, Oi = SG.sre[jj + 1, co], SG.sim[jj + 1, co]
        Fr = Er + cw * Or - sw * Oi; Fi = Ei + cw * Oi + sw * Or
        Er, Ei = SG.tre[jj + 1, ce], SG.tim[jj + 1, ce]; Or, Oi = SG.tre[jj + 1, co], SG.tim[jj + 1, co]
        Gr = Er + cw * Or - sw * Oi; Gi = Ei + cw * Oi + sw * Or
        f = Float64(jj) * hs.df
        th = -2pi * f * Ms * (hs.t_w - hs.t_a)
        c, s_ = cos(th), sin(th)
        Xr, Xi = Fr, -Fi
        Vr, Vi = Gr, -Gi
        FT.re[j, K, w] = Xr * c - Xi * s_; FT.im[j, K, w] = Xr * s_ + Xi * c
        Yr = Vr * c - Vi * s_; Yi = Vr * s_ + Vi * c
        FT.dre[j, K, w] = 2pi * Ms * Yi; FT.dim[j, K, w] = -2pi * Ms * Yr
    end
    return nothing
end

# ---------------------------------------------------------------------------------------------- evaluation
# Walker w's tables as the NamedTuple `GravitationalWaveforms._seobv5hm_spa_eval` reads ((node, mode) views) plus the windowed
# Fourier tables (`j0`: each mode's first kept 0-based bin).
@inline function _sdh_tab(HS, HT, FT, BLD, B, w)
    hs = HS[w]; bl = BLD[w]; b = B[w]
    return (success = hs.success, M_sec = b.M_sec, t_a = hs.t_a, nreal = bl.nspa, ts = view(HT.ts, :, w),
        phis = view(HT.phis, :, :, w), omegas = view(HT.oms, :, :, w), Qs = view(HT.Qs, :, :, w), dQs = view(HT.dQs, :, :, w),
        fmin = hs.fmin, fspa_max = hs.fspa_max, f_b1 = hs.f_b1, f_b2 = hs.f_b2, fmax = hs.fmax, df = hs.df, j0 = hs.j0,
        ft_re = view(FT.re, :, :, w), ft_im = view(FT.im, :, :, w), ft_dre = view(FT.dre, :, :, w), ft_dim = view(FT.dim, :, :, w))
end

# `GravitationalWaveforms._seobv5hm_ft_eval` on the window: bin j (0-based) is row j - j0 + 1.
@inline function _sdh_ft_eval(c, k, f)
    x = f / c.df
    j = clamp(unsafe_trunc(Int, floor(clamp(x, 0.0, 1.0e9))), c.j0[k], c.j0[k] + _SDH_NFH - 2)
    s = x - j
    jj = j - c.j0[k] + 1
    re, _ = _SDT_W._eobimr_hermite(c.ft_re[jj, k], c.ft_dre[jj, k], c.ft_re[jj + 1, k], c.ft_dre[jj + 1, k], c.df, s)
    im_, _ = _SDT_W._eobimr_hermite(c.ft_im[jj, k], c.ft_dim[jj, k], c.ft_im[jj + 1, k], c.ft_dim[jj + 1, k], c.df, s)
    fx = c.fmax[k]
    tap = 1.0 - _SDT_W._eobimr_smoothstep((f - 0.8 * fx) / (0.2 * fx))
    return tap * re, tap * im_
end

# `GravitationalWaveforms._seobv5hm_fd_eval` (reduced h_{l,-m}(f) of mode k, seconds) as (re, im) on the device tables.
@inline function _sdh_fd_eval(c, k, f)
    (!c.success || f < c.fmin[k] || f > c.fmax[k]) && return 0.0, 0.0
    if f <= c.f_b1[k]
        z = _SDT_W._seobv5hm_spa_eval(c, k, f)
        return real(z), imag(z)
    elseif f >= c.f_b2[k]
        return _sdh_ft_eval(c, k, f)
    end
    wb = _SDT_W._eobimr_smoothstep((f - c.f_b1[k]) / (c.f_b2[k] - c.f_b1[k]))
    z = _SDT_W._seobv5hm_spa_eval(c, k, f)
    fr, fi = _sdh_ft_eval(c, k, f)
    return (1.0 - wb) * real(z) + wb * fr, (1.0 - wb) * imag(z) + wb * fi
end

# ---------------------------------------------------------------------------------------------- buffers + driver
"""
    SEOBNRHMDeviceTables(backend, nwalkers)

Device buffers for [`seobnr_hm_device_tables!`](@ref): the (2,2) dynamics stages (`T22`, a
[`SEOBNRDeviceTables`](@ref)), the per-walker attachment records `HA`, the reduced modes at the dynamics nodes `HD`
((node, mode, walker)) and the per-mode SPA tables `HT`.
"""
struct SEOBNRHMDeviceTables{BK,T2,VA,A3,A2,VS}
    # concrete: with `backend::Any`, inference of the driver could not prune the KA CPU path, and Enzyme's host
    # thunk generator compiled CPU forward derivatives of every HM stage at the first call (~40 min on the V100
    # box, outside the kernel cache)
    backend::BK
    nw::Int
    T22::T2
    HA::VA
    HD::NamedTuple{(:A, :dA, :th, :dth),NTuple{4,A3}}
    HT::NamedTuple{(:ts, :phis, :oms, :Qs, :dQs),Tuple{A2,A3,A3,A3,A3}}
    HS::VS                                                  # per-walker band edges, segment layout, bin windows
    SG::NamedTuple{(:sre, :sim, :tre, :tim),NTuple{4,A2}}   # one mode's segment halves, (2048, 2 nw), reused per mode
    FT::NamedTuple{(:re, :im, :dre, :dim),NTuple{4,A3}}     # Fourier tables, (bin in window, mode, walker)
end

const _SDH_HS0 = (t_a = 0.0, t_w = 0.0, Lt = 0.0, dts = 0.0, df = 0.0, fmin = ntuple(_ -> 0.0, 6),
    fspa_max = ntuple(_ -> 0.0, 6), f_b1 = ntuple(_ -> 0.0, 6), f_b2 = ntuple(_ -> 0.0, 6), fmax = ntuple(_ -> 0.0, 6),
    j0 = ntuple(_ -> 0, 6), success = false)

# Zero prototype of the attachment record (`_seobv5hm_attach` on a CPU dynamics build).
function _sdh_attach_proto()
    B = GravitationalWaveforms._eobimr_build(GravitationalWaveforms.SEOBNRv5IMRAlignedSpinF2(), 30.0, 0.24, 0.1, 0.0)
    return _sdt_zero(GravitationalWaveforms._seobv5hm_attach(B))
end

function SEOBNRHMDeviceTables(backend, nw::Int)
    z(dims...) = KernelAbstractions.zeros(backend, Float64, dims...)
    a0 = _sdh_attach_proto()
    HA = KernelAbstractions.allocate(backend, typeof(a0), nw); fill!(HA, a0)
    HD = (A = z(_SDT_ND, _SDH_NM, nw), dA = z(_SDT_ND, _SDH_NM, nw), th = z(_SDT_ND, _SDH_NM, nw),
        dth = z(_SDT_ND, _SDH_NM, nw))
    HT = (ts = z(_SDT_NT, nw), phis = z(_SDT_NT, _SDH_NM, nw), oms = z(_SDT_NT, _SDH_NM, nw),
        Qs = z(_SDT_NT, _SDH_NM, nw), dQs = z(_SDT_NT, _SDH_NM, nw))
    HS = KernelAbstractions.allocate(backend, typeof(_SDH_HS0), nw); fill!(HS, _SDH_HS0)
    SG = (sre = z(_SDT_NS, 2nw), sim = z(_SDT_NS, 2nw), tre = z(_SDT_NS, 2nw), tim = z(_SDT_NS, 2nw))
    FT = (re = z(_SDH_NFH, _SDH_NM, nw), im = z(_SDH_NFH, _SDH_NM, nw), dre = z(_SDH_NFH, _SDH_NM, nw),
        dim = z(_SDH_NFH, _SDH_NM, nw))
    return SEOBNRHMDeviceTables(backend, nw, SEOBNRDeviceTables(backend, nw), HA, HD, HT, HS, SG, FT)
end

"""
    seobnr_hm_device_tables!(T::SEOBNRHMDeviceTables, pos; f_min = 20.0, TT = nothing, timers = nothing, table_kw...)

Build the SEOBNRv5HM tables of all walkers on the device; `pos` as for [`seobnr_device_tables!`](@ref) (4 x nwalkers:
chirp mass, eta, chi1, chi2). `TT`: four more `SEOBNRHMDeviceTables` whose `T22.POS` hold tangent directions (forward
mode, every buffer's tangent in `TT[k]`). `dirpar_nodes` / `dirpar_attach`: `_sdt_run!`'s `dirpar` for the node stages /
the attachment stage (default: 4 threads per walker up to 2048 walkers, 2 above). `table_kw` go to the (2,2) dynamics stages.
"""
function seobnr_hm_device_tables!(T::SEOBNRHMDeviceTables, pos; f_min = 20.0, TT = nothing, timers = nothing,
        dirpar_nodes = false, dirpar_attach = nothing, seg_split = true, table_kw...)
    be = T.backend; nw = T.nw
    fwd = TT !== nothing
    tg(f) = ntuple(k -> f(TT[k]), 4)
    sh(xs...) = fwd ? xs : nothing
    seobnr_device_tables!(T.T22, pos; f_min = f_min, timers = timers, build_only = true,
        TT = fwd ? tg(t -> t.T22) : nothing, table_kw...)
    t0 = Ref(time_ns())
    lap!(k) = timers === nothing || (KernelAbstractions.synchronize(be); t = time_ns();
        timers[k] = get(timers, k, 0.0) + (t - t0[]) / 1e9; t0[] = t)
    S = T.T22
    D = fwd ? tg(t -> t.T22.D) : nothing
    B = fwd ? tg(t -> t.T22.B) : nothing
    BLD = fwd ? tg(t -> t.T22.BLD) : nothing
    HA = fwd ? tg(t -> t.HA) : nothing
    HD = fwd ? tg(t -> t.HD) : nothing
    HT = fwd ? tg(t -> t.HT) : nothing
    # attachment stage (serial RK4 re-integration per walker) with the tangent directions on 4 (2) threads per walker:
    # V100: 1024 walkers 159 -> 79 ms (2 threads: 102), 4096: 207 -> 186 (2 threads: 174)
    dirpar_attach === nothing && (dirpar_attach = nw <= GPU.tuning(:SDH_ATTACH_DIRPAR_MAX, 2048) ? true : 2)
    _sdt_run!(be, _sdh_b_attach!, nw, (T.HA, S.D, S.DM, S.BLD, S.B), sh(HA, D, nothing, BLD, B); dirpar = dirpar_attach)
    lap!(:hm_attach)
    _sdt_run!(be, _sdh_b_nodes!, (_SDT_ND, nw), (T.HD, S.D, S.DM, S.B, T.HA), sh(HD, D, nothing, B, HA);
        dirpar = dirpar_nodes)
    _sdt_run!(be, _sdh_b_dnodes!, (_SDT_ND, nw), (T.HD, S.D, S.DM), sh(HD, D, nothing); dirpar = dirpar_nodes)
    lap!(:hm_nodes)
    _sdt_run!(be, _sdh_b_spa!, (_SDT_NT, nw), (T.HT, T.HD, S.D, S.DM, S.BLD, T.HA),
        sh(HT, HD, D, nothing, BLD, HA); dirpar = dirpar_nodes)
    _sdt_run!(be, _sdh_b_spa_finish!, (_SDT_NT, nw), (T.HT, S.BLD), sh(HT, BLD); dirpar = dirpar_nodes)
    lap!(:hm_spa)
    HS = fwd ? tg(t -> t.HS) : nothing
    SG = fwd ? tg(t -> t.SG) : nothing
    FT = fwd ? tg(t -> t.FT) : nothing
    _sdt_run!(be, _sdh_b_band!, nw, (T.HS, T.HT, S.D, S.DM, S.BLD, S.B, T.HA, T.HD), sh(HS, HT, D, nothing, BLD, B, HA, HD))
    lap!(:hm_band)
    for K in 1:_SDH_NM
        if seg_split
            for mr in (Val(false), Val(true))
                _sdt_run!(be, _sdh_b_seg1!, (_SDH_NSEG, nw), (T.SG, T.HS, S.D, S.DM, S.B, T.HA, T.HD, Val(K), mr),
                    sh(SG, HS, D, nothing, B, HA, HD, nothing, nothing); dirpar = dirpar_nodes)
            end
        else
            _sdt_run!(be, _sdh_b_seg!, (_SDT_NS, nw), (T.SG, T.HS, S.D, S.DM, S.B, T.HA, T.HD, Val(K)),
                sh(SG, HS, D, nothing, B, HA, HD, nothing); dirpar = dirpar_nodes)
        end
        lap!(:hm_seg)
        # the FFT is linear: tangents are transformed like the primal
        for G in (fwd ? (T.SG, SG...) : (T.SG,))
            _sdt_fft_shared_kernel!(be, _SDT_FFT_WG)(G.sre, G.sim, G.tre, G.tim; ndrange = _SDT_FFT_WG * 2 * (2nw))
        end
        lap!(:hm_fft)
        _sdt_run!(be, _sdh_b_bins!, (_SDH_NFH, nw), (T.FT, T.SG, T.HS, S.B, Val(K)), sh(FT, SG, HS, B, nothing);
            dirpar = dirpar_nodes)
        lap!(:hm_bins)
    end
    KernelAbstractions.synchronize(be)
    return T
end

"""
    seobnr_hm_device_spa_host(T::SEOBNRHMDeviceTables, w) -> NamedTuple

Walker `w`'s SPA tables (`ts`, `phis`, `omegas`, `Qs`, `dQs`, `nreal`) copied back, for validation against
`GravitationalWaveforms._seobv5hm_coeffs!`.
"""
function seobnr_hm_device_spa_host(T::SEOBNRHMDeviceTables, w::Int)
    HT = T.HT
    return (nreal = Array(T.T22.BLD)[w].nspa, ts = Array(view(HT.ts, :, w)), phis = Array(view(HT.phis, :, :, w)),
        omegas = Array(view(HT.oms, :, :, w)), Qs = Array(view(HT.Qs, :, :, w)), dQs = Array(view(HT.dQs, :, :, w)))
end

"""
    seobnr_hm_device_ft_host(T::SEOBNRHMDeviceTables, w) -> NamedTuple

Walker `w`'s band edges, bin windows and Fourier tables (`re`, `im`, `dre`, `dim`: (bin in window, mode); row j is the
0-based bin `j0[k] + j - 1`, i.e. row `j0[k] + j` of `GravitationalWaveforms._seobv5hm_coeffs`' `ft_*`), copied back.
"""
function seobnr_hm_device_ft_host(T::SEOBNRHMDeviceTables, w::Int)
    hs = Array(T.HS)[w]
    FT = T.FT
    return merge(hs, (re = Array(view(FT.re, :, :, w)), im = Array(view(FT.im, :, :, w)),
        dre = Array(view(FT.dre, :, :, w)), dim = Array(view(FT.dim, :, :, w))))
end
