# ------------------------------------------------------------------------------------------------
# SEOBNRv5PHM gradient, stage (4') with windowed tables: the per-sample reverse
# without arrays inside the differentiated function. Per sample:
#   1. locate (primal, no AD): the dynamics interval i (as `_seobv5phm_dyn_at`) and the PN-table window i0p .. i0p+3
#      around the interval of Omega(t) (as `seobnrv5phm_pn_state`; the gamma Hermite also reads the neighbours);
#   2. gather the nodes the sample reads into an isbits record `L` (151 floats: the model inputs of P, dynamics t and
#      its 8 node fields at i, i+1 (+ t at 1 and n for the clamp), the 4 x 6 mode tables at i, i+1, PN z / dz and the
#      gamma table on the window);
#   3. Enzyme reverse of the sample value as a function of (record subset, dynamics scalars, L): the GravitationalWaveforms
#      evaluators run unchanged on window types that mimic the arrays (a read outside the window gives NaN);
#   4. scatter dL back into the array shadows: an Enzyme reverse of the gather itself (`_rd(dL, gather)`), whose only
#      work is Enzyme's atomic shadow adds at the gathered positions.
# Inspiral (tq <= t_a) and merger-ringdown samples are differentiated separately, each with only the record fields it
# reads `Active` (`_phm_ri`: 51 floats, `_phm_rm`: 142 of the record's 248); a merger-ringdown sample reads no table.
# ------------------------------------------------------------------------------------------------

# window of a node vector: entries i0 .. i0 + K - 1; NaN elsewhere
struct _PHMWin{K,T}
    v::NTuple{K,T}
    i0::Int
end
@inline _phm_wsel(v::NTuple{2}, j, x) = ifelse(j == 0, v[1], ifelse(j == 1, v[2], x))
@inline _phm_wsel(v::NTuple{4}, j, x) = ifelse(j == 0, v[1], ifelse(j == 1, v[2], ifelse(j == 2, v[3], ifelse(j == 3, v[4], x))))
@inline Base.getindex(a::_PHMWin{K,T}, i::Integer) where {K,T<:AbstractFloat} = _phm_wsel(a.v, i - a.i0, T(NaN))
@inline Base.getindex(a::_PHMWin{K,SVector{M,T}}, i::Integer) where {K,M,T} =
    _phm_wsel(a.v, i - a.i0, SVector{M,T}(ntuple(_ -> T(NaN), Val(M))))
# the dynamics times: the window i, i + 1, and entries 1 and n (the clamp of `_seobv5phm_dyn_at`); `pos` is the
# precomputed `_eobimr_searchsortedlast` of the sample's (clamped) time
struct _PHMWinT{T}
    v::NTuple{4,T}                    # t[1], t[n], t[i], t[i+1]
    i0::Int
    n::Int
    pos::Int
end
@inline Base.getindex(a::_PHMWinT{T}, i::Integer) where {T} =
    ifelse(i == 1, a.v[1], ifelse(i == a.n, a.v[2], _phm_wsel((a.v[3], a.v[4]), i - a.i0, T(NaN))))
@inline GravitationalWaveforms._eobimr_searchsortedlast(a::_PHMWinT, x) = a.pos
# a mode table (node x mode) on the window i, i + 1
struct _PHMWinM{T}
    v::NTuple{2,SVector{6,T}}
    i0::Int
end
@inline Base.getindex(a::_PHMWinM{T}, i::Integer, k::Integer) where {T} = _phm_wsel((a.v[1][k], a.v[2][k]), i - a.i0, T(NaN))

# the record fields a sample reads: inspiral (tq <= t_a, `_seobv5phm_hpc_insp`) and merger-ringdown (`_seobv5phm_hpc_mr`),
# float-only (`Active` records with Bool fields broke Enzyme on GPUs, see `_rfl`): the anti-symmetric NQC flags and the
# mixed modes' `zero` flags go in as `Const` bitmasks (`_phm_rbits`), the dynamics scalars as (x1, h, oe) only
@inline _phm_ri(rec) = (nqc = rec.nqc, C = rec.C, qI2J = rec.qI2J, e1 = rec.e1, e2 = rec.e2, e3 = rec.e3,
    b6 = rec.asym.b6, b1 = rec.asym.b1)
@inline _phm_mixf(x) = (sph = x.sph, mu_mm = x.mu_mm, mu_lm = x.mu_lm, mu_ll = x.mu_ll)
@inline _phm_rm(rec) = (t_a = rec.t_a, mr = rec.mr, mix = (_phm_mixf(rec.mix[1]), _phm_mixf(rec.mix[2])),
    frame_a = rec.frame_a, precRate = rec.precRate, qI2J = rec.qI2J, amr = rec.asym.mr)
@inline _phm_rbits(rec) = _phm_fbits(rec.asym.flags) + 8 * Int(rec.mix[1].zero) + 16 * Int(rec.mix[2].zero)
@inline _phm_bit(fb, k) = (fb >> (k - 1)) & 1 != 0
@inline _phm_dmf(dm) = (x1 = dm.x1, h = dm.h, oe = dm.oe)
# both cotangents added into the full record's (`g`, zero outside)
@inline function _phm_rim_add(g, gi, gm)
    a = g.asym
    a = _nt_merge(a, (b6 = a.b6 + gi.b6, b1 = _ra(a.b1, gi.b1), mr = _ra(a.mr, gm.amr)))
    mx = g.mix
    mx = (_nt_merge(mx[1], _ra(_phm_mixf(mx[1]), gm.mix[1])), _nt_merge(mx[2], _ra(_phm_mixf(mx[2]), gm.mix[2])))
    return _nt_merge(g, (nqc = _ra(g.nqc, gi.nqc), C = _ra(g.C, gi.C), qI2J = _ra(_ra(g.qI2J, gi.qI2J), gm.qI2J),
        e1 = g.e1 + gi.e1, e2 = g.e2 + gi.e2, e3 = g.e3 + gi.e3, t_a = g.t_a + gm.t_a, mr = _ra(g.mr, gm.mr),
        mix = mx, frame_a = _ra(g.frame_a, gm.frame_a), precRate = g.precRate + gm.precRate, asym = a))
end

# 1. (pos, i, n, i0p) for the sample at dynamics time tq (the primal arrays)
@inline function _phm_locate(BUF, dm, w, tq)
    @inbounds begin
        n = dm.n
        tqc = clamp(tq, BUF[1, 1, w], BUF[n, 1, w])
        pos = GravitationalWaveforms._eobimr_searchsortedlast(view(BUF, :, 1, w), tqc)
        i = clamp(pos, 1, n - 1)
        # Omega at tq (the dynamics Hermite as `_seobv5phm_dyn_at`), then the PN interval as `seobnrv5phm_pn_state`
        h = BUF[i+1, 1, w] - BUF[i, 1, w]
        s = (tqc - BUF[i, 1, w]) / h
        _, om = GravitationalWaveforms._eobimr_hermite(BUF[i, 3, w], BUF[i, 7, w], BUF[i+1, 3, w], BUF[i+1, 7, w], h, s)
        i0p = clamp(_phm_pn_index(dm, om) - 1, 1, KernelFused._PHM_NPN - 3)
    end
    return (pos, i, n, i0p)
end
# the PN-table interval of Omega (as `seobnrv5phm_pn_state`)
@inline function _phm_pn_index(dm, om)
    g = GravitationalWaveforms._PNGrid(dm.x1, dm.h)
    xq = GravitationalWaveforms._seobv5phm_pn_x(om, 3.0)
    y = clamp((xq - g[1]) / (g[2] - g[1]), 0.0, Float64(dm.npn - 1))
    return min(unsafe_trunc(Int, floor(y)), dm.npn - 2) + 1
end

# 2. the gathered nodes (a pure read: its reverse is the scatter of 4.; no closures, see `_seobv5hm_nqc`)
@inline _phm_g2(BUF, i, c, w) = @inbounds (BUF[i, c, w], BUF[i+1, c, w])
@inline _phm_g6(MD, i, q, w) = @inbounds SVector{6}(MD[i, 1, q, w], MD[i, 2, q, w], MD[i, 3, q, w], MD[i, 4, q, w],
    MD[i, 5, q, w], MD[i, 6, q, w])
@inline _phm_gmd(MD, i, q, w) = (_phm_g6(MD, i, q, w), _phm_g6(MD, i + 1, q, w))
@inline _phm_g9(A, k, w) = @inbounds SVector{9}(A[1, k, w], A[2, k, w], A[3, k, w], A[4, k, w], A[5, k, w], A[6, k, w],
    A[7, k, w], A[8, k, w], A[9, k, w])
@inline _phm_gcol(A, k, w) = (_phm_g9(A, k, w), _phm_g9(A, k + 1, w), _phm_g9(A, k + 2, w), _phm_g9(A, k + 3, w))
@inline function _phm_gather(BUF, MD, Z, DZ, GT, P, w, idx)
    @inbounds begin
        _, i, n, i0p = idx
        return (pv = (P[1, w], P[8, w], P[9, w], P[10, w], P[11, w], P[12, w], BUF[1, 7, w]),
            t = (BUF[1, 1, w], BUF[n, 1, w], BUF[i, 1, w], BUF[i+1, 1, w]),
            r = _phm_g2(BUF, i, 2, w), phi = _phm_g2(BUF, i, 3, w), pr = _phm_g2(BUF, i, 4, w), pphi = _phm_g2(BUF, i, 5, w),
            dr = _phm_g2(BUF, i, 6, w), dphi = _phm_g2(BUF, i, 7, w), dpr = _phm_g2(BUF, i, 8, w),
            dpphi = _phm_g2(BUF, i, 9, w),
            md = (_phm_gmd(MD, i, 1, w), _phm_gmd(MD, i, 2, w), _phm_gmd(MD, i, 3, w), _phm_gmd(MD, i, 4, w)),
            z = _phm_gcol(Z, i0p, w), dz = _phm_gcol(DZ, i0p, w),
            g = (GT[i0p, w], GT[i0p+1, w], GT[i0p+2, w], GT[i0p+3, w]))
    end
end
@inline _phm_gather_dot(BUF, MD, Z, DZ, GT, P, w, idx, gL) = _rd(gL, _phm_gather(BUF, MD, Z, DZ, GT, P, w, idx))

# 3. the sample value Re(conj(xbar) h): inspiral from the record subset `ri`, the dynamics scalars and the gathered
# nodes; merger-ringdown from the subset `rm` and the model inputs `pv` only
# the inspiral sample's model (the windows + record subset as the `S` that `_seobv5phm_hpc_insp` reads)
@inline function _phm_iS(ri, dmf, L, idx, cst)
    pos, i, n, i0p = idx
    npn, fb = cst
    m1, om_ref, om_start = L.pv[1], L.pv[2], L.pv[3]
    tab = GravitationalWaveforms.SEOBNRv5PHMPNTable(GravitationalWaveforms._PNGrid(dmf.x1, dmf.h), _PHMWin(L.z, i0p), _PHMWin(L.dz, i0p), npn,
        dmf.oe, m1, 1 - m1, 3.0)
    b = GravitationalWaveforms._seobv5phm_base_tab(m1, 1 - m1, tab, om_ref, om_start)
    d = (t = _PHMWinT(L.t, i, n, pos), r = _PHMWin(L.r, i), phi = _PHMWin(L.phi, i), pr = _PHMWin(L.pr, i),
        pphi = _PHMWin(L.pphi, i), dr = _PHMWin(L.dr, i), dphi = _PHMWin(L.dphi, i), dpr = _PHMWin(L.dpr, i),
        dpphi = _PHMWin(L.dpphi, i), n = n, b = b)
    D = (A = _PHMWinM(L.md[1], i), dA = _PHMWinM(L.md[2], i), th = _PHMWinM(L.md[3], i), dth = _PHMWinM(L.md[4], i))
    asym = (flags = (_phm_bit(fb, 1), _phm_bit(fb, 2), _phm_bit(fb, 3)), b6 = ri.b6, b1 = ri.b1)
    return (nqc = ri.nqc, C = ri.C, qI2J = ri.qI2J, e1 = ri.e1, e2 = ri.e2, e3 = ri.e3, asym = asym, dyn = d, D = D, b = b,
        G = _PHMWin(L.g, i0p))
end
@inline function _phm_isample_dot(ri, dmf, L, idx, cst, jm1, xr, xi)
    S = _phm_iS(ri, dmf, L, idx, cst)
    m1, om_ref, om_start, incl, phiref, dt, om0 = L.pv
    t = jm1 * dt
    h = GravitationalWaveforms._seobv5phm_hpc_insp(S, t, incl, phiref) * GravitationalWaveforms._eobimr_turn_on(t / (4 * pi / om0))
    return xr * real(h) + xi * imag(h)
end
@inline function _phm_msample_dot(rm, pv, fb, jm1, xr, xi)
    m1, om_ref, om_start, incl, phiref, dt, om0 = pv
    mix = (merge(rm.mix[1], (zero = _phm_bit(fb, 4),)), merge(rm.mix[2], (zero = _phm_bit(fb, 5),)))
    S = (t_a = rm.t_a, mr = rm.mr, mix = mix, frame_a = rm.frame_a, precRate = rm.precRate, qI2J = rm.qI2J,
        asym = (mr = rm.amr,))
    t = jm1 * dt
    h = GravitationalWaveforms._seobv5phm_hpc_mr(S, t, incl, phiref) * GravitationalWaveforms._eobimr_turn_on(t / (4 * pi / om0))
    return xr * real(h) + xi * imag(h)
end
# (the whole sample, for checks)
@inline _phm_wsample_dot(rec, dm, L, idx, jm1, xr, xi) = jm1 * L.pv[6] <= rec.t_a ?
    _phm_isample_dot(_phm_ri(rec), _phm_dmf(dm), L, idx, (dm.npn, _phm_rbits(rec)), jm1, xr, xi) :
    _phm_msample_dot(_phm_rm(rec), L.pv, _phm_rbits(rec), jm1, xr, xi)

# (4'') chunk c of walker w's samples: the record-subset cotangents into RIB / RMB[c, w], the dynamics scalars' into
# DMB[c, w]; the node cotangents straight into the shadows. Samples past the waveform's end are zero for every input.
# `acc`: consecutive samples with the same window add their node cotangents in registers, scattered once per window.
@inline function _phm_wsample_rev_body!(c, w, RIB, RMB, DMB, XB, P, dP, Z, dZ, DZ, dDZ, BUF, dBUF, MD, dMD, GT, dGT, REC, DM,
        TP, N, SC, dSC, gpu, ::Val{acc} = Val(false)) where {acc}
    @inbounds begin
        rec = REC[w]; dm = DM[w]
        ri = _phm_ri(rec); rm = _phm_rm(rec); dmf = _phm_dmf(dm)
        fb = _phm_rbits(rec); cst = (dm.npn, fb)
        gri = _rz(ri); grm = _rz(rm); gdmf = _rz(dmf); gpv = ntuple(_ -> 0.0, Val(7))
        cur = (0, 0, 0, 0)
        gacc = _rz(_phm_gather(BUF, MD, Z, DZ, GT, P, w, (1, 1, 2, 1)))
        if TP[2, w] > 0
            jend = min(unsafe_trunc(Int, (rec.t_a + GravitationalWaveforms._EOBIMR_NTAU * rec.mr[1].tau) / P[12, w]) + 1, N)
            cs = cld(jend, _PHM_NSAMPT)
            ci = CartesianIndex(c, w)
            for j in (c - 1) * cs + 1:min(c * cs, jend)
                xb = XB[j, w]
                iszero(xb) && continue
                tq = (j - 1) * P[12, w]
                idx = _phm_locate(BUF, dm, w, tq)
                L = _phm_gather(BUF, MD, Z, DZ, GT, P, w, idx)
                if tq <= rec.t_a
                    r = _phm_grad(gpu, SC, dSC, ci, _phm_isample_dot, Enzyme.Active(ri), Enzyme.Active(dmf),
                        Enzyme.Active(L), Enzyme.Const(idx), Enzyme.Const(cst), Enzyme.Const(j - 1), Enzyme.Const(real(xb)),
                        Enzyme.Const(imag(xb)))
                    gri = _ra(gri, r[1]); gdmf = _ra(gdmf, r[2])
                    if !acc
                        _phm_scatter!(gpu, SC, dSC, ci, BUF, dBUF, MD, dMD, Z, dZ, DZ, dDZ, GT, dGT, P, dP, w, idx, r[3])
                    elseif idx == cur
                        gacc = _ra(gacc, r[3])
                    else
                        cur[2] > 0 && _phm_scatter!(gpu, SC, dSC, ci, BUF, dBUF, MD, dMD, Z, dZ, DZ, dDZ, GT, dGT, P, dP, w, cur,
                            gacc)
                        cur = idx; gacc = r[3]
                    end
                else
                    r = _phm_grad(gpu, SC, dSC, ci, _phm_msample_dot, Enzyme.Active(rm), Enzyme.Active(L.pv),
                        Enzyme.Const(fb), Enzyme.Const(j - 1), Enzyme.Const(real(xb)), Enzyme.Const(imag(xb)))
                    grm = _ra(grm, r[1]); gpv = _ra(gpv, r[2])
                end
            end
            acc && cur[2] > 0 && _phm_scatter!(gpu, SC, dSC, ci, BUF, dBUF, MD, dMD, Z, dZ, DZ, dDZ, GT, dGT, P, dP, w, cur, gacc)
            # the merger-ringdown samples' model-input cotangents (P rows 1, 8 .. 12; BUF dphi[1])
            _phm_grad(gpu, SC, dSC, ci, _phm_pv_dot, Enzyme.Duplicated(P, dP), Enzyme.Duplicated(BUF, dBUF), Enzyme.Const(w),
                Enzyme.Const(gpv))
        end
        RIB[c, w] = gri; RMB[c, w] = grm
        DMB[c, w] = _nt_merge(_rz(dm), (x1 = gdmf.x1, h = gdmf.h, oe = gdmf.oe))
    end
    return nothing
end
@inline _phm_scatter!(gpu, SC, dSC, ci, BUF, dBUF, MD, dMD, Z, dZ, DZ, dDZ, GT, dGT, P, dP, w, idx, g) =
    _phm_grad(gpu, SC, dSC, ci, _phm_gather_dot, Enzyme.Duplicated(BUF, dBUF), Enzyme.Duplicated(MD, dMD),
        Enzyme.Duplicated(Z, dZ), Enzyme.Duplicated(DZ, dDZ), Enzyme.Duplicated(GT, dGT), Enzyme.Duplicated(P, dP),
        Enzyme.Const(w), Enzyme.Const(idx), Enzyme.Const(g))
@inline _phm_pv_dot(P, BUF, w, g) = @inbounds _rd(g, (P[1, w], P[8, w], P[9, w], P[10, w], P[11, w], P[12, w], BUF[1, 7, w]))

# Sum of the chunk partials into the full record cotangent (zero outside the subsets) and the dynamics scalars'.
@inline function _phm_wchunk_sum_body!(w, GREC, GDM, RIB, RMB, DMB, REC, nc)
    @inbounds begin
        gi = RIB[1, w]; gm = RMB[1, w]; h = DMB[1, w]
        for c in 2:nc
            gi = _ra(gi, RIB[c, w]); gm = _ra(gm, RMB[c, w]); h = _ra(h, DMB[c, w])
        end
        GREC[w] = _phm_rim_add(_rz(REC[w]), gi, gm); GDM[w] = h
    end
    return nothing
end

# ---- the plunge step's reverse with a windowed PN table (`_phm_plunge_rev_body!`, `rev = :win`): the spins at omp
# read the table columns ip, ip + 1 only ----
@inline _phm_gather_pn2(P, Z, DZ, w, ip) = @inbounds (pv = (P[1, w], P[8, w], P[9, w]),
    z = (_phm_g9(Z, ip, w), _phm_g9(Z, ip + 1, w)), dz = (_phm_g9(DZ, ip, w), _phm_g9(DZ, ip + 1, w)))
@inline _phm_gather_pn2_dot(P, Z, DZ, w, ip, g) = _rd(g, _phm_gather_pn2(P, Z, DZ, w, ip))
@inline function _phm_wbase(dm, Lz, ip)
    m1, om_ref, om_start = Lz.pv
    tab = GravitationalWaveforms.SEOBNRv5PHMPNTable(GravitationalWaveforms._PNGrid(dm.x1, dm.h), _PHMWin(Lz.z, ip), _PHMWin(Lz.dz, ip), dm.npn,
        dm.oe, m1, 1 - m1, 3.0)
    return GravitationalWaveforms._seobv5phm_base_tab(m1, 1 - m1, tab, om_ref, om_start)
end
@inline function _phm_wstep_dot(dm, Lz, ip, sr, sph, spr, spp, st, omp, g)
    b = _phm_wbase(dm, Lz, ip)
    s = GravitationalWaveforms._seobv5phm_spins(b, omp)
    hstep, Dr, Dph, Dpr, Dpp = GravitationalWaveforms._seobv5phm_plunge_step(sr, spr, spp, s, omp, b,
        GravitationalWaveforms._SEOBV5PHM_C_OM, GravitationalWaveforms._SEOBV5PHM_C_R)
    nr = sr + Dr; npr = spr + Dpr; npp = spp + Dpp
    a, om, c, d, oc = GravitationalWaveforms._seobv5phm_rhs(nr, npr, npp, s, b)
    return _rd(g, (st + hstep, nr, sph + Dph, npr, npp, a, om, c, d, oc))
end
# the same step for the node-parallel plunge reverse (`_phm_pjac_rev_body!`): the dynamics scalars as (x1, h, oe) and
# without the t, phi accumulators (their cotangents are the suffix sums of the nodes' and go to hstep, Dph)
@inline function _phm_pstep(dmf, npn, Lz, ip, sr, spr, spp, omp)
    m1, om_ref, om_start = Lz.pv
    tab = GravitationalWaveforms.SEOBNRv5PHMPNTable(GravitationalWaveforms._PNGrid(dmf.x1, dmf.h), _PHMWin(Lz.z, ip), _PHMWin(Lz.dz, ip), npn,
        dmf.oe, m1, 1 - m1, 3.0)
    b = GravitationalWaveforms._seobv5phm_base_tab(m1, 1 - m1, tab, om_ref, om_start)
    s = GravitationalWaveforms._seobv5phm_spins(b, omp)
    hstep, Dr, Dph, Dpr, Dpp = GravitationalWaveforms._seobv5phm_plunge_step(sr, spr, spp, s, omp, b,
        GravitationalWaveforms._SEOBV5PHM_C_OM, GravitationalWaveforms._SEOBV5PHM_C_R)
    nr = sr + Dr; npr = spr + Dpr; npp = spp + Dpp
    a, om, c, d, oc = GravitationalWaveforms._seobv5phm_rhs(nr, npr, npp, s, b)
    return (hstep, nr, Dph, npr, npp, a, om, c, d, oc)
end
@inline _phm_pstep_dot(dmf, npn, Lz, ip, sr, spr, spp, omp, g) = _rd(g, _phm_pstep(dmf, npn, Lz, ip, sr, spr, spp, omp))
# the step stored into O[1:10, i, w], for its forward-mode Jacobian columns (`_phm_fwd`)
@inline function _phm_pstep_st!(O, i, w, dmf, npn, Lz, ip, sr, spr, spp, omp)
    o = _phm_pstep(dmf, npn, Lz, ip, sr, spr, spp, omp)
    @inbounds begin
        O[1, i, w] = o[1]; O[2, i, w] = o[2]; O[3, i, w] = o[3]; O[4, i, w] = o[4]; O[5, i, w] = o[5]
        O[6, i, w] = o[6]; O[7, i, w] = o[7]; O[8, i, w] = o[8]; O[9, i, w] = o[9]; O[10, i, w] = o[10]
    end
    return nothing
end
@inline _phm_fwd(gpu, f::F, args...) where {F} = _xasbp_ad(gpu, Enzyme.Forward, Enzyme.Const(f), Enzyme.Const, args...)
# K directions batched: the primal stored into JO, the tangents into JS[1:K] (separate arrays: shadows that are
# views of the primal's parent come back zero); an input seeded in direction k
@inline _phm_jb(JO, JS, ::Val{K}) where {K} = Enzyme.BatchDuplicated(JO, ntuple(b -> JS[b], Val(K)), false)
@inline _phm_bdir(x, k) = Enzyme.BatchDuplicated(x, ntuple(b -> Float64(b == k), Val(4)), false)
