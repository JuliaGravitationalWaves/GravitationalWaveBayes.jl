# ------------------------------------------------------------------------------------------------
# SEOBNRv5PHM log-likelihood gradient on a KernelAbstractions device:
# stagewise reverse through the stages of `KernelFused.seobnr_phm_device_loglikelihood!` (`SEOBNRPHMDevice.jl`),
# last stage first. This file, part 1 (the waveform tail):
#   (6') walker: the detector sums by hand -> the cotangent of the FFT bins Xbar (only the bins the data read), the
#        gradients along the extrinsic parameters (dL, t_c, ra, dec, psi) and along the walker scalars K = M/dL M_s,
#        dt (M) and t_peak (s);
#   (5') Xbar <- the adjoint of the unnormalized forward FFT, i.e. the unnormalized backward FFT;
#   (4') (chunk of samples) x walker: Enzyme reverse of each sample's Re(conj(Xbar_n) X_n) on the table nodes it
#        reads, gathered into registers, then scattered into the arrays' shadows (`SEOBNRPHMSampleRev.jl`).
# Cotangent convention for complex values: zbar = dL/dRe z + i dL/dIm z, so dL = Re(conj(zbar) dz).
# ------------------------------------------------------------------------------------------------

const _PHM_NSAMPT = 64            # reverse threads per walker in (4'), spread over the waveform's samples

# In-kernel reverse of a scalar function: its value goes into `out[j]` (`Duplicated`, shadow seeded with 1) and the
# call returns `nothing` (Enzyme's `autodiff_deferred` in a GPU kernel cannot return an Active scalar; see
# `_xasbp_bin_seed!`). Returns the cotangents of the arguments as `autodiff` does (`Active` ones as values).
@inline function _phm_store!(out, j, f::F, args...) where {F}
    @inbounds out[j] = f(args...)
    return nothing
end
# Fixed-arity methods (picked over the varargs one): with a splatted argument tuple Enzyme's GPU type analysis failed
# for record + array arguments (a byte load from a slot typed pointer-to-double).
for N in 1:16
    as = [Symbol(:a, i) for i in 1:N]
    @eval @inline function _phm_store!(out, j, f::F, $(as...)) where {F}
        @inbounds out[j] = f($(as...))
        return nothing
    end
end
# (runtime activity: static activity analysis rejects some records of views, e.g. the build post stage)
const _PHM_RREV = Enzyme.set_runtime_activity(Enzyme.Reverse)
# `_phm_grad`'s mode: plain reverse on GPUs (runtime activity gave wrong array cotangents in the t_peak stage there);
# runtime activity on the KernelAbstractions CPU backend, where static activity analysis rejects the records holding
# views of host arrays (the delta table `view(GT, :, w)`: "constant memory stored to a differentiable variable")
@inline _phm_rmode(::Val{true}) = Enzyme.Reverse
@inline _phm_rmode(::Val{false}) = _PHM_RREV
@inline function _phm_grad(gpu, out, dout, j, f::F, args...) where {F}
    @inbounds dout[j] = 1.0
    r = _xasbp_ad(gpu, _phm_rmode(gpu), Enzyme.Const(_phm_store!), Enzyme.Const{Nothing}, Enzyme.Duplicated(out, dout),
        Enzyme.Const(j), Enzyme.Const(f), args...)[1]
    return Base.tail(Base.tail(Base.tail(r)))
end

# ---- arithmetic on isbits records (nested NamedTuples / Tuples / SVectors; Bool and Int fields are labels) ----
@inline _rz(x::AbstractFloat) = zero(x)
@inline _rz(x::Complex) = zero(x)
@inline _rz(x::Union{Bool,Integer}) = x
@inline _rz(x::SVector) = zero(x)
@inline _rz(x::Tuple) = map(_rz, x)
@inline _rz(x::NamedTuple{K}) where {K} = NamedTuple{K}(map(_rz, values(x)))
@inline _ra(a::AbstractFloat, b) = a + b
@inline _ra(a::Complex, b) = a + b
@inline _ra(a::Union{Bool,Integer}, b) = a
@inline _ra(a::SVector, b) = a + b
@inline _ra(a::Tuple, b) = map(_ra, a, b)
@inline _ra(a::NamedTuple{K}, b) where {K} = NamedTuple{K}(map(_ra, values(a), values(b)))
@inline _phm_fbits(f) = Int(f[1]) + 2 * Int(f[2]) + 4 * Int(f[3])
# Float-only copy of a cotangent record (Bool / Int labels -> 0.0): as a `Const` argument of an in-kernel reverse, a
# record with Bool fields breaks Enzyme's GPU type analysis (a byte load in a slot typed pointer-to-double).
@inline _rfl(x::Union{Bool,Integer}) = 0.0
@inline _rfl(x::Union{AbstractFloat,Complex,SVector}) = x
@inline _rfl(x::Tuple) = map(_rfl, x)
@inline _rfl(x::NamedTuple{K}) where {K} = NamedTuple{K}(map(_rfl, values(x)))
# <a, b> with a a cotangent (complex entries: Re(conj(a) b), the convention above): seeds a record-valued function.
@inline _rd(a::AbstractFloat, b) = a * b
@inline _rd(a::Complex, b) = real(conj(a) * b)
@inline _rd(a::Union{Bool,Integer}, b) = 0.0
@inline _rd(a::SVector, b) = sum(a .* b)
@inline _rd(a::Tuple, b) = sum(map(_rd, a, b))
@inline _rd(a::NamedTuple, b) = sum(map(_rd, values(a), values(b)))

# ---- (6') ----
# One detector's sums and, given the cotangents (glp, ghh, gzr, gzi) of the scaled sums, Xbar += ..., and the
# gradients along tau (arrival time), Fp, Fc, K, dt, t_peak. `pass = 1`: sums only; `pass = 2`: the reverse.
# Calibration (`calv`, see `KernelFused._cal_bin`): the factor C joins c (so h, and every cotangent below, is the
# calibrated template's), and CG[o + k, w] += Re(conj(hb) dh/da_k), CG[o + NN + k, w] += Re(conj(hb) dh/dphi_k) with
# dh/da_k = WC[i, k] h / (1 + dA), dh/dphi_k = WC[i, k] h 4i / (4 + dphi^2).
@inline _phm_cal_rev!(::Nothing, CG, hb, h, i, w) = nothing
@inline function _phm_cal_rev!(calv::Tuple, CG, hb, h, i, w)
    CAL, WC, o, nnv = calv
    NN = KernelFused._valn(nnv)
    dA = 0.0; dP = 0.0
    @inbounds for k in 1:NN
        dA += WC[i, k] * CAL[o + k, w]
        dP += WC[i, k] * CAL[o + NN + k, w]
    end
    ga = real(conj(hb) * h) / (1 + dA)
    gp = real(conj(hb) * h * Complex(0.0, 4.0)) / (4 + dP * dP)
    @inbounds for k in 1:NN
        CG[o + k, w] += WC[i, k] * ga
        CG[o + NN + k, w] += WC[i, k] * gp
    end
    return nothing
end
@inline function _phm_det_rev!(XB, X, w, N, dtM, Kd, tpk, Fplus, Fcross, tsec::T, freqs, strain, psd, df, glp, ghh, gzr,
        gzi, calv = nothing, CG = nothing) where {T}
    gtau = zero(T); gFp = zero(T); gFc = zero(T); gK = zero(T); gdt = zero(T); gtp = zero(T)
    df_d = freqs[2] - freqs[1]
    @inbounds for i in eachindex(freqs)
        f = freqs[i]
        k = round(Int, f / df)
        km = mod(-k, N)
        sh = Complex{T}(cos(2 * T(pi) * f * tpk), sin(2 * T(pi) * f * tpk))
        Xa = X[k+1, w]; Xb = X[km+1, w]
        a = dtM * Xa * sh
        b = dtM * Xb * conj(sh)
        ph = Complex{T}(cos(-2 * T(pi) * f * tsec), sin(-2 * T(pi) * f * tsec))
        c = Kd / 2 * ph * KernelFused._cal_bin(calv, i, w)
        al = -Complex{T}(Fplus, Fcross)
        lam = c * al; mu = c * conj(al)
        h = lam * a + mu * conj(b)
        s = strain[i]; p = psd[i]
        wgt = KernelFused._safe_div(one(T), p)
        hb = (-4 * df_d * glp * (h - s) + 8 * df_d * ghh * h + 4 * df_d * (gzr * s + gzi * im * s)) * wgt
        ab = conj(lam) * hb
        bb = conj(hb) * mu
        XB[k+1, w] += dtM * conj(sh) * ab
        XB[km+1, w] += dtM * sh * bb
        gtau += real(conj(hb) * (-2 * T(pi) * im * f * h))
        gtp += real(conj(hb) * (2 * T(pi) * im * f * h))
        gK += real(conj(hb) * h) / Kd
        gdt += real(conj(hb) * h) / dtM
        gFp += real(conj(hb) * (-c * (a + conj(b))))
        gFc += real(conj(hb) * (-c * im * (a - conj(b))))
        _phm_cal_rev!(calv, CG, hb, h, i, w)
    end
    return gtau, gFp, gFc, gK, gdt, gtp
end

# Calibration marginalization (`KernelFused._PHMCalMarg`): as `_phm_det_rev!` with the template's bin cotangent
# hb = (AL[o + i, w] h + BE[o + i, w] d) / S from the curve-averaged coefficients (h uncalibrated).
@inline function _phm_det_rev_ab!(XB, X, w, N, dtM, Kd, tpk, Fplus, Fcross, tsec::T, freqs, strain, psd, df, AL, BE,
        o::Int) where {T}
    gtau = zero(T); gFp = zero(T); gFc = zero(T); gK = zero(T); gdt = zero(T); gtp = zero(T)
    @inbounds for i in eachindex(freqs)
        f = freqs[i]
        k = round(Int, f / df)
        km = mod(-k, N)
        sh = Complex{T}(cos(2 * T(pi) * f * tpk), sin(2 * T(pi) * f * tpk))
        Xa = X[k+1, w]; Xb = X[km+1, w]
        a = dtM * Xa * sh
        b = dtM * Xb * conj(sh)
        ph = Complex{T}(cos(-2 * T(pi) * f * tsec), sin(-2 * T(pi) * f * tsec))
        c = Kd / 2 * ph
        al = -Complex{T}(Fplus, Fcross)
        lam = c * al; mu = c * conj(al)
        h = lam * a + mu * conj(b)
        wgt = KernelFused._safe_div(one(T), psd[i])
        hb = (AL[o + i, w] * h + BE[o + i, w] * strain[i]) * wgt
        ab = conj(lam) * hb
        bb = conj(hb) * mu
        XB[k+1, w] += dtM * conj(sh) * ab
        XB[km+1, w] += dtM * sh * bb
        gtau += real(conj(hb) * (-2 * T(pi) * im * f * h))
        gtp += real(conj(hb) * (2 * T(pi) * im * f * h))
        gK += real(conj(hb) * h) / Kd
        gdt += real(conj(hb) * h) / dtM
        gFp += real(conj(hb) * (-c * (a + conj(b))))
        gFc += real(conj(hb) * (-c * im * (a - conj(b))))
    end
    return gtau, gFp, gFc, gK, gdt, gtp
end

# Extrinsic map of detector d: (t_c, ra, dec, psi) -> (tau_d, Fp_d, Fc_d) dotted with their cotangents.
@inline function _phm_ext_dot(t_c, ra, dec, psi, det_tuple, gmst_val, tref::Int, geo::Bool, gt, gp, gc)
    T = typeof(t_c)
    dt_ref = ifelse(geo & (tref > 0), time_delay_from_geocenter_gmst(det_tuple[max(tref, 1)], ra, dec, gmst_val), zero(T))
    acc = zero(T)
    for d in 1:length(det_tuple)
        dt_geo = ifelse(geo, time_delay_from_geocenter_gmst(det_tuple[d], ra, dec, gmst_val) - dt_ref, zero(T))
        Fp, Fc = antenna_response_gmst(det_tuple[d], ra, dec, psi, gmst_val)
        acc += gt[d] * (t_c + dt_geo) + gp[d] * Fp + gc[d] * Fc
    end
    return acc
end

# Walker w: logL (as the primal sums kernel), Xbar (zeroed first) and EG[:, w] = d logL / d(dL, t_c, ra, dec, psi, K, dt,
# t_peak[s]).
@inline function _phm_sums_rev_body!(w, ll, EG, XB, params, P, TP, X, N, df, freqs_tuple, strain_tuple, psd_tuple,
        det_tuple, gmst_val, lik_arr, tref::Int, geo::Bool, SC, dSC, gpu, cal = nothing, CG = nothing)
    @inbounds begin
        T = Float64
        for j in 1:N
            XB[j, w] = zero(Complex{T})
        end
        for k in 1:8
            EG[k, w] = zero(T)
        end
        lik = KernelFused._lik_from_array(lik_arr)
        dL, t_r, ra, dec, psi = params[5, w], params[6, w], params[8, w], params[9, w], params[11, w]
        # the sampled row is t_ref with `time_ref_frequency` (`_phm_tref_body!`); never reassigned (captured below)
        t_c = t_r - ifelse(lik.fref > 0, TP[3, w] + lik.toff, zero(T))
        dt_ref = ifelse(geo & (tref > 0), time_delay_from_geocenter_gmst(det_tuple[max(tref, 1)], ra, dec, gmst_val), zero(T))
        nd = length(freqs_tuple)
        taus = map(det -> t_c + ifelse(geo, time_delay_from_geocenter_gmst(det, ra, dec, gmst_val) - dt_ref, zero(T)), det_tuple)
        Fs = map(det -> antenna_response_gmst(det, ra, dec, psi, gmst_val), det_tuple)
        ll_plain = zero(T); hh = zero(T); zr = zero(T); zi = zero(T)
        for d in 1:nd
            a, h2, r, i = KernelFused._phm_det_sums(X, w, N, P[12, w], P[14, w], TP[1, w], Fs[d][1], Fs[d][2], taus[d],
                freqs_tuple[d], strain_tuple[d], psd_tuple[d], df, KernelFused._cal_det(cal, d))
            df_d = freqs_tuple[d][2] - freqs_tuple[d][1]
            ll_plain += T(-2) * df_d * a
            hh += T(4) * df_d * h2
            zr += T(4) * df_d * r
            zi += T(4) * df_d * i
        end
        v = KernelFused._marg_combine(ll_plain, hh, zr, zi, dL, lik)
        ok = (TP[2, w] > 0) & isfinite(v)
        ll[w] = ifelse(ok, v, T(KernelFused._SEOBV5F2_DEADZONE_LOGLIKELIHOOD))
        if ok
            gs = _phm_grad(gpu, SC, dSC, w, KernelFused._marg_combine, Enzyme.Active(ll_plain), Enzyme.Active(hh),
                Enzyme.Active(zr), Enzyme.Active(zi), Enzyme.Active(dL), Enzyme.Const(lik))
            glp, ghh, gzr, gzi, gdL = gs[1], gs[2], gs[3], gs[4], gs[5]
            gK = zero(T); gdt = zero(T); gtp = zero(T)
            gts = map(_ -> zero(T), freqs_tuple); gps = gts; gcs = gts
            for d in 1:nd
                r = _phm_det_rev!(XB, X, w, N, P[12, w], P[14, w], TP[1, w], Fs[d][1], Fs[d][2], taus[d], freqs_tuple[d],
                    strain_tuple[d], psd_tuple[d], df, glp, ghh, gzr, gzi, KernelFused._cal_det(cal, d), CG)
                gts = Base.setindex(gts, r[1], d); gps = Base.setindex(gps, r[2], d); gcs = Base.setindex(gcs, r[3], d)
                gK += r[4]; gdt += r[5]; gtp += r[6]
            end
            ge = _phm_grad(gpu, SC, dSC, w, _phm_ext_dot, Enzyme.Active(t_c), Enzyme.Active(ra), Enzyme.Active(dec),
                Enzyme.Active(psi), Enzyme.Const(det_tuple), Enzyme.Const(gmst_val),
                Enzyme.Const(tref), Enzyme.Const(geo), Enzyme.Const(gts), Enzyme.Const(gps), Enzyme.Const(gcs))
            EG[1, w] = gdL; EG[2, w] = ge[1]; EG[3, w] = ge[2]; EG[4, w] = ge[3]; EG[5, w] = ge[4]
            # t_ref: t_c = t_ref - (t_f M - t_peak M) - offset, so the peak-time cotangent gains g_tc (the t_f M part:
            # `_phm_tref_rev_body!`)
            EG[6, w] = gK; EG[7, w] = gdt; EG[8, w] = gtp + ifelse(lik.fref > 0, ge[1], zero(T))
        end
    end
    return nothing
end

# (6') with calibration marginalization: the curve-averaged bin coefficients AL, BE (`KernelFused._PHMCalMarg`, after the
# reverse GEMMs) and the explicit dL cotangent GDL; logL itself comes from the primal (`_phm_cm_body!`). Dead / non-finite
# walkers (ll == dead-zone value) get zero cotangents.
@inline function _phm_sums_rev_cm_body!(w, ll, EG, XB, AL, BE, GDL, params, P, TP, X, N, df, freqs_tuple, strain_tuple,
        psd_tuple, det_tuple, gmst_val, lik_arr, tref::Int, geo::Bool, SC, dSC, gpu, nb::Int)
    @inbounds begin
        T = Float64
        for j in 1:N
            XB[j, w] = zero(Complex{T})
        end
        for k in 1:8
            EG[k, w] = zero(T)
        end
        ok = (TP[2, w] > 0) & (ll[w] != T(KernelFused._SEOBV5F2_DEADZONE_LOGLIKELIHOOD))
        if ok
            lik = KernelFused._lik_from_array(lik_arr)
            t_r, ra, dec, psi = params[6, w], params[8, w], params[9, w], params[11, w]
            t_c = t_r - ifelse(lik.fref > 0, TP[3, w] + lik.toff, zero(T))
            dt_ref = ifelse(geo & (tref > 0), time_delay_from_geocenter_gmst(det_tuple[max(tref, 1)], ra, dec, gmst_val),
                zero(T))
            nd = length(freqs_tuple)
            taus = map(det -> t_c + ifelse(geo, time_delay_from_geocenter_gmst(det, ra, dec, gmst_val) - dt_ref, zero(T)),
                det_tuple)
            Fs = map(det -> antenna_response_gmst(det, ra, dec, psi, gmst_val), det_tuple)
            gK = zero(T); gdt = zero(T); gtp = zero(T)
            gts = map(_ -> zero(T), freqs_tuple); gps = gts; gcs = gts
            for d in 1:nd
                r = _phm_det_rev_ab!(XB, X, w, N, P[12, w], P[14, w], TP[1, w], Fs[d][1], Fs[d][2], taus[d], freqs_tuple[d],
                    strain_tuple[d], psd_tuple[d], df, AL, BE, (d - 1) * nb)
                gts = Base.setindex(gts, r[1], d); gps = Base.setindex(gps, r[2], d); gcs = Base.setindex(gcs, r[3], d)
                gK += r[4]; gdt += r[5]; gtp += r[6]
            end
            ge = _phm_grad(gpu, SC, dSC, w, _phm_ext_dot, Enzyme.Active(t_c), Enzyme.Active(ra), Enzyme.Active(dec),
                Enzyme.Active(psi), Enzyme.Const(det_tuple), Enzyme.Const(gmst_val),
                Enzyme.Const(tref), Enzyme.Const(geo), Enzyme.Const(gts), Enzyme.Const(gps), Enzyme.Const(gcs))
            EG[1, w] = GDL[w]; EG[2, w] = ge[1]; EG[3, w] = ge[2]; EG[4, w] = ge[3]; EG[5, w] = ge[4]
            EG[6, w] = gK; EG[7, w] = gdt; EG[8, w] = gtp + ifelse(lik.fref > 0, ge[1], zero(T))
        end
    end
    return nothing
end

# ---- (4') the sample reverse: `SEOBNRPHMSampleRev.jl` ----

# ---- (3') the build, in reverse order of `KernelFused._phm_b*_body!` ----
# (3'a) the post stage in four walker kernels (each its own, smaller GPU compile; one monolithic reverse of
# `_seobv5phm_build_post` + t_peak gave a 27 MB PTX that ptxas did not finish in 2.5 h): t_peak's Newton step, the
# anti-symmetric record, `_seobv5phm_post_rest!` (remnant, frames, delta table, merger-ringdown), `_seobv5phm_post_nqc`.
# Each takes what it reads as `Active` records (the stored primal ones) and `Duplicated` arrays, and is seeded with the
# cotangent of what it produced; fixed-arity store wrappers (a splatted argument tuple broke Enzyme's GPU type
# analysis), float-only `Const` cotangents, runtime activity (static activity analysis rejects records of views).
@inline function _phm_tpk_dot(rec, dm, P, Z, DZ, BUF, SCR, MD, GT, w, tm, gt)
    b = KernelFused._phm_base(P, Z, DZ, dm, w)
    S = GravitationalWaveforms._seobv5phm_from_rec(rec, KernelFused._phm_dyn(BUF, SCR, dm, b, w), KernelFused._phm_modes(MD, w), b,
        view(GT, :, w))
    return gt * GravitationalWaveforms._seobv5phm_tpeak_newton(S, tm)
end
# (the anti-symmetric NQC flags come in as a bitmask from the primal record: Booleans without derivative, and their
# 64-sample loop stays out of the differentiated code)
@inline function _phm_asym_dot(rec, sts, dm, P, Z, DZ, BUF, SCR, MD, GT, w, fbits, ga)
    b = KernelFused._phm_base(P, Z, DZ, dm, w)
    S = GravitationalWaveforms._seobv5phm_from_rec(rec, KernelFused._phm_dyn(BUF, SCR, dm, b, w), KernelFused._phm_modes(MD, w), b,
        view(GT, :, w))
    flags = ((fbits & 1) != 0, (fbits & 2) != 0, (fbits & 4) != 0)
    return _rd(ga, GravitationalWaveforms._seobv5phm_asym_build(S, sts, flags))
end
@inline function _phm_rest_dot(R, Q, dm, P, Z, DZ, BUF, SCR, GT, w, g0)
    b = KernelFused._phm_base(P, Z, DZ, dm, w)
    return _rd(g0, GravitationalWaveforms._seobv5phm_post_rest(view(GT, :, w), KernelFused._phm_dyn(BUF, SCR, dm, b, w), b, R, Q,
        KernelFused._phm_use_tref(P, w), P[8, w]))
end
@inline function _phm_rest_store!(out, j, R, Q, dm, P, Z, DZ, BUF, SCR, GT, w, g0)
    @inbounds out[j] = _phm_rest_dot(R, Q, dm, P, Z, DZ, BUF, SCR, GT, w, g0)
    return nothing
end

# t_peak (+ its share of TP[1] = t_peak M_s from the sums): the record's cotangent += d t_peak / d record
@inline function _phm_tpk_rev_body!(w, GDM, GREC, EG, P, dP, Z, dZ, DZ, dDZ, BUF, dBUF, SCR, dSCR, MD, dMD, GT, dGT, REC, DM, TM,
        TP, SC, dSC, gpu)
    @inbounds begin
        TP[2, w] > 0 || return nothing
        rec = REC[w]; g = GREC[w]
        gt = g.t_peak + EG[8, w] * P[13, w]
        dP[13, w] += EG[8, w] * rec.t_peak
        # (the sample stage's pattern: `_phm_grad`, plain reverse; runtime activity gave wrong array cotangents here)
        r = _phm_grad(gpu, SC, dSC, w, _phm_tpk_dot, Enzyme.Active(rec), Enzyme.Active(DM[w]), Enzyme.Duplicated(P, dP),
            Enzyme.Duplicated(Z, dZ), Enzyme.Duplicated(DZ, dDZ), Enzyme.Duplicated(BUF, dBUF), Enzyme.Duplicated(SCR, dSCR),
            Enzyme.Duplicated(MD, dMD), Enzyme.Duplicated(GT, dGT), Enzyme.Const(w), Enzyme.Const(TM[w]), Enzyme.Const(gt))
        GREC[w] = _ra(_nt_merge(g, (t_peak = 0.0,)), r[1])
        GDM[w] = _ra(GDM[w], r[2])
    end
    return nothing
end
# the anti-symmetric record: its cotangent -> the other record fields' and the pre-record's attachment states' (the
# build's samples; GPRE.sts is reversed later by the attachment re-integration's reverse)
@inline function _phm_asym_rev_body!(w, GDM, GREC, GPRE, P, dP, Z, dZ, DZ, dDZ, BUF, dBUF, SCR, dSCR, MD, dMD, GT, dGT, REC,
        RPRE, DM, TP, SC, dSC, gpu)
    @inbounds begin
        TP[2, w] > 0 || return nothing
        rec = REC[w]; g = GREC[w]
        r = _phm_grad(gpu, SC, dSC, w, _phm_asym_dot, Enzyme.Active(rec), Enzyme.Active(RPRE[w].sts), Enzyme.Active(DM[w]),
            Enzyme.Duplicated(P, dP), Enzyme.Duplicated(Z, dZ), Enzyme.Duplicated(DZ, dDZ), Enzyme.Duplicated(BUF, dBUF),
            Enzyme.Duplicated(SCR, dSCR), Enzyme.Duplicated(MD, dMD), Enzyme.Duplicated(GT, dGT), Enzyme.Const(w),
            Enzyme.Const(_phm_fbits(rec.asym.flags)), Enzyme.Const(_rfl(g.asym)))
        GREC[w] = _ra(_nt_merge(g, (asym = _rz(g.asym),)), r[1])
        GPRE[w] = _nt_merge(_rz(RPRE[w]), (sts = r[2],))  # (GPRE is reset by `_phm_rest_rev_body!`, which adds this)
        GDM[w] = _ra(GDM[w], r[3])
    end
    return nothing
end
# post_rest: the remaining record cotangent -> the pre record (GPRE), the NQC record (GQ), the delta table's shadow
# (read only here; the table itself is reversed by (3'a2))
@inline function _phm_rest_rev_body!(w, GPRE, GQ, GDM, GREC, P, dP, Z, dZ, DZ, dDZ, BUF, dBUF, SCR, dSCR, GT, dGT, RPRE, QREC,
        DM, TP, SC, dSC, gpu)
    @inbounds begin
        gsts = GPRE[w].sts                     # the anti-symmetric build's share (`_phm_asym_rev_body!`, run before)
        GPRE[w] = _rz(RPRE[w]); GQ[w] = _rz(QREC[w])
        TP[2, w] > 0 || return nothing
        g0 = _rfl(_nt_drop(GREC[w], Val((:asym, :use_asym, :t_peak))))
        dSC[w] = 1.0
        r = _xasbp_ad(gpu, _PHM_RREV, Enzyme.Const(_phm_rest_store!), Enzyme.Const{Nothing}, Enzyme.Duplicated(SC, dSC),
            Enzyme.Const(w), Enzyme.Active(RPRE[w]), Enzyme.Active(QREC[w]), Enzyme.Active(DM[w]), Enzyme.Duplicated(P, dP),
            Enzyme.Duplicated(Z, dZ), Enzyme.Duplicated(DZ, dDZ), Enzyme.Duplicated(BUF, dBUF), Enzyme.Duplicated(SCR, dSCR),
            Enzyme.Duplicated(GT, dGT), Enzyme.Const(w), Enzyme.Const(g0))[1]
        GPRE[w] = _nt_merge(r[3], (sts = _ra(r[3].sts, gsts),)); GQ[w] = r[4]
        GDM[w] = _ra(GDM[w], r[5])
    end
    return nothing
end
# post_nqc: the NQC record cotangent -> the pre record, one thread per (mode, walker) (`_seobv5phm_post_nqc_k`; the
# whole record in one thread was a 96 KB stack frame, 1.8 ms per walker at 4096 walkers), partials summed per walker
@inline function _phm_nqck_dot(R, dm, P, Z, DZ, w, k, gH, gn)
    Hk, nk = GravitationalWaveforms._seobv5phm_post_nqc_k(KernelFused._phm_base(P, Z, DZ, dm, w), R, k)
    return _rd(gH, Hk) + _rd(gn, nk)
end
@inline function _phm_nqck_rev_body!(k, w, GPREB, GDMB, GQ, P, dP, Z, dZ, DZ, dDZ, RPRE, DM, TP, SC, dSC, gpu)
    @inbounds begin
        R = RPRE[w]; dm = DM[w]
        GPREB[k, w] = _rz(R); GDMB[k, w] = _rz(dm)
        (k <= 6) & (TP[2, w] > 0) || return nothing
        q = GQ[w]
        r = _phm_grad(gpu, SC, dSC, CartesianIndex(k, w), _phm_nqck_dot, Enzyme.Active(R), Enzyme.Active(dm),
            Enzyme.Duplicated(P, dP), Enzyme.Duplicated(Z, dZ), Enzyme.Duplicated(DZ, dDZ), Enzyme.Const(w), Enzyme.Const(k),
            Enzyme.Const(q.Hs[k]), Enzyme.Const(q.nqc[k]))
        GPREB[k, w] = r[1]; GDMB[k, w] = r[2]
    end
    return nothing
end
@inline function _phm_nqc_sum_body!(w, GPRE, GDM, GPREB, GDMB, TP)
    @inbounds begin
        TP[2, w] > 0 || return nothing
        g = GPRE[w]; h = GDM[w]
        for k in 1:6
            g = _ra(g, GPREB[k, w]); h = _ra(h, GDMB[k, w])
        end
        GPRE[w] = g; GDM[w] = h
    end
    return nothing
end
# (3'a2) the delta table G[i+1] = G[i] + inc_i(PN table, e1, e2, e3) (G[1] = 0, G[i > n] = G[n]): walker suffix sums
# of its shadow -> inc_i's cotangent; node x walker VJPs of inc_i (the dynamics scalars and the frame vectors into
# NBG[1:12, i, w], the PN table arrays atomically); walker sums, and the frame vectors' cotangent -> J_f's. (The loop
# over the table inside an in-kernel reverse made Enzyme tape it on the device malloc heap, never freed.)
const _PHM_NBG = 12
@inline function _phm_gsuf_body!(w, GCB, dGT, DM, TP)
    @inbounds begin
        n = DM[w].npn
        L = size(dGT, 1)
        tail = 0.0
        for m in n+1:L
            tail += dGT[m, w]
        end
        acc = tail
        for i in n-1:-1:1
            acc += dGT[i+1, w]
            GCB[i, w] = ifelse(TP[2, w] > 0, acc, 0.0)
        end
    end
    return nothing
end
@inline _phm_ginc_dot(dm, P, Z, DZ, e1, e2, e3, w, i, c) =
    c * GravitationalWaveforms._seobv5phm_delta_inc(KernelFused._phm_base(P, Z, DZ, dm, w).tab, i, e1, e2, e3)
@inline function _phm_ginc_rev_body!(i, w, NBG, GCB, P, dP, Z, dZ, DZ, dDZ, DM, REC, TP, SC, dSC, gpu)
    @inbounds begin
        for k in 1:_PHM_NBG
            NBG[k, i, w] = 0.0
        end
        dm = DM[w]
        (i < dm.npn) & (TP[2, w] > 0) || return nothing
        rec = REC[w]
        r = _phm_grad(gpu, SC, dSC, CartesianIndex(i, w), _phm_ginc_dot, Enzyme.Active(dm), Enzyme.Duplicated(P, dP),
            Enzyme.Duplicated(Z, dZ), Enzyme.Duplicated(DZ, dDZ), Enzyme.Active(rec.e1), Enzyme.Active(rec.e2),
            Enzyme.Active(rec.e3), Enzyme.Const(w), Enzyme.Const(i), Enzyme.Const(GCB[i, w]))
        gdm, g1, g2, g3 = r[1], r[5], r[6], r[7]
        NBG[1, i, w] = gdm.x1; NBG[2, i, w] = gdm.h; NBG[3, i, w] = gdm.oe
        for k in 1:3
            NBG[3+k, i, w] = g1[k]; NBG[6+k, i, w] = g2[k]; NBG[9+k, i, w] = g3[k]
        end
    end
    return nothing
end
@inline _phm_jf_dot(Jf, g) = _rd(g, GravitationalWaveforms._seobv5phm_jframe(Jf / sqrt(Jf[1]^2 + Jf[2]^2 + Jf[3]^2)))
@inline _phm_nbg_add(acc, NBG, i, w) = ntuple(k -> @inbounds(acc[k] + NBG[k, i, w]), Val(_PHM_NBG))
@inline function _phm_gsum_body!(w, GPRE, GDM, NBG, DM, RPRE, TP, SC, dSC, gpu)
    @inbounds begin
        TP[2, w] > 0 || return nothing
        acc = ntuple(_ -> 0.0, Val(_PHM_NBG))
        for i in 1:DM[w].npn-1
            acc = _phm_nbg_add(acc, NBG, i, w)
        end
        h = GDM[w]
        GDM[w] = _nt_merge(h, (x1 = h.x1 + acc[1], h = h.h + acc[2], oe = h.oe + acc[3]))
        ge = (SVector(acc[4], acc[5], acc[6]), SVector(acc[7], acc[8], acc[9]), SVector(acc[10], acc[11], acc[12]))
        r = _phm_grad(gpu, SC, dSC, w, _phm_jf_dot, Enzyme.Active(RPRE[w].Jf), Enzyme.Const(ge))
        g = GPRE[w]
        GPRE[w] = _nt_merge(g, (Jf = g.Jf + r[1],))
    end
    return nothing
end

# the four, in order (host)
function _phm_post_rev!(backend, nw, GPRE, GQ, GDM, GREC, EG, P, dP, Z, dZ, DZ, dDZ, BUF, dBUF, SCR, dSCR, MD, dMD, GT, dGT,
        RPRE, QREC, REC, DM, TM, TP, SC, dSC, GPREB, GDMB, SCn, dSCn, gpu; stamp! = _ -> nothing)
    wg = backend isa KernelAbstractions.CPU ? 1 : KernelFused._phm_wg(nw)
    k(body, args) = KernelFused._phm_walker_k!(backend)(body, args; ndrange = nw, workgroupsize = wg)
    k(_phm_tpk_rev_body!, (GDM, GREC, EG, P, dP, Z, dZ, DZ, dDZ, BUF, dBUF, SCR, dSCR, MD, dMD, GT, dGT, REC, DM, TM, TP, SC,
        dSC, gpu))
    stamp!(:rev_tpk)
    k(_phm_asym_rev_body!, (GDM, GREC, GPRE, P, dP, Z, dZ, DZ, dDZ, BUF, dBUF, SCR, dSCR, MD, dMD, GT, dGT, REC, RPRE, DM, TP,
        SC, dSC, gpu))
    stamp!(:rev_asym)
    k(_phm_rest_rev_body!, (GPRE, GQ, GDM, GREC, P, dP, Z, dZ, DZ, dDZ, BUF, dBUF, SCR, dSCR, GT, dGT, RPRE, QREC, DM, TP, SC,
        dSC, gpu))
    stamp!(:rev_rest)
    KernelFused._phm_sample_k!(backend)(_phm_nqck_rev_body!, (GPREB, GDMB, GQ, P, dP, Z, dZ, DZ, dDZ, RPRE, DM, TP, SCn, dSCn,
        gpu); ndrange = (6, nw), workgroupsize = backend isa KernelAbstractions.CPU ? (6, 1) : (6, 16))
    k(_phm_nqc_sum_body!, (GPRE, GDM, GPREB, GDMB, TP))
    stamp!(:rev_nqc)
    return nothing
end

# (3'b) node x walker: the node derivatives (3-point stencils on the node times), reversed in place in the mode
# tables' shadow (dA, dth rows consumed, A, th rows i-1 .. i+1 and the node times accumulated).
@inline function _phm_nderiv!(i, w, BUF, MD, n)
    GravitationalWaveforms._seobv5phm_node_derivs!(KernelFused._phm_modes(MD, w), (t = view(BUF, :, 1, w), n = n), i)
    return nothing
end
@inline function _phm_nderiv_rev_body!(i, w, BUF, dBUF, MD, dMD, DM, TP, gpu)
    @inbounds begin
        n = DM[w].n
        (i <= n) & (TP[2, w] > 0) || return nothing
        _xasbp_ad(gpu, Enzyme.Reverse, Enzyme.Const(_phm_nderiv!), Enzyme.Const{Nothing}, Enzyme.Const(i), Enzyme.Const(w),
            Enzyme.Duplicated(BUF, dBUF), Enzyme.Duplicated(MD, dMD), Enzyme.Const(n))
    end
    return nothing
end

# (3'c) node x walker: the node modes (A, th rows consumed); the node's share of the cotangents of pref (6 complex),
# c21, c43 and of the dynamics scalars (x1, h, omega_end of the PN grid) into NB[1:17, i, w].
@inline function _phm_nmodes!(i, w, P, Z, DZ, BUF, SCR, MD, dm, pref, c21, c43)
    b = KernelFused._phm_base(P, Z, DZ, dm, w)
    GravitationalWaveforms._seobv5phm_node_modes!(KernelFused._phm_modes(MD, w), KernelFused._phm_dyn(BUF, SCR, dm, b, w), b,
        (pref = pref, c21 = c21, c43 = c43), i)
    return nothing
end
const _PHM_NNB = 17
@inline function _phm_nmodes_rev_body!(i, w, NB, P, dP, Z, dZ, DZ, dDZ, BUF, dBUF, SCR, dSCR, MD, dMD, DM, RPRE, TP, gpu)
    @inbounds begin
        for k in 1:_PHM_NNB
            NB[k, i, w] = 0.0
        end
        dm = DM[w]
        (i <= dm.n) & (TP[2, w] > 0) || return nothing
        R = RPRE[w]
        r = _xasbp_ad(gpu, Enzyme.Reverse, Enzyme.Const(_phm_nmodes!), Enzyme.Const{Nothing}, Enzyme.Const(i),
            Enzyme.Const(w), Enzyme.Duplicated(P, dP), Enzyme.Duplicated(Z, dZ), Enzyme.Duplicated(DZ, dDZ),
            Enzyme.Duplicated(BUF, dBUF), Enzyme.Duplicated(SCR, dSCR), Enzyme.Duplicated(MD, dMD), Enzyme.Active(dm),
            Enzyme.Active(R.pref), Enzyme.Active(R.c21), Enzyme.Active(R.c43))[1]
        gdm, gpref, g21, g43 = r[9], r[10], r[11], r[12]
        for k in 1:6
            NB[2k-1, i, w] = real(gpref[k]); NB[2k, i, w] = imag(gpref[k])
        end
        NB[13, i, w] = g21; NB[14, i, w] = g43
        NB[15, i, w] = gdm.x1; NB[16, i, w] = gdm.h; NB[17, i, w] = gdm.oe
    end
    return nothing
end
# (helpers: closures must not capture the reassigned accumulator -- Julia 1.13 boxes it, dynamic code on GPUs)
@inline _phm_nb_add(acc, NB, i, w) = ntuple(k -> @inbounds(acc[k] + NB[k, i, w]), Val(_PHM_NNB))
@inline _phm_pref_add(pref, acc) = ntuple(k -> pref[k] + complex(acc[2k-1], acc[2k]), Val(6))
# walker: the node shares summed into GPRE and GDM
@inline function _phm_nb_sum_body!(w, GPRE, GDM, NB, DM)
    @inbounds begin
        acc = ntuple(_ -> 0.0, Val(_PHM_NNB))
        for i in 1:DM[w].n
            acc = _phm_nb_add(acc, NB, i, w)
        end
        g = GPRE[w]
        GPRE[w] = _nt_merge(g, (pref = _phm_pref_add(g.pref, acc), c21 = g.c21 + acc[13],
            c43 = g.c43 + acc[14]))
        h = GDM[w]
        GDM[w] = _nt_merge(h, (x1 = h.x1 + acc[15], h = h.h + acc[16], oe = h.oe + acc[17]))
    end
    return nothing
end

# (3'd) the attachment record in three walker kernels (one reverse of all of `_seobv5phm_build_pre` was another
# multi-hour ptxas job): the special coefficients, the attachment states, the attachment times and spins; inputs from
# the stored record, cotangents accumulated in GPRE itself; then (3'e) the reference-phase shift.
const _PHM_ATT = GravitationalWaveforms._SEOBV5PHM_PRE_ATT
@inline function _phm_spec_dot(A, sts, dm, P, Z, DZ, w, g)
    c21, c43 = GravitationalWaveforms._seobv5phm_pre_spec(KernelFused._phm_base(P, Z, DZ, dm, w), A, sts, P[4, w], P[7, w])
    return g[1] * c21 + g[2] * c43
end
@inline function _phm_spec_rev_body!(w, GPRE, GDM, P, dP, Z, dZ, DZ, dDZ, RPRE, DM, TP, SC, dSC, gpu)
    @inbounds begin
        TP[2, w] > 0 || return nothing
        R = RPRE[w]; g = GPRE[w]
        r = _phm_grad(gpu, SC, dSC, w, _phm_spec_dot, Enzyme.Active(_nt_take(R, Val(_PHM_ATT))), Enzyme.Active(R.sts),
            Enzyme.Active(DM[w]), Enzyme.Duplicated(P, dP), Enzyme.Duplicated(Z, dZ), Enzyme.Duplicated(DZ, dDZ),
            Enzyme.Const(w), Enzyme.Const((g.c21, g.c43)))
        gA = _ra(_nt_take(g, Val(_PHM_ATT)), r[1])
        GPRE[w] = _nt_merge(_nt_merge(g, gA), (sts = _ra(g.sts, r[2]), c21 = 0.0, c43 = 0.0))
        GDM[w] = _ra(GDM[w], r[3])
    end
    return nothing
end
# The attachment states: `_seobv5phm_attach_states` re-integrates NLOC0 + 4 NF + 1 RK4 steps from node i0; reversed here
# step by step over the recomputed trajectory (one step VJP each, as the plunge). An Enzyme reverse of the whole loop
# tapes it on the device malloc heap: 0.56 s per walker on a V100, serialized over walkers.
const _PHM_NLOC0 = GravitationalWaveforms._SEOBV5PHM_ATT_N0
const _PHM_ATTNF = GravitationalWaveforms._SEOBV5PHM_ATT_NF
const _PHM_NAST = _PHM_NLOC0 + 4 * _PHM_ATTNF + 1
const _PHM_HF = 0.5 / _PHM_ATTNF                 # step between the states (M)
@inline function _phm_ast_dot(dm, P, Z, DZ, r, ph, pr, pp, oms, h, w, g)
    b = KernelFused._phm_base(P, Z, DZ, dm, w)
    return _rd(g, GravitationalWaveforms._seobv5phm_rk4(r, ph, pr, pp, h, oms, b))
end
# the snapshot cotangent at step NLOC0 + m (states at t_a - 1 + 0.5 m / NF, m = 0, NF, .., 4 NF), zero between
@inline _phm_snap(st, m) = m == 0 ? st[1] : m == _PHM_ATTNF ? st[2] : m == 2 * _PHM_ATTNF ? st[3] : m == 3 * _PHM_ATTNF ? st[4] :
    m == 4 * _PHM_ATTNF ? st[5] : (0.0, 0.0, 0.0, 0.0, 0.0)
@inline function _phm_sts_rev_body!(w, GPRE, GDM, P, dP, Z, dZ, DZ, dDZ, BUF, dBUF, SCR, dSCR, STB, RPRE, DM, TP, SC, dSC, gpu)
    @inbounds begin
        TP[2, w] > 0 || return nothing
        g = GPRE[w]; dm = DM[w]
        t_a = RPRE[w].t_a
        b = KernelFused._phm_base(P, Z, DZ, dm, w)
        n = dm.n
        i0 = clamp(GravitationalWaveforms._eobimr_searchsortedlast(view(BUF, :, 1, w), t_a - 1), 1, n)
        t0 = BUF[i0, 1, w]
        h0 = (t_a - 1 - t0) / _PHM_NLOC0
        # the trajectory (the state before each step)
        r, ph, pr, pp, oms = BUF[i0, 2, w], BUF[i0, 3, w], BUF[i0, 4, w], BUF[i0, 5, w], BUF[i0, 7, w]
        for j in 0:_PHM_NAST-1
            STB[1, j+1, w] = r; STB[2, j+1, w] = ph; STB[3, j+1, w] = pr; STB[4, j+1, w] = pp; STB[5, j+1, w] = oms
            r, ph, pr, pp, oms = GravitationalWaveforms._seobv5phm_rk4(r, ph, pr, pp, ifelse(j < _PHM_NLOC0, h0, _PHM_HF), oms, b)
        end
        # backwards: (rb .. ppb, osb) is the cotangent of the state after step j
        rb = phb = prb = ppb = osb = 0.0
        h0b = 0.0
        gdm = GDM[w]
        for j in _PHM_NAST-1:-1:0
            gs = _phm_snap(g.sts, j - _PHM_NLOC0)
            gout = (rb, phb, prb, ppb, osb + gs[5])
            q = _phm_grad(gpu, SC, dSC, w, _phm_ast_dot, Enzyme.Active(dm), Enzyme.Duplicated(P, dP), Enzyme.Duplicated(Z, dZ),
                Enzyme.Duplicated(DZ, dDZ), Enzyme.Active(STB[1, j+1, w]), Enzyme.Active(STB[2, j+1, w]),
                Enzyme.Active(STB[3, j+1, w]), Enzyme.Active(STB[4, j+1, w]), Enzyme.Active(STB[5, j+1, w]),
                Enzyme.Active(ifelse(j < _PHM_NLOC0, h0, _PHM_HF)), Enzyme.Const(w), Enzyme.Const(gout))
            gdm = _ra(gdm, q[1])
            rb = q[5] + gs[1]; phb = q[6] + gs[2]; prb = q[7] + gs[3]; ppb = q[8] + gs[4]; osb = q[9]
            h0b += ifelse(j < _PHM_NLOC0, q[10], 0.0)
        end
        GDM[w] = gdm
        dBUF[i0, 2, w] += rb; dBUF[i0, 3, w] += phb; dBUF[i0, 4, w] += prb; dBUF[i0, 5, w] += ppb; dBUF[i0, 7, w] += osb
        dBUF[i0, 1, w] -= h0b / _PHM_NLOC0
        GPRE[w] = _nt_merge(g, (t_a = g.t_a + h0b / _PHM_NLOC0, sts = _rz(g.sts)))
    end
    return nothing
end
# The same in parallel over the steps (as the plunge, `_phm_pjac_rev_body!`): step j maps (r, pr, pphi, omega_s) to
# (r, pr, pphi, Omega) of step j + 1 (phi accumulates); snapshot j's (r, phi, pr, pphi) cotangents sit on step j's input,
# its Omega cotangent on step j's output. SJ[:, j + 1, w]: A_j (1:16), c_j (17:20, then the cotangent of the step's
# output), the step's h partial (21) and dynamics-scalar partials (22:24).
@inline function _phm_ast_st!(O, i, w, dm, P, Z, DZ, r, ph, pr, pp, oms, h)
    b = KernelFused._phm_base(P, Z, DZ, dm, w)
    o = GravitationalWaveforms._seobv5phm_rk4(r, ph, pr, pp, h, oms, b)
    @inbounds begin
        O[1, i, w] = o[1]; O[2, i, w] = o[2]; O[3, i, w] = o[3]; O[4, i, w] = o[4]; O[5, i, w] = o[5]
    end
    return nothing
end
# phi cotangent of step j's output (the later snapshots') and the snapshot Omega cotangent on it
@inline function _phm_sts_own(sts, j)
    phb = 0.0
    for m in j+1:_PHM_NAST-1
        phb += _phm_snap(sts, m - _PHM_NLOC0)[2]
    end
    return phb, _phm_snap(sts, j - _PHM_NLOC0)[5]
end
@inline function _phm_sts_h(BUF, RPRE, dm, j, w)
    @inbounds begin
        t_a = RPRE[w].t_a
        i0 = clamp(GravitationalWaveforms._eobimr_searchsortedlast(view(BUF, :, 1, w), t_a - 1), 1, dm.n)
        return ifelse(j < _PHM_NLOC0, (t_a - 1 - BUF[i0, 1, w]) / _PHM_NLOC0, _PHM_HF)
    end
end
@inline function _phm_stsjac_rev_body!(jj, w, SJ, JO, JS, GPRE, P, Z, DZ, BUF, STB, RPRE, DM, TP, gpu)
    @inbounds begin
        TP[2, w] > 0 || return nothing
        j = jj - 1
        g = GPRE[w]; dm = DM[w]
        h = _phm_sts_h(BUF, RPRE, dm, j, w)
        phb, go = _phm_sts_own(g.sts, j)
        gs = _phm_snap(g.sts, j - _PHM_NLOC0)
        x = (STB[1, jj, w], STB[2, jj, w], STB[3, jj, w], STB[4, jj, w], STB[5, jj, w])
        # the Jacobian columns (forward mode, batched): A_j and c_j
        _phm_fwd(gpu, _phm_ast_st!, _phm_jb(JO, JS, Val(4)), Enzyme.Const(jj), Enzyme.Const(w), Enzyme.Const(dm),
            Enzyme.Const(P), Enzyme.Const(Z), Enzyme.Const(DZ), _phm_bdir(x[1], 1), Enzyme.Const(x[2]),
            _phm_bdir(x[3], 2), _phm_bdir(x[4], 3), _phm_bdir(x[5], 4), Enzyme.Const(h))
        for b in 1:4
            SJ[b, jj, w] = JS[b][1, jj, w]; SJ[4+b, jj, w] = JS[b][3, jj, w]; SJ[8+b, jj, w] = JS[b][4, jj, w]
            SJ[12+b, jj, w] = JS[b][5, jj, w]
            SJ[16+b, jj, w] = phb * JS[b][2, jj, w] + go * JS[b][5, jj, w] +
                (b == 1 ? gs[1] : b == 2 ? gs[3] : b == 3 ? gs[4] : 0.0)
        end
    end
    return nothing
end
# the re-integrated trajectory (the state before each step) into STB, as `_phm_sts_rev_body!`
@inline function _phm_ststraj_body!(w, STB, P, Z, DZ, BUF, RPRE, DM, TP)
    @inbounds begin
        TP[2, w] > 0 || return nothing
        dm = DM[w]
        t_a = RPRE[w].t_a
        b = KernelFused._phm_base(P, Z, DZ, dm, w)
        i0 = clamp(GravitationalWaveforms._eobimr_searchsortedlast(view(BUF, :, 1, w), t_a - 1), 1, dm.n)
        h0 = (t_a - 1 - BUF[i0, 1, w]) / _PHM_NLOC0
        r, ph, pr, pp, oms = BUF[i0, 2, w], BUF[i0, 3, w], BUF[i0, 4, w], BUF[i0, 5, w], BUF[i0, 7, w]
        for j in 0:_PHM_NAST-1
            STB[1, j+1, w] = r; STB[2, j+1, w] = ph; STB[3, j+1, w] = pr; STB[4, j+1, w] = pp; STB[5, j+1, w] = oms
            r, ph, pr, pp, oms = GravitationalWaveforms._seobv5phm_rk4(r, ph, pr, pp, ifelse(j < _PHM_NLOC0, h0, _PHM_HF), oms, b)
        end
    end
    return nothing
end
@inline function _phm_stschain_body!(w, SJ, GPRE, dBUF, BUF, RPRE, DM, TP)
    @inbounds begin
        TP[2, w] > 0 || return nothing
        g = GPRE[w]; dm = DM[w]
        t_a = RPRE[w].t_a
        i0 = clamp(GravitationalWaveforms._eobimr_searchsortedlast(view(BUF, :, 1, w), t_a - 1), 1, dm.n)
        s1 = s2 = s3 = s4 = 0.0
        for jj in _PHM_NAST:-1:1
            t1 = SJ[17, jj, w] + SJ[1, jj, w] * s1 + SJ[5, jj, w] * s2 + SJ[9, jj, w] * s3 + SJ[13, jj, w] * s4
            t2 = SJ[18, jj, w] + SJ[2, jj, w] * s1 + SJ[6, jj, w] * s2 + SJ[10, jj, w] * s3 + SJ[14, jj, w] * s4
            t3 = SJ[19, jj, w] + SJ[3, jj, w] * s1 + SJ[7, jj, w] * s2 + SJ[11, jj, w] * s3 + SJ[15, jj, w] * s4
            t4 = SJ[20, jj, w] + SJ[4, jj, w] * s1 + SJ[8, jj, w] * s2 + SJ[12, jj, w] * s3 + SJ[16, jj, w] * s4
            SJ[17, jj, w] = s1; SJ[18, jj, w] = s2; SJ[19, jj, w] = s3; SJ[20, jj, w] = s4
            s1, s2, s3, s4 = t1, t2, t3, t4
        end
        phb = 0.0
        for m in 0:_PHM_NAST-1
            phb += _phm_snap(g.sts, m - _PHM_NLOC0)[2]
        end
        dBUF[i0, 2, w] += s1; dBUF[i0, 3, w] += phb; dBUF[i0, 4, w] += s2; dBUF[i0, 5, w] += s3; dBUF[i0, 7, w] += s4
    end
    return nothing
end
# the parameter part of step j's VJP with its output's whole cotangent
@inline function _phm_stspar_rev_body!(jj, w, SJ, GPRE, P, dP, Z, dZ, DZ, dDZ, BUF, STB, RPRE, DM, TP, SC, dSC, gpu)
    @inbounds begin
        TP[2, w] > 0 || return nothing
        j = jj - 1
        dm = DM[w]
        phb, go = _phm_sts_own(GPRE[w].sts, j)
        gout = (SJ[17, jj, w], phb, SJ[18, jj, w], SJ[19, jj, w], SJ[20, jj, w] + go)
        q = _phm_grad(gpu, SC, dSC, CartesianIndex(jj, w), _phm_ast_dot, Enzyme.Active(dm), Enzyme.Duplicated(P, dP),
            Enzyme.Duplicated(Z, dZ), Enzyme.Duplicated(DZ, dDZ), Enzyme.Const(STB[1, jj, w]), Enzyme.Const(STB[2, jj, w]),
            Enzyme.Const(STB[3, jj, w]), Enzyme.Const(STB[4, jj, w]), Enzyme.Const(STB[5, jj, w]),
            Enzyme.Active(_phm_sts_h(BUF, RPRE, dm, j, w)), Enzyme.Const(w), Enzyme.Const(gout))
        SJ[21, jj, w] = q[10]; SJ[22, jj, w] = q[1].x1; SJ[23, jj, w] = q[1].h; SJ[24, jj, w] = q[1].oe
    end
    return nothing
end
@inline function _phm_stssum_body!(w, SJ, GPRE, GDM, dBUF, BUF, RPRE, DM, TP)
    @inbounds begin
        TP[2, w] > 0 || return nothing
        g = GPRE[w]; dm = DM[w]
        i0 = clamp(GravitationalWaveforms._eobimr_searchsortedlast(view(BUF, :, 1, w), RPRE[w].t_a - 1), 1, dm.n)
        h0b = 0.0; a1 = a2 = a3 = 0.0
        for jj in 1:_PHM_NAST
            jj <= _PHM_NLOC0 && (h0b += SJ[21, jj, w])
            a1 += SJ[22, jj, w]; a2 += SJ[23, jj, w]; a3 += SJ[24, jj, w]
        end
        gdm = GDM[w]
        GDM[w] = _nt_merge(gdm, (x1 = gdm.x1 + a1, h = gdm.h + a2, oe = gdm.oe + a3))
        dBUF[i0, 1, w] -= h0b / _PHM_NLOC0
        GPRE[w] = _nt_merge(g, (t_a = g.t_a + h0b / _PHM_NLOC0, sts = _rz(g.sts)))
    end
    return nothing
end
@inline function _phm_att_dot(dm, P, Z, DZ, BUF, SCR, w, g)
    b = KernelFused._phm_base(P, Z, DZ, dm, w)
    return _rd(g, GravitationalWaveforms._seobv5phm_pre_att(KernelFused._phm_dyn(BUF, SCR, dm, b, w), b))
end
@inline function _phm_att_rev_body!(w, GPRE, GDM, P, dP, Z, dZ, DZ, dDZ, BUF, dBUF, SCR, dSCR, DM, TP, SC, dSC, gpu)
    @inbounds begin
        TP[2, w] > 0 || return nothing
        r = _phm_grad(gpu, SC, dSC, w, _phm_att_dot, Enzyme.Active(DM[w]), Enzyme.Duplicated(P, dP), Enzyme.Duplicated(Z, dZ),
            Enzyme.Duplicated(DZ, dDZ), Enzyme.Duplicated(BUF, dBUF), Enzyme.Duplicated(SCR, dSCR), Enzyme.Const(w),
            Enzyme.Const(_nt_take(GPRE[w], Val(_PHM_ATT))))
        GDM[w] = _ra(GDM[w], r[1])
    end
    return nothing
end
# (3'e) the reference-phase shift phi -= phi(t_ref) (`use_tref`): the shift's cotangent is minus the sum of the
# shifted phases' cotangents; phi(t_ref) is linear in the phases, so the post-shift nodes give its derivative.
@inline _phm_refph_dot(dm, P, Z, DZ, BUF, SCR, w, c) = c * GravitationalWaveforms._seobv5phm_ref_phase(
    KernelFused._phm_dyn(BUF, SCR, dm, KernelFused._phm_base(P, Z, DZ, dm, w), w), P[8, w])
@inline function _phm_refph_rev_body!(w, GDM, P, dP, Z, dZ, DZ, dDZ, BUF, dBUF, SCR, dSCR, DM, TP, SC, dSC, gpu)
    @inbounds begin
        (TP[2, w] > 0) & KernelFused._phm_use_tref(P, w) || return nothing
        s = 0.0
        for i in 1:DM[w].n
            s += dBUF[i, 3, w]
        end
        r = _phm_grad(gpu, SC, dSC, w, _phm_refph_dot, Enzyme.Active(DM[w]), Enzyme.Duplicated(P, dP), Enzyme.Duplicated(Z, dZ),
            Enzyme.Duplicated(DZ, dDZ), Enzyme.Duplicated(BUF, dBUF), Enzyme.Duplicated(SCR, dSCR), Enzyme.Const(w),
            Enzyme.Const(-s))
        GDM[w] = _ra(GDM[w], r[1])
    end
    return nothing
end
function _phm_pre_rev!(backend, nw, GPRE, GDM, P, dP, Z, dZ, DZ, dDZ, BUF, dBUF, SCR, dSCR, STB, RPRE, DM, TP, SC, dSC, gpu;
        stamp! = _ -> nothing, par = nothing)
    wg = backend isa KernelAbstractions.CPU ? 1 : KernelFused._phm_wg(nw)
    k(body, args) = KernelFused._phm_walker_k!(backend)(body, args; ndrange = nw, workgroupsize = wg)
    k(_phm_spec_rev_body!, (GPRE, GDM, P, dP, Z, dZ, DZ, dDZ, RPRE, DM, TP, SC, dSC, gpu))
    stamp!(:rev_spec)
    if par === nothing
        k(_phm_sts_rev_body!, (GPRE, GDM, P, dP, Z, dZ, DZ, dDZ, BUF, dBUF, SCR, dSCR, STB, RPRE, DM, TP, SC, dSC, gpu))
    else
        SJ, JO, JS, SCn, dSCn = par
        nk(body, args) = KernelFused._phm_sample_k!(backend)(body, args; ndrange = (_PHM_NAST, nw),
            workgroupsize = backend isa KernelAbstractions.CPU ? (_PHM_NAST, 1) : (32, 1))
        k(_phm_ststraj_body!, (STB, P, Z, DZ, BUF, RPRE, DM, TP))
        nk(_phm_stsjac_rev_body!, (SJ, JO, JS, GPRE, P, Z, DZ, BUF, STB, RPRE, DM, TP, gpu))
        k(_phm_stschain_body!, (SJ, GPRE, dBUF, BUF, RPRE, DM, TP))
        nk(_phm_stspar_rev_body!, (SJ, GPRE, P, dP, Z, dZ, DZ, dDZ, BUF, STB, RPRE, DM, TP, SCn, dSCn, gpu))
        k(_phm_stssum_body!, (SJ, GPRE, GDM, dBUF, BUF, RPRE, DM, TP))
    end
    stamp!(:rev_sts)
    k(_phm_att_rev_body!, (GPRE, GDM, P, dP, Z, dZ, DZ, dDZ, BUF, dBUF, SCR, dSCR, DM, TP, SC, dSC, gpu))
    stamp!(:rev_att)
    k(_phm_refph_rev_body!, (GDM, P, dP, Z, dZ, DZ, dDZ, BUF, dBUF, SCR, dSCR, DM, TP, SC, dSC, gpu))
    stamp!(:rev_refph)
    return nothing
end

# ---- (2') the dynamics, in reverse order of `KernelFused._phm_dynamics!` ----
# Per-walker scalar cotangents DG[:, w]: 1 PA grid spacing. Node partials of the dynamics scalars (x1, h, omega_end
# of the PN grid) accumulate in NBP[1:3, i, w] (one thread per node per kernel), summed into GDM at the start stage.

# Transposes of the spline helpers (`_seobv5phm_spline_slopes!`, `_seobv5phm_spline_cumint!`) on a uniform grid of
# spacing h. slopes: s = A^-1 R(m), m_i = (y_{i+1} - y_i)/h; reverse: A^T z = sbar (Thomas), mbar = R^T z,
# ybar += D^T mbar / h, hbar = -sum(sbar s) / h. `z`, `c` scratch (length >= n).
@inline _phm_At_sub(i) = i == 2 ? 2.0 : 1.0
@inline _phm_At_diag(i, n) = (i == 1) | (i == n) ? 1.0 : 4.0
@inline _phm_At_sup(i, n) = i == n - 1 ? 2.0 : 1.0
@inline function _phm_slopes_T!(ybar, sbar, s, h, n, z, c)
    @inbounds begin
        c[1] = _phm_At_sup(1, n) / _phm_At_diag(1, n)
        z[1] = sbar[1] / _phm_At_diag(1, n)
        for i in 2:n
            den = _phm_At_diag(i, n) - _phm_At_sub(i) * c[i-1]
            c[i] = i < n ? _phm_At_sup(i, n) / den : 0.0
            z[i] = (sbar[i] - _phm_At_sub(i) * z[i-1]) / den
        end
        for i in n-1:-1:1
            z[i] -= c[i] * z[i+1]
        end
        hb = 0.0
        for i in 1:n
            hb -= sbar[i] * s[i]
        end
        hb /= h
        # mbar = R^T z, then ybar += D^T mbar / h (m_j = (y_{j+1} - y_j)/h)
        for j in 1:n-1
            mb = 0.0
            j == 1 && (mb += 2.5 * z[1])
            j == 2 && (mb += 0.5 * z[1])
            j <= n - 2 && (mb += 3 * z[j+1])          # row i = j + 1 (2 <= i <= n - 1) holds 3 m_{i-1}
            j >= 2 && (mb += 3 * z[j])                # row i = j (2 <= i <= n - 1) holds 3 m_i
            j == n - 2 && (mb += 0.5 * z[n])
            j == n - 1 && (mb += 2.5 * z[n])
            ybar[j+1] += mb / h
            ybar[j] -= mb / h
        end
    end
    return hb
end
# cumint: I_1 = 0, I_{i+1} = I_i + h/2 (y_i + y_{i+1}) + h^2/12 (s_i - s_{i+1}); reverse into ybar, sbar (+=), hbar.
@inline function _phm_cumint_T!(ybar, sbar, Ibar, y, s, h, n)
    hb = 0.0
    R = 0.0
    @inbounds for i in n-1:-1:1
        R += Ibar[i+1]
        ybar[i] += h / 2 * R; ybar[i+1] += h / 2 * R
        sbar[i] += h * h / 12 * R; sbar[i+1] -= h * h / 12 * R
        hb += R * ((y[i] + y[i+1]) / 2 + h / 6 * (s[i] - s[i+1]))
    end
    return hb
end

const _PHM_BCOLS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11)       # t, r, phi, pr, pphi, dr, dphi, dpr, dpphi, omega_circ
# (plunge step k from node k - 1 with the PN window of omp, `_phm_wstep_dot`)
@inline function _phm_plunge_rev_body!(w, GDM, P, dP, Z, dZ, DZ, dDZ, BUF, dBUF, DM, TP, SC, dSC, gpu)
    @inbounds begin
        TP[2, w] > 0 || return nothing
        dm = DM[w]
        n, npa = dm.n, dm.npa
        # padding t[i] = t[n] + (i - n)
        acc = 0.0
        for i in n+1:KernelFused._PHM_ND
            acc += dBUF[i, 1, w]; dBUF[i, 1, w] = 0.0
        end
        dBUF[n, 1, w] += acc
        gdm = GDM[w]
        for k in n:-1:npa+1
            j = k - 1
            omp = j == npa ? (dm.pa_success ? BUF[npa, 7, w] : P[9, w]) : BUF[j, 7, w]
            g = map(c -> dBUF[k, c, w], _PHM_BCOLS)
            gomp = dBUF[k, 10, w]
            ip = _phm_pn_index(dm, omp)
            Lz = _phm_gather_pn2(P, Z, DZ, w, ip)
            rw = _phm_grad(gpu, SC, dSC, w, _phm_wstep_dot, Enzyme.Active(dm), Enzyme.Active(Lz), Enzyme.Const(ip),
                Enzyme.Active(BUF[j, 2, w]), Enzyme.Active(BUF[j, 3, w]), Enzyme.Active(BUF[j, 4, w]),
                Enzyme.Active(BUF[j, 5, w]), Enzyme.Active(BUF[j, 1, w]), Enzyme.Active(omp), Enzyme.Const(g))
            _phm_grad(gpu, SC, dSC, w, _phm_gather_pn2_dot, Enzyme.Duplicated(P, dP), Enzyme.Duplicated(Z, dZ),
                Enzyme.Duplicated(DZ, dDZ), Enzyme.Const(w), Enzyme.Const(ip), Enzyme.Const(rw[2]))
            r = (rw[1], nothing, nothing, nothing, rw[4], rw[5], rw[6], rw[7], rw[8], rw[9])
            gdm = _ra(gdm, r[1])
            dBUF[j, 2, w] += r[5]; dBUF[j, 3, w] += r[6]; dBUF[j, 4, w] += r[7]; dBUF[j, 5, w] += r[8]
            dBUF[j, 1, w] += r[9]
            go = r[10] + gomp
            if (j == npa) & !dm.pa_success
                dP[9, w] += go
            else
                dBUF[j, 7, w] += go
            end
            for c in 1:11
                dBUF[k, c, w] = 0.0                      # consumed
            end
        end
        GDM[w] = gdm
    end
    return nothing
end
# The same reverse in parallel over the steps. Only (r, pr, pphi, omega) couple consecutive steps (t and phi are
# accumulators, the spins are read at the previous node's omega), so with the step's local Jacobian A_k
# (4 x 4: node k's (r, pr, pphi, dphi) against step k's (r, pr, pphi, omp)) the cotangent the later steps send to
# node k obeys s_{k-1} = c_k + A_k^T s_k, c_k the step's VJP of node k's own cotangents:
#   ptot (walker): t, phi cotangents -> their suffix sums;
#   pjac (node):   c_k, A_k (one reverse plus four with unit cotangents) and the parameter part of c_k;
#   pchain (walker): the recursion (4 x 4 per step) into s_k, and s at node npa into the PA nodes;
#   ppar (node):   the parameter part of A_k^T s_k;  psum (walker): the dynamics-scalar partials into GDM.
# PJ[:, i, w] for node k = npa + i: A_k (1:16, PJ[4(a - 1) + b] = d out_a / d in_b), c_k (17:20), s_k (21:24); NBQ the
# node partials of (x1, h, omega_end).
@inline _phm_omp(BUF, P, dm, j, w) = @inbounds j == dm.npa ? (dm.pa_success ? BUF[j, 7, w] : P[9, w]) : BUF[j, 7, w]
@inline function _phm_ptot_body!(w, dBUF, DM, TP)
    @inbounds begin
        TP[2, w] > 0 || return nothing
        dm = DM[w]
        n, npa = dm.n, dm.npa
        acc = 0.0                                        # padding t[i] = t[n] + (i - n)
        for i in n+1:KernelFused._PHM_ND
            acc += dBUF[i, 1, w]; dBUF[i, 1, w] = 0.0
        end
        dBUF[n, 1, w] += acc
        for k in n-1:-1:npa
            dBUF[k, 1, w] += dBUF[k+1, 1, w]; dBUF[k, 3, w] += dBUF[k+1, 3, w]
        end
    end
    return nothing
end
@inline function _phm_pjac_rev_body!(i, w, PJ, JO, JS, P, Z, DZ, BUF, dBUF, DM, TP, gpu)
    @inbounds begin
        dm = DM[w]
        k = dm.npa + i
        (TP[2, w] > 0) & (k <= dm.n) || return nothing
        j = k - 1
        omp = _phm_omp(BUF, P, dm, j, w)
        ip = _phm_pn_index(dm, omp)
        Lz = _phm_gather_pn2(P, Z, DZ, w, ip)
        dmf = _phm_dmf(dm)
        sr, spr, spp = BUF[j, 2, w], BUF[j, 4, w], BUF[j, 5, w]
        g = map(c -> dBUF[k, c, w], _PHM_BCOLS)
        # the Jacobian columns (forward mode, the four directions batched): A_k and c_k = J^T g
        _phm_fwd(gpu, _phm_pstep_st!, _phm_jb(JO, JS, Val(4)), Enzyme.Const(i), Enzyme.Const(w), Enzyme.Const(dmf),
            Enzyme.Const(dm.npn), Enzyme.Const(Lz), Enzyme.Const(ip), _phm_bdir(sr, 1), _phm_bdir(spr, 2),
            _phm_bdir(spp, 3), _phm_bdir(omp, 4))
        for b in 1:4
            PJ[b, i, w] = JS[b][2, i, w]; PJ[4+b, i, w] = JS[b][4, i, w]; PJ[8+b, i, w] = JS[b][5, i, w]
            PJ[12+b, i, w] = JS[b][7, i, w]
            c = 0.0
            for a in 1:10
                c += g[a] * JS[b][a, i, w]
            end
            PJ[16+b, i, w] = c
        end
        PJ[20, i, w] += dBUF[k, 10, w]
    end
    return nothing
end
@inline function _phm_pchain_body!(w, PJ, dP, dBUF, DM, TP)
    @inbounds begin
        TP[2, w] > 0 || return nothing
        dm = DM[w]
        n, npa = dm.n, dm.npa
        s1 = s2 = s3 = s4 = 0.0
        for k in n:-1:npa+1
            i = k - npa
            PJ[21, i, w] = s1; PJ[22, i, w] = s2; PJ[23, i, w] = s3; PJ[24, i, w] = s4
            t1 = PJ[17, i, w] + PJ[1, i, w] * s1 + PJ[5, i, w] * s2 + PJ[9, i, w] * s3 + PJ[13, i, w] * s4
            t2 = PJ[18, i, w] + PJ[2, i, w] * s1 + PJ[6, i, w] * s2 + PJ[10, i, w] * s3 + PJ[14, i, w] * s4
            t3 = PJ[19, i, w] + PJ[3, i, w] * s1 + PJ[7, i, w] * s2 + PJ[11, i, w] * s3 + PJ[15, i, w] * s4
            t4 = PJ[20, i, w] + PJ[4, i, w] * s1 + PJ[8, i, w] * s2 + PJ[12, i, w] * s3 + PJ[16, i, w] * s4
            s1, s2, s3, s4 = t1, t2, t3, t4
        end
        n > npa || return nothing
        dBUF[npa, 2, w] += s1; dBUF[npa, 4, w] += s2; dBUF[npa, 5, w] += s3
        if dm.pa_success
            dBUF[npa, 7, w] += s4
        else
            dP[9, w] += s4
        end
    end
    return nothing
end
# the parameter part of step k's VJP with node k's whole cotangent (its own and s_k); node k's row consumed
@inline function _phm_ppar_rev_body!(i, w, PJ, NBQ, P, dP, Z, dZ, DZ, dDZ, BUF, dBUF, DM, TP, SC, dSC, gpu)
    @inbounds begin
        dm = DM[w]
        k = dm.npa + i
        (TP[2, w] > 0) & (k <= dm.n) || return nothing
        g = map(c -> dBUF[k, c, w], _PHM_BCOLS)
        g = (g[1], g[2] + PJ[21, i, w], g[3], g[4] + PJ[22, i, w], g[5] + PJ[23, i, w], g[6], g[7] + PJ[24, i, w], g[8],
            g[9], g[10])
        for c in 1:11
            dBUF[k, c, w] = 0.0
        end
        j = k - 1
        omp = _phm_omp(BUF, P, dm, j, w)
        ip = _phm_pn_index(dm, omp)
        Lz = _phm_gather_pn2(P, Z, DZ, w, ip)
        ci = CartesianIndex(i, w)
        r = _phm_grad(gpu, SC, dSC, ci, _phm_pstep_dot, Enzyme.Active(_phm_dmf(dm)), Enzyme.Const(dm.npn),
            Enzyme.Active(Lz), Enzyme.Const(ip), Enzyme.Const(BUF[j, 2, w]), Enzyme.Const(BUF[j, 4, w]),
            Enzyme.Const(BUF[j, 5, w]), Enzyme.Const(omp), Enzyme.Const(g))
        _phm_grad(gpu, SC, dSC, ci, _phm_gather_pn2_dot, Enzyme.Duplicated(P, dP), Enzyme.Duplicated(Z, dZ),
            Enzyme.Duplicated(DZ, dDZ), Enzyme.Const(w), Enzyme.Const(ip), Enzyme.Const(r[3]))
        NBQ[1, i, w] = r[1].x1; NBQ[2, i, w] = r[1].h; NBQ[3, i, w] = r[1].oe
    end
    return nothing
end
@inline function _phm_psum_body!(w, GDM, NBQ, DM, TP)
    @inbounds begin
        TP[2, w] > 0 || return nothing
        dm = DM[w]
        a1 = a2 = a3 = 0.0
        for i in 1:dm.n-dm.npa
            a1 += NBQ[1, i, w]; a2 += NBQ[2, i, w]; a3 += NBQ[3, i, w]
        end
        g = GDM[w]
        GDM[w] = _nt_merge(g, (x1 = g.x1 + a1, h = g.h + a2, oe = g.oe + a3))
    end
    return nothing
end

# PA node outputs (BUF pr, pphi, omega_s copies; dt/dr, dphi/dr; the rates) from (r, PA final state)
@inline function _phm_panode_dot(dm, P, Z, DZ, r, pr, pp, om, w, g)
    b = KernelFused._phm_base(P, Z, DZ, dm, w)
    t1, t2 = GravitationalWaveforms._seobv5phm_pa_dtdr(r, pr, pp, om, b)
    a, f, c, d, oc = GravitationalWaveforms._seobv5phm_rhs(r, pr, pp, GravitationalWaveforms._seobv5phm_spins(b, om), b)
    return _rd(g, (pr, pp, om, t1, t2, a, f, c, d, oc))
end
@inline function _phm_panode_rev_body!(i, w, NBP, P, dP, Z, dZ, DZ, dDZ, BUF, dBUF, DM, PAS, dPAS, dPAY, TP, SC, dSC, gpu)
    @inbounds begin
        dm = DM[w]
        (TP[2, w] > 0) & dm.pa_success & (i <= dm.npa) || return nothing
        g = (dBUF[i, 4, w], dBUF[i, 5, w], dBUF[i, 10, w], dPAY[i, 1, w], dPAY[i, 2, w], dBUF[i, 6, w], dBUF[i, 7, w],
            dBUF[i, 8, w], dBUF[i, 9, w], dBUF[i, 11, w])
        r = _phm_grad(gpu, SC, dSC, CartesianIndex(i, w), _phm_panode_dot, Enzyme.Active(dm), Enzyme.Duplicated(P, dP),
            Enzyme.Duplicated(Z, dZ), Enzyme.Duplicated(DZ, dDZ), Enzyme.Active(BUF[i, 2, w]), Enzyme.Active(PAS[i, 1, 9, w]),
            Enzyme.Active(PAS[i, 2, 9, w]), Enzyme.Active(PAS[i, 3, 9, w]), Enzyme.Const(w), Enzyme.Const(g))
        NBP[1, i, w] += r[1].x1; NBP[2, i, w] += r[1].h; NBP[3, i, w] += r[1].oe
        dBUF[i, 2, w] += r[5]
        dPAS[i, 1, 9, w] += r[6]; dPAS[i, 2, 9, w] += r[7]; dPAS[i, 3, 9, w] += r[8]
    end
    return nothing
end
# t, phi integrals at the PA nodes (walker): their cotangents -> dt/dr, dphi/dr cotangents (dPAY) and the spacing
@inline function _phm_paint_rev_body!(w, DG, BUF, dBUF, SCR, DM, PAD, PAY, dPAY, TP)
    @inbounds begin
        dm = DM[w]
        (TP[2, w] > 0) & dm.pa_success || return nothing
        n = dm.npa
        h = BUF[2, 2, w] - BUF[1, 2, w]
        sb = view(SCR, :, 4, w)
        hb = 0.0
        for c in 1:2
            col = c == 1 ? 1 : 3
            for i in 1:n
                sb[i] = 0.0
            end
            hb += _phm_cumint_T!(view(dPAY, :, c, w), sb, view(dBUF, :, col, w), view(PAY, :, c, w), view(PAD, :, 8 + c, w), h, n)
            hb += _phm_slopes_T!(view(dPAY, :, c, w), sb, view(PAD, :, 8 + c, w), h, n, view(SCR, :, 2, w), view(SCR, :, 3, w))
        end
        DG[1, w] += hb
    end
    return nothing
end

# PA order k at node i: (r, pr, pphi, om, slope) -> the next order (pr, pphi, om)
@inline _phm_paupd_dot(dm, P, Z, DZ, r, pr, pphi, om, d, k, w, g) =
    _rd(g, GravitationalWaveforms._seobv5phm_pa_update(k, r, pr, pphi, om, d, KernelFused._phm_base(P, Z, DZ, dm, w)))
@inline function _phm_paupd_rev_body!(i, w, k, NBP, P, dP, Z, dZ, DZ, dDZ, BUF, dBUF, DM, PAS, dPAS, PAD, dPAD, TP, SC, dSC, gpu)
    @inbounds begin
        dm = DM[w]
        (TP[2, w] > 0) & dm.pa_success & (i <= dm.npa) || return nothing
        g = (dPAS[i, 1, k+1, w], dPAS[i, 2, k+1, w], dPAS[i, 3, k+1, w])
        r = _phm_grad(gpu, SC, dSC, CartesianIndex(i, w), _phm_paupd_dot, Enzyme.Active(dm), Enzyme.Duplicated(P, dP),
            Enzyme.Duplicated(Z, dZ), Enzyme.Duplicated(DZ, dDZ), Enzyme.Active(BUF[i, 2, w]), Enzyme.Active(PAS[i, 1, k, w]),
            Enzyme.Active(PAS[i, 2, k, w]), Enzyme.Active(PAS[i, 3, k, w]), Enzyme.Active(PAD[i, k, w]), Enzyme.Const(k),
            Enzyme.Const(w), Enzyme.Const(g))
        NBP[1, i, w] += r[1].x1; NBP[2, i, w] += r[1].h; NBP[3, i, w] += r[1].oe
        dBUF[i, 2, w] += r[5]
        dPAS[i, 1, k, w] += r[6]; dPAS[i, 2, k, w] += r[7]; dPAS[i, 3, k, w] += r[8]
        dPAD[i, k, w] = r[9]
    end
    return nothing
end
@inline function _phm_paslope_rev_body!(w, k, DG, BUF, SCR, DM, PAD, dPAS, dPAD, TP)
    @inbounds begin
        dm = DM[w]
        (TP[2, w] > 0) & dm.pa_success || return nothing
        h = BUF[2, 2, w] - BUF[1, 2, w]
        DG[1, w] += _phm_slopes_T!(view(dPAS, :, isodd(k) ? 2 : 1, k, w), view(dPAD, :, k, w), view(PAD, :, k, w), h, dm.npa,
            view(SCR, :, 2, w), view(SCR, :, 3, w))
    end
    return nothing
end
# adiabatic start at node i
# (from the stored converged pphi: one Newton step instead of the solver's loop, see `_seobv5phm_pa_adiabatic_from`)
@inline _phm_padiab_dot(dm, P, Z, DZ, r, w, jc, g) =
    _rd(g, GravitationalWaveforms._seobv5phm_pa_adiabatic_from(r, KernelFused._phm_base(P, Z, DZ, dm, w), jc))
@inline function _phm_padiab_rev_body!(i, w, NBP, P, dP, Z, dZ, DZ, dDZ, BUF, dBUF, DM, PAS, dPAS, TP, SC, dSC, gpu)
    @inbounds begin
        dm = DM[w]
        (TP[2, w] > 0) & dm.pa_success & (i <= dm.npa) || return nothing
        r = _phm_grad(gpu, SC, dSC, CartesianIndex(i, w), _phm_padiab_dot, Enzyme.Active(dm), Enzyme.Duplicated(P, dP),
            Enzyme.Duplicated(Z, dZ), Enzyme.Duplicated(DZ, dDZ), Enzyme.Active(BUF[i, 2, w]), Enzyme.Const(w),
            Enzyme.Const(PAS[i, 2, 1, w]), Enzyme.Const((dPAS[i, 2, 1, w], dPAS[i, 3, 1, w])))
        NBP[1, i, w] += r[1].x1; NBP[2, i, w] += r[1].h; NBP[3, i, w] += r[1].oe
        dBUF[i, 2, w] += r[5]
    end
    return nothing
end
# start (walker): the r grid, or the node-1 state when PA fails, then `_seobv5phm_dyn_start` (IC, PA grid decision)
# (from the stored converged initial conditions, see `_seobv5phm_dyn_start_from`: the solver's loops on an in-kernel
# reverse's tape were 15 ms per walker at 1024 walkers)
@inline _phm_ic_dot(dm, P, Z, DZ, w, ic, g) =
    _rd(g, GravitationalWaveforms._seobv5phm_dyn_start_from(KernelFused._phm_base(P, Z, DZ, dm, w), P[9, w], ic[1], ic[2], ic[3]))
@inline function _phm_node1_dot(dm, P, Z, DZ, r0, pr0, pphi0, w, g)
    b = KernelFused._phm_base(P, Z, DZ, dm, w)
    return _rd(g, GravitationalWaveforms._seobv5phm_rhs(r0, pr0, pphi0, GravitationalWaveforms._seobv5phm_spins(b, P[9, w]), b))
end
@inline function _phm_start_rev_body!(w, GDM, DG, NBP, P, dP, Z, dZ, DZ, dDZ, BUF, dBUF, DM, TP, SC, dSC, gpu)
    @inbounds begin
        TP[2, w] > 0 || return nothing
        dm = DM[w]
        gdm = GDM[w]
        for i in 1:KernelFused._PHM_NPA
            gdm = _nt_merge(gdm, (x1 = gdm.x1 + NBP[1, i, w], h = gdm.h + NBP[2, i, w], oe = gdm.oe + NBP[3, i, w]))
        end
        if dm.pa_success
            n = dm.npa
            ht = DG[1, w]
            r0b = 0.0
            for i in 1:n
                ht += (i - 1) * dBUF[i, 2, w]
                r0b += dBUF[i, 2, w]
            end
            g = (r0b - ht / (n - 1), 0.0, 0.0, ht / (n - 1))
        else
            # node 1 = (r0, pr0, pphi0, omega_s = omega_start) and its rates
            r = _phm_grad(gpu, SC, dSC, w, _phm_node1_dot, Enzyme.Active(dm), Enzyme.Duplicated(P, dP), Enzyme.Duplicated(Z, dZ),
                Enzyme.Duplicated(DZ, dDZ), Enzyme.Active(dm.r0), Enzyme.Active(dm.pr0), Enzyme.Active(dm.pphi0),
                Enzyme.Const(w), Enzyme.Const((dBUF[1, 6, w], dBUF[1, 7, w], dBUF[1, 8, w], dBUF[1, 9, w], dBUF[1, 11, w])))
            gdm = _ra(gdm, r[1])
            dP[9, w] += dBUF[1, 10, w]
            g = (dBUF[1, 2, w] + r[5], dBUF[1, 5, w] + r[7], dBUF[1, 4, w] + r[6], 0.0)
        end
        r = _phm_grad(gpu, SC, dSC, w, _phm_ic_dot, Enzyme.Active(dm), Enzyme.Duplicated(P, dP), Enzyme.Duplicated(Z, dZ),
            Enzyme.Duplicated(DZ, dDZ), Enzyme.Const(w), Enzyme.Const((dm.r0, dm.pphi0, dm.pr0)), Enzyme.Const(g))
        GDM[w] = _ra(gdm, r[1])
    end
    return nothing
end

# ---- (1') the PN spin table (walker): backward sweep over its RK4 steps, then over the carry steps, then the
# initial spins; and the grid scalars x1 = x(0.9 omega_start), h, omega_end into P (omega_start, omega_ref, m1, spins).
@inline _phm_nu(m1, m2) = m1 * m2 / (m1 + m2)^2
# the table's node count M (h = (x_hi - x_lo) / (M - 1), as `_seobv5phm_pn_fill!` picks it; piecewise constant)
@inline _phm_pn_m(x_lo, m1, m2) = GravitationalWaveforms._seobv5phm_pn_m(x_lo, 1 / GravitationalWaveforms._SEOBV5PHM_PN_OMEGA_MAX,
    KernelFused._PHM_NPN, GravitationalWaveforms._seobv5phm_pn_hstep(m1, m2))
@inline _phm_pnstep_dot(z, x, h, m1, m2, g) = sum(g .* GravitationalWaveforms._seobv5phm_pn_step(z, x, h, 3.0, _phm_nu(m1, m2), m1, m2)[1])
@inline _phm_pndx_dot(z, x, m1, m2, g) = sum(g .* GravitationalWaveforms._seobv5phm_pn_dx(z, x, 3.0, _phm_nu(m1, m2), m1, m2)[1])
@inline _phm_col9(A, i, w) = @inbounds SVector{9}(A[1, i, w], A[2, i, w], A[3, i, w], A[4, i, w], A[5, i, w], A[6, i, w],
    A[7, i, w], A[8, i, w], A[9, i, w])
@inline function _phm_pn_rev_body!(w, GDM, P, dP, Z, dZ, DZ, dDZ, ZC, DM, TP, SC, dSC, gpu)
    @inbounds begin
        TP[2, w] > 0 || return nothing
        dm = DM[w]; gdm = GDM[w]
        N = KernelFused._PHM_NPN
        n = dm.npn
        m1 = P[1, w]; m2 = 1 - m1
        x_lo = dm.x1; h = dm.h
        zb = zero(SVector{9,Float64})
        for i in N:-1:n+1                                  # columns past n hold the last state
            zb += _phm_col9(dZ, i, w)
        end
        xlb = gdm.x1; hb = gdm.h; m1b = 0.0; m2b = 0.0
        xe = x_lo + (n - 1) * h                             # omega_end = 1 / xe (p = 3)
        xlb -= gdm.oe / (xe * xe); hb -= gdm.oe * (n - 1) / (xe * xe)
        for i in n:-1:1
            x = x_lo + (i - 1) * h
            zi = _phm_col9(Z, i, w)
            xib = 0.0
            if i < n
                r = _phm_grad(gpu, SC, dSC, w, _phm_pnstep_dot, Enzyme.Active(zi), Enzyme.Active(x), Enzyme.Active(h),
                    Enzyme.Active(m1), Enzyme.Active(m2), Enzyme.Const(zb))
                zb = r[1]; xib += r[2]; hb += r[3]; m1b += r[4]; m2b += r[5]
            end                                         # (i = n: zb holds the padding columns')
            zb += _phm_col9(dZ, i, w)
            r = _phm_grad(gpu, SC, dSC, w, _phm_pndx_dot, Enzyme.Active(zi), Enzyme.Active(x), Enzyme.Active(m1),
                Enzyme.Active(m2), Enzyme.Const(_phm_col9(dDZ, i, w)))
            zb += r[1]; xib += r[2]; m1b += r[3]; m2b += r[4]
            xlb += xib; hb += (i - 1) * xib
        end
        xlb -= hb / (_phm_pn_m(x_lo, m1, m2) - 1)            # h = (x_hi - x_lo) / (M - 1)
        # carry steps x_ref -> x_lo
        x_ref = 1 / P[8, w]
        NB = clamp(ceil(Int, abs((x_lo - x_ref) / h)), 1, N)
        hc = (x_lo - x_ref) / NB
        hcb = 0.0; xrb = 0.0
        for j in NB-1:-1:0
            r = _phm_grad(gpu, SC, dSC, w, _phm_pnstep_dot, Enzyme.Active(_phm_col9(ZC, j + 1, w)), Enzyme.Active(x_ref + j * hc),
                Enzyme.Active(hc), Enzyme.Active(m1), Enzyme.Active(m2), Enzyme.Const(zb))
            zb = r[1]; xrb += r[2]; hcb += j * r[2] + r[3]; m1b += r[4]; m2b += r[5]
        end
        xlb += hcb / NB; xrb -= hcb / NB
        dP[9, w] -= xlb / (0.9 * P[9, w] * P[9, w])          # x_lo = 1 / (0.9 omega_start)
        dP[8, w] -= xrb / (P[8, w] * P[8, w])                # x_ref = 1 / omega_ref
        # z0 = (0, 0, 1, chi1 m1^2, chi2 m2^2)
        for c in 1:3
            dP[1+c, w] += zb[3+c] * m1 * m1
            dP[4+c, w] += zb[6+c] * m2 * m2
            m1b += 2 * m1 * P[1+c, w] * zb[3+c]
            m2b += 2 * m2 * P[4+c, w] * zb[6+c]
        end
        dP[1, w] += m1b - m2b
    end
    return nothing
end
# The same reverse in parallel over the steps (as the plunge, `_phm_pjac_rev_body!`): with the step Jacobian J_i
# (9 x 9), zb_i = J_i^T zb_{i+1} + c_i, c_i node i's own cotangent with its derivative's (and, at node n, the
# padding columns', which hold copies of the last state). Slot s <= N: table step s and the derivative at node s;
# slot N + 1 + j: carry step j. PNJ[1:81, s, w] = J (PNJ[9(a - 1) + b] = d z'_a / d z_b), PNJ[82:90] first c_s,
# then (`_phm_pnchain_body!`) the cotangent of the step's output; PNJ[91:94] the slot's (x, h, m1, m2) partials;
# ZB[:, w] the cotangent of the initial spins.
const _PHM_NPNJ = 94
@inline _phm_e9(a) = SVector{9}(ntuple(b -> Float64(a == b), Val(9)))
@inline _phm_pn_nb(dm, P, w) = @inbounds clamp(ceil(Int, abs((dm.x1 - 1 / P[8, w]) / dm.h)), 1, KernelFused._PHM_NPN)
@inline function _phm_pnslot(s, dm, P, Z, ZC, w)
    @inbounds begin
        N = KernelFused._PHM_NPN
        if s <= N
            return (s < dm.npn, _phm_col9(Z, s, w), dm.x1 + (s - 1) * dm.h, dm.h)
        end
        j = s - N - 1
        NB = _phm_pn_nb(dm, P, w)
        x_ref = 1 / P[8, w]
        hc = (dm.x1 - x_ref) / NB
        return (j < NB, _phm_col9(ZC, min(j + 1, N), w), x_ref + j * hc, hc)
    end
end
@inline function _phm_pnstep_st!(O, i, w, z, x, h, m1, m2)
    zn = GravitationalWaveforms._seobv5phm_pn_step(z, x, h, 3.0, _phm_nu(m1, m2), m1, m2)[1]
    @inbounds for a in 1:9
        O[a, i, w] = zn[a]
    end
    return nothing
end
@inline function _phm_pnjac_rev_body!(s, w, PNJ, JO, JS, P, Z, dZ, dDZ, ZC, DM, TP, SC, dSC, gpu)
    @inbounds begin
        TP[2, w] > 0 || return nothing
        dm = DM[w]
        N = KernelFused._PHM_NPN
        n = dm.npn
        m1 = P[1, w]
        ci = CartesianIndex(s, w)
        step, zi, x, h = _phm_pnslot(s, dm, P, Z, ZC, w)
        for k in 91:94
            PNJ[k, s, w] = 0.0
        end
        if s <= n
            c = _phm_col9(dZ, s, w)
            if s == n
                for i in n+1:N
                    c += _phm_col9(dZ, i, w)
                end
            end
            r = _phm_grad(gpu, SC, dSC, ci, _phm_pndx_dot, Enzyme.Active(zi), Enzyme.Active(x), Enzyme.Active(m1),
                Enzyme.Active(1 - m1), Enzyme.Const(_phm_col9(dDZ, s, w)))
            c += r[1]
            for b in 1:9
                PNJ[81+b, s, w] = c[b]
            end
            PNJ[91, s, w] = r[2]; PNJ[93, s, w] = r[3]; PNJ[94, s, w] = r[4]
        end
        step || return nothing
        # J (forward mode, the nine directions batched)
        _phm_fwd(gpu, _phm_pnstep_st!, _phm_jb(JO, JS, Val(9)), Enzyme.Const(s), Enzyme.Const(w),
            Enzyme.BatchDuplicated(zi, ntuple(_phm_e9, Val(9)), false), Enzyme.Const(x), Enzyme.Const(h), Enzyme.Const(m1),
            Enzyme.Const(1 - m1))
        for b in 1:9, a in 1:9
            PNJ[9a-9+b, s, w] = JS[b][a, s, w]
        end
    end
    return nothing
end
@inline _phm_pnjt(PNJ, s, w, zb) = @inbounds SVector{9}(ntuple(b -> PNJ[b, s, w] * zb[1] + PNJ[9+b, s, w] * zb[2] +
    PNJ[18+b, s, w] * zb[3] + PNJ[27+b, s, w] * zb[4] + PNJ[36+b, s, w] * zb[5] + PNJ[45+b, s, w] * zb[6] +
    PNJ[54+b, s, w] * zb[7] + PNJ[63+b, s, w] * zb[8] + PNJ[72+b, s, w] * zb[9], Val(9)))
@inline function _phm_pnchain_body!(w, PNJ, ZB, P, DM, TP)
    @inbounds begin
        TP[2, w] > 0 || return nothing
        dm = DM[w]
        N = KernelFused._PHM_NPN
        n = dm.npn
        zb = SVector{9}(ntuple(b -> PNJ[81+b, n, w], Val(9)))
        for i in n-1:-1:1
            c = SVector{9}(ntuple(b -> PNJ[81+b, i, w], Val(9)))
            for b in 1:9
                PNJ[81+b, i, w] = zb[b]
            end
            zb = _phm_pnjt(PNJ, i, w, zb) + c
        end
        for j in _phm_pn_nb(dm, P, w)-1:-1:0
            s = N + 1 + j
            for b in 1:9
                PNJ[81+b, s, w] = zb[b]
            end
            zb = _phm_pnjt(PNJ, s, w, zb)
        end
        for b in 1:9
            ZB[b, w] = zb[b]
        end
    end
    return nothing
end
@inline function _phm_pnpar_rev_body!(s, w, PNJ, P, Z, ZC, DM, TP, SC, dSC, gpu)
    @inbounds begin
        TP[2, w] > 0 || return nothing
        dm = DM[w]
        step, zi, x, h = _phm_pnslot(s, dm, P, Z, ZC, w)
        step || return nothing
        zb = SVector{9}(ntuple(b -> PNJ[81+b, s, w], Val(9)))
        iszero(zb) && return nothing
        m1 = P[1, w]
        r = _phm_grad(gpu, SC, dSC, CartesianIndex(s, w), _phm_pnstep_dot, Enzyme.Const(zi), Enzyme.Active(x),
            Enzyme.Active(h), Enzyme.Active(m1), Enzyme.Active(1 - m1), Enzyme.Const(zb))
        PNJ[91, s, w] += r[2]; PNJ[92, s, w] += r[3]; PNJ[93, s, w] += r[4]; PNJ[94, s, w] += r[5]
    end
    return nothing
end
# the scalars: as the end of `_phm_pn_rev_body!` from the slots' partials
@inline function _phm_pnsum_body!(w, GDM, PNJ, ZB, P, dP, DM, TP)
    @inbounds begin
        TP[2, w] > 0 || return nothing
        dm = DM[w]; gdm = GDM[w]
        N = KernelFused._PHM_NPN
        n = dm.npn
        m1 = P[1, w]; m2 = 1 - m1
        xlb = gdm.x1; hb = gdm.h; m1b = 0.0; m2b = 0.0
        xe = dm.x1 + (n - 1) * dm.h
        xlb -= gdm.oe / (xe * xe); hb -= gdm.oe * (n - 1) / (xe * xe)
        for i in 1:n
            xib = PNJ[91, i, w]
            xlb += xib; hb += (i - 1) * xib + PNJ[92, i, w]
            m1b += PNJ[93, i, w]; m2b += PNJ[94, i, w]
        end
        xlb -= hb / (_phm_pn_m(dm.x1, m1, m2) - 1)
        NB = _phm_pn_nb(dm, P, w)
        hcb = 0.0; xrb = 0.0
        for j in 0:NB-1
            s = N + 1 + j
            xrb += PNJ[91, s, w]; hcb += j * PNJ[91, s, w] + PNJ[92, s, w]
            m1b += PNJ[93, s, w]; m2b += PNJ[94, s, w]
        end
        xlb += hcb / NB; xrb -= hcb / NB
        dP[9, w] -= xlb / (0.9 * P[9, w] * P[9, w])
        dP[8, w] -= xrb / (P[8, w] * P[8, w])
        for c in 1:3
            dP[1+c, w] += ZB[3+c, w] * m1 * m1
            dP[4+c, w] += ZB[6+c, w] * m2 * m2
            m1b += 2 * m1 * P[1+c, w] * ZB[3+c, w]
            m2b += 2 * m2 * P[4+c, w] * ZB[6+c, w]
        end
        dP[1, w] += m1b - m2b
    end
    return nothing
end

# t_ref (`KernelFused._phm_tref_body!`): the t_c cotangent g (EG[2]) times -d(t_f M)/d(nodes i, i+1: t, Omega; M),
# t_f = t_i + s (t_i+1 - t_i), s = (pi f_ref M - Omega_i) / (Omega_i+1 - Omega_i) clamped to [0, 1]
@inline function _phm_tref_rev_body!(w, EG, dBUF, dP, BUF, DM, P, TP, fref)
    @inbounds begin
        (fref > 0) & (TP[2, w] > 0) || return nothing
        c = -EG[2, w]
        n = DM[w].n
        Ms = P[13, w]
        omr = pi * fref * Ms
        i = 1
        while (i < n - 1) & (BUF[i+1, 7, w] < omr)
            i += 1
        end
        t0, t1, o0, o1 = BUF[i, 1, w], BUF[i+1, 1, w], BUF[i, 7, w], BUF[i+1, 7, w]
        su = (omr - o0) / (o1 - o0)
        inside = (su > 0) & (su < 1)
        s = clamp(su, 0.0, 1.0)
        dt = t1 - t0
        tf = t0 + s * dt
        ds = ifelse(inside, Ms * dt, 0.0)                 # d(t_f M)/ds
        dBUF[i, 1, w] += c * Ms * (1 - s)
        dBUF[i+1, 1, w] += c * Ms * s
        dBUF[i, 7, w] += c * ds * (s - 1) / (o1 - o0)
        dBUF[i+1, 7, w] += c * ds * (-s) / (o1 - o0)
        dP[13, w] += c * (tf + ds * pi * fref / (o1 - o0))
    end
    return nothing
end

# ---- (0') walker: model inputs P -> the 15 physical rows (+ the direct extrinsic gradients of (6')) ----
@inline _phm_prep_dot(m1, m2, c1z, c2z, dL, ph, inc, c1x, c1y, c2x, c2y, fref, dt_s, fphi, g) =
    _rd(g, KernelFused._phm_prep_vals(m1, m2, c1z, c2z, dL, ph, inc, c1x, c1y, c2x, c2y, fref, dt_s, fphi))
const _PHM_PREP_ROWS = (1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15)
@inline function _phm_prep_rev_body!(w, G, params, dP, EG, TP, fref, dt_s, fphi, SC, dSC, gpu)
    @inbounds begin
        for k in 1:15
            G[k, w] = 0.0
        end
        TP[2, w] > 0 || return nothing
        g = ntuple(k -> dP[k, w] + (k == 14 ? EG[6, w] : k == 12 ? EG[7, w] : 0.0), Val(KernelFused._PHM_NP))
        a = KernelFused._phm_prep_args(params, w)
        r = _phm_grad(gpu, SC, dSC, w, _phm_prep_dot, Enzyme.Active(a[1]), Enzyme.Active(a[2]), Enzyme.Active(a[3]),
            Enzyme.Active(a[4]), Enzyme.Active(a[5]), Enzyme.Active(a[6]), Enzyme.Active(a[7]), Enzyme.Active(a[8]),
            Enzyme.Active(a[9]), Enzyme.Active(a[10]), Enzyme.Active(a[11]), Enzyme.Const(fref), Enzyme.Const(dt_s),
            Enzyme.Const(fphi), Enzyme.Const(g))
        # (unrolled: r is a mixed tuple, the Const arguments' entries are `nothing`; a runtime index into it is dynamic)
        ntuple(Val(length(_PHM_PREP_ROWS))) do j
            G[_PHM_PREP_ROWS[j], w] = r[j]
            nothing
        end
        G[5, w] += EG[1, w]; G[6, w] += EG[2, w]; G[8, w] += EG[3, w]; G[9, w] += EG[4, w]; G[11, w] += EG[5, w]
    end
    return nothing
end

# ---- the driver ----
struct _PHMRev{A2,A3,A4,XB,VR,VD,VP,VQ,DB,PB,IB,MB,JT}
    dP::A2; dZ::A3; dDZ::A3; dBUF::A3; dSCR::A3; dMD::A4; dGT::A2; dPAS::A4; dPAD::A3; dPAY::A3
    XB::XB; EG::A2; GREC::VR; GDM::VD; GPRE::VP; GQ::VQ; DMB::DB; NB::A3; NBP::A3; DG::A2
    SCw::A2; dSCw::A2; SCc::A2; dSCc::A2; SCn::A2; dSCn::A2
    GCB::A2; NBG::A3; SCg::A2; dSCg::A2; STB::A3; GPREB::PB; GDMB::DB
    RIB::IB; RMB::MB                  # record-subset cotangents per sample chunk (`SEOBNRPHMSampleRev.jl`)
    PJ::A3; NBQ::A3                   # plunge steps (`_phm_pjac_rev_body!`)
    PNJ::A3; ZB::A2                   # PN table steps (`_phm_pnjac_rev_body!`)
    SJ::A3                            # attachment-state steps (`_phm_stsjac_rev_body!`)
    JO::A3; JS::JT                    # the step chains' forward-mode Jacobian columns: primal, 9 tangent arrays
end
function _PHMRev(W)
    be = W.backend; nw = W.nw; N = W.N
    sh(A) = (B = similar(A); fill!(B, zero(eltype(A))); B)
    z(d...) = (A = KernelAbstractions.allocate(be, Float64, d...); fill!(A, 0.0); A)
    nc = _PHM_NSAMPT
    # the parallel step chains' buffers, each only up to its chain's threshold (1.8 GB at 1024 walkers)
    cp = _phm_chain_par_max()
    npl, npn, nst = (nw <= cp[c] ? nw : 0 for c in (:plunge, :pn, :sts))
    nj = max(npl, npn, nst)
    nd = max(KernelFused._PHM_NRK, 2 * KernelFused._PHM_NPN)
    return _PHMRev(sh(W.P), sh(W.Z), sh(W.DZ), sh(W.BUF), sh(W.SCR), sh(W.MD), sh(W.GT), sh(W.PAS), sh(W.PAD), sh(W.PAY),
        sh(W.X), z(8, nw), similar(W.REC), similar(W.DM), similar(W.RPRE), similar(W.QREC),
        KernelAbstractions.allocate(be, eltype(W.DM), nc, nw),
        z(_PHM_NNB, KernelFused._PHM_ND, nw), z(3, KernelFused._PHM_NPA, nw), z(1, nw),
        z(1, nw), z(1, nw), z(nc, nw), z(nc, nw), z(KernelFused._PHM_ND, nw), z(KernelFused._PHM_ND, nw),
        z(KernelFused._PHM_NPN, nw), z(_PHM_NBG, KernelFused._PHM_NPN, nw), z(KernelFused._PHM_NPN, nw),
        z(KernelFused._PHM_NPN, nw), z(5, _PHM_NAST, nw), KernelAbstractions.allocate(be, eltype(W.RPRE), 6, nw),
        KernelAbstractions.allocate(be, eltype(W.DM), 6, nw),
        KernelAbstractions.allocate(be, typeof(_phm_ri(KernelFused._phm_rec_proto())), nc, nw),
        KernelAbstractions.allocate(be, typeof(_phm_rm(KernelFused._phm_rec_proto())), nc, nw),
        z(24, KernelFused._PHM_NRK, npl), z(3, KernelFused._PHM_NRK, npl),
        z(_PHM_NPNJ, 2 * KernelFused._PHM_NPN, npn), z(9, npn), z(24, _PHM_NAST, nst),
        z(10, nd, nj), ntuple(_ -> z(10, nd, nj), npn > 0 ? 9 : 4))
end
const _PHM_REV = Ref{Any}(nothing)
# The step chains' reverses (plunge, PN table, attachment states) in parallel over the steps cost more work than
# the serial ones but not their per-walker latency: on a V100 they win below these walker counts
# (scripts/check_phm_chain_rev.jl, s par / serial: plunge 0.47 / 0.69 at 1536, 0.63 / 0.71 at 2048, 0.95 / 0.75 at 3072;
# PN 0.23 / 0.18 at 1536; attachment states 0.085 / 0.101 at 1024, 0.12 / 0.11 at 1536; the parallel cost grows
# linearly in the walkers, the serial one barely). `GRAVITATIONALWAVEBAYES_PHM_CHAIN_PAR_{PLUNGE,PN,STS}` override them (`GPU.tuning`).
_phm_chain_par_max() = (plunge = GPU.tuning(:PHM_CHAIN_PAR_PLUNGE, 2304), pn = GPU.tuning(:PHM_CHAIN_PAR_PN, 1152),
    sts = GPU.tuning(:PHM_CHAIN_PAR_STS, 1280))
_phm_chain_par(chain_rev, c, nw) = chain_rev === :par || (chain_rev === :auto && nw <= _phm_chain_par_max()[c])
function _phm_rev(W)
    R = _PHM_REV[]
    (R isa _PHMRev && size(R.XB) == size(W.X) && typeof(R.XB) == typeof(W.X)) && return R
    _PHM_REV[] = nothing
    GC.gc(true); GPU.reclaim()         # free the previous reverse buffers before allocating (see `_phm_work`)
    R = _PHMRev(W)
    _PHM_REV[] = R
    return R
end

"""
    seobnr_phm_device_logl_grad!(ll, G, params, cfg, data; backend, nfft = 0, timers = nothing, sample_rev = :win, chain_rev = :auto)

SEOBNRv5PHM log-likelihood (as `KernelFused.seobnr_phm_device_loglikelihood!`) and its gradient `G` (15 x walkers)
along the physical rows of `params` (m1, m2, chi1z, chi2z, dL, t_c, phase, ra, dec, incl, psi, chi1x, chi1y, chi2x,
chi2y) on `backend`: the primal stages, then their reverse (this file). Dead walkers get zero gradient.
`sample_rev`: the sample reverse's scatter (`SEOBNRPHMSampleRev.jl`): `:winacc` adds the node cotangents of
consecutive samples on one window before scattering them, `:win` scatters every sample's.
`chain_rev`: the plunge, PN-table and attachment-state reverses serial over their steps (`:serial`), in parallel
over them (`:par`), or each by the walker count (`:auto`, `_phm_chain_par_max`).
"""
function seobnr_phm_device_logl_grad!(ll, G, params, cfg::PEConfig, data; backend = KernelAbstractions.CPU(),
        nfft::Int = 0, timers = nothing, sample_rev::Symbol = :win, chain_rev::Symbol = :auto, cal = nothing, CG = nothing)
    # Enzyme's in-kernel reverse caches any loop left in a differentiated function on the device malloc heap (serialized,
    # and the 8 MB default overflowed at 33 walkers before the loops were taken out of the reverses): a margin of
    # 256 KB per walker
    backend isa KernelAbstractions.CPU || GPU.ensure_device_heap(max(64, size(params, 2) >> 2) << 20)
    KernelFused.seobnr_phm_device_loglikelihood!(ll, params, cfg, data; backend = backend, nfft = nfft, timers = timers, cal = cal)
    cal isa Tuple && fill!(CG, 0.0)            # calibration node gradients (`_phm_cal_rev!`), d logL / d node
    W = KernelFused._PHM_WORK[]
    R = _phm_rev(W)
    geo = data isa AbstractVector
    dets = geo ? data : [data]
    ndet = length(dets)
    nw = W.nw; N = W.N
    gmst_val = gmst(dets[1].gps_time)
    ft = ntuple(i -> dets[i].freqs, ndet); st = ntuple(i -> dets[i].strain, ndet); pt = ntuple(i -> dets[i].psd, ndet)
    geoms = ntuple(i -> geometry(dets[i].detector), ndet)
    fh = Array(ft[1][1:2]); df = fh[2] - fh[1]
    gpu = Val(!(backend isa KernelAbstractions.CPU))
    g = !(backend isa KernelAbstractions.CPU)
    wg = g ? KernelFused._phm_wg(nw) : 1
    sync() = KernelAbstractions.synchronize(backend)
    t0 = Ref(time())
    stamp!(k) = timers === nothing || (sync(); t = time(); timers[k] = get(timers, k, 0.0) + t - t0[]; t0[] = t)
    wk(body, args) = KernelFused._phm_walker_k!(backend)(body, args; ndrange = nw, workgroupsize = wg)
    nk(body, args, n1) = KernelFused._phm_sample_k!(backend)(body, args; ndrange = (n1, nw),
        workgroupsize = g ? (32, 1) : (n1, 1))
    for A in (R.dP, R.dZ, R.dDZ, R.dBUF, R.dSCR, R.dMD, R.dGT, R.dPAS, R.dPAD, R.dPAY, R.NBP, R.DG)
        fill!(A, 0.0)
    end
    copyto!(R.GDM, map(_rz, Array(W.DM)))
    stamp!(:rev_zero)
    # (6'), (5')
    if cal isa KernelFused._PHMCalMarg
        # alpha = A' G1, beta = B^H G2 into the forward V, U buffers (no longer needed)
        LinearAlgebra.mul!(cal.V, transpose(cal.A), cal.G1)
        LinearAlgebra.mul!(cal.U, adjoint(cal.B), cal.G2)
        wk(_phm_sums_rev_cm_body!, (ll, R.EG, R.XB, cal.V, cal.U, cal.GDL, params, W.P, W.TP, W.X, N, df, ft, st, pt, geoms,
            gmst_val, KernelFused._lik_array(cfg, backend), cfg.time_reference, geo, R.SCw, R.dSCw, gpu, cal.nb))
    else
        wk(_phm_sums_rev_body!, (ll, R.EG, R.XB, params, W.P, W.TP, W.X, N, df, ft, st, pt, geoms, gmst_val,
            KernelFused._lik_array(cfg, backend), cfg.time_reference, geo, R.SCw, R.dSCw, gpu, cal, CG))
    end
    wk(_phm_tref_rev_body!, (R.EG, R.dBUF, R.dP, W.BUF, W.DM, W.P, W.TP, Float64(cfg.time_ref_frequency)))
    KernelFused.FFTW.plan_bfft!(R.XB, 1) * R.XB
    stamp!(:rev_sums_fft)
    # (4')
    nc = _PHM_NSAMPT
    sample_rev in (:win, :winacc) || throw(ArgumentError("sample_rev: :win or :winacc"))
    chain_rev in (:auto, :par, :serial) || throw(ArgumentError("chain_rev: :auto, :par or :serial"))
    chain_rev === :par && min(size(R.PJ, 3), size(R.PNJ, 3), size(R.SJ, 3)) < nw &&
        throw(ArgumentError("chain_rev = :par: buffers allocated up to $(minimum(_phm_chain_par_max())) walkers"))
    nk(_phm_wsample_rev_body!, (R.RIB, R.RMB, R.DMB, R.XB, W.P, R.dP, W.Z, R.dZ, W.DZ, R.dDZ, W.BUF, R.dBUF, W.MD, R.dMD, W.GT,
        R.dGT, W.REC, W.DM, W.TP, N, R.SCc, R.dSCc, gpu, Val(sample_rev === :winacc)), nc)
    wk(_phm_wchunk_sum_body!, (R.GREC, R.GDM, R.RIB, R.RMB, R.DMB, W.REC, nc))
    stamp!(:rev_samples)
    # (3')
    ND = KernelFused._PHM_ND
    _phm_post_rev!(backend, nw, R.GPRE, R.GQ, R.GDM, R.GREC, R.EG, W.P, R.dP, W.Z, R.dZ, W.DZ, R.dDZ, W.BUF, R.dBUF, W.SCR,
        R.dSCR, W.MD, R.dMD, W.GT, R.dGT, W.RPRE, W.QREC, W.REC, W.DM, W.TM, W.TP, R.SCw, R.dSCw, R.GPREB, R.GDMB, R.SCn, R.dSCn,
        gpu; stamp! = stamp!)
    NPN = KernelFused._PHM_NPN
    wk(_phm_gsuf_body!, (R.GCB, R.dGT, W.DM, W.TP))
    nk(_phm_ginc_rev_body!, (R.NBG, R.GCB, W.P, R.dP, W.Z, R.dZ, W.DZ, R.dDZ, W.DM, W.REC, W.TP, R.SCg, R.dSCg, gpu), NPN)
    wk(_phm_gsum_body!, (R.GPRE, R.GDM, R.NBG, W.DM, W.RPRE, W.TP, R.SCw, R.dSCw, gpu))
    stamp!(:rev_gamma)
    nk(_phm_nderiv_rev_body!, (W.BUF, R.dBUF, W.MD, R.dMD, W.DM, W.TP, gpu), ND)
    stamp!(:rev_nderiv)
    nk(_phm_nmodes_rev_body!, (R.NB, W.P, R.dP, W.Z, R.dZ, W.DZ, R.dDZ, W.BUF, R.dBUF, W.SCR, R.dSCR, W.MD, R.dMD, W.DM,
        W.RPRE, W.TP, gpu), ND)
    wk(_phm_nb_sum_body!, (R.GPRE, R.GDM, R.NB, W.DM))
    stamp!(:rev_nmodes)
    _phm_pre_rev!(backend, nw, R.GPRE, R.GDM, W.P, R.dP, W.Z, R.dZ, W.DZ, R.dDZ, W.BUF, R.dBUF, W.SCR, R.dSCR, R.STB, W.RPRE, W.DM,
        W.TP, R.SCw, R.dSCw, gpu; stamp! = stamp!, par = _phm_chain_par(chain_rev, :sts, nw) ? (R.SJ, R.JO, R.JS, R.SCn, R.dSCn) : nothing)
    # (2')
    NPA = KernelFused._PHM_NPA
    if !_phm_chain_par(chain_rev, :plunge, nw)
        wk(_phm_plunge_rev_body!, (R.GDM, W.P, R.dP, W.Z, R.dZ, W.DZ, R.dDZ, W.BUF, R.dBUF, W.DM, W.TP, R.SCw, R.dSCw, gpu))
    else
        NRK = KernelFused._PHM_NRK
        wk(_phm_ptot_body!, (R.dBUF, W.DM, W.TP))
        nk(_phm_pjac_rev_body!, (R.PJ, R.JO, R.JS, W.P, W.Z, W.DZ, W.BUF, R.dBUF, W.DM, W.TP, gpu),
            NRK)
        stamp!(:rev_plunge_jac)
        wk(_phm_pchain_body!, (R.PJ, R.dP, R.dBUF, W.DM, W.TP))
        nk(_phm_ppar_rev_body!, (R.PJ, R.NBQ, W.P, R.dP, W.Z, R.dZ, W.DZ, R.dDZ, W.BUF, R.dBUF, W.DM, W.TP, R.SCn, R.dSCn, gpu), NRK)
        wk(_phm_psum_body!, (R.GDM, R.NBQ, W.DM, W.TP))
    end
    stamp!(:rev_plunge)
    wk(_phm_paint_rev_body!, (R.DG, W.BUF, R.dBUF, W.SCR, W.DM, W.PAD, W.PAY, R.dPAY, W.TP))
    nk(_phm_panode_rev_body!, (R.NBP, W.P, R.dP, W.Z, R.dZ, W.DZ, R.dDZ, W.BUF, R.dBUF, W.DM, W.PAS, R.dPAS, R.dPAY, W.TP,
        R.SCn, R.dSCn, gpu), NPA)
    stamp!(:rev_pa_nodes)
    for k in 8:-1:1
        nk(_phm_paupd_rev_body!, (k, R.NBP, W.P, R.dP, W.Z, R.dZ, W.DZ, R.dDZ, W.BUF, R.dBUF, W.DM, W.PAS, R.dPAS, W.PAD,
            R.dPAD, W.TP, R.SCn, R.dSCn, gpu), NPA)
        wk(_phm_paslope_rev_body!, (k, R.DG, W.BUF, W.SCR, W.DM, W.PAD, R.dPAS, R.dPAD, W.TP))
    end
    stamp!(:rev_pa_orders)
    nk(_phm_padiab_rev_body!, (R.NBP, W.P, R.dP, W.Z, R.dZ, W.DZ, R.dDZ, W.BUF, R.dBUF, W.DM, W.PAS, R.dPAS, W.TP, R.SCn,
        R.dSCn, gpu), NPA)
    stamp!(:rev_pa_adiab)
    wk(_phm_start_rev_body!, (R.GDM, R.DG, R.NBP, W.P, R.dP, W.Z, R.dZ, W.DZ, R.dDZ, W.BUF, R.dBUF, W.DM, W.TP, R.SCw, R.dSCw,
        gpu))
    stamp!(:rev_pa_start)
    # (1'), (0')
    if !_phm_chain_par(chain_rev, :pn, nw)
        wk(_phm_pn_rev_body!, (R.GDM, W.P, R.dP, W.Z, R.dZ, W.DZ, R.dDZ, W.ZC, W.DM, W.TP, R.SCw, R.dSCw, gpu))
    else
        NS = 2 * KernelFused._PHM_NPN
        nk(_phm_pnjac_rev_body!, (R.PNJ, R.JO, R.JS, W.P, W.Z, R.dZ, R.dDZ, W.ZC, W.DM, W.TP, R.SCn, R.dSCn, gpu), NS)
        stamp!(:rev_pn_jac)
        wk(_phm_pnchain_body!, (R.PNJ, R.ZB, W.P, W.DM, W.TP))
        nk(_phm_pnpar_rev_body!, (R.PNJ, W.P, W.Z, W.ZC, W.DM, W.TP, R.SCn, R.dSCn, gpu), NS)
        wk(_phm_pnsum_body!, (R.GDM, R.PNJ, R.ZB, W.P, R.dP, W.DM, W.TP))
    end
    stamp!(:rev_pn)
    wk(_phm_prep_rev_body!, (G, params, R.dP, R.EG, W.TP, fh[1], 1 / (N * df), Float64(cfg.phase_ref_frequency), R.SCw,
        R.dSCw, gpu))
    sync()
    stamp!(:rev_prep)
    return ll, G
end

# ---- the sampler interface: sampled coordinates (mc, q, chi1z, chi2z, dL, t_c, phase, ra, sin dec, cos incl, psi,
# chi1x, chi1y, chi2x, chi2y) <-> the physical rows, and the chain rule back through `_eobimr_physical` ----
@inline _phm_phys_dot(mc, q, sdec, cth, g) = _rd(g, _eobimr_physical(mc, q, sdec, cth))
function _phm_to_physical!(phys, positions)
    @inbounds for w in axes(positions, 2)
        for k in 1:15
            phys[k, w] = positions[k, w]
        end
        m1, m2, dec, th = _eobimr_physical(positions[1, w], positions[2, w], positions[9, w], positions[10, w])
        phys[1, w] = m1; phys[2, w] = m2; phys[9, w] = dec; phys[10, w] = th
    end
    return phys
end
function _phm_chain!(grad_out, G, positions)
    @inbounds for w in axes(positions, 2)
        for k in 1:15
            grad_out[k, w] = G[k, w]
        end
        g4 = (G[1, w], G[2, w], G[9, w], G[10, w])
        all(iszero, g4) && continue
        r = Enzyme.autodiff(Enzyme.Reverse, _phm_phys_dot, Enzyme.Active, Enzyme.Active(positions[1, w]),
            Enzyme.Active(positions[2, w]), Enzyme.Active(positions[9, w]), Enzyme.Active(positions[10, w]), Enzyme.Const(g4))[1]
        grad_out[1, w] = r[1]; grad_out[2, w] = r[2]; grad_out[9, w] = r[3]; grad_out[10, w] = r[4]
    end
    return grad_out
end


# Calibration for the sampler closures (`Calibration.calibration_prior`): positions rows 16 : 15 + NCAL are
# standard normals z, the node values mu + sigma z. Returns the device tuple for the kernels and a refresher.
function _phm_cal_setup(calibration, backend, nwalkers)
    calibration === nothing && return nothing, nothing, 0
    ncal = length(calibration.mu)
    CALh = zeros(ncal, nwalkers)
    CALd = KernelAbstractions.allocate(backend, Float64, ncal, nwalkers)
    WCd = KernelAbstractions.allocate(backend, Float64, size(calibration.W))
    copyto!(WCd, calibration.W)
    refresh! = function (positions)
        @inbounds for w in axes(positions, 2), r in 1:ncal
            CALh[r, w] = calibration.mu[r] + calibration.sigma[r] * positions[15 + r, w]
        end
        copyto!(CALd, CALh)
        return CALh
    end
    return (CALd, WCd, Val(calibration.nn)), refresh!, ncal
end

"""
    seobnr_phm_logp_and_grad(data, cfg; backend, nwalkers) -> logp_and_grad!(grad_out, logp_out, positions, active_mask)

SEOBNRv5PHM log-posterior (log-likelihood + the package's flat box prior) and gradient in the sampled coordinates
for host `positions` (15 x nwalkers), computed on `backend` by `seobnr_phm_device_logl_grad!` (stagewise reverse).
"""
function seobnr_phm_logp_and_grad(data, cfg::PEConfig{T}; backend, nwalkers::Int, calibration = nothing,
        calmarg = nothing, boxf = _call_log_prior_vec) where {T}
    cpu = backend isa KernelAbstractions.CPU
    dets = cpu ? data : _data_to_device(data)
    P = KernelAbstractions.allocate(backend, Float64, 15, nwalkers)
    G = KernelAbstractions.allocate(backend, Float64, 15, nwalkers)
    ll = KernelAbstractions.allocate(backend, Float64, nwalkers)
    phys = zeros(15, nwalkers)
    cal, calrefresh!, ncal = _phm_cal_setup(calibration, backend, nwalkers)
    # `calmarg`: response curves (`calibration_curves(...).C`) to marginalize over instead of sampling the calibration
    (calmarg === nothing || calibration === nothing) || throw(ArgumentError("calibration: sample OR marginalize, not both"))
    calmarg === nothing || (cal = KernelFused.phm_calmarg(calmarg, data, nwalkers; backend = backend))
    CG = ncal > 0 ? KernelAbstractions.allocate(backend, Float64, ncal, nwalkers) : nothing
    return function (grad_out::AbstractMatrix{T}, logp_out::AbstractVector{T}, positions::AbstractMatrix{T}, active_mask)
        box = boxf(positions)            # the package box, or the configured prior's support
        if !any(isfinite, box)
            fill!(logp_out, T(-Inf)); fill!(grad_out, zero(T))
            return nothing
        end
        size(positions, 1) == 15 + ncal || throw(ArgumentError("expected $(15 + ncal) rows, got $(size(positions, 1))"))
        _phm_to_physical!(phys, positions)
        _phm_substitute_outside!(phys, box)
        copyto!(P, phys)
        ncal > 0 && calrefresh!(positions)
        seobnr_phm_device_logl_grad!(ll, G, P, cfg, dets; backend = backend, cal = cal, CG = CG)
        _phm_chain!(grad_out, Array(G), positions)
        if ncal > 0
            CGh = Array(CG)
            @inbounds for w in axes(positions, 2), r in 1:ncal
                grad_out[15 + r, w] = calibration.sigma[r] * CGh[r, w]
            end
        end
        copyto!(logp_out, Array(ll))
        logp_out .+= box
        @inbounds for w in eachindex(box)
            isfinite(box[w]) || (grad_out[:, w] .= zero(T))
        end
        return nothing
    end
end

# Walkers outside the prior's support (e.g. spin magnitudes > 1, reachable inside the box) are never evaluated: the
# waveform can throw there (InexactError at |chi1| = 1.27), which aborts the whole device batch -- and the
# gradient path has no fallback. Their physical columns are replaced by a valid walker's (the batch keeps its shape);
# the caller gives them -Inf (and a zero gradient) from the prior.
function _phm_substitute_outside!(phys, box)
    v = findfirst(isfinite, box)
    v === nothing && return phys
    @inbounds for w in eachindex(box)
        isfinite(box[w]) || (phys[:, w] .= view(phys, :, v))
    end
    return phys
end

"""
    seobnr_phm_loglik(data, cfg; backend, nwalkers) -> loglik!(ll_out, positions)

SEOBNRv5PHM log-likelihood alone (plus the package's flat box prior, i.e. `-Inf` outside the box) in the sampled
coordinates for host `positions` (15 x nwalkers), computed on `backend` by `seobnr_phm_device_loglikelihood!`. For
gradient-free samplers (`smc_sample(...; moves = :rw)`): no gradient kernel is compiled and no reverse-pass device heap
is reserved.

If the device call throws (an exception inside a kernel aborts the whole batch), the batch is re-evaluated walker by
walker on the KernelAbstractions CPU backend at the same FFT length; walkers that throw there get `-Inf` and are
reported (parameters, and the first exception with its stack trace) -- so a sampler survives a waveform failure at a
rare parameter point, and the point is recorded for a fix.
"""
function seobnr_phm_loglik(data, cfg::PEConfig{T}; backend, nwalkers::Int, calibration = nothing,
        calmarg = nothing, boxf = _call_log_prior_vec) where {T}
    cpu = backend isa KernelAbstractions.CPU
    dets = cpu ? data : _data_to_device(data)
    freqs_host = Array((data isa AbstractVector ? data[1] : data).freqs)
    P = KernelAbstractions.allocate(backend, Float64, 15, nwalkers)
    ll = KernelAbstractions.allocate(backend, Float64, nwalkers)
    phys = zeros(15, nwalkers)
    cal, calrefresh!, ncal = _phm_cal_setup(calibration, backend, nwalkers)
    (calmarg === nothing || calibration === nothing) || throw(ArgumentError("calibration: sample OR marginalize, not both"))
    calmarg === nothing || (cal = KernelFused.phm_calmarg(calmarg, data, nwalkers; backend = backend))
    nfail = Ref(0)
    # `curve_ll` (calibration marginalization only): receives the per-curve log-likelihoods (ncurves x nwalkers), e.g. to
    # draw each sample's response curve afterwards (the calibration posterior)
    return function (ll_out::AbstractVector{T}, positions::AbstractMatrix{T}; curve_ll = nothing)
        box = boxf(positions)            # the package box, or the configured prior's support
        if !any(isfinite, box)
            fill!(ll_out, T(-Inf))
            return ll_out
        end
        size(positions, 1) == 15 + ncal || throw(ArgumentError("expected $(15 + ncal) rows, got $(size(positions, 1))"))
        _phm_to_physical!(phys, positions)
        _phm_substitute_outside!(phys, box)
        copyto!(P, phys)
        CALh = ncal > 0 ? calrefresh!(positions) : nothing
        N = KernelFused._phm_nfft(freqs_host, minimum(view(phys, 1, :) .+ view(phys, 2, :)))
        try
            KernelFused.seobnr_phm_device_loglikelihood!(ll, P, cfg, dets; backend = backend, nfft = N, cal = cal)
            copyto!(ll_out, Array(ll))
            (curve_ll === nothing || !(cal isa KernelFused._PHMCalMarg)) || copyto!(curve_ll, Array(cal.LK))
        catch err
            cpu && rethrow()
            @warn "SEOBNRv5PHM device likelihood threw; re-evaluating the batch walker by walker on the CPU" exception = err
            _phm_loglik_cpu_fallback!(ll_out, phys, cfg, data, N, nfail;
                                      cal = ncal > 0 ? (CALh, calibration.W, Val(calibration.nn)) :
                                            calmarg === nothing ? nothing :
                                            KernelFused.phm_calmarg(calmarg, data, 1; backend = KernelAbstractions.CPU()))
            _phm_device_bisect(phys, cfg, dets, backend, N)
        end
        ll_out .+= box
        return ll_out
    end
end

# A batch threw on the device: find the walkers that throw there by bisection (re-running halves of the batch, ~2 log2(n)
# small device calls) and print their physical parameters -- the CPU fallback may evaluate them fine (a device-only
# failure), so this is the only record of the point. Best effort: never throws.
function _phm_device_bisect(phys, cfg, dets, backend, N)
    try
        throws(cols) = try
            P = KernelAbstractions.allocate(backend, Float64, 15, length(cols))
            ll = KernelAbstractions.allocate(backend, Float64, length(cols))
            copyto!(P, phys[:, cols])
            KernelFused.seobnr_phm_device_loglikelihood!(ll, P, cfg, dets; backend = backend, nfft = N)
            false
        catch
            true
        end
        stack = [collect(axes(phys, 2))]
        found = 0
        while !isempty(stack) && found < 4
            cols = pop!(stack)
            throws(cols) || continue
            if length(cols) == 1
                found += 1
                println("SEOBNRv5PHM device likelihood throws (device only?) at physical parameters ", repr(phys[:, cols[1]]))
            else
                h = length(cols) ÷ 2
                push!(stack, cols[h+1:end], cols[1:h])
            end
        end
        found == 0 && println("SEOBNRv5PHM device likelihood: no single walker reproduces the throw (transient?)")
        flush(stdout)
    catch err
        println("SEOBNRv5PHM device bisection failed: ", sprint(showerror, err)[1:min(end, 200)])
    end
    return nothing
end

# Walker-by-walker CPU (KernelAbstractions) evaluation of a batch whose device call threw: -Inf for walkers that throw.
function _phm_loglik_cpu_fallback!(ll_out, phys, cfg, data, N, nfail; cal = nothing)
    be = KernelAbstractions.CPU()
    l1 = zeros(1)
    for w in axes(phys, 2)
        try
            calw = cal isa Tuple ? (cal[1][:, w:w], cal[2], cal[3]) : cal    # (marginalization: 1-walker buffers)
            KernelFused.seobnr_phm_device_loglikelihood!(l1, phys[:, w:w], cfg, data; backend = be, nfft = N, cal = calw)
            ll_out[w] = l1[1]
        catch err
            ll_out[w] = -Inf
            nfail[] += 1
            println("SEOBNRv5PHM likelihood throws at physical parameters ", repr(phys[:, w]), ": ",
                    sprint(showerror, err)[1:min(end, 300)])
            nfail[] == 1 && (showerror(stdout, err, catch_backtrace()); println())
            flush(stdout)
        end
    end
    return ll_out
end

# The CPU per-walker gradient entry (`gw_logp_and_grad` on the CPU): the same stagewise reverse on the
# KernelAbstractions CPU backend (Enzyme through the allocating CPU `hphc` fails to compile: an illegal address-space
# cast in the GC-invariant verifier).
function _eobimr_gradient!(grad_out::AbstractMatrix{T}, positions::AbstractMatrix{T}, freqs, cfg::PEConfig{T}, data,
        model::SEOBNRv5PHM; ll_out = nothing) where {T}
    nwalkers = size(positions, 2)
    size(positions, 1) == 15 || throw(ArgumentError("SEOBNRv5PHM samples 15 parameters, got $(size(positions, 1))"))
    phys = _phm_to_physical!(zeros(15, nwalkers), positions)
    ll = zeros(nwalkers); G = zeros(15, nwalkers)
    seobnr_phm_device_logl_grad!(ll, G, phys, cfg, data; backend = KernelAbstractions.CPU())
    _phm_chain!(grad_out, G, positions)
    ll_out === nothing || (ll_out .= ll)
    return nothing
end
