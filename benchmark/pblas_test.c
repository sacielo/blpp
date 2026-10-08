/*
 * pblas_test.c - verification for the pblas.h higher-level interface.
 * Runs all 13 physics extension operations at s, d, c, z precision and
 * compares against a long double reference, plus strided smoke cases.
 * Plain C99 (proves the header compiles as C).
 *
 * Build against an installed OpenBLAS (see CMakeLists.txt):
 *   gcc -O2 -I ../pkgs/openblas/include/openblas pblas_test.c \
 *       -L ../pkgs/openblas/lib -lopenblas -lm -o /tmp/pblas_test
 */
#include "pblas.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef long double ld;

#define N 8
#define VAL(S, i) ((ld)((S) + (i)) * 0.1L * (((i) % 3) == 0 ? -1.0L : 1.0L))

static int fails = 0;
static const char *prec = "?";

static void chk(const char *op, ld e, ld eps)
{
    if (e > eps) {
        printf("%s %-24s max_rel_err = %.3e  FAIL\n", prec, op, (double)e);
        fails++;
    } else {
        printf("%s %-24s max_rel_err = %.3e  ok\n", prec, op, (double)e);
    }
}

static ld rabs(ld a) { return a < 0 ? -a : a; }
static ld m1(ld a) { a = rabs(a); return a < 1 ? 1 : a; }
static ld mx(ld a, ld b) { return a > b ? a : b; }

/* long double "complex real" arithmetic for the reference */
typedef struct { ld re, im; } cr;

static cr cmul(cr a, cr b)
{
    cr c;
    c.re = a.re * b.re - a.im * b.im;
    c.im = a.re * b.im + a.im * b.re;
    return c;
}
static cr cadd(cr a, cr b) { cr c = { a.re + b.re, a.im + b.im }; return c; }

static void ccross3(const cr x[3], const cr y[3], cr w[3])
{
    cr p = cmul(x[1], y[2]);
    cr q = cmul(x[2], y[1]);
    w[0].re = p.re - q.re; w[0].im = p.im - q.im;
    p = cmul(x[2], y[0]);
    q = cmul(x[0], y[2]);
    w[1].re = p.re - q.re; w[1].im = p.im - q.im;
    p = cmul(x[0], y[1]);
    q = cmul(x[1], y[0]);
    w[2].re = p.re - q.re; w[2].im = p.im - q.im;
}

static cr cdot3c(const cr x[3], const cr w[3])
{
    cr s = { 0, 0 }, p;
    for (int c = 0; c < 3; c++) { p = cmul(x[c], w[c]); s = cadd(s, p); }
    return s;
}

static cr cnorm1(cr q)
{
    ld p0 = q.re * q.re - q.im * q.im;
    ld p1 = 2.0L * q.re * q.im;
    ld m = sqrtl(p0 * p0 + p1 * p1);
    cr r;
    r.re = sqrtl((m + p0) / 2.0L);
    r.im = (p1 >= 0) ? sqrtl((m - p0) / 2.0L) : -sqrtl((m - p0) / 2.0L);
    return r;
}

/* reference values for element i; seeds: x=(b,b+1,b+2), y=(b+3..b+5),
 * w=(b+6..b+8), q=b+9, t=b+10, yv=b+11, xv=b+12 */
typedef struct {
    cr ax, xy, dot, sqr, norm, crd, crsq, xw;
    cr had[3], crw[3], crs[3], u[3], v[3];
} refset;

static void refset_fill(int i, int b, int iscx, refset *R)
{
    cr x[3], y[3], w[3], q, t, xv, yv, s, a;
    ld im = iscx ? 0.7L : 0.0L;
    for (int c = 0; c < 3; c++) {
        ld v = VAL(b + c, i);
        x[c] = (cr){ v, v * im };
        v = VAL(b + 3 + c, i);
        y[c] = (cr){ v, v * im };
        v = VAL(b + 6 + c, i);
        w[c] = (cr){ v, v * im };
    }
    q = (cr){ VAL(b + 9, i), VAL(b + 9, i) * im };
    t = (cr){ VAL(b + 10, i), VAL(b + 10, i) * im };
    xv = (cr){ VAL(b + 12, i), VAL(b + 12, i) * im };
    yv = (cr){ VAL(b + 11, i), VAL(b + 11, i) * im };
    a = (cr){ 1.5L, iscx ? 0.5L : 0.0L };
    R->ax = cadd(cmul(a, xv), yv);
    R->xy = cadd(cmul(q, t), (cr){ 0.25L, iscx ? 0.1L : 0.0L });
    R->dot = cdot3c(x, y);
    for (int c = 0; c < 3; c++) R->had[c] = cmul(x[c], y[c]);
    s = (cr){ 0, 0 };
    for (int c = 0; c < 3; c++) s = cadd(s, cmul(x[c], x[c]));
    R->sqr = s;
    R->norm = cnorm1(q);
    ccross3(x, y, R->crw);
    for (int c = 0; c < 3; c++) R->crs[c] = cmul(a, R->crw[c]);
    R->crd = cdot3c(R->crw, w);
    R->crsq = cdot3c(R->crw, R->crw);
    ccross3(x, y, R->u);
    ccross3(x, w, R->v);
    R->xw = cdot3c(x, w);
}

/* extract one result element (re, im) from a kernel output array */
static void xval(int esz, int iscx, const unsigned char *p, cr *v)
{
    if (esz == 4) {
        float f; memcpy(&f, p, 4); v->re = f; v->im = 0;
    } else if (esz == 8 && !iscx) {
        double d; memcpy(&d, p, 8); v->re = d; v->im = 0;
    } else if (esz == 8) {
        float a, b; memcpy(&a, p, 4); memcpy(&b, p + 4, 4);
        v->re = a; v->im = b;
    } else {
        double a, b; memcpy(&a, p, 8); memcpy(&b, p + 8, 8);
        v->re = a; v->im = b;
    }
}

static ld relerr(cr g, cr r)
{
    return mx(rabs(g.re - r.re) / m1(r.re), rabs(g.im - r.im) / m1(r.im));
}

static cr field(const refset *r, int k)
{
    switch (k) {
    case 0: return r->ax;
    case 1: return r->xy;
    case 2: return r->dot;
    case 3: return r->had[0];
    case 4: return r->had[1];
    case 5: return r->had[2];
    case 6: return r->sqr;
    case 7: return r->norm;
    case 8: return r->crw[0];
    case 9: return r->crw[1];
    case 10: return r->crw[2];
    case 11: return r->crs[0];
    case 12: return r->crs[1];
    case 13: return r->crs[2];
    case 14: return r->crd;
    case 15: return r->crsq;
    case 16: return r->u[0];
    case 17: return r->u[1];
    case 18: return r->u[2];
    case 19: return r->v[0];
    case 20: return r->v[1];
    case 21: return r->v[2];
    case 22: return r->u[0];
    case 23: return r->u[1];
    case 24: return r->u[2];
    case 25: return r->xw;
    case 26: return r->dot;
    default: return r->xw;
    }
}

static const char *opnames[28] = {
    "v1axpy", "v1xypa", "v3dot",
    "v3had.x", "v3had.y", "v3had.z", "v3sqr", "v1norm",
    "v3cross.x", "v3cross.y", "v3cross.z",
    "v3crossscal.x", "v3crossscal.y", "v3crossscal.z",
    "v3crossdot", "v3crosssqr",
    "crossxy_crossxz.u.x", "crossxy_crossxz.u.y", "crossxy_crossxz.u.z",
    "crossxy_crossxz.v.x", "crossxy_crossxz.v.y", "crossxy_crossxz.v.z",
    "crossxy_dotxz.u.x", "crossxy_dotxz.u.y", "crossxy_dotxz.u.z",
    "crossxy_dotxz.r", "dotxy_dotxz.r", "dotxy_dotxz.q"
};

static void cmp_all(int esz, int iscx, const void *const g[28],
                    const refset *R, int n, ld eps)
{
    for (int k = 0; k < 28; k++) {
        ld e = 0;
        for (int i = 0; i < n; i++) {
            cr gv, rv = field(&R[i], k);
            xval(esz, iscx, (const unsigned char *)g[k] + (size_t)i * esz, &gv);
            if (relerr(gv, rv) > e) e = relerr(gv, rv);
        }
        chk(opnames[k], e, eps);
    }
}

/* data fillers: components of a 3-vector get seeds S, S+1, S+2 */
static void f1_s(v1s v, int S) { for (int i = 0; i < v.n; i++) v.x[i] = (float)VAL(S, i); }
static void f1_d(v1d v, int S) { for (int i = 0; i < v.n; i++) v.x[i] = (double)VAL(S, i); }
static void f1_c(v1c v, int S) { for (int i = 0; i < v.n; i++) { v.x[i].re = (float)VAL(S, i); v.x[i].im = (float)(VAL(S, i) * 0.7L); } }
static void f1_z(v1z v, int S) { for (int i = 0; i < v.n; i++) { v.x[i].re = VAL(S, i); v.x[i].im = VAL(S, i) * 0.7L; } }

static void f3_s(v3s v, int S) { for (int i = 0; i < v.n; i++) { v.x[i] = (float)VAL(S, i); v.y[i] = (float)VAL(S + 1, i); v.z[i] = (float)VAL(S + 2, i); } }
static void f3_d(v3d v, int S) { for (int i = 0; i < v.n; i++) { v.x[i] = (double)VAL(S, i); v.y[i] = (double)VAL(S + 1, i); v.z[i] = (double)VAL(S + 2, i); } }
static void f3_c(v3c v, int S) { for (int i = 0; i < v.n; i++) { v.x[i].re = (float)VAL(S, i); v.x[i].im = (float)(VAL(S, i) * 0.7L); v.y[i].re = (float)VAL(S + 1, i); v.y[i].im = (float)(VAL(S + 1, i) * 0.7L); v.z[i].re = (float)VAL(S + 2, i); v.z[i].im = (float)(VAL(S + 2, i) * 0.7L); } }
static void f3_z(v3z v, int S) { for (int i = 0; i < v.n; i++) { v.x[i].re = VAL(S, i); v.x[i].im = VAL(S, i) * 0.7L; v.y[i].re = VAL(S + 1, i); v.y[i].im = VAL(S + 1, i) * 0.7L; v.z[i].re = VAL(S + 2, i); v.z[i].im = VAL(S + 2, i) * 0.7L; } }

static void test_s(void)
{
    prec = "s";
    refset R[N];
    for (int i = 0; i < N; i++) refset_fill(i, 1, 0, &R[i]);
    v3s x = v3s_new(N), y = v3s_new(N), w = v3s_new(N);
    v1s q = v1s_new(N), t = v1s_new(N), xv = v1s_new(N), yv = v1s_new(N);
    f3_s(x, 1); f3_s(y, 4); f3_s(w, 7);
    f1_s(q, 10); f1_s(t, 11); f1_s(xv, 13); f1_s(yv, 12);

    v1s r_ax = v1axpy_s(xv, yv, 1.5f);
    v1s r_xy = v1xypa_s(q, t, 0.25f);
    v1s r_dot = v3dot_s(x, y);
    v3s r_had = v3had_s(x, y);
    v1s r_sqr = v3sqr_s(x);
    v1s r_norm = v1norm_s(q);
    v3s r_cr = v3cross_s(x, y);
    v3s r_crs = v3crossscal_s(x, y, 1.5f);
    v1s r_crd = v3crossdot_s(x, y, w);
    v1s r_crsq = v3crosssqr_s(x, y);
    v3uvs p33 = v3crossxy_crossxz_s(x, y, w);
    v3urs p31 = v3crossxy_dotxz_s(x, y, w);
    v1rqs p11 = v3dotxy_dotxz_s(x, y, w);

    const void *g[28] = {
        r_ax.x, r_xy.x, r_dot.x, r_had.x, r_had.y, r_had.z,
        r_sqr.x, r_norm.x, r_cr.x, r_cr.y, r_cr.z,
        r_crs.x, r_crs.y, r_crs.z, r_crd.x, r_crsq.x,
        p33.u.x, p33.u.y, p33.u.z, p33.v.x, p33.v.y, p33.v.z,
        p31.u.x, p31.u.y, p31.u.z, p31.r.x, p11.r.x, p11.q.x
    };
    cmp_all(4, 0, g, R, N, 1e-5L);

    v1s_free(&r_ax); v1s_free(&r_xy); v1s_free(&r_dot); v3s_free(&r_had);
    v1s_free(&r_sqr); v1s_free(&r_norm); v3s_free(&r_cr); v3s_free(&r_crs);
    v1s_free(&r_crd); v1s_free(&r_crsq);
    v3uvs_free(p33); v3urs_free(p31); v1rqs_free(p11);
    v3s_free(&x); v3s_free(&y); v3s_free(&w);
    v1s_free(&q); v1s_free(&t); v1s_free(&xv); v1s_free(&yv);
}

static void test_d(void)
{
    prec = "d";
    refset R[N];
    for (int i = 0; i < N; i++) refset_fill(i, 1, 0, &R[i]);
    v3d x = v3d_new(N), y = v3d_new(N), w = v3d_new(N);
    v1d q = v1d_new(N), t = v1d_new(N), xv = v1d_new(N), yv = v1d_new(N);
    f3_d(x, 1); f3_d(y, 4); f3_d(w, 7);
    f1_d(q, 10); f1_d(t, 11); f1_d(xv, 13); f1_d(yv, 12);

    v1d r_ax = v1axpy_d(xv, yv, 1.5);
    v1d r_xy = v1xypa_d(q, t, 0.25);
    v1d r_dot = v3dot_d(x, y);
    v3d r_had = v3had_d(x, y);
    v1d r_sqr = v3sqr_d(x);
    v1d r_norm = v1norm_d(q);
    v3d r_cr = v3cross_d(x, y);
    v3d r_crs = v3crossscal_d(x, y, 1.5);
    v1d r_crd = v3crossdot_d(x, y, w);
    v1d r_crsq = v3crosssqr_d(x, y);
    v3uvd p33 = v3crossxy_crossxz_d(x, y, w);
    v3urd p31 = v3crossxy_dotxz_d(x, y, w);
    v1rqd p11 = v3dotxy_dotxz_d(x, y, w);

    const void *g[28] = {
        r_ax.x, r_xy.x, r_dot.x, r_had.x, r_had.y, r_had.z,
        r_sqr.x, r_norm.x, r_cr.x, r_cr.y, r_cr.z,
        r_crs.x, r_crs.y, r_crs.z, r_crd.x, r_crsq.x,
        p33.u.x, p33.u.y, p33.u.z, p33.v.x, p33.v.y, p33.v.z,
        p31.u.x, p31.u.y, p31.u.z, p31.r.x, p11.r.x, p11.q.x
    };
    cmp_all(8, 0, g, R, N, 1e-12L);

    /* strided smoke: v3cross_d on components with inc = 2 */
    {
        double bx[2 * N], by[2 * N], bz[2 * N];
        double byx[2 * N], byy[2 * N], byz[2 * N];
        refset RS[N];
        v3d sw;
        for (int i = 0; i < N; i++) {
            bx[2 * i] = (double)VAL(21, i);
            by[2 * i] = (double)VAL(22, i);
            bz[2 * i] = (double)VAL(23, i);
            byx[2 * i] = (double)VAL(24, i);
            byy[2 * i] = (double)VAL(25, i);
            byz[2 * i] = (double)VAL(26, i);
            refset_fill(i, 21, 0, &RS[i]);
        }
        v3d sx = v3d_view(bx, by, bz, 2, 2, 2, N);
        v3d sy = v3d_view(byx, byy, byz, 2, 2, 2, N);
        sw = v3cross_d(sx, sy);
        for (int c = 0; c < 3; c++) {
            ld e = 0;
            const double *a = (const double *[]){ sw.x, sw.y, sw.z }[c];
            for (int i = 0; i < N; i++) {
                cr gv = { a[i], 0 }, rv = RS[i].crw[c];
                if (relerr(gv, rv) > e) e = relerr(gv, rv);
            }
            chk("strided v3cross_d", e, 1e-12L);
        }
        v3d_free(&sw);
    }

    v1d_free(&r_ax); v1d_free(&r_xy); v1d_free(&r_dot); v3d_free(&r_had);
    v1d_free(&r_sqr); v1d_free(&r_norm); v3d_free(&r_cr); v3d_free(&r_crs);
    v1d_free(&r_crd); v1d_free(&r_crsq);
    v3uvd_free(p33); v3urd_free(p31); v1rqd_free(p11);
    v3d_free(&x); v3d_free(&y); v3d_free(&w);
    v1d_free(&q); v1d_free(&t); v1d_free(&xv); v1d_free(&yv);
}

static void test_c(void)
{
    prec = "c";
    refset R[N];
    for (int i = 0; i < N; i++) refset_fill(i, 1, 1, &R[i]);
    v3c x = v3c_new(N), y = v3c_new(N), w = v3c_new(N);
    v1c q = v1c_new(N), t = v1c_new(N), xv = v1c_new(N), yv = v1c_new(N);
    f3_c(x, 1); f3_c(y, 4); f3_c(w, 7);
    f1_c(q, 10); f1_c(t, 11); f1_c(xv, 13); f1_c(yv, 12);

    v1c r_ax = v1axpy_c(xv, yv, pblas_cc(1.5f, 0.5f));
    v1c r_xy = v1xypa_c(q, t, pblas_cc(0.25f, 0.1f));
    v1c r_dot = v3dot_c(x, y);
    v3c r_had = v3had_c(x, y);
    v1c r_sqr = v3sqr_c(x);
    v1c r_norm = v1norm_c(q);
    v3c r_cr = v3cross_c(x, y);
    v3c r_crs = v3crossscal_c(x, y, pblas_cc(1.5f, 0.5f));
    v1c r_crd = v3crossdot_c(x, y, w);
    v1c r_crsq = v3crosssqr_c(x, y);
    v3uvc p33 = v3crossxy_crossxz_c(x, y, w);
    v3urc p31 = v3crossxy_dotxz_c(x, y, w);
    v1rqc p11 = v3dotxy_dotxz_c(x, y, w);

    const void *g[28] = {
        r_ax.x, r_xy.x, r_dot.x, r_had.x, r_had.y, r_had.z,
        r_sqr.x, r_norm.x, r_cr.x, r_cr.y, r_cr.z,
        r_crs.x, r_crs.y, r_crs.z, r_crd.x, r_crsq.x,
        p33.u.x, p33.u.y, p33.u.z, p33.v.x, p33.v.y, p33.v.z,
        p31.u.x, p31.u.y, p31.u.z, p31.r.x, p11.r.x, p11.q.x
    };
    cmp_all(8, 1, g, R, N, 1e-5L);

    /* strided smoke: v1xypa_c with q at inc = 2 */
    {
        pblas_floatcomplex bq[2 * N], bt[N];
        refset RS[N];
        v1c sr;
        for (int i = 0; i < N; i++) {
            bq[2 * i].re = (float)VAL(21, i);
            bq[2 * i].im = (float)(VAL(21, i) * 0.7L);
            bt[i].re = (float)VAL(22, i);
            bt[i].im = (float)(VAL(22, i) * 0.7L);
            refset_fill(i, 12, 1, &RS[i]); /* q = seed 21, t = seed 22 */
        }
        v1c sq = v1c_view(bq, 2, N);
        v1c st = v1c_view(bt, 1, N);
        sr = v1xypa_c(sq, st, pblas_cc(0.25f, 0.1f));
        ld e = 0;
        for (int i = 0; i < N; i++) {
            cr gv = { sr.x[i].re, sr.x[i].im };
            if (relerr(gv, RS[i].xy) > e) e = relerr(gv, RS[i].xy);
        }
        chk("strided v1xypa_c", e, 1e-5L);
        v1c_free(&sr);
    }

    v1c_free(&r_ax); v1c_free(&r_xy); v1c_free(&r_dot); v3c_free(&r_had);
    v1c_free(&r_sqr); v1c_free(&r_norm); v3c_free(&r_cr); v3c_free(&r_crs);
    v1c_free(&r_crd); v1c_free(&r_crsq);
    v3uvc_free(p33); v3urc_free(p31); v1rqc_free(p11);
    v3c_free(&x); v3c_free(&y); v3c_free(&w);
    v1c_free(&q); v1c_free(&t); v1c_free(&xv); v1c_free(&yv);
}

static void test_z(void)
{
    prec = "z";
    refset R[N];
    for (int i = 0; i < N; i++) refset_fill(i, 1, 1, &R[i]);
    v3z x = v3z_new(N), y = v3z_new(N), w = v3z_new(N);
    v1z q = v1z_new(N), t = v1z_new(N), xv = v1z_new(N), yv = v1z_new(N);
    f3_z(x, 1); f3_z(y, 4); f3_z(w, 7);
    f1_z(q, 10); f1_z(t, 11); f1_z(xv, 13); f1_z(yv, 12);

    v1z r_ax = v1axpy_z(xv, yv, pblas_zc(1.5, 0.5));
    v1z r_xy = v1xypa_z(q, t, pblas_zc(0.25, 0.1));
    v1z r_dot = v3dot_z(x, y);
    v3z r_had = v3had_z(x, y);
    v1z r_sqr = v3sqr_z(x);
    v1z r_norm = v1norm_z(q);
    v3z r_cr = v3cross_z(x, y);
    v3z r_crs = v3crossscal_z(x, y, pblas_zc(1.5, 0.5));
    v1z r_crd = v3crossdot_z(x, y, w);
    v1z r_crsq = v3crosssqr_z(x, y);
    v3uvz p33 = v3crossxy_crossxz_z(x, y, w);
    v3urz p31 = v3crossxy_dotxz_z(x, y, w);
    v1rqz p11 = v3dotxy_dotxz_z(x, y, w);

    const void *g[28] = {
        r_ax.x, r_xy.x, r_dot.x, r_had.x, r_had.y, r_had.z,
        r_sqr.x, r_norm.x, r_cr.x, r_cr.y, r_cr.z,
        r_crs.x, r_crs.y, r_crs.z, r_crd.x, r_crsq.x,
        p33.u.x, p33.u.y, p33.u.z, p33.v.x, p33.v.y, p33.v.z,
        p31.u.x, p31.u.y, p31.u.z, p31.r.x, p11.r.x, p11.q.x
    };
    cmp_all(16, 1, g, R, N, 1e-12L);

    v1z_free(&r_ax); v1z_free(&r_xy); v1z_free(&r_dot); v3z_free(&r_had);
    v1z_free(&r_sqr); v1z_free(&r_norm); v3z_free(&r_cr); v3z_free(&r_crs);
    v1z_free(&r_crd); v1z_free(&r_crsq);
    v3uvz_free(p33); v3urz_free(p31); v1rqz_free(p11);
    v3z_free(&x); v3z_free(&y); v3z_free(&w);
    v1z_free(&q); v1z_free(&t); v1z_free(&xv); v1z_free(&yv);
}

int main(void)
{
    test_s();
    test_d();
    test_c();
    test_z();
    printf("pblas_test: %s (%d failures)\n", fails ? "FAIL" : "OK", fails);
    return fails ? 1 : 0;
}
