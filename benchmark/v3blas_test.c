/*
 * v3blas_test.c - verification for the v3blas.h higher-level interface.
 * Runs all 19 physics extension operations at s, d, c, z precision and
 * compares against a long double reference, plus a strided v3 smoke
 * case.  All arrays are the caller's own (the library never allocates).
 * Plain C99 (proves the header compiles as C).
 *
 * Build against an installed OpenBLAS (see CMakeLists.txt):
 *   gcc -O2 -I ../pkgs/openblas/include/openblas v3blas_test.c \
 *       -L ../pkgs/openblas/lib -lopenblas -lm -o /tmp/v3blas_test
 */
#include "v3blas.h"
#include <stdio.h>
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
static cr csub(cr a, cr b) { cr c = { a.re - b.re, a.im - b.im }; return c; }
static cr cdiv(cr a, cr b)
{
    ld d = b.re * b.re + b.im * b.im;
    cr c;
    c.re = (a.re * b.re + a.im * b.im) / d;
    c.im = (a.im * b.re - a.re * b.im) / d;
    return c;
}

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

static cr csqrt_cr(cr z)
{
    ld m = sqrtl(z.re * z.re + z.im * z.im);
    cr r;
    r.re = sqrtl((m + z.re) / 2.0L);
    r.im = (z.im >= 0) ? sqrtl((m - z.re) / 2.0L)
                       : -sqrtl((m - z.re) / 2.0L);
    return r;
}

/* reference values for element i; seeds: x=(b,b+1,b+2), y=(b+3..b+5),
 * w=(b+6..b+8), q=b+9, t=b+10, yv=b+11, xv=b+12 */
typedef struct {
    cr ax, xy, dot, sqr, norm, crd, crsq, xw;
    cr had[3], crw[3], crs[3], u[3], v[3];
    cr cc[3], nur, nu[3], rf[3], ex[3], dr[3], mm[3], kk;
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
    /* new CFD ops: real a = 1.5, eps = 0.25 in every precision */
    {
        cr tmp[3], epsc = { 0.25L, 0.0L }, asc = { 1.5L, 0.0L };
        cr half = { 0.5L, 0.0L }, sxx, dyy, ff;
        ccross3(x, y, tmp);
        ccross3(tmp, w, R->cc);
        sxx = cdot3c(x, x);
        R->nur = csqrt_cr(cadd(sxx, epsc));           /* sqrt(x.x + eps) */
        for (int c = 0; c < 3; c++) R->nu[c] = cdiv(x[c], R->nur);
        dyy = cdot3c(y, y);
        ff = cdiv(cmul(asc, cdot3c(x, y)), cadd(dyy, epsc));
        for (int c = 0; c < 3; c++) R->rf[c] = csub(x[c], cmul(ff, y[c]));
        ff = cdiv(asc, cadd(dyy, epsc));
        for (int c = 0; c < 3; c++) R->ex[c] = cmul(ff, R->crw[c]);
        for (int c = 0; c < 3; c++) R->dr[c] = cmul(cmul(asc, x[c]), R->nur);
        for (int c = 0; c < 3; c++) R->mm[c] = cmul(q, x[c]);
        R->kk = cmul(cmul(half, q), sxx);
    }
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
    case 28: return r->cc[0];
    case 29: return r->cc[1];
    case 30: return r->cc[2];
    case 31: return r->nur;
    case 32: return r->nu[0];
    case 33: return r->nu[1];
    case 34: return r->nu[2];
    case 35: return r->rf[0];
    case 36: return r->rf[1];
    case 37: return r->rf[2];
    case 38: return r->ex[0];
    case 39: return r->ex[1];
    case 40: return r->ex[2];
    case 41: return r->dr[0];
    case 42: return r->dr[1];
    case 43: return r->dr[2];
    case 44: return r->mm[0];
    case 45: return r->mm[1];
    case 46: return r->mm[2];
    case 47: return r->kk;
    default: return r->xw;
    }
}

static const char *opnames[48] = {
    "v1axpy", "v1xypa", "v3dot",
    "v3had.x", "v3had.y", "v3had.z", "v3sqr", "v1norm",
    "v3cross.x", "v3cross.y", "v3cross.z",
    "v3crossscal.x", "v3crossscal.y", "v3crossscal.z",
    "v3crossdot", "v3crosssqr",
    "crossxy_crossxz.u.x", "crossxy_crossxz.u.y", "crossxy_crossxz.u.z",
    "crossxy_crossxz.v.x", "crossxy_crossxz.v.y", "crossxy_crossxz.v.z",
    "crossxy_dotxz.u.x", "crossxy_dotxz.u.y", "crossxy_dotxz.u.z",
    "crossxy_dotxz.r", "dotxy_dotxz.r", "dotxy_dotxz.q",
    "crosscross.x", "crosscross.y", "crosscross.z",
    "norm_unit.r", "norm_unit.u.x", "norm_unit.u.y", "norm_unit.u.z",
    "refl.x", "refl.y", "refl.z",
    "exb.x", "exb.y", "exb.z",
    "drag.x", "drag.y", "drag.z",
    "mom_ke.x", "mom_ke.y", "mom_ke.z", "mom_ke.k"
};

static void cmp_all(int esz, int iscx, const void *const g[48],
                    const refset *R, int n, ld eps)
{
    for (int k = 0; k < 48; k++) {
        ld e = 0;
        for (int i = 0; i < n; i++) {
            cr gv, rv = field(&R[i], k);
            xval(esz, iscx, (const unsigned char *)g[k] + (size_t)i * esz, &gv);
            if (relerr(gv, rv) > e) e = relerr(gv, rv);
        }
        chk(opnames[k], e, eps);
    }
}

/* data fills: components of a 3-vector get seeds S, S+1, S+2 */
static void f1_s(float *v, int S) { for (int i = 0; i < N; i++) v[i] = (float)VAL(S, i); }
static void f1_d(double *v, int S) { for (int i = 0; i < N; i++) v[i] = (double)VAL(S, i); }
static void f1_c(v3blas_floatcomplex *v, int S) { for (int i = 0; i < N; i++) { v[i].re = (float)VAL(S, i); v[i].im = (float)(VAL(S, i) * 0.7L); } }
static void f1_z(v3blas_doublecomplex *v, int S) { for (int i = 0; i < N; i++) { v[i].re = VAL(S, i); v[i].im = VAL(S, i) * 0.7L; } }

static void f3_s(float *a0, float *a1, float *a2, int S) { for (int i = 0; i < N; i++) { a0[i] = (float)VAL(S, i); a1[i] = (float)VAL(S + 1, i); a2[i] = (float)VAL(S + 2, i); } }
static void f3_d(double *a0, double *a1, double *a2, int S) { for (int i = 0; i < N; i++) { a0[i] = (double)VAL(S, i); a1[i] = (double)VAL(S + 1, i); a2[i] = (double)VAL(S + 2, i); } }
static void f3_c(v3blas_floatcomplex *a0, v3blas_floatcomplex *a1, v3blas_floatcomplex *a2, int S) { for (int i = 0; i < N; i++) { a0[i].re = (float)VAL(S, i); a0[i].im = (float)(VAL(S, i) * 0.7L); a1[i].re = (float)VAL(S + 1, i); a1[i].im = (float)(VAL(S + 1, i) * 0.7L); a2[i].re = (float)VAL(S + 2, i); a2[i].im = (float)(VAL(S + 2, i) * 0.7L); } }
static void f3_z(v3blas_doublecomplex *a0, v3blas_doublecomplex *a1, v3blas_doublecomplex *a2, int S) { for (int i = 0; i < N; i++) { a0[i].re = VAL(S, i); a0[i].im = VAL(S, i) * 0.7L; a1[i].re = VAL(S + 1, i); a1[i].im = VAL(S + 1, i) * 0.7L; a2[i].re = VAL(S + 2, i); a2[i].im = VAL(S + 2, i) * 0.7L; } }


static void test_s(void)
{
    prec = "s";
    refset R[N];
    for (int i = 0; i < N; i++) refset_fill(i, 1, 0, &R[i]);
    float x0[N], x1[N], x2[N];
    float y0[N], y1[N], y2[N];
    float w0[N], w1[N], w2[N];
    float q[N], t[N], xv[N], yv[N];
    f3_s(x0, x1, x2, 1); f3_s(y0, y1, y2, 4); f3_s(w0, w1, w2, 7);
    f1_s(q, 10); f1_s(t, 11); f1_s(xv, 13); f1_s(yv, 12);
    v3s x = v3s_wrap(x0, x1, x2, N), y = v3s_wrap(y0, y1, y2, N);
    v3s w = v3s_wrap(w0, w1, w2, N);

    float ax[N], xy[N], dot[N], sqr[N], norm[N];
    float had0[N], had1[N], had2[N];
    float cr0[N], cr1[N], cr2[N];
    float cs0[N], cs1[N], cs2[N];
    float crd[N], crsq[N];
    float uu0[N], uu1[N], uu2[N];
    float vv0[N], vv1[N], vv2[N];
    float pu0[N], pu1[N], pu2[N], pur[N];
    float dxy_r[N], dxy_q[N];

    v1axpy_s(1.5f, xv, yv, ax, N);
    v1xypa_s(q, t, 0.25f, xy, N);
    v3dot_s(x, y, dot);
    v3had_s(x, y, had0, had1, had2);
    v3sqr_s(x, sqr);
    v1norm_s(q, norm, N);
    v3cross_s(x, y, cr0, cr1, cr2);
    v3crossscal_s(x, y, 1.5f, cs0, cs1, cs2);
    v3crossdot_s(x, y, w, crd);
    v3crosssqr_s(x, y, crsq);
    v3crossxy_crossxz_s(x, y, w, uu0, uu1, uu2, vv0, vv1, vv2);
    v3crossxy_dotxz_s(x, y, w, pu0, pu1, pu2, pur);
    v3dotxy_dotxz_s(x, y, w, dxy_r, dxy_q);

    float cc0[N], cc1[N], cc2[N], nur[N], nn0[N], nn1[N], nn2[N];
    float rf0[N], rf1[N], rf2[N], ex0[N], ex1[N], ex2[N];
    float dr0[N], dr1[N], dr2[N], mm0[N], mm1[N], mm2[N], ke[N];
    v3crosscross_s(x, y, w, cc0, cc1, cc2);
    v3norm_unit_s(x, 0.25f, nur, nn0, nn1, nn2);
    v3refl_s(x, y, 1.5f, 0.25f, rf0, rf1, rf2);
    v3exb_s(x, y, 1.5f, 0.25f, ex0, ex1, ex2);
    v3drag_s(x, 1.5f, 0.25f, dr0, dr1, dr2);
    v3mom_ke_s(q, x, mm0, mm1, mm2, ke);

    const void *g[48] = {
        ax, xy, dot, had0, had1, had2, sqr, norm,
        cr0, cr1, cr2, cs0, cs1, cs2, crd, crsq,
        uu0, uu1, uu2, vv0, vv1, vv2,
        pu0, pu1, pu2, pur, dxy_r, dxy_q,
        cc0, cc1, cc2, nur, nn0, nn1, nn2,
        rf0, rf1, rf2, ex0, ex1, ex2,
        dr0, dr1, dr2, mm0, mm1, mm2, ke
    };
    cmp_all(4, 0, g, R, N, 1e-5L);
}

static void test_d(void)
{
    prec = "d";
    refset R[N];
    for (int i = 0; i < N; i++) refset_fill(i, 1, 0, &R[i]);
    double x0[N], x1[N], x2[N];
    double y0[N], y1[N], y2[N];
    double w0[N], w1[N], w2[N];
    double q[N], t[N], xv[N], yv[N];
    f3_d(x0, x1, x2, 1); f3_d(y0, y1, y2, 4); f3_d(w0, w1, w2, 7);
    f1_d(q, 10); f1_d(t, 11); f1_d(xv, 13); f1_d(yv, 12);
    v3d x = v3d_wrap(x0, x1, x2, N), y = v3d_wrap(y0, y1, y2, N);
    v3d w = v3d_wrap(w0, w1, w2, N);

    double ax[N], xy[N], dot[N], sqr[N], norm[N];
    double had0[N], had1[N], had2[N];
    double cr0[N], cr1[N], cr2[N];
    double cs0[N], cs1[N], cs2[N];
    double crd[N], crsq[N];
    double uu0[N], uu1[N], uu2[N];
    double vv0[N], vv1[N], vv2[N];
    double pu0[N], pu1[N], pu2[N], pur[N];
    double dxy_r[N], dxy_q[N];

    v1axpy_d(1.5, xv, yv, ax, N);
    v1xypa_d(q, t, 0.25, xy, N);
    v3dot_d(x, y, dot);
    v3had_d(x, y, had0, had1, had2);
    v3sqr_d(x, sqr);
    v1norm_d(q, norm, N);
    v3cross_d(x, y, cr0, cr1, cr2);
    v3crossscal_d(x, y, 1.5, cs0, cs1, cs2);
    v3crossdot_d(x, y, w, crd);
    v3crosssqr_d(x, y, crsq);
    v3crossxy_crossxz_d(x, y, w, uu0, uu1, uu2, vv0, vv1, vv2);
    v3crossxy_dotxz_d(x, y, w, pu0, pu1, pu2, pur);
    v3dotxy_dotxz_d(x, y, w, dxy_r, dxy_q);

    double cc0[N], cc1[N], cc2[N], nur[N], nn0[N], nn1[N], nn2[N];
    double rf0[N], rf1[N], rf2[N], ex0[N], ex1[N], ex2[N];
    double dr0[N], dr1[N], dr2[N], mm0[N], mm1[N], mm2[N], ke[N];
    v3crosscross_d(x, y, w, cc0, cc1, cc2);
    v3norm_unit_d(x, 0.25, nur, nn0, nn1, nn2);
    v3refl_d(x, y, 1.5, 0.25, rf0, rf1, rf2);
    v3exb_d(x, y, 1.5, 0.25, ex0, ex1, ex2);
    v3drag_d(x, 1.5, 0.25, dr0, dr1, dr2);
    v3mom_ke_d(q, x, mm0, mm1, mm2, ke);

    const void *g[48] = {
        ax, xy, dot, had0, had1, had2, sqr, norm,
        cr0, cr1, cr2, cs0, cs1, cs2, crd, crsq,
        uu0, uu1, uu2, vv0, vv1, vv2,
        pu0, pu1, pu2, pur, dxy_r, dxy_q,
        cc0, cc1, cc2, nur, nn0, nn1, nn2,
        rf0, rf1, rf2, ex0, ex1, ex2,
        dr0, dr1, dr2, mm0, mm1, mm2, ke
    };
    cmp_all(8, 0, g, R, N, 1e-12L);

    /* strided smoke: v3cross_d on components with inc = 2 */
    {
        double bx[2 * N], by[2 * N], bz[2 * N];
        double byx[2 * N], byy[2 * N], byz[2 * N];
        double sw0[N], sw1[N], sw2[N];
        refset RS[N];
        for (int i = 0; i < N; i++) {
            bx[2 * i] = (double)VAL(21, i);
            by[2 * i] = (double)VAL(22, i);
            bz[2 * i] = (double)VAL(23, i);
            byx[2 * i] = (double)VAL(24, i);
            byy[2 * i] = (double)VAL(25, i);
            byz[2 * i] = (double)VAL(26, i);
            refset_fill(i, 21, 0, &RS[i]);
        }
        v3d sx = v3d_wrap(bx, by, bz, N);
        v3d sy = v3d_wrap(byx, byy, byz, N);
        sx.incx = 2; sx.incy = 2; sx.incz = 2;
        sy.incx = 2; sy.incy = 2; sy.incz = 2;
        v3cross(sx, sy, sw0, sw1, sw2);  /* C11 _Generic name */
        for (int c = 0; c < 3; c++) {
            ld e = 0;
            const double *a = (const double *[]){ sw0, sw1, sw2 }[c];
            for (int i = 0; i < N; i++) {
                cr gv = { a[i], 0 }, rv = RS[i].crw[c];
                if (relerr(gv, rv) > e) e = relerr(gv, rv);
            }
            chk("strided v3cross_d", e, 1e-12L);
        }
    }
}

static void test_c(void)
{
    prec = "c";
    refset R[N];
    for (int i = 0; i < N; i++) refset_fill(i, 1, 1, &R[i]);
    typedef v3blas_floatcomplex Z;
    Z x0[N], x1[N], x2[N];
    Z y0[N], y1[N], y2[N];
    Z w0[N], w1[N], w2[N];
    Z q[N], t[N], xv[N], yv[N];
    f3_c(x0, x1, x2, 1); f3_c(y0, y1, y2, 4); f3_c(w0, w1, w2, 7);
    f1_c(q, 10); f1_c(t, 11); f1_c(xv, 13); f1_c(yv, 12);
    v3c x = v3c_wrap(x0, x1, x2, N), y = v3c_wrap(y0, y1, y2, N);
    v3c w = v3c_wrap(w0, w1, w2, N);

    Z ax[N], xy[N], dot[N], sqr[N], norm[N];
    Z had0[N], had1[N], had2[N];
    Z cr0[N], cr1[N], cr2[N];
    Z cs0[N], cs1[N], cs2[N];
    Z crd[N], crsq[N];
    Z uu0[N], uu1[N], uu2[N];
    Z vv0[N], vv1[N], vv2[N];
    Z pu0[N], pu1[N], pu2[N], pur[N];
    Z dxy_r[N], dxy_q[N];

    v1axpy_c(v3blas_cc(1.5f, 0.5f), xv, yv, ax, N);
    v1xypa_c(q, t, v3blas_cc(0.25f, 0.1f), xy, N);
    v3dot_c(x, y, dot);
    v3had_c(x, y, had0, had1, had2);
    v3sqr_c(x, sqr);
    v1norm_c(q, norm, N);
    v3cross_c(x, y, cr0, cr1, cr2);
    v3crossscal_c(x, y, v3blas_cc(1.5f, 0.5f), cs0, cs1, cs2);
    v3crossdot_c(x, y, w, crd);
    v3crosssqr_c(x, y, crsq);
    v3crossxy_crossxz_c(x, y, w, uu0, uu1, uu2, vv0, vv1, vv2);
    v3crossxy_dotxz_c(x, y, w, pu0, pu1, pu2, pur);
    v3dotxy_dotxz_c(x, y, w, dxy_r, dxy_q);

    Z cc0[N], cc1[N], cc2[N], nur[N], nn0[N], nn1[N], nn2[N];
    Z rf0[N], rf1[N], rf2[N], ex0[N], ex1[N], ex2[N];
    Z dr0[N], dr1[N], dr2[N], mm0[N], mm1[N], mm2[N], ke[N];
    v3crosscross_c(x, y, w, cc0, cc1, cc2);
    v3norm_unit_c(x, 0.25f, nur, nn0, nn1, nn2);
    v3refl_c(x, y, 1.5f, 0.25f, rf0, rf1, rf2);
    v3exb_c(x, y, 1.5f, 0.25f, ex0, ex1, ex2);
    v3drag_c(x, 1.5f, 0.25f, dr0, dr1, dr2);
    v3mom_ke_c(q, x, mm0, mm1, mm2, ke);

    const void *g[48] = {
        ax, xy, dot, had0, had1, had2, sqr, norm,
        cr0, cr1, cr2, cs0, cs1, cs2, crd, crsq,
        uu0, uu1, uu2, vv0, vv1, vv2,
        pu0, pu1, pu2, pur, dxy_r, dxy_q,
        cc0, cc1, cc2, nur, nn0, nn1, nn2,
        rf0, rf1, rf2, ex0, ex1, ex2,
        dr0, dr1, dr2, mm0, mm1, mm2, ke
    };
    cmp_all(8, 1, g, R, N, 1e-5L);
}

static void test_z(void)
{
    prec = "z";
    refset R[N];
    for (int i = 0; i < N; i++) refset_fill(i, 1, 1, &R[i]);
    typedef v3blas_doublecomplex Z;
    Z x0[N], x1[N], x2[N];
    Z y0[N], y1[N], y2[N];
    Z w0[N], w1[N], w2[N];
    Z q[N], t[N], xv[N], yv[N];
    f3_z(x0, x1, x2, 1); f3_z(y0, y1, y2, 4); f3_z(w0, w1, w2, 7);
    f1_z(q, 10); f1_z(t, 11); f1_z(xv, 13); f1_z(yv, 12);
    v3z x = v3z_wrap(x0, x1, x2, N), y = v3z_wrap(y0, y1, y2, N);
    v3z w = v3z_wrap(w0, w1, w2, N);

    Z ax[N], xy[N], dot[N], sqr[N], norm[N];
    Z had0[N], had1[N], had2[N];
    Z cr0[N], cr1[N], cr2[N];
    Z cs0[N], cs1[N], cs2[N];
    Z crd[N], crsq[N];
    Z uu0[N], uu1[N], uu2[N];
    Z vv0[N], vv1[N], vv2[N];
    Z pu0[N], pu1[N], pu2[N], pur[N];
    Z dxy_r[N], dxy_q[N];

    v1axpy_z(v3blas_zc(1.5, 0.5), xv, yv, ax, N);
    v1xypa_z(q, t, v3blas_zc(0.25, 0.1), xy, N);
    v3dot_z(x, y, dot);
    v3had_z(x, y, had0, had1, had2);
    v3sqr_z(x, sqr);
    v1norm_z(q, norm, N);
    v3cross_z(x, y, cr0, cr1, cr2);
    v3crossscal_z(x, y, v3blas_zc(1.5, 0.5), cs0, cs1, cs2);
    v3crossdot_z(x, y, w, crd);
    v3crosssqr_z(x, y, crsq);
    v3crossxy_crossxz_z(x, y, w, uu0, uu1, uu2, vv0, vv1, vv2);
    v3crossxy_dotxz_z(x, y, w, pu0, pu1, pu2, pur);
    v3dotxy_dotxz_z(x, y, w, dxy_r, dxy_q);

    Z cc0[N], cc1[N], cc2[N], nur[N], nn0[N], nn1[N], nn2[N];
    Z rf0[N], rf1[N], rf2[N], ex0[N], ex1[N], ex2[N];
    Z dr0[N], dr1[N], dr2[N], mm0[N], mm1[N], mm2[N], ke[N];
    v3crosscross_z(x, y, w, cc0, cc1, cc2);
    v3norm_unit_z(x, 0.25, nur, nn0, nn1, nn2);
    v3refl_z(x, y, 1.5, 0.25, rf0, rf1, rf2);
    v3exb_z(x, y, 1.5, 0.25, ex0, ex1, ex2);
    v3drag_z(x, 1.5, 0.25, dr0, dr1, dr2);
    v3mom_ke_z(q, x, mm0, mm1, mm2, ke);

    const void *g[48] = {
        ax, xy, dot, had0, had1, had2, sqr, norm,
        cr0, cr1, cr2, cs0, cs1, cs2, crd, crsq,
        uu0, uu1, uu2, vv0, vv1, vv2,
        pu0, pu1, pu2, pur, dxy_r, dxy_q,
        cc0, cc1, cc2, nur, nn0, nn1, nn2,
        rf0, rf1, rf2, ex0, ex1, ex2,
        dr0, dr1, dr2, mm0, mm1, mm2, ke
    };
    cmp_all(16, 1, g, R, N, 1e-12L);
}

int main(void)
{
    test_s();
    test_d();
    test_c();
    test_z();
    printf("v3blas_test: %s (%d failures)\n", fails ? "FAIL" : "OK", fails);
    return fails ? 1 : 0;
}
