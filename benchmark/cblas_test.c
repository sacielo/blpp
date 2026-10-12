/*
 * cblas_test.c - verification for the raw cblas_* physics extension entry
 * points: 19 element-wise ops + the lap5/lap7 stencils, at s, d, c, z,
 * all compared against a long double reference.  Correctness only: no
 * timing (that is cblas_bench's job), no allocation.
 *
 * Ends with a MATLAB/Octave verification block (ops_octave.h): paste it
 * into Octave and recompute element 1 of every output by hand.  Products
 * print with .* because the complex ops are bilinear, not sesquilinear.
 *
 * Build against an installed OpenBLAS (see CMakeLists.txt):
 *   gcc -O2 -I ../pkgs/openblas/include/openblas cblas_test.c \
 *       -L ../pkgs/openblas/lib -lopenblas -lm -o /tmp/cblas_test
 */
#include "cblas.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "test_ops.h"
#include "ops_octave.h"

/* stencil reference: pairwise +/- pairs, then center term (kernel order) */
static void chk_lap(int lap7, int esz, int iscx, const void *out,
                    const ld *ure, const ld *uim, ld eps)
{
    ld e = 0;
    for (int i = 0; i < N; i++) {
        cr g, r;
        int c = 16 + i;
        ld t = (ure[c - 3] + ure[c + 3]) + (ure[c - 7] + ure[c + 7]);
        if (lap7) t = (t + (ure[c - 11] + ure[c + 11]));
        r.re = (ld)0.25 * (t - (ld)(lap7 ? 6 : 4) * ure[c]);
        ld ti = 0;
        if (iscx) {
            ti = (uim[c - 3] + uim[c + 3]) + (uim[c - 7] + uim[c + 7]);
            if (lap7) ti = ti + (uim[c - 11] + uim[c + 11]);
            ti = (ld)0.25 * (ti - (ld)(lap7 ? 6 : 4) * uim[c]);
        }
        r.im = ti;
        xval(esz, iscx, (const unsigned char *)out + (size_t)i * esz, &g);
        ld ei = mx(rabs(g.re - r.re) / m1(r.re), rabs(g.im - r.im) / m1(r.im));
        if (ei > e) e = ei;
    }
    char nm[24];
    snprintf(nm, sizeof nm, lap7 ? "%slap7" : "%slap5", prec);
    chk(nm, e, eps);
}

/*
 * One body, four precisions.
 *   ICX: 0 real, 1 complex.  INIT sets the three complex-capable scalars
 *   (axpy a = 1.5[+0.5i], 1xypa a = 0.25[+0.1i], 3crossscal a); CAX/CXA/
 *   CCS pass them the way each prototype wants (value for real, pointer
 *   for complex).  SETV stores a long double sample into the typed lap
 *   buffer and the two long double reference copies.
 *   lap5/lap7 keep alpha, jx, jy, jz real in every precision.
 */
#define XT x0, 1, x1, 1, x2, 1
#define YT y0, 1, y1, 1, y2, 1
#define WT w0, 1, w1, 1, w2, 1

#define RUNTEST(P, PCH, T, ICX, ESZ, EPS, INIT, CAX, CXA, CCS, SETV)         \
static void test_ ## P(void)                                                 \
{                                                                            \
    prec = #P;                                                               \
    refset R[N];                                                             \
    for (int i = 0; i < N; i++) refset_fill(i, 1, ICX, &R[i]);               \
    T x0[N], x1[N], x2[N], y0[N], y1[N], y2[N], w0[N], w1[N], w2[N];         \
    T q[N], t[N], xv[N], yv[N];                                              \
    f3_ ## P(x0, x1, x2, 1); f3_ ## P(y0, y1, y2, 4);                        \
    f3_ ## P(w0, w1, w2, 7);                                                 \
    f1_ ## P(q, 10); f1_ ## P(t, 11); f1_ ## P(xv, 13); f1_ ## P(yv, 12);    \
    T s_ax, s_xa, s_cs;                                                      \
    INIT;                                                                    \
    void *pax = &s_ax, *pxa = &s_xa, *pcs = &s_cs;                           \
    (void)pax; (void)pxa; (void)pcs;                                         \
    T ax[N], xy[N], dot[N], sqr[N], norm[N];                                 \
    T had0[N], had1[N], had2[N];                                             \
    T cr0[N], cr1[N], cr2[N], cs0[N], cs1[N], cs2[N], crd[N], crsq[N];       \
    T uu0[N], uu1[N], uu2[N], vv0[N], vv1[N], vv2[N];                        \
    T pu0[N], pu1[N], pu2[N], pur[N], dxy_r[N], dxy_q[N];                    \
    T cc0[N], cc1[N], cc2[N], nur[N], nn0[N], nn1[N], nn2[N];                \
    T rf0[N], rf1[N], rf2[N], ex0[N], ex1[N], ex2[N];                        \
    T dr0[N], dr1[N], dr2[N], mm0[N], mm1[N], mm2[N], ke[N];                 \
    T l5[N], l7[N];                                                          \
                                                                             \
    memcpy(ax, yv, sizeof ax);                                               \
    cblas_ ## P ## axpy(N, CAX, xv, 1, ax, 1);                               \
    cblas_ ## P ## 1xypa(N, CXA, q, 1, t, 1, xy, 1);                         \
    cblas_ ## P ## 3dot(N, XT, YT, dot, 1);                                  \
    cblas_ ## P ## 3had(N, XT, YT, had0, 1, had1, 1, had2, 1);               \
    cblas_ ## P ## 3sqr(N, XT, sqr, 1);                                      \
    cblas_ ## P ## 1norm(N, q, 1, norm, 1);                                  \
    cblas_ ## P ## 3cross(N, XT, YT, cr0, 1, cr1, 1, cr2, 1);                \
    cblas_ ## P ## 3crossscal(N, CCS, XT, YT, cs0, 1, cs1, 1, cs2, 1);       \
    cblas_ ## P ## 3crossdot(N, XT, YT, WT, crd, 1);                         \
    cblas_ ## P ## 3crosssqr(N, XT, YT, crsq, 1);                            \
    cblas_ ## P ## 3crossxy_crossxz(N, XT, YT, WT,                           \
        uu0, 1, uu1, 1, uu2, 1, vv0, 1, vv1, 1, vv2, 1);                     \
    cblas_ ## P ## 3crossxy_dotxz(N, XT, YT, WT,                             \
        pu0, 1, pu1, 1, pu2, 1, pur, 1);                                     \
    cblas_ ## P ## 3dotxy_dotxz(N, XT, YT, WT, dxy_r, 1, dxy_q, 1);          \
    cblas_ ## P ## 3crosscross(N, XT, YT, WT, cc0, 1, cc1, 1, cc2, 1);       \
    cblas_ ## P ## 3norm_unit(N, XT, 0.25, nur, 1, nn0, 1, nn1, 1, nn2, 1);  \
    cblas_ ## P ## 3refl(N, XT, YT, 1.5, 0.25, rf0, 1, rf1, 1, rf2, 1);      \
    cblas_ ## P ## 3exb(N, XT, YT, 1.5, 0.25, ex0, 1, ex1, 1, ex2, 1);       \
    cblas_ ## P ## 3drag(N, XT, 1.5, 0.25, dr0, 1, dr1, 1, dr2, 1);          \
    cblas_ ## P ## 3mom_ke(N, q, 1, XT, mm0, 1, mm1, 1, mm2, 1, ke, 1);      \
                                                                             \
    T ubuf[N + 32];                                                          \
    ld Ure[N + 32], Uim[N + 32];                                             \
    for (int i = 0; i < N + 32; i++) {                                       \
        ld v = VAL(16, i - 16);                                              \
        SETV;                                                                \
    }                                                                        \
    cblas_ ## P ## lap5(N, 0.25, ubuf + 16, 1, 3, 7, l5, 1);                 \
    cblas_ ## P ## lap7(N, 0.25, ubuf + 16, 1, 3, 7, 11, l7, 1);             \
                                                                             \
    const void *g[50] = {                                                    \
        ax, xy, dot, had0, had1, had2, sqr, norm,                            \
        cr0, cr1, cr2, cs0, cs1, cs2, crd, crsq,                             \
        uu0, uu1, uu2, vv0, vv1, vv2,                                        \
        pu0, pu1, pu2, pur, dxy_r, dxy_q,                                    \
        cc0, cc1, cc2, nur, nn0, nn1, nn2,                                   \
        rf0, rf1, rf2, ex0, ex1, ex2,                                        \
        dr0, dr1, dr2, mm0, mm1, mm2, ke, l5, l7                             \
    };                                                                       \
    cmp_all(ESZ, ICX, g, R, N, EPS);              \
    chk_lap(0, ESZ, ICX, l5, Ure, Uim, EPS);                                 \
    chk_lap(1, ESZ, ICX, l7, Ure, Uim, EPS);                                 \
    {                                                                        \
        const void *in[13] = { x0, x1, x2, y0, y1, y2, w0, w1, w2,           \
                               q, t, xv, yv };                               \
        octave_dump(PCH, ESZ, ICX, 50, g, in, ubuf);                         \
    }                                                                        \
}

static v3blas_floatcomplex mkcc(double r, double i)
{ v3blas_floatcomplex c; c.re = (float)r; c.im = (float)i; return c; }
static v3blas_doublecomplex mkzz(double r, double i)
{ v3blas_doublecomplex c; c.re = r; c.im = i; return c; }

#define SET_S ubuf[i] = (float)v; Ure[i] = v; Uim[i] = 0
#define SET_D ubuf[i] = (double)v; Ure[i] = v; Uim[i] = 0
#define SET_C ubuf[i].re = (float)v; ubuf[i].im = (float)(v * 0.7L);         \
              Ure[i] = v; Uim[i] = v * 0.7L
#define SET_Z ubuf[i].re = v; ubuf[i].im = v * 0.7L;                         \
              Ure[i] = v; Uim[i] = v * 0.7L

RUNTEST(s, 's', float, 0, 4, 1e-5L,
        s_ax = 1.5f; s_xa = 0.25f; s_cs = 1.5f,
        s_ax, s_xa, s_cs, SET_S)
RUNTEST(d, 'd', double, 0, 8, 1e-12L,
        s_ax = 1.5; s_xa = 0.25; s_cs = 1.5,
        s_ax, s_xa, s_cs, SET_D)
RUNTEST(c, 'c', v3blas_floatcomplex, 1, 8, 1e-5L,
        s_ax = mkcc(1.5, 0.5); s_xa = mkcc(0.25, 0.1); s_cs = mkcc(1.5, 0.5),
        pax, pxa, pcs, SET_C)
RUNTEST(z, 'z', v3blas_doublecomplex, 1, 16, 1e-12L,
        s_ax = mkzz(1.5, 0.5); s_xa = mkzz(0.25, 0.1); s_cs = mkzz(1.5, 0.5),
        pax, pxa, pcs, SET_Z)

#undef RUNTEST
#undef XT
#undef YT
#undef WT

static int want(const char *sel, const char *tag)
{
    return !*sel || !strcmp(sel, tag);
}

int main(int argc, char **argv)
{
    const char *sel = argc > 1 ? argv[1] : "";
    if (*sel && !strchr("sdcz", sel[0])) {
        fprintf(stderr, "usage: %s [s|d|c|z]   (default: all four)\n", argv[0]);
        return 2;
    }
    if (want(sel, "s")) test_s();
    if (want(sel, "d")) test_d();
    if (want(sel, "c")) test_c();
    if (want(sel, "z")) test_z();
    printf("cblas_test: %s (%d failures)\n", fails ? "FAIL" : "OK", fails);
    return fails ? 1 : 0;
}
