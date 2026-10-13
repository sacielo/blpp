/*
 * v3blas_test.c - verification for the v3blas.h higher-level interface.
 * Runs all 18 physics extension operations at s, d, c, z precision and
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

#include "test_ops.h"
#include "ops_octave.h"
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

    memcpy(ax, yv, sizeof ax); cblas_saxpy(N, 1.5f, xv, 1, ax, 1);  /* baseline */
    v1xypa_s(q, t, 0.25f, xy, N);
    v3dot_s(x, y, dot);
    v3had_s(x, y, v3s_wrap(had0, had1, had2, N));
    v3sqr_s(x, sqr);
    v1norm_s(q, norm, N);
    v3cross_s(x, y, v3s_wrap(cr0, cr1, cr2, N));
    v3crossscal_s(x, y, 1.5f, v3s_wrap(cs0, cs1, cs2, N));
    v3crossdot_s(x, y, w, crd);
    v3crosssqr_s(x, y, crsq);
    v3crossxy_crossxz_s(x, y, w, v3s_wrap(uu0, uu1, uu2, N),
                                 v3s_wrap(vv0, vv1, vv2, N));
    v3crossxy_dotxz_s(x, y, w, v3s_wrap(pu0, pu1, pu2, N), pur);
    v3dotxy_dotxz_s(x, y, w, dxy_r, dxy_q);

    float cc0[N], cc1[N], cc2[N], nur[N], nn0[N], nn1[N], nn2[N];
    float rf0[N], rf1[N], rf2[N], ex0[N], ex1[N], ex2[N];
    float dr0[N], dr1[N], dr2[N], mm0[N], mm1[N], mm2[N], ke[N];
    v3crosscross_s(x, y, w, v3s_wrap(cc0, cc1, cc2, N));
    v3norm_unit_s(x, 0.25f, nur, v3s_wrap(nn0, nn1, nn2, N));
    v3refl_s(x, y, 1.5f, 0.25f, v3s_wrap(rf0, rf1, rf2, N));
    v3exb_s(x, y, 1.5f, 0.25f, v3s_wrap(ex0, ex1, ex2, N));
    v3drag_s(x, 1.5f, 0.25f, v3s_wrap(dr0, dr1, dr2, N));
    v3mom_ke_s(q, x, v3s_wrap(mm0, mm1, mm2, N), ke);

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
    {
        const void *in[13] = { x0, x1, x2, y0, y1, y2, w0, w1, w2,
                                 q, t, xv, yv };
        octave_dump('s', 4, 0, 48, g, in, NULL);
    }
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

    memcpy(ax, yv, sizeof ax); cblas_daxpy(N, 1.5, xv, 1, ax, 1);   /* baseline */
    v1xypa_d(q, t, 0.25, xy, N);
    v3dot_d(x, y, dot);
    v3had_d(x, y, v3d_wrap(had0, had1, had2, N));
    v3sqr_d(x, sqr);
    v1norm_d(q, norm, N);
    v3cross_d(x, y, v3d_wrap(cr0, cr1, cr2, N));
    v3crossscal_d(x, y, 1.5, v3d_wrap(cs0, cs1, cs2, N));
    v3crossdot_d(x, y, w, crd);
    v3crosssqr_d(x, y, crsq);
    v3crossxy_crossxz_d(x, y, w, v3d_wrap(uu0, uu1, uu2, N),
                                 v3d_wrap(vv0, vv1, vv2, N));
    v3crossxy_dotxz_d(x, y, w, v3d_wrap(pu0, pu1, pu2, N), pur);
    v3dotxy_dotxz_d(x, y, w, dxy_r, dxy_q);

    double cc0[N], cc1[N], cc2[N], nur[N], nn0[N], nn1[N], nn2[N];
    double rf0[N], rf1[N], rf2[N], ex0[N], ex1[N], ex2[N];
    double dr0[N], dr1[N], dr2[N], mm0[N], mm1[N], mm2[N], ke[N];
    v3crosscross_d(x, y, w, v3d_wrap(cc0, cc1, cc2, N));
    v3norm_unit_d(x, 0.25, nur, v3d_wrap(nn0, nn1, nn2, N));
    v3refl_d(x, y, 1.5, 0.25, v3d_wrap(rf0, rf1, rf2, N));
    v3exb_d(x, y, 1.5, 0.25, v3d_wrap(ex0, ex1, ex2, N));
    v3drag_d(x, 1.5f, 0.25f, v3d_wrap(dr0, dr1, dr2, N));
    v3mom_ke_d(q, x, v3d_wrap(mm0, mm1, mm2, N), ke);

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
    {
        const void *in[13] = { x0, x1, x2, y0, y1, y2, w0, w1, w2,
                                 q, t, xv, yv };
        octave_dump('d', 8, 0, 48, g, in, NULL);
    }

    /* strided smoke: v3cross_d on components with inc = 2 */
    {
        double bx[2 * N], by[2 * N], bz[2 * N];
        double byx[2 * N], byy[2 * N], byz[2 * N];
        double sw0[2 * N], sw1[2 * N], sw2[2 * N];
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
        v3d so = v3d_wrap(sw0, sw1, sw2, N);   /* strided OUTPUTS now too */
        so.incx = 2; so.incy = 2; so.incz = 2;
        v3cross(sx, sy, so);  /* C11 _Generic name */
        for (int c = 0; c < 3; c++) {
            ld e = 0;
            const double *a = (const double *[]){ sw0, sw1, sw2 }[c];
            for (int i = 0; i < N; i++) {
                cr gv = { a[2 * i], 0 }, rv = RS[i].crw[c];
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

    { Z ac = v3blas_cc(1.5f, 0.5f);
      memcpy(ax, yv, sizeof ax); cblas_caxpy(N, &ac, xv, 1, ax, 1); /* baseline */ }
    v1xypa_c(q, t, v3blas_cc(0.25f, 0.1f), xy, N);
    v3dot_c(x, y, dot);
    v3had_c(x, y, v3c_wrap(had0, had1, had2, N));
    v3sqr_c(x, sqr);
    v1norm_c(q, norm, N);
    v3cross_c(x, y, v3c_wrap(cr0, cr1, cr2, N));
    v3crossscal_c(x, y, v3blas_cc(1.5f, 0.5f), v3c_wrap(cs0, cs1, cs2, N));
    v3crossdot_c(x, y, w, crd);
    v3crosssqr_c(x, y, crsq);
    v3crossxy_crossxz_c(x, y, w, v3c_wrap(uu0, uu1, uu2, N),
                                 v3c_wrap(vv0, vv1, vv2, N));
    v3crossxy_dotxz_c(x, y, w, v3c_wrap(pu0, pu1, pu2, N), pur);
    v3dotxy_dotxz_c(x, y, w, dxy_r, dxy_q);

    Z cc0[N], cc1[N], cc2[N], nur[N], nn0[N], nn1[N], nn2[N];
    Z rf0[N], rf1[N], rf2[N], ex0[N], ex1[N], ex2[N];
    Z dr0[N], dr1[N], dr2[N], mm0[N], mm1[N], mm2[N], ke[N];
    v3crosscross_c(x, y, w, v3c_wrap(cc0, cc1, cc2, N));
    v3norm_unit_c(x, 0.25f, nur, v3c_wrap(nn0, nn1, nn2, N));
    v3refl_c(x, y, 1.5f, 0.25f, v3c_wrap(rf0, rf1, rf2, N));
    v3exb_c(x, y, 1.5f, 0.25f, v3c_wrap(ex0, ex1, ex2, N));
    v3drag_c(x, 1.5f, 0.25f, v3c_wrap(dr0, dr1, dr2, N));
    v3mom_ke_c(q, x, v3c_wrap(mm0, mm1, mm2, N), ke);

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
    {
        const void *in[13] = { x0, x1, x2, y0, y1, y2, w0, w1, w2,
                                 q, t, xv, yv };
        octave_dump('c', 8, 1, 48, g, in, NULL);
    }
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

    { Z az = v3blas_zc(1.5, 0.5);
      memcpy(ax, yv, sizeof ax); cblas_zaxpy(N, &az, xv, 1, ax, 1);  /* baseline */ }
    v1xypa_z(q, t, v3blas_zc(0.25, 0.1), xy, N);
    v3dot_z(x, y, dot);
    v3had_z(x, y, v3z_wrap(had0, had1, had2, N));
    v3sqr_z(x, sqr);
    v1norm_z(q, norm, N);
    v3cross_z(x, y, v3z_wrap(cr0, cr1, cr2, N));
    v3crossscal_z(x, y, v3blas_zc(1.5, 0.5), v3z_wrap(cs0, cs1, cs2, N));
    v3crossdot_z(x, y, w, crd);
    v3crosssqr_z(x, y, crsq);
    v3crossxy_crossxz_z(x, y, w, v3z_wrap(uu0, uu1, uu2, N),
                                 v3z_wrap(vv0, vv1, vv2, N));
    v3crossxy_dotxz_z(x, y, w, v3z_wrap(pu0, pu1, pu2, N), pur);
    v3dotxy_dotxz_z(x, y, w, dxy_r, dxy_q);

    Z cc0[N], cc1[N], cc2[N], nur[N], nn0[N], nn1[N], nn2[N];
    Z rf0[N], rf1[N], rf2[N], ex0[N], ex1[N], ex2[N];
    Z dr0[N], dr1[N], dr2[N], mm0[N], mm1[N], mm2[N], ke[N];
    v3crosscross_z(x, y, w, v3z_wrap(cc0, cc1, cc2, N));
    v3norm_unit_z(x, 0.25, nur, v3z_wrap(nn0, nn1, nn2, N));
    v3refl_z(x, y, 1.5, 0.25, v3z_wrap(rf0, rf1, rf2, N));
    v3exb_z(x, y, 1.5, 0.25, v3z_wrap(ex0, ex1, ex2, N));
    v3drag_z(x, 1.5f, 0.25f, v3z_wrap(dr0, dr1, dr2, N));
    v3mom_ke_z(q, x, v3z_wrap(mm0, mm1, mm2, N), ke);

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
    {
        const void *in[13] = { x0, x1, x2, y0, y1, y2, w0, w1, w2,
                                 q, t, xv, yv };
        octave_dump('z', 16, 1, 48, g, in, NULL);
    }
}

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
    printf("v3blas_test: %s (%d failures)\n", fails ? "FAIL" : "OK", fails);
    return fails ? 1 : 0;
}
