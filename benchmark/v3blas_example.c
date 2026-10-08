/* v3blas_example.c - a tour of the v3blas.h interface (plain C99).
 *
 * Build (against an installed OpenBLAS, see CMakeLists.txt):
 *   gcc -O2 -std=c99 -I ../pkgs/openblas/include/openblas \
 *       v3blas_example.c -L ../pkgs/openblas/lib -lopenblas -lm \
 *       -o v3blas_example
 *
 * Every operation below is one line.  v3blas never allocates: every
 * array here is a plain stack buffer the caller owns; the returns are
 * just views of those buffers.
 */
#include <stdio.h>
#include "v3blas.h"

#define N 16

static void ramp1(double *v, double base){
    for (int i = 0; i < N; i++)
        v[i] = base + 0.1 * i;
}

static void ramp3(v3d v, double base){
    for (int i = 0; i < v.n; i++) {
        v.x[i] = base + 0.1 * i;
        v.y[i] = base + 10 + 0.1 * i;
        v.z[i] = base + 20 + 0.1 * i;
    }
}

int main(void){
    double alpha = 1.5, a = 0.25, b = 1.5;

    /* inputs: three 3-vectors and four 1-vectors, all caller-owned */
    double x0[N], x1[N], x2[N], y0[N], y1[N], y2[N], w0[N], w1[N], w2[N];
    double q[N], t[N], yy[N], xx[N];
    v3d x = v3d_wrap(x0, x1, x2, N), y = v3d_wrap(y0, y1, y2, N);
    v3d w = v3d_wrap(w0, w1, w2, N);
    ramp3(x, 1.0); ramp3(y, 4.0); ramp3(w, 7.0);
    ramp1(q, 10.0); ramp1(t, 11.0); ramp1(yy, 12.0); ramp1(xx, 13.0);

    /* output buffers, reused between demos */
    double o0[N], o1[N], o2[N], o3[N], o4[N], o5[N];

    printf("n = %d\n\n", N);

    /* r = alpha*x + y (y itself is untouched) */
    v1axpy_d(alpha, xx, yy, o0, N);
    printf("v1axpy_d       r = %gx + y        [0] = %.4f\n", alpha, o0[0]);

    /* r = q.*t + a */
    v1xypa_d(q, t, a, o0, N);
    printf("v1xypa_d       r = q.*t + %g     [0] = %.4f\n", a, o0[0]);

    /* r = x1y1 + x2y2 + x3y3 */
    v3dot_d(x, y, o0);
    printf("v3dot_d        r = x1y1+x2y2+x3y3 [0] = %.4f\n", o0[0]);

    /* w = x .* y  (Hadamard, component-wise) */
    v3d v3 = v3had_d(x, y, o0, o1, o2);
    printf("v3had_d        w = x.*y           [0] = (%.4f, %.4f, %.4f)\n",
           v3.x[0], v3.y[0], v3.z[0]);

    /* r = x1^2 + x2^2 + x3^2 */
    v3sqr_d(x, o0);
    printf("v3sqr_d        r = x1^2+x2^2+x3^2 [0] = %.4f\n", o0[0]);

    /* r = sqrt(q·q) */
    v1norm_d(q, o0, N);
    printf("v1norm_d       r = sqrt(q·q)      [0] = %.4f\n", o0[0]);

    /* w = x∧y  (cross product) */
    v3 = v3cross_d(x, y, o0, o1, o2);
    printf("v3cross_d      w = x∧y            [0] = (%.4f, %.4f, %.4f)\n",
           v3.x[0], v3.y[0], v3.z[0]);

    /* w = a·(x∧y) */
    v3 = v3crossscal_d(x, y, b, o0, o1, o2);
    printf("v3crossscal_d  w = %g·(x∧y)       [0] = (%.4f, %.4f, %.4f)\n",
           b, v3.x[0], v3.y[0], v3.z[0]);

    /* r = (x∧y)·w */
    v3crossdot_d(x, y, w, o0);
    printf("v3crossdot_d   r = (x∧y)·w        [0] = %.4f\n", o0[0]);

    /* r = (x∧y)·(x∧y) */
    v3crosssqr_d(x, y, o0);
    printf("v3crosssqr_d   r = (x∧y)·(x∧y)    [0] = %.4f\n", o0[0]);

    /* two outputs: u = x∧y, v = x∧w  (one struct holds both views) */
    v3uvd uv = v3crossxy_crossxz_d(x, y, w, o0, o1, o2, o3, o4, o5);
    printf("v3crossxy_crossxz_d  u = x∧y, v = x∧w  [0] u = (%.4f, %.4f, %.4f)\n",
           uv.u.x[0], uv.u.y[0], uv.u.z[0]);

    /* u = x∧y, r = x·w */
    v3urd ur = v3crossxy_dotxz_d(x, y, w, o0, o1, o2, o3);
    printf("v3crossxy_dotxz_d  u = x∧y, r = x·w  [0] r = %.4f\n", ur.r[0]);

    /* r = x·y, q = x·w */
    rqd rq = v3dotxy_dotxz_d(x, y, w, o0, o1);
    printf("v3dotxy_dotxz_d  r = x·y, q = x·w  [0] = (%.4f, %.4f)\n",
           rq.r[0], rq.q[0]);

    /* complex: same one-liners, elements are {re, im} pairs */
    v3blas_doublecomplex cx0[N], cx1[N], cx2[N];
    v3blas_doublecomplex cy0[N], cy1[N], cy2[N];
    v3blas_doublecomplex cz0[N], cz1[N], cz2[N];
    for (int i = 0; i < N; i++) {
        cx0[i] = v3blas_zc(1.0, 0.2 * i);
        cx1[i] = v3blas_zc(4.0, 0.1 * i);
        cx2[i] = v3blas_zc(7.0, 0.3 * i);
        cy0[i] = v3blas_zc(2.0, 0.3 * i);
        cy1[i] = v3blas_zc(5.0, 0.2 * i);
        cy2[i] = v3blas_zc(8.0, 0.1 * i);
    }
    v3z cx = v3z_wrap(cx0, cx1, cx2, N), cy = v3z_wrap(cy0, cy1, cy2, N);
    v3z cz = v3crossscal_z(cx, cy, v3blas_zc(1.5, 0.5), cz0, cz1, cz2);
    printf("\nv3crossscal_z  w = a·(x∧y)      [0] = (%.4f + %.4f i, "
           "%.4f + %.4f i, %.4f + %.4f i)\n",
           cz.x[0].re, cz.x[0].im, cz.y[0].re, cz.y[0].im,
           cz.z[0].re, cz.z[0].im);

    /* strided input: the view fields are public, set the incs by hand */
    double raw[3 * 2 * N];
    for (int i = 0; i < 3 * 2 * N; i++)
        raw[i] = 0.01 * i;
    v3d vx = v3d_wrap(raw, raw + 2 * N, raw + 4 * N, N);
    vx.incx = 2; vx.incy = 2; vx.incz = 2;
    v3sqr_d(vx, o0);
    printf("v3sqr_d (strided) r = x1^2+x2^2+x3^2 [0] = %.4f\n", o0[0]);

    /* nothing to free: the caller owned everything all along */
    return 0;
}
