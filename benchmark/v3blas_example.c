/* v3blas_example.c - a tour of the v3blas.h interface (plain C11).
 *
 * Build (against an installed OpenBLAS, see CMakeLists.txt):
 *   gcc -O2 -std=c11 -I ../pkgs/openblas/include/openblas \
 *       v3blas_example.c -L ../pkgs/openblas/lib -lopenblas -lm \
 *       -o v3blas_example
 *
 * Every operation below is one line.  v3blas never allocates: every
 * array here is a plain stack buffer the caller owns, and every
 * operation is a void function that writes into it.  The precision-
 * generic names (v3cross, v3dot, ...) pick s/d/c/z from the type of
 * the output pointer via C11 _Generic; the suffixed names (v3cross_d)
 * are always available too.
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
    v1axpy(alpha, xx, yy, o0, N);
    printf("v1axpy       r = %gx + y        [0] = %.4f\n", alpha, o0[0]);

    /* r = q.*t + a */
    v1xypa(q, t, a, o0, N);
    printf("v1xypa       r = q.*t + %g     [0] = %.4f\n", a, o0[0]);

    /* r = x1y1 + x2y2 + x3y3 */
    v3dot(x, y, o0);
    printf("v3dot        r = x1y1+x2y2+x3y3 [0] = %.4f\n", o0[0]);

    /* w = x .* y  (Hadamard, component-wise) */
    v3had(x, y, o0, o1, o2);
    printf("v3had        w = x.*y           [0] = (%.4f, %.4f, %.4f)\n",
           o0[0], o1[0], o2[0]);

    /* r = x1^2 + x2^2 + x3^2 */
    v3sqr(x, o0);
    printf("v3sqr        r = x1^2+x2^2+x3^2 [0] = %.4f\n", o0[0]);

    /* r = sqrt(q·q) */
    v1norm(q, o0, N);
    printf("v1norm       r = sqrt(q·q)      [0] = %.4f\n", o0[0]);

    /* w = x∧y  (cross product) */
    v3cross(x, y, o0, o1, o2);
    printf("v3cross      w = x∧y            [0] = (%.4f, %.4f, %.4f)\n",
           o0[0], o1[0], o2[0]);

    /* w = a·(x∧y) */
    v3crossscal(x, y, b, o0, o1, o2);
    printf("v3crossscal  w = %g·(x∧y)       [0] = (%.4f, %.4f, %.4f)\n",
           b, o0[0], o1[0], o2[0]);

    /* r = (x∧y)·w */
    v3crossdot(x, y, w, o0);
    printf("v3crossdot   r = (x∧y)·w        [0] = %.4f\n", o0[0]);

    /* r = (x∧y)·(x∧y) */
    v3crosssqr(x, y, o0);
    printf("v3crosssqr   r = (x∧y)·(x∧y)    [0] = %.4f\n", o0[0]);

    /* two outputs, one call: u = x∧y, v = x∧w */
    v3crossxy_crossxz(x, y, w, o0, o1, o2, o3, o4, o5);
    printf("v3crossxy_crossxz  u = x∧y, v = x∧w  [0] u = (%.4f, %.4f, %.4f)\n",
           o0[0], o1[0], o2[0]);

    /* u = x∧y, r = x·w */
    v3crossxy_dotxz(x, y, w, o0, o1, o2, o3);
    printf("v3crossxy_dotxz  u = x∧y, r = x·w  [0] r = %.4f\n", o3[0]);

    /* r = x·y, q = x·w */
    v3dotxy_dotxz(x, y, w, o0, o1);
    printf("v3dotxy_dotxz  r = x·y, q = x·w  [0] = (%.4f, %.4f)\n",
           o0[0], o1[0]);

    /* complex: the same generic names, elements are {re, im} pairs */
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
    v3crossscal(cx, cy, v3blas_zc(1.5, 0.5), cz0, cz1, cz2);
    printf("\nv3crossscal (z)  w = a·(x∧y)   [0] = (%.4f + %.4f i, "
           "%.4f + %.4f i, %.4f + %.4f i)\n",
           cz0[0].re, cz0[0].im, cz1[0].re, cz1[0].im,
           cz2[0].re, cz2[0].im);

    /* strided input: the view fields are public, set the incs by hand */
    double raw[3 * 2 * N];
    for (int i = 0; i < 3 * 2 * N; i++)
        raw[i] = 0.01 * i;
    v3d vx = v3d_wrap(raw, raw + 2 * N, raw + 4 * N, N);
    vx.incx = 2; vx.incy = 2; vx.incz = 2;
    v3sqr_d(vx, o0);   /* suffixed name - a v3d view says the precision */
    printf("v3sqr_d (strided) r = x1^2+x2^2+x3^2 [0] = %.4f\n", o0[0]);

    /* nothing to free: the caller owned everything all along */
    return 0;
}
