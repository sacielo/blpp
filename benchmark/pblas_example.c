/* pblas_example.c - a tour of the pblas.h interface (plain C99).
 *
 * Build (against an installed OpenBLAS, see CMakeLists.txt):
 *   gcc -O2 -std=c99 -I ../pkgs/openblas/include/openblas \
 *       pblas_example.c -L ../pkgs/openblas/lib -lopenblas -lm \
 *       -o pblas_example
 *
 * Every operation below is one line; the results are freshly allocated
 * by pblas and are always freed with the matching _free.
 */
#include <stdio.h>
#include "pblas.h"

#define N 16

static void fill1(v1d v, double base)
{
    for (int i = 0; i < v.n; i++)
        v.x[i] = base + 0.1 * i;
}

static void fill3(v3d v, double base)
{
    for (int i = 0; i < v.n; i++) {
        v.x[i] = base + 0.1 * i;
        v.y[i] = base + 10 + 0.1 * i;
        v.z[i] = base + 20 + 0.1 * i;
    }
}

int main(void)
{
    double alpha = 1.5, a = 0.25, b = 1.5;

    /* inputs: three 3-vectors and three 1-vectors (borrowed by pblas) */
    v3d x = v3d_new(N), y = v3d_new(N), w = v3d_new(N);
    v1d q = v1d_new(N), t = v1d_new(N), yy = v1d_new(N), xx = v1d_new(N);
    fill3(x, 1.0); fill3(y, 4.0); fill3(w, 7.0);
    fill1(q, 10.0); fill1(t, 11.0); fill1(yy, 12.0); fill1(xx, 13.0);

    printf("n = %d\n\n", N);

    v1d r;
    v3d v3;

    /* y = alpha*x + y  (fresh vector: the old y is untouched) */
    r = v1axpy_d(xx, yy, alpha);
    printf("v1axpy_d       y = %gx + y        [0] = %.4f\n", alpha, r.x[0]);
    v1d_free(&r);

    /* r = q.*t + a */
    r = v1xypa_d(q, t, a);
    printf("v1xypa_d       r = q.*t + %g     [0] = %.4f\n", a, r.x[0]);
    v1d_free(&r);

    /* r = x1*y1 + x2*y2 + x3*y3 */
    r = v3dot_d(x, y);
    printf("v3dot_d        r = x1y1+x2y2+x3y3 [0] = %.4f\n", r.x[0]);
    v1d_free(&r);

    /* w = x .* y  (Hadamard, component-wise) */
    v3 = v3had_d(x, y);
    printf("v3had_d        w = x.*y           [0] = (%.4f, %.4f, %.4f)\n",
           v3.x[0], v3.y[0], v3.z[0]);
    v3d_free(&v3);

    /* r = x1^2 + x2^2 + x3^2 */
    r = v3sqr_d(x);
    printf("v3sqr_d        r = x1^2+x2^2+x3^2 [0] = %.4f\n", r.x[0]);
    v1d_free(&r);

    /* r = sqrt(q.q) */
    r = v1norm_d(q);
    printf("v1norm_d       r = sqrt(q.q)      [0] = %.4f\n", r.x[0]);
    v1d_free(&r);

    /* w = x^y  (cross product) */
    v3 = v3cross_d(x, y);
    printf("v3cross_d      w = x^y            [0] = (%.4f, %.4f, %.4f)\n",
           v3.x[0], v3.y[0], v3.z[0]);
    v3d_free(&v3);

    /* w = a*(x^y) */
    v3 = v3crossscal_d(x, y, b);
    printf("v3crossscal_d  w = %g*(x^y)       [0] = (%.4f, %.4f, %.4f)\n",
           b, v3.x[0], v3.y[0], v3.z[0]);
    v3d_free(&v3);

    /* r = (x^y).w */
    r = v3crossdot_d(x, y, w);
    printf("v3crossdot_d   r = (x^y).w        [0] = %.4f\n", r.x[0]);
    v1d_free(&r);

    /* r = (x^y).(x^y) */
    r = v3crosssqr_d(x, y);
    printf("v3crosssqr_d   r = (x^y).(x^y)    [0] = %.4f\n", r.x[0]);
    v1d_free(&r);

    /* two outputs: u = x^y, v = x^w  (one struct holds both) */
    v3uvd uv = v3crossxy_crossxz_d(x, y, w);
    printf("v3crossxy_crossxz_d  u = x^y, v = x^w  [0] u = (%.4f, %.4f, %.4f)\n",
           uv.u.x[0], uv.u.y[0], uv.u.z[0]);
    v3uvd_free(uv);

    /* u = x^y, r = x.w */
    v3urd ur = v3crossxy_dotxz_d(x, y, w);
    printf("v3crossxy_dotxz_d  u = x^y, r = x.w  [0] r = %.4f\n", ur.r.x[0]);
    v3urd_free(ur);

    /* r = x.y, q = x.w */
    v1rqd rq = v3dotxy_dotxz_d(x, y, w);
    printf("v3dotxy_dotxz_d  r = x.y, q = x.w  [0] = (%.4f, %.4f)\n",
           rq.r.x[0], rq.q.x[0]);
    v1rqd_free(rq);

    /* complex: same one-liners, elements are {re, im} pairs */
    v3z cx = v3z_new(N), cy = v3z_new(N);
    for (int i = 0; i < N; i++) {
        cx.x[i] = pblas_zc(1.0, 0.2 * i);
        cx.y[i] = pblas_zc(4.0, 0.1 * i);
        cx.z[i] = pblas_zc(7.0, 0.3 * i);
        cy.x[i] = pblas_zc(2.0, 0.3 * i);
        cy.y[i] = pblas_zc(5.0, 0.2 * i);
        cy.z[i] = pblas_zc(8.0, 0.1 * i);
    }
    v3z cz = v3crossscal_z(cx, cy, pblas_zc(1.5, 0.5));
    printf("\nv3crossscal_z  w = a*(x^y)      [0] = (%.4f + %.4f i, "
           "%.4f + %.4f i, %.4f + %.4f i)\n",
           cz.x[0].re, cz.x[0].im, cz.y[0].re, cz.y[0].im,
           cz.z[0].re, cz.z[0].im);
    v3z_free(&cz);

    /* strided input through a view (borrowed, inc 2) */
    double raw[3 * 2 * N];
    for (int i = 0; i < 3 * 2 * N; i++)
        raw[i] = 0.01 * i;
    v3d vx = v3d_view(raw, raw + 2 * N, raw + 4 * N, 2, 2, 2, N);
    v1d rs = v3sqr_d(vx);
    printf("v3sqr_d (view) r = x1^2+x2^2+x3^2 [0] = %.4f\n", rs.x[0]);
    v1d_free(&rs);

    /* inputs were allocated with _new, so they are freed too */
    v3d_free(&x); v3d_free(&y); v3d_free(&w); v3z_free(&cx); v3z_free(&cy);
    v1d_free(&q); v1d_free(&t); v1d_free(&yy); v1d_free(&xx);
    return 0;
}
