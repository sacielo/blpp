/* v3blas_hpp_test.cpp - verify the v3blas.hpp overloaded front-end.
 *
 * The overloads must (a) resolve the right precision purely from the
 * argument types and (b) be bitwise identical to the direct cblas_*
 * extension calls they are supposed to wrap.
 *
 * Build:
 *   g++ -O2 -std=c++11 -I ../pkgs/openblas/include/openblas \
 *       v3blas_hpp_test.cpp -L ../pkgs/openblas/lib -lopenblas -lm \
 *       -o v3blas_hpp_test
 */
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include "v3blas.hpp"

static int failures = 0;

/* murmur3 finalizer, same as the benchmark: data in [-0.5, 0.5) */
static uint32_t hash2(uint32_t s, uint32_t i){
    uint32_t h = s ^ i;
    h ^= h >> 16; h *= 0x85ebca6bU;
    h ^= h >> 13; h *= 0xc2b2ae35U;
    h ^= h >> 16;
    return h;
}
#define VAL(S, i) ((long double)((double)hash2((uint32_t)(S), (uint32_t)(i))  \
                                 / 4294967296.0) - 0.5L)

template <class T> static void setelem(T &v, long double x){ v = (T)x; }
template <> void setelem(v3blas_floatcomplex &v, long double x){
    v.re = (float)x; v.im = (float)(x * 0.7L);
}
template <> void setelem(v3blas_doublecomplex &v, long double x){
    v.re = (double)x; v.im = x * 0.7L;
}
template <class T> static void fill(T *v, int n, int S){
    for (int i = 0; i < n; i++) setelem(v[i], VAL(S, i));
}
template <class T> static void fill3(T *a0, T *a1, T *a2, int n, int S){
    for (int i = 0; i < n; i++) {
        setelem(a0[i], VAL(S, i));
        setelem(a1[i], VAL(S + 1, i));
        setelem(a2[i], VAL(S + 2, i));
    }
}
template <class T> static T mkscal(long double re, long double im){
    T s; setelem(s, re); (void)im; return s;
}
template <> v3blas_floatcomplex mkscal<v3blas_floatcomplex>(
    long double re, long double im){ v3blas_floatcomplex s = { (float)re, (float)im }; return s; }
template <> v3blas_doublecomplex mkscal<v3blas_doublecomplex>(
    long double re, long double im){ v3blas_doublecomplex s = { (double)re, (double)im }; return s; }

static int chk(const char *name, const void *a, const void *b, size_t bytes){
    int ok = memcmp(a, b, bytes) == 0;
    printf("%-24s %s\n", name, ok ? "OK" : "FAIL");
    if (!ok) failures++;
    return !ok;
}

/* CAX/CXA/CCS are the cblas scalar args: by value for s/d, by pointer
 * for c/z (the cblas complex convention). */
#define RUN_ALL(P, T, CAX, CXA, CCS) do {                                      \
    const int n = 64, nb = n * (int)sizeof(T);                                 \
    T bx0[64], bx1[64], bx2[64], by0[64], by1[64], by2[64];                    \
    T bw0[64], bw1[64], bw2[64], bq[64], bt[64], bxv[64], byv[64];             \
    T wo[6][64], ro[6][64];                                                    \
    memset(wo, 0, sizeof wo); memset(ro, 0, sizeof ro);                        \
    fill3(bx0, bx1, bx2, n, 1); fill3(by0, by1, by2, n, 4);                    \
    fill3(bw0, bw1, bw2, n, 7); fill(bq, n, 10); fill(bt, n, 11);              \
    fill(bxv, n, 12); fill(byv, n, 13);                                        \
    v3 ## P x = v3 ## P ## _wrap(bx0, bx1, bx2, n);                            \
    v3 ## P y = v3 ## P ## _wrap(by0, by1, by2, n);                            \
    v3 ## P w = v3 ## P ## _wrap(bw0, bw1, bw2, n);                            \
    T s_ax = mkscal<T>(1.5L, 0.5L), s_xa = mkscal<T>(0.25L, 0.1L);             \
    T s_cs = mkscal<T>(1.5L, 0.5L);                                            \
    int bad = 0;                                                               \
    memcpy(ro[0], byv, nb); cblas_ ## P ## axpy(n, CAX, bxv, 1, ro[0], 1);     \
    v1axpy(s_ax, bxv, byv, wo[0], n);                                          \
    bad += chk("v1axpy_" #P, wo[0], ro[0], nb);                                \
    cblas_ ## P ## 1xypa(n, CXA, bq, 1, bt, 1, ro[0], 1);                      \
    v1xypa(bq, bt, s_xa, wo[0], n);                                            \
    bad += chk("v1xypa_" #P, wo[0], ro[0], nb);                                \
    cblas_ ## P ## 1norm(n, bq, 1, ro[0], 1);                                  \
    v1norm(bq, wo[0], n);                                                      \
    bad += chk("v1norm_" #P, wo[0], ro[0], nb);                                \
    cblas_ ## P ## 3dot(n, x.x, 1, x.y, 1, x.z, 1,                             \
                        y.x, 1, y.y, 1, y.z, 1, ro[0], 1);                     \
    v3dot(x, y, wo[0]);                                                        \
    bad += chk("v3dot_" #P, wo[0], ro[0], nb);                                 \
    cblas_ ## P ## 3sqr(n, x.x, 1, x.y, 1, x.z, 1, ro[0], 1);                  \
    v3sqr(x, wo[0]);                                                           \
    bad += chk("v3sqr_" #P, wo[0], ro[0], nb);                                 \
    cblas_ ## P ## 3crossdot(n, x.x, 1, x.y, 1, x.z, 1,                        \
                             y.x, 1, y.y, 1, y.z, 1,                           \
                             w.x, 1, w.y, 1, w.z, 1, ro[0], 1);                \
    v3crossdot(x, y, w, wo[0]);                                                \
    bad += chk("v3crossdot_" #P, wo[0], ro[0], nb);                            \
    cblas_ ## P ## 3crosssqr(n, x.x, 1, x.y, 1, x.z, 1,                        \
                             y.x, 1, y.y, 1, y.z, 1, ro[0], 1);                \
    v3crosssqr(x, y, wo[0]);                                                   \
    bad += chk("v3crosssqr_" #P, wo[0], ro[0], nb);                            \
    cblas_ ## P ## 3dotxy_dotxz(n, x.x, 1, x.y, 1, x.z, 1,                     \
                                y.x, 1, y.y, 1, y.z, 1,                        \
                                w.x, 1, w.y, 1, w.z, 1, ro[0], 1, ro[1], 1);   \
    v3dotxy_dotxz(x, y, w, wo[0], wo[1]);                                      \
    bad += chk("v3dotxy_dotxz_" #P " r", wo[0], ro[0], nb);                    \
    bad += chk("v3dotxy_dotxz_" #P " q", wo[1], ro[1], nb);                    \
    cblas_ ## P ## 3had(n, x.x, 1, x.y, 1, x.z, 1,                             \
                        y.x, 1, y.y, 1, y.z, 1,                                \
                        ro[0], 1, ro[1], 1, ro[2], 1);                         \
    v3had(x, y, wo[0], wo[1], wo[2]);                                          \
    bad += chk("v3had_" #P, wo[0], ro[0], 3 * nb);                             \
    cblas_ ## P ## 3cross(n, x.x, 1, x.y, 1, x.z, 1,                           \
                          y.x, 1, y.y, 1, y.z, 1,                              \
                          ro[0], 1, ro[1], 1, ro[2], 1);                       \
    v3cross(x, y, wo[0], wo[1], wo[2]);                                        \
    bad += chk("v3cross_" #P, wo[0], ro[0], 3 * nb);                           \
    cblas_ ## P ## 3crossscal(n, CCS, x.x, 1, x.y, 1, x.z, 1,                  \
                              y.x, 1, y.y, 1, y.z, 1,                          \
                              ro[0], 1, ro[1], 1, ro[2], 1);                   \
    v3crossscal(x, y, s_cs, wo[0], wo[1], wo[2]);                              \
    bad += chk("v3crossscal_" #P, wo[0], ro[0], 3 * nb);                       \
    cblas_ ## P ## 3crossxy_crossxz(n, x.x, 1, x.y, 1, x.z, 1,                 \
                                    y.x, 1, y.y, 1, y.z, 1,                    \
                                    w.x, 1, w.y, 1, w.z, 1,                    \
                                    ro[0], 1, ro[1], 1, ro[2], 1,              \
                                    ro[3], 1, ro[4], 1, ro[5], 1);             \
    v3crossxy_crossxz(x, y, w, wo[0], wo[1], wo[2], wo[3], wo[4], wo[5]);      \
    bad += chk("v3crossxy_crossxz_" #P, wo[0], ro[0], 6 * nb);                 \
    cblas_ ## P ## 3crossxy_dotxz(n, x.x, 1, x.y, 1, x.z, 1,                   \
                                  y.x, 1, y.y, 1, y.z, 1,                      \
                                  w.x, 1, w.y, 1, w.z, 1,                      \
                                  ro[0], 1, ro[1], 1, ro[2], 1, ro[3], 1);     \
    v3crossxy_dotxz(x, y, w, wo[0], wo[1], wo[2], wo[3]);                      \
    bad += chk("v3crossxy_dotxz_" #P, wo[0], ro[0], 4 * nb);                   \
    cblas_ ## P ## 3crosscross(n, x.x, 1, x.y, 1, x.z, 1,                      \
                               y.x, 1, y.y, 1, y.z, 1,                         \
                               w.x, 1, w.y, 1, w.z, 1,                         \
                               ro[0], 1, ro[1], 1, ro[2], 1);                  \
    v3crosscross(x, y, w, wo[0], wo[1], wo[2]);                                \
    bad += chk("v3crosscross_" #P, wo[0], ro[0], 3 * nb);                      \
    cblas_ ## P ## 3norm_unit(n, x.x, 1, x.y, 1, x.z, 1, 0.25,                 \
                              ro[0], 1, ro[1], 1, ro[2], 1, ro[3], 1);         \
    v3norm_unit(x, 0.25, wo[0], wo[1], wo[2], wo[3]);                          \
    bad += chk("v3norm_unit_" #P, wo[0], ro[0], 4 * nb);                       \
    cblas_ ## P ## 3refl(n, x.x, 1, x.y, 1, x.z, 1,                            \
                         y.x, 1, y.y, 1, y.z, 1, 1.5, 0.25,                    \
                         ro[0], 1, ro[1], 1, ro[2], 1);                        \
    v3refl(x, y, 1.5, 0.25, wo[0], wo[1], wo[2]);                              \
    bad += chk("v3refl_" #P, wo[0], ro[0], 3 * nb);                            \
    cblas_ ## P ## 3exb(n, x.x, 1, x.y, 1, x.z, 1,                             \
                        y.x, 1, y.y, 1, y.z, 1, 1.5, 0.25,                     \
                        ro[0], 1, ro[1], 1, ro[2], 1);                         \
    v3exb(x, y, 1.5, 0.25, wo[0], wo[1], wo[2]);                               \
    bad += chk("v3exb_" #P, wo[0], ro[0], 3 * nb);                             \
    cblas_ ## P ## 3drag(n, x.x, 1, x.y, 1, x.z, 1, 1.5, 0.25,                 \
                         ro[0], 1, ro[1], 1, ro[2], 1);                        \
    v3drag(x, 1.5, 0.25, wo[0], wo[1], wo[2]);                                 \
    bad += chk("v3drag_" #P, wo[0], ro[0], 3 * nb);                            \
    cblas_ ## P ## 3mom_ke(n, bq, 1, x.x, 1, x.y, 1, x.z, 1,                   \
                           ro[0], 1, ro[1], 1, ro[2], 1, ro[3], 1);            \
    v3mom_ke(bq, x, wo[0], wo[1], wo[2], wo[3]);                               \
    bad += chk("v3mom_ke_" #P, wo[0], ro[0], 4 * nb);                          \
    if (bad) printf("%d failures in " #P "\n", bad);                           \
} while (0)

int main(){
    RUN_ALL(s, float, s_ax, s_xa, s_cs);
    RUN_ALL(d, double, s_ax, s_xa, s_cs);
    RUN_ALL(c, v3blas_floatcomplex, &s_ax, &s_xa, &s_cs);
    RUN_ALL(z, v3blas_doublecomplex, &s_ax, &s_xa, &s_cs);
    printf("v3blas_hpp_test: %s (%d failures)\n",
           failures ? "FAILED" : "OK", failures);
    return failures != 0;
}
