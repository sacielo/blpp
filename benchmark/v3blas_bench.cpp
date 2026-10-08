/*
 * v3blas_bench.cpp - benchmark the v3blas.h higher-level interface against
 * raw cblas extension calls, and verify both produce IDENTICAL output.
 *
 * Neither side allocates inside the timed region: all arrays are
 * allocated once up front (v3blas itself never allocates at all), so
 * the "overhead" column is pure interface cost: view structs by value,
 * no free bookkeeping.
 *
 * Correctness: max_rel_err compares the v3blas and cblas outputs
 * element-wise (not against a reference).  The kernels are bit-level
 * identical to the interface computations (generic, deterministic), so
 * max_rel_err must be exactly 0.000e+00; any other value means the
 * interface is lying about what it calls.
 *
 * Build (see CMakeLists.txt):
 *   g++ -O2 -I <prefix>/include/openblas v3blas_bench.cpp \
 *       -L <prefix>/lib -lopenblas -o v3blas_bench
 * Run:  ./v3blas_bench [n] [s|d|c|z]
 */
#include "v3blas.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <cmath>
#include <string>
#include <vector>
#include <functional>

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

struct Timing {
    double t_run;   /* seconds spent in work iterations */
    long iters;
};

static Timing timed_run(const std::function<void()> &fn)
{
    static const double kMinTime = 0.5;   /* seconds */
    static const long kMinIters = 3;
    static const double kMaxTime = 10.0;
    Timing tm = { 0.0, 0 };
    double t0 = now_sec();
    while (true) {
        fn();
        tm.iters++;
        double el = now_sec() - t0;
        if (el >= kMinTime && tm.iters >= kMinIters) { tm.t_run = el; return tm; }
        if (el >= kMaxTime) { tm.t_run = el; return tm; }
    }
}

static uint32_t hash2(uint32_t s, uint32_t i)
{
    uint32_t h = s * 2654435761u + i * 2246822519u;
    h ^= h >> 16; h *= 2246822519u;
    h ^= h >> 13; h *= 3266489917u;
    h ^= h >> 16;
    return h;
}

#define VAL(S, i) ((long double)((double)hash2((uint32_t)(S), (uint32_t)(i)) / 4294967296.0) - 0.5L)

static long double rabs_ld(long double a) { return a < 0 ? -a : a; }

/* relative error of got[] vs want[] (element size esz bytes, ICX: complex
 * elements get separate re/im checks); 2^-40 tolerance so "identical"
 * really means identical */
static double max_rel_diff(size_t esz, int icx, const void *gotv,
                           const void *wantv, int n)
{
    const unsigned char *g = (const unsigned char *)gotv;
    const unsigned char *w = (const unsigned char *)wantv;
    long double worst = 0.0L;
    for (int i = 0; i < n; i++) {
        long double gr, gi, wr, wi;
        if (esz == 4) {
            float a, b; memcpy(&a, g + i * 4, 4); memcpy(&b, w + i * 4, 4);
            gr = a; wr = b; gi = 0; wi = 0;
        } else if (esz == 8 && !icx) {
            double a, b; memcpy(&a, g + i * 8, 8); memcpy(&b, w + i * 8, 8);
            gr = a; wr = b; gi = 0; wi = 0;
        } else if (esz == 8) {
            float a, b, c, d;
            memcpy(&a, g + i * 8, 4); memcpy(&b, g + i * 8 + 4, 4);
            memcpy(&c, w + i * 8, 4); memcpy(&d, w + i * 8 + 4, 4);
            gr = a; gi = b; wr = c; wi = d;
        } else {
            double a, b, c, d;
            memcpy(&a, g + i * 16, 8); memcpy(&b, g + i * 16 + 8, 8);
            memcpy(&c, w + i * 16, 8); memcpy(&d, w + i * 16 + 8, 8);
            gr = a; gi = b; wr = c; wi = d;
        }
        long double ar = rabs_ld(gr - wr) / (rabs_ld(wr) > 1 ? rabs_ld(wr) : 1);
        long double ai = rabs_ld(gi - wi) / (rabs_ld(wi) > 1 ? rabs_ld(wi) : 1);
        long double e = ar > ai ? ar : ai;
        if (e > worst) worst = e;
    }
    return (double)worst;
}

struct Result {
    std::string name;
    double ms_v3blas;    /* per-call ms, v3blas.h interface */
    double ms_cblas;     /* per-call ms, raw cblas call */
    double overhead;     /* v3blas/cblas time ratio, ~1.00 expected */
    double flops_rate;   /* math FLOP/s delivered via v3blas */
    double max_rel_err;  /* v3blas vs cblas, must be 0 */
    long iters;
    bool ok;
    std::string eq;
};

static std::vector<Result> results;

// data fills: pseudo-random in [-0.5, 0.5), deterministic (murmur3 finalizer);
// components of a 3-vector get seeds S, S+1, S+2
static void fill1_s(float *v, int n, int S) { for (int i = 0; i < n; i++) v[i] = (float)VAL(S, i); }
static void fill1_d(double *v, int n, int S) { for (int i = 0; i < n; i++) v[i] = (double)VAL(S, i); }
static void fill1_c(v3blas_floatcomplex *v, int n, int S) { for (int i = 0; i < n; i++) { v[i].re = (float)VAL(S, i); v[i].im = (float)(VAL(S, i) * 0.7L); } }
static void fill1_z(v3blas_doublecomplex *v, int n, int S) { for (int i = 0; i < n; i++) { v[i].re = VAL(S, i); v[i].im = VAL(S, i) * 0.7L; } }

static void fill3_s(float *a0, float *a1, float *a2, int n, int S) { for (int i = 0; i < n; i++) { a0[i] = (float)VAL(S, i); a1[i] = (float)VAL(S + 1, i); a2[i] = (float)VAL(S + 2, i); } }
static void fill3_d(double *a0, double *a1, double *a2, int n, int S) { for (int i = 0; i < n; i++) { a0[i] = (double)VAL(S, i); a1[i] = (double)VAL(S + 1, i); a2[i] = (double)VAL(S + 2, i); } }
static void fill3_c(v3blas_floatcomplex *a0, v3blas_floatcomplex *a1, v3blas_floatcomplex *a2, int n, int S) { for (int i = 0; i < n; i++) { a0[i].re = (float)VAL(S, i); a0[i].im = (float)(VAL(S, i) * 0.7L); a1[i].re = (float)VAL(S + 1, i); a1[i].im = (float)(VAL(S + 1, i) * 0.7L); a2[i].re = (float)VAL(S + 2, i); a2[i].im = (float)(VAL(S + 2, i) * 0.7L); } }
static void fill3_z(v3blas_doublecomplex *a0, v3blas_doublecomplex *a1, v3blas_doublecomplex *a2, int n, int S) { for (int i = 0; i < n; i++) { a0[i].re = VAL(S, i); a0[i].im = VAL(S, i) * 0.7L; a1[i].re = VAL(S + 1, i); a1[i].im = VAL(S + 1, i) * 0.7L; a2[i].re = VAL(S + 2, i); a2[i].im = VAL(S + 2, i) * 0.7L; } }

static void add_result(const char *name, const Timing &tp, const Timing &tc,
                       double err, double fpse, int n, const char *eq)
{
    double per_p = tp.t_run / tp.iters;
    double per_c = tc.t_run / tc.iters;
    results.push_back({name, per_p * 1e3, per_c * 1e3, per_p / per_c,
                       fpse * n / per_p, err, tp.iters, err == 0.0, eq});
}

/*
 * One body, four precisions. ICX: 0 = real, 1 = complex element.
 * INIT: statements that initialize the three scalars (comma-free).
 * CAX/CXA/CCS: the scalar arguments in the form the cblas prototypes
 * expect (by value for real, as void* for complex); the v3blas calls use
 * the local s_ax/s_xa/s_cs directly.
 * Inputs are one malloc'd slab sliced into 13 unit-stride arrays.
 * ro[0..5] are the cblas reference outputs, wo[0..5] the v3blas ones.
 */
#define BENCH_PREC(P, T, ICX, INIT, CAX, CXA, CCS)                         \
static void bench_all_ ## P(int n)                                         \
{                                                                          \
    T s_ax, s_xa, s_cs;                                                    \
    INIT;                                                                  \
    const void *pax = (const void *)&s_ax;                                 \
    const void *pxa = (const void *)&s_xa;                                 \
    const void *pcs = (const void *)&s_cs;                                 \
    (void)pax; (void)pxa; (void)pcs;                                       \
    const size_t esz = sizeof(T);                                          \
                                                                           \
    T *in = (T *)malloc(13 * (size_t)n * esz);                             \
    T *bx0 = in, *bx1 = in + n, *bx2 = in + 2 * n, *by0 = in + 3 * n,      \
      *by1 = in + 4 * n, *by2 = in + 5 * n, *bw0 = in + 6 * n,             \
      *bw1 = in + 7 * n, *bw2 = in + 8 * n, *bq = in + 9 * n,              \
      *bt = in + 10 * n, *bxv = in + 11 * n, *byv = in + 12 * n;           \
    v3 ## P x = v3 ## P ## _wrap(bx0, bx1, bx2, n);                        \
    v3 ## P y = v3 ## P ## _wrap(by0, by1, by2, n);                        \
    v3 ## P w = v3 ## P ## _wrap(bw0, bw1, bw2, n);                        \
    fill3_ ## P(bx0, bx1, bx2, n, 1); fill3_ ## P(by0, by1, by2, n, 4);    \
    fill3_ ## P(bw0, bw1, bw2, n, 7); fill1_ ## P(bq, n, 10);              \
    fill1_ ## P(bt, n, 11); fill1_ ## P(bxv, n, 13);                       \
    fill1_ ## P(byv, n, 12);                                               \
                                                                           \
    T *scratch = (T *)malloc(12 * (size_t)n * esz);                        \
    T *ro[6], *wo[6];                                                      \
    for (int k = 0; k < 6; k++) {                                          \
        ro[k] = scratch + (size_t)k * n;                                   \
        wo[k] = scratch + (size_t)(6 + k) * n;                             \
    }                                                                      \
                                                                           \
    { memcpy(ro[0], byv, (size_t)n * esz);                                 \
      cblas_ ## P ## axpy(n, CAX, bxv, 1, ro[0], 1);                       \
      double e = max_rel_diff(esz, ICX, v1axpy_ ## P(s_ax, bxv, byv,       \
                                                     wo[0], n), ro[0], n); \
      Timing tp = timed_run([&]{ v1axpy_ ## P(s_ax, bxv, byv, wo[0], n); }); \
      Timing tc = timed_run([&]{ memcpy(ro[0], byv, (size_t)n * esz);      \
          cblas_ ## P ## axpy(n, CAX, bxv, 1, ro[0], 1); });               \
      add_result("v1axpy_" #P, tp, tc, e, (ICX ? 8.0 : 2.0), n, "y=a*x+y"); } \
                                                                           \
    { cblas_ ## P ## 1xypa(n, CXA, bq, 1, bt, 1, ro[0], 1);                \
      double e = max_rel_diff(esz, ICX,                                    \
                              v1xypa_ ## P(bq, bt, s_xa, wo[0], n),        \
                              ro[0], n);                                   \
      Timing tp = timed_run([&]{ v1xypa_ ## P(bq, bt, s_xa, wo[0], n); }); \
      Timing tc = timed_run([&]{ cblas_ ## P ## 1xypa(n, CXA, bq, 1,       \
          bt, 1, ro[0], 1); });                                            \
      add_result("v1xypa_" #P, tp, tc, e, (ICX ? 8.0 : 2.0), n, "r=q.*t+a"); } \
                                                                           \
    { cblas_ ## P ## 3dot(n, x.x, 1, x.y, 1, x.z, 1,                       \
                          y.x, 1, y.y, 1, y.z, 1, ro[0], 1);               \
      double e = max_rel_diff(esz, ICX, v3dot_ ## P(x, y, wo[0]),          \
                              ro[0], n);                                   \
      Timing tp = timed_run([&]{ v3dot_ ## P(x, y, wo[0]); });             \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3dot(n,                    \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1, ro[0], 1); });   \
      add_result("v3dot_" #P, tp, tc, e, (ICX ? 22.0 : 5.0), n, "r=x1y1+x2y2+x3y3"); } \
                                                                           \
    { cblas_ ## P ## 3had(n, x.x, 1, x.y, 1, x.z, 1,                       \
                          y.x, 1, y.y, 1, y.z, 1,                          \
                          ro[0], 1, ro[1], 1, ro[2], 1);                   \
      v3 ## P r = v3had_ ## P(x, y, wo[0], wo[1], wo[2]);                  \
      double e = max_rel_diff(esz, ICX, r.x, ro[0], n);                    \
      e = fmax(e, max_rel_diff(esz, ICX, r.y, ro[1], n));                  \
      e = fmax(e, max_rel_diff(esz, ICX, r.z, ro[2], n));                  \
      Timing tp = timed_run([&]{ v3had_ ## P(x, y, wo[0], wo[1], wo[2]); }); \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3had(n,                    \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1,                  \
          ro[0], 1, ro[1], 1, ro[2], 1); });                               \
      add_result("v3had_" #P, tp, tc, e, (ICX ? 18.0 : 3.0), n, "w=x.*y"); } \
                                                                           \
    { cblas_ ## P ## 3sqr(n, x.x, 1, x.y, 1, x.z, 1, ro[0], 1);            \
      double e = max_rel_diff(esz, ICX, v3sqr_ ## P(x, wo[0]),             \
                              ro[0], n);                                   \
      Timing tp = timed_run([&]{ v3sqr_ ## P(x, wo[0]); });                \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3sqr(n,                    \
          x.x, 1, x.y, 1, x.z, 1, ro[0], 1); });                           \
      add_result("v3sqr_" #P, tp, tc, e, (ICX ? 22.0 : 5.0), n, "r=x1^2+x2^2+x3^2"); } \
                                                                           \
    { cblas_ ## P ## 1norm(n, bq, 1, ro[0], 1);                            \
      double e = max_rel_diff(esz, ICX, v1norm_ ## P(bq, wo[0], n),        \
                              ro[0], n);                                   \
      Timing tp = timed_run([&]{ v1norm_ ## P(bq, wo[0], n); });           \
      Timing tc = timed_run([&]{ cblas_ ## P ## 1norm(n, bq, 1,            \
          ro[0], 1); });                                                   \
      add_result("v1norm_" #P, tp, tc, e, (ICX ? 16.0 : 2.0), n, "r=sqrt(q·q)"); } \
                                                                           \
    { cblas_ ## P ## 3cross(n, x.x, 1, x.y, 1, x.z, 1,                     \
                            y.x, 1, y.y, 1, y.z, 1,                        \
                            ro[0], 1, ro[1], 1, ro[2], 1);                 \
      v3 ## P r = v3cross_ ## P(x, y, wo[0], wo[1], wo[2]);                \
      double e = max_rel_diff(esz, ICX, r.x, ro[0], n);                    \
      e = fmax(e, max_rel_diff(esz, ICX, r.y, ro[1], n));                  \
      e = fmax(e, max_rel_diff(esz, ICX, r.z, ro[2], n));                  \
      Timing tp = timed_run([&]{ v3cross_ ## P(x, y, wo[0], wo[1], wo[2]); }); \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3cross(n,                  \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1,                  \
          ro[0], 1, ro[1], 1, ro[2], 1); });                               \
      add_result("v3cross_" #P, tp, tc, e, (ICX ? 42.0 : 9.0), n, "w=x∧y"); } \
                                                                           \
    { cblas_ ## P ## 3crossscal(n, CCS, x.x, 1, x.y, 1, x.z, 1,            \
                                y.x, 1, y.y, 1, y.z, 1,                    \
                                ro[0], 1, ro[1], 1, ro[2], 1);             \
      v3 ## P r = v3crossscal_ ## P(x, y, s_cs, wo[0], wo[1], wo[2]);      \
      double e = max_rel_diff(esz, ICX, r.x, ro[0], n);                    \
      e = fmax(e, max_rel_diff(esz, ICX, r.y, ro[1], n));                  \
      e = fmax(e, max_rel_diff(esz, ICX, r.z, ro[2], n));                  \
      Timing tp = timed_run([&]{ v3crossscal_ ## P(x, y, s_cs,             \
          wo[0], wo[1], wo[2]); });                                        \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3crossscal(n, CCS,         \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1,                  \
          ro[0], 1, ro[1], 1, ro[2], 1); });                               \
      add_result("v3crossscal_" #P, tp, tc, e, (ICX ? 60.0 : 12.0), n, "w=a·(x∧y)"); } \
                                                                           \
    { cblas_ ## P ## 3crossdot(n, x.x, 1, x.y, 1, x.z, 1,                  \
                               y.x, 1, y.y, 1, y.z, 1,                     \
                               w.x, 1, w.y, 1, w.z, 1, ro[0], 1);          \
      double e = max_rel_diff(esz, ICX, v3crossdot_ ## P(x, y, w, wo[0]),  \
                              ro[0], n);                                   \
      Timing tp = timed_run([&]{ v3crossdot_ ## P(x, y, w, wo[0]); });     \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3crossdot(n,               \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1,                  \
          w.x, 1, w.y, 1, w.z, 1, ro[0], 1); });                           \
      add_result("v3crossdot_" #P, tp, tc, e, (ICX ? 64.0 : 14.0), n, "r=(x∧y)·w"); } \
                                                                           \
    { cblas_ ## P ## 3crosssqr(n, x.x, 1, x.y, 1, x.z, 1,                  \
                               y.x, 1, y.y, 1, y.z, 1, ro[0], 1);          \
      double e = max_rel_diff(esz, ICX, v3crosssqr_ ## P(x, y, wo[0]),     \
                              ro[0], n);                                   \
      Timing tp = timed_run([&]{ v3crosssqr_ ## P(x, y, wo[0]); });        \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3crosssqr(n,               \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1, ro[0], 1); });   \
      add_result("v3crosssqr_" #P, tp, tc, e, (ICX ? 64.0 : 14.0), n, "r=(x∧y)·(x∧y)"); } \
                                                                           \
    { cblas_ ## P ## 3crossxy_crossxz(n, x.x, 1, x.y, 1, x.z, 1,           \
                                      y.x, 1, y.y, 1, y.z, 1,              \
                                      w.x, 1, w.y, 1, w.z, 1,              \
                                      ro[0], 1, ro[1], 1, ro[2], 1,        \
                                      ro[3], 1, ro[4], 1, ro[5], 1);       \
      v3uv ## P r = v3crossxy_crossxz_ ## P(x, y, w,                        \
          wo[0], wo[1], wo[2], wo[3], wo[4], wo[5]);                       \
      double e = max_rel_diff(esz, ICX, r.u.x, ro[0], n);                  \
      e = fmax(e, max_rel_diff(esz, ICX, r.u.y, ro[1], n));                \
      e = fmax(e, max_rel_diff(esz, ICX, r.u.z, ro[2], n));                \
      e = fmax(e, max_rel_diff(esz, ICX, r.v.x, ro[3], n));                \
      e = fmax(e, max_rel_diff(esz, ICX, r.v.y, ro[4], n));                \
      e = fmax(e, max_rel_diff(esz, ICX, r.v.z, ro[5], n));                \
      Timing tp = timed_run([&]{ v3crossxy_crossxz_ ## P(x, y, w,          \
          wo[0], wo[1], wo[2], wo[3], wo[4], wo[5]); });                   \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3crossxy_crossxz(n,        \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1,                  \
          w.x, 1, w.y, 1, w.z, 1,                                          \
          ro[0], 1, ro[1], 1, ro[2], 1, ro[3], 1, ro[4], 1, ro[5], 1); }); \
      add_result("v3crossxy_crossxz_" #P, tp, tc, e,                       \
                 (ICX ? 84.0 : 18.0), n, "u=x∧y,v=x∧w"); }                 \
                                                                           \
    { cblas_ ## P ## 3crossxy_dotxz(n, x.x, 1, x.y, 1, x.z, 1,             \
                                    y.x, 1, y.y, 1, y.z, 1,                \
                                    w.x, 1, w.y, 1, w.z, 1,                \
                                    ro[0], 1, ro[1], 1, ro[2], 1,          \
                                    ro[3], 1);                             \
      v3ur ## P r = v3crossxy_dotxz_ ## P(x, y, w,                          \
          wo[0], wo[1], wo[2], wo[3]);                                     \
      double e = max_rel_diff(esz, ICX, r.u.x, ro[0], n);                  \
      e = fmax(e, max_rel_diff(esz, ICX, r.u.y, ro[1], n));                \
      e = fmax(e, max_rel_diff(esz, ICX, r.u.z, ro[2], n));                \
      e = fmax(e, max_rel_diff(esz, ICX, r.r, ro[3], n));                  \
      Timing tp = timed_run([&]{ v3crossxy_dotxz_ ## P(x, y, w,            \
          wo[0], wo[1], wo[2], wo[3]); });                                 \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3crossxy_dotxz(n,          \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1,                  \
          w.x, 1, w.y, 1, w.z, 1,                                          \
          ro[0], 1, ro[1], 1, ro[2], 1, ro[3], 1); });                     \
      add_result("v3crossxy_dotxz_" #P, tp, tc, e,                         \
                 (ICX ? 64.0 : 14.0), n, "u=x∧y,r=x·w"); }                 \
                                                                           \
    { cblas_ ## P ## 3dotxy_dotxz(n, x.x, 1, x.y, 1, x.z, 1,               \
                                  y.x, 1, y.y, 1, y.z, 1,                  \
                                  w.x, 1, w.y, 1, w.z, 1,                  \
                                  ro[0], 1, ro[1], 1);                     \
      rq ## P r = v3dotxy_dotxz_ ## P(x, y, w, wo[0], wo[1]);              \
      double e = max_rel_diff(esz, ICX, r.r, ro[0], n);                    \
      e = fmax(e, max_rel_diff(esz, ICX, r.q, ro[1], n));                  \
      Timing tp = timed_run([&]{ v3dotxy_dotxz_ ## P(x, y, w,              \
          wo[0], wo[1]); });                                               \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3dotxy_dotxz(n,            \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1,                  \
          w.x, 1, w.y, 1, w.z, 1, ro[0], 1, ro[1], 1); });                 \
      add_result("v3dotxy_dotxz_" #P, tp, tc, e,                           \
                 (ICX ? 44.0 : 10.0), n, "r=x·y,q=x·w"); }                 \
                                                                           \
    free(in);                                                              \
    free(scratch);                                                         \
}

BENCH_PREC(s, float, 0, s_ax = 1.5f; s_xa = 0.25f; s_cs = 1.5f,
           s_ax, s_xa, s_cs)
BENCH_PREC(d, double, 0, s_ax = 1.5; s_xa = 0.25; s_cs = 1.5,
           s_ax, s_xa, s_cs)
BENCH_PREC(c, v3blas_floatcomplex, 1,
           s_ax.re = 1.5f; s_ax.im = 0.5f; s_xa.re = 0.25f; s_xa.im = 0.1f; s_cs.re = 1.5f; s_cs.im = 0.5f,
           pax, pxa, pcs)
BENCH_PREC(z, v3blas_doublecomplex, 1,
           s_ax.re = 1.5; s_ax.im = 0.5; s_xa.re = 0.25; s_xa.im = 0.1; s_cs.re = 1.5; s_cs.im = 0.5,
           pax, pxa, pcs)

#undef BENCH_PREC

int main(int argc, char **argv)
{
    int n = 1000;
    char prec = 'd';
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.size() == 1 && strchr("sdcz", a[0]))
            prec = a[0];
        else {
            long v = std::strtol(a.c_str(), nullptr, 10);
            if (v > 0)
                n = (int)v;
        }
    }

    std::printf("n = %d  prec = %c\n", n, prec);
    std::printf("v3blas = wrapper call on caller-allocated arrays; cblas = raw call\n");
    std::printf("max_rel_err compares the v3blas and cblas outputs element-wise\n");

    if (prec == 's') bench_all_s(n);
    else if (prec == 'c') bench_all_c(n);
    else if (prec == 'z') bench_all_z(n);
    else bench_all_d(n);

    std::printf("%-19s %13s %13s %8s %12s %13s %7s  %s  %s\n",
                "kernel", "v3blas ms/run", "cblas ms/run", "overhead",
                "FLOP/s", "max_rel_err", "iters", "status", "equation");
    for (const Result &res : results)
        std::printf("%-19s %13.3f %13.3f %8.2f %12.3e %13.3e %7ld  %s  %s\n",
                    res.name.c_str(), res.ms_v3blas, res.ms_cblas,
                    res.overhead, res.flops_rate, res.max_rel_err, res.iters,
                    res.ok ? "OK" : "FAIL", res.eq.c_str());
    return 0;
}
