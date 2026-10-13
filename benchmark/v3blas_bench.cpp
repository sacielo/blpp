/*
 * v3blas_bench.cpp - timing for the v3blas.h interface, side by side with
 * the raw cblas_* extension calls it dispatches to.  The overhead column
 * is the point of this file: what a v3blas_*(...) call costs relative to
 * the cblas call it wraps (view structs by value, no bookkeeping).
 *
 * Timing only.  Correctness of every one of these calls, bitwise against
 * cblas and against a long-double reference, is v3blas_test's job - see
 * that file.  Nothing here validates results.
 *
 * Neither side allocates inside a timed region (v3blas never allocates
 * at all); all buffers come from two mallocs up front.
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

#ifndef K_MIN_TIME
#define K_MIN_TIME 0.5      /* seconds per kernel; -DK_MIN_TIME=... to slim */
#endif
#ifndef K_MIN_ITERS
#define K_MIN_ITERS 3
#endif
#ifndef K_MAX_TIME
#define K_MAX_TIME 10.0
#endif

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

struct Timing {
    double t_run;   /* seconds spent inside fn only */
    long iters;
};

/* run fn until kMinTime; returns pure fn time (restore excluded) */
static Timing timed_run(const std::function<void()> &fn)
{
    static const double kMinTime = K_MIN_TIME;   /* seconds */
    static const long kMinIters = K_MIN_ITERS;
    static const double kMaxTime = K_MAX_TIME;
    Timing tm = { 0.0, 0 };
    double t0 = now_sec();
    while (true) {
        fn();
        tm.iters++;
        double el = now_sec() - t0;
        if ((el >= kMinTime && tm.iters >= kMinIters) || el >= kMaxTime) {
            tm.t_run = el;
            return tm;
        }
    }
}

/* for in-place kernels (axpy): put dst back, then time fn alone */
static Timing timed_run_restore(const std::function<void()> &restore,
                                const std::function<void()> &fn)
{
    static const double kMinTime = K_MIN_TIME;
    static const long kMinIters = K_MIN_ITERS;
    static const double kMaxTime = K_MAX_TIME;
    Timing tm = { 0.0, 0 };
    restore();
    fn();
    double t0 = now_sec();
    while (true) {
        restore();
        double b = now_sec();
        fn();
        tm.t_run += now_sec() - b;
        tm.iters++;
        double el = now_sec() - t0;
        if ((el >= kMinTime && tm.iters >= kMinIters) || el >= kMaxTime)
            return tm;
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

struct Result {
    std::string name;
    double ms_v3blas;    /* per-call ms, v3blas.h interface */
    double ms_cblas;     /* per-call ms, raw cblas call */
    double overhead;     /* v3blas/cblas time ratio, ~1.00 expected */
    double flops_rate;   /* math FLOP/s delivered through v3blas */
    long iters;
    std::string eq;      /* FLOP-count legend, see cblas_bench for formulas */
};

static std::vector<Result> results;

/* deterministic pseudo-random fills in [-0.5, 0.5); a 3-vector's
 * components get seeds S, S+1, S+2; complex im = 0.7*re */
static void fill1_s(float *v, int n, int S) { for (int i = 0; i < n; i++) v[i] = (float)VAL(S, i); }
static void fill1_d(double *v, int n, int S) { for (int i = 0; i < n; i++) v[i] = (double)VAL(S, i); }
static void fill1_c(v3blas_floatcomplex *v, int n, int S) { for (int i = 0; i < n; i++) { v[i].re = (float)VAL(S, i); v[i].im = (float)(VAL(S, i) * 0.7L); } }
static void fill1_z(v3blas_doublecomplex *v, int n, int S) { for (int i = 0; i < n; i++) { v[i].re = VAL(S, i); v[i].im = VAL(S, i) * 0.7L; } }

static void fill3_s(float *a0, float *a1, float *a2, int n, int S) { for (int i = 0; i < n; i++) { a0[i] = (float)VAL(S, i); a1[i] = (float)VAL(S + 1, i); a2[i] = (float)VAL(S + 2, i); } }
static void fill3_d(double *a0, double *a1, double *a2, int n, int S) { for (int i = 0; i < n; i++) { a0[i] = (double)VAL(S, i); a1[i] = (double)VAL(S + 1, i); a2[i] = (double)VAL(S + 2, i); } }
static void fill3_c(v3blas_floatcomplex *a0, v3blas_floatcomplex *a1, v3blas_floatcomplex *a2, int n, int S) { for (int i = 0; i < n; i++) { a0[i].re = (float)VAL(S, i); a0[i].im = (float)(VAL(S, i) * 0.7L); a1[i].re = (float)VAL(S + 1, i); a1[i].im = (float)(VAL(S + 1, i) * 0.7L); a2[i].re = (float)VAL(S + 2, i); a2[i].im = (float)(VAL(S + 2, i) * 0.7L); } }
static void fill3_z(v3blas_doublecomplex *a0, v3blas_doublecomplex *a1, v3blas_doublecomplex *a2, int n, int S) { for (int i = 0; i < n; i++) { a0[i].re = VAL(S, i); a0[i].im = VAL(S, i) * 0.7L; a1[i].re = VAL(S + 1, i); a1[i].im = VAL(S + 1, i) * 0.7L; a2[i].re = VAL(S + 2, i); a2[i].im = VAL(S + 2, i) * 0.7L; } }

static void add_result(const char *name, const Timing &tp, const Timing &tc,
                       double fpse, int n, const char *eq)
{
    double per_p = tp.t_run / tp.iters;
    double per_c = tc.t_run / tc.iters;
    results.push_back({name, per_p * 1e3, per_c * 1e3, per_p / per_c,
                       fpse * n / per_p, tp.iters, eq});
}

/*
 * One body, four precisions. ICX: 0 = real, 1 = complex element.
 * INIT: statements initializing the three scalars (comma-free).
 * CAX/CXA/CCS pass those scalars the way each cblas prototype wants them
 * (by value for real, as void* for complex); v3blas calls use the locals.
 * Inputs: one malloc'd slab sliced into 13 unit-stride arrays.
 * wo[] are the v3blas output buffers, ro[] the cblas ones.
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
    v3 ## P ow = v3 ## P ## _wrap(wo[0], wo[1], wo[2], n);                 \
    v3 ## P ou = v3 ## P ## _wrap(wo[1], wo[2], wo[3], n);                 \
    v3 ## P ov = v3 ## P ## _wrap(wo[3], wo[4], wo[5], n);                 \
                                                                           \
    { Timing tp = timed_run([&]{ v1axpy_ ## P(s_ax, bxv, byv, wo[0], n); }); \
      Timing tc = timed_run_restore([&]{ memcpy(ro[0], byv, (size_t)n * esz); }, \
          [&]{ cblas_ ## P ## axpy(n, CAX, bxv, 1, ro[0], 1); });          \
      add_result("v1axpy_" #P, tp, tc, (ICX ? 8.0 : 2.0), n, "y=a*x+y"); } \
                                                                           \
    { v1xypa_ ## P(bq, bt, s_xa, wo[0], n);                                \
      Timing tp = timed_run([&]{ v1xypa_ ## P(bq, bt, s_xa, wo[0], n); }); \
      Timing tc = timed_run([&]{ cblas_ ## P ## 1xypa(n, bq, 1,           \
          bt, 1, CXA, ro[0], 1); });                                     \
      add_result("v1xypa_" #P, tp, tc, (ICX ? 8.0 : 2.0), n, "r=q.*t+a"); } \
                                                                           \
    { v3dot_ ## P(x, y, wo[0]);                                            \
      Timing tp = timed_run([&]{ v3dot_ ## P(x, y, wo[0]); });             \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3dot(n,                    \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1, ro[0], 1); });   \
      add_result("v3dot_" #P, tp, tc, (ICX ? 22.0 : 5.0), n, "r=x1y1+x2y2+x3y3"); } \
                                                                           \
    { v3had_ ## P(x, y, ow);                              \
      Timing tp = timed_run([&]{ v3had_ ## P(x, y, ow); }); \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3had(n,                    \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1,                  \
          ro[0], 1, ro[1], 1, ro[2], 1); });                               \
      add_result("v3had_" #P, tp, tc, (ICX ? 18.0 : 3.0), n, "w=x.*y"); }  \
                                                                           \
    { v3sqr_ ## P(x, wo[0]);                                               \
      Timing tp = timed_run([&]{ v3sqr_ ## P(x, wo[0]); });                \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3sqr(n,                    \
          x.x, 1, x.y, 1, x.z, 1, ro[0], 1); });                           \
      add_result("v3sqr_" #P, tp, tc, (ICX ? 22.0 : 5.0), n, "r=x1^2+x2^2+x3^2"); } \
                                                                           \
    { v1norm_ ## P(bq, wo[0], n);                                          \
      Timing tp = timed_run([&]{ v1norm_ ## P(bq, wo[0], n); });           \
      Timing tc = timed_run([&]{ cblas_ ## P ## 1norm(n, bq, 1,            \
          ro[0], 1); });                                                   \
      add_result("v1norm_" #P, tp, tc, (ICX ? 16.0 : 2.0), n, "r=sqrt(q\xc2\xb7q)"); } \
                                                                           \
    { v3cross_ ## P(x, y, ow);                            \
      Timing tp = timed_run([&]{ v3cross_ ## P(x, y, ow); }); \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3cross(n,                  \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1,                  \
          ro[0], 1, ro[1], 1, ro[2], 1); });                               \
      add_result("v3cross_" #P, tp, tc, (ICX ? 42.0 : 9.0), n, "w=x\xe2\x88\xa7y"); } \
                                                                           \
    { v3crossscal_ ## P(x, y, s_cs, ow);                  \
      Timing tp = timed_run([&]{ v3crossscal_ ## P(x, y, s_cs, ow); });                                        \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3crossscal(n, CCS,         \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1,                  \
          ro[0], 1, ro[1], 1, ro[2], 1); });                               \
      add_result("v3crossscal_" #P, tp, tc, (ICX ? 60.0 : 12.0), n,        \
                 "w=a\xc2\xb7(x\xe2\x88\xa7y)"); }                         \
                                                                           \
    { v3crossdot_ ## P(x, y, w, wo[0]);                                    \
      Timing tp = timed_run([&]{ v3crossdot_ ## P(x, y, w, wo[0]); });     \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3crossdot(n,               \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1,                  \
          w.x, 1, w.y, 1, w.z, 1, ro[0], 1); });                           \
      add_result("v3crossdot_" #P, tp, tc, (ICX ? 64.0 : 14.0), n,         \
                 "r=(x\xe2\x88\xa7y)\xc2\xb7w"); }                         \
                                                                           \
    { v3crosssqr_ ## P(x, y, wo[0]);                                       \
      Timing tp = timed_run([&]{ v3crosssqr_ ## P(x, y, wo[0]); });        \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3crosssqr(n,               \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1, ro[0], 1); });   \
      add_result("v3crosssqr_" #P, tp, tc, (ICX ? 64.0 : 14.0), n,         \
                 "r=(x\xe2\x88\xa7y)\xc2\xb7(x\xe2\x88\xa7y)"); }          \
                                                                           \
    { v3crossxy_crossxz_ ## P(x, y, w, ow, ov);                       \
      Timing tp = timed_run([&]{ v3crossxy_crossxz_ ## P(x, y, w, ow, ov); });                   \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3crossxy_crossxz(n,        \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1,                  \
          w.x, 1, w.y, 1, w.z, 1, ro[0], 1, ro[1], 1, ro[2], 1,            \
          ro[3], 1, ro[4], 1, ro[5], 1); });                               \
      add_result("v3crossxy_crossxz_" #P, tp, tc, (ICX ? 108.0 : 18.0), n, \
                 "u=x\xe2\x88\xa7y,v=x\xe2\x88\xa7z"); }                   \
                                                                           \
    { v3crossxy_dotxz_ ## P(x, y, w, ow, wo[3]);          \
      Timing tp = timed_run([&]{ v3crossxy_dotxz_ ## P(x, y, w, ow, wo[3]); });                                 \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3crossxy_dotxz(n,          \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1,                  \
          w.x, 1, w.y, 1, w.z, 1, ro[0], 1, ro[1], 1, ro[2], 1,            \
          ro[3], 1); });                                                   \
      add_result("v3crossxy_dotxz_" #P, tp, tc, (ICX ? 86.0 : 17.0), n,    \
                 "u=x\xe2\x88\xa7y,r=x\xe2\x88\xa7z"); }                   \
                                                                           \
    { v3dotxy_dotxz_ ## P(x, y, w, wo[0], wo[1]);                          \
      Timing tp = timed_run([&]{ v3dotxy_dotxz_ ## P(x, y, w,              \
          wo[0], wo[1]); });                                               \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3dotxy_dotxz(n,            \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1,                  \
          w.x, 1, w.y, 1, w.z, 1, ro[0], 1, ro[1], 1); });                 \
      add_result("v3dotxy_dotxz_" #P, tp, tc, (ICX ? 64.0 : 14.0), n,      \
                 "r=x\xc2\xb7y,q=x\xc2\xb7z"); }                           \
                                                                           \
    { v3crosscross_ ## P(x, y, w, ow);                    \
      Timing tp = timed_run([&]{ v3crosscross_ ## P(x, y, w, ow); });                                        \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3crosscross(n,             \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1,                  \
          w.x, 1, w.y, 1, w.z, 1, ro[0], 1, ro[1], 1, ro[2], 1); });       \
      add_result("v3crosscross_" #P, tp, tc, (ICX ? 84.0 : 18.0), n,       \
                 "r=(x\xe2\x88\xa7y)\xe2\x88\xa7w"); }                     \
                                                                           \
    { v3norm_unit_ ## P(x, 0.25, wo[0], ou);              \
      Timing tp = timed_run([&]{ v3norm_unit_ ## P(x, 0.25, wo[0], ou); });                                 \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3norm_unit(n,              \
          x.x, 1, x.y, 1, x.z, 1, 0.25, ro[0], 1,                          \
          ro[1], 1, ro[2], 1, ro[3], 1); });                               \
      add_result("v3norm_unit_" #P, tp, tc, (ICX ? 76.0 : 12.0), n,        \
                 "u=x/sqrt(x\xc2\xb7x+eps)"); }                            \
                                                                           \
    { v3refl_ ## P(x, y, 1.5, 0.25, ow);                  \
      Timing tp = timed_run([&]{ v3refl_ ## P(x, y, 1.5, 0.25, ow); });                                        \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3refl(n,                   \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1, 1.5, 0.25,       \
          ro[0], 1, ro[1], 1, ro[2], 1); });                               \
      add_result("v3refl_" #P, tp, tc, (ICX ? 108.0 : 18.0), n,            \
                 "q=x-a(x\xc2\xb7y)/(y\xc2\xb7y+eps)y"); }                 \
                                                                           \
    { v3exb_ ## P(x, y, 1.5, 0.25, ow);                   \
      Timing tp = timed_run([&]{ v3exb_ ## P(x, y, 1.5, 0.25, ow); });                                        \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3exb(n,                    \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1, 1.5, 0.25,       \
          ro[0], 1, ro[1], 1, ro[2], 1); });                               \
      add_result("v3exb_" #P, tp, tc, (ICX ? 52.0 : 20.0), n,              \
                 "r=a(x\xe2\x88\xa7y)/(y\xc2\xb7y+eps)"); }                \
                                                                           \
    { v3drag_ ## P(x, 1.5, 0.25, ow);                     \
      Timing tp = timed_run([&]{ v3drag_ ## P(x, 1.5, 0.25, ow); });                                        \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3drag(n,                   \
          x.x, 1, x.y, 1, x.z, 1, 1.5, 0.25,                               \
          ro[0], 1, ro[1], 1, ro[2], 1); });                               \
      add_result("v3drag_" #P, tp, tc, (ICX ? 48.0 : 27.0), n,             \
                 "r=a*x*sqrt(x\xc2\xb7x+eps)"); }                          \
                                                                           \
    { v3mom_ke_ ## P(bq, x, ow, wo[3]);                   \
      Timing tp = timed_run([&]{ v3mom_ke_ ## P(bq, x, ow, wo[3]); });                                 \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3mom_ke(n, bq, 1,          \
          x.x, 1, x.y, 1, x.z, 1,                                          \
          ro[0], 1, ro[1], 1, ro[2], 1, ro[3], 1); });                     \
      add_result("v3mom_ke_" #P, tp, tc, (ICX ? 28.0 : 10.0), n,           \
                 "m=s*x,k=0.5*s*(x\xc2\xb7x)"); }                          \
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
    std::printf("correctness lives in v3blas_test; this table is timing only\n");

    if (prec == 's') bench_all_s(n);
    else if (prec == 'c') bench_all_c(n);
    else if (prec == 'z') bench_all_z(n);
    else bench_all_d(n);

    std::printf("%-19s %13s %13s %8s %12s %7s  %s\n",
                "kernel", "v3blas ms/run", "cblas ms/run", "overhead",
                "FLOP/s", "iters", "equation");
    for (const Result &res : results)
        std::printf("%-19s %13.3f %13.3f %8.2f %12.3e %7ld  %s\n",
                    res.name.c_str(), res.ms_v3blas, res.ms_cblas,
                    res.overhead, res.flops_rate, res.iters, res.eq.c_str());
    return 0;
}
