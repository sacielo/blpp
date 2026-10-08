// v3blas_bench.cpp - benchmark of the v3blas.h higher-level interface.
//
// usage: v3blas_bench [n] [s|d|c|z]
//   n    vector size (default 1000)
//   sdcz precision (default d); order of the two arguments is free
//
// For every operation the table compares two timed loops:
//   v3blas:  the v3blas call (which heap-allocates its result) + the
//           matching _free, i.e. the cost a v3blas user actually pays;
//   cblas:  the raw cblas extension call with pre-allocated output
//           buffers, i.e. the kernel alone.
// The overhead column is v3blas/cblas.
//
// Correctness here means v3blas == cblas on identical inputs: both are
// run once before timing and their outputs compared element-wise
// (max_rel_err; both paths run the same kernel, so 0 is expected).
// A high-precision reference is v3blas_test.c's job, not this file's.
// The last column is the kernel equation, a single space-free token so
// whole lines can be parsed with awk (same convention as benchmark.cpp).
//
// Links against an installed OpenBLAS (see CMakeLists.txt); this
// project does not build OpenBLAS.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "v3blas.h"

using Clock = std::chrono::steady_clock;
using ld = long double;

namespace {

constexpr double kMinTime = 0.5;   // seconds of runtime per kernel
constexpr int kMinIters = 3;
constexpr double kMaxTime = 10.0;  // cap for slow machines

// bounded incoherent value of (seed, element) in [-0.5, 0.5): the
// murmur3 finalizer avalanches fully, so neither elements nor component
// seeds are correlated; deliberately not a ramp (see benchmark.cpp's
// seeded() for why)
inline unsigned hash2(int a, int b)
{
    unsigned h = (unsigned)a * 0x9E3779B1u + (unsigned)b * 0x85EBCA6Bu;
    h ^= h >> 16; h *= 0x7FEB352Du;
    h ^= h >> 15; h *= 0x846CA68Bu;
    h ^= h >> 16;
    return h;
}

inline ld VAL(int S, int i)
{
    return (ld)hash2(S, i) / 4294967296.0L - 0.5L;
}

struct Result {
    std::string name;
    double ms_v3blas;
    double ms_cblas;
    double overhead;   // ms_v3blas / ms_cblas
    double flops_rate; // FLOP/s from v3blas timing
    double max_rel_err; // v3blas result vs. cblas result
    int iters;         // v3blas timing loop
    bool ok;
    std::string eq;    // space-free equation token, printed last
};

std::vector<Result> results;

// warmup, then time repeated runs; returns total run time and iteration count
struct Timing { double t_run; int iters; };

Timing timed_run(const std::function<void()> &call)
{
    call(); // warmup

    double t_run = 0.0;
    int iters = 0;
    auto t_start = Clock::now();
    for (;;) {
        auto b = Clock::now();
        call();
        t_run += std::chrono::duration<double>(Clock::now() - b).count();
        ++iters;
        double elapsed = std::chrono::duration<double>(Clock::now() - t_start).count();
        if ((iters >= kMinIters && elapsed >= kMinTime) || elapsed >= kMaxTime)
            break;
    }
    return {t_run, iters};
}

// read one output element of any precision as (re, im)
static void xread(size_t esz, int iscx, const unsigned char *p, ld *re, ld *im)
{
    if (esz == 4) {
        float f; memcpy(&f, p, 4); *re = f; *im = 0;
    } else if (esz == 8 && !iscx) {
        double d; memcpy(&d, p, 8); *re = d; *im = 0;
    } else if (esz == 8) {
        float a, b; memcpy(&a, p, 4); memcpy(&b, p + 4, 4); *re = a; *im = b;
    } else {
        double a, b; memcpy(&a, p, 8); memcpy(&b, p + 8, 8); *re = a; *im = b;
    }
}

static ld rabs(ld a) { return a < 0 ? -a : a; }
static ld m1(ld a) { a = rabs(a); return a < 1 ? 1 : a; }

// max element-wise relative difference between two output arrays
static double max_rel_diff(size_t esz, int iscx, const void *g,
                           const void *r, int n)
{
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        ld g0, g1, r0, r1;
        xread(esz, iscx, (const unsigned char *)g + (size_t)i * esz, &g0, &g1);
        xread(esz, iscx, (const unsigned char *)r + (size_t)i * esz, &r0, &r1);
        ld e = rabs(g0 - r0) / m1(r0);
        ld d = rabs(g1 - r1) / m1(r1);
        if (d > e) e = d;
        if ((double)e > m) m = (double)e;
    }
    return m;
}

// data fillers: components of a 3-vector get seeds S, S+1, S+2
static void fill1_s(v1s v, int S) { for (int i = 0; i < v.n; i++) v.x[i] = (float)VAL(S, i); }
static void fill1_d(v1d v, int S) { for (int i = 0; i < v.n; i++) v.x[i] = (double)VAL(S, i); }
static void fill1_c(v1c v, int S) { for (int i = 0; i < v.n; i++) { v.x[i].re = (float)VAL(S, i); v.x[i].im = (float)(VAL(S, i) * 0.7L); } }
static void fill1_z(v1z v, int S) { for (int i = 0; i < v.n; i++) { v.x[i].re = VAL(S, i); v.x[i].im = VAL(S, i) * 0.7L; } }

static void fill3_s(v3s v, int S) { for (int i = 0; i < v.n; i++) { v.x[i] = (float)VAL(S, i); v.y[i] = (float)VAL(S + 1, i); v.z[i] = (float)VAL(S + 2, i); } }
static void fill3_d(v3d v, int S) { for (int i = 0; i < v.n; i++) { v.x[i] = (double)VAL(S, i); v.y[i] = (double)VAL(S + 1, i); v.z[i] = (double)VAL(S + 2, i); } }
static void fill3_c(v3c v, int S) { for (int i = 0; i < v.n; i++) { v.x[i].re = (float)VAL(S, i); v.x[i].im = (float)(VAL(S, i) * 0.7L); v.y[i].re = (float)VAL(S + 1, i); v.y[i].im = (float)(VAL(S + 1, i) * 0.7L); v.z[i].re = (float)VAL(S + 2, i); v.z[i].im = (float)(VAL(S + 2, i) * 0.7L); } }
static void fill3_z(v3z v, int S) { for (int i = 0; i < v.n; i++) { v.x[i].re = VAL(S, i); v.x[i].im = VAL(S, i) * 0.7L; v.y[i].re = VAL(S + 1, i); v.y[i].im = VAL(S + 1, i) * 0.7L; v.z[i].re = VAL(S + 2, i); v.z[i].im = VAL(S + 2, i) * 0.7L; } }

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
 * the local s_ax/s_xa/s_cs directly. o0..o5 are the pre-allocated
 * cblas output buffers, also used as the comparison reference.
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
                                                                           \
    v3 ## P x = v3 ## P ## _new(n), y = v3 ## P ## _new(n), w = v3 ## P ## _new(n); \
    v1 ## P q = v1 ## P ## _new(n), t = v1 ## P ## _new(n),                \
              xv = v1 ## P ## _new(n), yv = v1 ## P ## _new(n);            \
    fill3_ ## P(x, 1); fill3_ ## P(y, 4); fill3_ ## P(w, 7);               \
    fill1_ ## P(q, 10); fill1_ ## P(t, 11); fill1_ ## P(xv, 13);           \
    fill1_ ## P(yv, 12);                                                   \
                                                                           \
    std::vector<T> scratch(6 * n);                                         \
    T *o0 = scratch.data(), *o1 = o0 + n, *o2 = o1 + n,                    \
      *o3 = o2 + n, *o4 = o3 + n, *o5 = o4 + n;                            \
    const size_t esz = sizeof(T);                                          \
                                                                           \
    { memcpy(o0, yv.x, (size_t)n * sizeof(T));                             \
      cblas_ ## P ## axpy(n, CAX, xv.x, 1, o0, 1);                         \
      v1 ## P r = v1axpy_ ## P(xv, yv, s_ax);                              \
      double e = max_rel_diff(esz, ICX, r.x, o0, n);                       \
      v1 ## P ## _free(&r);                                                \
      Timing tp = timed_run([&]{ v1 ## P a = v1axpy_ ## P(xv, yv, s_ax);   \
                                 v1 ## P ## _free(&a); });                 \
      Timing tc = timed_run([&]{ memcpy(o0, yv.x, (size_t)n * sizeof(T));  \
                                 cblas_ ## P ## axpy(n, CAX, xv.x, 1, o0, 1); }); \
      add_result("v1axpy_" #P, tp, tc, e, (ICX ? 8.0 : 2.0), n, "y=a*x+y"); }                 \
                                                                           \
    { cblas_ ## P ## 1xypa(n, CXA, q.x, 1, t.x, 1, o0, 1);                 \
      v1 ## P r = v1xypa_ ## P(q, t, s_xa);                                \
      double e = max_rel_diff(esz, ICX, r.x, o0, n);                       \
      v1 ## P ## _free(&r);                                                \
      Timing tp = timed_run([&]{ v1 ## P a = v1xypa_ ## P(q, t, s_xa);     \
                                 v1 ## P ## _free(&a); });                 \
      Timing tc = timed_run([&]{ cblas_ ## P ## 1xypa(n, CXA, q.x, 1,      \
                                                      t.x, 1, o0, 1); });  \
      add_result("v1xypa_" #P, tp, tc, e, (ICX ? 8.0 : 2.0), n, "r=q.*t+a"); }                \
                                                                           \
    { cblas_ ## P ## 3dot(n, x.x, 1, x.y, 1, x.z, 1,                       \
                          y.x, 1, y.y, 1, y.z, 1, o0, 1);                  \
      v1 ## P r = v3dot_ ## P(x, y);                                       \
      double e = max_rel_diff(esz, ICX, r.x, o0, n);                       \
      v1 ## P ## _free(&r);                                                \
      Timing tp = timed_run([&]{ v1 ## P a = v3dot_ ## P(x, y);            \
                                 v1 ## P ## _free(&a); });                 \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3dot(n,                    \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1, o0, 1); });      \
      add_result("v3dot_" #P, tp, tc, e, (ICX ? 22.0 : 5.0), n, "r=x1y1+x2y2+x3y3"); }         \
                                                                           \
    { cblas_ ## P ## 3had(n, x.x, 1, x.y, 1, x.z, 1,                       \
                          y.x, 1, y.y, 1, y.z, 1,                          \
                          o0, 1, o1, 1, o2, 1);                            \
      v3 ## P r = v3had_ ## P(x, y);                                       \
      double e = max_rel_diff(esz, ICX, r.x, o0, n);                       \
      e = fmax(e, max_rel_diff(esz, ICX, r.y, o1, n));                     \
      e = fmax(e, max_rel_diff(esz, ICX, r.z, o2, n));                     \
      v3 ## P ## _free(&r);                                                \
      Timing tp = timed_run([&]{ v3 ## P a = v3had_ ## P(x, y);            \
                                 v3 ## P ## _free(&a); });                 \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3had(n,                    \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1,                  \
          o0, 1, o1, 1, o2, 1); });                                        \
      add_result("v3had_" #P, tp, tc, e, (ICX ? 18.0 : 3.0), n, "w=x.*y"); }                   \
                                                                           \
    { cblas_ ## P ## 3sqr(n, x.x, 1, x.y, 1, x.z, 1, o0, 1);               \
      v1 ## P r = v3sqr_ ## P(x);                                          \
      double e = max_rel_diff(esz, ICX, r.x, o0, n);                       \
      v1 ## P ## _free(&r);                                                \
      Timing tp = timed_run([&]{ v1 ## P a = v3sqr_ ## P(x);               \
                                 v1 ## P ## _free(&a); });                 \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3sqr(n,                    \
          x.x, 1, x.y, 1, x.z, 1, o0, 1); });                              \
      add_result("v3sqr_" #P, tp, tc, e, (ICX ? 22.0 : 5.0), n, "r=x1^2+x2^2+x3^2"); }         \
                                                                           \
    { cblas_ ## P ## 1norm(n, q.x, 1, o0, 1);                              \
      v1 ## P r = v1norm_ ## P(q);                                         \
      double e = max_rel_diff(esz, ICX, r.x, o0, n);                       \
      v1 ## P ## _free(&r);                                                \
      Timing tp = timed_run([&]{ v1 ## P a = v1norm_ ## P(q);              \
                                 v1 ## P ## _free(&a); });                 \
      Timing tc = timed_run([&]{ cblas_ ## P ## 1norm(n, q.x, 1, o0, 1); }); \
      add_result("v1norm_" #P, tp, tc, e, (ICX ? 16.0 : 2.0), n, "r=sqrt(q·q)"); }             \
                                                                           \
    { cblas_ ## P ## 3cross(n, x.x, 1, x.y, 1, x.z, 1,                     \
                            y.x, 1, y.y, 1, y.z, 1,                        \
                            o0, 1, o1, 1, o2, 1);                          \
      v3 ## P r = v3cross_ ## P(x, y);                                     \
      double e = max_rel_diff(esz, ICX, r.x, o0, n);                       \
      e = fmax(e, max_rel_diff(esz, ICX, r.y, o1, n));                     \
      e = fmax(e, max_rel_diff(esz, ICX, r.z, o2, n));                     \
      v3 ## P ## _free(&r);                                                \
      Timing tp = timed_run([&]{ v3 ## P a = v3cross_ ## P(x, y);          \
                                 v3 ## P ## _free(&a); });                 \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3cross(n,                  \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1,                  \
          o0, 1, o1, 1, o2, 1); });                                        \
      add_result("v3cross_" #P, tp, tc, e, (ICX ? 42.0 : 9.0), n, "w=x∧y"); }                  \
                                                                           \
    { cblas_ ## P ## 3crossscal(n, CCS, x.x, 1, x.y, 1, x.z, 1,            \
                                y.x, 1, y.y, 1, y.z, 1,                    \
                                o0, 1, o1, 1, o2, 1);                      \
      v3 ## P r = v3crossscal_ ## P(x, y, s_cs);                           \
      double e = max_rel_diff(esz, ICX, r.x, o0, n);                       \
      e = fmax(e, max_rel_diff(esz, ICX, r.y, o1, n));                     \
      e = fmax(e, max_rel_diff(esz, ICX, r.z, o2, n));                     \
      v3 ## P ## _free(&r);                                                \
      Timing tp = timed_run([&]{ v3 ## P a = v3crossscal_ ## P(x, y, s_cs);\
                                 v3 ## P ## _free(&a); });                 \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3crossscal(n, CCS,         \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1,                  \
          o0, 1, o1, 1, o2, 1); });                                        \
      add_result("v3crossscal_" #P, tp, tc, e, (ICX ? 60.0 : 12.0), n, "w=a·(x∧y)"); }          \
                                                                           \
    { cblas_ ## P ## 3crossdot(n, x.x, 1, x.y, 1, x.z, 1,                  \
                               y.x, 1, y.y, 1, y.z, 1,                     \
                               w.x, 1, w.y, 1, w.z, 1, o0, 1);             \
      v1 ## P r = v3crossdot_ ## P(x, y, w);                               \
      double e = max_rel_diff(esz, ICX, r.x, o0, n);                       \
      v1 ## P ## _free(&r);                                                \
      Timing tp = timed_run([&]{ v1 ## P a = v3crossdot_ ## P(x, y, w);    \
                                 v1 ## P ## _free(&a); });                 \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3crossdot(n,               \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1,                  \
          w.x, 1, w.y, 1, w.z, 1, o0, 1); });                              \
      add_result("v3crossdot_" #P, tp, tc, e, (ICX ? 64.0 : 14.0), n, "r=(x∧y)·w"); }           \
                                                                           \
    { cblas_ ## P ## 3crosssqr(n, x.x, 1, x.y, 1, x.z, 1,                  \
                               y.x, 1, y.y, 1, y.z, 1, o0, 1);             \
      v1 ## P r = v3crosssqr_ ## P(x, y);                                  \
      double e = max_rel_diff(esz, ICX, r.x, o0, n);                       \
      v1 ## P ## _free(&r);                                                \
      Timing tp = timed_run([&]{ v1 ## P a = v3crosssqr_ ## P(x, y);       \
                                 v1 ## P ## _free(&a); });                 \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3crosssqr(n,               \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1, o0, 1); });      \
      add_result("v3crosssqr_" #P, tp, tc, e, (ICX ? 64.0 : 14.0), n, "r=(x∧y)·(x∧y)"); }       \
                                                                           \
    { cblas_ ## P ## 3crossxy_crossxz(n, x.x, 1, x.y, 1, x.z, 1,           \
                                      y.x, 1, y.y, 1, y.z, 1,              \
                                      w.x, 1, w.y, 1, w.z, 1,              \
                                      o0, 1, o1, 1, o2, 1,                 \
                                      o3, 1, o4, 1, o5, 1);                \
      v3uv ## P r = v3crossxy_crossxz_ ## P(x, y, w);                      \
      double e = max_rel_diff(esz, ICX, r.u.x, o0, n);                     \
      e = fmax(e, max_rel_diff(esz, ICX, r.u.y, o1, n));                   \
      e = fmax(e, max_rel_diff(esz, ICX, r.u.z, o2, n));                   \
      e = fmax(e, max_rel_diff(esz, ICX, r.v.x, o3, n));                   \
      e = fmax(e, max_rel_diff(esz, ICX, r.v.y, o4, n));                   \
      e = fmax(e, max_rel_diff(esz, ICX, r.v.z, o5, n));                   \
      v3uv ## P ## _free(r);                                               \
      Timing tp = timed_run([&]{ v3uv ## P a = v3crossxy_crossxz_ ## P(x, y, w); \
                                 v3uv ## P ## _free(a); });                \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3crossxy_crossxz(n,        \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1,                  \
          w.x, 1, w.y, 1, w.z, 1,                                          \
          o0, 1, o1, 1, o2, 1, o3, 1, o4, 1, o5, 1); });                   \
      add_result("v3crossxy_crossxz_" #P, tp, tc, e, (ICX ? 84.0 : 18.0), n, "u=x∧y,v=x∧w"); }  \
                                                                           \
    { cblas_ ## P ## 3crossxy_dotxz(n, x.x, 1, x.y, 1, x.z, 1,             \
                                    y.x, 1, y.y, 1, y.z, 1,                \
                                    w.x, 1, w.y, 1, w.z, 1,                \
                                    o0, 1, o1, 1, o2, 1, o3, 1);           \
      v3ur ## P r = v3crossxy_dotxz_ ## P(x, y, w);                        \
      double e = max_rel_diff(esz, ICX, r.u.x, o0, n);                     \
      e = fmax(e, max_rel_diff(esz, ICX, r.u.y, o1, n));                   \
      e = fmax(e, max_rel_diff(esz, ICX, r.u.z, o2, n));                   \
      e = fmax(e, max_rel_diff(esz, ICX, r.r.x, o3, n));                   \
      v3ur ## P ## _free(r);                                               \
      Timing tp = timed_run([&]{ v3ur ## P a = v3crossxy_dotxz_ ## P(x, y, w); \
                                 v3ur ## P ## _free(a); });                \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3crossxy_dotxz(n,          \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1,                  \
          w.x, 1, w.y, 1, w.z, 1,                                          \
          o0, 1, o1, 1, o2, 1, o3, 1); });                                 \
      add_result("v3crossxy_dotxz_" #P, tp, tc, e, (ICX ? 64.0 : 14.0), n, "u=x∧y,r=x·w"); }    \
                                                                           \
    { cblas_ ## P ## 3dotxy_dotxz(n, x.x, 1, x.y, 1, x.z, 1,               \
                                  y.x, 1, y.y, 1, y.z, 1,                  \
                                  w.x, 1, w.y, 1, w.z, 1,                  \
                                  o0, 1, o1, 1);                           \
      v1rq ## P r = v3dotxy_dotxz_ ## P(x, y, w);                          \
      double e = max_rel_diff(esz, ICX, r.r.x, o0, n);                     \
      e = fmax(e, max_rel_diff(esz, ICX, r.q.x, o1, n));                   \
      v1rq ## P ## _free(r);                                               \
      Timing tp = timed_run([&]{ v1rq ## P a = v3dotxy_dotxz_ ## P(x, y, w); \
                                 v1rq ## P ## _free(a); });                \
      Timing tc = timed_run([&]{ cblas_ ## P ## 3dotxy_dotxz(n,            \
          x.x, 1, x.y, 1, x.z, 1, y.x, 1, y.y, 1, y.z, 1,                  \
          w.x, 1, w.y, 1, w.z, 1,                                          \
          o0, 1, o1, 1); });                                               \
      add_result("v3dotxy_dotxz_" #P, tp, tc, e, (ICX ? 44.0 : 10.0), n, "r=x·y,q=x·w"); }      \
                                                                           \
    v3 ## P ## _free(&x); v3 ## P ## _free(&y); v3 ## P ## _free(&w);      \
    v1 ## P ## _free(&q); v1 ## P ## _free(&t);                            \
    v1 ## P ## _free(&xv); v1 ## P ## _free(&yv);                          \
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

}  // namespace

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
    std::printf("v3blas = v3blas call + result free; cblas = raw call, pre-allocated outputs\n");
    std::printf("max_rel_err compares the v3blas and cblas outputs element-wise\n");

    if (prec == 's') bench_all_s(n);
    else if (prec == 'c') bench_all_c(n);
    else if (prec == 'z') bench_all_z(n);
    else bench_all_d(n);

    std::printf("%-19s %13s %13s %8s %12s %13s %7s  %s  %s\n",
                "kernel", "v3blas ms/run", "cblas ms/run", "overhead",
                "FLOP/s", "max_rel_err", "iters", "status", "equation");
    for (const Result &res : results)
        std::printf("%-19s %13.3f %13.3f %8.2f %12.3e %13.3e %7d  %s  %s\n",
                    res.name.c_str(), res.ms_v3blas, res.ms_cblas,
                    res.overhead, res.flops_rate, res.max_rel_err, res.iters,
                    res.ok ? "OK" : "FAIL", res.eq.c_str());
    return 0;
}
