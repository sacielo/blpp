/*
 * cblas_bench.cpp - timing table for the raw cblas_* physics extension
 * entry points (s d c z).  Pure benchmark: no reference values, no
 * verification, no equation dumps.  Correctness of these entries lives
 * in cblas_test.c (plus the in-tree OpenBLAS utests); the equation
 * column here is only the FLOP-count legend for the row.
 *
 * usage: cblas_bench [n] [s|d|c|z]
 *
 * Columns:
 *   ms/run      per-call wall time (min 0.5 s / 3 iters per kernel)
 *   FLOP/s      standard count: 1 per add/mul/sqrt; complex bilinear
 *               product = 6, complex add = 2 (printed exponential)
 *   Melem/s     million elements per second
 *   F/cyc       expected flops per element per loop iteration (FMA=1
 *               for real; complex as implemented: cmul 6, cadd 2)
 *   equation    space-free token so whole lines parse with awk
 *
 * Data: hash2(seed, element) in [-0.5, 0.5), full avalanche, no ramps.
 * Links an installed OpenBLAS only (see CMakeLists.txt).
 */
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <complex>
#include <functional>
#include <string>
#include <vector>

#include "cblas.h"

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
    double t_run;
    long iters;
};

static Timing timed_run(const std::function<void()> &fn)
{
    static const double kMinTime = K_MIN_TIME;
    static const long kMinIters = K_MIN_ITERS;
    static const double kMaxTime = K_MAX_TIME;
    Timing tm = { 0.0, 0 };
    fn();  /* warmup */
    double t0 = now_sec();
    while (true) {
        double b = now_sec();
        fn();
        tm.t_run += now_sec() - b;
        tm.iters++;
        double el = now_sec() - t0;
        if ((el >= kMinTime && tm.iters >= kMinIters) || el >= kMaxTime)
            return tm;
    }
}

/* in-place kernels: restore() runs before every call, outside the clock */
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

/* one element from a hash value: im = 0.7 * re for complex */
static float mk_s(long double v) { return (float)v; }
static double mk_d(long double v) { return (double)v; }
static std::complex<float> mk_c(long double v)
{ return std::complex<float>((float)v, (float)(0.7L * v)); }
static std::complex<double> mk_z(long double v)
{ return std::complex<double>((double)v, 0.7L * (double)v); }

static void fill1_s(float *v, int n, int S) { for (int i = 0; i < n; i++) v[i] = mk_s(VAL(S, i)); }
static void fill1_d(double *v, int n, int S) { for (int i = 0; i < n; i++) v[i] = mk_d(VAL(S, i)); }
static void fill1_c(std::complex<float> *v, int n, int S) { for (int i = 0; i < n; i++) v[i] = mk_c(VAL(S, i)); }
static void fill1_z(std::complex<double> *v, int n, int S) { for (int i = 0; i < n; i++) v[i] = mk_z(VAL(S, i)); }

struct Result {
    std::string name;
    double ms_per_run;
    double flops_rate;
    double meps;
    double fcyc;
    long iters;
    std::string eq;
};

static std::vector<Result> results;

/* one row: op name gets the precision prefix P appended */
#define PUSH(op, FL, FC, tm_, n_, eq)                                        \
    do {                                                                     \
        double per = (tm_).t_run / (tm_).iters;                              \
        char nm[24];                                                         \
        std::snprintf(nm, sizeof nm, "%s%s", pre, op);                        \
        results.push_back({nm, per * 1e3, (FL) * (n_) / per,                 \
                           (n_) / per / 1e6, (FC), (tm_).iters, eq});        \
    } while (0)

#define X1 bx0, 1, bx1, 1, bx2, 1
#define Y1 by0, 1, by1, 1, by2, 1
#define W1 bw0, 1, bw1, 1, bw2, 1

/*
 * All four precisions from one body.
 *   ICX:  0 real, 1 complex (selects the complex FLOP convention)
 *   INIT: sets the scalars s_ax (axpy), s_xa (xypa), s_cs (crossscal)
 *   CAX / CXA / CCS: those scalars as the cblas prototypes want them
 *   (by value for real, void* for complex)
 * Inputs: one malloc'd slab, 13 unit-stride arrays.  Outputs: ro[0..5]
 * in a second slab; axpy runs in place on ro[0] and is restored from
 * ay0 between timed calls.  lap reads a padded u (pad 16 elements).
 */
#define BENCH_PREC(P, T, ICX, INIT, CAX, CXA, CCS)                           \
static void bench_all_ ## P(int n)                                           \
{                                                                            \
    const char pre[] = #P;                                                   \
    T s_ax, s_xa, s_cs;                                                      \
    INIT;                                                                    \
    const void *pax = (const void *)&s_ax;                                   \
    const void *pxa = (const void *)&s_xa;                                   \
    const void *pcs = (const void *)&s_cs;                                   \
    (void)pax; (void)pxa; (void)pcs;                                         \
    const size_t esz = sizeof(T);                                            \
                                                                             \
    T *in = (T *)malloc(13 * (size_t)n * esz);                               \
    T *bx0 = in, *bx1 = in + n, *bx2 = in + 2 * n, *by0 = in + 3 * n,        \
      *by1 = in + 4 * n, *by2 = in + 5 * n, *bw0 = in + 6 * n,               \
      *bw1 = in + 7 * n, *bw2 = in + 8 * n, *bq = in + 9 * n,                \
      *bt = in + 10 * n, *bxv = in + 11 * n, *byv = in + 12 * n;             \
    for (int a = 0; a < 13; a++)                                             \
        fill1_ ## P(in + (size_t)a * n, n, 1 + a);                           \
    T *u = (T *)malloc(((size_t)n + 32) * esz);                              \
    for (int i = -16; i < n + 16; i++) u[16 + i] = mk_ ## P(VAL(16, i));     \
    T *lap = (T *)malloc((size_t)n * esz);                                   \
    T *scratch = (T *)malloc(7 * (size_t)n * esz);                           \
    T *ro[6];                                                                \
    for (int k = 0; k < 6; k++) ro[k] = scratch + (size_t)k * n;             \
    T *ay0 = scratch + 6 * (size_t)n;                                        \
    for (int i = 0; i < n; i++) { ay0[i] = byv[i]; ro[0][i] = byv[i]; }      \
                                                                             \
    {                                                                        \
      Timing t = timed_run_restore(                                          \
          [&] { std::memcpy(ro[0], ay0, (size_t)n * esz); },                 \
          [&] { cblas_ ## P ## axpy(n, CAX, bxv, 1, ro[0], 1); });           \
      PUSH("axpy", (ICX ? 8.0 : 2.0), (ICX ? 8.0 : 1.0), t, n, "y=a*x+y");   \
    }                                                                        \
    {                                                                        \
      Timing t = timed_run([&] { cblas_ ## P ## 1xypa(n, bq, 1,             \
          bt, 1, CXA, ro[0], 1); });                                     \
      PUSH("1xypa", (ICX ? 8.0 : 2.0), (ICX ? 8.0 : 1.0), t, n,              \
           "r=q.*t+a");                                                      \
    }                                                                        \
    {                                                                        \
      Timing t = timed_run([&] { cblas_ ## P ## lap5(n, 0.25, u + 16, 1,     \
          3, 7, lap, 1); });                                                 \
      PUSH("lap5", (ICX ? 12.0 : 6.0), (ICX ? 3.0 : 1.5), t, n,              \
           "lap=a*((u(-jx)+u(jx))+(u(-jy)+u(jy))-4u)");                      \
    }                                                                        \
    {                                                                        \
      Timing t = timed_run([&] { cblas_ ## P ## lap7(n, 0.25, u + 16, 1,     \
          3, 7, 11, lap, 1); });                                             \
      PUSH("lap7", (ICX ? 18.0 : 9.0), (ICX ? 4.0 : 2.0), t, n,              \
           "lap=a*((u(-jx)+u(jx))+(u(-jy)+u(jy))+(u(-jz)+u(jz))-6u)");       \
    }                                                                        \
    {                                                                        \
      Timing t = timed_run([&] { cblas_ ## P ## 3dot(n, X1, Y1,              \
          ro[0], 1); });                                                     \
      PUSH("3dot", (ICX ? 22.0 : 5.0), (ICX ? 22.0 : 3.0), t, n,             \
           "r=x1y1+x2y2+x3y3");                                              \
    }                                                                        \
    {                                                                        \
      Timing t = timed_run([&] { cblas_ ## P ## 3had(n, X1, Y1,              \
          ro[0], 1, ro[1], 1, ro[2], 1); });                                 \
      PUSH("3had", (ICX ? 18.0 : 3.0), (ICX ? 18.0 : 3.0), t, n, "w=x.*y");  \
    }                                                                        \
    {                                                                        \
      Timing t = timed_run([&] { cblas_ ## P ## 3sqr(n, X1, ro[0], 1); });   \
      PUSH("3sqr", (ICX ? 22.0 : 5.0), (ICX ? 22.0 : 3.0), t, n,             \
           "r=x1^2+x2^2+x3^2");                                              \
    }                                                                        \
    {                                                                        \
      Timing t = timed_run([&] { cblas_ ## P ## 1norm(n, bq, 1,              \
          ro[0], 1); });                                                     \
      PUSH("1norm", (ICX ? 16.0 : 2.0), (ICX ? 16.0 : 2.0), t, n,            \
           "r=sqrt(q\xc2\xb7" "q)");                                         \
    }                                                                        \
    {                                                                        \
      Timing t = timed_run([&] { cblas_ ## P ## 3cross(n, X1, Y1,            \
          ro[0], 1, ro[1], 1, ro[2], 1); });                                 \
      PUSH("3cross", (ICX ? 42.0 : 9.0), (ICX ? 42.0 : 6.0), t, n,           \
           "w=x\xe2\x88\xa7y");                                              \
    }                                                                        \
    {                                                                        \
      Timing t = timed_run([&] { cblas_ ## P ## 3crossscal(n, CCS, X1, Y1,   \
          ro[0], 1, ro[1], 1, ro[2], 1); });                                 \
      PUSH("3crossscal", (ICX ? 60.0 : 12.0), (ICX ? 60.0 : 9.0), t, n,      \
           "w=a\xc2\xb7" "(x\xe2\x88\xa7" "y)");                             \
    }                                                                        \
    {                                                                        \
      Timing t = timed_run([&] { cblas_ ## P ## 3crossdot(n, X1, Y1, W1,     \
          ro[0], 1); });                                                     \
      PUSH("3crossdot", (ICX ? 64.0 : 14.0), (ICX ? 64.0 : 9.0), t, n,       \
           "r=(x\xe2\x88\xa7y)\xc2\xb7" "w");                                \
    }                                                                        \
    {                                                                        \
      Timing t = timed_run([&] { cblas_ ## P ## 3crosssqr(n, X1, Y1,         \
          ro[0], 1); });                                                     \
      PUSH("3crosssqr", (ICX ? 64.0 : 14.0), (ICX ? 64.0 : 9.0), t, n,       \
           "r=(x\xe2\x88\xa7y)\xc2\xb7" "(x\xe2\x88\xa7" "y)");              \
    }                                                                        \
    {                                                                        \
      Timing t = timed_run([&] { cblas_ ## P ## 3crossxy_crossxz(n, X1,      \
          Y1, W1, ro[0], 1, ro[1], 1, ro[2], 1, ro[3], 1, ro[4], 1,          \
          ro[5], 1); });                                                     \
      PUSH("3crossxy_crossxz", (ICX ? 84.0 : 18.0), (ICX ? 84.0 : 12.0),     \
           t, n, "u=x\xe2\x88\xa7y,v=x\xe2\x88\xa7" "w");                    \
    }                                                                        \
    {                                                                        \
      Timing t = timed_run([&] { cblas_ ## P ## 3crossxy_dotxz(n, X1,        \
          Y1, W1, ro[0], 1, ro[1], 1, ro[2], 1, ro[3], 1); });               \
      PUSH("3crossxy_dotxz", (ICX ? 64.0 : 14.0), (ICX ? 64.0 : 9.0), t, n,  \
           "u=x\xe2\x88\xa7y,r=x\xc2\xb7" "w");                              \
    }                                                                        \
    {                                                                        \
      Timing t = timed_run([&] { cblas_ ## P ## 3dotxy_dotxz(n, X1, Y1,      \
          W1, ro[0], 1, ro[1], 1); });                                       \
      PUSH("3dotxy_dotxz", (ICX ? 44.0 : 10.0), (ICX ? 44.0 : 6.0), t, n,    \
           "r=x\xc2\xb7y,q=x\xc2\xb7" "w");                                  \
    }                                                                        \
    {                                                                        \
      Timing t = timed_run([&] { cblas_ ## P ## 3crosscross(n, X1, Y1,       \
          W1, ro[0], 1, ro[1], 1, ro[2], 1); });                             \
      PUSH("3crosscross", (ICX ? 72.0 : 24.0), (ICX ? 72.0 : 12.0), t, n,    \
           "r=(x\xe2\x88\xa7y)\xe2\x88\xa7" "w");                            \
    }                                                                        \
    {                                                                        \
      Timing t = timed_run([&] { cblas_ ## P ## 3norm_unit(n, X1, 0.25,      \
          ro[0], 1, ro[1], 1, ro[2], 1, ro[3], 1); });                       \
      PUSH("3norm_unit", (ICX ? 56.0 : 24.0), (ICX ? 56.0 : 12.0), t, n,     \
           "r=sqrt(x\xc2\xb7x+eps),u=x/r");                                  \
    }                                                                        \
    {                                                                        \
      Timing t = timed_run([&] { cblas_ ## P ## 3refl(n, X1, Y1, 1.5,        \
          0.25, ro[0], 1, ro[1], 1, ro[2], 1); });                           \
      PUSH("3refl", (ICX ? 44.0 : 20.0), (ICX ? 44.0 : 10.0), t, n,          \
           "q=x-a(x\xc2\xb7y)/(y\xc2\xb7y+eps)y");                           \
    }                                                                        \
    {                                                                        \
      Timing t = timed_run([&] { cblas_ ## P ## 3exb(n, X1, Y1, 1.5,         \
          0.25, ro[0], 1, ro[1], 1, ro[2], 1); });                           \
      PUSH("3exb", (ICX ? 52.0 : 20.0), (ICX ? 52.0 : 10.0), t, n,           \
           "r=a(x\xe2\x88\xa7y)/(y\xc2\xb7y+eps)");                          \
    }                                                                        \
    {                                                                        \
      Timing t = timed_run([&] { cblas_ ## P ## 3drag(n, X1, 1.5,            \
          0.25, ro[0], 1, ro[1], 1, ro[2], 1); });                           \
      PUSH("3drag", (ICX ? 48.0 : 27.0), (ICX ? 48.0 : 14.0), t, n,          \
           "r=a*x*sqrt(x\xc2\xb7" "x+eps)");                                 \
    }                                                                        \
    {                                                                        \
      Timing t = timed_run([&] { cblas_ ## P ## 3mom_ke(n, bq, 1, X1,        \
          ro[0], 1, ro[1], 1, ro[2], 1, ro[3], 1); });                       \
      PUSH("3mom_ke", (ICX ? 28.0 : 10.0), (ICX ? 28.0 : 5.0), t, n,         \
           "m=s*x,k=0.5*s*(x\xc2\xb7" "x)");                                 \
    }                                                                        \
                                                                             \
    free(in); free(u); free(lap); free(scratch);                             \
}

BENCH_PREC(s, float, 0, s_ax = 1.5f; s_xa = 0.25f; s_cs = 1.5f,
           s_ax, s_xa, s_cs)
BENCH_PREC(d, double, 0, s_ax = 1.5; s_xa = 0.25; s_cs = 1.5,
           s_ax, s_xa, s_cs)
BENCH_PREC(c, std::complex<float>, 1,
           s_ax = std::complex<float>(1.5f, 0.5f); s_xa = std::complex<float>(0.25f, 0.1f); s_cs = std::complex<float>(1.5f, 0.5f),
           pax, pxa, pcs)
BENCH_PREC(z, std::complex<double>, 1,
           s_ax = std::complex<double>(1.5, 0.5); s_xa = std::complex<double>(0.25, 0.1); s_cs = std::complex<double>(1.5, 0.5),
           pax, pxa, pcs)

#undef BENCH_PREC
#undef X1
#undef Y1
#undef W1

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

    if (prec == 's') bench_all_s(n);
    else if (prec == 'c') bench_all_c(n);
    else if (prec == 'z') bench_all_z(n);
    else bench_all_d(n);

    std::printf("%-19s %10s %12s %12s %6s %7s  %s\n",
                "kernel", "ms/run", "FLOP/s", "Melem/s", "F/cyc",
                "iters", "equation");
    for (const Result &res : results)
        std::printf("%-19s %10.3f %12.3e %12.3f %6.1f %7ld  %s\n",
                    res.name.c_str(), res.ms_per_run, res.flops_rate,
                    res.meps, res.fcyc, res.iters, res.eq.c_str());
    return 0;
}
