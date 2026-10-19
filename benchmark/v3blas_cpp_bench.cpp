/* v3blas_cpp_bench.cpp - timing for the v3blas.hpp overloaded front-end.
 *
 * Four representative ops per precision, called as plain templates:
 * overload resolution picks the precision from the argument types.
 * Timing only - bitwise equality of these calls with cblas_* is
 * v3blas_cpp_test's job.  Budget loop as in the other benches: run each
 * kernel until 0.005 s or 3 calls (override: -DK_MIN_TIME=... etc).
 *
 * Build:
 *   g++ -O2 -std=c++11 -I ../pkgs/openblas/include/openblas \
 *       v3blas_cpp_bench.cpp -L ../pkgs/openblas/lib -lopenblas -lm \
 *       -o v3blas_cpp_bench
 */
#include "v3blas.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <string>
#include <vector>
#include <functional>

#ifndef K_MIN_TIME
#define K_MIN_TIME 0.005
#endif
#ifndef K_MIN_ITERS
#define K_MIN_ITERS 5
#endif
#ifndef K_MAX_TIME
#define K_MAX_TIME 1.0
#endif

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static double timed_ms(const std::function<void()> &fn)
{
    long iters = 0;
    double t0 = now_sec();
    while (true) {
        fn();
        iters++;
        double el = now_sec() - t0;
        if ((el >= K_MIN_TIME && iters >= K_MIN_ITERS) || el >= K_MAX_TIME)
            return el / iters * 1e3;
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

struct Row { std::string name; double ms; double gflops; std::string eq; };
static std::vector<Row> rows;

template <class T> static void setv(T &v, long double x) { v = (T)x; }
template <> void setv(v3blas_floatcomplex &v, long double x)
{ v.re = (float)x; v.im = (float)(x * 0.7L); }
template <> void setv(v3blas_doublecomplex &v, long double x)
{ v.re = (double)x; v.im = x * 0.7L; }
template <class T> static void fill(T *v, int n, int S)
{ for (int i = 0; i < n; i++) setv(v[i], VAL(S, i)); }

/* P: view struct suffix (s/d/c/z); fl[]: math FLOPs per element */
#define BENCH_CPP(P, T)                                                      \
static void bench_ ## P(int n)                                               \
{                                                                            \
    T *x1 = (T *)malloc(6 * (size_t)n * sizeof(T));                          \
    T *x2 = x1 + n, *x3 = x1 + 2 * n;                                        \
    T *y1 = x1 + 3 * n, *y2 = x1 + 4 * n, *y3 = x1 + 5 * n;                  \
    T *o1 = (T *)malloc(4 * (size_t)n * sizeof(T));                          \
    T *o2 = o1 + n, *o3 = o1 + 2 * n, *o4 = o1 + 3 * n;                      \
    fill(x1, n, 1); fill(x2, n, 2); fill(x3, n, 3);                          \
    fill(y1, n, 4); fill(y2, n, 5); fill(y3, n, 6);                          \
    v3 ## P x = v3 ## P ## _wrap(x1, x2, x3, n);                             \
    v3 ## P y = v3 ## P ## _wrap(y1, y2, y3, n);                             \
    v3 ## P ow = v3 ## P ## _wrap(o1, o2, o3, n);   /* output view */         \
    double ms;                                                               \
    ms = timed_ms([&]{ v3dot(x, y, o1); });                                  \
    rows.push_back({"v3dot_" #P, ms, 5.0 * n / ms / 1.0e6, "r=x1y1+x2y2+x3y3"}); \
    ms = timed_ms([&]{ v3cross(x, y, ow); });                                \
    rows.push_back({"v3cross_" #P, ms, 9.0 * n / ms / 1.0e6, "w=x^y"});      \
    ms = timed_ms([&]{ v3drag(x, 1.5, 0.25, ow); });                         \
    rows.push_back({"v3drag_" #P, ms, 27.0 * n / ms / 1.0e6,                 \
                    "r=a*x*sqrt(x.x+eps)"});                                 \
    ms = timed_ms([&]{ v3mom_ke(y1, x, ow, o4); });                          \
    rows.push_back({"v3mom_ke_" #P, ms, 10.0 * n / ms / 1.0e6,               \
                    "m=s*x, k=0.5*s*(x.x)"});                                \
    free(x1);                                                                \
    free(o1);                                                                \
}

BENCH_CPP(s, float)
BENCH_CPP(d, double)
BENCH_CPP(c, v3blas_floatcomplex)
BENCH_CPP(z, v3blas_doublecomplex)

#undef BENCH_CPP

int main(int argc, char **argv)
{
    int n = 4096;
    if (argc > 1) {
        long v = std::strtol(argv[1], nullptr, 10);
        if (v > 0) n = (int)v;
    }
    std::printf("v3blas_cpp_bench: timing only (correctness: v3blas_cpp_test)\n");
    std::printf("n = %d\n", n);
    std::printf("%-12s %10s %10s  %s\n", "kernel", "ms/run", "GFLOP/s", "equation");
    bench_s(n); bench_d(n); bench_c(n); bench_z(n);
    for (const Row &r : rows)
        std::printf("%-12s %10.4f %10.3f  %s\n",
                    r.name.c_str(), r.ms, r.gflops, r.eq.c_str());
    return 0;
}
