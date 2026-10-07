// Benchmark for memory-bound, element-wise BLAS L1 kernels.
//
// usage: benchmark [n] [s|d|c|z]
//   n    vector size (default 1000)
//   sdcz precision (default d); order of the two arguments is free
//
// Each kernel is first checked against a known result (independently
// computed reference in long double), then timed over repeated runs.
// In a non-NDEBUG build (CMake debug mode) each kernel is run once and
// the first element of its expression is printed for hand verification;
// no timing is done. With NDEBUG the performance table is printed.
// The table's last column is the kernel equation, a single space-free
// token (e.g. w=a*(x^y)) so whole lines can be parsed with awk.
//
// FLOP conventions:
//   GFLOP/s: standard count, one flop per add/mul/sqrt; a complex
//     bilinear product is 6, a complex add is 2.
//   F/cyc:  expected flops per element per loop iteration, assuming a
//     fused multiply-add counts as 1 for real; complex kernels are
//     counted as implemented (bilinear product 6, add 2, csqrt 16).
//
// Add new kernels as new bench_*() functions and list them in main().
//
// Links against a pre-built OpenBLAS; this project does not build OpenBLAS.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "cblas.h"

using Clock = std::chrono::steady_clock;
using ld = long double;

namespace {

constexpr double kMinTime = 0.5;   // seconds of runtime per kernel
constexpr int kMinIters = 3;
constexpr double kMaxTime = 10.0;  // cap for slow machines
constexpr double kEps = 1e-12;     // relative tolerance vs. the reference (d/z)
constexpr double kEpsF = 1e-5;     // ditto for float precision (s/c)

struct Result {
    std::string name;
    double flops;       // FLOPs per element (GFLOP/s count)
    double fcyc;        // expected FLOPs per element per iteration
    double ms_per_run;
    double gflops;
    double meps;        // million elements / second
    double max_rel_err; // vs. the long double reference
    int iters;
    bool ok;
    std::string eq;     // space-free equation token, printed last
};

Result make_result(const std::string &name, double flops_per_elem,
                   double fcyc, double per_run, int n, int iters,
                   double err, double eps, const char *eq)
{
    return {name, flops_per_elem, fcyc, per_run * 1e3,
            flops_per_elem * n / per_run / 1e9, n / per_run / 1e6,
            err, iters, err <= eps, eq};
}

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

// same, but with a restore() step before each timed call (in-place kernels)
Timing timed_run_restore(const std::function<void()> &restore,
                         const std::function<void()> &call)
{
    restore();
    call(); // warmup

    double t_run = 0.0;
    int iters = 0;
    auto t_start = Clock::now();
    for (;;) {
        restore();
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

template <typename T>
double max_rel_err(const T *v, const ld *ref, int n)
{
    double m = 0.0;
    for (int i = 0; i < n; ++i) {
        double r = (double)ref[i];
        double e = std::fabs((double)v[i] - r);
        m = std::fmax(m, e / std::fmax(1.0, std::fabs(r)));
    }
    return m;
}

// interleaved complex arrays, 2n elements each
template <typename T>
double max_rel_err_c(const T *v, const ld *ref, int n)
{
    double m = 0.0;
    for (int i = 0; i < 2 * n; ++i) {
        double r = (double)ref[i];
        double e = std::fabs((double)v[i] - r);
        m = std::fmax(m, e / std::fmax(1.0, std::fabs(r)));
    }
    return m;
}

template <typename T>
void seeded(std::vector<T> &v, int seed)
{
    for (int i = 0; i < (int)v.size(); ++i)
        v[i] = 1e-3 * ((i + seed) % 997 - 498);
}

// interleaved complex: re from seed, im from seed+7
template <typename T>
void seeded_c(std::vector<T> &v, int seed)
{
    int n = (int)v.size() / 2;
    for (int i = 0; i < n; ++i) {
        v[2 * i]     = T(1e-3 * ((i + seed) % 997 - 498));
        v[2 * i + 1] = T(1e-3 * ((i + seed + 7) % 997 - 498));
    }
}

// bilinear complex product (no conjugation)
inline void cprod(ld a0, ld a1, ld b0, ld b1, ld &o0, ld &o1)
{
    o0 = a0 * b0 - a1 * b1;
    o1 = a0 * b1 + a1 * b0;
}

// bilinear cross product, element i; component arrays interleaved
template <typename T>
void ccross(const T *a1, const T *a2, const T *a3,
            const T *b1, const T *b2, const T *b3,
            ld *w1, ld *w2, ld *w3, int i)
{
    ld p0, p1, q0, q1;
    cprod((ld)a2[2 * i], (ld)a2[2 * i + 1], (ld)b3[2 * i], (ld)b3[2 * i + 1], p0, p1);
    cprod((ld)a3[2 * i], (ld)a3[2 * i + 1], (ld)b2[2 * i], (ld)b2[2 * i + 1], q0, q1);
    w1[2 * i] = p0 - q0;
    w1[2 * i + 1] = p1 - q1;
    cprod((ld)a3[2 * i], (ld)a3[2 * i + 1], (ld)b1[2 * i], (ld)b1[2 * i + 1], p0, p1);
    cprod((ld)a1[2 * i], (ld)a1[2 * i + 1], (ld)b3[2 * i], (ld)b3[2 * i + 1], q0, q1);
    w2[2 * i] = p0 - q0;
    w2[2 * i + 1] = p1 - q1;
    cprod((ld)a1[2 * i], (ld)a1[2 * i + 1], (ld)b2[2 * i], (ld)b2[2 * i + 1], p0, p1);
    cprod((ld)a2[2 * i], (ld)a2[2 * i + 1], (ld)b1[2 * i], (ld)b1[2 * i + 1], q0, q1);
    w3[2 * i] = p0 - q0;
    w3[2 * i + 1] = p1 - q1;
}

// bilinear triple dot product x.y, element i
template <typename T>
void cdot3(const T *x1, const T *x2, const T *x3,
           const T *y1, const T *y2, const T *y3,
           ld &o0, ld &o1, int i)
{
    ld s0 = 0, s1 = 0, p0, p1;
    cprod((ld)x1[2 * i], (ld)x1[2 * i + 1], (ld)y1[2 * i], (ld)y1[2 * i + 1], p0, p1);
    s0 += p0; s1 += p1;
    cprod((ld)x2[2 * i], (ld)x2[2 * i + 1], (ld)y2[2 * i], (ld)y2[2 * i + 1], p0, p1);
    s0 += p0; s1 += p1;
    cprod((ld)x3[2 * i], (ld)x3[2 * i + 1], (ld)y3[2 * i], (ld)y3[2 * i + 1], p0, p1);
    s0 += p0; s1 += p1;
    o0 = s0;
    o1 = s1;
}

// y = alpha*x + y  (in place on y, restored from y0 before every run)
template <typename T, typename Call>
Result bench_axpy_t(int n, const char *nm, double eps, T alpha,
                    Call call)
{
    std::vector<T> x(n), y0(n), y(n);
    std::vector<ld> ref(n);
    seeded(x, 1);
    seeded(y0, 2);
    for (int i = 0; i < n; ++i)
        ref[i] = (ld)alpha * x[i] + y0[i];

    const double flops = 2.0, fcyc = 1.0;
    const char *eq = "y=a*x+y";
#ifndef NDEBUG
    std::memcpy(y.data(), y0.data(), n * sizeof(T));
    call(n, alpha, x.data(), 1, y.data(), 1);
    double err = max_rel_err(y.data(), ref.data(), n);
    std::printf("%s[0]: a*x[0] + y[0] = %g*%g + %g = %g\n", nm,
                (double)alpha, (double)x[0], (double)y0[0], (double)y[0]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run_restore(
        [&] { std::memcpy(y.data(), y0.data(), n * sizeof(T)); },
        [&] { call(n, alpha, x.data(), 1, y.data(), 1); });
    double err = max_rel_err(y.data(), ref.data(), n);
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// r = q.*t + a  (q and t are read-only, r is fully rewritten each run)
template <typename T, typename Call>
Result bench_xypa_t(int n, const char *nm, double eps, T a,
                    Call call)
{
    std::vector<T> q(n), t(n), r(n);
    std::vector<ld> ref(n);
    seeded(q, 3);
    seeded(t, 4);
    for (int i = 0; i < n; ++i)
        ref[i] = (ld)q[i] * t[i] + (ld)a;

    const double flops = 2.0, fcyc = 1.0;
    const char *eq = "r=q.*t+a";
#ifndef NDEBUG
    call(n, a, q.data(), 1, t.data(), 1, r.data(), 1);
    double err = max_rel_err(r.data(), ref.data(), n);
    std::printf("%s[0]: q[0] .* t[0] + a = %g .* %g + %g = %g\n", nm,
                (double)q[0], (double)t[0], (double)a, (double)r[0]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing tm = timed_run([&] { call(n, a, q.data(), 1, t.data(), 1, r.data(), 1); });
    double err = max_rel_err(r.data(), ref.data(), n);
    return make_result(nm, flops, fcyc, tm.t_run / tm.iters, n, tm.iters, err, eps, eq);
#endif
}

// r = x1y1 + x2y2 + x3y3
template <typename T, typename Call>
Result bench_3dot_t(int n, const char *nm, double eps,
                    Call call)
{
    std::vector<T> x1(n), x2(n), x3(n), y1(n), y2(n), y3(n), r(n);
    std::vector<ld> ref(n);
    seeded(x1, 11);
    seeded(x2, 12);
    seeded(x3, 13);
    seeded(y1, 14);
    seeded(y2, 15);
    seeded(y3, 16);
    for (int i = 0; i < n; ++i)
        ref[i] = (ld)x1[i] * y1[i] + (ld)x2[i] * y2[i] + (ld)x3[i] * y3[i];

    const double flops = 5.0, fcyc = 3.0;
    const char *eq = "r=x1y1+x2y2+x3y3";
#ifndef NDEBUG
    call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
         y1.data(), 1, y2.data(), 1, y3.data(), 1, r.data(), 1);
    double err = max_rel_err(r.data(), ref.data(), n);
    std::printf("%s[0]: (x1[0] x2[0] x3[0]) cdot (y1[0] y2[0] y3[0]) = r[0]\n"
                "(%g %g %g) cdot (%g %g %g) = %g\n", nm,
                (double)x1[0], (double)x2[0], (double)x3[0],
                (double)y1[0], (double)y2[0], (double)y3[0], (double)r[0]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run([&] {
        call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
             y1.data(), 1, y2.data(), 1, y3.data(), 1, r.data(), 1);
    });
    double err = max_rel_err(r.data(), ref.data(), n);
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// w = x.*y  (Hadamard product)
template <typename T, typename Call>
Result bench_3had_t(int n, const char *nm, double eps,
                    Call call)
{
    std::vector<T> x1(n), x2(n), x3(n), y1(n), y2(n), y3(n);
    std::vector<T> w1(n), w2(n), w3(n);
    std::vector<ld> w1_ref(n), w2_ref(n), w3_ref(n);
    seeded(x1, 21);
    seeded(x2, 22);
    seeded(x3, 23);
    seeded(y1, 24);
    seeded(y2, 25);
    seeded(y3, 26);
    for (int i = 0; i < n; ++i) {
        w1_ref[i] = (ld)x1[i] * y1[i];
        w2_ref[i] = (ld)x2[i] * y2[i];
        w3_ref[i] = (ld)x3[i] * y3[i];
    }

    const double flops = 3.0, fcyc = 3.0;
    const char *eq = "w=x.*y";
#ifndef NDEBUG
    call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
         y1.data(), 1, y2.data(), 1, y3.data(), 1,
         w1.data(), 1, w2.data(), 1, w3.data(), 1);
    double err = std::fmax(max_rel_err(w1.data(), w1_ref.data(), n),
                   std::fmax(max_rel_err(w2.data(), w2_ref.data(), n),
                             max_rel_err(w3.data(), w3_ref.data(), n)));
    std::printf("%s[0]: x1[0] .* y1[0] = w1[0]  etc.\n"
                "(%g .* %g = %g)  (%g .* %g = %g)  (%g .* %g = %g)\n", nm,
                (double)x1[0], (double)y1[0], (double)w1[0],
                (double)x2[0], (double)y2[0], (double)w2[0],
                (double)x3[0], (double)y3[0], (double)w3[0]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run([&] {
        call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
             y1.data(), 1, y2.data(), 1, y3.data(), 1,
             w1.data(), 1, w2.data(), 1, w3.data(), 1);
    });
    double err = std::fmax(max_rel_err(w1.data(), w1_ref.data(), n),
                   std::fmax(max_rel_err(w2.data(), w2_ref.data(), n),
                             max_rel_err(w3.data(), w3_ref.data(), n)));
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// r = x1^2 + x2^2 + x3^2
template <typename T, typename Call>
Result bench_3sqr_t(int n, const char *nm, double eps,
                    Call call)
{
    std::vector<T> x1(n), x2(n), x3(n), r(n);
    std::vector<ld> ref(n);
    seeded(x1, 31);
    seeded(x2, 32);
    seeded(x3, 33);
    for (int i = 0; i < n; ++i)
        ref[i] = (ld)x1[i] * x1[i] + (ld)x2[i] * x2[i] + (ld)x3[i] * x3[i];

    const double flops = 5.0, fcyc = 3.0;
    const char *eq = "r=x1^2+x2^2+x3^2";
#ifndef NDEBUG
    call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1, r.data(), 1);
    double err = max_rel_err(r.data(), ref.data(), n);
    std::printf("%s[0]: x1[0]^2 + x2[0]^2 + x3[0]^2 = r[0]\n"
                "(%g^2 + %g^2 + %g^2 = %g)\n", nm,
                (double)x1[0], (double)x2[0], (double)x3[0], (double)r[0]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run([&] {
        call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1, r.data(), 1);
    });
    double err = max_rel_err(r.data(), ref.data(), n);
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// r = sqrt(q.*q)
template <typename T, typename Call>
Result bench_1norm_t(int n, const char *nm, double eps,
                     Call call)
{
    std::vector<T> q(n), r(n);
    std::vector<ld> ref(n);
    seeded(q, 41);
    for (int i = 0; i < n; ++i)
        ref[i] = sqrtl((ld)q[i] * q[i]);

    const double flops = 2.0, fcyc = 2.0;
    const char *eq = "r=sqrt(q.q)";
#ifndef NDEBUG
    call(n, q.data(), 1, r.data(), 1);
    double err = max_rel_err(r.data(), ref.data(), n);
    std::printf("%s[0]: sqrt(q[0] * q[0]) = sqrt(%g * %g) = %g\n", nm,
                (double)q[0], (double)q[0], (double)r[0]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run([&] { call(n, q.data(), 1, r.data(), 1); });
    double err = max_rel_err(r.data(), ref.data(), n);
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// w = x^y  (cross product)
template <typename T, typename Call>
Result bench_3cross_t(int n, const char *nm, double eps,
                      Call call)
{
    std::vector<T> x1(n), x2(n), x3(n), y1(n), y2(n), y3(n);
    std::vector<T> w1(n), w2(n), w3(n);
    std::vector<ld> w1_ref(n), w2_ref(n), w3_ref(n);
    seeded(x1, 51);
    seeded(x2, 52);
    seeded(x3, 53);
    seeded(y1, 54);
    seeded(y2, 55);
    seeded(y3, 56);
    for (int i = 0; i < n; ++i) {
        w1_ref[i] = (ld)x2[i] * y3[i] - (ld)x3[i] * y2[i];
        w2_ref[i] = (ld)x3[i] * y1[i] - (ld)x1[i] * y3[i];
        w3_ref[i] = (ld)x1[i] * y2[i] - (ld)x2[i] * y1[i];
    }

    const double flops = 9.0, fcyc = 6.0;
    const char *eq = "w=x^y";
#ifndef NDEBUG
    call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
         y1.data(), 1, y2.data(), 1, y3.data(), 1,
         w1.data(), 1, w2.data(), 1, w3.data(), 1);
    double err = std::fmax(max_rel_err(w1.data(), w1_ref.data(), n),
                   std::fmax(max_rel_err(w2.data(), w2_ref.data(), n),
                             max_rel_err(w3.data(), w3_ref.data(), n)));
    std::printf("%s[0]: x^y = w\n"
                "(%g %g %g)^(%g %g %g) = (%g %g %g)\n", nm,
                (double)x1[0], (double)x2[0], (double)x3[0],
                (double)y1[0], (double)y2[0], (double)y3[0],
                (double)w1[0], (double)w2[0], (double)w3[0]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run([&] {
        call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
             y1.data(), 1, y2.data(), 1, y3.data(), 1,
             w1.data(), 1, w2.data(), 1, w3.data(), 1);
    });
    double err = std::fmax(max_rel_err(w1.data(), w1_ref.data(), n),
                   std::fmax(max_rel_err(w2.data(), w2_ref.data(), n),
                             max_rel_err(w3.data(), w3_ref.data(), n)));
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// w = a*(x^y)
template <typename T, typename Call>
Result bench_3crossscal_t(int n, const char *nm, double eps, T a,
                          Call call)
{
    std::vector<T> x1(n), x2(n), x3(n), y1(n), y2(n), y3(n);
    std::vector<T> w1(n), w2(n), w3(n);
    std::vector<ld> w1_ref(n), w2_ref(n), w3_ref(n);
    seeded(x1, 61);
    seeded(x2, 62);
    seeded(x3, 63);
    seeded(y1, 64);
    seeded(y2, 65);
    seeded(y3, 66);
    for (int i = 0; i < n; ++i) {
        w1_ref[i] = (ld)a * ((ld)x2[i] * y3[i] - (ld)x3[i] * y2[i]);
        w2_ref[i] = (ld)a * ((ld)x3[i] * y1[i] - (ld)x1[i] * y3[i]);
        w3_ref[i] = (ld)a * ((ld)x1[i] * y2[i] - (ld)x2[i] * y1[i]);
    }

    const double flops = 12.0, fcyc = 9.0;
    const char *eq = "w=a*(x^y)";
#ifndef NDEBUG
    call(n, a, x1.data(), 1, x2.data(), 1, x3.data(), 1,
         y1.data(), 1, y2.data(), 1, y3.data(), 1,
         w1.data(), 1, w2.data(), 1, w3.data(), 1);
    double err = std::fmax(max_rel_err(w1.data(), w1_ref.data(), n),
                   std::fmax(max_rel_err(w2.data(), w2_ref.data(), n),
                             max_rel_err(w3.data(), w3_ref.data(), n)));
    std::printf("%s[0]: a*(x^y) = w  (a = %g)\n"
                "(%g %g %g)^(%g %g %g) = (%g %g %g)\n", nm, (double)a,
                (double)x1[0], (double)x2[0], (double)x3[0],
                (double)y1[0], (double)y2[0], (double)y3[0],
                (double)w1[0], (double)w2[0], (double)w3[0]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run([&] {
        call(n, a, x1.data(), 1, x2.data(), 1, x3.data(), 1,
             y1.data(), 1, y2.data(), 1, y3.data(), 1,
             w1.data(), 1, w2.data(), 1, w3.data(), 1);
    });
    double err = std::fmax(max_rel_err(w1.data(), w1_ref.data(), n),
                   std::fmax(max_rel_err(w2.data(), w2_ref.data(), n),
                             max_rel_err(w3.data(), w3_ref.data(), n)));
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// r = (x^y).w
template <typename T, typename Call>
Result bench_3crossdot_t(int n, const char *nm, double eps,
                         Call call)
{
    std::vector<T> x1(n), x2(n), x3(n), y1(n), y2(n), y3(n);
    std::vector<T> w1(n), w2(n), w3(n), r(n);
    std::vector<ld> ref(n);
    seeded(x1, 71);
    seeded(x2, 72);
    seeded(x3, 73);
    seeded(y1, 74);
    seeded(y2, 75);
    seeded(y3, 76);
    seeded(w1, 77);
    seeded(w2, 78);
    seeded(w3, 79);
    for (int i = 0; i < n; ++i) {
        ld c1 = (ld)x2[i] * y3[i] - (ld)x3[i] * y2[i];
        ld c2 = (ld)x3[i] * y1[i] - (ld)x1[i] * y3[i];
        ld c3 = (ld)x1[i] * y2[i] - (ld)x2[i] * y1[i];
        ref[i] = c1 * w1[i] + c2 * w2[i] + c3 * w3[i];
    }

    const double flops = 14.0, fcyc = 9.0;
    const char *eq = "r=(x^y).w";
#ifndef NDEBUG
    call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
         y1.data(), 1, y2.data(), 1, y3.data(), 1,
         w1.data(), 1, w2.data(), 1, w3.data(), 1, r.data(), 1);
    double err = max_rel_err(r.data(), ref.data(), n);
    std::printf("%s[0]: (x^y).w = r\n"
                "((%g %g %g)^(%g %g %g)).(%g %g %g) = %g\n", nm,
                (double)x1[0], (double)x2[0], (double)x3[0],
                (double)y1[0], (double)y2[0], (double)y3[0],
                (double)w1[0], (double)w2[0], (double)w3[0], (double)r[0]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run([&] {
        call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
             y1.data(), 1, y2.data(), 1, y3.data(), 1,
             w1.data(), 1, w2.data(), 1, w3.data(), 1, r.data(), 1);
    });
    double err = max_rel_err(r.data(), ref.data(), n);
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// r = (x^y).(x^y)
template <typename T, typename Call>
Result bench_3crosssqr_t(int n, const char *nm, double eps,
                         Call call)
{
    std::vector<T> x1(n), x2(n), x3(n), y1(n), y2(n), y3(n), r(n);
    std::vector<ld> ref(n);
    seeded(x1, 81);
    seeded(x2, 82);
    seeded(x3, 83);
    seeded(y1, 84);
    seeded(y2, 85);
    seeded(y3, 86);
    for (int i = 0; i < n; ++i) {
        ld c1 = (ld)x2[i] * y3[i] - (ld)x3[i] * y2[i];
        ld c2 = (ld)x3[i] * y1[i] - (ld)x1[i] * y3[i];
        ld c3 = (ld)x1[i] * y2[i] - (ld)x2[i] * y1[i];
        ref[i] = c1 * c1 + c2 * c2 + c3 * c3;
    }

    const double flops = 14.0, fcyc = 9.0;
    const char *eq = "r=(x^y).(x^y)";
#ifndef NDEBUG
    call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
         y1.data(), 1, y2.data(), 1, y3.data(), 1, r.data(), 1);
    double err = max_rel_err(r.data(), ref.data(), n);
    std::printf("%s[0]: (x^y).(x^y) = r\n"
                "((%g %g %g)^(%g %g %g)).(same) = %g\n", nm,
                (double)x1[0], (double)x2[0], (double)x3[0],
                (double)y1[0], (double)y2[0], (double)y3[0], (double)r[0]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run([&] {
        call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
             y1.data(), 1, y2.data(), 1, y3.data(), 1, r.data(), 1);
    });
    double err = max_rel_err(r.data(), ref.data(), n);
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// u = x^y, v = x^w
template <typename T, typename Call>
Result bench_3crossxy_crossxz_t(int n, const char *nm, double eps,
                                Call call)
{
    std::vector<T> x1(n), x2(n), x3(n), y1(n), y2(n), y3(n);
    std::vector<T> w1(n), w2(n), w3(n);
    std::vector<T> u1(n), u2(n), u3(n), v1(n), v2(n), v3(n);
    std::vector<ld> u1_ref(n), u2_ref(n), u3_ref(n);
    std::vector<ld> v1_ref(n), v2_ref(n), v3_ref(n);
    seeded(x1, 91);
    seeded(x2, 92);
    seeded(x3, 93);
    seeded(y1, 94);
    seeded(y2, 95);
    seeded(y3, 96);
    seeded(w1, 97);
    seeded(w2, 98);
    seeded(w3, 99);
    for (int i = 0; i < n; ++i) {
        u1_ref[i] = (ld)x2[i] * y3[i] - (ld)x3[i] * y2[i];
        u2_ref[i] = (ld)x3[i] * y1[i] - (ld)x1[i] * y3[i];
        u3_ref[i] = (ld)x1[i] * y2[i] - (ld)x2[i] * y1[i];
        v1_ref[i] = (ld)x2[i] * w3[i] - (ld)x3[i] * w2[i];
        v2_ref[i] = (ld)x3[i] * w1[i] - (ld)x1[i] * w3[i];
        v3_ref[i] = (ld)x1[i] * w2[i] - (ld)x2[i] * w1[i];
    }

    const double flops = 18.0, fcyc = 12.0;
    const char *eq = "u=x^y,v=x^w";
    double err;
#ifndef NDEBUG
    call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
         y1.data(), 1, y2.data(), 1, y3.data(), 1,
         w1.data(), 1, w2.data(), 1, w3.data(), 1,
         u1.data(), 1, u2.data(), 1, u3.data(), 1,
         v1.data(), 1, v2.data(), 1, v3.data(), 1);
    err = std::fmax(max_rel_err(u1.data(), u1_ref.data(), n),
              std::fmax(max_rel_err(u2.data(), u2_ref.data(), n),
                        std::fmax(max_rel_err(u3.data(), u3_ref.data(), n),
                                  std::fmax(max_rel_err(v1.data(), v1_ref.data(), n),
                                            std::fmax(max_rel_err(v2.data(), v2_ref.data(), n),
                                                      max_rel_err(v3.data(), v3_ref.data(), n))))));
    std::printf("%s[0]: x^y = u, x^w = v\n"
                "u = (%g %g %g)  v = (%g %g %g)\n", nm,
                (double)u1[0], (double)u2[0], (double)u3[0],
                (double)v1[0], (double)v2[0], (double)v3[0]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run([&] {
        call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
             y1.data(), 1, y2.data(), 1, y3.data(), 1,
             w1.data(), 1, w2.data(), 1, w3.data(), 1,
             u1.data(), 1, u2.data(), 1, u3.data(), 1,
             v1.data(), 1, v2.data(), 1, v3.data(), 1);
    });
    err = std::fmax(max_rel_err(u1.data(), u1_ref.data(), n),
              std::fmax(max_rel_err(u2.data(), u2_ref.data(), n),
                        std::fmax(max_rel_err(u3.data(), u3_ref.data(), n),
                                  std::fmax(max_rel_err(v1.data(), v1_ref.data(), n),
                                            std::fmax(max_rel_err(v2.data(), v2_ref.data(), n),
                                                      max_rel_err(v3.data(), v3_ref.data(), n))))));
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// u = x^y, r = x.w
template <typename T, typename Call>
Result bench_3crossxy_dotxz_t(int n, const char *nm, double eps,
                              Call call)
{
    std::vector<T> x1(n), x2(n), x3(n), y1(n), y2(n), y3(n);
    std::vector<T> w1(n), w2(n), w3(n);
    std::vector<T> u1(n), u2(n), u3(n), r(n);
    std::vector<ld> u1_ref(n), u2_ref(n), u3_ref(n), ref(n);
    seeded(x1, 101);
    seeded(x2, 102);
    seeded(x3, 103);
    seeded(y1, 104);
    seeded(y2, 105);
    seeded(y3, 106);
    seeded(w1, 107);
    seeded(w2, 108);
    seeded(w3, 109);
    for (int i = 0; i < n; ++i) {
        u1_ref[i] = (ld)x2[i] * y3[i] - (ld)x3[i] * y2[i];
        u2_ref[i] = (ld)x3[i] * y1[i] - (ld)x1[i] * y3[i];
        u3_ref[i] = (ld)x1[i] * y2[i] - (ld)x2[i] * y1[i];
        ref[i] = (ld)x1[i] * w1[i] + (ld)x2[i] * w2[i]
               + (ld)x3[i] * w3[i];
    }

    const double flops = 14.0, fcyc = 9.0;
    const char *eq = "u=x^y,r=x.w";
    double err;
#ifndef NDEBUG
    call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
         y1.data(), 1, y2.data(), 1, y3.data(), 1,
         w1.data(), 1, w2.data(), 1, w3.data(), 1,
         u1.data(), 1, u2.data(), 1, u3.data(), 1, r.data(), 1);
    err = std::fmax(max_rel_err(u1.data(), u1_ref.data(), n),
              std::fmax(max_rel_err(u2.data(), u2_ref.data(), n),
                        std::fmax(max_rel_err(u3.data(), u3_ref.data(), n),
                                  max_rel_err(r.data(), ref.data(), n))));
    std::printf("%s[0]: x^y = u, x.w = r\n"
                "u = (%g %g %g)  r = %g\n", nm,
                (double)u1[0], (double)u2[0], (double)u3[0], (double)r[0]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run([&] {
        call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
             y1.data(), 1, y2.data(), 1, y3.data(), 1,
             w1.data(), 1, w2.data(), 1, w3.data(), 1,
             u1.data(), 1, u2.data(), 1, u3.data(), 1, r.data(), 1);
    });
    err = std::fmax(max_rel_err(u1.data(), u1_ref.data(), n),
              std::fmax(max_rel_err(u2.data(), u2_ref.data(), n),
                        std::fmax(max_rel_err(u3.data(), u3_ref.data(), n),
                                  max_rel_err(r.data(), ref.data(), n))));
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// r = x.y, q = x.w
template <typename T, typename Call>
Result bench_3dotxy_dotxz_t(int n, const char *nm, double eps,
                            Call call)
{
    std::vector<T> x1(n), x2(n), x3(n), y1(n), y2(n), y3(n);
    std::vector<T> w1(n), w2(n), w3(n);
    std::vector<T> r(n), q(n);
    std::vector<ld> r_ref(n), q_ref(n);
    seeded(x1, 111);
    seeded(x2, 112);
    seeded(x3, 113);
    seeded(y1, 114);
    seeded(y2, 115);
    seeded(y3, 116);
    seeded(w1, 117);
    seeded(w2, 118);
    seeded(w3, 119);
    for (int i = 0; i < n; ++i) {
        r_ref[i] = (ld)x1[i] * y1[i] + (ld)x2[i] * y2[i]
                 + (ld)x3[i] * y3[i];
        q_ref[i] = (ld)x1[i] * w1[i] + (ld)x2[i] * w2[i]
                 + (ld)x3[i] * w3[i];
    }

    const double flops = 10.0, fcyc = 6.0;
    const char *eq = "r=x.y,q=x.w";
    double err;
#ifndef NDEBUG
    call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
         y1.data(), 1, y2.data(), 1, y3.data(), 1,
         w1.data(), 1, w2.data(), 1, w3.data(), 1,
         r.data(), 1, q.data(), 1);
    err = std::fmax(max_rel_err(r.data(), r_ref.data(), n),
              max_rel_err(q.data(), q_ref.data(), n));
    std::printf("%s[0]: x.y = r, x.w = q\n"
                "r = %g  q = %g\n", nm, (double)r[0], (double)q[0]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run([&] {
        call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
             y1.data(), 1, y2.data(), 1, y3.data(), 1,
             w1.data(), 1, w2.data(), 1, w3.data(), 1,
             r.data(), 1, q.data(), 1);
    });
    err = std::fmax(max_rel_err(r.data(), r_ref.data(), n),
              max_rel_err(q.data(), q_ref.data(), n));
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}


// ---------------- complex (c/z) kernels, interleaved storage ----------------
// T is the scalar type (float for c, double for z); arrays hold 2n elements,
// element i is the pair (v[2*i], v[2*i+1]) = (re, im). Scalars are T[2]
// {re, im} passed as the c/z void* scalar.

// y = alpha*x + y
template <typename T, typename Call>
Result bench_axpy_x(int n, const char *nm, double eps, const T alpha[2],
                    Call call)
{
    std::vector<T> x(2 * n), y0(2 * n), y(2 * n);
    std::vector<ld> ref(2 * n);
    seeded_c(x, 1);
    seeded_c(y0, 2);
    for (int i = 0; i < n; ++i) {
        ld p0, p1;
        cprod((ld)alpha[0], (ld)alpha[1], (ld)x[2 * i], (ld)x[2 * i + 1], p0, p1);
        ref[2 * i] = p0 + y0[2 * i];
        ref[2 * i + 1] = p1 + y0[2 * i + 1];
    }

    const double flops = 8.0, fcyc = 8.0;
    const char *eq = "y=a*x+y";
#ifndef NDEBUG
    std::memcpy(y.data(), y0.data(), 2 * n * sizeof(T));
    call(n, alpha, x.data(), 1, y.data(), 1);
    double err = max_rel_err_c(y.data(), ref.data(), n);
    std::printf("%s[0]: a*x[0] + y[0] = (%g %g)*(%g %g) + (%g %g) = (%g %g)\n", nm,
                (double)alpha[0], (double)alpha[1], (double)x[0], (double)x[1],
                (double)y0[0], (double)y0[1], (double)y[0], (double)y[1]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run_restore(
        [&] { std::memcpy(y.data(), y0.data(), 2 * n * sizeof(T)); },
        [&] { call(n, alpha, x.data(), 1, y.data(), 1); });
    double err = max_rel_err_c(y.data(), ref.data(), n);
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// r = q.*t + a
template <typename T, typename Call>
Result bench_xypa_x(int n, const char *nm, double eps, const T a[2],
                    Call call)
{
    std::vector<T> q(2 * n), t(2 * n), r(2 * n);
    std::vector<ld> ref(2 * n);
    seeded_c(q, 3);
    seeded_c(t, 4);
    for (int i = 0; i < n; ++i) {
        ld p0, p1;
        cprod((ld)q[2 * i], (ld)q[2 * i + 1], (ld)t[2 * i], (ld)t[2 * i + 1], p0, p1);
        ref[2 * i] = p0 + (ld)a[0];
        ref[2 * i + 1] = p1 + (ld)a[1];
    }

    const double flops = 8.0, fcyc = 8.0;
    const char *eq = "r=q.*t+a";
#ifndef NDEBUG
    call(n, a, q.data(), 1, t.data(), 1, r.data(), 1);
    double err = max_rel_err_c(r.data(), ref.data(), n);
    std::printf("%s[0]: q[0] .* t[0] + a = (%g %g).*(%g %g) + (%g %g) = (%g %g)\n",
                nm, (double)q[0], (double)q[1], (double)t[0], (double)t[1],
                (double)a[0], (double)a[1], (double)r[0], (double)r[1]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing tm = timed_run([&] { call(n, a, q.data(), 1, t.data(), 1, r.data(), 1); });
    double err = max_rel_err_c(r.data(), ref.data(), n);
    return make_result(nm, flops, fcyc, tm.t_run / tm.iters, n, tm.iters, err, eps, eq);
#endif
}

// r = x1y1 + x2y2 + x3y3 (bilinear products)
template <typename T, typename Call>
Result bench_3dot_x(int n, const char *nm, double eps, Call call)
{
    std::vector<T> x1(2 * n), x2(2 * n), x3(2 * n);
    std::vector<T> y1(2 * n), y2(2 * n), y3(2 * n), r(2 * n);
    std::vector<ld> ref(2 * n);
    seeded_c(x1, 11);
    seeded_c(x2, 12);
    seeded_c(x3, 13);
    seeded_c(y1, 14);
    seeded_c(y2, 15);
    seeded_c(y3, 16);
    for (int i = 0; i < n; ++i) {
        ld s0 = 0, s1 = 0, p0, p1;
        cprod((ld)x1[2 * i], (ld)x1[2 * i + 1], (ld)y1[2 * i], (ld)y1[2 * i + 1], p0, p1);
        s0 += p0; s1 += p1;
        cprod((ld)x2[2 * i], (ld)x2[2 * i + 1], (ld)y2[2 * i], (ld)y2[2 * i + 1], p0, p1);
        s0 += p0; s1 += p1;
        cprod((ld)x3[2 * i], (ld)x3[2 * i + 1], (ld)y3[2 * i], (ld)y3[2 * i + 1], p0, p1);
        s0 += p0; s1 += p1;
        ref[2 * i] = s0;
        ref[2 * i + 1] = s1;
    }

    const double flops = 22.0, fcyc = 22.0;
    const char *eq = "r=x1y1+x2y2+x3y3";
#ifndef NDEBUG
    call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
         y1.data(), 1, y2.data(), 1, y3.data(), 1, r.data(), 1);
    double err = max_rel_err_c(r.data(), ref.data(), n);
    std::printf("%s[0]: r = x1y1+x2y2+x3y3 = (%g %g)\n", nm,
                (double)r[0], (double)r[1]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run([&] {
        call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
             y1.data(), 1, y2.data(), 1, y3.data(), 1, r.data(), 1);
    });
    double err = max_rel_err_c(r.data(), ref.data(), n);
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// w = x.*y (bilinear products)
template <typename T, typename Call>
Result bench_3had_x(int n, const char *nm, double eps, Call call)
{
    std::vector<T> x1(2 * n), x2(2 * n), x3(2 * n);
    std::vector<T> y1(2 * n), y2(2 * n), y3(2 * n);
    std::vector<T> w1(2 * n), w2(2 * n), w3(2 * n);
    std::vector<ld> w1_ref(2 * n), w2_ref(2 * n), w3_ref(2 * n);
    seeded_c(x1, 21);
    seeded_c(x2, 22);
    seeded_c(x3, 23);
    seeded_c(y1, 24);
    seeded_c(y2, 25);
    seeded_c(y3, 26);
    for (int i = 0; i < n; ++i) {
        cprod((ld)x1[2 * i], (ld)x1[2 * i + 1], (ld)y1[2 * i], (ld)y1[2 * i + 1],
              w1_ref[2 * i], w1_ref[2 * i + 1]);
        cprod((ld)x2[2 * i], (ld)x2[2 * i + 1], (ld)y2[2 * i], (ld)y2[2 * i + 1],
              w2_ref[2 * i], w2_ref[2 * i + 1]);
        cprod((ld)x3[2 * i], (ld)x3[2 * i + 1], (ld)y3[2 * i], (ld)y3[2 * i + 1],
              w3_ref[2 * i], w3_ref[2 * i + 1]);
    }

    const double flops = 18.0, fcyc = 18.0;
    const char *eq = "w=x.*y";
#ifndef NDEBUG
    call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
         y1.data(), 1, y2.data(), 1, y3.data(), 1,
         w1.data(), 1, w2.data(), 1, w3.data(), 1);
    double err = std::fmax(max_rel_err_c(w1.data(), w1_ref.data(), n),
                   std::fmax(max_rel_err_c(w2.data(), w2_ref.data(), n),
                             max_rel_err_c(w3.data(), w3_ref.data(), n)));
    std::printf("%s[0]: w = x.*y\n"
                "w1 = (%g %g)  w2 = (%g %g)  w3 = (%g %g)\n", nm,
                (double)w1[0], (double)w1[1], (double)w2[0], (double)w2[1],
                (double)w3[0], (double)w3[1]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run([&] {
        call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
             y1.data(), 1, y2.data(), 1, y3.data(), 1,
             w1.data(), 1, w2.data(), 1, w3.data(), 1);
    });
    double err = std::fmax(max_rel_err_c(w1.data(), w1_ref.data(), n),
                   std::fmax(max_rel_err_c(w2.data(), w2_ref.data(), n),
                             max_rel_err_c(w3.data(), w3_ref.data(), n)));
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// r = x1^2 + x2^2 + x3^2 (bilinear squares)
template <typename T, typename Call>
Result bench_3sqr_x(int n, const char *nm, double eps, Call call)
{
    std::vector<T> x1(2 * n), x2(2 * n), x3(2 * n), r(2 * n);
    std::vector<ld> ref(2 * n);
    seeded_c(x1, 31);
    seeded_c(x2, 32);
    seeded_c(x3, 33);
    for (int i = 0; i < n; ++i) {
        ld s0 = 0, s1 = 0, p0, p1;
        cprod((ld)x1[2 * i], (ld)x1[2 * i + 1], (ld)x1[2 * i], (ld)x1[2 * i + 1], p0, p1);
        s0 += p0; s1 += p1;
        cprod((ld)x2[2 * i], (ld)x2[2 * i + 1], (ld)x2[2 * i], (ld)x2[2 * i + 1], p0, p1);
        s0 += p0; s1 += p1;
        cprod((ld)x3[2 * i], (ld)x3[2 * i + 1], (ld)x3[2 * i], (ld)x3[2 * i + 1], p0, p1);
        s0 += p0; s1 += p1;
        ref[2 * i] = s0;
        ref[2 * i + 1] = s1;
    }

    const double flops = 22.0, fcyc = 22.0;
    const char *eq = "r=x1^2+x2^2+x3^2";
#ifndef NDEBUG
    call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1, r.data(), 1);
    double err = max_rel_err_c(r.data(), ref.data(), n);
    std::printf("%s[0]: r = x1^2+x2^2+x3^2 = (%g %g)\n", nm,
                (double)r[0], (double)r[1]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run([&] {
        call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1, r.data(), 1);
    });
    double err = max_rel_err_c(r.data(), ref.data(), n);
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// r = sqrt(q.*q), complex hypotenuse as implemented in the kernel
template <typename T, typename Call>
Result bench_1norm_x(int n, const char *nm, double eps, Call call)
{
    std::vector<T> q(2 * n), r(2 * n);
    std::vector<ld> ref(2 * n);
    seeded_c(q, 41);
    for (int i = 0; i < n; ++i) {
        ld qr = q[2 * i], qi = q[2 * i + 1];
        ld p0 = qr * qr - qi * qi;
        ld p1 = 2.0L * qr * qi;
        ld m = sqrtl(p0 * p0 + p1 * p1);
        ld rr = sqrtl((m + p0) / 2.0L);
        ld ri = (p1 >= 0) ? sqrtl((m - p0) / 2.0L) : -sqrtl((m - p0) / 2.0L);
        ref[2 * i] = rr;
        ref[2 * i + 1] = ri;
    }

    const double flops = 16.0, fcyc = 16.0;
    const char *eq = "r=sqrt(q.q)";
#ifndef NDEBUG
    call(n, q.data(), 1, r.data(), 1);
    double err = max_rel_err_c(r.data(), ref.data(), n);
    std::printf("%s[0]: sqrt(q[0] * q[0]) = sqrt((%g %g)*(%g %g)) = (%g %g)\n", nm,
                (double)q[0], (double)q[1], (double)q[0], (double)q[1],
                (double)r[0], (double)r[1]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run([&] { call(n, q.data(), 1, r.data(), 1); });
    double err = max_rel_err_c(r.data(), ref.data(), n);
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// w = x^y
template <typename T, typename Call>
Result bench_3cross_x(int n, const char *nm, double eps, Call call)
{
    std::vector<T> x1(2 * n), x2(2 * n), x3(2 * n);
    std::vector<T> y1(2 * n), y2(2 * n), y3(2 * n);
    std::vector<T> w1(2 * n), w2(2 * n), w3(2 * n);
    std::vector<ld> w1_ref(2 * n), w2_ref(2 * n), w3_ref(2 * n);
    seeded_c(x1, 51);
    seeded_c(x2, 52);
    seeded_c(x3, 53);
    seeded_c(y1, 54);
    seeded_c(y2, 55);
    seeded_c(y3, 56);
    for (int i = 0; i < n; ++i)
        ccross(x1.data(), x2.data(), x3.data(),
               y1.data(), y2.data(), y3.data(),
               w1_ref.data(), w2_ref.data(), w3_ref.data(), i);

    const double flops = 42.0, fcyc = 42.0;
    const char *eq = "w=x^y";
#ifndef NDEBUG
    call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
         y1.data(), 1, y2.data(), 1, y3.data(), 1,
         w1.data(), 1, w2.data(), 1, w3.data(), 1);
    double err = std::fmax(max_rel_err_c(w1.data(), w1_ref.data(), n),
                   std::fmax(max_rel_err_c(w2.data(), w2_ref.data(), n),
                             max_rel_err_c(w3.data(), w3_ref.data(), n)));
    std::printf("%s[0]: w = x^y\n"
                "w1 = (%g %g)  w2 = (%g %g)  w3 = (%g %g)\n", nm,
                (double)w1[0], (double)w1[1], (double)w2[0], (double)w2[1],
                (double)w3[0], (double)w3[1]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run([&] {
        call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
             y1.data(), 1, y2.data(), 1, y3.data(), 1,
             w1.data(), 1, w2.data(), 1, w3.data(), 1);
    });
    double err = std::fmax(max_rel_err_c(w1.data(), w1_ref.data(), n),
                   std::fmax(max_rel_err_c(w2.data(), w2_ref.data(), n),
                             max_rel_err_c(w3.data(), w3_ref.data(), n)));
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// w = a*(x^y)
template <typename T, typename Call>
Result bench_3crossscal_x(int n, const char *nm, double eps, const T a[2],
                          Call call)
{
    std::vector<T> x1(2 * n), x2(2 * n), x3(2 * n);
    std::vector<T> y1(2 * n), y2(2 * n), y3(2 * n);
    std::vector<T> w1(2 * n), w2(2 * n), w3(2 * n);
    std::vector<ld> w1_ref(2 * n), w2_ref(2 * n), w3_ref(2 * n);
    seeded_c(x1, 61);
    seeded_c(x2, 62);
    seeded_c(x3, 63);
    seeded_c(y1, 64);
    seeded_c(y2, 65);
    seeded_c(y3, 66);
    for (int i = 0; i < n; ++i) {
        ccross(x1.data(), x2.data(), x3.data(),
               y1.data(), y2.data(), y3.data(),
               w1_ref.data(), w2_ref.data(), w3_ref.data(), i);
        ld p0, p1;
        cprod((ld)a[0], (ld)a[1], w1_ref[2 * i], w1_ref[2 * i + 1], p0, p1);
        w1_ref[2 * i] = p0;
        w1_ref[2 * i + 1] = p1;
        cprod((ld)a[0], (ld)a[1], w2_ref[2 * i], w2_ref[2 * i + 1], p0, p1);
        w2_ref[2 * i] = p0;
        w2_ref[2 * i + 1] = p1;
        cprod((ld)a[0], (ld)a[1], w3_ref[2 * i], w3_ref[2 * i + 1], p0, p1);
        w3_ref[2 * i] = p0;
        w3_ref[2 * i + 1] = p1;
    }

    const double flops = 60.0, fcyc = 60.0;
    const char *eq = "w=a*(x^y)";
#ifndef NDEBUG
    call(n, a, x1.data(), 1, x2.data(), 1, x3.data(), 1,
         y1.data(), 1, y2.data(), 1, y3.data(), 1,
         w1.data(), 1, w2.data(), 1, w3.data(), 1);
    double err = std::fmax(max_rel_err_c(w1.data(), w1_ref.data(), n),
                   std::fmax(max_rel_err_c(w2.data(), w2_ref.data(), n),
                             max_rel_err_c(w3.data(), w3_ref.data(), n)));
    std::printf("%s[0]: w = a*(x^y), a = (%g %g)\n"
                "w1 = (%g %g)  w2 = (%g %g)  w3 = (%g %g)\n", nm,
                (double)a[0], (double)a[1],
                (double)w1[0], (double)w1[1], (double)w2[0], (double)w2[1],
                (double)w3[0], (double)w3[1]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run([&] {
        call(n, a, x1.data(), 1, x2.data(), 1, x3.data(), 1,
             y1.data(), 1, y2.data(), 1, y3.data(), 1,
             w1.data(), 1, w2.data(), 1, w3.data(), 1);
    });
    double err = std::fmax(max_rel_err_c(w1.data(), w1_ref.data(), n),
                   std::fmax(max_rel_err_c(w2.data(), w2_ref.data(), n),
                             max_rel_err_c(w3.data(), w3_ref.data(), n)));
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// r = (x^y).w
template <typename T, typename Call>
Result bench_3crossdot_x(int n, const char *nm, double eps, Call call)
{
    std::vector<T> x1(2 * n), x2(2 * n), x3(2 * n);
    std::vector<T> y1(2 * n), y2(2 * n), y3(2 * n);
    std::vector<T> w1(2 * n), w2(2 * n), w3(2 * n), r(2 * n);
    std::vector<ld> ref(2 * n);
    seeded_c(x1, 71);
    seeded_c(x2, 72);
    seeded_c(x3, 73);
    seeded_c(y1, 74);
    seeded_c(y2, 75);
    seeded_c(y3, 76);
    seeded_c(w1, 77);
    seeded_c(w2, 78);
    seeded_c(w3, 79);
    for (int i = 0; i < n; ++i) {
        ld c1r, c1i, c2r, c2i, c3r, c3i;
        {
            ld p0, p1, q0, q1;
            cprod((ld)x2[2 * i], (ld)x2[2 * i + 1], (ld)y3[2 * i], (ld)y3[2 * i + 1], p0, p1);
            cprod((ld)x3[2 * i], (ld)x3[2 * i + 1], (ld)y2[2 * i], (ld)y2[2 * i + 1], q0, q1);
            c1r = p0 - q0; c1i = p1 - q1;
            cprod((ld)x3[2 * i], (ld)x3[2 * i + 1], (ld)y1[2 * i], (ld)y1[2 * i + 1], p0, p1);
            cprod((ld)x1[2 * i], (ld)x1[2 * i + 1], (ld)y3[2 * i], (ld)y3[2 * i + 1], q0, q1);
            c2r = p0 - q0; c2i = p1 - q1;
            cprod((ld)x1[2 * i], (ld)x1[2 * i + 1], (ld)y2[2 * i], (ld)y2[2 * i + 1], p0, p1);
            cprod((ld)x2[2 * i], (ld)x2[2 * i + 1], (ld)y1[2 * i], (ld)y1[2 * i + 1], q0, q1);
            c3r = p0 - q0; c3i = p1 - q1;
        }
        ld s0 = 0, s1 = 0, p0, p1;
        cprod(c1r, c1i, (ld)w1[2 * i], (ld)w1[2 * i + 1], p0, p1);
        s0 += p0; s1 += p1;
        cprod(c2r, c2i, (ld)w2[2 * i], (ld)w2[2 * i + 1], p0, p1);
        s0 += p0; s1 += p1;
        cprod(c3r, c3i, (ld)w3[2 * i], (ld)w3[2 * i + 1], p0, p1);
        s0 += p0; s1 += p1;
        ref[2 * i] = s0;
        ref[2 * i + 1] = s1;
    }

    const double flops = 64.0, fcyc = 64.0;
    const char *eq = "r=(x^y).w";
#ifndef NDEBUG
    call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
         y1.data(), 1, y2.data(), 1, y3.data(), 1,
         w1.data(), 1, w2.data(), 1, w3.data(), 1, r.data(), 1);
    double err = max_rel_err_c(r.data(), ref.data(), n);
    std::printf("%s[0]: r = (x^y).w = (%g %g)\n", nm, (double)r[0], (double)r[1]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run([&] {
        call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
             y1.data(), 1, y2.data(), 1, y3.data(), 1,
             w1.data(), 1, w2.data(), 1, w3.data(), 1, r.data(), 1);
    });
    double err = max_rel_err_c(r.data(), ref.data(), n);
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// r = (x^y).(x^y)
template <typename T, typename Call>
Result bench_3crosssqr_x(int n, const char *nm, double eps, Call call)
{
    std::vector<T> x1(2 * n), x2(2 * n), x3(2 * n);
    std::vector<T> y1(2 * n), y2(2 * n), y3(2 * n), r(2 * n);
    std::vector<ld> ref(2 * n);
    seeded_c(x1, 81);
    seeded_c(x2, 82);
    seeded_c(x3, 83);
    seeded_c(y1, 84);
    seeded_c(y2, 85);
    seeded_c(y3, 86);
    for (int i = 0; i < n; ++i) {
        ld c1r, c1i, c2r, c2i, c3r, c3i, p0, p1, q0, q1, s0, s1;
        cprod((ld)x2[2 * i], (ld)x2[2 * i + 1], (ld)y3[2 * i], (ld)y3[2 * i + 1], p0, p1);
        cprod((ld)x3[2 * i], (ld)x3[2 * i + 1], (ld)y2[2 * i], (ld)y2[2 * i + 1], q0, q1);
        c1r = p0 - q0; c1i = p1 - q1;
        cprod((ld)x3[2 * i], (ld)x3[2 * i + 1], (ld)y1[2 * i], (ld)y1[2 * i + 1], p0, p1);
        cprod((ld)x1[2 * i], (ld)x1[2 * i + 1], (ld)y3[2 * i], (ld)y3[2 * i + 1], q0, q1);
        c2r = p0 - q0; c2i = p1 - q1;
        cprod((ld)x1[2 * i], (ld)x1[2 * i + 1], (ld)y2[2 * i], (ld)y2[2 * i + 1], p0, p1);
        cprod((ld)x2[2 * i], (ld)x2[2 * i + 1], (ld)y1[2 * i], (ld)y1[2 * i + 1], q0, q1);
        c3r = p0 - q0; c3i = p1 - q1;
        s0 = 0; s1 = 0;
        cprod(c1r, c1i, c1r, c1i, p0, p1);
        s0 += p0; s1 += p1;
        cprod(c2r, c2i, c2r, c2i, p0, p1);
        s0 += p0; s1 += p1;
        cprod(c3r, c3i, c3r, c3i, p0, p1);
        s0 += p0; s1 += p1;
        ref[2 * i] = s0;
        ref[2 * i + 1] = s1;
    }

    const double flops = 64.0, fcyc = 64.0;
    const char *eq = "r=(x^y).(x^y)";
#ifndef NDEBUG
    call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
         y1.data(), 1, y2.data(), 1, y3.data(), 1, r.data(), 1);
    double err = max_rel_err_c(r.data(), ref.data(), n);
    std::printf("%s[0]: r = (x^y).(x^y) = (%g %g)\n", nm, (double)r[0], (double)r[1]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run([&] {
        call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
             y1.data(), 1, y2.data(), 1, y3.data(), 1, r.data(), 1);
    });
    double err = max_rel_err_c(r.data(), ref.data(), n);
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// u = x^y, v = x^w
template <typename T, typename Call>
Result bench_3crossxy_crossxz_x(int n, const char *nm, double eps, Call call)
{
    std::vector<T> x1(2 * n), x2(2 * n), x3(2 * n);
    std::vector<T> y1(2 * n), y2(2 * n), y3(2 * n);
    std::vector<T> w1(2 * n), w2(2 * n), w3(2 * n);
    std::vector<T> u1(2 * n), u2(2 * n), u3(2 * n), v1(2 * n), v2(2 * n), v3(2 * n);
    std::vector<ld> u1r(2 * n), u2r(2 * n), u3r(2 * n);
    std::vector<ld> v1r(2 * n), v2r(2 * n), v3r(2 * n);
    seeded_c(x1, 91);
    seeded_c(x2, 92);
    seeded_c(x3, 93);
    seeded_c(y1, 94);
    seeded_c(y2, 95);
    seeded_c(y3, 96);
    seeded_c(w1, 97);
    seeded_c(w2, 98);
    seeded_c(w3, 99);
    for (int i = 0; i < n; ++i) {
        ccross(x1.data(), x2.data(), x3.data(),
               y1.data(), y2.data(), y3.data(),
               u1r.data(), u2r.data(), u3r.data(), i);
        ccross(x1.data(), x2.data(), x3.data(),
               w1.data(), w2.data(), w3.data(),
               v1r.data(), v2r.data(), v3r.data(), i);
    }

    const double flops = 84.0, fcyc = 84.0;
    const char *eq = "u=x^y,v=x^w";
#ifndef NDEBUG
    call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
         y1.data(), 1, y2.data(), 1, y3.data(), 1,
         w1.data(), 1, w2.data(), 1, w3.data(), 1,
         u1.data(), 1, u2.data(), 1, u3.data(), 1,
         v1.data(), 1, v2.data(), 1, v3.data(), 1);
    double err = std::fmax(max_rel_err_c(u1.data(), u1r.data(), n),
                   std::fmax(max_rel_err_c(u2.data(), u2r.data(), n),
                             std::fmax(max_rel_err_c(u3.data(), u3r.data(), n),
                                       std::fmax(max_rel_err_c(v1.data(), v1r.data(), n),
                                                 std::fmax(max_rel_err_c(v2.data(), v2r.data(), n),
                                                           max_rel_err_c(v3.data(), v3r.data(), n))))));
    std::printf("%s[0]: u = x^y, v = x^w\n"
                "u1 = (%g %g)  v1 = (%g %g)\n", nm,
                (double)u1[0], (double)u1[1], (double)v1[0], (double)v1[1]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run([&] {
        call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
             y1.data(), 1, y2.data(), 1, y3.data(), 1,
             w1.data(), 1, w2.data(), 1, w3.data(), 1,
             u1.data(), 1, u2.data(), 1, u3.data(), 1,
             v1.data(), 1, v2.data(), 1, v3.data(), 1);
    });
    double err = std::fmax(max_rel_err_c(u1.data(), u1r.data(), n),
                   std::fmax(max_rel_err_c(u2.data(), u2r.data(), n),
                             std::fmax(max_rel_err_c(u3.data(), u3r.data(), n),
                                       std::fmax(max_rel_err_c(v1.data(), v1r.data(), n),
                                                 std::fmax(max_rel_err_c(v2.data(), v2r.data(), n),
                                                           max_rel_err_c(v3.data(), v3r.data(), n))))));
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// u = x^y, r = x.w
template <typename T, typename Call>
Result bench_3crossxy_dotxz_x(int n, const char *nm, double eps, Call call)
{
    std::vector<T> x1(2 * n), x2(2 * n), x3(2 * n);
    std::vector<T> y1(2 * n), y2(2 * n), y3(2 * n);
    std::vector<T> w1(2 * n), w2(2 * n), w3(2 * n);
    std::vector<T> u1(2 * n), u2(2 * n), u3(2 * n), r(2 * n);
    std::vector<ld> u1r(2 * n), u2r(2 * n), u3r(2 * n), ref(2 * n);
    seeded_c(x1, 101);
    seeded_c(x2, 102);
    seeded_c(x3, 103);
    seeded_c(y1, 104);
    seeded_c(y2, 105);
    seeded_c(y3, 106);
    seeded_c(w1, 107);
    seeded_c(w2, 108);
    seeded_c(w3, 109);
    for (int i = 0; i < n; ++i) {
        ccross(x1.data(), x2.data(), x3.data(),
               y1.data(), y2.data(), y3.data(),
               u1r.data(), u2r.data(), u3r.data(), i);
        cdot3(x1.data(), x2.data(), x3.data(),
              w1.data(), w2.data(), w3.data(),
              ref[2 * i], ref[2 * i + 1], i);
    }

    const double flops = 64.0, fcyc = 64.0;
    const char *eq = "u=x^y,r=x.w";
#ifndef NDEBUG
    call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
         y1.data(), 1, y2.data(), 1, y3.data(), 1,
         w1.data(), 1, w2.data(), 1, w3.data(), 1,
         u1.data(), 1, u2.data(), 1, u3.data(), 1, r.data(), 1);
    double err = std::fmax(max_rel_err_c(u1.data(), u1r.data(), n),
                   std::fmax(max_rel_err_c(u2.data(), u2r.data(), n),
                             std::fmax(max_rel_err_c(u3.data(), u3r.data(), n),
                                       max_rel_err_c(r.data(), ref.data(), n))));
    std::printf("%s[0]: u = x^y, r = x.w\n"
                "u1 = (%g %g)  r = (%g %g)\n", nm,
                (double)u1[0], (double)u1[1], (double)r[0], (double)r[1]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run([&] {
        call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
             y1.data(), 1, y2.data(), 1, y3.data(), 1,
             w1.data(), 1, w2.data(), 1, w3.data(), 1,
             u1.data(), 1, u2.data(), 1, u3.data(), 1, r.data(), 1);
    });
    double err = std::fmax(max_rel_err_c(u1.data(), u1r.data(), n),
                   std::fmax(max_rel_err_c(u2.data(), u2r.data(), n),
                             std::fmax(max_rel_err_c(u3.data(), u3r.data(), n),
                                       max_rel_err_c(r.data(), ref.data(), n))));
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// r = x.y, q = x.w
template <typename T, typename Call>
Result bench_3dotxy_dotxz_x(int n, const char *nm, double eps, Call call)
{
    std::vector<T> x1(2 * n), x2(2 * n), x3(2 * n);
    std::vector<T> y1(2 * n), y2(2 * n), y3(2 * n);
    std::vector<T> w1(2 * n), w2(2 * n), w3(2 * n);
    std::vector<T> r(2 * n), q(2 * n);
    std::vector<ld> rr(2 * n), qr(2 * n);
    seeded_c(x1, 111);
    seeded_c(x2, 112);
    seeded_c(x3, 113);
    seeded_c(y1, 114);
    seeded_c(y2, 115);
    seeded_c(y3, 116);
    seeded_c(w1, 117);
    seeded_c(w2, 118);
    seeded_c(w3, 119);
    for (int i = 0; i < n; ++i) {
        cdot3(x1.data(), x2.data(), x3.data(),
              y1.data(), y2.data(), y3.data(),
              rr[2 * i], rr[2 * i + 1], i);
        cdot3(x1.data(), x2.data(), x3.data(),
              w1.data(), w2.data(), w3.data(),
              qr[2 * i], qr[2 * i + 1], i);
    }

    const double flops = 44.0, fcyc = 44.0;
    const char *eq = "r=x.y,q=x.w";
#ifndef NDEBUG
    call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
         y1.data(), 1, y2.data(), 1, y3.data(), 1,
         w1.data(), 1, w2.data(), 1, w3.data(), 1,
         r.data(), 1, q.data(), 1);
    double err = std::fmax(max_rel_err_c(r.data(), rr.data(), n),
                   max_rel_err_c(q.data(), qr.data(), n));
    std::printf("%s[0]: r = x.y, q = x.w\n"
                "r = (%g %g)  q = (%g %g)\n", nm,
                (double)r[0], (double)r[1], (double)q[0], (double)q[1]);
    return {nm, flops, fcyc, 0.0, 0.0, 0.0, err, 0, err <= eps, eq};
#else
    Timing t = timed_run([&] {
        call(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
             y1.data(), 1, y2.data(), 1, y3.data(), 1,
             w1.data(), 1, w2.data(), 1, w3.data(), 1,
             r.data(), 1, q.data(), 1);
    });
    double err = std::fmax(max_rel_err_c(r.data(), rr.data(), n),
                   max_rel_err_c(q.data(), qr.data(), n));
    return make_result(nm, flops, fcyc, t.t_run / t.iters, n, t.iters, err, eps, eq);
#endif
}

// ---------------- precision dispatchers ----------------

Result bench_axpy(int n, char prec)
{
    switch (prec) {
    case 's': return bench_axpy_t<float>(n, "saxpy", kEpsF, 1.5f, cblas_saxpy);
    case 'd': return bench_axpy_t<double>(n, "daxpy", kEps, 1.5, cblas_daxpy);
    case 'c': { float a[2] = {1.5f, 0.5f};
                return bench_axpy_x<float>(n, "caxpy", kEpsF, a, cblas_caxpy); }
    default:  { double a[2] = {1.5, 0.5};
                return bench_axpy_x<double>(n, "zaxpy", kEps, a, cblas_zaxpy); }
    }
}

Result bench_xypa(int n, char prec)
{
    switch (prec) {
    case 's': return bench_xypa_t<float>(n, "s1xypa", kEpsF, 0.25f, cblas_s1xypa);
    case 'd': return bench_xypa_t<double>(n, "d1xypa", kEps, 0.25, cblas_d1xypa);
    case 'c': { float a[2] = {0.25f, 0.1f};
                return bench_xypa_x<float>(n, "c1xypa", kEpsF, a, cblas_c1xypa); }
    default:  { double a[2] = {0.25, 0.1};
                return bench_xypa_x<double>(n, "z1xypa", kEps, a, cblas_z1xypa); }
    }
}

Result bench_3dot(int n, char prec)
{
    switch (prec) {
    case 's': return bench_3dot_t<float>(n, "s3dot", kEpsF, cblas_s3dot);
    case 'd': return bench_3dot_t<double>(n, "d3dot", kEps, cblas_d3dot);
    case 'c': return bench_3dot_x<float>(n, "c3dot", kEpsF, cblas_c3dot);
    default:  return bench_3dot_x<double>(n, "z3dot", kEps, cblas_z3dot);
    }
}

Result bench_3had(int n, char prec)
{
    switch (prec) {
    case 's': return bench_3had_t<float>(n, "s3had", kEpsF, cblas_s3had);
    case 'd': return bench_3had_t<double>(n, "d3had", kEps, cblas_d3had);
    case 'c': return bench_3had_x<float>(n, "c3had", kEpsF, cblas_c3had);
    default:  return bench_3had_x<double>(n, "z3had", kEps, cblas_z3had);
    }
}

Result bench_3sqr(int n, char prec)
{
    switch (prec) {
    case 's': return bench_3sqr_t<float>(n, "s3sqr", kEpsF, cblas_s3sqr);
    case 'd': return bench_3sqr_t<double>(n, "d3sqr", kEps, cblas_d3sqr);
    case 'c': return bench_3sqr_x<float>(n, "c3sqr", kEpsF, cblas_c3sqr);
    default:  return bench_3sqr_x<double>(n, "z3sqr", kEps, cblas_z3sqr);
    }
}

Result bench_1norm(int n, char prec)
{
    switch (prec) {
    case 's': return bench_1norm_t<float>(n, "s1norm", kEpsF, cblas_s1norm);
    case 'd': return bench_1norm_t<double>(n, "d1norm", kEps, cblas_d1norm);
    case 'c': return bench_1norm_x<float>(n, "c1norm", kEpsF, cblas_c1norm);
    default:  return bench_1norm_x<double>(n, "z1norm", kEps, cblas_z1norm);
    }
}

Result bench_3cross(int n, char prec)
{
    switch (prec) {
    case 's': return bench_3cross_t<float>(n, "s3cross", kEpsF, cblas_s3cross);
    case 'd': return bench_3cross_t<double>(n, "d3cross", kEps, cblas_d3cross);
    case 'c': return bench_3cross_x<float>(n, "c3cross", kEpsF, cblas_c3cross);
    default:  return bench_3cross_x<double>(n, "z3cross", kEps, cblas_z3cross);
    }
}

Result bench_3crossscal(int n, char prec)
{
    switch (prec) {
    case 's': return bench_3crossscal_t<float>(n, "s3crossscal", kEpsF, 1.5f,
                                              cblas_s3crossscal);
    case 'd': return bench_3crossscal_t<double>(n, "d3crossscal", kEps, 1.5,
                                                cblas_d3crossscal);
    case 'c': { float a[2] = {1.5f, 0.5f};
                return bench_3crossscal_x<float>(n, "c3crossscal", kEpsF, a,
                                                 cblas_c3crossscal); }
    default:  { double a[2] = {1.5, 0.5};
                return bench_3crossscal_x<double>(n, "z3crossscal", kEps, a,
                                                  cblas_z3crossscal); }
    }
}

Result bench_3crossdot(int n, char prec)
{
    switch (prec) {
    case 's': return bench_3crossdot_t<float>(n, "s3crossdot", kEpsF, cblas_s3crossdot);
    case 'd': return bench_3crossdot_t<double>(n, "d3crossdot", kEps, cblas_d3crossdot);
    case 'c': return bench_3crossdot_x<float>(n, "c3crossdot", kEpsF, cblas_c3crossdot);
    default:  return bench_3crossdot_x<double>(n, "z3crossdot", kEps, cblas_z3crossdot);
    }
}

Result bench_3crosssqr(int n, char prec)
{
    switch (prec) {
    case 's': return bench_3crosssqr_t<float>(n, "s3crosssqr", kEpsF, cblas_s3crosssqr);
    case 'd': return bench_3crosssqr_t<double>(n, "d3crosssqr", kEps, cblas_d3crosssqr);
    case 'c': return bench_3crosssqr_x<float>(n, "c3crosssqr", kEpsF, cblas_c3crosssqr);
    default:  return bench_3crosssqr_x<double>(n, "z3crosssqr", kEps, cblas_z3crosssqr);
    }
}

Result bench_3crossxy_crossxz(int n, char prec)
{
    switch (prec) {
    case 's': return bench_3crossxy_crossxz_t<float>(n, "s3crossxy_crossxz", kEpsF,
                                                     cblas_s3crossxy_crossxz);
    case 'd': return bench_3crossxy_crossxz_t<double>(n, "d3crossxy_crossxz", kEps,
                                                      cblas_d3crossxy_crossxz);
    case 'c': return bench_3crossxy_crossxz_x<float>(n, "c3crossxy_crossxz", kEpsF,
                                                     cblas_c3crossxy_crossxz);
    default:  return bench_3crossxy_crossxz_x<double>(n, "z3crossxy_crossxz", kEps,
                                                      cblas_z3crossxy_crossxz);
    }
}

Result bench_3crossxy_dotxz(int n, char prec)
{
    switch (prec) {
    case 's': return bench_3crossxy_dotxz_t<float>(n, "s3crossxy_dotxz", kEpsF,
                                                   cblas_s3crossxy_dotxz);
    case 'd': return bench_3crossxy_dotxz_t<double>(n, "d3crossxy_dotxz", kEps,
                                                    cblas_d3crossxy_dotxz);
    case 'c': return bench_3crossxy_dotxz_x<float>(n, "c3crossxy_dotxz", kEpsF,
                                                   cblas_c3crossxy_dotxz);
    default:  return bench_3crossxy_dotxz_x<double>(n, "z3crossxy_dotxz", kEps,
                                                    cblas_z3crossxy_dotxz);
    }
}

Result bench_3dotxy_dotxz(int n, char prec)
{
    switch (prec) {
    case 's': return bench_3dotxy_dotxz_t<float>(n, "s3dotxy_dotxz", kEpsF,
                                                 cblas_s3dotxy_dotxz);
    case 'd': return bench_3dotxy_dotxz_t<double>(n, "d3dotxy_dotxz", kEps,
                                                  cblas_d3dotxy_dotxz);
    case 'c': return bench_3dotxy_dotxz_x<float>(n, "c3dotxy_dotxz", kEpsF,
                                                 cblas_c3dotxy_dotxz);
    default:  return bench_3dotxy_dotxz_x<double>(n, "z3dotxy_dotxz", kEps,
                                                  cblas_z3dotxy_dotxz);
    }
}

} // namespace

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

    std::vector<Result> results;
    results.push_back(bench_axpy(n, prec));
    results.push_back(bench_xypa(n, prec));
    results.push_back(bench_3dot(n, prec));
    results.push_back(bench_3had(n, prec));
    results.push_back(bench_3sqr(n, prec));
    results.push_back(bench_1norm(n, prec));
    results.push_back(bench_3cross(n, prec));
    results.push_back(bench_3crossscal(n, prec));
    results.push_back(bench_3crossdot(n, prec));
    results.push_back(bench_3crosssqr(n, prec));
    results.push_back(bench_3crossxy_crossxz(n, prec));
    results.push_back(bench_3crossxy_dotxz(n, prec));
    results.push_back(bench_3dotxy_dotxz(n, prec));

    std::printf("%-19s %10s %12s %12s %6s %13s %7s  %s  %s\n",
                "kernel", "ms/run", "GFLOP/s", "Melem/s", "F/cyc",
                "max_rel_err", "iters", "status", "equation");
    for (const Result &res : results)
        std::printf("%-19s %10.3f %12.3f %12.3f %6.0f %13.3e %7d  %s  %s\n",
                    res.name.c_str(), res.ms_per_run, res.gflops, res.meps,
                    res.fcyc, res.max_rel_err, res.iters,
                    res.ok ? "OK" : "FAIL", res.eq.c_str());
    return 0;
}
