// Benchmark for memory-bound, element-wise BLAS L1 kernels.
//
// usage: benchmark [n]    (vector size, default 1000)
//
// Each kernel is first checked against a known result (independently
// computed reference in long double), then timed over repeated runs.
// In a non-NDEBUG build (CMake debug mode) each kernel is run once and
// the first element of its expression is printed for hand verification;
// no timing is done. With NDEBUG the performance table is printed.
// Add new kernels as new bench_*() functions and list them in main().
//
// Links against a pre-built OpenBLAS; this project does not build OpenBLAS.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cblas.h"

using Clock = std::chrono::steady_clock;

namespace {

constexpr double kMinTime = 0.5;   // seconds of runtime per kernel
constexpr int kMinIters = 3;
constexpr double kMaxTime = 10.0;  // cap for slow machines
constexpr double kEps = 1e-12;     // relative tolerance vs. the reference

struct Result {
    std::string name;
    double flops;       // FLOPs per element
    double ms_per_run;
    double gflops;
    double meps;        // million elements / second
    double max_rel_err; // vs. the long double reference
    int iters;
    bool ok;
};

Result make_result(const std::string &name, double flops_per_elem,
                   double per_run, int n, int iters, double err)
{
    return {name, flops_per_elem, per_run * 1e3,
            flops_per_elem * n / per_run / 1e9, n / per_run / 1e6,
            err, iters, err <= kEps};
}

double max_rel_err(const double *v, const long double *ref, int n)
{
    double m = 0.0;
    for (int i = 0; i < n; ++i) {
        double r = (double)ref[i];
        double e = std::fabs(v[i] - r);
        m = std::fmax(m, e / std::fmax(1.0, std::fabs(r)));
    }
    return m;
}

void seeded(std::vector<double> &v, int seed)
{
    for (int i = 0; i < (int)v.size(); ++i)
        v[i] = 1e-3 * ((i + seed) % 997 - 498);
}

// y = alpha*x + y  (in place on y, restored from y0 before every run)
Result bench_axpy(int n)
{
    const double alpha = 1.5;
    std::vector<double> x(n), y0(n), y(n);
    std::vector<long double> ref(n);
    seeded(x, 1);
    seeded(y0, 2);
    for (int i = 0; i < n; ++i)
        ref[i] = (long double)alpha * x[i] + y0[i];

    std::memcpy(y.data(), y0.data(), n * sizeof(double));
#ifndef NDEBUG
    cblas_daxpy(n, alpha, x.data(), 1, y.data(), 1);
    double err = max_rel_err(y.data(), ref.data(), n);
    std::printf("daxpy[0]: a*x[0] + y[0] = %.6f*%.6f + %.6f = %.6f\n",
                alpha, x[0], y0[0], y[0]);
    return {"daxpy", 2.0, 0.0, 0.0, 0.0, err, 0, err <= kEps};
#else
    cblas_daxpy(n, alpha, x.data(), 1, y.data(), 1); // warmup

    double t_run = 0.0;
    int iters = 0;
    auto t_start = Clock::now();
    for (;;) {
        std::memcpy(y.data(), y0.data(), n * sizeof(double));
        auto b = Clock::now();
        cblas_daxpy(n, alpha, x.data(), 1, y.data(), 1);
        t_run += std::chrono::duration<double>(Clock::now() - b).count();
        ++iters;
        double elapsed = std::chrono::duration<double>(Clock::now() - t_start).count();
        if ((iters >= kMinIters && elapsed >= kMinTime) || elapsed >= kMaxTime)
            break;
    }
    double err = max_rel_err(y.data(), ref.data(), n);
    return make_result("daxpy (y = a*x + y)", 2.0, t_run / iters, n, iters, err);
#endif
}

// r = q.*t + a  (q and t are read-only, r is fully rewritten each run)
Result bench_xypa(int n)
{
    const double a = 0.25;
    std::vector<double> q(n), t(n), r(n);
    std::vector<long double> ref(n);
    seeded(q, 3);
    seeded(t, 4);
    for (int i = 0; i < n; ++i)
        ref[i] = (long double)q[i] * t[i] + a;

#ifndef NDEBUG
    cblas_d1xypa(n, a, q.data(), 1, t.data(), 1, r.data(), 1);
    double err = max_rel_err(r.data(), ref.data(), n);
    std::printf("d1xypa[0]: q[0] .* t[0] + a = %.6f .* %.6f + %.6f = %.6f\n",
                q[0], t[0], a, r[0]);
    return {"d1xypa", 2.0, 0.0, 0.0, 0.0, err, 0, err <= kEps};
#else
    cblas_d1xypa(n, a, q.data(), 1, t.data(), 1, r.data(), 1); // warmup

    double t_run = 0.0;
    int iters = 0;
    auto t_start = Clock::now();
    for (;;) {
        auto b = Clock::now();
        cblas_d1xypa(n, a, q.data(), 1, t.data(), 1, r.data(), 1);
        t_run += std::chrono::duration<double>(Clock::now() - b).count();
        ++iters;
        double elapsed = std::chrono::duration<double>(Clock::now() - t_start).count();
        if ((iters >= kMinIters && elapsed >= kMinTime) || elapsed >= kMaxTime)
            break;
    }
    double err = max_rel_err(r.data(), ref.data(), n);
    return make_result("d1xypa (r = q.*t + a)", 2.0, t_run / iters, n, iters, err);
#endif
}

// r = x1.*y1 + x2.*y2 + x3.*y3  (inputs read-only, r fully rewritten each run)
Result bench_3dot(int n)
{
    std::vector<double> x1(n), x2(n), x3(n), y1(n), y2(n), y3(n), r(n);
    std::vector<long double> ref(n);
    seeded(x1, 11);
    seeded(x2, 12);
    seeded(x3, 13);
    seeded(y1, 14);
    seeded(y2, 15);
    seeded(y3, 16);
    for (int i = 0; i < n; ++i)
        ref[i] = (long double)x1[i] * y1[i] + (long double)x2[i] * y2[i]
               + (long double)x3[i] * y3[i];

#ifndef NDEBUG
    cblas_d3dot(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                y1.data(), 1, y2.data(), 1, y3.data(), 1, r.data(), 1);
    double err = max_rel_err(r.data(), ref.data(), n);
    std::printf("d3dot[0]: (x1[0] x2[0] x3[0]) cdot (y1[0] y2[0] y3[0]) = r[0]\n"
                "(%f %f %f) cdot (%f %f %f) = %f\n",
                x1[0], x2[0], x3[0], y1[0], y2[0], y3[0], r[0]);
    return {"d3dot", 5.0, 0.0, 0.0, 0.0, err, 0, err <= kEps};
#else
    cblas_d3dot(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                y1.data(), 1, y2.data(), 1, y3.data(), 1, r.data(), 1); // warmup

    double t_run = 0.0;
    int iters = 0;
    auto t_start = Clock::now();
    for (;;) {
        auto b = Clock::now();
        cblas_d3dot(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                    y1.data(), 1, y2.data(), 1, y3.data(), 1, r.data(), 1);
        t_run += std::chrono::duration<double>(Clock::now() - b).count();
        ++iters;
        double elapsed = std::chrono::duration<double>(Clock::now() - t_start).count();
        if ((iters >= kMinIters && elapsed >= kMinTime) || elapsed >= kMaxTime)
            break;
    }
    double err = max_rel_err(r.data(), ref.data(), n);
    return make_result("d3dot (r = x1y1+x2y2+x3y3)", 5.0, t_run / iters, n, iters, err);
#endif
}

// w1 = x1.*y1, w2 = x2.*y2, w3 = x3.*y3  (inputs read-only, w fully rewritten each run)
Result bench_3had(int n)
{
    std::vector<double> x1(n), x2(n), x3(n), y1(n), y2(n), y3(n);
    std::vector<double> w1(n), w2(n), w3(n);
    std::vector<long double> ref(n), w2_ref(n), w3_ref(n);
    seeded(x1, 21);
    seeded(x2, 22);
    seeded(x3, 23);
    seeded(y1, 24);
    seeded(y2, 25);
    seeded(y3, 26);
    for (int i = 0; i < n; ++i) {
        ref[i] = (long double)x1[i] * y1[i];
        w2_ref[i] = (long double)x2[i] * y2[i];
        w3_ref[i] = (long double)x3[i] * y3[i];
    }

#ifndef NDEBUG
    cblas_d3had(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                y1.data(), 1, y2.data(), 1, y3.data(), 1,
                w1.data(), 1, w2.data(), 1, w3.data(), 1);
    double err = std::fmax(max_rel_err(w1.data(), ref.data(), n),
                   std::fmax(max_rel_err(w2.data(), w2_ref.data(), n),
                             max_rel_err(w3.data(), w3_ref.data(), n)));
    std::printf("d3had[0]: x1[0] .* y1[0] = w1[0]  etc.\n"
                "(%.6f .* %.6f = %.6f)  (%.6f .* %.6f = %.6f)  (%.6f .* %.6f = %.6f)\n",
                x1[0], y1[0], w1[0], x2[0], y2[0], w2[0], x3[0], y3[0], w3[0]);
    return {"d3had", 3.0, 0.0, 0.0, 0.0, err, 0, err <= kEps};
#else
    cblas_d3had(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                y1.data(), 1, y2.data(), 1, y3.data(), 1,
                w1.data(), 1, w2.data(), 1, w3.data(), 1); // warmup

    double t_run = 0.0;
    int iters = 0;
    auto t_start = Clock::now();
    for (;;) {
        auto b = Clock::now();
        cblas_d3had(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                    y1.data(), 1, y2.data(), 1, y3.data(), 1,
                    w1.data(), 1, w2.data(), 1, w3.data(), 1);
        t_run += std::chrono::duration<double>(Clock::now() - b).count();
        ++iters;
        double elapsed = std::chrono::duration<double>(Clock::now() - t_start).count();
        if ((iters >= kMinIters && elapsed >= kMinTime) || elapsed >= kMaxTime)
            break;
    }
    double err = std::fmax(max_rel_err(w1.data(), ref.data(), n),
                   std::fmax(max_rel_err(w2.data(), w2_ref.data(), n),
                             max_rel_err(w3.data(), w3_ref.data(), n)));
    return make_result("d3had (w = x.*y)", 3.0, t_run / iters, n, iters, err);
#endif
}

// r = x1.^2 + x2.^2 + x3.^2  (inputs read-only, r fully rewritten each run)
Result bench_3sqr(int n)
{
    std::vector<double> x1(n), x2(n), x3(n), r(n);
    std::vector<long double> ref(n);
    seeded(x1, 31);
    seeded(x2, 32);
    seeded(x3, 33);
    for (int i = 0; i < n; ++i)
        ref[i] = (long double)x1[i] * x1[i] + (long double)x2[i] * x2[i]
               + (long double)x3[i] * x3[i];

#ifndef NDEBUG
    cblas_d3sqr(n, x1.data(), 1, x2.data(), 1, x3.data(), 1, r.data(), 1);
    double err = max_rel_err(r.data(), ref.data(), n);
    std::printf("d3sqr[0]: x1[0]^2 + x2[0]^2 + x3[0]^2 = r[0]\n"
                "(%.6f^2 + %.6f^2 + %.6f^2 = %.6f)\n",
                x1[0], x2[0], x3[0], r[0]);
    return {"d3sqr", 5.0, 0.0, 0.0, 0.0, err, 0, err <= kEps};
#else
    cblas_d3sqr(n, x1.data(), 1, x2.data(), 1, x3.data(), 1, r.data(), 1); // warmup

    double t_run = 0.0;
    int iters = 0;
    auto t_start = Clock::now();
    for (;;) {
        auto b = Clock::now();
        cblas_d3sqr(n, x1.data(), 1, x2.data(), 1, x3.data(), 1, r.data(), 1);
        t_run += std::chrono::duration<double>(Clock::now() - b).count();
        ++iters;
        double elapsed = std::chrono::duration<double>(Clock::now() - t_start).count();
        if ((iters >= kMinIters && elapsed >= kMinTime) || elapsed >= kMaxTime)
            break;
    }
    double err = max_rel_err(r.data(), ref.data(), n);
    return make_result("d3sqr (r = x1^2+x2^2+x3^2)", 5.0, t_run / iters, n, iters, err);
#endif
}


// r = sqrt(q.*q)  (inputs read-only, r fully rewritten each run)
Result bench_1norm(int n)
{
    std::vector<double> q(n), r(n);
    std::vector<long double> ref(n);
    seeded(q, 41);
    for (int i = 0; i < n; ++i)
        ref[i] = sqrtl((long double)q[i] * q[i]);

#ifndef NDEBUG
    cblas_d1norm(n, q.data(), 1, r.data(), 1);
    double err = max_rel_err(r.data(), ref.data(), n);
    std::printf("d1norm[0]: sqrt(q[0] * q[0]) = sqrt(%.6f * %.6f) = %.6f\n",
                q[0], q[0], r[0]);
    return {"d1norm", 2.0, 0.0, 0.0, 0.0, err, 0, err <= kEps};
#else
    cblas_d1norm(n, q.data(), 1, r.data(), 1); // warmup

    double t_run = 0.0;
    int iters = 0;
    auto t_start = Clock::now();
    for (;;) {
        auto b = Clock::now();
        cblas_d1norm(n, q.data(), 1, r.data(), 1);
        t_run += std::chrono::duration<double>(Clock::now() - b).count();
        ++iters;
        double elapsed = std::chrono::duration<double>(Clock::now() - t_start).count();
        if ((iters >= kMinIters && elapsed >= kMinTime) || elapsed >= kMaxTime)
            break;
    }
    double err = max_rel_err(r.data(), ref.data(), n);
    return make_result("d1norm (r = sqrt(q.q))", 2.0, t_run / iters, n, iters, err);
#endif
}

// w = x^y  (inputs read-only, w fully rewritten each run)
Result bench_3cross(int n)
{
    std::vector<double> x1(n), x2(n), x3(n), y1(n), y2(n), y3(n);
    std::vector<double> w1(n), w2(n), w3(n);
    std::vector<long double> w1_ref(n), w2_ref(n), w3_ref(n);
    seeded(x1, 51);
    seeded(x2, 52);
    seeded(x3, 53);
    seeded(y1, 54);
    seeded(y2, 55);
    seeded(y3, 56);
    for (int i = 0; i < n; ++i) {
        w1_ref[i] = (long double)x2[i] * y3[i] - (long double)x3[i] * y2[i];
        w2_ref[i] = (long double)x3[i] * y1[i] - (long double)x1[i] * y3[i];
        w3_ref[i] = (long double)x1[i] * y2[i] - (long double)x2[i] * y1[i];
    }

#ifndef NDEBUG
    cblas_d3cross(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                  y1.data(), 1, y2.data(), 1, y3.data(), 1,
                  w1.data(), 1, w2.data(), 1, w3.data(), 1);
    double err = std::fmax(max_rel_err(w1.data(), w1_ref.data(), n),
                   std::fmax(max_rel_err(w2.data(), w2_ref.data(), n),
                             max_rel_err(w3.data(), w3_ref.data(), n)));
    std::printf("d3cross[0]: x^y = w\n"
                "(%f %f %f)^(%f %f %f) = (%f %f %f)\n",
                x1[0], x2[0], x3[0], y1[0], y2[0], y3[0], w1[0], w2[0], w3[0]);
    return {"d3cross", 9.0, 0.0, 0.0, 0.0, err, 0, err <= kEps};
#else
    cblas_d3cross(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                  y1.data(), 1, y2.data(), 1, y3.data(), 1,
                  w1.data(), 1, w2.data(), 1, w3.data(), 1); // warmup

    double t_run = 0.0;
    int iters = 0;
    auto t_start = Clock::now();
    for (;;) {
        auto b = Clock::now();
        cblas_d3cross(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                      y1.data(), 1, y2.data(), 1, y3.data(), 1,
                      w1.data(), 1, w2.data(), 1, w3.data(), 1);
        t_run += std::chrono::duration<double>(Clock::now() - b).count();
        ++iters;
        double elapsed = std::chrono::duration<double>(Clock::now() - t_start).count();
        if ((iters >= kMinIters && elapsed >= kMinTime) || elapsed >= kMaxTime)
            break;
    }
    double err = std::fmax(max_rel_err(w1.data(), w1_ref.data(), n),
                   std::fmax(max_rel_err(w2.data(), w2_ref.data(), n),
                             max_rel_err(w3.data(), w3_ref.data(), n)));
    return make_result("d3cross (w = x^y)", 9.0, t_run / iters, n, iters, err);
#endif
}

// w = a*(x^y)  (inputs read-only, w fully rewritten each run)
Result bench_3crossscal(int n)
{
    const double a = 1.5;
    std::vector<double> x1(n), x2(n), x3(n), y1(n), y2(n), y3(n);
    std::vector<double> w1(n), w2(n), w3(n);
    std::vector<long double> w1_ref(n), w2_ref(n), w3_ref(n);
    seeded(x1, 61);
    seeded(x2, 62);
    seeded(x3, 63);
    seeded(y1, 64);
    seeded(y2, 65);
    seeded(y3, 66);
    for (int i = 0; i < n; ++i) {
        w1_ref[i] = (long double)a * ((long double)x2[i] * y3[i] - (long double)x3[i] * y2[i]);
        w2_ref[i] = (long double)a * ((long double)x3[i] * y1[i] - (long double)x1[i] * y3[i]);
        w3_ref[i] = (long double)a * ((long double)x1[i] * y2[i] - (long double)x2[i] * y1[i]);
    }

#ifndef NDEBUG
    cblas_d3crossscal(n, a, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                      y1.data(), 1, y2.data(), 1, y3.data(), 1,
                      w1.data(), 1, w2.data(), 1, w3.data(), 1);
    double err = std::fmax(max_rel_err(w1.data(), w1_ref.data(), n),
                   std::fmax(max_rel_err(w2.data(), w2_ref.data(), n),
                             max_rel_err(w3.data(), w3_ref.data(), n)));
    std::printf("d3crossscal[0]: a*(x^y) = w  (a = %f)\n"
                "(%f %f %f)^(%f %f %f) = (%f %f %f)\n",
                a, x1[0], x2[0], x3[0], y1[0], y2[0], y3[0], w1[0], w2[0], w3[0]);
    return {"d3crossscal", 12.0, 0.0, 0.0, 0.0, err, 0, err <= kEps};
#else
    cblas_d3crossscal(n, a, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                      y1.data(), 1, y2.data(), 1, y3.data(), 1,
                      w1.data(), 1, w2.data(), 1, w3.data(), 1); // warmup

    double t_run = 0.0;
    int iters = 0;
    auto t_start = Clock::now();
    for (;;) {
        auto b = Clock::now();
        cblas_d3crossscal(n, a, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                          y1.data(), 1, y2.data(), 1, y3.data(), 1,
                          w1.data(), 1, w2.data(), 1, w3.data(), 1);
        t_run += std::chrono::duration<double>(Clock::now() - b).count();
        ++iters;
        double elapsed = std::chrono::duration<double>(Clock::now() - t_start).count();
        if ((iters >= kMinIters && elapsed >= kMinTime) || elapsed >= kMaxTime)
            break;
    }
    double err = std::fmax(max_rel_err(w1.data(), w1_ref.data(), n),
                   std::fmax(max_rel_err(w2.data(), w2_ref.data(), n),
                             max_rel_err(w3.data(), w3_ref.data(), n)));
    return make_result("d3crossscal (w = a*(x^y))", 12.0, t_run / iters, n, iters, err);
#endif
}

// r = (x^y).w  (inputs read-only, r fully rewritten each run)
Result bench_3crossdot(int n)
{
    std::vector<double> x1(n), x2(n), x3(n), y1(n), y2(n), y3(n);
    std::vector<double> w1(n), w2(n), w3(n), r(n);
    std::vector<long double> ref(n);
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
        long double c1 = (long double)x2[i] * y3[i] - (long double)x3[i] * y2[i];
        long double c2 = (long double)x3[i] * y1[i] - (long double)x1[i] * y3[i];
        long double c3 = (long double)x1[i] * y2[i] - (long double)x2[i] * y1[i];
        ref[i] = c1 * w1[i] + c2 * w2[i] + c3 * w3[i];
    }

#ifndef NDEBUG
    cblas_d3crossdot(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                     y1.data(), 1, y2.data(), 1, y3.data(), 1,
                     w1.data(), 1, w2.data(), 1, w3.data(), 1, r.data(), 1);
    double err = max_rel_err(r.data(), ref.data(), n);
    std::printf("d3crossdot[0]: (x^y).w = r\n"
                "((%f %f %f)^(%f %f %f)).(%f %f %f) = %f\n",
                x1[0], x2[0], x3[0], y1[0], y2[0], y3[0], w1[0], w2[0], w3[0], r[0]);
    return {"d3crossdot", 14.0, 0.0, 0.0, 0.0, err, 0, err <= kEps};
#else
    cblas_d3crossdot(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                     y1.data(), 1, y2.data(), 1, y3.data(), 1,
                     w1.data(), 1, w2.data(), 1, w3.data(), 1, r.data(), 1); // warmup

    double t_run = 0.0;
    int iters = 0;
    auto t_start = Clock::now();
    for (;;) {
        auto b = Clock::now();
        cblas_d3crossdot(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                         y1.data(), 1, y2.data(), 1, y3.data(), 1,
                         w1.data(), 1, w2.data(), 1, w3.data(), 1, r.data(), 1);
        t_run += std::chrono::duration<double>(Clock::now() - b).count();
        ++iters;
        double elapsed = std::chrono::duration<double>(Clock::now() - t_start).count();
        if ((iters >= kMinIters && elapsed >= kMinTime) || elapsed >= kMaxTime)
            break;
    }
    double err = max_rel_err(r.data(), ref.data(), n);
    return make_result("d3crossdot (r = (x^y).w)", 14.0, t_run / iters, n, iters, err);
#endif
}

// r = (x^y).(x^y)  (inputs read-only, r fully rewritten each run)
Result bench_3crosssqr(int n)
{
    std::vector<double> x1(n), x2(n), x3(n), y1(n), y2(n), y3(n), r(n);
    std::vector<long double> ref(n);
    seeded(x1, 81);
    seeded(x2, 82);
    seeded(x3, 83);
    seeded(y1, 84);
    seeded(y2, 85);
    seeded(y3, 86);
    for (int i = 0; i < n; ++i) {
        long double c1 = (long double)x2[i] * y3[i] - (long double)x3[i] * y2[i];
        long double c2 = (long double)x3[i] * y1[i] - (long double)x1[i] * y3[i];
        long double c3 = (long double)x1[i] * y2[i] - (long double)x2[i] * y1[i];
        ref[i] = c1 * c1 + c2 * c2 + c3 * c3;
    }

#ifndef NDEBUG
    cblas_d3crosssqr(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                     y1.data(), 1, y2.data(), 1, y3.data(), 1, r.data(), 1);
    double err = max_rel_err(r.data(), ref.data(), n);
    std::printf("d3crosssqr[0]: (x^y).(x^y) = r\n"
                "((%f %f %f)^(%f %f %f)).(same) = %f\n",
                x1[0], x2[0], x3[0], y1[0], y2[0], y3[0], r[0]);
    return {"d3crosssqr", 14.0, 0.0, 0.0, 0.0, err, 0, err <= kEps};
#else
    cblas_d3crosssqr(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                     y1.data(), 1, y2.data(), 1, y3.data(), 1, r.data(), 1); // warmup

    double t_run = 0.0;
    int iters = 0;
    auto t_start = Clock::now();
    for (;;) {
        auto b = Clock::now();
        cblas_d3crosssqr(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                         y1.data(), 1, y2.data(), 1, y3.data(), 1, r.data(), 1);
        t_run += std::chrono::duration<double>(Clock::now() - b).count();
        ++iters;
        double elapsed = std::chrono::duration<double>(Clock::now() - t_start).count();
        if ((iters >= kMinIters && elapsed >= kMinTime) || elapsed >= kMaxTime)
            break;
    }
    double err = max_rel_err(r.data(), ref.data(), n);
    return make_result("d3crosssqr (r = (x^y).(x^y))", 14.0, t_run / iters, n, iters, err);
#endif
}

// u = x^y, v = x^w  (inputs read-only, u and v fully rewritten each run)
Result bench_3crossxy_crossxz(int n)
{
    std::vector<double> x1(n), x2(n), x3(n), y1(n), y2(n), y3(n);
    std::vector<double> w1(n), w2(n), w3(n);
    std::vector<double> u1(n), u2(n), u3(n), v1(n), v2(n), v3(n);
    std::vector<long double> u1_ref(n), u2_ref(n), u3_ref(n);
    std::vector<long double> v1_ref(n), v2_ref(n), v3_ref(n);
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
        u1_ref[i] = (long double)x2[i] * y3[i] - (long double)x3[i] * y2[i];
        u2_ref[i] = (long double)x3[i] * y1[i] - (long double)x1[i] * y3[i];
        u3_ref[i] = (long double)x1[i] * y2[i] - (long double)x2[i] * y1[i];
        v1_ref[i] = (long double)x2[i] * w3[i] - (long double)x3[i] * w2[i];
        v2_ref[i] = (long double)x3[i] * w1[i] - (long double)x1[i] * w3[i];
        v3_ref[i] = (long double)x1[i] * w2[i] - (long double)x2[i] * w1[i];
    }

    double err;
#ifndef NDEBUG
    cblas_d3crossxy_crossxz(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
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
    std::printf("d3crossxy_crossxz[0]: x^y = u, x^w = v\n"
                "u = (%f %f %f)  v = (%f %f %f)\n",
                u1[0], u2[0], u3[0], v1[0], v2[0], v3[0]);
    return {"d3crossxy_crossxz", 18.0, 0.0, 0.0, 0.0, err, 0, err <= kEps};
#else
    cblas_d3crossxy_crossxz(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                            y1.data(), 1, y2.data(), 1, y3.data(), 1,
                            w1.data(), 1, w2.data(), 1, w3.data(), 1,
                            u1.data(), 1, u2.data(), 1, u3.data(), 1,
                            v1.data(), 1, v2.data(), 1, v3.data(), 1); // warmup

    double t_run = 0.0;
    int iters = 0;
    auto t_start = Clock::now();
    for (;;) {
        auto b = Clock::now();
        cblas_d3crossxy_crossxz(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                                y1.data(), 1, y2.data(), 1, y3.data(), 1,
                                w1.data(), 1, w2.data(), 1, w3.data(), 1,
                                u1.data(), 1, u2.data(), 1, u3.data(), 1,
                                v1.data(), 1, v2.data(), 1, v3.data(), 1);
        t_run += std::chrono::duration<double>(Clock::now() - b).count();
        ++iters;
        double elapsed = std::chrono::duration<double>(Clock::now() - t_start).count();
        if ((iters >= kMinIters && elapsed >= kMinTime) || elapsed >= kMaxTime)
            break;
    }
    err = std::fmax(max_rel_err(u1.data(), u1_ref.data(), n),
              std::fmax(max_rel_err(u2.data(), u2_ref.data(), n),
                        std::fmax(max_rel_err(u3.data(), u3_ref.data(), n),
                                  std::fmax(max_rel_err(v1.data(), v1_ref.data(), n),
                                            std::fmax(max_rel_err(v2.data(), v2_ref.data(), n),
                                                      max_rel_err(v3.data(), v3_ref.data(), n))))));
    return make_result("d3crossxy_crossxz (u = x^y, v = x^w)", 18.0, t_run / iters, n, iters, err);
#endif
}

// u = x^y, r = x.w  (inputs read-only, u and r fully rewritten each run)
Result bench_3crossxy_dotxz(int n)
{
    std::vector<double> x1(n), x2(n), x3(n), y1(n), y2(n), y3(n);
    std::vector<double> w1(n), w2(n), w3(n);
    std::vector<double> u1(n), u2(n), u3(n), r(n);
    std::vector<long double> u1_ref(n), u2_ref(n), u3_ref(n), ref(n);
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
        u1_ref[i] = (long double)x2[i] * y3[i] - (long double)x3[i] * y2[i];
        u2_ref[i] = (long double)x3[i] * y1[i] - (long double)x1[i] * y3[i];
        u3_ref[i] = (long double)x1[i] * y2[i] - (long double)x2[i] * y1[i];
        ref[i] = (long double)x1[i] * w1[i] + (long double)x2[i] * w2[i]
               + (long double)x3[i] * w3[i];
    }

    double err;
#ifndef NDEBUG
    cblas_d3crossxy_dotxz(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                          y1.data(), 1, y2.data(), 1, y3.data(), 1,
                          w1.data(), 1, w2.data(), 1, w3.data(), 1,
                          u1.data(), 1, u2.data(), 1, u3.data(), 1, r.data(), 1);
    err = std::fmax(max_rel_err(u1.data(), u1_ref.data(), n),
              std::fmax(max_rel_err(u2.data(), u2_ref.data(), n),
                        std::fmax(max_rel_err(u3.data(), u3_ref.data(), n),
                                  max_rel_err(r.data(), ref.data(), n))));
    std::printf("d3crossxy_dotxz[0]: x^y = u, x.w = r\n"
                "u = (%f %f %f)  r = %f\n", u1[0], u2[0], u3[0], r[0]);
    return {"d3crossxy_dotxz", 14.0, 0.0, 0.0, 0.0, err, 0, err <= kEps};
#else
    cblas_d3crossxy_dotxz(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                          y1.data(), 1, y2.data(), 1, y3.data(), 1,
                          w1.data(), 1, w2.data(), 1, w3.data(), 1,
                          u1.data(), 1, u2.data(), 1, u3.data(), 1, r.data(), 1); // warmup

    double t_run = 0.0;
    int iters = 0;
    auto t_start = Clock::now();
    for (;;) {
        auto b = Clock::now();
        cblas_d3crossxy_dotxz(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                              y1.data(), 1, y2.data(), 1, y3.data(), 1,
                              w1.data(), 1, w2.data(), 1, w3.data(), 1,
                              u1.data(), 1, u2.data(), 1, u3.data(), 1, r.data(), 1);
        t_run += std::chrono::duration<double>(Clock::now() - b).count();
        ++iters;
        double elapsed = std::chrono::duration<double>(Clock::now() - t_start).count();
        if ((iters >= kMinIters && elapsed >= kMinTime) || elapsed >= kMaxTime)
            break;
    }
    err = std::fmax(max_rel_err(u1.data(), u1_ref.data(), n),
              std::fmax(max_rel_err(u2.data(), u2_ref.data(), n),
                        std::fmax(max_rel_err(u3.data(), u3_ref.data(), n),
                                  max_rel_err(r.data(), ref.data(), n))));
    return make_result("d3crossxy_dotxz (u = x^y, r = x.w)", 14.0, t_run / iters, n, iters, err);
#endif
}

// r = x.y, q = x.w  (inputs read-only, r and q fully rewritten each run)
Result bench_3dotxy_dotxz(int n)
{
    std::vector<double> x1(n), x2(n), x3(n), y1(n), y2(n), y3(n);
    std::vector<double> w1(n), w2(n), w3(n);
    std::vector<double> r(n), q(n);
    std::vector<long double> r_ref(n), q_ref(n);
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
        r_ref[i] = (long double)x1[i] * y1[i] + (long double)x2[i] * y2[i]
                 + (long double)x3[i] * y3[i];
        q_ref[i] = (long double)x1[i] * w1[i] + (long double)x2[i] * w2[i]
                 + (long double)x3[i] * w3[i];
    }

    double err;
#ifndef NDEBUG
    cblas_d3dotxy_dotxz(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                        y1.data(), 1, y2.data(), 1, y3.data(), 1,
                        w1.data(), 1, w2.data(), 1, w3.data(), 1,
                        r.data(), 1, q.data(), 1);
    err = std::fmax(max_rel_err(r.data(), r_ref.data(), n),
              max_rel_err(q.data(), q_ref.data(), n));
    std::printf("d3dotxy_dotxz[0]: x.y = r, x.w = q\n"
                "r = %f  q = %f\n", r[0], q[0]);
    return {"d3dotxy_dotxz", 10.0, 0.0, 0.0, 0.0, err, 0, err <= kEps};
#else
    cblas_d3dotxy_dotxz(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                        y1.data(), 1, y2.data(), 1, y3.data(), 1,
                        w1.data(), 1, w2.data(), 1, w3.data(), 1,
                        r.data(), 1, q.data(), 1); // warmup

    double t_run = 0.0;
    int iters = 0;
    auto t_start = Clock::now();
    for (;;) {
        auto b = Clock::now();
        cblas_d3dotxy_dotxz(n, x1.data(), 1, x2.data(), 1, x3.data(), 1,
                            y1.data(), 1, y2.data(), 1, y3.data(), 1,
                            w1.data(), 1, w2.data(), 1, w3.data(), 1,
                            r.data(), 1, q.data(), 1);
        t_run += std::chrono::duration<double>(Clock::now() - b).count();
        ++iters;
        double elapsed = std::chrono::duration<double>(Clock::now() - t_start).count();
        if ((iters >= kMinIters && elapsed >= kMinTime) || elapsed >= kMaxTime)
            break;
    }
    err = std::fmax(max_rel_err(r.data(), r_ref.data(), n),
              max_rel_err(q.data(), q_ref.data(), n));
    return make_result("d3dotxy_dotxz (r = x.y, q = x.w)", 10.0, t_run / iters, n, iters, err);
#endif
}

} // namespace

int main(int argc, char **argv)
{
    int n = 1000;
    if (argc > 1) {
        n = std::atoi(argv[1]);
        if (n <= 0) {
            std::fprintf(stderr, "usage: %s [n]  (n > 0)\n", argv[0]);
            return 2;
        }
    }

    std::vector<Result> results;
    results.push_back(bench_axpy(n));
    results.push_back(bench_xypa(n));
    results.push_back(bench_3dot(n));
    results.push_back(bench_3had(n));
    results.push_back(bench_3sqr(n));
    results.push_back(bench_1norm(n));
    results.push_back(bench_3cross(n));
    results.push_back(bench_3crossscal(n));
    results.push_back(bench_3crossdot(n));
    results.push_back(bench_3crosssqr(n));
    results.push_back(bench_3crossxy_crossxz(n));
    results.push_back(bench_3crossxy_dotxz(n));
    results.push_back(bench_3dotxy_dotxz(n));

    bool ok = true;
#ifndef NDEBUG
    // debug mode: the vector prints above are the only output
    for (const Result &res : results)
        ok = ok && res.ok;
    return ok ? 0 : 1;
#else
    std::printf("n = %d\n\n", n);
    std::printf("%-26s %10s %12s %12s %13s %7s  %s\n",
                "kernel", "ms/run", "GFLOP/s", "Melem/s", "max_rel_err",
                "iters", "status");
    for (const Result &res : results) {
        ok = ok && res.ok;
        std::printf("%-26s %10.3f %12.2e %12.2e %13.3e %7d  %s\n",
                    res.name.c_str(), res.ms_per_run, res.gflops, res.meps,
                    res.max_rel_err, res.iters, res.ok ? "OK" : "FAIL");
    }
    return ok ? 0 : 1;
#endif
}
