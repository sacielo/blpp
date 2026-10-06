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

    bool ok = true;
#ifndef NDEBUG
    // debug mode: the vector prints above are the only output
    for (const Result &res : results)
        ok = ok && res.ok;
    return ok ? 0 : 1;
#else
    std::printf("n = %d\n\n", n);
    std::printf("%-22s %10s %12s %12s %13s %7s  %s\n",
                "kernel", "ms/run", "GFLOP/s", "Melem/s", "max_rel_err",
                "iters", "status");
    for (const Result &res : results) {
        ok = ok && res.ok;
        std::printf("%-22s %10.3f %12.2e %12.2e %13.3e %7d  %s\n",
                    res.name.c_str(), res.ms_per_run, res.gflops, res.meps,
                    res.max_rel_err, res.iters, res.ok ? "OK" : "FAIL");
    }
    return ok ? 0 : 1;
#endif
}
