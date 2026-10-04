// Benchmark for memory-bound, element-wise BLAS L1 kernels.
//
// usage: benchmark [n]    (vector size, default 1000)
//
// Each kernel is first checked against a known result (independently
// computed reference in long double), then timed over repeated runs.
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

void fill(std::vector<double> &v, int seed)
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
    fill(x, 1);
    fill(y0, 2);
    for (int i = 0; i < n; ++i)
        ref[i] = (long double)alpha * x[i] + y0[i];

    std::memcpy(y.data(), y0.data(), n * sizeof(double));
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
}

// r = q.*t + a  (q and t are read-only, r is fully rewritten each run)
Result bench_xypa(int n)
{
    const double a = 0.25;
    std::vector<double> q(n), t(n), r(n);
    std::vector<long double> ref(n);
    fill(q, 3);
    fill(t, 4);
    for (int i = 0; i < n; ++i)
        ref[i] = (long double)q[i] * t[i] + a;

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

    bool ok = true;
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
}
