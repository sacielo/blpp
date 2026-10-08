# Physics kernels in OpenBLAS — progress

Order: user-specified first two (`1xypa`, then `3dot`), then the rest in the
order written in `blasKernels.md`. Every kernel: `s d c z` precisions,
Fortran + cblas interfaces, unit + cblas tests, generic kernel, plain C.
All work lives in `/home/sac/blpp/openblas`.

| # | symbol | math | status |
|---|---|---|---|
| 1 | `<p>1xypa` | `r = q.*t + a` | done 2026-07-20 — 9 unit + 44 cblas tests pass |
| 2 | `<p>3dot` | `r = x·y` | done 2026-10-06 — unit + cblas tests pass, bench 0.095 ms/run @ n=1000 |
| 3 | `<p>3had` | `w = x.*y` | done 2026-10-06 — unit + cblas tests pass, bench 0.236 ms/run @ n=1000 |
| 4 | `<p>3cross` | `w = x∧y` | done 2026-10-07 — unit + cblas tests pass, bench 0.201 ms/run @ n=1000 |
| 5 | `<p>3crossscal` | `w = a·(x∧y)` | done 2026-10-07 — unit + cblas tests pass, bench 0.260 ms/run @ n=1000 |
| 6 | `<p>3sqr` | `r = x·x` | done 2026-10-06 — unit + cblas tests pass, bench 0.084 ms/run @ n=1000 |
| 7 | `<p>3crossdot` | `r = (x∧y)·w` | done 2026-10-07 — unit + cblas tests pass, bench 0.237 ms/run @ n=1000 |
| 8 | `<p>3crosssqr` | `r = (x∧y)·(x∧y)` | done 2026-10-07 — unit + cblas tests pass, bench 0.154 ms/run @ n=1000 |
| 9 | `<p>3crossxy_crossxz` | `u = x∧y`, `v = x∧w` | done 2026-10-07 — unit + cblas tests pass, bench 0.653 ms/run @ n=1000 |
| 10 | `<p>3crossxy_dotxz` | `u = x∧y`, `r = x·w` | done 2026-10-07 — unit + cblas tests pass, bench 0.485 ms/run @ n=1000 |
| 11 | `<p>3dotxy_dotxz` | `r = x·y`, `q = x·w` | done 2026-10-07 — unit + cblas tests pass, bench 0.379 ms/run @ n=1000 |
| 12 | `<p>1norm` | `r = √(q·q)` | done 2026-10-07 — unit + cblas tests pass, bench 0.084 ms/run @ n=1000 |

## pblas.h — header-only higher-level interface (done 2026-10-08)

MATLAB/Octave-style API over the 13 kernels, design in `PBLAS_API.md`.

- `openblas/pblas.h` (plain C, C++-safe, single macro × 4 precisions):
  - structs `v1<p> {T *x; blasint inc, n;}`, `v3<p> {T *x,*y,*z; blasint incx,incy,incz, n;}`
    (`<p>` = s d c z; `blasint` = the cblas extension index type);
  - complex element = `pblas_floatcomplex`/`pblas_doublecomplex` `{re, im}` — the
    interleaved cblas c/z layout, zero-copy, re/im always explicit;
  - ops named like the kernels: `v1axpy_<p>`, `v1xypa_<p>`, `v3dot_<p>`, `v3had_<p>`,
    `v3sqr_<p>`, `v1norm_<p>`, `v3cross_<p>`, `v3crossscal_<p>`, `v3crossdot_<p>`,
    `v3crosssqr_<p>`, `v3crossxy_crossxz_<p>`, `v3crossxy_dotxz_<p>`, `v3dotxy_dotxz_<p>`;
  - multi-output result structs: `v3uv<p> {v3 u; v3 v;}`, `v3ur<p> {v3 u; v1 r;}`,
    `v1rq<p> {v1 r; v1 q;}` (members named after the equation output tokens);
  - results heap-allocated (inc 1) via `*_new`, released by the matching `_free`
    (v1/v3 _free take a pointer, combined _free by value); inputs borrowed.
- `benchmark/pblas_test.c` (C99, long double reference): all 13 ops × 4 precisions
  + strided smoke cases (v3cross_d inc=2, v1xypa_c inc=2) — all pass
  (d/z exact, s/c ≤ ~3e-7). C++ compile check passes.
- Root `CMakeLists.txt`: installs `pblas.h` next to `cblas.h`.

## Conventions settled while doing #1

- Symbols: `s1xypa`, `d1xypa`, `c1xypa`, `z1xypa`; cblas: `cblas_s1xypa` etc.
- cblas prototype (output last, BLAS convention):
  `void cblas_<p>1xypa(blasint n, <T> a, const <T> *q, blasint incq, const <T> *t, blasint inct, <T> *r, blasint incr)`
  (`<T>` = `void*` for `c`/`z`, complex scalar is a pair of reals)
- Fortran entry: all arguments by pointer.
- Complex product is bilinear (no conjugation):
  `r = (qr*tr − qi*ti + ar) + i(qr*ti + qi*tr + ai)`
- Kernel signature (threaded level-1 calling convention):
  real: `int CNAME(BLASLONG n, BLASLONG d0, BLASLONG d1, FLOAT a, FLOAT *q, BLASLONG incq, FLOAT *t, BLASLONG inct, FLOAT *r, BLASLONG incr, FLOAT *dummy, BLASLONG dummy2)`
  complex: same with `FLOAT ar, FLOAT ai` after the dummies; component strides doubled inside the kernel.
- Macro name in headers is `XYPA_K` (C identifiers cannot start with a digit).
- Files per kernel: `kernel/generic/1xypa.c` (s/d), `kernel/generic/z1xypa.c` (c/z),
  `interface/1xypa.c`, `interface/z1xypa.c`,
  `utest/test_1xypa.c` (unit, Fortran entries),
  `utest/test_extensions/test_{s,d,c,z}1xypa.c` (cblas tests).
- Wiring points (per kernel):
  - `kernel/Makefile.L1`: `SBLASOBJS/DBLASOBJS/CBLASOBJS/ZBLASOBJS += <p>1xypa_k` + per-precision compile rules
  - `kernel/CMakeLists.txt` (float loop, next to geadd) + `cmake/kernel.cmake` (`SetDefaultL1` fallbacks)
  - `common_level1.h` prototypes; `common_{s,d,c,z}.h` `XYPA_K` macro (both non-DYNAMIC and DYNAMIC blocks);
    `common_macro.h` alias blocks (SINGLE/DOUBLE/COMPLEX/ZCOMPLEX); `common_param.h` `gotoblas_t` field;
    `kernel/setparam-ref.c` initializer; `common_interface.h` prototypes
  - `interface/Makefile`: objects into `SBLAS1OBJS`/`DBLAS1OBJS`/`CBLAS1OBJS`/`ZBLAS1OBJS`
    (precision comes from `Makefile.tail` list overrides) + cblas objects into `CSBLAS1OBJS`/`CDBLAS1OBJS`/`CCBLAS1OBJS`/`CZBLAS1OBJS` + rules
  - `interface/CMakeLists.txt`: `1xypa.c` into `BLAS1_MANGLED_SOURCES`
  - `cblas.h`: `cblas_<p>1xypa` declarations
  - `exports/gensymbol`: `s1xypa` etc. in per-precision lists, `cblas_*` in cblas lists
  - `utest/Makefile` (`OBJS` + `OBJS_EXT`) and `utest/CMakeLists.txt`
- Build: CMake only (this checkout has no working `configure`/make prebuild):
  `cmake -B build -DNO_LAPACK=1` then `make -j1 openblas_static openblas_utest openblas_utest_ext`,
  run `./build/utest/openblas_utest` and `./build/utest/openblas_utest_ext` (filter: `1xypa`, `s1xypa`, ...).

## Side experiments (parked)

- Oversubscription: run the benchmark with 2 OpenBLAS threads on this 1-core machine and see whether a second thread overlaps the in-order core's load stalls (memory-level parallelism). Blocked on the current build: `MAX_CPU_NUMBER=1` is compiled in (CMake `CORE_COUNT`), which clamps `OPENBLAS_NUM_THREADS` to 1 (verified: `openblas_get_num_threads()` returns 1 regardless of the env var). Unblock = reconfigure with `-DCORE_COUNT=2` + full rebuild (~1.5–2 h); note this also changes blocking parameters for all kernels, so results wouldn't isolate the oversubscription effect.

## Oddjobs

- fix the 1xypa kernel template so that scalar argument a comes after vector argument x and y (6 arguments each)
