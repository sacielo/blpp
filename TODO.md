# Physics kernels in OpenBLAS — progress

Order: user-specified first two (`1xypa`, then `3dot`), then the rest in the
order written in `blasKernels.md`. Every kernel: `s d c z` precisions,
Fortran + cblas interfaces, unit + cblas tests, generic kernel, plain C.
All work lives in `/home/sac/blpp/openblas`.

| # | symbol | math | status |
|---|---|---|---|
| 1 | `<p>1xypa` | `r = q.*t + a` | done 2026-07-20 — 9 unit + 44 cblas tests pass |
| 2 | `<p>3dot` | `r = x·y` | todo |
| 3 | `<p>3had` | `w = x.*y` | todo |
| 4 | `<p>3cross` | `w = x∧y` | todo |
| 5 | `<p>3crossscal` | `w = a·(x∧y)` | todo |
| 6 | `<p>3sqr` | `r = x·x` | todo |
| 7 | `<p>3crossdot` | `r = (x∧y)·w` | todo |
| 8 | `<p>3crosssqr` | `r = (x∧y)·(x∧y)` | todo |
| 9 | `<p>3crossxy_crossxz` | `u = x∧y`, `v = x∧w` | todo |
| 10 | `<p>3crossxy_dotxz` | `u = x∧y`, `r = x·w` | todo |
| 11 | `<p>3dotxy_dotxz` | `r = x·y`, `q = x·w` | todo |
| 12 | `<p>1norm` | `r = √(q·q)` | todo |

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
