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

## v3blas — higher-level interface, three front-ends (done 2026-10-08)

MATLAB/Octave-style API over the 13 kernels, design in `V3BLAS_API.md`.

- `openblas/v3blas.h` (plain C, C++-safe, single macro × 4 precisions):
  - 1-vectors are plain unit-stride C arrays (no struct, no wrapper);
    structs `v3<p> {T *x,*y,*z; blasint incx,incy,incz, n;}`
    (`<p>` = s d c z; `blasint` = the cblas extension index type);
  - complex element = `v3blas_floatcomplex`/`v3blas_doublecomplex` `{re, im}` — the
    interleaved cblas c/z layout, zero-copy, re/im always explicit;
  - ops named like the kernels: `v1axpy_<p>`, `v1xypa_<p>`, `v3dot_<p>`, `v3had_<p>`,
    `v3sqr_<p>`, `v1norm_<p>`, `v3cross_<p>`, `v3crossscal_<p>`, `v3crossdot_<p>`,
    `v3crosssqr_<p>`, `v3crossxy_crossxz_<p>`, `v3crossxy_dotxz_<p>`, `v3dotxy_dotxz_<p>`;
  - all ops are `void`: outputs are trailing caller-provided pointer
    arguments (`T *r`, `T *wx, *wy, *wz`, ...); no result structs, no
    return values, nothing to alias;
  - C11 `_Generic` macros give precision-generic names (`v3cross(...)` →
    `v3cross_s/_d/_c/_z` from the output pointer type); suffixed names
    stay canonical and work everywhere;
  - the library NEVER allocates: outputs are caller arrays (unit stride);
    inputs borrowed and never modified; `v3<p>_wrap` sets incs to 1,
    fields are public.
- `openblas/v3blas.hpp` (C++ front-end): same signatures as real `inline`
  overloads per precision (no macros under `__cplusplus`); `v3blas_hpp_test.cpp`
  checks every op × precision bitwise (`memcmp`) against `cblas_*`.
- `openblas/v3blas.f90` (Fortran front-end, F2003 `iso_c_binding` module):
  generic interfaces `v1axpy ... v3dotxy_dotxz` over 52 `bind(C)` wrappers
  (BLAS-style arg order, `n` first, unit stride only); `gfortran
  -std=f2008 -Wall` clean; `v3blas_f90_test.f90` checks 13 generics ×
  4 precisions against Fortran-recomputed equations (52 checks, 0 fail).
- `benchmark/v3blas_test.c` (C99, long double reference): all 13 ops × 4 precisions
  + strided smoke case (v3cross_d inc=2) — all pass
  (d/z exact, s/c ≤ ~3e-7). C++ compile check passes.
- `openblas/CMakeLists.txt`: installs `v3blas.h`, `v3blas.hpp`, `v3blas.f90`
  next to `cblas.h`.

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

## Benchmark data and linking

- benchmark data is hash2(seed, element) in [-0.5, 0.5), full avalanche, no
  ramps/affine structure: ramp data makes every cross product a difference of
  nearly equal products, which cancels catastrophically at float precision as
  n grows (the old (i+seed)%997 sawtooth was a ramp with equally spaced
  components; LCG-from-seed is also affine and just hides it).
- v3blas_bench verifies v3blas == cblas bitwise on identical inputs (its subject
  is API overhead: call + result free vs raw call with pre-allocated outputs);
  true-reference checks stay in v3blas_test.c and the unit tests.
- benchmark/ links only an *installed* OpenBLAS (cmake --install build
  --prefix ../pkgs/openblas; headers in include/openblas/). Referencing the
  source + build trees instead duplicates the same headers/macros.
