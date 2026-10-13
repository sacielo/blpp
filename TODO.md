# Physics kernels in OpenBLAS — progress

Big picture and what comes next (CFD layering, stencils, threading):
[`grand_plan.md`](grand_plan.md).

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

## Stencils — `lap5` / `lap7` (done 2026-10-11)

`<p>lap5` / `<p>lap7` (s d c z): signed element offsets `jx jy [jz]` and
scale `alpha` carried as the scalar type, so one call maps a strided row of
a 2-D/3-D grid (`jy = ld*incy`); summation order pinned in the spec
(pairwise ± neighbours, pairs last) for bitwise conformance; complex is
linear, no conjugates, real `alpha`; no v3blas front-end (stride-stencil
RFC track). utest 220 (8 f77), utest_ext 2765 (+120), bench rows live.
Gotchas on the way: the f77 hand-expectation for `dlap5` was miscomputed
(kernel was right, test fixed); the interface-list wiring exposed a stale
duplicate block in `interface/CMakeLists.txt` (cleaned).

## CFD menu ops — six 3-vector composites (done 2026-10-10)

`3crosscross` `r=(x∧y)∧z`, `3norm_unit` `r=√(x·x+eps), u=x/r`,
`3refl` `q=x−a(x·y)y/(y·y+eps)`, `3exb` `r=a(x∧y)/(y·y+eps)`,
`3drag` `r=a·x·√(x·x+eps)`, `3mom_ke` `m=s·x, k=½·s·(x·x)` — full
pipeline: generic kernels + interfaces + exports (24 symbols), ext
utests (2645 checks), wrapper utests (212), v3blas.h/.hpp/.f90
front-ends (19 ops), C/C++/F90 driver tests (195/80/76 checks),
benchmark templates ×4 precisions. `3minmod`/`3sel` stay candidates
(`cfd_kernel_candidates.md` was folded into the README and deleted).
Gotchas fixed on the way: complex `sqrt` references must take the
principal sqrt of the operand, not of its square (`cnorm1` semantics);
Fortran test scratch must never be an output array (`q`) and comparison
constants must match what the call actually passed (`al` vs `1.5`).

## v3blas — higher-level interface, three front-ends (done 2026-10-08)

MATLAB/Octave-style API over the 19 element-wise kernels, design in
`V3BLAS_API.md` (grew from the first 13; lap5/lap7 stay raw-cblas).

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
  generic interfaces `v1axpy ... v3mom_ke` over 76 wrappers; 3-vector ops
  take whole fields `u(n, 3)` (columns = component arrays, `n` implicit,
  runtime `(n,3)` guard); 1-vector ops keep explicit `n` (2026-10-13
  field-form rewrite); `gfortran -std=f2008 -Wall` clean;
  `v3blas_f90_test.f90` checks 19 generics × 4 precisions against
  Fortran-recomputed equations (76 checks, 0 fail; `[s|d|c|z]` CLI).
- `benchmark/v3blas_test.c` (C99, long double reference): all 19 ops × 4 precisions
  + strided smoke case (v3cross_d inc=2) — all pass
  (d/z exact, s/c ≤ ~3e-7). C++ compile check passes. All four `_test`
  programs accept `[s|d|c|z]` to run one precision (default: all).
- `openblas/CMakeLists.txt`: installs `v3blas.h`, `v3blas.hpp`, `v3blas.f90`
  next to `cblas.h`.

## Conventions settled while doing #1

- Symbols: `s1xypa`, `d1xypa`, `c1xypa`, `z1xypa`; cblas: `cblas_s1xypa` etc.
- cblas prototype (name-ordered args, output last — `1xypa` literally
  reads `x y p(a)lus a`, so vectors first, scalar `a` after them; reordered
  2026-10-13, f77 entry matches):
  `void cblas_<p>1xypa(blasint n, const <T> *q, blasint incq, const <T> *t, blasint inct, <T> a, <T> *r, blasint incr)`
  (`<T>` = `void*` for `c`/`z`, complex scalar is a pair of reals;
  the internal kernel CNAME keeps the OpenBLAS scalar-first convention —
  only the public layers are name-ordered)
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

- Oversubscription: run the benchmark with 2 OpenBLAS threads on this 1-core machine and see whether a second thread overlaps the in-order core's load stalls (memory-level parallelism). NOTE (2026-10-13): the threading layer work already rebuilt with `MAX_CPU_NUMBER=4` and validated at forced `NUM_THREADS=4` (bitwise T1-vs-T4), so the old `CORE_COUNT=1` clamp note is obsolete — the experiment only needs bench runs at NUM_THREADS ∈ {1,2,4}, still meaningless on 1 core beyond correctness.

## Oddjobs

- [x] Reorder `1xypa` so scalar `a` comes after the vector arguments
      (done 2026-10-13: cblas/f77/common_interface/v3blas.h/.f90 + all
      call sites; kernel CNAME unchanged — scalar-after-dummies is the
      internal calling convention the thread marshaller relies on)

## Threading layer — `blas_physics_thread` (done 2026-10-11)

The OpenBLAS way: interface decides, driver dispatches. New
`driver/others/blas_physics_thread.c` adds
`blas_physics_thread(shape, mode, n, scalars, ptrs, incs, nptr, kfunc, nthreads)`
which slices `n` like `blas_level1_thread` but strides every pointer with its
own increment (the old L1 marshaller never strides C — the 1xypa bug), then
runs the kernel through the normal `exec_blas()` worker pool. Interfaces carry
a small `#ifdef SMP` block: a shape id (`PHYS_3HAD … PHYS_LAP7` in
`common_level1.h`), the scalar/pointer slots in `phys_slice_t`, and the serial
gate `any inc == 0 || n <= PHYS_MT_MIN (10000) ? 1 : num_cpu_avail(1)`.
Threaded: `3had 3cross 3crossscal 3crosscross 3crossxy_crossxz 3norm_unit
3refl 3exb 3drag 3mom_ke lap5 lap7` × s d c z. Reductions (`3dot &co`) and
`1xypa` stay serial by design (pinned summation order = bitwise contract);
`lap` results are order-independent since each output reads inputs only.
Validated on a forced `-DNUM_THREADS=4` build on the 1-core Pi: utest 220 +
ext 2765 green with default and `OPENBLAS_NUM_THREADS=4`; an out-of-tree
checker runs all 40 threaded cblas entries at T=1 then T=4 and `memcmp`s the
outputs — 40/40 bitwise (`MT-CHECK-OK`). Gotchas: C macro params don't thread
through nested file-level macros (solved with per-precision exec functions
opening with `typedef float FT;`); `3crosscross` needs 12 pointer slots,
`3crossxy_crossxz` 15; only `3crossscal` has a complex scalar (re/im pair in
slots 0–1); driver entry compiled only under `if (USE_THREAD)` in CMake.
(One "MT corruption" scare was the checker itself: `3crossxy_crossxz`'s third
group is the INPUT `w` — passing the shared z region there poisoned the next
op's input. Library was innocent.)

## Big iron (HPC target) — remaining

- [ ] Multicore perf check: Pi threading is correctness-only (1 core). Rerun
      the bench + checker on a real box, n ∈ {9999, 10001, 100000} ×
      OPENBLAS_NUM_THREADS ∈ {1, 2, 4, 8}, all precisions, incl. inc=0/2/-1,
      and tune `PHYS_MT_MIN` from the crossover curve.
- [ ] Measure first whether L1 threading is even worth it: these are
      bandwidth-bound streaming kernels; upstream itself disables L1 threading
      below n=10000 for perf reasons. Bench on a real multicore box before
      investing in the driver patch.
- [ ] Think about who actually threads at scale. Typical HPC physics codes are
      MPI × OpenMP with BLAS called *inside* the parallel region — intra-kernel
      threading then oversubscribes (the classic NUMA footgun); for those the
      single-threaded kernels are the right default and the element-wise ops
      parallelize trivially at the caller level. Intra-BLAS threading mostly
      matters for single-rank workloads.
- [ ] Is OpenBLAS the right substrate for the HPC endgame? Options to weigh:
      keep patching OpenBLAS (per-arch optimized kernels would be the next real
      step — ours are generic-C only; threading + per-arch = upstreaming
      territory); BLIS (explicitly built for adding kernels, has a kernel
      framework and a threading layer, community says it extends more cleanly);
      MKL (fast, has vector math libs, proprietary + no custom kernels); write
      the ops as vendor-offload pragmas/kernels in the app (CUDA/HIP/SYCL) and
      keep BLAS as the CPU fallback. Decide after the multicore bench above,
      not before.

## Standardization track (the endgame — see README "Roadmap")

Goal: same op names in every BLAS; users write `v3cross(A,B,W)`, link
their local BLAS, done. Providers adopt spec + conformance suite, not PRs
they must trust.

- [ ] Conformance runner: standalone binary, `dlopen`s an arbitrary BLAS
      (shared or static MKL/ESSL), probes symbols (LP64 + ILP64 name
      variants), runs the 19 ops × 4 precisions against the long-double
      reference vectors from `v3blas_test.c`, prints a per-op pass/fail
      table. Must run on a vendor lib with zero recompilation — this is
      the outreach weapon ("your lib fails 12/12, here's the 2-page spec").
- [ ] Versioned spec document: ABI (names, arg order, strides), ILP64
      convention (`_ilp64` vs `_64` — providers will fight; propose one),
      complex scalar = re/im pair like cblas, inc<=0 and aliasing rules,
      NaN/Inf semantics, and the fused multi-output ops flagged as
      beyond-classic-BLAS semantics (needs a Level-1 extension API).
- [ ] OpenBLAS RFC discussion first (names + opt-in flag `PHYSICS_EXT`),
      then the PR as a commit series (per-op kernel+interface+tests,
      wiring split, DCO sign-off, rebase e016600 → master).
      Ready: 3 representative `benchmark/` programs (3dot, 3crosscross,
      3dotxy_dotxz × 4 prec, `.goto` only) already in the patch.
- [ ] BLIS port of one kernel as feasibility probe (refk + kern structs;
      their axpby precedent) — second adopter = FOMO lever.
- [ ] BLAS Extensions forum proposal for the subset whose shape fits
      their Level-1 grammar (1xypa/3had/3dot family); fused ops as a
      separate API-extension request.

## Benchmark/test suite restructure (user) — DONE

- [x] Separate correctness tests from benchmarks: benches are timing-only
      (no verify path, no reference code); tests time nothing. Shared
      machinery: `benchmark/test_ops.h` (long-double reference engine) +
      `benchmark/ops_octave.h` (Octave block printer), included by both
      C tests only.
- [x] Complete matrix, one program per cell: 4 interfaces
      (cblas, v3blas C11, v3blas C++, v3blas F90) x {test, bench, example}
      = 12 programs + `v3blas_example.c` (C11 example predates this).
      Timing budgets are `-DK_MIN_TIME/-DK_MIN_ITERS/-DK_MAX_TIME`.
- [x] Only TESTS print the worked first-element equations, as a
      pasteable MATLAB/Octave block (`ops_octave.h`): named inputs,
      bilinear `.*` formulas (correct for complex too — the ops are
      bilinear, not sesquilinear), `% => value` so the block doubles as
      a self-check; lap stencils print the padded `U` array with base-17
      indexing. Benches keep an equation *token* column only, as the
      FLOP-count legend. Blocks cross-validated against the in-test
      long-double references (real + complex).
- [x] EXAMPLES = pure tutorials: one `main`, textbook-shaped calls, no
      checks, no timing, minimal output
      (`cblas_example.c`, `v3blas_example.c`, `v3blas_cpp_example.cpp`,
      `v3blas_f90_example.f90`).

## Benchmark data and linking

- benchmark data is hash2(seed, element) in [-0.5, 0.5), full avalanche, no
  ramps/affine structure: ramp data makes every cross product a difference of
  nearly equal products, which cancels catastrophically at float precision as
  n grows (the old (i+seed)%997 sawtooth was a ramp with equally spaced
  components; LCG-from-seed is also affine and just hides it).
- v3blas_bench is timing-only since the test/bench split (API overhead
  comparison v3blas vs raw cblas); true-reference checks live in
  v3blas_test.c and the unit tests.
- benchmark/ links only an *installed* OpenBLAS (cmake --install build
  --prefix ../pkgs/openblas; headers in include/openblas/). Referencing the
  source + build trees instead duplicates the same headers/macros.
