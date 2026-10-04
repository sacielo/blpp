# Answers found by exploring the OpenBLAS tree

## Where the kernel code goes
- OpenBLAS's generic kernels live in `kernel/generic/` (the tree has `kernel`, not `kernels`). A new kernel file is dropped in there and compiled for every architecture build, exactly like the existing `geadd`/`axpy` kernels.
- One C file is enough per kernel family: OpenBLAS compiles the same file 4 times (s/d/c/z) with `-D` precision macros (`FLOAT` = float or double). Real (s, d) can share one file; complex (c, z) usually get their own file because the arithmetic differs. That's how geadd is done, and I'll copy that.
- So: yes, OpenBLAS uses macros for precision, and it means **2 C files total** (one real, one complex) for this kernel, not 4 and not 1.

## How a kernel becomes "part of openblas"
The plumbing (all copied from how `geadd`/`axpy` are done):
1. `kernel/generic/<name>.c` — the loop itself.
2. `kernel/Makefile.L1` + `kernel/CMakeLists.txt` — compile it, one object per precision.
3. `common_*.h` — prototypes and the per-precision dispatch macros (this is how the code picks `s…/d…/c…/z…` at compile time).
4. `interface/<name>.c` — the entry points. One file gives **both** the Fortran symbol (`s1xtypa_`) and the CBLAS symbol (`cblas_s1xtypa`); it's compiled twice, once per flavor. This satisfies "fortran and cblas".
5. `cblas.h` — declaration of the `cblas_*` entry.
6. `exports/gensymbol` — the shared library only exports names listed here; the new 8 symbols must be added or they won't be visible to your code.

## Threading
- "OpenBLAS' own threading scheme" for level-1 ops = the interface splits `n` across threads via the same helper `axpy` uses (`blas_level1_thread`), and falls back to one thread for small n or zero strides. Copying axpy covers this; no new threading machinery needed.

## FMA
- The loop is plain C (`r[i] = q[i]*t[i] + a`); whether the compiler fuses the mul+add depends on the optimization level, which this build does not set by default. ARMv6 (this machine) has no true FMA instruction — the fused form is VFPv2 `VMLA`, emitted by gcc at `-O2`+. See the optimization section below.

## "unit and cblas test"
- The standard Netlib test suites in `test/` and `ctest/` can only test *stock* BLAS routines, so new-kernel tests live in the `utest/` framework:
  - **unit test** = `utest/test_<name>.c`, calls the Fortran-style entry (like `test_axpy.c` does).
  - **cblas test** = `utest/test_extensions/test_<p><name>.c` (one per precision, like `test_sgeadd.c`), calls the `cblas_*` entry.
- Both are registered in `utest/Makefile` and `utest/CMakeLists.txt`. `make test` builds and runs them.

## Building on this machine
- It's a 1-core ARMv6 with ~420 MB RAM. I'll build serially and skip LAPACK (`NO_LAPACK=1`) for the verification build, since the kernel and its tests don't need it; expect it to take a while.

## Optimization / performance of the physics kernels
- The stock build compiles generic C kernels with **no optimization flags at all** (`-O0`), so a plain loop runs far below the hand-tuned asm kernels: `daxpy` on this machine uses `vmla.f64` (VFPv2 fused multiply-add), 4x unrolling, multi-register `vldmia`/`vstmia`, and `pld` prefetch. Byte counts are identical (3n doubles each), so the gap is instructions/element + latency hiding, not bandwidth.
- In-kernel fix: the `1xypa` kernel uses the standard axpy structure — unrolled fast path for `inc==1` plus the strided fallback loop (the generic `scal`-style idiom), mirroring axpy's fast/slow dispatch. Effect: 3.8x at n=1000, 1.17x at n=100000.
- Build-flag fix: CMake option **`PHYS_KERNELS_O3`** (default ON) compiles only the physics extension kernels with `-O3`; a configure-time warning is emitted in both cases (ON and OFF) so the state is always visible and it can be changed with `-DPHYS_KERNELS_O3=OFF`. Future kernels opt in by calling `PhysicsKernelOptimize(<generated wrappers>)` in their `kernel/CMakeLists.txt` entry (only `1xypa` does so today).
- Measured at n=100000, double precision. `d1xypa` ms/run: 19.4 (naive -O0 loop) → 16.5 (unrolled, -O0) → **11.3 (-O3)**, stable across sessions. `daxpy` wobbles 8.9–11.3 ms per run (1-core thermal/frequency effects), so the ratio is 1.01× in some runs and 1.22× against its best time — i.e. the generic C kernel is now at parity with (or within noise of) the tuned asm kernel, versus 2.1× before. The worst-case residual ~1.2× is consistent with the `pld` prefetch and hand scheduling in the `.S` kernel; gcc for ARMv6 emits no prefetch.
- At small n (1000) the generic kernel is still ~2× off `daxpy`: the working set is a few cache lines, so it's a latency/constant-overhead game where hand-scheduled asm wins. Not worth chasing in generic C.
- `-O3` verified in the shipped library: the four `1xypa` objects contain fused `vmla` instructions (7 in s/d, 21 in c/z), vs none at `-O0`.
- We deliberately do **not** use `-Ofast`: it adds `-ffast-math` (reassociation etc.), which changes the per-operation rounding that BLAS correctness guarantees are built on. `-O3` is the safe maximum.
- Benchmark lives in `/home/sac/blpp/benchmark/` (`benchmark.cpp` + own CMakeLists that only links the pre-built `libopenblas.a`, never rebuilds it): vector size is a CLI argument (default 1000), reports GFLOP/s and Melem/s, and every kernel is checked against a long-double reference (non-zero exit on mismatch). New kernels = one `bench_*()` function + one line in `main()`.
