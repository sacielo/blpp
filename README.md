# blpp — physics kernels for OpenBLAS

A set of **element-wise BLAS extension kernels for field-based physics
computations**, implemented in [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS)
(pinned at v0.3.34, carried as a submodule, delivered as a single patch),
plus a header-only, allocation-free API in three front-ends (C11, C++,
Fortran).

## Rationale

Physics codes (MHD, particle-in-cell, rigid bodies, …) rarely move plain
vectors around; they move **fields of 2- or 3-component vectors**. The
hot loops are element-wise: per-cell dot products, cross products,
Hadamard products, squared magnitudes — always element-wise, never
reduced across the field. Standard BLAS has exactly one element-wise
operation (`axpy`), so everything else degenerates into hand-written
loops that a BLAS kernel could do better (blocking, vectorization,
threading) — if BLAS exposed them.

This repo adds the twelve kernels of [`blasKernels.md`](blasKernels.md)
to OpenBLAS — plain-C **generic** kernels (no per-arch assembly, so they
stay portable and reviewable), for all four precisions (`s d c z`), each
with the usual Fortran 77 / cblas interfaces, wired into the build and
export lists, and covered by unit + cblas tests in OpenBLAS' own utest
suite:

| kernel `<p>` | equation | kernel `<p>` | equation |
|---|---|---|---|
| `1xypa` | `r = q.*t + a` | `3crossdot` | `r = (x∧y)·w` |
| `3dot` | `r = x·y` | `3crosssqr` | `r = (x∧y)·(x∧y)` |
| `3had` | `w = x.*y` | `3crossxy_crossxz` | `u = x∧y, v = x∧w` |
| `3sqr` | `r = x·x` | `3crossxy_dotxz` | `u = x∧y, r = x·w` |
| `1norm` | `r = √(q·q)` | `3dotxy_dotxz` | `r = x·y, q = x·w` |
| `3cross` | `w = x∧y` | `3crossscal` | `w = a·(x∧y)` |

(13 operations in the API: the thirteenth reuses `cblas_*axpy`.)

On top of the raw `cblas_<p><kernel>` entry points sits **`v3blas`**, a
thin front-end with MATLAB-flavoured names ([`V3BLAS_API.md`](V3BLAS_API.md)):
3-vectors are three component arrays with independent strides, every
operation is `void` with outputs as trailing caller-owned arrays — the
library never allocates — and the precision is either a suffix
(`v3cross_d`) or picked from the argument types (`_Generic` in C11,
overloads in C++, generic module procedures in Fortran). The front-ends
are headers/a source file only: `v3blas.h`, `v3blas.hpp`, `v3blas.f90`,
installed next to `cblas.h`.

## Roadmap — the actual goal: every BLAS, not this BLAS

Every physics project re-implements cross products in its first week,
forever. The fix is not one fast library; it is **one symbol name in
every BLAS**, so users write `v3cross(A, B, W)`, link their local BLAS
(mine, the cluster's, the vendor's) and go — no paradigm choice, no new
dependency, nothing to install. Diffusion model: the `cblas_*` ABI,
which became portable precisely because every provider emits the same
names, not because of performance.

The durable deliverable is therefore a **versioned ABI + semantics
spec** (symbols, argument order, LP64/ILP64 variants, complex scalar
convention, increment/aliasing/NaN rules) plus a **conformance suite
runnable against any vendor library at runtime** (dlopen, no
recompile). Providers adopt standards they can test against, not
opinions they must trust. Order of attack: OpenBLAS (open, first
mover — this repo is its working implementation), BLIS (framework
built for extension; its group already added axpby), the BLAS
Extensions standardisation effort (the front that cuBLAS/rocBLAS/MKL
can follow once a standard exists), then vendor libraries on user
demand. FOMO after the second adopter. The two-layer split is what
keeps this portable: providers only ever implement the C ABI layer;
users only ever write the front-end layer.

## Quickstart

Requirements: git, CMake ≥ 3.16, a C compiler, and a Fortran compiler
(gfortran) for the unit tests and the Fortran front-end test. The patch
applies to the pinned submodule commit (OpenBLAS v0.3.34).

Everything below — clone, patch, build, install into `pkgs/openblas`,
run OpenBLAS' own tests, then build and run the v3blas tests and
benchmarks — is one paste:

```sh
# 1. Clone with the OpenBLAS submodule and apply the physics-kernel patch
git clone --recurse-submodules https://github.com/sacielo/blpp.git
cd blpp
git -C openblas apply ../physics_kernels.patch

# 2. Build OpenBLAS and install it to pkgs/openblas (benchmarks link only
#    against this installed tree; use -j1 on small/low-RAM machines)
cmake -S openblas -B openblas/build -DCMAKE_BUILD_TYPE=Release \
      -DNO_LAPACK=1
cmake --build openblas/build -j"$(nproc)"
cmake --install openblas/build --prefix "$PWD/pkgs/openblas"

# 3. OpenBLAS' own test suites (the *_ext suite holds the cblas tests of
#    the 12 new kernels; both suites must end with 0 failures)
./openblas/build/utest/openblas_utest
./openblas/build/utest/openblas_utest_ext

# 4. Build the v3blas tests/examples/benchmarks against the installed
#    tree and run them (every line must say OK; 188 + 2285 + 115 + 52 +
#    57 checks in total)
cmake -S benchmark -B benchmark/build
cmake --build benchmark/build -j"$(nproc)"
./benchmark/build/v3blas_test                     # all ops x 4 precisions vs long-double ref
./benchmark/build/v3blas_example | tail -n 3      # API tour
./benchmark/build/v3blas_bench                    # v3blas vs raw cblas: timing + bitwise err
./benchmark/build/v3blas_hpp_test | tail -n 1     # C++ overloads, bitwise vs cblas
./benchmark/build/v3blas_f90_test | tail -n 1     # Fortran generics vs Fortran-recomputed equations
```

The OpenBLAS build takes ~1.5–2 h on a single ARMv6 core and minutes on
a desktop; `v3blas_bench` re-times every op until stable (up to a few
minutes on slow hardware). Benchmarks on this repo's dev box report
v3blas == cblas with 0.000e+00 error (the interface costs only the call
itself); numbers in [`TODO.md`](TODO.md).

## Layout

| path | what |
|---|---|
| `openblas/` | OpenBLAS submodule (v0.3.34, unmodified upstream commit) |
| `physics_kernels.patch` | all kernel/interface/test wiring, as one `git diff` |
| `pkgs/openblas/` | install prefix for the patched build |
| `benchmark/` | v3blas tests, examples and benchmarks (standalone CMake, links only the installed tree) |
| `blasKernels.md` | the kernel spec this implements |
| `V3BLAS_API.md` | front-end design: vectors, ownership, the 13 ops, three languages |
| `TODO.md` | progress log, wiring points per kernel, benchmark data notes |
