# blpp — element-wise physics kernels for BLAS

Physics codes (MHD, CFD, particle-in-cell, rigid bodies) move **fields of
3-component vectors**, and their hot loops are element-wise: per-cell
cross products, dot products, Hadamard products, squared magnitudes —
never reduced across the field. Standard BLAS has exactly one element-wise
operation (`axpy`), so everything else degenerates into hand-written loops
that a BLAS kernel would do better. This project defines those operations
— **20 kernels × 4 precisions (`s d c z`), plus a MATLAB-flavoured
front-end in C11, C++ and Fortran** — and maintains working backends.

Backends are pluggable by design: the semantics and the conformance
matter, the library does not. Today one backend ships:

| backend | status | details |
|---|---|---|
| [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) | working — one patch on a pinned commit | [`docs/backends/OPENBLAS.md`](docs/backends/OPENBLAS.md) |
| oneMATH (FOSS oneMKL), CLBlast, BLIS | planned | [`docs/TODO.md`](docs/TODO.md) |

## Quickstart (OpenBLAS backend)

To get started with OpenBLAS, just run the script:

```sh
git clone https://github.com/sacielo/blpp.git && cd blpp
./install/openblas.sh 4                  # jobs as arg 1; 1 on small machines
```

It clones OpenBLAS at the pinned commit (if not already there), applies
`physics_kernels.patch`, builds and installs into `pkgs/openblas/`,
requires OpenBLAS' own suites to pass, and **compiles** (without
running) the benchmark/test/example suite and the mini-apps. Then run
the tests yourself -
every `*_test` must end `OK (0 failures)`:

```sh
cd benchmark/build && ./cblas_test && ./v3blas_test && ./v3blas_cpp_test && ./v3blas_f90_test
```

Everything links against the *installed* tree in `pkgs/openblas/`:
headers in `include/openblas/` (`cblas.h` + the three front-end files),
static library in `lib/` (some distros install to `lib64/`, which the
benchmark CMake also finds).

## Using the APIs

Two layers, one library. The **raw layer** is the plain BLAS-style ABI,
`cblas_<p><kernel>`, always available with explicit `n` and per-component
strides (standard level-1 like `axpy` included - they are not
repeated on top). On top sits **`v3blas`**, a header-only, allocation-free
front-end in three languages over the same symbols:

| language | file | precision chosen… |
|---|---|---|
| C (C11) | `v3blas.h` | by `_Generic` on the arguments, or suffix: `v3cross_d(…)` |
| C++ (11+) | `v3blas.hpp` | by overload (no macros; include exactly one header) |
| Fortran (F2003+) | `v3blas.f90` | by argument types (generic module procedures) |

The whole API in three rules:

1. **The library never allocates.** Every operation is `void`; outputs
   are trailing arguments you provide (like the `y` of `cblas_daxpy`) and
   may alias inputs. Inputs are never modified.
2. **A 3-vector is three component arrays sharing one element stride.**
   In C: a `v3<P>` view — `{T *x, *y, *z; blasint inc, n;}` — built with
   `v3<P>_wrap(x, y, z, n)` (contiguous); its `.inc` is set by hand for
   the other field layouts: array-of-structs (3), subgrid slice (grid
   stride), broadcast constant (0 — each component repeats its own base,
   so `wrap(&zero, &one, &zero, n)` *is* the vector (0,1,0) everywhere).
   One stride per view is the tightened definition of a field: layouts
   that mix strides between components are not vector fields and live
   one layer down — the `cblas_*` calls keep per-component incs as the
   documented escape hatch. Indexing is `blasint`, strides are BLAS
   semantics (see [`docs/KERNELS.md`](docs/KERNELS.md) for the full
   contract).
3. **Complex is bilinear** (no conjugation anywhere); guard parameters
   (`eps`) and coefficients (`a`) are *real* scalars in every precision.

```c
#include <v3blas.h>                        /* C11; or <v3blas.hpp> for C++ */
v3d x = v3d_wrap(x1, x2, x3, n), y = v3d_wrap(y1, y2, y3, n);
v3d w = v3d_wrap(w1, w2, w3, n);           /* output: your arrays */
v3cross(x, y, w);                          /* w = x∧y            */
v3drag(x, 0.5, 1e-30, w);                  /* w = 0.5·x·√(x·x+ε) */
v1norm(q, r1, n);                        /* one-vector ops keep explicit n */
```

Fortran hands the same kernels **whole fields**: a 3-vector is one
`u(n, 3)` array — column-major, so each column `u(:, c)` *is* one
contiguous component array. No `n`, no strides, the shape is guarded at
runtime:

```fortran
use v3blas
real(8) :: x(n,3), y(n,3), w(n,3), r(n)
call v3cross(x, y, w)            ! w = x∧y
call v3norm_unit(x, eps, r, w)   ! r = √(x·x+eps), w = x/r
```

Raw F77 triplets (`call d3cross(n, x1, 1, …)`) stay one call away; the
front-ends are convenience, not a gate.

### Which operations

The 18 front-end operations - equations, raw `cblas_*` prototypes, and
the stride/aliasing/NaN contracts of all 20 kernels (the stencils
`lap5`/`lap7` live at the raw layer only) - are listed in one place:
**[`docs/KERNELS.md`](docs/KERNELS.md)**. Shown for `d`; the generic
names pick the precision from the arguments, and `n` is explicit for the
one-vector ops (`x.n`, Fortran `size(x,1)`, otherwise).

## Documentation map

| file | what |
|---|---|
| [`docs/KERNELS.md`](docs/KERNELS.md) | the kernel catalog: quick list, equations, prototypes, stride/aliasing/NaN contracts |
| [`docs/AGENT.md`](docs/AGENT.md) | how to implement kernels (rules, traps, test discipline) — read before writing code |
| [`docs/backends/OPENBLAS.md`](docs/backends/OPENBLAS.md) | OpenBLAS specifics: patch workflow, wiring, threading, build flags |
| [`docs/TODO.md`](docs/TODO.md) | open work + closed decisions |
| [`docs/specs/`](docs/specs/) | the originating specs, frozen (`blasKernels.md`, `grand_plan.md`) |
| `benchmark/` | the test/bench/example suite (standalone CMake, links only an installed backend) |
| `miniapp/` | working physics codes on the public API — the API's harshest reviewer (2D Sod tube; batched rigid tops where BLAS-2 joins) |

## Roadmap — every BLAS, not this BLAS

Every physics project re-implements cross products in its first week,
forever. The fix is not one fast library; it is **one symbol name in every
BLAS**: users write `v3cross(A, B, W)`, link their local BLAS, go. The
durable deliverable is a versioned ABI + semantics spec plus a
conformance suite runnable against any vendor library at runtime —
providers adopt standards they can test against, not opinions they must
trust. OpenBLAS (this repo) is the working reference; BLIS, the BLAS
Extensions forum and the vendor libraries follow. The two-layer split is
what keeps it portable: providers implement only the C ABI layer, users
only the front-end layer. Committees ratify shipping practice — the spec
and forum pitch come *after* first open merges, not before.
