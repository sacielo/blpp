# blpp — element-wise physics kernels for BLAS

Physics codes (MHD, CFD, particle-in-cell, rigid bodies) move **fields of
2-/3-component vectors**, and their hot loops are element-wise: per-cell
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

```sh
git clone --recurse-submodules https://github.com/sacielo/blpp.git && cd blpp
./docs/backends/OPENBLAS.install.sh     # patch + build + install -> pkgs/openblas
                                        # (JOBS=1 on small machines)

# OpenBLAS' own suites must end with 0 failures (220 and 2765 checks)
./openblas/build/utest/openblas_utest
./openblas/build/utest/openblas_utest_ext

# the front-end suite: tests verify, benches time, examples teach
cmake -S benchmark -B benchmark/build && cmake --build benchmark/build
./benchmark/build/cblas_test          # every *_test must end OK (0 failures)
```

Everything links against the *installed* tree in `pkgs/openblas/`:
headers in `include/openblas/` (`cblas.h` + the three front-end files),
`lib/libopenblas.a`.

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
2. **A 3-vector is three component arrays.** In C: a `v3<P>` view —
   `{T *x, *y, *z; blasint incx, incy, incz, n;}` — built with
   `v3<P>_wrap(x, y, z, n)`; its fields are public, so strided or
   scattered components are just set by hand. `inc = 0` broadcasts
   (constant fields are free). Indexing is `blasint`, strides are BLAS
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

### The 18 front-end operations

Shown for `d`; generic names pick the precision from the arguments.
Equations are element-wise; `n` is explicit for one-vector ops and comes
from `x.n` (Fortran: `size(x,1)`) otherwise. Full semantics, edge
contracts and the raw `cblas_*` prototypes for all 20 kernels:
[`docs/KERNELS.md`](docs/KERNELS.md).

| equation | call | equation | call |
|---|---|---|---|
| `r = q.*t + a` | `v1xypa(q, t, a, r, n)` | `r = (x∧y)·w` | `v3crossdot(x, y, w, r)` |
| `r = √(q·q)` | `v1norm(q, r, n)` | `r = (x∧y)·(x∧y)` | `v3crosssqr(x, y, r)` |
| `r = x·y` | `v3dot(x, y, r)` | `u = x∧y, v = x∧w` | `v3crossxy_crossxz(x, y, w, u, v)` |
| `w = x.*y` | `v3had(x, y, w)` | `u = x∧y, r = x·w` | `v3crossxy_dotxz(x, y, w, u, r)` |
| `r = x·x` | `v3sqr(x, r)` | `r = x·y, q = x·w` | `v3dotxy_dotxz(x, y, w, r, q)` |
| `w = x∧y` | `v3cross(x, y, w)` | `r = (x∧y)∧z` | `v3crosscross(x, y, z, r)` |
| `w = a·(x∧y)` | `v3crossscal(x, y, a, w)` | `r = √(x·x+ε), u = x/r` | `v3norm_unit(x, eps, r, u)` |
| `r = a·x·√(x·x+ε)` | `v3drag(x, a, eps, r)` | `q = x − a(x·y)y/(y·y+ε)` | `v3refl(x, y, a, eps, q)` |
| `m = s·x, k = ½s(x·x)` | `v3mom_ke(s, x, m, k)` | `r = a(x∧y)/(y·y+ε)` | `v3exb(x, y, a, eps, r)` |
| | | (`lap5`, `lap7`: raw layer only) | |

## Documentation map

| file | what |
|---|---|
| [`docs/KERNELS.md`](docs/KERNELS.md) | the kernel catalog: quick list, equations, prototypes, stride/aliasing/NaN contracts |
| [`docs/AGENT.md`](docs/AGENT.md) | how to implement kernels (rules, traps, test discipline) — read before writing code |
| [`docs/backends/OPENBLAS.md`](docs/backends/OPENBLAS.md) | OpenBLAS specifics: patch workflow, wiring, threading, build flags |
| [`docs/TODO.md`](docs/TODO.md) | open work + closed decisions |
| [`docs/specs/`](docs/specs/) | the originating specs, frozen (`blasKernels.md`, `grand_plan.md`) |
| `benchmark/` | the test/bench/example suite (standalone CMake, links only an installed backend) |

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
