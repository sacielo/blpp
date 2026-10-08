# v3blas — higher-level interface for the physics extension kernels

A header-only, MATLAB/Octave-style API on top of the 13 element-wise
extension kernels (`s`/`d`/`c`/`z`), in three front-ends over one set of
`cblas_*` entry points:

| language            | file          | how to pick precision          |
|---------------------|---------------|--------------------------------|
| C (C11)             | `v3blas.h`    | `_Generic` macros, or `_s.._z` suffixes |
| C++ (C++11 or 98+)  | `v3blas.hpp`  | real function overloads        |
| Fortran (F2003+)    | `v3blas.f90`  | generic module procedures      |

All three live at the OpenBLAS root next to `cblas.h` (the header and the
module are installed with it); link `-lopenblas` as usual. Nothing is
compiled ahead of time: the C/C++ front-ends are `static inline` /
`inline`, the Fortran one is a single source file to compile.

**The library never allocates.** You own every array, always.

## Vectors

A **1-vector is a plain C array** of `n` elements, unit stride — exactly
what BLAS has always taken. No struct, no wrapper: pass the pointer.
(Need a strided 1-vector? Call the `cblas_*` entry point directly.)

A **3-vector** is a field of 3-component vectors: three component arrays,
each with its own stride — exactly the shape the cblas extension calls
take, so the wrapper is a zero-copy one-liner. Index fields are
`blasint`, the index type the cblas extension prototypes declare.

```c
typedef struct { float  *x, *y, *z; blasint incx, incy, incz, n; } v3s;
typedef struct { double *x, *y, *z; blasint incx, incy, incz, n; } v3d;
typedef struct { v3blas_floatcomplex  *x, *y, *z; blasint incx, incy, incz, n; } v3c;
typedef struct { v3blas_doublecomplex *x, *y, *z; blasint incx, incy, incz, n; } v3z;
```

- `v3<P>_wrap(x, y, z, n)` bundles three unit-stride arrays into a view.
  The struct fields are public: set `.incx/.incy/.incz` by hand for
  strided components.
- `v3blas_floatcomplex = { float  re, im; }`, `v3blas_doublecomplex = { double re, im; }`
  — the interleaved (re, im) layout the cblas `c`/`z` extension routines
  consume. Separate re/im pointers would force a conversion copy on every
  call, so the complex structs keep the cblas layout as-is. Both real and
  imaginary parts are always explicit (`.re` / `.im`).
- The suffixed names (`v3cross_d(...)`, `v3cross_z(...)`, ...) are the
  canonical ones in every front-end; the generic names dispatch to them.

## Outputs are arguments (the only rule to remember)

Every operation is a `void` procedure: outputs are **trailing pointer
arguments you provide**, exactly like the `y` of `cblas_daxpy`. Nothing
is allocated, nothing is freed, nothing is returned. Outputs may alias
the inputs. Vector outputs are written unit stride (that is what the
kernels write).

## The 13 operations (C / C++)

Equation → call (shown for `d`; the generic name selects the precision
from an output pointer type under C11 `_Generic`, or just suffix the
name). Scalars: real by value; complex by value of `v3blas_*complex`
(construct with `v3blas_cc(re, im)` / `v3blas_zc(re, im)`).
`n` is explicit for pure-1-vector ops and comes from `x.n` when any
input is a `v3<P>`.

| equation                          | v3blas call                                   |
|-----------------------------------|-----------------------------------------------|
| `y = alpha*x + y`                 | `v1axpy(alpha, x, y, r, n);`                  |
| `r = q.*t + a`                    | `v1xypa(q, t, a, r, n);`                      |
| `r = x1*y1+x2*y2+x3*y3`           | `v3dot(x, y, r);`                             |
| `w = x.*y`  (per component)       | `v3had(x, y, wx, wy, wz);`                    |
| `r = x1²+x2²+x3²`                 | `v3sqr(x, r);`                                |
| `r = sqrt(q·q)`                   | `v1norm(q, r, n);`                            |
| `w = x∧y`                         | `v3cross(x, y, wx, wy, wz);`                  |
| `w = a·(x∧y)`                     | `v3crossscal(x, y, a, wx, wy, wz);`           |
| `r = (x∧y)·w`                     | `v3crossdot(x, y, w, r);`                     |
| `r = (x∧y)·(x∧y)`                 | `v3crosssqr(x, y, r);`                        |
| `u = x∧y,  v = x∧w`               | `v3crossxy_crossxz(x, y, w, ux,uy,uz, vx,vy,vz);` |
| `u = x∧y,  r = x·w`               | `v3crossxy_dotxz(x, y, w, ux,uy,uz, r);`      |
| `r = x·y,  q = x·w`               | `v3dotxy_dotxz(x, y, w, r, q);`               |

Under C++ include `<v3blas.hpp>` instead — same signatures, resolved by
overload (no `_Generic`, no suffixes, and the macros are not defined).
Both headers are C- and C++-clean; include exactly one.

## The Fortran front-end (`v3blas.f90`)

Compile the module, then your program; same names via F2003 generic
interfaces, precision picked from the argument types. BLAS-style
argument order: `n` first, then inputs, then outputs; explicit-shape or
assumed-shape arrays, unit stride only.

```fortran
use v3blas
call v3cross(n, x1, x2, x3, y1, y2, y3, w1, w2, w3)  ! w = x∧y
call v3dot(n, x1, x2, x3, y1, y2, y3, r)             ! r = x·y
call v1axpy(n, alpha, x, y)                          ! y = alpha*x + y
```

`v1axpy` is the in-place BLAS one here (y is updated); the C front-end
additionally keeps `y` untouched by writing into `r` first. `v1norm` on
complex data is the principal complex `sqrt(q·q)`, on real data `|q|`.

## Conventions

- `n` travels with the first `v3<P>` input (`x.n`) in C/C++; the caller
  guarantees all inputs agree (BLAS trust-the-caller convention).
- Inputs are **never modified** (MATLAB `z = cross(x, y)` leaves `x`, `y`
  alone — same here). `v1axpy` therefore copies `y` into `r` before the
  in-place `cblas_*axpy` (one extra memory pass for the invariant).
- The C header is one macro (`V3BLAS_DEFINE`) instantiated four times —
  real scalars by value, complex scalars as pointers to the two-
  component struct, exactly matching the cblas prototypes.
- The Fortran module is generated (52 bind(C) wrappers) but is ordinary
  F2003 source; `gfortran -std=f2008 -Wall` clean.

## Verification

`benchmark/v3blas_test.c` (plain C): all 13 operations at all 4
precisions against a long-double reference, generic and suffixed names,
plus a strided (inc = 2) `v3cross_d` smoke case.

`benchmark/v3blas_example.c` tours the API; `v3blas_bench.cpp` times all
13 × 4 precisions against the raw `cblas_*` calls (err must be 0).

`benchmark/v3blas_hpp_test.cpp`: C++ overloads, bitwise (`memcmp`)
against `cblas_*` for every op × precision.

`benchmark/v3blas_f90_test.f90`: all 13 generics × 4 precisions through
the module, checked against the equations recomputed with Fortran array
expressions.

```
gcc  -O2 -std=c11 -I pkgs/openblas/include/openblas benchmark/v3blas_test.c \
     -L pkgs/openblas/lib -lopenblas -lm -o /tmp/v3blas_test && /tmp/v3blas_test
g++  -O2 -std=c++11 -I pkgs/openblas/include/openblas benchmark/v3blas_hpp_test.cpp \
     -L pkgs/openblas/lib -lopenblas -lm -o /tmp/v3blas_hpp_test && /tmp/v3blas_hpp_test
gfortran -Wall pkgs/openblas/include/openblas/v3blas.f90 benchmark/v3blas_f90_test.f90 \
     -L pkgs/openblas/lib -lopenblas -lm -o /tmp/v3blas_f90_test && /tmp/v3blas_f90_test
```
