# v3blas — higher-level interface for the physics extension kernels

A header-only, MATLAB/Octave-style API on top of the 19 element-wise
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

Every operation is a `void` procedure: outputs are **trailing arguments
you provide** — exactly like the `y` of `cblas_daxpy`. Nothing is
allocated, nothing is freed, nothing is returned. Outputs may alias the
inputs.

- A **scalar output** (`r`, `k`) is a plain `T *` array.
- A **3-vector output** is a `v3<P>` view you built with `v3<P>_wrap`.
  Its `.incx/.incy/.incz` are the *write* strides for the three
  components (a view with `incx=2` scatters the x-component). The length
  `n` always comes from the input view `x.n` — the output's own `.n` is
  unused.

So one view type serves inputs and outputs; there is no second
"result" shape to learn.

## The 19 operations (C / C++)

Equation → call (shown for `d`; the generic name selects the precision
from the first `v3<P>` input — or from an output pointer for pure-1-vector
ops — under C11 `_Generic`, or just suffix the name). Scalars: real by
value; complex by value of `v3blas_*complex` (construct with
`v3blas_cc(re, im)` / `v3blas_zc(re, im)`).
`n` is explicit for pure-1-vector ops and comes from `x.n` when any
input is a `v3<P>`.

| equation                          | v3blas call                             |
|-----------------------------------|-----------------------------------------|
| `y = alpha*x + y`                 | `v1axpy(alpha, x, y, r, n);`            |
| `r = q.*t + a`                    | `v1xypa(q, t, a, r, n);`                |
| `r = x1*y1+x2*y2+x3*y3`           | `v3dot(x, y, r);`                       |
| `w = x.*y`  (per component)       | `v3had(x, y, w);`                       |
| `r = x1²+x2²+x3²`                 | `v3sqr(x, r);`                          |
| `r = sqrt(q·q)`                   | `v1norm(q, r, n);`                      |
| `w = x∧y`                         | `v3cross(x, y, w);`                     |
| `w = a·(x∧y)`                     | `v3crossscal(x, y, a, w);`              |
| `r = (x∧y)·w`                     | `v3crossdot(x, y, w, r);`               |
| `r = (x∧y)·(x∧y)`                 | `v3crosssqr(x, y, r);`                  |
| `u = x∧y,  v = x∧w`               | `v3crossxy_crossxz(x, y, w, u, v);`     |
| `u = x∧y,  r = x·w`               | `v3crossxy_dotxz(x, y, w, u, r);`       |
| `r = x·y,  q = x·w`               | `v3dotxy_dotxz(x, y, w, r, q);`         |
| `r = (x∧y)∧z`                     | `v3crosscross(x, y, z, r);`             |
| `r = sqrt(x·x+eps), u = x/r`      | `v3norm_unit(x, eps, r, u);`            |
| `q = x − a(x·y)y/(y·y+eps)`       | `v3refl(x, y, a, eps, q);`              |
| `r = a(x∧y)/(y·y+eps)`            | `v3exb(x, y, a, eps, r);`               |
| `r = a*x*sqrt(x·x+eps)`           | `v3drag(x, a, eps, r);`                 |
| `m = s*x,  k = 0.5*s*(x·x)`       | `v3mom_ke(s, x, m, k);`                 |

The six CFD-menu ops take their guard `eps` and coefficient `a` as
**real** scalars by value in every precision (`float` for `s`/`c`,
`double` for `d`/`z`); `s` in `v3mom_ke` is a field (a per-element
scalar array, e.g. density), not a scalar.

Under C++ include `<v3blas.hpp>` instead — same signatures, resolved by
overload (no `_Generic`, no suffixes, and the macros are not defined).
Both headers are C- and C++-clean; include exactly one.

## The Fortran front-end (`v3blas.f90`)

Compile the module, then your program; same names via F2003 generic
interfaces, precision picked from the argument types. A 3-vector is one
**field array `u(n, 3)`**: a row per point, and in column-major storage
each column `u(:, c)` is one contiguous component array — exactly the
SoA blocks the kernels read. There is no view struct in Fortran; the
field *is* the data, and the call takes it whole: no `n`, no strides
(`n = size(u, 1)` inside). Argument order matches `v3blas.h`, outputs
trail. The one-vector ops are plain arrays and keep the BLAS-style
explicit `n`.

```fortran
use v3blas
real(8) :: x(n,3), y(n,3), w(n,3), r(n)
call v3cross(x, y, w)             ! w = x∧y
call v3dot(x, y, r)               ! r = x·y
call v3drag(x, a, eps, w)         ! w = a*x*sqrt(x·x + eps)
call v3norm_unit(x, eps, r, w)    ! r = sqrt(x·x+eps), w = x/r
call v1axpy(n, alpha, x1, y1)     ! y = alpha*x + y (one-vector: explicit n)
```

Each field is guarded at runtime for `size(u, 2) == 3` (passing a
transposed `u(3, n)` would otherwise be silently misread); agreement of
`n` between fields is caller trust, BLAS-style. Output fields may alias
inputs. Passing a non-contiguous actual (a slice, a transpose) makes
gfortran route the call through a copy — correct, but it defeats the
point; pass whole `(n, 3)` arrays. Component-by-component arguments and
non-unit strides remain available one layer down: either the
`cblas_*3cross` entry points or the raw F77 symbols (`call d3cross(n, x1,
1, x2, 1, x3, 1, ...)`) — the module is convenience, not a gate.

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
- The Fortran module is generated (76 wrappers over 76 bind(C)
  interfaces: 16 field-form ops + 3 one-vector ops, 4 precisions) but is
  ordinary F2003 source; `gfortran -std=f2008 -Wall` clean. Dummy fields
  are all-assumed-shape `(:,:)` with a runtime shape guard, which is the
  form gfortran accepts and associates without copies.
- Public argument order follows the formula in the name: `1xypa(n, q,
  t, a, r, …)` literally reads *x y plus a* — vectors first, then the
  scalar, output last (same in the f77 entry). The CFD family obeys the
  rule too (`3drag(n, x…, a, eps, r…)`); `3crossscal(n, a, x…, y…, w…)`
  is the one deliberate exception, keeping the BLAS `scal` tradition of
  a leading scalar.
- Raw-layer stride contract (probed and pinned by the `contract` block
  in `cblas_test.c`): `n ≤ 0` is a no-op; `inc < 0` walks the span
  `base..base+(n-1)|inc|` **backwards** (the interfaces shift the base
  pointer, so pass the forward start); `inc = 0` broadcasts a scalar;
  `w_c` may alias `x_c`/`y_c` in place, but NEVER alias across
  components (they run sequentially) and partial overlap is UB; NaN
  propagates silently (`eps < 0` making a sqrt argument negative →
  IEEE NaN, no trap).

## Verification

`benchmark/v3blas_test.c` (plain C): all 19 operations at all 4
precisions against a long-double reference, generic and suffixed names,
plus a strided smoke case (inc = 2 on the *output* view of `v3cross`).
All four `_test` programs (cblas, C, C++, F90) take an optional
`s|d|c|z` argument to run a single precision; default is all four.
`benchmark/cblas_test.c` additionally pins the raw-layer contract above
(five `d contract …` checks).

`benchmark/v3blas_example.c` tours the API; `v3blas_bench.cpp` times all
19 × 4 precisions against the raw `cblas_*` calls (timing only —
correctness is the tests' job).

`benchmark/v3blas_cpp_test.cpp`: C++ overloads, bitwise (`memcmp`)
against `cblas_*` for every op × precision.

`benchmark/v3blas_f90_test.f90`: all 19 generics × 4 precisions through
the module in field form, checked against the equations recomputed with
Fortran array expressions over the same fields. Accepts `s|d|c|z` as an
argument to run one precision.

```
gcc  -O2 -std=c11 -I pkgs/openblas/include/openblas benchmark/v3blas_test.c \
     -L pkgs/openblas/lib -lopenblas -lm -o /tmp/v3blas_test && /tmp/v3blas_test
g++  -O2 -std=c++11 -I pkgs/openblas/include/openblas benchmark/v3blas_cpp_test.cpp \
     -L pkgs/openblas/lib -lopenblas -lm -o /tmp/v3blas_cpp_test && /tmp/v3blas_cpp_test
gfortran -Wall pkgs/openblas/include/openblas/v3blas.f90 benchmark/v3blas_f90_test.f90 \
     -L pkgs/openblas/lib -lopenblas -lm -o /tmp/v3blas_f90_test && /tmp/v3blas_f90_test
```
