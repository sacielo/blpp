# v3blas.h — higher-level interface for the physics extension kernels

Header-only, MATLAB/Octave-style API on top of the 13 element-wise
extension kernels (`s`/`d`/`c`/`z`). Lives at the OpenBLAS root next to
`cblas.h`; links against `libopenblas` as usual. No build-system changes
beyond installing the header.

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
  strided components (`_view` was just a wrap with incs, so it is gone).
- `v3blas_floatcomplex = { float  re, im; }`, `v3blas_doublecomplex = { double re, im; }`
  — the interleaved (re, im) layout the cblas `c`/`z` extension routines
  consume. Separate re/im pointers would force a conversion copy on every
  call, so the complex structs keep the cblas layout as-is. Both real and
  imaginary parts are always explicit (`.re` / `.im`).
- The precision letter (`s d c z`) is a suffix, the cblas idiom:
  `v3cross_d(...)`, `v3cross_z(...)`.

## Ownership (the only rule to remember)

There is nothing to remember: nothing is allocated, nothing is freed.
Every operation writes into output **arrays you provided** and returns a
view of them (a `T *` for scalar results, a `v3<P>` — or a struct of
them — for vector ones). Outputs may alias the inputs.

## The 13 operations

MATLAB-style equation → v3blas call (shown for `d`; suffix the name with
`s`/`c`/`z` as needed). Scalars: real by value; complex by value of
`v3blas_*complex` (construct with `v3blas_cc(re, im)` / `v3blas_zc(re, im)`).
`n` is explicit for pure-1-vector ops and comes from `x.n` when any
input is a `v3<P>`.

| MATLAB-ish                        | v3blas call                                   | returns |
|-----------------------------------|----------------------------------------------|---------|
| `z = cross(x, y)`                 | `z = v3cross_d(x, y, wx, wy, wz);`           | `v3d`   |
| `z = a*(cross(x, y))`             | `z = v3crossscal_d(x, y, a, wx, wy, wz);`    | `v3d`   |
| `z = x.*y`  (per component)       | `z = v3had_d(x, y, wx, wy, wz);`             | `v3d`   |
| `d = x(1)*y(1)+x(2)*y(2)+x(3)*y(3)` (element-wise) | `r = v3dot_d(x, y, r);`            | `double *` |
| `d = x(1)^2+x(2)^2+x(3)^2`        | `r = v3sqr_d(x, r);`                         | `double *` |
| `d = sqrt(q·q)`                   | `r = v1norm_d(q, r, n);`                     | `double *` |
| `r = q.*t + a`                    | `r = v1xypa_d(q, t, a, r, n);`               | `double *` |
| `y = alpha*x + y`                 | `r = v1axpy_d(alpha, x, y, r, n);`           | `double *` |
| `d = (x∧y)·w`                     | `r = v3crossdot_d(x, y, w, r);`              | `double *` |
| `d = (x∧y)·(x∧y)`                 | `r = v3crosssqr_d(x, y, r);`                 | `double *` |
| `[u, v] = cross(x,y), cross(x,w)` | `p = v3crossxy_crossxz_d(x, y, w, ux,uy,uz, vx,vy,vz);` → `p.u`, `p.v` | `v3uv_d` |
| `[u, r] = cross(x,y), x·w`        | `p = v3crossxy_dotxz_d(x, y, w, ux,uy,uz, r);` → `p.u`, `p.r` | `v3ur_d` |
| `[r, q] = x·y, x·w`               | `p = v3dotxy_dotxz_d(x, y, w, r, q);` → `p.r`, `p.q` | `rq_d` |

Output arrays are always unit stride (that is what the kernels write).

### Multi-output kernels

C has no MATLAB `[u, v] = f(...)`. Each multi-output kernel returns a
small struct whose members are named exactly like the equation's
outputs (the equation tokens from the benchmark table) and are plain
views of the caller's output arrays:

```c
typedef struct { v3d u, v; }        v3uv_d;  /* u = x∧y,  v = x∧w  */
typedef struct { v3d u; double *r; blasint n; } v3ur_d;  /* u = x∧y, r = x·w */
typedef struct { double *r, *q; blasint n; }   rq_d;    /* r = x·y,  q = x·w  */
```

## Conventions

- `n` travels with the first `v3<P>` input (`x.n`); the caller
  guarantees all inputs agree (BLAS trust-the-caller convention).
- Inputs are **never modified** (MATLAB `z = cross(x, y)` leaves `x`, `y`
  alone — same here). `v1axpy` therefore copies `y` into `r` before the
  in-place `cblas_*axpy` (one extra memory pass for the invariant).
- Header is generated from one macro (`V3BLAS_DEFINE`) instantiated four
  times — real scalars are passed by value, complex scalars as pointers
  to a two-component struct, exactly matching the cblas prototypes.
  Everything is `static inline`; nothing else to link.

## Verification

`benchmark/v3blas_test.c` (plain C, ad-hoc compile against the installed
`libopenblas.a`, not part of the OpenBLAS utest suite): runs all 13
operations at all 4 precisions against a long-double reference, plus a
strided (inc = 2) `v3cross_d` smoke case. (Strided 1-vectors are no
longer part of this API — that is what `cblas_*` is for.)

```
gcc -O2 -I pkgs/openblas/include/openblas benchmark/v3blas_test.c \
    -L pkgs/openblas/lib -lopenblas -lm -o /tmp/v3blas_test && /tmp/v3blas_test
```
