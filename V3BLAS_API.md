# v3blas.h — higher-level interface for the physics extension kernels

Header-only, MATLAB/Octave-style API on top of the 13 element-wise
extension kernels (`s`/`d`/`c`/`z`). Lives at the OpenBLAS root next to
`cblas.h`; links against `libopenblas` as usual. No build-system changes
beyond installing the header.

## Vectors

A **1-vector** is a field of scalars (n elements, one stride). A
**3-vector** is a field of 3-component vectors: three component arrays,
each with its own stride — exactly the shape the cblas extension calls
take, so the wrapper is a zero-copy one-liner. Index fields are
`blasint`, the index type the cblas extension prototypes declare.

```c
typedef struct { float  *x;                    blasint inc, n; }       v1s;
typedef struct { double *x;                    blasint inc, n; }       v1d;
typedef struct { v3blas_floatcomplex  *x;       blasint inc, n; }       v1c;
typedef struct { v3blas_doublecomplex *x;       blasint inc, n; }       v1z;

typedef struct { float  *x, *y, *z;            blasint incx, incy, incz, n; } v3s;
typedef struct { double *x, *y, *z;            blasint incx, incy, incz, n; } v3d;
typedef struct { v3blas_floatcomplex  *x, *y, *z; blasint incx, incy, incz, n; } v3c;
typedef struct { v3blas_doublecomplex *x, *y, *z; blasint incx, incy, incz, n; } v3z;
```

- `v3blas_floatcomplex = { float  re, im; }`, `v3blas_doublecomplex = { double re, im; }`
  — the interleaved (re, im) layout the cblas `c`/`z` extension routines
  consume. Separate re/im pointers would force a conversion copy on every
  call, so the complex structs keep the cblas layout as-is. Both real and
  imaginary parts are always explicit (`.re` / `.im`).
- The precision letter (`s d c z`) is a suffix, the cblas idiom:
  `v3cross_d(...)`, `v3cross_z(...)`. `v3cross(x, y)` in the sketch maps to
  `v3d z = v3cross_d(x, y);`.
- The shape prefix states the **return type**: `v3*` functions return a
  3-vector, `v1*` functions return a 1-vector.

## Ownership (the only rule to remember)

- Results returned by the operation functions are heap-allocated
  (inc = 1); release them with the matching `_free()`.
- Inputs are **borrowed and never modified** (MATLAB `z = cross(x, y)`
  leaves `x`, `y` alone — same here).
- `v1*_new(n)` / `v3*_new(n)` allocate; `v1*_view` / `v3*_view` wrap
  user-provided arrays (borrowed — never `_free` those).
- `v1*_free` / `v3*_free` take a pointer and clear the struct; the
  combined frees (`v3uv_*_free` etc.) take the result struct by value and
  consume it.

## The 13 operations

MATLAB-style equation → v3blas call (shown for `d`; suffix the name with
`s`/`c`/`z` as needed). Scalars: real by value; complex by value of
`v3blas_*complex` (construct with `v3blas_cc(re, im)` / `v3blas_zc(re, im)`).

| MATLAB-ish                        | v3blas call                                   | returns |
|-----------------------------------|----------------------------------------------|---------|
| `z = cross(x, y)`                 | `z = v3cross_d(x, y);`                       | `v3d`   |
| `z = a*(cross(x, y))`             | `z = v3crossscal_d(x, y, a);`                | `v3d`   |
| `z = x.*y`  (per component)       | `z = v3had_d(x, y);`                         | `v3d`   |
| `d = x(1)*y(1)+x(2)*y(2)+x(3)*y(3)` (element-wise) | `r = v3dot_d(x, y);` | `v1d`   |
| `d = x(1)^2+x(2)^2+x(3)^2`        | `r = v3sqr_d(x);`                            | `v1d`   |
| `d = sqrt(q.*q)`                  | `r = v1norm_d(q);`                           | `v1d`   |
| `r = q.*t + a`                    | `r = v1xypa_d(q, t, a);`                     | `v1d`   |
| `y = alpha*x + y`                 | `y = v1axpy_d(x, y, alpha);` (new `y`)       | `v1d`   |
| `d = (x∧y)·w`                     | `r = v3crossdot_d(x, y, w);`                 | `v1d`   |
| `d = (x∧y)·(x∧y)`                 | `r = v3crosssqr_d(x, y);`                    | `v1d`   |
| `[u, v] = cross(x,y), cross(x,w)` | `p = v3crossxy_crossxz_d(x, y, w);` → `p.u`, `p.v` | `v3uv_d` |
| `[u, r] = cross(x,y), x·w`        | `p = v3crossxy_dotxz_d(x, y, w);` → `p.u`, `p.r` | `v3ur_d` |
| `[r, q] = x·y, x·w`               | `p = v3dotxy_dotxz_d(x, y, w);` → `p.r`, `p.q` | `v1rq_d` |

### Multi-output kernels (the "more complicated" part)

C has no MATLAB `[u, v] = f(...)`. Each multi-output kernel returns a
small result struct whose members are named exactly like the equation's
outputs (the equation tokens from the benchmark table):

```c
typedef struct { v3d u; v3d v; } v3uv_d;   /* u = x∧y,  v = x∧w  */
typedef struct { v3d u; v1d r; } v3ur_d;   /* u = x∧y,  r = x·w  */
typedef struct { v1d r; v1d q; } v1rq_d;   /* r = x·y,  q = x·w  */
```

Each member is an independent owned vector (individually freeable with
`v3d_free`/`v1d_free`); the combined `v3uv_d_free(p)` / `v3ur_d_free(p)` /
`v1rq_d_free(p)` frees all members at once.

## Conventions

- `result.n` = first input's `n`; the caller guarantees all inputs share
  `n` (BLAS trust-the-caller convention).
- Results always have stride 1.
- `v1axpy` computes `alpha*x + y` into a **new** vector (copy `y`, then
  call `cblas_*axpy`) rather than overwriting `y`: one extra memory pass,
  in exchange for the uniform "inputs are borrowed" invariant.
- Header is generated from one macro (`V3BLAS_DEFINE`) instantiated four
  times — real scalars are passed by value, complex scalars are passed as
  pointers to a two-component struct, exactly matching the cblas
  prototypes. Everything is `static inline`; nothing else to link.

## Verification

`benchmark/v3blas_test.c` (plain C, ad-hoc compile against the installed
`libopenblas.a`, not part of the OpenBLAS utest suite): runs all 13
operations at all 4 precisions against a long-double reference, plus
strided (inc ≠ 1) cases for `v3cross_d` and `v1xypa_c`.

```
gcc -O2 -I pkgs/openblas/include/openblas benchmark/v3blas_test.c \
    -L pkgs/openblas/lib -lopenblas -lm -o /tmp/v3blas_test && /tmp/v3blas_test
```
