# Physics kernel report (temporary)

Headers below are quoted verbatim from the files in the tree
(`kernel/generic/<stem>.c` via `awk '/^int CNAME/{...}'`, `cblas.h` via `grep`).

Conventions shared by all kernels:
- A "3-vector" is 3 separate arrays `x1,x2,x3`, each with its own stride `incx1..incx3`.
  All are component-wise over `n` elements; none of them is a global reduction.
- Strides are BLAS/Fortran-style: `inc==0` means the first element is repeated,
  `inc<0` means traverse backwards.
- Complex (`c`,`z`) arithmetic is bilinear (no conjugation). `FLOAT` in the kernel
  header is the component type (`float` for `s`/`c`, `double` for `d`/`z`); a complex
  array is 2×FLOAT interleaved, so the complex interface doubles the strides internally.
- `dummy0/dummy1` (front) and `dummy/dummy2` (back) are ABI padding only — unused.
- The `z<stem>.c` complex kernel header is identical to `<stem>.c` except where noted
  (scalar kernels split the scalar into `ar, ai`).

---

## 1. `1xypa` — implemented

**Does:** `r[i] = q[i]*t[i] + a` — element-wise multiply of two 1-vectors plus a scalar.

**Kernel header** (`kernel/generic/1xypa.c`):
```c
int CNAME(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, FLOAT a, FLOAT *q,
          BLASLONG incq, FLOAT *t, BLASLONG inct, FLOAT *r, BLASLONG incr,
          FLOAT *dummy, BLASLONG dummy2)
```
`z1xypa.c` has `FLOAT ar, FLOAT ai` instead of `FLOAT a`.

**cblas header** (`cblas.h`):
```c
void cblas_s1xypa(OPENBLAS_CONST blasint n, OPENBLAS_CONST float a, OPENBLAS_CONST float *q, OPENBLAS_CONST blasint incq, OPENBLAS_CONST float *t, OPENBLAS_CONST blasint inct, float *r, OPENBLAS_CONST blasint incr);
```
`d` = same with `double`; `c`/`z` = same with `void *` (scalar passed as pointer).

| arg | meaning |
|---|---|
| `n` | number of elements |
| `a` | scalar addend (`ar,ai` for c/z) |
| `q`, `incq` | input 1-vector and its stride |
| `t`, `inct` | input 1-vector and its stride |
| `r`, `incr` | output 1-vector and its stride |

---

## 2. `3dot` — implemented

**Does:** `r[i] = x1[i]*y1[i] + x2[i]*y2[i] + x3[i]*y3[i]` — element-wise dot of two 3-vectors (not a global sum).

**Kernel header** (`kernel/generic/3dot.c`, `z3dot.c` identical):
```c
int CNAME(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1,
          FLOAT *x1, BLASLONG incx1, FLOAT *x2, BLASLONG incx2, FLOAT *x3, BLASLONG incx3,
          FLOAT *y1, BLASLONG incy1, FLOAT *y2, BLASLONG incy2, FLOAT *y3, BLASLONG incy3,
          FLOAT *r, BLASLONG incr, FLOAT *dummy, BLASLONG dummy2)
```

**cblas header** (`cblas.h`):
```c
void cblas_s3dot(OPENBLAS_CONST blasint n, OPENBLAS_CONST float *x1, OPENBLAS_CONST blasint incx1, OPENBLAS_CONST float *x2, OPENBLAS_CONST blasint incx2, OPENBLAS_CONST float *x3, OPENBLAS_CONST blasint incx3, OPENBLAS_CONST float *y1, OPENBLAS_CONST blasint incy1, OPENBLAS_CONST float *y2, OPENBLAS_CONST blasint incy2, OPENBLAS_CONST float *y3, OPENBLAS_CONST blasint incy3, float *r, OPENBLAS_CONST blasint incr);
```

| arg | meaning |
|---|---|
| `x1..x3`, `incx1..incx3` | components of input 3-vector x |
| `y1..y3`, `incy1..incy3` | components of input 3-vector y |
| `r`, `incr` | scalar output array (one element per i) |

---

## 3. `3had` — stub (-42 fill)

**Does:** Hadamard (component-wise) product of two 3-vectors: `wj[i] = xj[i]*yj[i]` for j = 1..3.

**Kernel header** (`kernel/generic/3had.c`, `z3had.c` identical):
```c
int CNAME(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1,
          FLOAT *x1, BLASLONG incx1, FLOAT *x2, BLASLONG incx2, FLOAT *x3, BLASLONG incx3,
          FLOAT *y1, BLASLONG incy1, FLOAT *y2, BLASLONG incy2, FLOAT *y3, BLASLONG incy3,
          FLOAT *w1, BLASLONG incw1, FLOAT *w2, BLASLONG incw2, FLOAT *w3, BLASLONG incw3,
          FLOAT *dummy, BLASLONG dummy2)
```

**cblas header** (`cblas.h`):
```c
void cblas_s3had(OPENBLAS_CONST blasint n, OPENBLAS_CONST float *x1, OPENBLAS_CONST blasint incx1, OPENBLAS_CONST float *x2, OPENBLAS_CONST blasint incx2, OPENBLAS_CONST float *x3, OPENBLAS_CONST blasint incx3, OPENBLAS_CONST float *y1, OPENBLAS_CONST blasint incy1, OPENBLAS_CONST float *y2, OPENBLAS_CONST blasint incy2, OPENBLAS_CONST float *y3, OPENBLAS_CONST blasint incy3, float *w1, OPENBLAS_CONST blasint incw1, float *w2, OPENBLAS_CONST blasint incw2, float *w3, OPENBLAS_CONST blasint incw3);
```

| arg | meaning |
|---|---|
| `x1..x3`/`incx*`, `y1..y3`/`incy*` | input 3-vectors |
| `w1..w3`, `incw1..incw3` | output 3-vector (component-wise product) |

---

## 4. `3cross` — stub (-42 fill)

**Does:** Cross product of two 3-vectors:
`w1[i] = x2[i]*y3[i] − x3[i]*y2[i]`,
`w2[i] = x3[i]*y1[i] − x1[i]*y3[i]`,
`w3[i] = x1[i]*y2[i] − x2[i]*y1[i]`.

**Kernel header** (`kernel/generic/3cross.c`, `z3cross.c` identical):
```c
int CNAME(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1,
          FLOAT *x1, BLASLONG incx1, FLOAT *x2, BLASLONG incx2, FLOAT *x3, BLASLONG incx3,
          FLOAT *y1, BLASLONG incy1, FLOAT *y2, BLASLONG incy2, FLOAT *y3, BLASLONG incy3,
          FLOAT *w1, BLASLONG incw1, FLOAT *w2, BLASLONG incw2, FLOAT *w3, BLASLONG incw3,
          FLOAT *dummy, BLASLONG dummy2)
```

**cblas header** (`cblas.h`):
```c
void cblas_s3cross(OPENBLAS_CONST blasint n, OPENBLAS_CONST float *x1, OPENBLAS_CONST blasint incx1, OPENBLAS_CONST float *x2, OPENBLAS_CONST blasint incx2, OPENBLAS_CONST float *x3, OPENBLAS_CONST blasint incx3, OPENBLAS_CONST float *y1, OPENBLAS_CONST blasint incy1, OPENBLAS_CONST float *y2, OPENBLAS_CONST blasint incy2, OPENBLAS_CONST float *y3, OPENBLAS_CONST blasint incy3, float *w1, OPENBLAS_CONST blasint incw1, float *w2, OPENBLAS_CONST blasint incw2, float *w3, OPENBLAS_CONST blasint incw3);
```

| arg | meaning |
|---|---|
| `x1..x3`/`incx*`, `y1..y3`/`incy*` | input 3-vectors |
| `w1..w3`, `incw1..incw3` | output cross product |

---

## 5. `3crossscal` — stub (-42 fill)

**Does:** Scaled cross product: `w[i] = a * (x[i] ∧ y[i])`.

**Kernel header** (`kernel/generic/3crossscal.c`):
```c
int CNAME(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, FLOAT a,
          FLOAT *x1, BLASLONG incx1, FLOAT *x2, BLASLONG incx2, FLOAT *x3, BLASLONG incx3,
          FLOAT *y1, BLASLONG incy1, FLOAT *y2, BLASLONG incy2, FLOAT *y3, BLASLONG incy3,
          FLOAT *w1, BLASLONG incw1, FLOAT *w2, BLASLONG incw2, FLOAT *w3, BLASLONG incw3,
          FLOAT *dummy, BLASLONG dummy2)
```
`z3crossscal.c` has `FLOAT ar, FLOAT ai` instead of `FLOAT a`.

**cblas header** (`cblas.h`):
```c
void cblas_s3crossscal(OPENBLAS_CONST blasint n, OPENBLAS_CONST float a, OPENBLAS_CONST float *x1, OPENBLAS_CONST blasint incx1, OPENBLAS_CONST float *x2, OPENBLAS_CONST blasint incx2, OPENBLAS_CONST float *x3, OPENBLAS_CONST blasint incx3, OPENBLAS_CONST float *y1, OPENBLAS_CONST blasint incy1, OPENBLAS_CONST float *y2, OPENBLAS_CONST blasint incy2, OPENBLAS_CONST float *y3, OPENBLAS_CONST blasint incy3, float *w1, OPENBLAS_CONST blasint incw1, float *w2, OPENBLAS_CONST blasint incw2, float *w3, OPENBLAS_CONST blasint incw3);
```
(`c`/`z`: `void *a`.)

| arg | meaning |
|---|---|
| `a` | scalar scale (`ar,ai` for c/z) |
| `x1..x3`/`incx*`, `y1..y3`/`incy*` | input 3-vectors |
| `w1..w3`, `incw*` | output scaled cross product |

---

## 6. `3sqr` — stub (-42 fill)

**Does:** Squared norm per element: `r[i] = x1[i]*x1[i] + x2[i]*x2[i] + x3[i]*x3[i]`.

**Kernel header** (`kernel/generic/3sqr.c`, `z3sqr.c` identical):
```c
int CNAME(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1,
          FLOAT *x1, BLASLONG incx1, FLOAT *x2, BLASLONG incx2, FLOAT *x3, BLASLONG incx3,
          FLOAT *r, BLASLONG incr, FLOAT *dummy, BLASLONG dummy2)
```

**cblas header** (`cblas.h`):
```c
void cblas_s3sqr(OPENBLAS_CONST blasint n, OPENBLAS_CONST float *x1, OPENBLAS_CONST blasint incx1, OPENBLAS_CONST float *x2, OPENBLAS_CONST blasint incx2, OPENBLAS_CONST float *x3, OPENBLAS_CONST blasint incx3, float *r, OPENBLAS_CONST blasint incr);
```

| arg | meaning |
|---|---|
| `x1..x3`, `incx*` | input 3-vector |
| `r`, `incr` | scalar output array |

---

## 7. `3crossdot` — stub (-42 fill)

**Does:** `r[i] = (x[i] ∧ y[i]) · w[i]` — dot of the cross product x∧y with a third 3-vector w.

**Kernel header** (`kernel/generic/3crossdot.c`, `z3crossdot.c` identical):
```c
int CNAME(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1,
          FLOAT *x1, BLASLONG incx1, FLOAT *x2, BLASLONG incx2, FLOAT *x3, BLASLONG incx3,
          FLOAT *y1, BLASLONG incy1, FLOAT *y2, BLASLONG incy2, FLOAT *y3, BLASLONG incy3,
          FLOAT *w1, BLASLONG incw1, FLOAT *w2, BLASLONG incw2, FLOAT *w3, BLASLONG incw3,
          FLOAT *r, BLASLONG incr, FLOAT *dummy, BLASLONG dummy2)
```

**cblas header** (`cblas.h`):
```c
void cblas_s3crossdot(OPENBLAS_CONST blasint n, OPENBLAS_CONST float *x1, OPENBLAS_CONST blasint incx1, OPENBLAS_CONST float *x2, OPENBLAS_CONST blasint incx2, OPENBLAS_CONST float *x3, OPENBLAS_CONST blasint incx3, OPENBLAS_CONST float *y1, OPENBLAS_CONST blasint incy1, OPENBLAS_CONST float *y2, OPENBLAS_CONST blasint incy2, OPENBLAS_CONST float *y3, OPENBLAS_CONST blasint incy3, OPENBLAS_CONST float *w1, OPENBLAS_CONST blasint incw1, OPENBLAS_CONST float *w2, OPENBLAS_CONST blasint incw2, OPENBLAS_CONST float *w3, OPENBLAS_CONST blasint incw3, float *r, OPENBLAS_CONST blasint incr);
```

| arg | meaning |
|---|---|
| `x1..x3`/`incx*`, `y1..y3`/`incy*` | first two 3-vectors (crossed) |
| `w1..w3`, `incw*` | third 3-vector (dotted with x∧y) |
| `r`, `incr` | scalar output array |

---

## 8. `3crosssqr` — stub (-42 fill)

**Does:** Squared length of the cross product: `r[i] = (x[i] ∧ y[i]) · (x[i] ∧ y[i])`.

**Kernel header** (`kernel/generic/3crosssqr.c`, `z3crosssqr.c` identical):
```c
int CNAME(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1,
          FLOAT *x1, BLASLONG incx1, FLOAT *x2, BLASLONG incx2, FLOAT *x3, BLASLONG incx3,
          FLOAT *y1, BLASLONG incy1, FLOAT *y2, BLASLONG incy2, FLOAT *y3, BLASLONG incy3,
          FLOAT *r, BLASLONG incr, FLOAT *dummy, BLASLONG dummy2)
```

**cblas header** (`cblas.h`):
```c
void cblas_s3crosssqr(OPENBLAS_CONST blasint n, OPENBLAS_CONST float *x1, OPENBLAS_CONST blasint incx1, OPENBLAS_CONST float *x2, OPENBLAS_CONST blasint incx2, OPENBLAS_CONST float *x3, OPENBLAS_CONST blasint incx3, OPENBLAS_CONST float *y1, OPENBLAS_CONST blasint incy1, OPENBLAS_CONST float *y2, OPENBLAS_CONST blasint incy2, OPENBLAS_CONST float *y3, OPENBLAS_CONST blasint incy3, float *r, OPENBLAS_CONST blasint incr);
```

| arg | meaning |
|---|---|
| `x1..x3`/`incx*`, `y1..y3`/`incy*` | input 3-vectors |
| `r`, `incr` | scalar output array |

---

## 9. `3crossxy_crossxz` — stub (-42 fill)

**Does:** Two cross products sharing the first operand:
`u[i] = x[i] ∧ y[i]` and `v[i] = x[i] ∧ w[i]`.

**Kernel header** (`kernel/generic/3crossxy_crossxz.c`, `z3crossxy_crossxz.c` identical):
```c
int CNAME(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1,
          FLOAT *x1, BLASLONG incx1, FLOAT *x2, BLASLONG incx2, FLOAT *x3, BLASLONG incx3,
          FLOAT *y1, BLASLONG incy1, FLOAT *y2, BLASLONG incy2, FLOAT *y3, BLASLONG incy3,
          FLOAT *w1, BLASLONG incw1, FLOAT *w2, BLASLONG incw2, FLOAT *w3, BLASLONG incw3,
          FLOAT *u1, BLASLONG incu1, FLOAT *u2, BLASLONG incu2, FLOAT *u3, BLASLONG incu3,
          FLOAT *v1, BLASLONG incv1, FLOAT *v2, BLASLONG incv2, FLOAT *v3, BLASLONG incv3,
          FLOAT *dummy, BLASLONG dummy2)
```

**cblas header** (`cblas.h`):
```c
void cblas_s3crossxy_crossxz(OPENBLAS_CONST blasint n, OPENBLAS_CONST float *x1, OPENBLAS_CONST blasint incx1, OPENBLAS_CONST float *x2, OPENBLAS_CONST blasint incx2, OPENBLAS_CONST float *x3, OPENBLAS_CONST blasint incx3, OPENBLAS_CONST float *y1, OPENBLAS_CONST blasint incy1, OPENBLAS_CONST float *y2, OPENBLAS_CONST blasint incy2, OPENBLAS_CONST float *y3, OPENBLAS_CONST blasint incy3, OPENBLAS_CONST float *w1, OPENBLAS_CONST blasint incw1, OPENBLAS_CONST float *w2, OPENBLAS_CONST blasint incw2, OPENBLAS_CONST float *w3, OPENBLAS_CONST blasint incw3, float *u1, OPENBLAS_CONST blasint incu1, float *u2, OPENBLAS_CONST blasint incu2, float *u3, OPENBLAS_CONST blasint incu3, float *v1, OPENBLAS_CONST blasint incv1, float *v2, OPENBLAS_CONST blasint incv2, float *v3, OPENBLAS_CONST blasint incv3);
```

| arg | meaning |
|---|---|
| `x1..x3`/`incx*` | shared first operand |
| `y1..y3`/`incy*` | second operand of first cross |
| `w1..w3`/`incw*` | second operand of second cross |
| `u1..u3`/`incu*` | output x∧y |
| `v1..v3`/`incv*` | output x∧w |

---

## 10. `3crossxy_dotxz` — stub (-42 fill)

**Does:** Cross + dot sharing the first operand:
`u[i] = x[i] ∧ y[i]` and `r[i] = x[i] · w[i]`.

**Kernel header** (`kernel/generic/3crossxy_dotxz.c`, `z3crossxy_dotxz.c` identical):
```c
int CNAME(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1,
          FLOAT *x1, BLASLONG incx1, FLOAT *x2, BLASLONG incx2, FLOAT *x3, BLASLONG incx3,
          FLOAT *y1, BLASLONG incy1, FLOAT *y2, BLASLONG incy2, FLOAT *y3, BLASLONG incy3,
          FLOAT *w1, BLASLONG incw1, FLOAT *w2, BLASLONG incw2, FLOAT *w3, BLASLONG incw3,
          FLOAT *u1, BLASLONG incu1, FLOAT *u2, BLASLONG incu2, FLOAT *u3, BLASLONG incu3,
          FLOAT *r, BLASLONG incr, FLOAT *dummy, BLASLONG dummy2)
```

**cblas header** (`cblas.h`):
```c
void cblas_s3crossxy_dotxz(OPENBLAS_CONST blasint n, OPENBLAS_CONST float *x1, OPENBLAS_CONST blasint incx1, OPENBLAS_CONST float *x2, OPENBLAS_CONST blasint incx2, OPENBLAS_CONST float *x3, OPENBLAS_CONST blasint incx3, OPENBLAS_CONST float *y1, OPENBLAS_CONST blasint incy1, OPENBLAS_CONST float *y2, OPENBLAS_CONST blasint incy2, OPENBLAS_CONST float *y3, OPENBLAS_CONST blasint incy3, OPENBLAS_CONST float *w1, OPENBLAS_CONST blasint incw1, OPENBLAS_CONST float *w2, OPENBLAS_CONST blasint incw2, OPENBLAS_CONST float *w3, OPENBLAS_CONST blasint incw3, float *u1, OPENBLAS_CONST blasint incu1, float *u2, OPENBLAS_CONST blasint incu2, float *u3, OPENBLAS_CONST blasint incu3, float *r, OPENBLAS_CONST blasint incr);
```

| arg | meaning |
|---|---|
| `x1..x3`/`incx*` | shared first operand |
| `y1..y3`/`incy*` | cross partner of x |
| `w1..w3`/`incw*` | dot partner of x |
| `u1..u3`/`incu*` | output x∧y |
| `r`, `incr` | scalar output x·w |

---

## 11. `3dotxy_dotxz` — stub (-42 fill)

**Does:** Two element-wise dots sharing the first operand:
`r[i] = x[i] · y[i]` and `q[i] = x[i] · w[i]`.

**Kernel header** (`kernel/generic/3dotxy_dotxz.c`, `z3dotxy_dotxz.c` identical):
```c
int CNAME(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1,
          FLOAT *x1, BLASLONG incx1, FLOAT *x2, BLASLONG incx2, FLOAT *x3, BLASLONG incx3,
          FLOAT *y1, BLASLONG incy1, FLOAT *y2, BLASLONG incy2, FLOAT *y3, BLASLONG incy3,
          FLOAT *w1, BLASLONG incw1, FLOAT *w2, BLASLONG incw2, FLOAT *w3, BLASLONG incw3,
          FLOAT *r, BLASLONG incr, FLOAT *q, BLASLONG incq, FLOAT *dummy, BLASLONG dummy2)
```

**cblas header** (`cblas.h`, canonical copy at line 481):
```c
void cblas_s3dotxy_dotxz(OPENBLAS_CONST blasint n, OPENBLAS_CONST float *x1, OPENBLAS_CONST blasint incx1, OPENBLAS_CONST float *x2, OPENBLAS_CONST blasint incx2, OPENBLAS_CONST float *x3, OPENBLAS_CONST blasint incx3, OPENBLAS_CONST float *y1, OPENBLAS_CONST blasint incy1, OPENBLAS_CONST float *y2, OPENBLAS_CONST blasint incy2, OPENBLAS_CONST float *y3, OPENBLAS_CONST blasint incy3, OPENBLAS_CONST float *w1, OPENBLAS_CONST blasint incw1, OPENBLAS_CONST float *w2, OPENBLAS_CONST blasint incw2, OPENBLAS_CONST float *w3, OPENBLAS_CONST blasint incw3, float *r, OPENBLAS_CONST blasint incr, float *q, OPENBLAS_CONST blasint incq);
```

| arg | meaning |
|---|---|
| `x1..x3`/`incx*` | shared first operand |
| `y1..y3`/`incy*` | partner of first dot |
| `w1..w3`/`incw*` | partner of second dot |
| `r`, `incr` | scalar output x·y |
| `q`, `incq` | scalar output x·w |

---

## 12. `1norm` — stub (-42 fill)

**Does:** `r[i] = sqrt(q[i] * q[i])` — element-wise square root of the square of a 1-vector.
Real: equals `|q[i]|`. Complex: principal square root of the bilinear square (i.e.
`q[i]` when `Re(q[i]) >= 0`, `−q[i]` otherwise).

**Kernel header** (`kernel/generic/1norm.c`, `z1norm.c` identical):
```c
int CNAME(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1,
          FLOAT *q, BLASLONG incq, FLOAT *r, BLASLONG incr, FLOAT *dummy, BLASLONG dummy2)
```

**cblas header** (`cblas.h`):
```c
void cblas_s1norm(OPENBLAS_CONST blasint n, OPENBLAS_CONST float *q, OPENBLAS_CONST blasint incq, float *r, OPENBLAS_CONST blasint incr);
```

| arg | meaning |
|---|---|
| `q`, `incq` | input 1-vector |
| `r`, `incr` | output 1-vector |

---

## Known issues found while pulling these headers

1. **`cblas.h` duplicate block** — lines 489–492 repeat all four `cblas_*3dotxy_dotxz`
   prototypes with different spacing (`float * x1`), right after the canonical 1norm block.
   Legally a redeclaration, but it is a wiring leftover and should be deleted.
2. **`common_level1.h` truncation** — a wiring script once cut the file after
   `int scopy_k(...)`, losing `dcopy_k … zaxpby_k` and the closing
   `#ifdef __CUDACC__ } #endif / #endif`. Restored in the working tree (verified:
   `#if` balance 0, diff is now purely additive, 120 insertions / 0 deletions).
3. **Stub state** — kernels 3–12 currently fill their outputs with `-42`; all
   signatures, interfaces, build wiring and tests are in final form.
4. **Ext test files** — 40 generated files exist in `utest/test_extensions/`; a
   compile bug in the generated `n_zero` CTEST calls (missing stride args) is known
   and pending fix; the 10 unit-test files in `utest/` are unaffected.
