# Kernel catalog

The single source of truth for **what the kernels are**: equations, argument
semantics, prototypes, edge-case contracts. How they are wired into a
specific BLAS lives in [`backends/`](backends/); how to implement one lives
in [`AGENT.md`](AGENT.md).

## Notation

- `<p>` = precision: `s`, `d`, `c`, `z`. `T` = real element type: `float`
  for `s`/`c`, `double` for `d`/`z`. One `<p>` symbol is four real symbols.
- A **3-vector** is three plain arrays (the components), each with its own
  stride. In prototypes, a triplet argument `x` below expands to
  `x1, x2, x3` with increments `incx1, incx2, incx3` — component `k` of a
  3-vector is the array `x(k+1)`. No grid, no layout assumption beyond the
  strides: SoA falls out naturally from how the caller allocates.
- Over `c`/`z` one element is two adjacent `T` (interleaved re, im — the
  cblas layout); the component stride there counts reals, so complex
  interfaces double the user's element strides internally.
- **All products over `c`/`z` are bilinear — no conjugation anywhere.**
  `x·y = (xr·yr − xi·yi) + i(xr·yi + xi·yr)`.
- `∧` = cross product, `·` = dot, `.*` = Hadamard (per-component) product,
  juxtaposition/`*` = scalar times array. Equations are element-wise at
  index `i`; nothing here is a global reduction over the field.

## Quick list

20 operations, 4 precisions = 80 kernel symbols. "Front-end" = the op also
exists in the `v3blas` C/C++/Fortran layer ([README](../README.md)).

| symbol | equation | family | front-end | threading¹ |
|---|---|---|---|---|
| `<p>1xypa` | `r = q.*t + a` | scalar-array | ✓ | serial² |
| `<p>1norm` | `r = √(q·q)` | scalar-array | ✓ | serial² |
| `<p>lap5` | `lap = a·((u(±jx) + u(±jy)) − 4u)` | stencil | — | threaded |
| `<p>lap7` | `lap = a·((u(±jx) + u(±jy) + u(±jz)) − 6u)` | stencil | — | threaded |
| `<p>3had` | `w = x.*y` | 3-vector | ✓ | threaded |
| `<p>3cross` | `w = x∧y` | 3-vector | ✓ | threaded |
| `<p>3dot` | `r = x·y` | 3-vector | ✓ | serial² |
| `<p>3sqr` | `r = x·x` | 3-vector | ✓ | serial² |
| `<p>3crossscal` | `w = a·(x∧y)` | 3-vector | ✓ | threaded |
| `<p>3crossdot` | `r = (x∧y)·w` | 3-vector | ✓ | serial² |
| `<p>3crosssqr` | `r = (x∧y)·(x∧y)` | 3-vector | ✓ | serial² |
| `<p>3crossxy_crossxz` | `u = x∧y`, `v = x∧w` | fork | ✓ | threaded |
| `<p>3crossxy_dotxz` | `u = x∧y`, `r = x·w` | fork | ✓ | serial² |
| `<p>3dotxy_dotxz` | `r = x·y`, `q = x·w` | fork | ✓ | serial² |
| `<p>3crosscross` | `r = (x∧y)∧z` | 3-vector (CFD) | ✓ | threaded |
| `<p>3norm_unit` | `r = √(x·x+eps)`, `u = x/r` | 3-vector (CFD) | ✓ | threaded |
| `<p>3refl` | `q = x − a·(x·y)·y/(y·y+eps)` | 3-vector (CFD) | ✓ | threaded |
| `<p>3exb` | `r = a·(x∧y)/(y·y+eps)` | 3-vector (CFD) | ✓ | threaded |
| `<p>3drag` | `r = a·x·√(x·x+eps)` | 3-vector (CFD) | ✓ | threaded |
| `<p>3mom_ke` | `m = s·x`, `k = ½·s·(x·x)` | 3-vector (CFD) | ✓ | threaded |

¹ Threading = the backend may slice `n` across threads transparently; see
your backend's doc for the mechanism. Semantically irrelevant to the caller.
² Ops whose outputs are per-element dot products keep a **pinned summation
order** (see Contracts) and run serially to preserve it bitwise.

Not kernels but worth knowing: `axpby` (upstream in modern OpenBLAS) covers
`r = a·x + b·y`; the front-end's `v1axpy` maps to `axpy`. Still-candidate
ops: `3minmod` (TVD limiter), `3sel` (branchless upwind selection), and the
indirection/gather-scatter ABI for unstructured meshes
([`specs/grand_plan.md`](specs/grand_plan.md)).

## Shared conventions

**Argument order reads the name.** `1xypa(n, q, t, a, r, …)` is literally
*x y plus a*, output `r` last. The CFD family obeys it
(`3drag(n, x…, a, eps, r…)`). Two deliberate exceptions: `3crossscal(n, a,
x…, y…, w…)` keeps the BLAS `scal` tradition of a leading scalar, and the
stencils lead with `alpha` (`lap5(n, alpha, u, …)`).

**Scalars.** `a`, `eps`, `alpha`, and the stencil offsets `jx jy jz` are the
*real* type in all four precisions (`float` for `s`/`c`, `double` for
`d`/`z`) — they are physical damping/geometry factors, never complex. The
one complex scalar in the set is `a` of `3crossscal` (a complex circulation
factor is meaningful), passed by address like every cblas complex scalar.
Complex vector/scalar arguments are passed by address (`void *` in cblas);
real scalars by value.

**Strides (`inc`).** BLAS/Fortran semantics, pinned by contract tests:

| `inc` | meaning |
|---|---|
| `> 0` | element `i` is at `base + i·inc` |
| `= 0` | broadcast: element 0 repeated — how you pass constant fields (gravity, `B̂₀`, rotation axis) for free |
| `< 0` | walk backwards: the span is `base … base + (n−1)·|inc|`, traversed from its far end. Pass the *forward start*; the interface shifts the base. |
| `n ≤ 0` | no-op; nothing is read or written |

**Aliasing.** An output component may alias the input component at the same
slot (`w2 == x2` is an in-place op). Never alias *across* components (they
may be computed sequentially — cross-component aliasing compounds
nondeterministically) and never rely on partial overlap: different strides
sharing elements is undefined. `lap` may alias `u` provided the backend
computes the same-element order; in practice keep them disjoint.

**NaN/Inf.** IEEE, silent: a negative `eps` making a sqrt argument negative
yields NaN, no trap, no error return — BLAS convention.

**Reductions are per-element, summation order pinned.** `3dot`-family
outputs are `((x1·y1 + x2·y2) + x3·y3)` evaluated in that order (complex:
each `·` is the bilinear pair product, the sum pairs components first). The
stencil sums `u(−j)+u(+j)` pairwise, then the pairs, the whole sum last.
This is what makes same-library results bitwise reproducible and
cross-library comparison tolerance-based.

**`eps` guards.** Guarded divisions/sqrts take `eps` as an explicit
argument; `eps = 0` means the exact (unguarded) formula.

**Complex `1norm` is not |q|.** `√(q·q)` is the *principal square root of
the bilinear square*: `q` when `Re q ≥ 0`, `−q` otherwise. Genuinely
complex-valued outputs exist (`3sqr`, `1sqr`-shaped ops) — that is why √ is
the only place a branch convention is needed.

## Templates

The raw ABI in full (the `s` variant; `d` swaps the type, `c`/`z` swap
vectors and complex scalars to `void *` and double the strides inside):

```c
void cblas_s3cross(OPENBLAS_CONST blasint n,
    OPENBLAS_CONST float * x1, OPENBLAS_CONST blasint incx1,
    OPENBLAS_CONST float * x2, OPENBLAS_CONST blasint incx2,
    OPENBLAS_CONST float * x3, OPENBLAS_CONST blasint incx3,
    OPENBLAS_CONST float * y1, OPENBLAS_CONST blasint incy1,
    OPENBLAS_CONST float * y2, OPENBLAS_CONST blasint incy2,
    OPENBLAS_CONST float * y3, OPENBLAS_CONST blasint incy3,
                 float * w1, OPENBLAS_CONST blasint incw1,
                 float * w2, OPENBLAS_CONST blasint incw2,
                 float * w3, OPENBLAS_CONST blasint incw3);
```

Below, each op is given in compact form: triplets collapsed to their
letter, increments implicit (`(x)` = the three pointer/inc pairs). The
Fortran 77 entry mirrors the cblas argument list with everything passed by
address and the trailing scalar-precision suffix (`call d3cross(n, x1, 1,
x2, 1, ...)` is legal, raw, and always available).

### Scalar-array (Layer 0)

| op | compact cblas form | detail |
|---|---|---|
| `1xypa` | `cblas_<p>1xypa(n, q, t, a, r)` | `r = q.*t + a`. `a` by value (complex: by address). The Williamson-RK step (`q += h·y; y = f(q) + β·y` is two of these). |
| `1norm` | `cblas_<p>1norm(n, q, r)` | `r = √(q·q)`; real: `|q|`, complex: principal √ of the bilinear square. |

### Stencils

| op | compact cblas form | detail |
|---|---|---|
| `lap5` | `cblas_<p>lap5(n, alpha, u, jx, jy, lap)` | 5-point 2-D Laplacian on a strided row: `jy = ld·incy` maps a grid row in one call. `jx jy` signed element offsets, real type. |
| `lap7` | `cblas_<p>lap7(n, alpha, u, jx, jy, jz, lap)` | 7-point 3-D analogue. |

`alpha` folds `1/h²`. Bounds are the caller's business: the kernel reads
`u(i±j)` unconditionally, so the walked range needs real elements on both
sides (periodic/folded rows are caller-side choices). No front-end:
stride-stencils are a separate ABI track.

### 3-vectors (Layer 1)

| op | compact cblas form | detail |
|---|---|---|
| `3had` | `cblas_<p>3had(n, x, y, w)` | `w = x.*y`, per component. |
| `3cross` | `cblas_<p>3cross(n, x, y, w)` | `w = x∧y`: `w1 = x2y3−x3y2`, `w2 = x3y1−x1y3`, `w3 = x1y2−x2y1`. |
| `3dot` | `cblas_<p>3dot(n, x, y, r)` | `r = x1y1+x2y2+x3y3`, one output scalar per element. |
| `3sqr` | `cblas_<p>3sqr(n, x, r)` | `r = x·x`. |
| `3crossscal` | `cblas_<p>3crossscal(n, a, x, y, w)` | `w = a·(x∧y)`; the one complex scalar (`a` by address in `c`/`z`), leading-scalar exception to the name-order rule. |
| `3crossdot` | `cblas_<p>3crossdot(n, x, y, w, r)` | `r = (x∧y)·w`; also spells `x·(y∧z)`. |
| `3crosssqr` | `cblas_<p>3crosssqr(n, x, y, r)` | `r = (x∧y)·(x∧y)` — a `3dot` on a cross with itself, fused. |

### Forks — two outputs, one pass

All forks share the first operand `x`; passing the same handle twice is
legal and free. Forks cap at two outputs (register pressure, not taste).

| op | compact cblas form | detail |
|---|---|---|
| `3crossxy_crossxz` | `cblas_<p>3crossxy_crossxz(n, x, y, w, u, v)` | `u = x∧y`, `v = x∧w`. Mind the arg slots: `w` is an **input**. |
| `3crossxy_dotxz` | `cblas_<p>3crossxy_dotxz(n, x, y, w, u, r)` | `u = x∧y`, `r = x·w`. |
| `3dotxy_dotxz` | `cblas_<p>3dotxy_dotxz(n, x, y, w, r, q)` | `r = x·y`, `q = x·w` — e.g. Ohmic heating and helicity in one pass. |

### CFD menu (composite 3-vector ops)

Guarded ops take real `eps` (0 = exact); `a` is real here too.

| op | compact cblas form | detail |
|---|---|---|
| `3crosscross` | `cblas_<p>3crosscross(n, x, y, z, r)` | `r = (x∧y)∧z` — the `(u∧B)∧B` Lorentz/MHD relaxation shape. |
| `3norm_unit` | `cblas_<p>3norm_unit(n, x, eps, r, u)` | `r = √(x·x+eps)` and `u = x/r` — norm *and* unit vector in one pass (`u(:,c)` of an `(n,3)` field). |
| `3refl` | `cblas_<p>3refl(n, x, y, a, eps, q)` | `q = x − a·(x·y)·y/(y·y+eps)` — wall/boundary reflection (`a=1` reflect, `a=2` specular mirror). |
| `3exb` | `cblas_<p>3exb(n, x, y, a, eps, r)` | `r = a·(x∧y)/(y·y+eps)` — E∧B drift, `(∧B)/B²` operators. |
| `3drag` | `cblas_<p>3drag(n, x, a, eps, r)` | `r = a·x·√(x·x+eps)` — quadratic drag with optional speed floor. |
| `3mom_ke` | `cblas_<p>3mom_ke(n, s, x, m, k)` | `m = s·x`, `k = ½·s·(x·x)` — primitive→conservative map; `s` is a *field* (density), not a scalar. |

## Front-end mapping

Every front-end op is a zero-cost one-liner onto the table above; the C11
signature shape is `void v<name>_d(<inputs..., scalars>, <outputs...>)`
with views instead of triplets — see [README](../README.md) for the API
model and the full op-call table.
