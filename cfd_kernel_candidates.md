# 3-vector kernel candidates for CFD workloads

Menu draft in the `blasKernels.md` idiom: element-wise over fields of
3-vectors (three component arrays, per-component strides), pure C,
generic kernel + `cblas_<p><stem>` ABI, fused multi-output where it
saves a memory pass. `∧` = cross (right-hand rule), `·` = dot.
Scalar parameters follow the `1xypa` precedent (real argument beside
the arrays). `e` is a small guard supplied by the caller.

## Proposed ops

| stem | equation | why CFD/plasma codes want it |
|---|---|---|
| `3unitnorm` | `r = sqrt(x·x + e)`, `e = x/r` (2 outputs) | unit vectors everywhere: cell-face normals n̂, magnetic direction b̂ = B/|B|, velocity direction in Riemann data; fusing norm+division halves traffic vs two passes |
| `3triple` | `s = x·(y∧z)` | mapping Jacobian J = ∂x/∂ξ · (∂y/∂ξ ∧ ∂z/∂ξ) in every curvilinear/ALE metric computation; signed tet/hex volumes; discrete de Rham / mimetic Hodge star ingredients |
| `3crossabxc` | `v = (x∧y)∧z` | Lorentz/MHD force (u∧B)∧B, magnetic pressure/tension split, rotating-frame inertial terms; Lagrange identity `(x·z)y − (y·z)x` gives a cheaper fused path (2 dots + scaled adds) no naive code takes |
| `3refl` | `q = x − a·(x·y)·y/(y·y + e)` | one kernel, three uses: a=2 → reflection off walls (slip walls, IBM, DSMC/particle BCs); a=1 → tangential component ⟂ b̂ (Braginskii perpendicular transport, mirror/actuator-disk models; parallel part then `p = x − q` via axpy) |
| `3exb` | `r = a·(x∧y)/(y·y + e)` | E∧B/B² drift velocity — the single most-used term in edge-plasma turbulence (BOUT++, TOKAM3K, GENE-adjacent); sign/coeff flips give grad-B and curvature drifts |
| `3drag` | `r = a·x·sqrt(x·x + e)` | quadratic drag |u|u: wind loading, Forchheimer porous drag, shallow-water bottom friction, outflow damping; fuses sqrt-norm+scale, saves one full read of x |
| `3momke` | `m = s·x`, `k = ½·s·(x·x)` (2 outputs) | primitive→conservative map per cell (momentum + kinetic energy from density s and velocity x); the k output feeds pressure p = (γ−1)(E−k) and Mach/sound-speed diagnostics |

Close relatives, same ABI shape but acting component-wise on three
generic scalar fields (still our layout, not "3-vectors" physically):

- `3minmod`: `r = minmod(x, y, z)` — TVD slope limiters (left/right/centered candidates).
- `3sel`: `r = (s > 0) ? x : y` — branchless upwind selection in flux assembly.

## Deliberately NOT in this grammar (next tiers)

- **curl / div / grad**: stencils — need neighbor offsets, not strides; a
  different ABI (a "Level-0.5 stencil" spec, later).
- **Riemann solvers, flux Jacobians, characteristic decomposition**:
  per-face small matrix/vector work — matvec/eigen, not element-wise.
- **Dyadics** (`B⊗B`, double contractions `B:(vv)`): per-element 3×3
  matmul — batched-GEMM territory, tensor-core conversation.
- **Global reductions / norms over the mesh**: BLAS Level-1 proper and
  collectives territory — out by design (no reductions, per spec).

## Stride tricks: these need no new kernels

- Constant vectors (gravity g, rotation axis ω, fixed B̂₀): pass the
  constant array with `inc = 0` — e.g. Coriolis `ω∧u` is plain `3cross`
  with ω's three scalars at stride 0.
- Component broadcast/selection: stride 0 and negative strides already
  cover mirroring symmetries.

## Design notes

- Division kernels (`3refl`, `3exb`) carry the guard `e` as an explicit
  argument rather than a hardcoded `+1e-30`: callers choose between
  exactness (e=0) and robustness, and the conformance vectors can pin
  both behaviors.
- `3crossabxc` should be written to pick whichever of the double-cross
  / Lagrange-identity forms the compiler can't beat, but results must
  fix one — pick the double-cross evaluation order for the spec and
  keep tolerance tiers for backends that take the identity shortcut.
- New ops inherit everything: 4 precisions, `v3blas` front-ends, utest
  template, cblas ABI, conformance vectors. Cost per op ≈ one generic
  kernel + one interface file + tests, same pipeline as the first 13.
