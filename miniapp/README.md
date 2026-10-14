# `miniapp/` — the API, used for real

Mini-apps are working physics codes built **only** on the public API
(`cblas_*`, `v3blas_*` via `v3blas.h`) plus standard BLAS. They are the
API's harshest reviewer: stride semantics, in-place aliasing and
broadcast idioms get exercised the way physics code actually uses them.

| app | physics | kernels exercised beyond std BLAS |
|---|---|---|
| `sod2d.c` | 2D compressible Euler, Sod tube | `1xypa 3had 3mom_ke 3sqr 3refl lap5` |
| `tops3d.c` | N rigid tops (attitude dynamics) | `1xypa 3had 3cross 3crossscal 3dot` + **L2**: `dsbmv dgbmv` |

## Build & run

Built by the same script as the benchmarks — see
[`../docs/backends/OPENBLAS.install.sh`](../docs/backends/OPENBLAS.install.sh)
— or standalone:

```sh
cmake -S miniapp -B miniapp/build
cmake --build miniapp/build
./miniapp/build/sod2d                 # 96x48, ~9 s on a Pi 1
./miniapp/build/sod2d -nx 384 -ny 8 -av 0 -out 500
./miniapp/build/tops3d                # 1024 tops, t=0.5, ~12 s on a Pi 1
./miniapp/build/tops3d -nT 64 -tend 2 -dt 1e-3
```

Writes `sod2d.vtk` (ASCII, one quad per cell) — open in VisIt, ParaView
or matplotlib+meshio. Flags: `-nx -ny -tend -cfl -av -out`; the grid is
2 cells thick in y at minimum so the y-sweep and wall reflection always
execute.

## Validation baked into the defaults

At `t = 0.2` the exact Sod solution is
`u* = 0.9275, p* = 0.3031, rho* = 0.4263, post-shock rho = 0.2656`,
contact at `x = 0.686`, shock at `x = 0.852`. At `-nx 1152 -ny 4 -av 0`
the reported profiles match those numbers to four decimals; mass is
conserved to ~1e-7 relative. First-order Rusanov fluxes + dimensional
splitting are the deliberate choice: monotone, no limiters, every op
element-wise.

## The stride lesson this app found (true story)

`inc = 0` means *"repeat element 0 of this component's array"* — scalar
broadcast, the BLAS convention. Multiplying a 3-vector field by a
*scalar field* (`v = m .* (1/d)`) is **not** a broadcast: pass the same
array in all three component slots with **stride 1**:

```c
v3d inv3 = v3d_wrap(invd, invd, invd, n);   /* stride 1: v[i] uses 1/d[i] */
v3had(m, inv3, v);
```

Setting `inv3.inc = 0` freezes the factor at `invd[0]` — silently
wrong, physically plausible-looking, caught only by checking invariants
against the analytic solution. A true broadcast (wall-normal `(0,1,0)`,
constant `eps`) is where stride 0 belongs:

```c
v3d nrm = v3d_wrap(&zero, &one, &zero, rows);
nrm.inc = 0;
v3refl(m_ghost, nrm, nrm, eps0, m_reflected);   /* specular wall */
```

## `tops3d` — a batch of rigid bodies, where BLAS-2 joins in

N independent tops integrated in the **body frame** (Euler's equation,
where the inertia tensor is constant and the ODE is autonomous), with
attitude carried as unit quaternions. Body 0 is fixed check data, body 1
is a heavy fast top that precesses, the rest are random free asymmetric
tops with genuinely non-diagonal inertia (ω and L point in different
directions — the interesting case).

The Level-2 meeting point, honestly: a batch of symmetric 3×3 tensors
stored block-diagonally **is** a symmetric banded matrix with `KD = 2`,
`ld = 3`, so one `dsbmv` call computes `omega[b] = Iinv[b] L[b]` for
*every* body, streaming; likewise a batch of general 3×3 rotation matrices
is a `dgbmv` (`KL = KU = 2`), used to pull the world gravity force into
body axes. The banded batch wants per-body-contiguous (AoS) vectors while
the element-wise physics wants SoA fields; the bridge is a 3×nT
mini-transpose done with three strided `dcopy`s (`inc = 3`) — the most
common plumbing operation in real BLAS codes, made explicit here.

Validation contract (printed by the app; compare with
`./miniapp/build/reftop`):

| check | expected | observed (defaults) |
|---|---|---|
| free-top energy `½ ω·L` | exact invariant | drift `-5e-13` |
| free-top `\|L\|²` | exact invariant | drift `-2e-13` |
| body 0 final L vs independent hand-written reference | agreement | to `~1e-14` |
| heavy-top precession vs `mgl/(I₃ω₃)` | fast-top order | `0.544` vs `0.514` (≈6%) |

The reference (`ref/reftop.c`, plain loops, own 3×3 inverse, zero shared
code, does not even link BLAS) reproduces body 0 to printed precision —
the strided pack, the `dsbmv` band fill and the `v3cross` Euler term are
jointly pinned.

The VTK output draws body 0's **polhode** (ω in body axes — Euler's
exact elliptic solution, visibly a closed curve on the inertia
ellipsoid) and body 1's world axis sweeping its precession cone.

## Lessons `tops3d` found (true stories, all app bugs — library held)

1. **Zero quaternion = NaN faucet.** Bodies initialized with `q = 0`
   (calloc) divide by `|q|` at the first renormalization: `1/0 = inf`,
   `0·inf = NaN`, every field poisoned within one step. Invariants
   caught it; the energy print localized it.
2. **`dgbmv` slot is `ku + i − j`, not `kl + ku + i − j`.** One off, and
   the fill silently writes past the last band column — heap scribble
   that surfaced as NaN far from the cause. The symmetric `dsbmv` upper
   slot `kd + i − j` is fine as Netlib documents.
3. **Measure precession on L̂, not on the figure axis.** The initial
   conditions maximally excite free nutation: the axis whirls about L at
   `~L/I₁` (≈18 rad/s here) and its azimuth is mush; but
   `|dL̂/dt| = |τ|/L` is slow by torque budget, so the world-frame L̂
   azimuth is the clean precession signal. The physics was right the
   first time — the *probe* was wrong.

## Verification discipline

`sod2d`: any change to the flux path must still reproduce the exact
Riemann invariants (`u + 2c/(gamma-1)` constant across the fan) and the
exact star state. `tops3d`: any change must keep the two exact free-top
invariants at the 1e-12 level and `L_final` of body 0 equal to
`./miniapp/build/reftop` output. Independent references are part of the
apps, not afterthoughts.

## Planned

1. **MHD version** (`sod2d_mhd`): adds `3cross`/`3crosscross` for
   J = curl B and the induction term, `3exb` for drift diagnostics.
2. **TVD upgrade**: needs a componentwise `minmod` — a candidate new
   kernel (`3minmod`) admitted only because a real consumer wants it.
3. **Lagrangian particle pusher** (`push3d`): `3crossscal` (Boris
   rotation), `3norm_unit`, `1norm` — exercises the `-t`-style output
   scalar contract.

## Considered and parked — permanent side notes, not roadmap

Kept on record because they are real consumers, with one line on what
kernel mix each would lean on. None is scheduled.

- **FDTD Maxwell on a Yee lattice** — the purest showcase for the fused
  fork kernel `3crossxy_crossxz` (both curl components in one sweep),
  staggered E/B fields are exactly the SoA 3-vector grammar; L2 would
  only appear in token roles (lumped-element ports), which is why it is
  the weakest fit for this repo's L2 story.
- **Linear blend skinning** (game character deformation) — per-vertex
  blended bone transforms are a `gemm` of the (nVerts × nBones) weight
  matrix against per-bone 4×4 pose deltas, then per-vertex 3-vector
  normalize/orthogonalize; its validation is "looks right", which fails
  this folder's contract.
- **Articulated robot arm** (Featherstone ABA) — the maximal-L2 cousin
  of `tops3d` (`dsymv`/`dger`/`dtrsv` on the joint-space inertia all
  arise naturally), but the recursive spatial-vector code is twice the
  size of `tops3d` and has no clean analytic gate to validate against;
  `tops3d` is its free-body simplification with the same API story.
