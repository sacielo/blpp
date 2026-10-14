# `miniapp/` — the API, used for real

Mini-apps are working physics codes built **only** on the public API
(`cblas_*`, `v3blas_*` via `v3blas.h`) plus standard BLAS. They are the
API's harshest reviewer: stride semantics, in-place aliasing and
broadcast idioms get exercised the way physics code actually uses them.

| app | physics | kernels exercised beyond std BLAS |
|---|---|---|
| `sod2d.c` | 2D compressible Euler, Sod tube | `1xypa 3had 3mom_ke 3sqr 3refl lap5` |

## Build & run

Built by the same script as the benchmarks — see
[`../docs/backends/OPENBLAS.install.sh`](../docs/backends/OPENBLAS.install.sh)
— or standalone:

```sh
cmake -S miniapp -B miniapp/build
cmake --build miniapp/build
./miniapp/build/sod2d                 # 96x48, ~9 s on a Pi 1
./miniapp/build/sod2d -nx 384 -ny 8 -av 0 -out 500
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

Setting those strides to 0 freezes the factor at `invd[0]` — silently
wrong, physically plausible-looking, caught only by checking invariants
against the analytic solution. A true broadcast (wall-normal `(0,1,0)`,
constant `eps`) is where stride 0 belongs:

```c
v3d nrm = v3d_wrap(&zero, &one, &zero, rows);
nrm.incx = nrm.incy = nrm.incz = 0;
v3refl(m_ghost, nrm, nrm, eps0, m_reflected);   /* specular wall */
```

## Verification discipline

`ref1d`-style checks are part of the app: any change to the flux path
must still reproduce the exact Riemann invariants
(`u + 2c/(gamma-1)` constant across the fan) and the exact star state.

## Planned

1. **MHD version** (`sod2d_mhd`): adds `3cross`/`3crosscross` for
   J = curl B and the induction term, `3exb` for drift diagnostics.
2. **TVD upgrade**: needs a componentwise `minmod` — a candidate new
   kernel (`3minmod`) admitted only because a real consumer wants it.
3. **Lagrangian particle pusher** (`push3d`): `3crossscal` (Boris
   rotation), `3norm_unit`, `1norm` — exercises the `-t`-style output
   scalar contract.
