# Grand plan: from physics kernels to a CFD stack

Where the element-wise kernels shipped in `physics_kernels.patch` sit in a
PDE-solver architecture, what is still missing, and in which order to
build it. Working title for the strategy: *LAPACK did for BLAS what we
want element-wise physics to do for CFD* — a versioned ABI + semantics
spec plus a conformance suite wins, performance alone never did.

## The layering

```
solver      Navier-Stokes, RK, Newton, preconditioners   <- do NOT standardize; write ONE exemplar
geometry    volumes, normals, metrics, Jacobians         <- elementwise; already our kernel shapes
stencil     gather -> flux/Riemann -> scatter, lap5/lap7 <- next ABI ask (indirection)
physics     3dot 3cross 3drag 3mom_ke 1xypa ...          <- SHIPPED (dense, strided, s/d/c/z)
```

LAPACK is the *algorithm* layer (factorizations); the analogy that holds
is that LAPACK won by standardizing **semantics with a reference
implementation**, not performance. spec.md + conformance.c already are
that for the physics layer.

## Runnable today with zero ABI changes

- **Structured meshes: strides *are* stencils.** A 2-D 5-point Laplacian
  is `daxpy`-shaped calls with `inc = n` between neighbour rows; a fused
  `lap5`/`lap7` generic kernel takes four stride args and follows the
  exact same calling discipline (and threads for free, see below).
- **Low-storage RK is `axpby`/`1xypa`.** RK2/3/4-Williamson is
  `q += h*y; y = f(q) + beta*y` — the Williamson pair is precisely the
  2-register `1xypa` shape. That is why 1xypa is on the menu.
- **Geometry is our kernels in disguise.** Face normals = `3cross` of
  tangents, areas = `1norm`, metrics = `3dot`/`3crossdot`. A mesher
  replacing six hand-written loops with `v3blas` calls needs no
  committee — a quiet, standalone adoption channel.

## The real missing ABI: indirection

Unstructured CFD lives in `x[idx[i]]`: face->cell gather, Riemann solve
(still exactly our kernel shape), cell scatter. Phase-2 spec addition:
`blasint *idx` arguments, same semantics (no allocations, restrict,
per-array increments, bitwise reference). Ask only after the dense
element-wise subset lands — it is a much bigger standardization bite.

## Where LAPACK legitimately enters

Incompressible/projection steps. A pressure-Poisson CG is
`dot` + `axpy` + spmv — the solver is *assembled from* the layer, not
shipped in it; Helmholtz in complex mode uses the `c`/`z` variants.

## Ordered steps

1. **Exemplar toy solver in `benchmark/`**: structured Euler or
   diffusion, RK via `axpby`/`1xypa`, upwind flux via scalars, written
   using *only* v3blas calls — no hand-fused kernels. Converts "nice
   kernels" into "a programming model" and exposes exactly which call
   the model gets stuck on.
2. **Add the stencil-shaped generic kernels it demands**: `lap5`/`lap7`
   (stride params) and `3flux` (Rusanov-flavour face kernel). Both
   dense, so they fit the current ABI and ship like the others.
3. **Multithread the interfaces** (next section) and rerun the sweep on
   a multicore machine — adoption needs scaling numbers.
4. **BLAS Extensions forum RFC**: dense element-wise + stride-stencil
   subset with the conformance suite; indirection/CSR-shaped ops
   explicitly parked as Phase 2.
5. In the wake of acceptance: BLIS/OpenBLAS stencil ports, and geometry
   kernels ("meshkit") as their own small spec.

## Multithreading status and options

Today **all 19 physics ops are serial**: the interface files call the
`*_K` macro (the raw kernel) directly, e.g. `interface/axpy.c` by
contrast builds `nthreads = num_cpu_avail(1)` and dispatches through
`blas_level1_thread(...)`. No environment variable can change this.

Two credible routes, both semantics-neutral (element-wise ops have no
cross-thread reductions, so results stay bitwise-identical under any
slicing — unlike `dot`):

- **Interface route (upstream style, preferred for a PR):** mirror the
  `axpy.c` pattern. Complication: `blas_level1_thread()` has a fixed
  axpy-shaped argument list; our ops take up to 8 arrays. Needs either
  a generalized sibling in `driver/others/blas_l1_thread.c` or a small
  op-local pool loop. Zero new build options, reuses their thread pool.
- **OpenMP route (simplest code, best for the RFC demo):**
  `#pragma omp parallel for` on the element loop in
  `kernel/generic/{,z}*.c`, active only under `#ifdef USE_OPENMP`
  (already a recognized make/cmake option) and above a size threshold.
  Users build `-DUSE_OPENMP=1` and control width with
  `OMP_NUM_THREADS` (or `OPENBLAS_NUM_THREADS` when built that way).

`1xypa` stays single-threaded on purpose (register-coupled recursion).

## Running the benchmarks on a big machine

The README "one paste" (after "Quick start") is the full recipe; on a
many-core box:

```sh
git clone --recurse-submodules https://github.com/sacielo/blpp.git
cd blpp
git -C openblas apply ../blpp/physics_kernels.patch   # submodule sits at e016600
cmake -S openblas -B openblas/build -DCMAKE_BUILD_TYPE=Release -DNO_LAPACK=1
cmake --build openblas/build -j"$(nproc)"             # full RAM here, no -j1 needed
cmake --install openblas/build --prefix "$PWD/pkgs/openblas"
./openblas/build/utest/openblas_utest      # 212 checks
./openblas/build/utest/openblas_utest_ext  # 2645 checks
cmake -S benchmark -B benchmark/build && cmake --build benchmark/build
ctest --test-dir benchmark/build --output-on-failure
```

Then the 25-run sizemap sweep, plots into `benchmark/runs-<hostname>/`:

```sh
cd benchmark && NREP=25 ./sweep25.sh && python3 analyze25.py   # numpy+matplotlib
```

Caveats on a multicore box:

- The physics rows are single-threaded today; the bigger machine still
  wins via clock/L2/L3, and `runs-<host>`/`sizemap25-<host>.png` keep
  results from different boxes side by side.
- Classic OpenBLAS rows (`daxpy`, `ddot`, gemm) DO spawn
  `nproc` threads by default — pin them with `OPENBLAS_NUM_THREADS=1`
  for sizemaps, or set it >1 to see what interface threading would buy
  our ops.
- For the OpenMP route once implemented: rebuild with
  `-DUSE_OPENMP=1` and sweep with `OMP_NUM_THREADS=1..n`.
