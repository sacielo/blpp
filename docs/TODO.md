# TODO / live backlog

Done work is not tracked here — [`KERNELS.md`](KERNELS.md) lists what
exists, git log is the history. This file is what is *not yet done*, plus
decisions already closed so they don't reopen.

## Big iron (HPC target)

- [ ] Multicore perf: Pi threading is correctness-only (1 core). Rerun
      bench + MT checker on a real box, n ∈ {9999, 10001, 100000} ×
      `OPENBLAS_NUM_THREADS` ∈ {1,2,4,8}, all precisions, incl.
      inc=0/2/−1; tune `PHYS_MT_MIN` from the crossover curve.
- [ ] Before that, measure whether L1 threading is even wanted: these are
      bandwidth-bound streaming kernels (upstream disables L1 threading
      below n≈10000 for exactly this reason).
- [ ] Who threads at scale: MPI×OpenMP codes call BLAS *inside* the
      parallel region — intra-kernel threading oversubscribes (the NUMA
      footgun); for them single-thread kernels are the right default.
      Intra-BLAS threading serves single-rank workloads.
- [ ] Substrate for the HPC endgame: keep OpenBLAS + per-arch kernels
      (upstreaming territory) vs BLIS (built to extend; axpby precedent)
      vs app-side CUDA/HIP/SYCL with BLAS as CPU fallback. Decide after
      the multicore bench, not before.
- [ ] Oversubscription curiosity: 2 OpenBLAS threads on the 1-core Pi —
      does the 2nd thread overlap load stalls? Runs at
      `OPENBLAS_NUM_THREADS` ∈ {1,2,4}; perf-meaningless on 1 core,
      correctness-only.

## Standardization track (endgame — README "Roadmap")

- [ ] Conformance runner: standalone binary, `dlopen`s any BLAS (static
      MKL/ESSL too), probes symbols (LP64 + ILP64 name variants), runs
      all ops × 4 precisions against the long-double reference vectors
      from `v3blas_test.c`, prints a per-op pass/fail table. Must run on
      a vendor lib with zero recompilation — the outreach weapon.
- [ ] Versioned spec document (from KERNELS.md): ABI names/arg order/
      strides, ILP64 convention (`_ilp64` vs `_64` — propose one),
      complex scalar = re/im pair, inc≤0 + aliasing + NaN rules, fused
      multi-output ops flagged as beyond-classic-BLAS (a Level-1
      extension API ask).
- [ ] OpenBLAS RFC discussion (names + opt-in `PHYSICS_EXT` flag), then
      the PR as a commit series (per-op kernel+interface+tests, DCO
      sign-off, rebase onto master). Ready: the three representative
      `openblas/benchmark/` programs (3dot, 3crosscross, 3dotxy_dotxz ×
      4 prec, `.goto` style) already ship in the patch.
- [ ] BLIS port of one kernel as feasibility probe (refk + kern structs).
      Second adopter = the FOMO lever.
- [ ] BLAS Extensions forum: propose the subset whose shape fits their
      Level-1 grammar (`1xypa`/`3had`/`3dot` family); fused forks as a
      separate API-extension request.
- [ ] Migration of real community codes (mesher geometry loops, an MHD
      toy) is roadmap work of equal rank to any PR — the binding
      constraint is users, not engineering.

## New kernels

- [ ] `3minmod` (TVD slope limiter), `3sel` (branchless upwind selection)
      — shapes fit today's ABI, nobody has asked yet.
- [ ] Indirection ABI (`x[idx[i]]` gather/flux/scatter) for unstructured
      meshes — design in [`specs/grand_plan.md`](specs/grand_plan.md);
      separate spec version, do not bend today's strides to fake it.

## New backends (the actual point)

- [ ] oneMATH (FOSS oneMKL) port probe — same 80 symbols, their
      test/export machinery.
- [ ] CLBlast probe (OpenCL): which ops map to kernels cheaply.
- [ ] A second backend is what forces the conformance runner above to
      exist first.

## Closed decisions (don't reopen silently)

- Public arg order reads the name; `1xypa` scalar moved to name position
  (2026-10-13); the two documented exceptions (`3crossscal` leading
  scalar, stencils leading `alpha`) stand.
- Internal OpenBLAS CNAME keeps scalar-after-dummies (the SMP marshaller
  ABI); only public layers reorder.
- Reduction-output ops stay serial for the pinned summation order.
- Benches time, tests verify, examples teach (2026-10-12 split).
- Octave equation blocks print from tests only.
- Fortran front-end stays atop the cblas kernels — no change to the
  cblas layer for Fortran's sake.
