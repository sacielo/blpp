# Agent guide — implementing physics kernels

Instructions and hard-won rules for anyone (human or agent) implementing or
extending these kernels **in any BLAS backend**. What to implement (shapes,
equations, contracts) is [`KERNELS.md`](KERNELS.md); how a specific library
is wired is in [`backends/`](backends/). This file is the process.

## Prime directives

1. **Portability is the product.** The same op name and semantics in every
   BLAS is the goal; any single library's cleverness is the enemy. Kernels
   are plain, portable C — no per-arch assembly, no intrinsics, no new
   dependencies, no code generators, no Python, no external tooling. Write
   the C yourself.
2. **Blend in.** In each backend, the extension must be indistinguishable
   from native kernels: that backend's own build system, own test
   framework, own symbol/export machinery. Modify existing build files;
   do not bolt on parallel ones.
3. **When in doubt, copy `axpy`.** The reference kernel for every
   convention (file layout, dispatch, unrolled unit-stride fast path +
   strided fallback, test style, interface shape).
4. **One kernel, end to end.** First one kernel all the way (kernel →
   interfaces → build wiring → exports → tests → green run), *then* the
   next. Never mass-generate half-wired kernels.
5. **Minimal comments; never describe absence.** No "this is not a
   reduction", no cross-references to other docs/files, no developer notes.
   Comments say what the code does, or nothing.

## Code rules

- Generic-only kernels: one implementation per backend's *generic* tier;
  architecture specialization is a later, separate decision (threading
  layers and compiler flags already give most of the win).
- Precision via the backend's own mechanism (OpenBLAS: compile the same
  file per precision with `-DFLOAT=…` macros). Typically 2 files per op:
  one real (`s`/`d` share), one complex (`c`/`z` share — arithmetic, not
  just type, differs).
- 3-vectors are 3 arrays + 3 strides, always. Interfaces are long — fine.
  No support structs in kernels or tests beyond what the backend already
  has.
- Complex arithmetic bilinear, never sesquilinear; guarded-division `eps`
  and geometric scalars real in every precision (see KERNELS contracts).
- Public argument order reads the kernel's name; outputs last; the
  documented exceptions are exactly that — don't add new ones casually.
- Strided ops: unrolled unit-stride fast path, correct general fallback.
  Mapping to FMA/mul is the compiler's job at `-O3` — don't hand-fuse,
  don't `volatile`, never `-ffast-math` (it breaks the per-operation
  rounding that BLAS correctness is built on; `-O3` is the ceiling).
- Fixed summation orders in KERNELS are contracts, not style — a future
  per-arch kernel must reproduce them bitwise.

## Test discipline

- Every op × precision gets the backend-native unit test (Fortran entry)
  **and** a cblas test, plus front-end tests where the op exists.
- References are computed in **long double** inside the test, from the
  same inputs, by the equation — never by a second implementation.
- Correctness and timing never mix: `*_test` programs verify and time
  nothing; `*_bench` programs time and verify nothing; `*_example`
  programs teach (one `main`, textbook calls, printed output is the doc).
- Tests double as spec documentation: they print a paste-ready
  MATLAB/Octave block of the worked first-element equations (bilinear
  notation, `% => value` self-checks).
- Benchmark input data is `hash2(seed, i)`-derived in [−0.5, 0.5) — full
  avalanche. Never ramp/affine/LCG data: nearly-equal products in cross
  products cancel catastrophically at float precision as n grows.
- Aliasing/edge contracts (`n≤0`, `inc<0`, `inc=0`, in-place, NaN) are
  pinned by permanent checks in the raw-layer test, not prose alone.

## C preprocessor traps (all bitten, twice)

- Commas inside braces split macro arguments — wrap initializers in
  helper macros (`mkcc(1,2)` style) or extra parens.
- The preprocessor substitutes macro-parameter tokens *after a dot*
  (`x.P` where `P` is a parameter) — name macro parameters UPPERCASE and
  avoid member names that collide.
- `#P` stringification needs the parameter to exist; `char` passed to
  `%s` segfaults; a missing `;` before a line-continuing `\` changes
  meaning silently.

## Fortran traps (gfortran 14, and friends)

- Names are case-insensitive: `X1` *is* `x1` — never differ only by case.
- Mixed assumed-shape dummies (`a(3, :)`) are rejected; only full `(:,:)`
  (or `(:)`) is portable. Assumed-shape *elements* cannot be actuals to
  array dummies; contiguous whole sections `x(:, c)` pass zero-copy.
- `cmplx(...)` needs explicit `kind=8` for double components.
- Free-form line limit is 131 characters; `sed` on format strings eats
  closing parens — edit formats by hand.
- Raw F77 symbols take every argument by reference, including scalars.

## The machine (dev box facts)

- Reference dev box: 1-core ARMv6, 426 MiB RAM. **Build serially (`-j1`)**
  — a parallel build OOMs; a full BLAS build is ~1.5–2 h. Long builds go
  into `setsid nohup ... &` logs you poll, never foreground waits.
- Never make full-tree copies of the source to "diff safely"; never run
  heavy benchmarks (25-rep sweeps) on the Pi.
- `pkill -f` can match your own shell — check PIDs. gdb is unreliable
  here; debug with `stdbuf -o0` printfs. No Octave on the Pi — print the
  Octave block and let the user run it elsewhere.
- Shell/tooling: the bash tool's cwd resets per call; `&&`-chain failures
  still let later `;`-separated statements run (masked a failed-script
  incident) — check exit codes of the thing that matters. Fancy `sed`
  surgery on C source breaks; when a scratch file tangles, rewrite it
  wholesale.

## Git hygiene (this repo)

- Backends are submodules kept **unmodified**; all work ships as one
  accumulating patch per backend (see backend doc for the workflow).
  Commit in the superproject with explicit paths; never stage a dirty
  submodule pointer or `.mod` build artifacts; commit identity:
  `sacielo` / `29252493+sacielo@users.noreply.github.com`.
- Push with `git push origin HEAD`.
- **Never commit performance numbers** — perf is host-scoped, lives in no
  file in the repo (host docs or nowhere).
- Commit messages say *why*; the diff says *what*.

## North star (why the shapes are what they are)

The deliverable is a **versioned ABI + semantics spec plus a conformance
suite runnable against any vendor library at runtime** — providers adopt
standards they can test against, not opinions they must trust. Two-layer
split keeps it portable: providers implement only the C ABI layer; users
write only the front-end layer. Conformance is tiered: tolerance-based
across providers (FMA/rounding differ), bitwise only within one library.
Committees ratify shipping practice — the spec/forum pitch comes *after*
first open merges; the cautionary precedent is XBLAS (reference
implementation, test deck, no third-party callers, never adopted). The
invariant is the spec + conformance vectors; library linkage is one era's
delivery vehicle. Current backlog and order of attack:
[`TODO.md`](TODO.md).
