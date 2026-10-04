# Adding physics kernels to blas

Every MHD, CFD and electromagnetics code re-implements the same few-line
element-wise cross / dot / Hadamard on N grid points. The operations are
memory-bound, same roofline class as SAXPY, arithmetic intensity well below 1.
They belong in BLAS, and BLAS implementations can take extensions now.

Example applications for physics include:

    F = (∇∧B) ∧ B / μ₀     Lorentz force
    S = E ∧ B / μ₀         Poynting flux
    Q = (∇∧B)·(∇∧B) / σ    Ohmic heating
    h = B·(∇∧B)            helicity


---

## Design principles
### Data types
- These are meant to be parallel operations on arrays (like Blas L1 kernel)
- 3-vectors are to be managed as 3 separate components, each being a plain C array, that the user will allocate independently and manage by themselves
- This gives automatically a SoA layout to the data, but you don't have to think of it, it will come naturally
- Don't worry if interfaces are long
- beware that none of these kernels is a reduction, they are all just component-wise operations on array
- the kernel prototype must then include 3 argument (the 3 components) plus one (inc) per each 3-vector
- Beware that there is no "grid" or space distribution of the vectors; these are just represented as such single-array component

### How to implement
- Kernels are added inside Openblas folder, alongside existing kernels, and only there. Not in dedicated places, but alongside existing kernels
- Do not use generators; do not create external tools; do not use python. Code them yourself in plain C, and modify OpenBLAS' Makefile and CMakeLists ONLY
- kernels won't have any compiled support struct, nor will tests; aside from structures that already exist in OpenBLAS
- Keep comment to a minimum! Especially never say what the code is not, or what doesn't exist; don't put developer notes in comments; don't put cross-references to other docs, to other files, and to any material you create. 
- In notes the following operators have to be used: wedge for cross product, dot or cdot for dot product, ".*" for hadamard (elementwise) product, justaposition or "*" for regular multiplication between scalars  
- Beware that this system has very low ram and cpu. When you compile, do it serially. Expect compilation to take a while.
- when in doubt, copy how everything is done for the axpy kernel
- start by implementing just one kernel, the "1xypa" kernel. Once that is working, notify me so I will test it.

### OpenBLAS operability
- Kernels are build whenever openblas is compiled; they must use either openblas' own Makefile AND CMakeLists
- the kernels must efficiently map to ISA (as fma or mul)
- the kernels must implemented in OpenBLAS' "generic" verison only (located under OpenBLAS/kernels/generic); not specifying any specific architecture by desing 
- the kernels will support openblas own threading scheme
- kernels must be available for both fortran and cblas; have s d c z precision, and have unit and cblas test
- If OpenBLAS uses macros or loops to specify the precision, you have to use those. Check if this means writing a single C file per kernel. 
- this extension has to be a part of openblas indistinguishable from native kernels.
- complex kernels will NOT use complex conjugation (for the time being)
- These kernels must be usable by simply compiling openblas and linking it to our code


## Kernel set

**`<p>` stands for the precision** — `s`, `d`, `c` or `z`. Every kernel,
constructor and type below is written once in that generic form, because
nothing about the mathematics or the calling convention changes between the
four. `<p>3cross` is four real symbols: `s3cross`, `d3cross`, `c3cross`,
`z3cross`. Where a concrete name matters — a stock BLAS symbol it must not
collide with, a test that only holds in one precision — the real name is
spelled out.

**`T` is the element type**, and it is a *different* thing from `<p>`: `T` is
the family's real scalar, `float` for `s` and `double` for the other three,
where one complex value occupies two adjacent `T`s. That is what CBLAS and
OpenBLAS both do, so a `<p>3v` for `c` or `z` points into a buffer of `n`
complex values and its component stride is `2 * cinc` reals.

**Over `c` and `z` every product in the tables below is bilinear.** Component
`k` is the pair at `x + 2*k*cinc`, written `x_k = (x_kr, x_ki)`, and

    x_k · y_k  =  (x_kr·y_kr − x_ki·y_ki)  +  i·(x_kr·y_ki + x_ki·y_kr)

with `+`, `−` and `a·u` componentwise on the pair. The cross is
`c₀ = x₁y₂ − x₂y₁`, `c₁ = x₂y₀ − x₀y₂`, `c₂ = x₀y₁ − x₁y₀`, each `x_j·y_k`
being that product. No conjugate appears anywhere. `<p>1sqr` is `x·x` and is
genuinely complex-valued; `<p>1sqrt` is the principal square root of it, which
is why √ is the only stem in the set that needs a branch convention at all.
This convention is v1's and is versioned like any other semantics — see
§Versioning.

**The mathematics column is the right-hand side only.** No kernel writes its
first operand in place — every output is a new value, spelled at the call site
by passing an input handle again if that is what you want. `a·x + y` is the
shape, not the assignment.

19 bodies × 4 precisions is **76 kernel symbols**. Add the support symbols
they need — 24 Layer-1 constructors, 12 Layer-0 constructors, 8 status
functions — for **120 exported symbols** in all (§Acceptance criteria).

### Layer 1 — 3-vectors (10)

| symbol | mathematics | example |
|---|---|---|
| `<p>3had` | `w = x.*y` | field coefficient ρ·v |
| `<p>3cross` | `w = x∧y` | |
| `<p>3crossscal` | `w = a·(x∧y)` | F, S |
| `<p>3dot` | `r = x·y` | h, Q |
| `<p>3sqr` | `r = x·x` | |
| `<p>3crossdot` | `r = (x∧y)·w` | α-effect |
| `<p>3crosssqr` | `r = (x∧y)·(x∧y)` | Alfvén speed |

Composites name the inner op first: `crossdot` is `(x∧y)·z`, and it also
spells `x·(y∧z)` by commutativity. `<p>3crosssqr` is `<p>3dot` on a cross with
itself. `<p>3sqr` is `<p>3dot(x,x)` spelled shorter; the duplication is admitted.

### Forks (3)

Two outputs, one body, one symbol. `_` separates clauses; a clause spells its
operands. All three share the first operand and vary the second.

| symbol | shape |
|---|---|
| `<p>3crossxy_crossxz` | `u=x∧y`, `v=x∧w` — div B |
| `<p>3crossxy_dotxz` | `u=x∧y`, `r=x·w` |
| `<p>3dotxy_dotxz` | `r=x·y`, `q=x·w` — Q, h |

With x = y = ∇∧B and z = B, `<p>3dotxy_dotxz` gives Q₀ and h in one pass.
Passing the same handle twice is legal and free.

F and S share `B` as the *second* operand of a non-commutative cross, so no
fork shape can hold both.

When a fork's second and third operands are the same handle, both clauses read
one operand pair instead of two, and an implementation may take a shorter path
for it. That is a licence, not a requirement: the result is the same either way.

Forks cap at two outputs: a cross has three live component expressions, so two
clauses need six and three would need nine, over x86-64's sixteen vector
registers.

### Layer 0 — scalar arrays (6)

| symbol | mathematics | stock equivalent |
|---|---|---|
| `<p>1xypa` | `r= q.*t+a`| none |
| `<p>1norm` | `r= √(q·q)` | none |


## Naming

| part | means |
|---|---|
| `s` `d` `c` `z` | precision |
| `r` `q` `t` | arrays of scalars |
| `u` `v`  `x` `y` `w`  | vectors (logically, 3 separate arrays) |
| `cross` `dot` `had` `sqr` `sqrt` | element operation |
| juxtaposition | composition, inner op first |
| `_` | fork separator |

