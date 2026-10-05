# Compilation bugs — current state

One confirmed compile error:

1. n_zero CTEST in the 40 generated ext test files (utest/test_extensions/test_*{had,cross,...}.c)
   The generator emitted check_<stem>(0) (or check_<stem>(0, a_val) for 3crossscal) — but check_<stem> requires all stride arguments. gcc -fsyntax-only
   gives:
   ```
     error: too few arguments to function 'check_3had'
   ```
   One line per file, all 40 files affected. Fix is trivial (append the full stride list, e.g. all-ones) — I had just started it when you paused.

Wiring errors found by inspection (not compile errors, but they would have broken the build):

2. common_level1.h was truncated — an earlier sed-based insertion destroyed the file tail (dcopy_k … zaxpby_k + the closing #ifdef __CUDACC__ block). Fixed
   and verified (restored, #if balance 0, diff now purely additive: +120/−0).

3. cblas.h duplicate prototypes — all four cblas_*3dotxy_dotxz declarations appear twice (canonical at line 481 + a leftover copy at 489 with different
   spacing). Legal C (redeclaration), compiles fine, but should be deleted.

Everything else is unverified, not broken:
The kernel stubs, interfaces, common_*.h, setparam-ref.c, gensymbol, and the 10 hand-written unit tests have never been compiled — that's the whole point of
the stub plan: one cmake + one build. Expected risk spots for the first build: positional alignment of setparam-ref.c, Fortran entry names at link, and the
10 unit-test files (the one I syntax-checked was clean).

And the structural issue you already flagged: those 40 files came from a bash generator, which violates your no-generators rule — they'll need hand
verification (or rewriting) regardless.
