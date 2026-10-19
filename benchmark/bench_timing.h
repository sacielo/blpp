#ifndef BENCH_TIMING_H
#define BENCH_TIMING_H

/* Timing budget for every bench in this folder, C++ and Fortran
 * (v3blas_f90_bench.F90 includes this file through the preprocessor).
 * Run each kernel until kMinTime of wall time has passed AND at least
 * kMinIters calls completed; stop at kMaxTime no matter what.  The
 * floors exist to keep a measurement above chronometer sensitivity,
 * nothing else; repetition statistics live in the sweep (sweep25.sh +
 * analyze25.py), not in one window.  Override any value at compile
 * time: -DK_MIN_TIME=... etc. */
#ifndef K_MIN_TIME
#define K_MIN_TIME 0.005 /* seconds per kernel  */
#endif
#ifndef K_MIN_ITERS
#define K_MIN_ITERS 5 /* calls per kernel    */
#endif
#ifndef K_MAX_TIME
#define K_MAX_TIME 1.0 /* seconds per kernel  */
#endif

#endif /* BENCH_TIMING_H */
