#!/bin/sh
# blpp OpenBLAS backend: clone, patch, build, install, verify, and build
# the benchmark/test suite. Just run it:
#
#     ./docs/backends/OPENBLAS.install.sh
#
# Env knobs:
#   JOBS     parallelism (default: nproc; use JOBS=1 on small machines)
#   PREFIX   install prefix (default: <repo>/pkgs/openblas)
#   CMAKE    cmake executable (default: cmake)
# Any extra arguments are appended to the OpenBLAS cmake configure call.
set -e

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$root"
prefix=${PREFIX:-$root/pkgs/openblas}
jobs=${JOBS:-$(nproc 2>/dev/null || echo 1)}
cmake=${CMAKE:-cmake}
sha=e0166008be8e466242aa76b2ff75ce3f0fbf574a   # pinned upstream commit

# 1. OpenBLAS checkout at the pinned commit (clone if not there)
if [ ! -e openblas/CMakeLists.txt ]; then
    echo "==> cloning OpenBLAS @ $sha"
    git submodule update --init openblas 2>/dev/null ||
        git clone https://github.com/OpenMathLib/OpenBLAS.git openblas
    git -C openblas checkout -q "$sha"
fi

# 2. Apply the patch unless applied already (idempotent).
#    git -C runs inside openblas/, so the patch path is openblas-relative.
if git -C openblas apply --check ../physics_kernels.patch 2>/dev/null; then
    echo "==> applying physics_kernels.patch"
    git -C openblas apply ../physics_kernels.patch
fi

# 3. Build and install the library (threaded: the physics kernels ride
#    OpenBLAS's own SMP machinery, so tests must cover the threaded path)
echo "==> building OpenBLAS (-j$jobs; on small machines: JOBS=1)"
"$cmake" -S openblas -B openblas/build \
    -DCMAKE_BUILD_TYPE=Release -DNO_LAPACK=1 \
    -DSMP_SERVER -DUSE_THREAD -DMAX_CPU_NUMBER="$jobs" "$@"
"$cmake" --build openblas/build -j"$jobs"
"$cmake" --install openblas/build --prefix "$prefix"

# 4. Upstream's own suites must pass on the patched library
echo "==> OpenBLAS test suites (expect 220 and 2765 checks, 0 failures)"
./openblas/build/utest/openblas_utest
./openblas/build/utest/openblas_utest_ext

# 5. Compile (but do not run) the benchmark/test/example suite
echo "==> building benchmark/"
"$cmake" -S benchmark -B benchmark/build
"$cmake" --build benchmark/build -j"$jobs"

# 6. Compile (but do not run) the mini-apps
echo "==> building miniapp/"
"$cmake" -S miniapp -B miniapp/build
"$cmake" --build miniapp/build -j"$jobs"

cat <<EOF

==> done. installed: $prefix
    The suite is built, now you can run it from benchmark/build/:
      ./cblas_test          raw cblas_* layer   (205 checks)
      ./v3blas_test         C11 front-end       (195 checks)
      ./v3blas_cpp_test     C++ front-end
      ./v3blas_f90_test     Fortran front-end   (72 checks)
      ./cblas_bench [n] [s|d|c|z]   timing tables; *_example: tutorials
    Every *_test must end with "OK (0 failures)".
    Mini-apps (see miniapp/README.md):
      ../../miniapp/build/sod2d        2D Sod tube -> sod2d.vtk (~9 s)
EOF
