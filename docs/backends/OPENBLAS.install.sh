#!/bin/sh
# Build and install the OpenBLAS backend of blpp from scratch.
# Run it from anywhere, once, on a clean clone:
#     ./docs/backends/OPENBLAS.install.sh
#
# Env knobs:
#   JOBS     parallelism (default: nproc; use JOBS=1 on small machines)
#   PREFIX   install prefix (default: <repo>/pkgs/openblas)
#   CMAKE    cmake executable (default: cmake)
# Any extra arguments are appended to the cmake configure call.
set -e

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$root"
prefix=${PREFIX:-$root/pkgs/openblas}
jobs=${JOBS:-$(nproc 2>/dev/null || echo 1)}
cmake=${CMAKE:-cmake}

# submodule at the pinned commit
[ -e openblas/CMakeLists.txt ] || git submodule update --init openblas

# apply the patch unless it is applied already (idempotent)
if git -C openblas apply --check physics_kernels.patch 2>/dev/null; then
    git -C openblas apply physics_kernels.patch
    echo "patched openblas @ $(git -C openblas rev-parse --short HEAD)"
fi

"$cmake" -S openblas -B openblas/build \
    -DCMAKE_BUILD_TYPE=Release -DNO_LAPACK=1 "$@"
"$cmake" --build openblas/build -j"$jobs"
"$cmake" --install openblas/build --prefix "$prefix"

cat <<EOF

installed: $prefix
verify:    ./openblas/build/utest/openblas_utest       (220 checks)
           ./openblas/build/utest/openblas_utest_ext  (2765 checks)
           cmake -S benchmark -B benchmark/build && cmake --build benchmark/build
EOF
