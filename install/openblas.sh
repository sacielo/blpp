#!/bin/sh
echo "1. ==> cloning OpenBLAS"
git submodule update --init openblas

echo "2. ==> applying physics_kernels.patch"
git -C openblas apply ../physics_kernels.patch

echo "3. ==> building OpenBLAS"
if command -v ccmake ; then MYCMAKE="ccmake"; else MYCMAKE="cmake"; fi
$MYCMAKE -S openblas -B openblas/build -DCMAKE_BUILD_TYPE=Release \
 -DBUILD_BENCHMARKS=OFF -DBUILD_LAPACK_DEPRECATED=OFF -DBUILD_RELAPACK=OFF \
 -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DBUILD_TESTING=ON -DC_LAPACK=OFF \
 -DBUILD_WITHOUT_CBLAS=OFF -DBUILD_WITHOUT_LAPACK=ON -DBUILD_WITHOUT_LAPACKE=ON \
 -DNO_WARMUP=ON -DPHYS_KERNELS_O3=ON  -DCMAKE_INSTALL_PREFIX=${2-"pkgs/openblas"}
cmake --build openblas/build -j$1
cmake --install openblas/build --prefix ${2-"pkgs/openblas"}

echo "4. ==> OpenBLAS test suites"
./openblas/build/utest/openblas_utest
./openblas/build/utest/openblas_utest_ext

echo "5. ==> building benchmark/"
cmake -S benchmark -B benchmark/build
cmake --build benchmark/build -j$1

echo "6. ==> building miniapp/"
cmake -S miniapp -B miniapp/build
cmake --build miniapp/build -j$1

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
