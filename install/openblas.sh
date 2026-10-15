#!/bin/sh
echo "1. ==> cloning OpenBLAS"
git submodule update --init openblas

echo "2. ==> applying physics_kernels.patch"
git -C openblas apply ../physics_kernels.patch

echo "3. ==> building OpenBLAS"
ccmake -S openblas -B openblas/build -DCMAKE_BUILD_TYPE=Release
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
