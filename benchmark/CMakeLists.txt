cmake_minimum_required(VERSION 3.16)
project(benchmark CXX C Fortran)

# This project only builds these executables. It links against an
# *installed* OpenBLAS and nothing else (source and build trees are not
# added to the include path: they carry duplicate copies of the same
# headers/macros). Install OpenBLAS once, e.g.:
#   cd ../openblas
#   cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-O2" -DNO_LAPACK=1
#   cmake --build build -j1
#   cmake --install build --prefix ../pkgs/openblas
set(OPENBLAS_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/../pkgs/openblas
    CACHE PATH "OpenBLAS install prefix (include/openblas + lib)")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# OpenBLAS installs its headers (and v3blas.hpp / v3blas.f90) to
# <prefix>/include/openblas - that is the only place these targets look.
function(add_ob_target target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE ${OPENBLAS_PREFIX}/include/openblas)
  target_link_libraries(${target} PRIVATE ${OPENBLAS_PREFIX}/lib/libopenblas.a pthread m)
endfunction()

add_ob_target(benchmark benchmark.cpp)

# v3blas_bench: cost of the v3blas.h interface (call + result free) vs the
# raw cblas extension kernels with pre-allocated outputs.
add_ob_target(v3blas_bench v3blas_bench.cpp)

# v3blas_example: minimal C11 example of the v3blas.h interface (the
# precision-generic names need C11 _Generic).
add_ob_target(v3blas_example v3blas_example.c)

# v3blas_hpp_test: C++ overloads from v3blas.hpp, bitwise vs cblas_*.
add_ob_target(v3blas_hpp_test v3blas_hpp_test.cpp)

# v3blas_f90_test: Fortran generic module procedures, vs equations
# recomputed in Fortran.  The module itself comes from the installed
# v3blas.f90; CMake resolves the module dependency and compile order.
add_ob_target(v3blas_f90_test
  ${OPENBLAS_PREFIX}/include/openblas/v3blas.f90 v3blas_f90_test.f90)

set_target_properties(benchmark v3blas_bench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
set_target_properties(v3blas_example PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
set_target_properties(v3blas_hpp_test PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON)
