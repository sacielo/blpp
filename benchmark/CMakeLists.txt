cmake_minimum_required(VERSION 3.16)
project(benchmark CXX C Fortran)

# This project only builds these executables. It links against an
# *installed* OpenBLAS and nothing else (source and build trees are not
# added to the include path: they carry duplicate copies of the same
# headers/macros). To produce that installed tree, run
#   ../docs/backends/OPENBLAS.install.sh
set(OPENBLAS_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/../pkgs/openblas
    CACHE PATH "OpenBLAS install prefix (include/openblas + lib or lib64)")

# pkgs/ is not in git; it is what the install step puts there.
if(NOT EXISTS ${OPENBLAS_PREFIX}/include/openblas/v3blas.f90)
  message(FATAL_ERROR
    "No patched OpenBLAS found under ${OPENBLAS_PREFIX}.\n"
    "Build and install it first:  docs/backends/OPENBLAS.install.sh")
endif()

# Fedora/openSUSE-style toolchains install to lib64/ instead of lib/.
set(OPENBLAS_LIB ${OPENBLAS_PREFIX}/lib/libopenblas.a)
if(NOT EXISTS ${OPENBLAS_LIB})
  set(OPENBLAS_LIB ${OPENBLAS_PREFIX}/lib64/libopenblas.a)
endif()
if(NOT EXISTS ${OPENBLAS_LIB})
  message(FATAL_ERROR "libopenblas.a not found under ${OPENBLAS_PREFIX}/lib[64]/"
    " - rerun docs/backends/OPENBLAS.install.sh")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# OpenBLAS installs its headers (and v3blas.hpp / v3blas.f90) to
# <prefix>/include/openblas - that is the only place these targets look.
# The Fortran targets also compile the installed v3blas.f90 module; CMake
# resolves the module dependency and the compile order.
function(add_ob_target target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE ${OPENBLAS_PREFIX}/include/openblas)
  target_link_libraries(${target} PRIVATE ${OPENBLAS_LIB} pthread m)
endfunction()

# Three roles, per interface (cblas_*, v3blas C11/C++/F90):
#   *_test     correctness: every op x precision vs a long double
#              reference; prints an Octave/MATLAB block to re-verify by
#              hand.  Nothing is timed.
#   *_bench    timing only: ms/run and FLOP/s tables.  Nothing is
#              verified (v3blas_bench's "overhead" column is the v3blas
#              call vs the cblas call it dispatches to).
#   *_example  minimal copy-paste tutorials, no tables.

# ---- raw cblas_* extension entry points --------------------------------
add_ob_target(cblas_test cblas_test.c)
add_ob_target(cblas_bench cblas_bench.cpp)
add_ob_target(cblas_example cblas_example.c)

# ---- v3blas.h (C11 _Generic) --------------------------------------------
# v3blas_bench also carries the per-kernel v3blas-vs-cblas overhead column.
add_ob_target(v3blas_test v3blas_test.c)
add_ob_target(v3blas_bench v3blas_bench.cpp)
add_ob_target(v3blas_example v3blas_example.c)

# ---- v3blas.hpp (C++ overloads) ------------------------------------------
add_ob_target(v3blas_cpp_test v3blas_cpp_test.cpp)
add_ob_target(v3blas_cpp_bench v3blas_cpp_bench.cpp)
add_ob_target(v3blas_cpp_example v3blas_cpp_example.cpp)

# ---- v3blas.f90 (Fortran generic module) ---------------------------------
add_ob_target(v3blas_f90_test
  ${OPENBLAS_PREFIX}/include/openblas/v3blas.f90 v3blas_f90_test.f90)
add_ob_target(v3blas_f90_bench
  ${OPENBLAS_PREFIX}/include/openblas/v3blas.f90 v3blas_f90_bench.f90)
add_ob_target(v3blas_f90_example
  ${OPENBLAS_PREFIX}/include/openblas/v3blas.f90 v3blas_f90_example.f90)

set_target_properties(cblas_bench v3blas_bench PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
set_target_properties(cblas_test cblas_example v3blas_test v3blas_example
  PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
set_target_properties(v3blas_cpp_test v3blas_cpp_bench v3blas_cpp_example
  PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON)
