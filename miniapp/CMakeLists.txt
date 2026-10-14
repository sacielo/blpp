cmake_minimum_required(VERSION 3.20)
project(blpp_miniapp C)

# Mini-apps: working physics codes on the public API (sod2d: 2D Euler;
# tops3d: batched rigid tops + banded L2).  Links against an *installed*
# patched OpenBLAS only - produce it with ../docs/backends/OPENBLAS.install.sh.
set(OPENBLAS_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/../pkgs/openblas
    CACHE PATH "OpenBLAS install prefix (include/openblas + lib or lib64)")

if(NOT EXISTS ${OPENBLAS_PREFIX}/include/openblas/v3blas.f90)
  message(FATAL_ERROR
    "No patched OpenBLAS found under ${OPENBLAS_PREFIX}. "
    "Run docs/backends/OPENBLAS.install.sh first.")
endif()

set(OPENBLAS_LIB ${OPENBLAS_PREFIX}/lib/libopenblas.a)
if(NOT EXISTS ${OPENBLAS_LIB})
  set(OPENBLAS_LIB ${OPENBLAS_PREFIX}/lib64/libopenblas.a)
endif()
if(NOT EXISTS ${OPENBLAS_LIB})
  message(FATAL_ERROR "libopenblas.a not found under ${OPENBLAS_PREFIX}/lib[64]/"
    " - rerun docs/backends/OPENBLAS.install.sh")
endif()

add_executable(sod2d sod2d.c)
target_include_directories(sod2d PRIVATE ${OPENBLAS_PREFIX}/include/openblas)
target_link_libraries(sod2d PRIVATE ${OPENBLAS_LIB} pthread m)

add_executable(tops3d tops3d.c)
target_include_directories(tops3d PRIVATE ${OPENBLAS_PREFIX}/include/openblas)
target_link_libraries(tops3d PRIVATE ${OPENBLAS_LIB} pthread m)

# Independent reference for tops3d body 0: must NOT link BLAS.
add_executable(reftop ref/reftop.c)
