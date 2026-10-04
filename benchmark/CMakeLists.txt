cmake_minimum_required(VERSION 3.16)
project(benchmark CXX)

# This project only builds the benchmark. It links against an OpenBLAS
# that must already be built, e.g.:
#   cmake -B build -DNO_LAPACK=1 && make -j1 openblas_static
# inside the OpenBLAS tree.
set(OPENBLAS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../openblas
    CACHE PATH "OpenBLAS source tree (pre-built into ${OPENBLAS_ROOT}/build)")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(benchmark benchmark.cpp)
target_compile_features(benchmark PRIVATE cxx_std_17)
target_include_directories(benchmark PRIVATE
  ${OPENBLAS_ROOT}
  ${OPENBLAS_ROOT}/build)
target_link_libraries(benchmark PRIVATE
  ${OPENBLAS_ROOT}/build/lib/libopenblas.a
  pthread
  m)
