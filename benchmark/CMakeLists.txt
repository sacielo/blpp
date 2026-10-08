cmake_minimum_required(VERSION 3.16)
project(benchmark CXX C)

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

# OpenBLAS installs its headers to <prefix>/include/openblas
function(add_ob_target target source)
  add_executable(${target} ${source})
  target_include_directories(${target} PRIVATE ${OPENBLAS_PREFIX}/include/openblas)
  target_link_libraries(${target} PRIVATE ${OPENBLAS_PREFIX}/lib/libopenblas.a pthread m)
endfunction()

add_ob_target(benchmark benchmark.cpp)

# pblas_bench: cost of the pblas.h interface (call + result free) vs the
# raw cblas extension kernels with pre-allocated outputs.
add_ob_target(pblas_bench pblas_bench.cpp)

# pblas_example: minimal C99 example of the pblas.h interface.
add_ob_target(pblas_example pblas_example.c)

set_target_properties(benchmark pblas_bench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
set_target_properties(pblas_example PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)
