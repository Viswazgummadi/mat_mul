cmake_minimum_required(VERSION 3.10)
project(MatrixMulOpt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall")

include_directories(include)

# Source files
set(SOURCES
    src/naive.cpp
    src/loop_reorder.cpp
    # Add other implementations here
)

# Benchmark executable
add_executable(benchmark_runner benchmark/benchmark_harness.cpp ${SOURCES})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_libraries(benchmark_runner pthread)
endif()
