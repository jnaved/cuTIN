cmake_minimum_required(VERSION 3.18)
project(cuTIN LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default CUDA architectures (Turing sm_75, Ampere sm_80/sm_86)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
endif()

# Compiler optimization flags
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/O2>
        $<$<COMPILE_LANGUAGE:CXX>:/W3>
        $<$<COMPILE_LANGUAGE:CXX>:/openmp>
    )
else()
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    )
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
endif()

# Find OpenMP for multi-threaded CPU acceleration
find_package(OpenMP REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Core Library (DataLoader + 3 Backends: Sequential, OpenMP, CUDA)
add_library(cutin_core STATIC
    src/dataloader.cpp
    src/sequential.cpp
    src/openmp.cpp
    src/cuda.cu
)
target_include_directories(cutin_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cutin_core PUBLIC OpenMP::OpenMP_CXX)

# Main Training & Benchmark Runner
add_executable(cuTIN src/main.cpp)
target_link_libraries(cuTIN PRIVATE cutin_core)

# Numerical Equivalence & Unit Test Suite
add_executable(test_equivalence tests/test_equivalence.cpp)
target_link_libraries(test_equivalence PRIVATE cutin_core)
