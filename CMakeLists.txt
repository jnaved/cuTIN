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
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found. Multi-threading enabled for OpenMP module.")
else()
    message(WARNING "OpenMP NOT found. OpenMP module will compile with stubs or fallback.")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 1. Common Module (DataLoader, Image Processing, Utilities)
add_library(tin_common STATIC
    src/common/dataloader.cpp
)
target_include_directories(tin_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 2. Sequential Reference Module (Teammate - Sequential)
add_library(tin_sequential STATIC
    src/sequential/conv.cpp
    src/sequential/dense.cpp
    src/sequential/train.cpp
)
target_include_directories(tin_sequential PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 3. OpenMP Parallel Module (Teammate - OpenMP)
add_library(tin_openmp STATIC
    src/openmp/conv.cpp
    src/openmp/dense.cpp
    src/openmp/train.cpp
)
target_include_directories(tin_openmp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tin_openmp PUBLIC OpenMP::OpenMP_CXX)
endif()

# 4. CUDA Accelerated Module
add_library(tin_cuda STATIC
    src/cuda/conv.cu
    src/cuda/dense.cu
    src/cuda/train.cu
)
target_include_directories(tin_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Main Training & Inference Executable
add_executable(cuTIN src/main.cpp)
target_link_libraries(cuTIN PRIVATE
    tin_cuda
    tin_openmp
    tin_sequential
    tin_common
)

# Test & Numerical Equivalence Suite Executable
add_executable(test_equivalence tests/test_equivalence.cpp)
target_link_libraries(test_equivalence PRIVATE
    tin_cuda
    tin_openmp
    tin_sequential
    tin_common
)
