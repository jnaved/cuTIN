#ifndef CUTIN_UTILS_HPP
#define CUTIN_UTILS_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <string>
#include <iomanip>
#include <algorithm>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error at %s:%d - %s\n",                   \
                         __FILE__, __LINE__, cudaGetErrorString(err));           \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)
#else
#define CUDA_CHECK(call) (call)
#endif

namespace cutin {

// CPU high-resolution timer
class CPUTimer {
public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time_;
        return elapsed.count(); // in milliseconds
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

#ifdef __CUDACC__
// GPU event-based timer
class GPUTimer {
public:
    GPUTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~GPUTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_, stream);
    }

    float stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_, stream);
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms; // in milliseconds
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};
#endif

// Weight initialization utilities
inline void kaiming_normal_init(float* weights, size_t size, size_t fan_in, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    float stddev = std::sqrt(2.0f / static_cast<float>(fan_in));
    std::normal_distribution<float> dist(0.0f, stddev);
    for (size_t i = 0; i < size; ++i) {
        weights[i] = dist(gen);
    }
}

inline void kaiming_uniform_init(float* weights, size_t size, size_t fan_in, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    float bound = std::sqrt(6.0f / static_cast<float>(fan_in));
    std::uniform_real_distribution<float> dist(-bound, bound);
    for (size_t i = 0; i < size; ++i) {
        weights[i] = dist(gen);
    }
}

inline void zero_init(float* data, size_t size) {
    std::fill(data, data + size, 0.0f);
}

// Numerical comparison helper for tests
inline bool check_close(const float* a, const float* b, size_t size, float atol = 1e-4f, float rtol = 1e-3f, size_t max_errors = 5) {
    size_t errors = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float diff = std::fabs(a[i] - b[i]);
        float tol = atol + rtol * std::fabs(b[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tol) {
            if (errors < max_errors) {
                std::cerr << "Mismatch at index " << i << ": a=" << a[i]
                          << ", b=" << b[i] << ", diff=" << diff << " > tol=" << tol << "\n";
            }
            errors++;
        }
    }
    if (errors > 0) {
        std::cerr << "Total mismatches: " << errors << " / " << size
                  << " (Max absolute diff = " << max_diff << ")\n";
        return false;
    }
    return true;
}

} // namespace cutin

#endif // CUTIN_UTILS_HPP

