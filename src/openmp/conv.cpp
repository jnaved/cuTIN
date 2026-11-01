#include "network.hpp"
#include <iostream>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace cutin {

// =========================================================================
// SKELETON REFERENCE FOR OPENMP IMPLEMENTATION
// Teammate note: Parallelize the CPU loops below using OpenMP (#pragma omp parallel for).
// =========================================================================

void omp_relu_forward(const float* x, float* y, size_t size) {
    // TODO: [Teammate - OpenMP] Parallelize element-wise ReLU: y = max(0, x) using #pragma omp parallel for
    (void)x; (void)y; (void)size;
}

void omp_relu_backward(const float* dy, const float* y, float* dx, size_t size) {
    // TODO: [Teammate - OpenMP] Parallelize ReLU backward: dx = (y > 0) ? dy : 0 using #pragma omp parallel for
    (void)dy; (void)y; (void)dx; (void)size;
}

void omp_maxpool2d_forward(const float* x, float* y, int* mask,
                           int B, int C, int H_in, int W_in, int K, int stride) {
    // TODO: [Teammate - OpenMP] Parallelize 2D Max Pooling forward using #pragma omp parallel for
    (void)x; (void)y; (void)mask;
    (void)B; (void)C; (void)H_in; (void)W_in; (void)K; (void)stride;
}

void omp_maxpool2d_backward(const float* dy, const int* mask, float* dx,
                            int B, int C, int H_out, int W_out, int H_in, int W_in, int K, int stride) {
    // TODO: [Teammate - OpenMP] Parallelize 2D Max Pooling backward using #pragma omp parallel for
    (void)dy; (void)mask; (void)dx;
    (void)B; (void)C; (void)H_out; (void)W_out; (void)H_in; (void)W_in; (void)K; (void)stride;
}

void omp_conv2d_forward(const float* x, const float* w, const float* b, float* y,
                        int B, int C_in, int H_in, int W_in,
                        int C_out, int K, int pad, int stride) {
    // TODO: [Teammate - OpenMP] Parallelize 2D Convolution forward:
    // Parallelize over batches B and output channels C_out
    (void)x; (void)w; (void)b; (void)y;
    (void)B; (void)C_in; (void)H_in; (void)W_in; (void)C_out; (void)K; (void)pad; (void)stride;
}

void omp_conv2d_backward(const float* dy, const float* x, const float* w,
                         float* dx, float* dw, float* db,
                         int B, int C_in, int H_in, int W_in,
                         int C_out, int K, int pad, int stride) {
    // TODO: [Teammate - OpenMP] Parallelize 2D Convolution backward:
    // Compute gradient w.r.t input (dx), weights (dw), and biases (db) using OpenMP
    (void)dy; (void)x; (void)w; (void)dx; (void)dw; (void)db;
    (void)B; (void)C_in; (void)H_in; (void)W_in; (void)C_out; (void)K; (void)pad; (void)stride;
}

} // namespace cutin

