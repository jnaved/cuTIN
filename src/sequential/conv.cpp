#include "network.hpp"
#include <iostream>

namespace cutin {

// =========================================================================
// SKELETON REFERENCE FOR SEQUENTIAL IMPLEMENTATION
// Teammate note: Implement the sequential CPU loops below. Do not use OpenMP or CUDA here.
// =========================================================================

void seq_relu_forward(const float* x, float* y, size_t size) {
    // TODO: [Teammate - Sequential] Implement element-wise ReLU: y = max(0, x)
    (void)x; (void)y; (void)size;
}

void seq_relu_backward(const float* dy, const float* y, float* dx, size_t size) {
    // TODO: [Teammate - Sequential] Implement element-wise ReLU backward: dx = (y > 0) ? dy : 0
    (void)dy; (void)y; (void)dx; (void)size;
}

void seq_maxpool2d_forward(const float* x, float* y, int* mask,
                           int B, int C, int H_in, int W_in, int K, int stride) {
    // TODO: [Teammate - Sequential] Implement 2D Max Pooling forward and store winning indices in mask
    (void)x; (void)y; (void)mask;
    (void)B; (void)C; (void)H_in; (void)W_in; (void)K; (void)stride;
}

void seq_maxpool2d_backward(const float* dy, const int* mask, float* dx,
                            int B, int C, int H_out, int W_out, int H_in, int W_in, int K, int stride) {
    // TODO: [Teammate - Sequential] Implement 2D Max Pooling backward routing dy to winning indices in dx
    (void)dy; (void)mask; (void)dx;
    (void)B; (void)C; (void)H_out; (void)W_out; (void)H_in; (void)W_in; (void)K; (void)stride;
}

void seq_conv2d_forward(const float* x, const float* w, const float* b, float* y,
                        int B, int C_in, int H_in, int W_in,
                        int C_out, int K, int pad, int stride) {
    // TODO: [Teammate - Sequential] Implement 2D Convolution forward:
    // Output shape: (B, C_out, H_out, W_out) where H_out = (H_in + 2*pad - K)/stride + 1
    (void)x; (void)w; (void)b; (void)y;
    (void)B; (void)C_in; (void)H_in; (void)W_in; (void)C_out; (void)K; (void)pad; (void)stride;
}

void seq_conv2d_backward(const float* dy, const float* x, const float* w,
                         float* dx, float* dw, float* db,
                         int B, int C_in, int H_in, int W_in,
                         int C_out, int K, int pad, int stride) {
    // TODO: [Teammate - Sequential] Implement 2D Convolution backward:
    // Compute gradient w.r.t input (dx), weights (dw), and biases (db)
    (void)dy; (void)x; (void)w; (void)dx; (void)dw; (void)db;
    (void)B; (void)C_in; (void)H_in; (void)W_in; (void)C_out; (void)K; (void)pad; (void)stride;
}

} // namespace cutin

