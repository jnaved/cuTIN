#include "network.hpp"
#include "utils.hpp"
#include <cuda_runtime.h>
#include <cfloat>

namespace cutin {

// =========================================================================
// Linear / Dense Forward
// =========================================================================
__global__ void k_linear_forward(const float* __restrict__ x,
                                 const float* __restrict__ w,
                                 const float* __restrict__ b,
                                 float* __restrict__ y,
                                 int B, int in_features, int out_features) {
    int total = B * out_features;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;

    int out_idx = idx % out_features;
    int b_idx = idx / out_features;

    float sum = (b != nullptr) ? b[out_idx] : 0.0f;

    const float* x_row = x + b_idx * in_features;
    const float* w_row = w + out_idx * in_features;

    for (int k = 0; k < in_features; ++k) {
        sum += x_row[k] * w_row[k];
    }

    y[idx] = sum;
}

void cuda_linear_forward(const float* d_x, const float* d_w, const float* d_b, float* d_y,
                         int B, int in_features, int out_features) {
    int total = B * out_features;
    int block = 256;
    int grid = (total + block - 1) / block;
    k_linear_forward<<<grid, block>>>(d_x, d_w, d_b, d_y, B, in_features, out_features);
    CUDA_CHECK(cudaGetLastError());
}

// =========================================================================
// Linear / Dense Backward
// =========================================================================
__global__ void k_linear_backward_input(const float* __restrict__ dy,
                                        const float* __restrict__ w,
                                        float* __restrict__ dx,
                                        int B, int in_features, int out_features) {
    int total = B * in_features;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;

    int in_idx = idx % in_features;
    int b_idx = idx / in_features;

    float sum = 0.0f;
    const float* dy_row = dy + b_idx * out_features;

    for (int j = 0; j < out_features; ++j) {
        sum += dy_row[j] * w[j * in_features + in_idx];
    }

    dx[idx] = sum;
}

__global__ void k_linear_backward_weights(const float* __restrict__ dy,
                                          const float* __restrict__ x,
                                          float* __restrict__ dw,
                                          int B, int in_features, int out_features) {
    int total = out_features * in_features;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;

    int in_idx = idx % in_features;
    int out_idx = idx / in_features;

    float sum = 0.0f;
    for (int b = 0; b < B; ++b) {
        sum += dy[b * out_features + out_idx] * x[b * in_features + in_idx];
    }

    dw[idx] = sum;
}

__global__ void k_linear_backward_bias(const float* __restrict__ dy,
                                       float* __restrict__ db,
                                       int B, int out_features) {
    int out_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (out_idx >= out_features) return;

    float sum = 0.0f;
    for (int b = 0; b < B; ++b) {
        sum += dy[b * out_features + out_idx];
    }

    db[out_idx] = sum;
}

void cuda_linear_backward(const float* d_dy, const float* d_x, const float* d_w,
                          float* d_dx, float* d_dw, float* d_db,
                          int B, int in_features, int out_features) {
    if (d_dx != nullptr) {
        int total_dx = B * in_features;
        int block = 256;
        int grid = (total_dx + block - 1) / block;
        k_linear_backward_input<<<grid, block>>>(d_dy, d_w, d_dx, B, in_features, out_features);
    }

    if (d_dw != nullptr) {
        int total_dw = out_features * in_features;
        int block = 256;
        int grid = (total_dw + block - 1) / block;
        k_linear_backward_weights<<<grid, block>>>(d_dy, d_x, d_dw, B, in_features, out_features);
    }

    if (d_db != nullptr) {
        int block = 128;
        int grid = (out_features + block - 1) / block;
        k_linear_backward_bias<<<grid, block>>>(d_dy, d_db, B, out_features);
    }

    CUDA_CHECK(cudaGetLastError());
}

// =========================================================================
// Softmax + Categorical Cross-Entropy (Forward & Backward)
// =========================================================================
__global__ void k_softmax_cross_entropy(const float* __restrict__ logits,
                                        const int* __restrict__ labels,
                                        float* __restrict__ probs,
                                        float* __restrict__ dlogits,
                                        float* __restrict__ batch_loss,
                                        int B, int num_classes) {
    int b = blockDim.x * blockIdx.x + threadIdx.x;
    if (b >= B) return;

    const float* row = logits + b * num_classes;
    float* p_row = probs + b * num_classes;
    float* dl_row = dlogits + b * num_classes;
    int target = labels[b];

    // Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int c = 0; c < num_classes; ++c) {
        if (row[c] > max_val) max_val = row[c];
    }

    // Compute exp sum
    float sum_exp = 0.0f;
    for (int c = 0; c < num_classes; ++c) {
        float val = expf(row[c] - max_val);
        p_row[c] = val;
        sum_exp += val;
    }

    float inv_sum = 1.0f / (sum_exp + 1e-12f);
    float inv_B = 1.0f / static_cast<float>(B);

    // Compute probabilities and gradient (p - 1{y == c}) / B
    for (int c = 0; c < num_classes; ++c) {
        float p = p_row[c] * inv_sum;
        p_row[c] = p;

        float target_indicator = (c == target) ? 1.0f : 0.0f;
        dl_row[c] = (p - target_indicator) * inv_B;
    }

    // Negative log likelihood loss
    if (batch_loss != nullptr) {
        if (target >= 0 && target < num_classes) {
            float p_target = fmaxf(p_row[target], 1e-12f);
            batch_loss[b] = -logf(p_target);
        } else {
            batch_loss[b] = 0.0f;
        }
    }
}

void cuda_softmax_cross_entropy_forward_backward(const float* d_logits, const int* d_labels,
                                                 float* d_probs, float* d_dlogits,
                                                 float* d_batch_loss, int B, int num_classes) {
    int block = 64;
    int grid = (B + block - 1) / block;
    k_softmax_cross_entropy<<<grid, block>>>(d_logits, d_labels, d_probs, d_dlogits,
                                             d_batch_loss, B, num_classes);
    CUDA_CHECK(cudaGetLastError());
}

// =========================================================================
// SGD with Momentum & Weight Decay Update
// =========================================================================
__global__ void k_sgd_update(float* __restrict__ param,
                             const float* __restrict__ grad,
                             float* __restrict__ velocity,
                             size_t size, float lr, float momentum, float weight_decay) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;

    float p = param[idx];
    float g = grad[idx];

    if (weight_decay != 0.0f) {
        g += weight_decay * p;
    }

    float v = velocity[idx];
    v = momentum * v + g;
    velocity[idx] = v;

    param[idx] = p - lr * v;
}

void cuda_sgd_update(float* d_w, float* d_dw, float* d_vw,
                     float* d_b, float* d_db, float* d_vb,
                     size_t w_size, size_t b_size,
                     float lr, float momentum, float weight_decay) {
    int block = 256;
    if (w_size > 0 && d_w != nullptr) {
        int grid = static_cast<int>((w_size + block - 1) / block);
        k_sgd_update<<<grid, block>>>(d_w, d_dw, d_vw, w_size, lr, momentum, weight_decay);
    }
    if (b_size > 0 && d_b != nullptr) {
        int grid = static_cast<int>((b_size + block - 1) / block);
        // Biases typically do not use weight decay
        k_sgd_update<<<grid, block>>>(d_b, d_db, d_vb, b_size, lr, momentum, 0.0f);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cutin

