#include "network.hpp"
#include "utils.hpp"
#include <cuda_runtime.h>
#include <cfloat>

namespace cutin {

// =========================================================================
// ReLU Forward & Backward
// =========================================================================
__global__ void k_relu_forward(const float* __restrict__ x, float* __restrict__ y, size_t size) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        y[idx] = val > 0.0f ? val : 0.0f;
    }
}

__global__ void k_relu_backward(const float* __restrict__ dy, const float* __restrict__ y,
                                float* __restrict__ dx, size_t size) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        dx[idx] = (y[idx] > 0.0f) ? dy[idx] : 0.0f;
    }
}

void cuda_relu_forward(const float* d_x, float* d_y, size_t size) {
    int block = 256;
    int grid = static_cast<int>((size + block - 1) / block);
    k_relu_forward<<<grid, block>>>(d_x, d_y, size);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_relu_backward(const float* d_dy, const float* d_y, float* d_dx, size_t size) {
    int block = 256;
    int grid = static_cast<int>((size + block - 1) / block);
    k_relu_backward<<<grid, block>>>(d_dy, d_y, d_dx, size);
    CUDA_CHECK(cudaGetLastError());
}

// =========================================================================
// MaxPool2D Forward & Backward (2x2, stride=2)
// =========================================================================
__global__ void k_maxpool2d_forward(const float* __restrict__ x,
                                    float* __restrict__ y,
                                    int* __restrict__ mask,
                                    int B, int C, int H_in, int W_in,
                                    int H_out, int W_out,
                                    int K, int stride) {
    int total = B * C * H_out * W_out;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp = tmp / H_out;
    int c = tmp % C;
    int b = tmp / C;

    int h_start = h_out * stride;
    int w_start = w_out * stride;

    float max_val = -FLT_MAX;
    int max_idx = -1;

    int in_batch_channel_offset = (b * C + c) * H_in * W_in;

    for (int kh = 0; kh < K; ++kh) {
        int h = h_start + kh;
        if (h < H_in) {
            for (int kw = 0; kw < K; ++kw) {
                int w = w_start + kw;
                if (w < W_in) {
                    int in_idx = in_batch_channel_offset + h * W_in + w;
                    float val = x[in_idx];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = in_idx;
                    }
                }
            }
        }
    }

    y[idx] = max_val;
    mask[idx] = max_idx;
}

__global__ void k_maxpool2d_backward(const float* __restrict__ dy,
                                     const int* __restrict__ mask,
                                     float* __restrict__ dx,
                                     int total_out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total_out) return;

    int target_idx = mask[idx];
    if (target_idx >= 0) {
        atomicAdd(&dx[target_idx], dy[idx]);
    }
}

void cuda_maxpool2d_forward(const float* d_x, float* d_y, int* d_mask,
                            int B, int C, int H_in, int W_in, int K, int stride) {
    int H_out = (H_in - K) / stride + 1;
    int W_out = (W_in - K) / stride + 1;
    int total = B * C * H_out * W_out;

    int block = 256;
    int grid = (total + block - 1) / block;
    k_maxpool2d_forward<<<grid, block>>>(d_x, d_y, d_mask, B, C, H_in, W_in, H_out, W_out, K, stride);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_maxpool2d_backward(const float* d_dy, const int* d_mask, float* d_dx,
                             int B, int C, int H_out, int W_out, int H_in, int W_in, int K, int stride) {
    int total_in = B * C * H_in * W_in;
    CUDA_CHECK(cudaMemset(d_dx, 0, total_in * sizeof(float)));

    int total_out = B * C * H_out * W_out;
    int block = 256;
    int grid = (total_out + block - 1) / block;
    k_maxpool2d_backward<<<grid, block>>>(d_dy, d_mask, d_dx, total_out);
    CUDA_CHECK(cudaGetLastError());
}

// =========================================================================
// Conv2D Forward
// =========================================================================
__global__ void k_conv2d_forward(const float* __restrict__ x,
                                 const float* __restrict__ w,
                                 const float* __restrict__ b,
                                 float* __restrict__ y,
                                 int B, int C_in, int H_in, int W_in,
                                 int C_out, int H_out, int W_out,
                                 int K, int pad, int stride) {
    int total = B * C_out * H_out * W_out;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp = tmp / H_out;
    int c_out = tmp % C_out;
    int b_idx = tmp / C_out;

    float sum = (b != nullptr) ? b[c_out] : 0.0f;

    int h_in_base = h_out * stride - pad;
    int w_in_base = w_out * stride - pad;

    for (int c_in = 0; c_in < C_in; ++c_in) {
        int x_channel_offset = ((b_idx * C_in + c_in) * H_in);
        int w_channel_offset = ((c_out * C_in + c_in) * K);

        for (int kh = 0; kh < K; ++kh) {
            int h_in = h_in_base + kh;
            if (h_in >= 0 && h_in < H_in) {
                int x_row_offset = (x_channel_offset + h_in) * W_in;
                int w_row_offset = (w_channel_offset + kh) * K;

                for (int kw = 0; kw < K; ++kw) {
                    int w_in = w_in_base + kw;
                    if (w_in >= 0 && w_in < W_in) {
                        sum += x[x_row_offset + w_in] * w[w_row_offset + kw];
                    }
                }
            }
        }
    }

    y[idx] = sum;
}

void cuda_conv2d_forward(const float* d_x, const float* d_w, const float* d_b, float* d_y,
                         int B, int C_in, int H_in, int W_in,
                         int C_out, int K, int pad, int stride) {
    int H_out = (H_in + 2 * pad - K) / stride + 1;
    int W_out = (W_in + 2 * pad - K) / stride + 1;
    int total = B * C_out * H_out * W_out;

    int block = 256;
    int grid = (total + block - 1) / block;
    k_conv2d_forward<<<grid, block>>>(d_x, d_w, d_b, d_y,
                                      B, C_in, H_in, W_in,
                                      C_out, H_out, W_out,
                                      K, pad, stride);
    CUDA_CHECK(cudaGetLastError());
}

// =========================================================================
// Conv2D Backward
// =========================================================================
__global__ void k_conv2d_backward_input(const float* __restrict__ dy,
                                        const float* __restrict__ w,
                                        float* __restrict__ dx,
                                        int B, int C_in, int H_in, int W_in,
                                        int C_out, int H_out, int W_out,
                                        int K, int pad, int stride) {
    int total = B * C_in * H_in * W_in;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;

    int w_in = idx % W_in;
    int tmp = idx / W_in;
    int h_in = tmp % H_in;
    tmp = tmp / H_in;
    int c_in = tmp % C_in;
    int b_idx = tmp / C_in;

    float grad = 0.0f;

    for (int c_out = 0; c_out < C_out; ++c_out) {
        int dy_channel_offset = ((b_idx * C_out + c_out) * H_out);
        int w_channel_offset = ((c_out * C_in + c_in) * K);

        for (int kh = 0; kh < K; ++kh) {
            int h_out_coord = h_in + pad - kh;
            if (h_out_coord % stride == 0) {
                int h_out = h_out_coord / stride;
                if (h_out >= 0 && h_out < H_out) {
                    int dy_row_offset = (dy_channel_offset + h_out) * W_out;
                    int w_row_offset = (w_channel_offset + kh) * K;

                    for (int kw = 0; kw < K; ++kw) {
                        int w_out_coord = w_in + pad - kw;
                        if (w_out_coord % stride == 0) {
                            int w_out = w_out_coord / stride;
                            if (w_out >= 0 && w_out < W_out) {
                                grad += dy[dy_row_offset + w_out] * w[w_row_offset + kw];
                            }
                        }
                    }
                }
            }
        }
    }

    dx[idx] = grad;
}

__global__ void k_conv2d_backward_weights(const float* __restrict__ dy,
                                          const float* __restrict__ x,
                                          float* __restrict__ dw,
                                          int B, int C_in, int H_in, int W_in,
                                          int C_out, int H_out, int W_out,
                                          int K, int pad, int stride) {
    int total = C_out * C_in * K * K;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;

    int kw = idx % K;
    int tmp = idx / K;
    int kh = tmp % K;
    tmp = tmp / K;
    int c_in = tmp % C_in;
    int c_out = tmp / C_in;

    float grad = 0.0f;

    for (int b = 0; b < B; ++b) {
        int dy_channel_offset = ((b * C_out + c_out) * H_out);
        int x_channel_offset = ((b * C_in + c_in) * H_in);

        for (int h_out = 0; h_out < H_out; ++h_out) {
            int h_in = h_out * stride - pad + kh;
            if (h_in >= 0 && h_in < H_in) {
                int dy_row_offset = (dy_channel_offset + h_out) * W_out;
                int x_row_offset = (x_channel_offset + h_in) * W_in;

                for (int w_out = 0; w_out < W_out; ++w_out) {
                    int w_in = w_out * stride - pad + kw;
                    if (w_in >= 0 && w_in < W_in) {
                        grad += dy[dy_row_offset + w_out] * x[x_row_offset + w_in];
                    }
                }
            }
        }
    }

    dw[idx] = grad;
}

__global__ void k_conv2d_backward_bias(const float* __restrict__ dy,
                                       float* __restrict__ db,
                                       int B, int C_out, int H_out, int W_out) {
    int c_out = blockDim.x * blockIdx.x + threadIdx.x;
    if (c_out >= C_out) return;

    float sum = 0.0f;
    for (int b = 0; b < B; ++b) {
        int offset = ((b * C_out + c_out) * H_out);
        for (int h = 0; h < H_out; ++h) {
            int row = (offset + h) * W_out;
            for (int w = 0; w < W_out; ++w) {
                sum += dy[row + w];
            }
        }
    }
    db[c_out] = sum;
}

void cuda_conv2d_backward(const float* d_dy, const float* d_x, const float* d_w,
                          float* d_dx, float* d_dw, float* d_db,
                          int B, int C_in, int H_in, int W_in,
                          int C_out, int K, int pad, int stride) {
    int H_out = (H_in + 2 * pad - K) / stride + 1;
    int W_out = (W_in + 2 * pad - K) / stride + 1;

    // 1. d_dx
    if (d_dx != nullptr) {
        int total_dx = B * C_in * H_in * W_in;
        int block = 256;
        int grid = (total_dx + block - 1) / block;
        k_conv2d_backward_input<<<grid, block>>>(d_dy, d_w, d_dx,
                                                 B, C_in, H_in, W_in,
                                                 C_out, H_out, W_out,
                                                 K, pad, stride);
    }

    // 2. d_dw
    if (d_dw != nullptr) {
        int total_dw = C_out * C_in * K * K;
        int block = 256;
        int grid = (total_dw + block - 1) / block;
        k_conv2d_backward_weights<<<grid, block>>>(d_dy, d_x, d_dw,
                                                   B, C_in, H_in, W_in,
                                                   C_out, H_out, W_out,
                                                   K, pad, stride);
    }

    // 3. d_db
    if (d_db != nullptr) {
        int block = 64;
        int grid = (C_out + block - 1) / block;
        k_conv2d_backward_bias<<<grid, block>>>(d_dy, d_db, B, C_out, H_out, W_out);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace cutin

