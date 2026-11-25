#include "network.hpp"
#include <cuda_runtime.h>
#include <cfloat>

namespace cutin {

// =========================================================================
// CUDA GPU ACCELERATED BACKEND (Member 3)
// High-performance massively parallel kernels executed on GPU threads.
// =========================================================================

// --- 1. ReLU Kernels ---
__global__ void k_relu_forward(const float* __restrict__ x, float* __restrict__ y, size_t size) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) y[idx] = (x[idx] > 0.0f) ? x[idx] : 0.0f;
}

__global__ void k_relu_backward(const float* __restrict__ dy, const float* __restrict__ y, float* __restrict__ dx, size_t size) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) dx[idx] = (y[idx] > 0.0f) ? dy[idx] : 0.0f;
}

void cuda_relu_forward(const float* d_x, float* d_y, size_t size) {
    int b = 256; int g = (size + b - 1) / b;
    k_relu_forward<<<g, b>>>(d_x, d_y, size);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_relu_backward(const float* d_dy, const float* d_y, float* d_dx, size_t size) {
    int b = 256; int g = (size + b - 1) / b;
    k_relu_backward<<<g, b>>>(d_dy, d_y, d_dx, size);
    CUDA_CHECK(cudaGetLastError());
}

// --- 2. MaxPool2D Kernels ---
__global__ void k_maxpool2d_forward(const float* __restrict__ x, float* __restrict__ y, int* __restrict__ mask,
                                    int B, int C, int H_in, int W_in, int H_out, int W_out, int K, int stride) {
    int total = B * C * H_out * W_out;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;

    int w_out = idx % W_out; int tmp = idx / W_out;
    int h_out = tmp % H_out; tmp /= H_out;
    int c = tmp % C; int b = tmp / C;
    int in_off = (b * C + c) * H_in * W_in;

    float max_v = -FLT_MAX; int max_i = -1;
    for (int kh = 0; kh < K; ++kh) {
        int h = h_out * stride + kh;
        if (h < H_in) {
            for (int kw = 0; kw < K; ++kw) {
                int w = w_out * stride + kw;
                if (w < W_in) {
                    int in_idx = in_off + h * W_in + w;
                    float v = x[in_idx];
                    if (v > max_v) { max_v = v; max_i = in_idx; }
                }
            }
        }
    }
    y[idx] = max_v; mask[idx] = max_i;
}

__global__ void k_maxpool2d_backward(const float* __restrict__ dy, const int* __restrict__ mask, float* __restrict__ dx, int total_out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total_out) {
        int target = mask[idx];
        if (target >= 0) atomicAdd(&dx[target], dy[idx]);
    }
}

void cuda_maxpool2d_forward(const float* d_x, float* d_y, int* d_mask, int B, int C, int H_in, int W_in, int K, int stride) {
    int H_out = (H_in - K) / stride + 1, W_out = (W_in - K) / stride + 1, total = B * C * H_out * W_out;
    int b = 256; int g = (total + b - 1) / b;
    k_maxpool2d_forward<<<g, b>>>(d_x, d_y, d_mask, B, C, H_in, W_in, H_out, W_out, K, stride);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_maxpool2d_backward(const float* d_dy, const int* d_mask, float* d_dx,
                             int B, int C, int H_out, int W_out, int H_in, int W_in, int K, int stride) {
    (void)K; (void)stride;
    int total_in = B * C * H_in * W_in;
    CUDA_CHECK(cudaMemset(d_dx, 0, total_in * sizeof(float)));
    int total_out = B * C * H_out * W_out;
    int b = 256; int g = (total_out + b - 1) / b;
    k_maxpool2d_backward<<<g, b>>>(d_dy, d_mask, d_dx, total_out);
    CUDA_CHECK(cudaGetLastError());
}

// --- 3. Conv2D Kernels ---
__global__ void k_conv2d_forward(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, float* __restrict__ y,
                                 int B, int C_in, int H_in, int W_in, int C_out, int H_out, int W_out, int K, int pad, int stride) {
    int total = B * C_out * H_out * W_out;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;

    int w_out = idx % W_out; int tmp = idx / W_out;
    int h_out = tmp % H_out; tmp /= H_out;
    int c_out = tmp % C_out; int b_idx = tmp / C_out;

    float sum = b ? b[c_out] : 0.0f;
    int h_base = h_out * stride - pad; int w_base = w_out * stride - pad;

    for (int ci = 0; ci < C_in; ++ci) {
        int x_ch = (b_idx * C_in + ci) * H_in;
        int w_ch = (c_out * C_in + ci) * K;
        for (int kh = 0; kh < K; ++kh) {
            int hi = h_base + kh;
            if (hi >= 0 && hi < H_in) {
                for (int kw = 0; kw < K; ++kw) {
                    int wi = w_base + kw;
                    if (wi >= 0 && wi < W_in) sum += x[(x_ch + hi) * W_in + wi] * w[(w_ch + kh) * K + kw];
                }
            }
        }
    }
    y[idx] = sum;
}

__global__ void k_conv2d_backward_input(const float* __restrict__ dy, const float* __restrict__ w, float* __restrict__ dx,
                                        int B, int C_in, int H_in, int W_in, int C_out, int H_out, int W_out, int K, int pad, int stride) {
    int total = B * C_in * H_in * W_in;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;

    int wi = idx % W_in; int tmp = idx / W_in;
    int hi = tmp % H_in; tmp /= H_in;
    int ci = tmp % C_in; int b_idx = tmp / C_in;

    float grad = 0.0f;
    for (int co = 0; co < C_out; ++co) {
        int dy_ch = (b_idx * C_out + co) * H_out;
        int w_ch = (co * C_in + ci) * K;
        for (int kh = 0; kh < K; ++kh) {
            int hc = hi + pad - kh;
            if (hc % stride == 0) {
                int ho = hc / stride;
                if (ho >= 0 && ho < H_out) {
                    for (int kw = 0; kw < K; ++kw) {
                        int wc = wi + pad - kw;
                        if (wc % stride == 0) {
                            int wo = wc / stride;
                            if (wo >= 0 && wo < W_out) grad += dy[(dy_ch + ho) * W_out + wo] * w[(w_ch + kh) * K + kw];
                        }
                    }
                }
            }
        }
    }
    dx[idx] = grad;
}

__global__ void k_conv2d_backward_weights(const float* __restrict__ dy, const float* __restrict__ x, float* __restrict__ dw,
                                          int B, int C_in, int H_in, int W_in, int C_out, int H_out, int W_out, int K, int pad, int stride) {
    int total = C_out * C_in * K * K;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;

    int kw = idx % K; int tmp = idx / K;
    int kh = tmp % K; tmp /= K;
    int ci = tmp % C_in; int co = tmp / C_in;

    float grad = 0.0f;
    for (int b = 0; b < B; ++b) {
        int dy_ch = (b * C_out + co) * H_out;
        int x_ch = (b * C_in + ci) * H_in;
        for (int ho = 0; ho < H_out; ++ho) {
            int hi = ho * stride - pad + kh;
            if (hi >= 0 && hi < H_in) {
                for (int wo = 0; wo < W_out; ++wo) {
                    int wi = wo * stride - pad + kw;
                    if (wi >= 0 && wi < W_in) grad += dy[(dy_ch + ho) * W_out + wo] * x[(x_ch + hi) * W_in + wi];
                }
            }
        }
    }
    dw[idx] = grad;
}

__global__ void k_conv2d_backward_bias(const float* __restrict__ dy, float* __restrict__ db, int B, int C_out, int H_out, int W_out) {
    int co = blockDim.x * blockIdx.x + threadIdx.x;
    if (co >= C_out) return;
    float sum = 0.0f;
    for (int b = 0; b < B; ++b) {
        int off = (b * C_out + co) * H_out;
        for (int h = 0; h < H_out; ++h) {
            int row = (off + h) * W_out;
            for (int w = 0; w < W_out; ++w) sum += dy[row + w];
        }
    }
    db[co] = sum;
}

void cuda_conv2d_forward(const float* d_x, const float* d_w, const float* d_b, float* d_y,
                         int B, int C_in, int H_in, int W_in, int C_out, int K, int pad, int stride) {
    int H_out = (H_in + 2 * pad - K) / stride + 1, W_out = (W_in + 2 * pad - K) / stride + 1, total = B * C_out * H_out * W_out;
    int b = 256; int g = (total + b - 1) / b;
    k_conv2d_forward<<<g, b>>>(d_x, d_w, d_b, d_y, B, C_in, H_in, W_in, C_out, H_out, W_out, K, pad, stride);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_conv2d_backward(const float* d_dy, const float* d_x, const float* d_w, float* d_dx, float* d_dw, float* d_db,
                          int B, int C_in, int H_in, int W_in, int C_out, int K, int pad, int stride) {
    int H_out = (H_in + 2 * pad - K) / stride + 1, W_out = (W_in + 2 * pad - K) / stride + 1;
    if (d_dx) { int tot = B * C_in * H_in * W_in; int b = 256; k_conv2d_backward_input<<<(tot+b-1)/b, b>>>(d_dy, d_w, d_dx, B, C_in, H_in, W_in, C_out, H_out, W_out, K, pad, stride); }
    if (d_dw) { int tot = C_out * C_in * K * K; int b = 256; k_conv2d_backward_weights<<<(tot+b-1)/b, b>>>(d_dy, d_x, d_dw, B, C_in, H_in, W_in, C_out, H_out, W_out, K, pad, stride); }
    if (d_db) { int b = 64; k_conv2d_backward_bias<<<(C_out+b-1)/b, b>>>(d_dy, d_db, B, C_out, H_out, W_out); }
    CUDA_CHECK(cudaGetLastError());
}

// --- 4. Linear / Dense Kernels ---
__global__ void k_linear_forward(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, float* __restrict__ y,
                                 int B, int in_f, int out_f) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= B * out_f) return;
    int out = idx % out_f; int b_idx = idx / out_f;
    float sum = b ? b[out] : 0.0f;
    const float* xr = x + b_idx * in_f; const float* wr = w + out * in_f;
    for (int k = 0; k < in_f; ++k) sum += xr[k] * wr[k];
    y[idx] = sum;
}

__global__ void k_linear_backward_input(const float* __restrict__ dy, const float* __restrict__ w, float* __restrict__ dx, int B, int in_f, int out_f) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= B * in_f) return;
    int in = idx % in_f; int b_idx = idx / in_f;
    float sum = 0.0f; const float* dyr = dy + b_idx * out_f;
    for (int j = 0; j < out_f; ++j) sum += dyr[j] * w[j * in_f + in];
    dx[idx] = sum;
}

__global__ void k_linear_backward_weights(const float* __restrict__ dy, const float* __restrict__ x, float* __restrict__ dw, int B, int in_f, int out_f) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= out_f * in_f) return;
    int in = idx % in_f; int out = idx / in_f;
    float sum = 0.0f;
    for (int b = 0; b < B; ++b) sum += dy[b * out_f + out] * x[b * in_f + in];
    dw[idx] = sum;
}

__global__ void k_linear_backward_bias(const float* __restrict__ dy, float* __restrict__ db, int B, int out_f) {
    int out = blockDim.x * blockIdx.x + threadIdx.x;
    if (out >= out_f) return;
    float sum = 0.0f;
    for (int b = 0; b < B; ++b) sum += dy[b * out_f + out];
    db[out] = sum;
}

void cuda_linear_forward(const float* d_x, const float* d_w, const float* d_b, float* d_y, int B, int in_f, int out_f) {
    int tot = B * out_f; int b = 256; k_linear_forward<<<(tot+b-1)/b, b>>>(d_x, d_w, d_b, d_y, B, in_f, out_f);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_linear_backward(const float* d_dy, const float* d_x, const float* d_w, float* d_dx, float* d_dw, float* d_db, int B, int in_f, int out_f) {
    if (d_dx) { int tot = B * in_f; int b = 256; k_linear_backward_input<<<(tot+b-1)/b, b>>>(d_dy, d_w, d_dx, B, in_f, out_f); }
    if (d_dw) { int tot = out_f * in_f; int b = 256; k_linear_backward_weights<<<(tot+b-1)/b, b>>>(d_dy, d_x, d_dw, B, in_f, out_f); }
    if (d_db) { int b = 128; k_linear_backward_bias<<<(out_f+b-1)/b, b>>>(d_dy, d_db, B, out_f); }
    CUDA_CHECK(cudaGetLastError());
}

// --- 5. Softmax + CrossEntropy & SGD ---
__global__ void k_softmax_ce(const float* __restrict__ logits, const int* __restrict__ labels,
                             float* __restrict__ probs, float* __restrict__ dlogits, float* __restrict__ batch_loss, int B, int num_classes) {
    int b = blockDim.x * blockIdx.x + threadIdx.x;
    if (b >= B) return;
    const float* row = logits + b * num_classes; float* p_row = probs + b * num_classes; float* dl = dlogits + b * num_classes;
    int target = labels[b];

    float max_v = -FLT_MAX;
    for (int c = 0; c < num_classes; ++c) if (row[c] > max_v) max_v = row[c];
    float sum_exp = 0.0f;
    for (int c = 0; c < num_classes; ++c) { float v = expf(row[c] - max_v); p_row[c] = v; sum_exp += v; }
    float inv_sum = 1.0f / (sum_exp + 1e-12f), inv_B = 1.0f / static_cast<float>(B);

    for (int c = 0; c < num_classes; ++c) {
        float p = p_row[c] * inv_sum; p_row[c] = p;
        dl[c] = (p - (c == target ? 1.0f : 0.0f)) * inv_B;
    }
    if (batch_loss) {
        batch_loss[b] = (target >= 0 && target < num_classes) ? -logf(fmaxf(p_row[target], 1e-12f)) : 0.0f;
    }
}

void cuda_softmax_cross_entropy_forward_backward(const float* d_logits, const int* d_labels, float* d_probs, float* d_dlogits, float* d_batch_loss, int B, int num_classes) {
    int b = 64; k_softmax_ce<<<(B+b-1)/b, b>>>(d_logits, d_labels, d_probs, d_dlogits, d_batch_loss, B, num_classes);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_sgd(float* __restrict__ p, const float* __restrict__ g, float* __restrict__ v, size_t sz, float lr, float mom, float wd) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= sz) return;
    float grad = g[idx] + (wd != 0.0f ? wd * p[idx] : 0.0f);
    float vel = mom * v[idx] + grad;
    v[idx] = vel; p[idx] -= lr * vel;
}

void cuda_sgd_update(float* d_w, float* d_dw, float* d_vw, float* d_b, float* d_db, float* d_vb, size_t w_sz, size_t b_sz, float lr, float mom, float wd) {
    int b = 256;
    if (w_sz && d_w) k_sgd<<<(w_sz+b-1)/b, b>>>(d_w, d_dw, d_vw, w_sz, lr, mom, wd);
    if (b_sz && d_b) k_sgd<<<(b_sz+b-1)/b, b>>>(d_b, d_db, d_vb, b_sz, lr, mom, 0.0f);
    CUDA_CHECK(cudaGetLastError());
}

// =========================================================================
// CUDA DEVICE BUFFERS & HIGH-LEVEL PIPELINE
// =========================================================================
namespace {
struct CudaBuffers {
    float *d_in, *d_c1, *d_p1, *d_c2, *d_p2, *d_fc1, *d_fc1_r, *d_fc2, *d_probs, *d_loss;
    int *d_labels, *d_p1_mask, *d_p2_mask;
    float *d_dl, *d_dfc1_r, *d_dfc1, *d_dp2, *d_dc2, *d_dp1, *d_dc1;

    explicit CudaBuffers(int B) {
        CUDA_CHECK(cudaMalloc(&d_in, B * ArchConfig::IN_C * ArchConfig::IN_H * ArchConfig::IN_W * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_c1, B * ArchConfig::C1_OUT_C * ArchConfig::C1_OUT_H * ArchConfig::C1_OUT_W * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_p1, B * ArchConfig::C1_OUT_C * ArchConfig::P1_OUT_H * ArchConfig::P1_OUT_W * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_p1_mask, B * ArchConfig::C1_OUT_C * ArchConfig::P1_OUT_H * ArchConfig::P1_OUT_W * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_c2, B * ArchConfig::C2_OUT_C * ArchConfig::C2_OUT_H * ArchConfig::C2_OUT_W * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_p2, B * ArchConfig::FLATTEN_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_p2_mask, B * ArchConfig::FLATTEN_SIZE * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_fc1, B * ArchConfig::FC1_OUT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_fc1_r, B * ArchConfig::FC1_OUT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_fc2, B * ArchConfig::FC2_OUT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_probs, B * ArchConfig::FC2_OUT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_loss, B * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_labels, B * sizeof(int)));

        CUDA_CHECK(cudaMalloc(&d_dl, B * ArchConfig::FC2_OUT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dfc1_r, B * ArchConfig::FC1_OUT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dfc1, B * ArchConfig::FC1_OUT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dp2, B * ArchConfig::FLATTEN_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dc2, B * ArchConfig::C2_OUT_C * ArchConfig::C2_OUT_H * ArchConfig::C2_OUT_W * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dp1, B * ArchConfig::C1_OUT_C * ArchConfig::P1_OUT_H * ArchConfig::P1_OUT_W * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dc1, B * ArchConfig::C1_OUT_C * ArchConfig::C1_OUT_H * ArchConfig::C1_OUT_W * sizeof(float)));
    }

    ~CudaBuffers() {
        cudaFree(d_in); cudaFree(d_c1); cudaFree(d_p1); cudaFree(d_p1_mask);
        cudaFree(d_c2); cudaFree(d_p2); cudaFree(d_p2_mask); cudaFree(d_fc1);
        cudaFree(d_fc1_r); cudaFree(d_fc2); cudaFree(d_probs); cudaFree(d_loss); cudaFree(d_labels);
        cudaFree(d_dl); cudaFree(d_dfc1_r); cudaFree(d_dfc1); cudaFree(d_dp2);
        cudaFree(d_dc2); cudaFree(d_dp1); cudaFree(d_dc1);
    }
};

void forward_batch(CudaBuffers& b, const ModelParameters& p, int B, bool compute_loss) {
    cuda_conv2d_forward(b.d_in, p.w_c1, p.b_c1, b.d_c1, B, ArchConfig::IN_C, ArchConfig::IN_H, ArchConfig::IN_W,
                        ArchConfig::C1_OUT_C, ArchConfig::C1_K, ArchConfig::C1_PAD, ArchConfig::C1_STRIDE);
    cuda_relu_forward(b.d_c1, b.d_c1, B * ArchConfig::C1_OUT_C * ArchConfig::C1_OUT_H * ArchConfig::C1_OUT_W);
    cuda_maxpool2d_forward(b.d_c1, b.d_p1, b.d_p1_mask, B, ArchConfig::C1_OUT_C, ArchConfig::C1_OUT_H, ArchConfig::C1_OUT_W, ArchConfig::P1_K, ArchConfig::P1_STRIDE);

    cuda_conv2d_forward(b.d_p1, p.w_c2, p.b_c2, b.d_c2, B, ArchConfig::C1_OUT_C, ArchConfig::P1_OUT_H, ArchConfig::P1_OUT_W,
                        ArchConfig::C2_OUT_C, ArchConfig::C2_K, ArchConfig::C2_PAD, ArchConfig::C2_STRIDE);
    cuda_relu_forward(b.d_c2, b.d_c2, B * ArchConfig::C2_OUT_C * ArchConfig::C2_OUT_H * ArchConfig::C2_OUT_W);
    cuda_maxpool2d_forward(b.d_c2, b.d_p2, b.d_p2_mask, B, ArchConfig::C2_OUT_C, ArchConfig::C2_OUT_H, ArchConfig::C2_OUT_W, ArchConfig::P2_K, ArchConfig::P2_STRIDE);

    cuda_linear_forward(b.d_p2, p.w_fc1, p.b_fc1, b.d_fc1, B, ArchConfig::FLATTEN_SIZE, ArchConfig::FC1_OUT);
    cuda_relu_forward(b.d_fc1, b.d_fc1_r, B * ArchConfig::FC1_OUT);
    cuda_linear_forward(b.d_fc1_r, p.w_fc2, p.b_fc2, b.d_fc2, B, ArchConfig::FC1_OUT, ArchConfig::FC2_OUT);

    cuda_softmax_cross_entropy_forward_backward(b.d_fc2, b.d_labels, b.d_probs, b.d_dl,
                                                compute_loss ? b.d_loss : nullptr, B, ArchConfig::FC2_OUT);
}

void backward_and_update(CudaBuffers& b, ModelParameters& p, int B, const TrainingConfig& cfg) {
    cuda_linear_backward(b.d_dl, b.d_fc1_r, p.w_fc2, b.d_dfc1_r, p.dw_fc2, p.db_fc2, B, ArchConfig::FC1_OUT, ArchConfig::FC2_OUT);
    cuda_relu_backward(b.d_dfc1_r, b.d_fc1_r, b.d_dfc1, B * ArchConfig::FC1_OUT);
    cuda_linear_backward(b.d_dfc1, b.d_p2, p.w_fc1, b.d_dp2, p.dw_fc1, p.db_fc1, B, ArchConfig::FLATTEN_SIZE, ArchConfig::FC1_OUT);

    cuda_maxpool2d_backward(b.d_dp2, b.d_p2_mask, b.d_dc2, B, ArchConfig::C2_OUT_C, ArchConfig::P2_OUT_H, ArchConfig::P2_OUT_W, ArchConfig::C2_OUT_H, ArchConfig::C2_OUT_W, ArchConfig::P2_K, ArchConfig::P2_STRIDE);
    cuda_relu_backward(b.d_dc2, b.d_c2, b.d_dc2, B * ArchConfig::C2_OUT_C * ArchConfig::C2_OUT_H * ArchConfig::C2_OUT_W);
    cuda_conv2d_backward(b.d_dc2, b.d_p1, p.w_c2, b.d_dp1, p.dw_c2, p.db_c2, B, ArchConfig::C1_OUT_C, ArchConfig::P1_OUT_H, ArchConfig::P1_OUT_W, ArchConfig::C2_OUT_C, ArchConfig::C2_K, ArchConfig::C2_PAD, ArchConfig::C2_STRIDE);

    cuda_maxpool2d_backward(b.d_dp1, b.d_p1_mask, b.d_dc1, B, ArchConfig::C1_OUT_C, ArchConfig::P1_OUT_H, ArchConfig::P1_OUT_W, ArchConfig::C1_OUT_H, ArchConfig::C1_OUT_W, ArchConfig::P1_K, ArchConfig::P1_STRIDE);
    cuda_relu_backward(b.d_dc1, b.d_c1, b.d_dc1, B * ArchConfig::C1_OUT_C * ArchConfig::C1_OUT_H * ArchConfig::C1_OUT_W);
    cuda_conv2d_backward(b.d_dc1, b.d_in, p.w_c1, nullptr, p.dw_c1, p.db_c1, B, ArchConfig::IN_C, ArchConfig::IN_H, ArchConfig::IN_W, ArchConfig::C1_OUT_C, ArchConfig::C1_K, ArchConfig::C1_PAD, ArchConfig::C1_STRIDE);

    cuda_sgd_update(p.w_c1, p.dw_c1, p.vw_c1, p.b_c1, p.db_c1, p.vb_c1, ArchConfig::C1_W_SIZE, ArchConfig::C1_B_SIZE, cfg.learning_rate, cfg.momentum, cfg.weight_decay);
    cuda_sgd_update(p.w_c2, p.dw_c2, p.vw_c2, p.b_c2, p.db_c2, p.vb_c2, ArchConfig::C2_W_SIZE, ArchConfig::C2_B_SIZE, cfg.learning_rate, cfg.momentum, cfg.weight_decay);
    cuda_sgd_update(p.w_fc1, p.dw_fc1, p.vw_fc1, p.b_fc1, p.db_fc1, p.vb_fc1, ArchConfig::FC1_W_SIZE, ArchConfig::FC1_B_SIZE, cfg.learning_rate, cfg.momentum, cfg.weight_decay);
    cuda_sgd_update(p.w_fc2, p.dw_fc2, p.vw_fc2, p.b_fc2, p.db_fc2, p.vb_fc2, ArchConfig::FC2_W_SIZE, ArchConfig::FC2_B_SIZE, cfg.learning_rate, cfg.momentum, cfg.weight_decay);
}

void alloc_gpu_params(ModelParameters& d, const ModelParameters& h) {
    auto cp = [](float*& dp, const float* hp, size_t sz) {
        CUDA_CHECK(cudaMalloc(&dp, sz * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dp, hp, sz * sizeof(float), cudaMemcpyHostToDevice));
    };
    auto zc = [](float*& dp, size_t sz) {
        CUDA_CHECK(cudaMalloc(&dp, sz * sizeof(float)));
        CUDA_CHECK(cudaMemset(dp, 0, sz * sizeof(float)));
    };
    cp(d.w_c1, h.w_c1, ArchConfig::C1_W_SIZE); cp(d.b_c1, h.b_c1, ArchConfig::C1_B_SIZE);
    zc(d.dw_c1, ArchConfig::C1_W_SIZE);       zc(d.db_c1, ArchConfig::C1_B_SIZE);
    zc(d.vw_c1, ArchConfig::C1_W_SIZE);       zc(d.vb_c1, ArchConfig::C1_B_SIZE);

    cp(d.w_c2, h.w_c2, ArchConfig::C2_W_SIZE); cp(d.b_c2, h.b_c2, ArchConfig::C2_B_SIZE);
    zc(d.dw_c2, ArchConfig::C2_W_SIZE);       zc(d.db_c2, ArchConfig::C2_B_SIZE);
    zc(d.vw_c2, ArchConfig::C2_W_SIZE);       zc(d.vb_c2, ArchConfig::C2_B_SIZE);

    cp(d.w_fc1, h.w_fc1, ArchConfig::FC1_W_SIZE); cp(d.b_fc1, h.b_fc1, ArchConfig::FC1_B_SIZE);
    zc(d.dw_fc1, ArchConfig::FC1_W_SIZE);         zc(d.db_fc1, ArchConfig::FC1_B_SIZE);
    zc(d.vw_fc1, ArchConfig::FC1_W_SIZE);         zc(d.vb_fc1, ArchConfig::FC1_B_SIZE);

    cp(d.w_fc2, h.w_fc2, ArchConfig::FC2_W_SIZE); cp(d.b_fc2, h.b_fc2, ArchConfig::FC2_B_SIZE);
    zc(d.dw_fc2, ArchConfig::FC2_W_SIZE);         zc(d.db_fc2, ArchConfig::FC2_B_SIZE);
    zc(d.vw_fc2, ArchConfig::FC2_W_SIZE);         zc(d.vb_fc2, ArchConfig::FC2_B_SIZE);
}

void free_gpu_params(ModelParameters& d, ModelParameters* h = nullptr) {
    auto dl = [](float* dp, float* hp, size_t sz) {
        if (hp && dp) {
            CUDA_CHECK(cudaMemcpy(hp, dp, sz * sizeof(float), cudaMemcpyDeviceToHost));
        }
        if (dp) cudaFree(dp);
    };
    dl(d.w_c1, h ? h->w_c1 : nullptr, ArchConfig::C1_W_SIZE);
    dl(d.b_c1, h ? h->b_c1 : nullptr, ArchConfig::C1_B_SIZE);
    cudaFree(d.dw_c1); cudaFree(d.db_c1); cudaFree(d.vw_c1); cudaFree(d.vb_c1);

    dl(d.w_c2, h ? h->w_c2 : nullptr, ArchConfig::C2_W_SIZE);
    dl(d.b_c2, h ? h->b_c2 : nullptr, ArchConfig::C2_B_SIZE);
    cudaFree(d.dw_c2); cudaFree(d.db_c2); cudaFree(d.vw_c2); cudaFree(d.vb_c2);

    dl(d.w_fc1, h ? h->w_fc1 : nullptr, ArchConfig::FC1_W_SIZE);
    dl(d.b_fc1, h ? h->b_fc1 : nullptr, ArchConfig::FC1_B_SIZE);
    cudaFree(d.dw_fc1); cudaFree(d.db_fc1); cudaFree(d.vw_fc1); cudaFree(d.vb_fc1);

    dl(d.w_fc2, h ? h->w_fc2 : nullptr, ArchConfig::FC2_W_SIZE);
    dl(d.b_fc2, h ? h->b_fc2 : nullptr, ArchConfig::FC2_B_SIZE);
    cudaFree(d.dw_fc2); cudaFree(d.db_fc2); cudaFree(d.vw_fc2); cudaFree(d.vb_fc2);
}
} // namespace

float cuda_evaluate(const TinyImageNetDataset& val_data, const ModelParameters& params, int batch_size) {
    if (val_data.size() == 0 || batch_size <= 0) return 0.0f;
    ModelParameters d_p;
    alloc_gpu_params(d_p, params);
    CudaBuffers buf(batch_size);

    std::vector<float> h_in(batch_size * TinyImageNetDataset::IMAGE_SIZE);
    std::vector<int> h_lbl(batch_size);
    std::vector<float> h_probs(batch_size * ArchConfig::FC2_OUT);

    int correct = 0, evaluated = 0;
    for (size_t start = 0; start < val_data.size(); start += batch_size) {
        int actual = val_data.get_batch(start, batch_size, h_in.data(), h_lbl.data());
        if (actual == 0) break;

        CUDA_CHECK(cudaMemcpy(buf.d_in, h_in.data(), batch_size * TinyImageNetDataset::IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buf.d_labels, h_lbl.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice));

        forward_batch(buf, d_p, batch_size, false);
        CUDA_CHECK(cudaMemcpy(h_probs.data(), buf.d_probs, batch_size * ArchConfig::FC2_OUT * sizeof(float), cudaMemcpyDeviceToHost));

        for (int i = 0; i < actual; ++i) {
            int pred = 0; float best = -1.0f;
            for (int c = 0; c < ArchConfig::FC2_OUT; ++c) {
                float v = h_probs[i * ArchConfig::FC2_OUT + c];
                if (v > best) { best = v; pred = c; }
            }
            if (pred == h_lbl[i]) correct++;
        }
        evaluated += actual;
    }

    free_gpu_params(d_p, nullptr);
    return evaluated ? (100.0f * correct / evaluated) : 0.0f;
}

void cuda_train(const TinyImageNetDataset& train_data, const TinyImageNetDataset& val_data,
                ModelParameters& h_params, const TrainingConfig& config) {
    std::cout << "\n=======================================================\n"
              << " Starting Deep CNN Training using CUDA C / C++\n"
              << " Classes: " << ArchConfig::FC2_OUT << " | Batch: " << config.batch_size
              << " | Epochs: " << config.epochs << " | LR: " << config.learning_rate << "\n"
              << "=======================================================\n";

    ModelParameters d_p;
    alloc_gpu_params(d_p, h_params);
    CudaBuffers buf(config.batch_size);

    std::vector<float> h_in(config.batch_size * TinyImageNetDataset::IMAGE_SIZE);
    std::vector<int> h_lbl(config.batch_size);
    std::vector<float> h_loss(config.batch_size);
    std::vector<float> h_probs(config.batch_size * ArchConfig::FC2_OUT);

    CPUTimer epoch_timer, batch_timer;
    size_t total_batches = (train_data.size() + config.batch_size - 1) / config.batch_size;

    std::string stats_file = config.stats_file.empty() ? "cuda_stats.csv" : config.stats_file;
    std::ofstream csv(stats_file);
    if (csv.is_open()) {
        csv << "epoch,train_loss,train_acc_pct,train_error_pct,val_acc_pct,val_error_pct,epoch_time_sec,throughput_img_sec\n";
    }

    for (int epoch = 1; epoch <= config.epochs; ++epoch) {
        epoch_timer.start();
        float epoch_loss = 0.0f;
        int epoch_correct = 0, samples_processed = 0;
        const_cast<TinyImageNetDataset&>(train_data).shuffle(epoch * 1000 + 42);

        for (size_t b_idx = 0; b_idx < total_batches; ++b_idx) {
            batch_timer.start();
            size_t start = b_idx * config.batch_size;
            int actual = train_data.get_batch(start, config.batch_size, h_in.data(), h_lbl.data());
            if (actual == 0) break;

            CUDA_CHECK(cudaMemcpy(buf.d_in, h_in.data(), config.batch_size * TinyImageNetDataset::IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(buf.d_labels, h_lbl.data(), config.batch_size * sizeof(int), cudaMemcpyHostToDevice));

            forward_batch(buf, d_p, config.batch_size, true);
            backward_and_update(buf, d_p, config.batch_size, config);

            CUDA_CHECK(cudaMemcpy(h_loss.data(), buf.d_loss, config.batch_size * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_probs.data(), buf.d_probs, config.batch_size * ArchConfig::FC2_OUT * sizeof(float), cudaMemcpyDeviceToHost));

            float b_loss = 0.0f;
            int b_hits = 0;
            for (int i = 0; i < actual; ++i) {
                b_loss += h_loss[i];
                int pred = 0; float best = -1.0f;
                for (int c = 0; c < ArchConfig::FC2_OUT; ++c) {
                    float v = h_probs[i * ArchConfig::FC2_OUT + c];
                    if (v > best) { best = v; pred = c; }
                }
                if (pred == h_lbl[i]) b_hits++;
            }

            epoch_loss += b_loss;
            epoch_correct += b_hits;
            samples_processed += actual;

            if (b_idx % 10 == 0 || b_idx == total_batches - 1) {
                double ms = batch_timer.stop();
                float avg_l = b_loss / actual;
                float acc = 100.0f * b_hits / actual;
                float throughput = actual / (ms / 1000.0f);
                std::cout << "\r[Epoch " << epoch << "/" << config.epochs << "] Batch " << (b_idx + 1) << "/" << total_batches
                          << " | Loss: " << std::fixed << std::setprecision(4) << avg_l
                          << " | Acc: " << std::setprecision(2) << acc << "%"
                          << " | " << std::setprecision(1) << throughput << " img/s" << std::flush;
            }
        }

        double ep_ms = epoch_timer.stop();
        if (samples_processed == 0) break;
        float tr_loss = epoch_loss / samples_processed;
        float tr_acc = 100.0f * epoch_correct / samples_processed;
        float tr_err = 100.0f - tr_acc;
        float val_acc = 0.0f;
        float val_err = 0.0f;

        std::cout << "\n--> Epoch " << epoch << " Finished in " << std::fixed << std::setprecision(2) << ep_ms / 1000.0 << "s"
                  << " | Train Loss: " << tr_loss
                  << " | Train Acc: " << tr_acc << "%\n";

        if (val_data.size() > 0) {
            CUDA_CHECK(cudaMemcpy(h_params.w_c1, d_p.w_c1, ArchConfig::C1_W_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_params.b_c1, d_p.b_c1, ArchConfig::C1_B_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_params.w_c2, d_p.w_c2, ArchConfig::C2_W_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_params.b_c2, d_p.b_c2, ArchConfig::C2_B_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_params.w_fc1, d_p.w_fc1, ArchConfig::FC1_W_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_params.b_fc1, d_p.b_fc1, ArchConfig::FC1_B_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_params.w_fc2, d_p.w_fc2, ArchConfig::FC2_W_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_params.b_fc2, d_p.b_fc2, ArchConfig::FC2_B_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            val_acc = cuda_evaluate(val_data, h_params, config.batch_size);
            val_err = 100.0f - val_acc;
            std::cout << "--> Validation Accuracy: " << std::setprecision(2) << val_acc << "%\n";
        }

        double ep_sec = ep_ms / 1000.0;
        double throughput = (ep_sec > 0.0) ? (samples_processed / ep_sec) : 0.0;
        if (csv.is_open()) {
            csv << epoch << "," << std::fixed << std::setprecision(4) << tr_loss << ","
                << std::setprecision(2) << tr_acc << "," << tr_err << ","
                << val_acc << "," << val_err << ","
                << ep_sec << "," << throughput << "\n";
            csv.flush();
        }
    }

    free_gpu_params(d_p, &h_params);

    std::string model_file = config.model_save_path.empty() ? "cuda_model.bin" : config.model_save_path;
    if (h_params.save_to_file(model_file)) {
        std::cout << "--> Model checkpoint successfully saved to: " << model_file << "\n";
    }
    if (csv.is_open()) {
        std::cout << "--> Epoch training & testing statistics saved to: " << stats_file << "\n";
    }

    std::cout << "=======================================================\n"
              << " CUDA Training Completed Successfully!\n"
              << "=======================================================\n";
}

} // namespace cutin
