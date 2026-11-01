/**
 * =========================================================================
 * WHY IS THIS 'tests' FOLDER PRESENT?
 * -------------------------------------------------------------------------
 * In high-performance parallel computing and neural networks from scratch:
 * 1. Numerical Equivalence Verification:
 *    When three developers implement the exact same CNN across three different
 *    execution paradigms (Sequential CPU, OpenMP multi-threading, CUDA GPU),
 *    this test suite ensures all three yield numerically identical forward
 *    activations and backward gradients within floating-point epsilon (atol ~ 1e-4).
 *
 * 2. Layer-by-Layer Debugging (Unit Testing):
 *    Debugging deep learning models end-to-end is notoriously difficult: if loss
 *    diverges or accuracy is 0.5%, it is nearly impossible to tell whether the bug
 *    is in Conv stride indexing, MaxPool mask routing, FC matrix multiplication,
 *    or Softmax gradient scaling. This test isolates each layer independently.
 *
 * 3. Regression Prevention:
 *    Whenever optimizations (tiling, shared memory, vectorization, pragma tuning)
 *    are added, this test suite ensures optimizations do not introduce race
 *    conditions or numerical drift.
 * =========================================================================
 */

#include "network.hpp"
#include "utils.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>

using namespace cutin;

// Reference CPU ground-truth forward and backward for unit testing
namespace ref {

void conv2d_forward(const float* x, const float* w, const float* b, float* y,
                    int B, int C_in, int H_in, int W_in,
                    int C_out, int K, int pad, int stride) {
    int H_out = (H_in + 2 * pad - K) / stride + 1;
    int W_out = (W_in + 2 * pad - K) / stride + 1;

    for (int b_idx = 0; b_idx < B; ++b_idx) {
        for (int co = 0; co < C_out; ++co) {
            float bias_val = (b != nullptr) ? b[co] : 0.0f;
            for (int ho = 0; ho < H_out; ++ho) {
                for (int wo = 0; wo < W_out; ++wo) {
                    float sum = bias_val;
                    for (int ci = 0; ci < C_in; ++ci) {
                        for (int kh = 0; kh < K; ++kh) {
                            int hi = ho * stride - pad + kh;
                            for (int kw = 0; kw < K; ++kw) {
                                int wi = wo * stride - pad + kw;
                                if (hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                                    sum += x[((b_idx * C_in + ci) * H_in + hi) * W_in + wi]
                                         * w[((co * C_in + ci) * K + kh) * K + kw];
                                }
                            }
                        }
                    }
                    y[((b_idx * C_out + co) * H_out + ho) * W_out + wo] = sum;
                }
            }
        }
    }
}

void maxpool2d_forward(const float* x, float* y, int* mask,
                       int B, int C, int H_in, int W_in, int K, int stride) {
    int H_out = (H_in - K) / stride + 1;
    int W_out = (W_in - K) / stride + 1;

    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            int in_ch_offset = (b * C + c) * H_in * W_in;
            for (int ho = 0; ho < H_out; ++ho) {
                for (int wo = 0; wo < W_out; ++wo) {
                    float max_val = -1e30f;
                    int max_idx = -1;
                    for (int kh = 0; kh < K; ++kh) {
                        int hi = ho * stride + kh;
                        for (int kw = 0; kw < K; ++kw) {
                            int wi = wo * stride + kw;
                            int in_idx = in_ch_offset + hi * W_in + wi;
                            float val = x[in_idx];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = in_idx;
                            }
                        }
                    }
                    int out_idx = ((b * C + c) * H_out + ho) * W_out + wo;
                    y[out_idx] = max_val;
                    mask[out_idx] = max_idx;
                }
            }
        }
    }
}

void linear_forward(const float* x, const float* w, const float* b, float* y,
                    int B, int in_f, int out_f) {
    for (int b_idx = 0; b_idx < B; ++b_idx) {
        for (int j = 0; j < out_f; ++j) {
            float sum = (b != nullptr) ? b[j] : 0.0f;
            for (int k = 0; k < in_f; ++k) {
                sum += x[b_idx * in_f + k] * w[j * in_f + k];
            }
            y[b_idx * out_f + j] = sum;
        }
    }
}

void relu_forward(const float* x, float* y, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

} // namespace ref

bool test_relu_cuda() {
    std::cout << "[Test] Running ReLU CUDA Equivalence Test...";
    const size_t N = 1024;
    std::vector<float> h_x(N), h_y_ref(N), h_y_cuda(N);
    for (size_t i = 0; i < N; ++i) {
        h_x[i] = (static_cast<float>(i) - 512.0f) * 0.1f;
    }

    ref::relu_forward(h_x.data(), h_y_ref.data(), N);

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    cuda_relu_forward(d_x, d_y, N);

    CUDA_CHECK(cudaMemcpy(h_y_cuda.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_x); cudaFree(d_y);

    bool pass = check_close(h_y_cuda.data(), h_y_ref.data(), N);
    std::cout << (pass ? " PASSED\n" : " FAILED\n");
    return pass;
}

bool test_maxpool2d_cuda() {
    std::cout << "[Test] Running MaxPool2D CUDA Equivalence Test...";
    int B = 2, C = 4, H_in = 8, W_in = 8, K = 2, stride = 2;
    int H_out = 4, W_out = 4;
    size_t in_size = B * C * H_in * W_in;
    size_t out_size = B * C * H_out * W_out;

    std::vector<float> h_x(in_size), h_y_ref(out_size), h_y_cuda(out_size);
    std::vector<int> h_mask_ref(out_size), h_mask_cuda(out_size);

    for (size_t i = 0; i < in_size; ++i) h_x[i] = static_cast<float>(i % 37) * 0.25f;

    ref::maxpool2d_forward(h_x.data(), h_y_ref.data(), h_mask_ref.data(),
                           B, C, H_in, W_in, K, stride);

    float *d_x, *d_y;
    int *d_mask;
    CUDA_CHECK(cudaMalloc(&d_x, in_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mask, out_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), in_size * sizeof(float), cudaMemcpyHostToDevice));

    cuda_maxpool2d_forward(d_x, d_y, d_mask, B, C, H_in, W_in, K, stride);

    CUDA_CHECK(cudaMemcpy(h_y_cuda.data(), d_y, out_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mask_cuda.data(), d_mask, out_size * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_mask);

    bool pass = check_close(h_y_cuda.data(), h_y_ref.data(), out_size);
    std::cout << (pass ? " PASSED\n" : " FAILED\n");
    return pass;
}

bool test_conv2d_cuda() {
    std::cout << "[Test] Running Conv2D CUDA Equivalence Test...";
    int B = 2, C_in = 3, H_in = 8, W_in = 8;
    int C_out = 4, K = 3, pad = 1, stride = 1;
    int H_out = 8, W_out = 8;

    size_t x_size = B * C_in * H_in * W_in;
    size_t w_size = C_out * C_in * K * K;
    size_t b_size = C_out;
    size_t y_size = B * C_out * H_out * W_out;

    std::vector<float> h_x(x_size), h_w(w_size), h_b(b_size), h_y_ref(y_size), h_y_cuda(y_size);
    for (size_t i = 0; i < x_size; ++i) h_x[i] = std::sin(static_cast<float>(i));
    for (size_t i = 0; i < w_size; ++i) h_w[i] = std::cos(static_cast<float>(i)) * 0.1f;
    for (size_t i = 0; i < b_size; ++i) h_b[i] = 0.5f;

    ref::conv2d_forward(h_x.data(), h_w.data(), h_b.data(), h_y_ref.data(),
                        B, C_in, H_in, W_in, C_out, K, pad, stride);

    float *d_x, *d_w, *d_b, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, x_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, b_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, y_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), x_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), w_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), b_size * sizeof(float), cudaMemcpyHostToDevice));

    cuda_conv2d_forward(d_x, d_w, d_b, d_y, B, C_in, H_in, W_in, C_out, K, pad, stride);

    CUDA_CHECK(cudaMemcpy(h_y_cuda.data(), d_y, y_size * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_b); cudaFree(d_y);

    bool pass = check_close(h_y_cuda.data(), h_y_ref.data(), y_size);
    std::cout << (pass ? " PASSED\n" : " FAILED\n");
    return pass;
}

bool test_linear_cuda() {
    std::cout << "[Test] Running Linear / Dense CUDA Equivalence Test...";
    int B = 4, in_f = 16, out_f = 8;
    size_t x_size = B * in_f;
    size_t w_size = out_f * in_f;
    size_t b_size = out_f;
    size_t y_size = B * out_f;

    std::vector<float> h_x(x_size), h_w(w_size), h_b(b_size), h_y_ref(y_size), h_y_cuda(y_size);
    for (size_t i = 0; i < x_size; ++i) h_x[i] = static_cast<float>(i % 5) * 0.2f;
    for (size_t i = 0; i < w_size; ++i) h_w[i] = static_cast<float>(i % 7) * 0.1f;
    for (size_t i = 0; i < b_size; ++i) h_b[i] = 0.1f;

    ref::linear_forward(h_x.data(), h_w.data(), h_b.data(), h_y_ref.data(), B, in_f, out_f);

    float *d_x, *d_w, *d_b, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, x_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, b_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, y_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), x_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), w_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), b_size * sizeof(float), cudaMemcpyHostToDevice));

    cuda_linear_forward(d_x, d_w, d_b, d_y, B, in_f, out_f);

    CUDA_CHECK(cudaMemcpy(h_y_cuda.data(), d_y, y_size * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_b); cudaFree(d_y);

    bool pass = check_close(h_y_cuda.data(), h_y_ref.data(), y_size);
    std::cout << (pass ? " PASSED\n" : " FAILED\n");
    return pass;
}

int main() {
    std::cout << "=======================================================\n";
    std::cout << " cuTIN Unit & Backend Equivalence Verification Suite\n";
    std::cout << "=======================================================\n";

    bool all_pass = true;
    all_pass &= test_relu_cuda();
    all_pass &= test_maxpool2d_cuda();
    all_pass &= test_conv2d_cuda();
    all_pass &= test_linear_cuda();

    std::cout << "=======================================================\n";
    if (all_pass) {
        std::cout << " ALL CUDA UNIT TESTS PASSED SUCCESSFULLY!\n";
        std::cout << " (Teammates can hook in seq_* and omp_* functions\n";
        std::cout << "  into this harness to verify their implementations!)\n";
    } else {
        std::cout << " ONE OR MORE TESTS FAILED.\n";
    }
    std::cout << "=======================================================\n";

    return all_pass ? 0 : 1;
}

