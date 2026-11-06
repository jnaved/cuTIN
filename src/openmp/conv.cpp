#include "network.hpp"
#include <cfloat>
#include <cstddef>

namespace cutin {

void omp_relu_forward(const float* x, float* y, size_t size) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(size); ++i)
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

void omp_relu_backward(const float* dy, const float* y, float* dx, size_t size) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(size); ++i)
        dx[i] = y[i] > 0.0f ? dy[i] : 0.0f;
}

void omp_maxpool2d_forward(const float* x, float* y, int* mask,
                           int B, int C, int H_in, int W_in, int K, int stride) {
    const int H_out = (H_in - K) / stride + 1;
    const int W_out = (W_in - K) / stride + 1;
    const long long total = 1LL * B * C * H_out * W_out;

    #pragma omp parallel for schedule(static)
    for (long long idx = 0; idx < total; ++idx) {
        int w_out = idx % W_out;
        long long tmp = idx / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int b = tmp / C;
        int h_start = h_out * stride;
        int w_start = w_out * stride;
        int input_offset = (b * C + c) * H_in * W_in;
        float max_value = -FLT_MAX;
        int max_index = -1;

        for (int kh = 0; kh < K; ++kh) {
            int h = h_start + kh;
            if (h >= H_in) continue;
            for (int kw = 0; kw < K; ++kw) {
                int w = w_start + kw;
                if (w >= W_in) continue;
                int in_idx = input_offset + h * W_in + w;
                if (x[in_idx] > max_value) {
                    max_value = x[in_idx];
                    max_index = in_idx;
                }
            }
        }
        y[idx] = max_value;
        mask[idx] = max_index;
    }
}

void omp_maxpool2d_backward(const float* dy, const int* mask, float* dx,
                            int B, int C, int H_out, int W_out,
                            int H_in, int W_in, int K, int stride) {
    (void)C; (void)H_out; (void)W_out; (void)K; (void)stride;
    const long long total_in = 1LL * B * C * H_in * W_in;
    const long long total_out = 1LL * B * C * H_out * W_out;
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < total_in; ++i) dx[i] = 0.0f;

    // Atomic addition matches CUDA atomicAdd when multiple pooling windows
    // route gradients to the same input element (for overlapping pools).
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < total_out; ++i) {
        int target = mask[i];
        if (target >= 0 && target < total_in) {
            #pragma omp atomic update
            dx[target] += dy[i];
        }
    }
}

void omp_conv2d_forward(const float* x, const float* w, const float* b, float* y,
                        int B, int C_in, int H_in, int W_in,
                        int C_out, int K, int pad, int stride) {
    const int H_out = (H_in + 2 * pad - K) / stride + 1;
    const int W_out = (W_in + 2 * pad - K) / stride + 1;
    const long long total = 1LL * B * C_out * H_out * W_out;

    #pragma omp parallel for schedule(static)
    for (long long idx = 0; idx < total; ++idx) {
        int w_out = idx % W_out;
        long long tmp = idx / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int c_out = tmp % C_out;
        int batch = tmp / C_out;
        float sum = b ? b[c_out] : 0.0f;
        int h_base = h_out * stride - pad;
        int w_base = w_out * stride - pad;

        for (int c = 0; c < C_in; ++c) {
            int x_channel = (batch * C_in + c) * H_in * W_in;
            int w_channel = (c_out * C_in + c) * K * K;
            for (int kh = 0; kh < K; ++kh) {
                int h = h_base + kh;
                if (h < 0 || h >= H_in) continue;
                for (int kw = 0; kw < K; ++kw) {
                    int wi = w_base + kw;
                    if (wi >= 0 && wi < W_in)
                        sum += x[x_channel + h * W_in + wi] * w[w_channel + kh * K + kw];
                }
            }
        }
        y[idx] = sum;
    }
}

void omp_conv2d_backward(const float* dy, const float* x, const float* w,
                         float* dx, float* dw, float* db,
                         int B, int C_in, int H_in, int W_in,
                         int C_out, int K, int pad, int stride) {
    const int H_out = (H_in + 2 * pad - K) / stride + 1;
    const int W_out = (W_in + 2 * pad - K) / stride + 1;

    if (dx) {
        const long long total_dx = 1LL * B * C_in * H_in * W_in;
        #pragma omp parallel for schedule(static)
        for (long long idx = 0; idx < total_dx; ++idx) {
            int wi = idx % W_in;
            long long tmp = idx / W_in;
            int hi = tmp % H_in;
            tmp /= H_in;
            int ci = tmp % C_in;
            int batch = tmp / C_in;
            float grad = 0.0f;
            for (int co = 0; co < C_out; ++co) {
                int dy_channel = (batch * C_out + co) * H_out * W_out;
                int w_channel = (co * C_in + ci) * K * K;
                for (int kh = 0; kh < K; ++kh) {
                    int h_coord = hi + pad - kh;
                    if (h_coord % stride != 0) continue;
                    int ho = h_coord / stride;
                    if (ho < 0 || ho >= H_out) continue;
                    for (int kw = 0; kw < K; ++kw) {
                        int w_coord = wi + pad - kw;
                        if (w_coord % stride != 0) continue;
                        int wo = w_coord / stride;
                        if (wo >= 0 && wo < W_out)
                            grad += dy[dy_channel + ho * W_out + wo] * w[w_channel + kh * K + kw];
                    }
                }
            }
            dx[idx] = grad;
        }
    }

    if (dw) {
        const long long total_dw = 1LL * C_out * C_in * K * K;
        #pragma omp parallel for schedule(static)
        for (long long idx = 0; idx < total_dw; ++idx) {
            int kw = idx % K;
            long long tmp = idx / K;
            int kh = tmp % K;
            tmp /= K;
            int ci = tmp % C_in;
            int co = tmp / C_in;
            float grad = 0.0f;
            for (int batch = 0; batch < B; ++batch) {
                int dy_channel = (batch * C_out + co) * H_out * W_out;
                int x_channel = (batch * C_in + ci) * H_in * W_in;
                for (int ho = 0; ho < H_out; ++ho) {
                    int hi = ho * stride - pad + kh;
                    if (hi < 0 || hi >= H_in) continue;
                    for (int wo = 0; wo < W_out; ++wo) {
                        int wi = wo * stride - pad + kw;
                        if (wi >= 0 && wi < W_in)
                            grad += dy[dy_channel + ho * W_out + wo] * x[x_channel + hi * W_in + wi];
                    }
                }
            }
            dw[idx] = grad;
        }
    }

    if (db) {
        #pragma omp parallel for schedule(static)
        for (int co = 0; co < C_out; ++co) {
            float grad = 0.0f;
            for (int batch = 0; batch < B; ++batch)
                for (int h = 0; h < H_out; ++h)
                    for (int wi = 0; wi < W_out; ++wi)
                        grad += dy[((batch * C_out + co) * H_out + h) * W_out + wi];
            db[co] = grad;
        }
    }
}

} // namespace cutin
