
#include "network.hpp"
#include <iostream>

namespace cutin {

// =========================================================================
// SEQUENTIAL CPU IMPLEMENTATION
// No OpenMP, CUDA, STL algorithms, or library math functions.
// Tensor layout: NCHW
// Weight layout: OIHW
// =========================================================================


// -------------------------------------------------------------------------
// 1. ReLU FORWARD
// y = max(0, x)
// -------------------------------------------------------------------------

void seq_relu_forward(const float* x, float* y, size_t size) {

    for (size_t i = 0; i < size; i++) {

        if (x[i] > 0.0f) {
            y[i] = x[i];
        }
        else {
            y[i] = 0.0f;
        }
    }
}


// -------------------------------------------------------------------------
// 2. ReLU BACKWARD
// dx = dy if y > 0, otherwise 0
// -------------------------------------------------------------------------

void seq_relu_backward(const float* dy, const float* y,
                       float* dx, size_t size) {

    for (size_t i = 0; i < size; i++) {

        if (y[i] > 0.0f) {
            dx[i] = dy[i];
        }
        else {
            dx[i] = 0.0f;
        }
    }
}


// -------------------------------------------------------------------------
// 3. MAXPOOL2D FORWARD
// Input:  (B, C, H_in, W_in)
// Output: (B, C, H_out, W_out)
// H_out = (H_in - K) / stride + 1
// W_out = (W_in - K) / stride + 1
//
// mask stores the absolute flattened input index of the maximum value.
// -------------------------------------------------------------------------

void seq_maxpool2d_forward(const float* x, float* y, int* mask,
                           int B, int C, int H_in, int W_in,
                           int K, int stride) {

    int H_out = (H_in - K) / stride + 1;
    int W_out = (W_in - K) / stride + 1;

    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            for (int ho = 0; ho < H_out; ho++) {
                for (int wo = 0; wo < W_out; wo++) {

                    float max_value = -3.402823466e+38F;
                    int max_index = -1;

                    // Starting position in input
                    int h_start = ho * stride;
                    int w_start = wo * stride;

                    // Search inside the pooling window
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {

                            int hi = h_start + kh;
                            int wi = w_start + kw;

                            int input_index =
                                ((b * C + c) * H_in + hi) * W_in + wi;

                            if (x[input_index] > max_value) {

                                max_value = x[input_index];
                                max_index = input_index;
                            }
                        }
                    }

                    int output_index =
                        ((b * C + c) * H_out + ho) * W_out + wo;

                    y[output_index] = max_value;
                    mask[output_index] = max_index;
                }
            }
        }
    }
}


// -------------------------------------------------------------------------
// 4. MAXPOOL2D BACKWARD
// Route each output gradient to the input position that won in forward.
// -------------------------------------------------------------------------

void seq_maxpool2d_backward(const float* dy, const int* mask,
                            float* dx,
                            int B, int C,
                            int H_out, int W_out,
                            int H_in, int W_in,
                            int K, int stride) {

    // Initialize input gradients to zero
    int input_size = B * C * H_in * W_in;

    for (int i = 0; i < input_size; i++) {
        dx[i] = 0.0f;
    }

    int output_size = B * C * H_out * W_out;

    // Send each output gradient to its winning input position
    for (int i = 0; i < output_size; i++) {

        int input_index = mask[i];

        if (input_index >= 0 && input_index < input_size) {
            dx[input_index] += dy[i];
        }
    }

    (void)K;
    (void)stride;
}


// -------------------------------------------------------------------------
// 5. CONV2D FORWARD
//
// Input:  (B, C_in, H_in, W_in)
// Weight: (C_out, C_in, K, K)
// Bias:   (C_out)
// Output: (B, C_out, H_out, W_out)
//
// H_out = (H_in + 2*pad - K) / stride + 1
// W_out = (W_in + 2*pad - K) / stride + 1
// -------------------------------------------------------------------------

void seq_conv2d_forward(const float* x, const float* w,
                        const float* b, float* y,
                        int B, int C_in, int H_in, int W_in,
                        int C_out, int K, int pad, int stride) {

    int H_out = (H_in + 2 * pad - K) / stride + 1;
    int W_out = (W_in + 2 * pad - K) / stride + 1;

    for (int batch = 0; batch < B; batch++) {
        for (int co = 0; co < C_out; co++) {
            for (int ho = 0; ho < H_out; ho++) {
                for (int wo = 0; wo < W_out; wo++) {

                    float sum = 0.0f;

                    if (b != 0) {
                        sum = b[co];
                    }

                    // Loop over input channels
                    for (int ci = 0; ci < C_in; ci++) {
                        for (int kh = 0; kh < K; kh++) {
                            for (int kw = 0; kw < K; kw++) {

                                int hi = ho * stride + kh - pad;
                                int wi = wo * stride + kw - pad;

                                // Ignore positions outside input boundaries
                                if (hi >= 0 && hi < H_in &&
                                    wi >= 0 && wi < W_in) {

                                    int input_index =
                                        ((batch * C_in + ci) * H_in + hi)
                                        * W_in + wi;

                                    int weight_index =
                                        ((co * C_in + ci) * K + kh)
                                        * K + kw;

                                    sum += x[input_index] * w[weight_index];
                                }
                            }
                        }
                    }

                    int output_index =
                        ((batch * C_out + co) * H_out + ho)
                        * W_out + wo;

                    y[output_index] = sum;
                }
            }
        }
    }
}


// -------------------------------------------------------------------------
// 6. CONV2D BACKWARD
//
// Computes:
// dx = gradient with respect to input
// dw = gradient with respect to weights
// db = gradient with respect to bias
// -------------------------------------------------------------------------

void seq_conv2d_backward(const float* dy, const float* x,
                         const float* w,
                         float* dx, float* dw, float* db,
                         int B, int C_in, int H_in, int W_in,
                         int C_out, int K, int pad, int stride) {

    int H_out = (H_in + 2 * pad - K) / stride + 1;
    int W_out = (W_in + 2 * pad - K) / stride + 1;


    // -----------------------------------------------------
    // Step 1: Initialize gradients to zero
    // -----------------------------------------------------

    int input_size = B * C_in * H_in * W_in;
    int weight_size = C_out * C_in * K * K;

    for (int i = 0; i < input_size; i++) {
        dx[i] = 0.0f;
    }

    for (int i = 0; i < weight_size; i++) {
        dw[i] = 0.0f;
    }

    for (int i = 0; i < C_out; i++) {
        db[i] = 0.0f;
    }


    // -----------------------------------------------------
    // Step 2: Calculate dx, dw and db
    // -----------------------------------------------------

    for (int batch = 0; batch < B; batch++) {
        for (int co = 0; co < C_out; co++) {
            for (int ho = 0; ho < H_out; ho++) {
                for (int wo = 0; wo < W_out; wo++) {

                    int output_index =
                        ((batch * C_out + co) * H_out + ho)
                        * W_out + wo;

                    float grad = dy[output_index];

                    // Bias gradient
                    db[co] += grad;

                    // Input channel and kernel loops
                    for (int ci = 0; ci < C_in; ci++) {
                        for (int kh = 0; kh < K; kh++) {
                            for (int kw = 0; kw < K; kw++) {

                                int hi = ho * stride + kh - pad;
                                int wi = wo * stride + kw - pad;

                                // Check input boundaries
                                if (hi >= 0 && hi < H_in &&
                                    wi >= 0 && wi < W_in) {

                                    int input_index =
                                        ((batch * C_in + ci) * H_in + hi)
                                        * W_in + wi;

                                    int weight_index =
                                        ((co * C_in + ci) * K + kh)
                                        * K + kw;

                                    // Input gradient
                                    dx[input_index] +=
                                        grad * w[weight_index];

                                    // Weight gradient
                                    dw[weight_index] +=
                                        grad * x[input_index];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace cutin
