
#include "network.hpp"
#include <iostream>

namespace cutin {

// =========================================================================
// HELPER FUNCTIONS
// Manual exponential and logarithm calculations.
// No standard math functions are used.
// =========================================================================

static float seq_exp(float x) {

    // Prevent extremely small values from causing unnecessary calculations
    if (x < -80.0f) {
        return 0.0f;
    }

    // Range reduction: exp(x) = exp(x/16)^16
    float value = x / 16.0f;

    // Taylor series for exp(value)
    float sum = 1.0f;
    float term = 1.0f;

    for (int i = 1; i <= 24; i++) {
        term = term * value / (float)i;
        sum += term;
    }

    // Raise the result to the 16th power
    sum = sum * sum;
    sum = sum * sum;
    sum = sum * sum;
    sum = sum * sum;

    return sum;
}


static float seq_log(float x) {

    if (x <= 0.0f) {
        return -80.0f;
    }

    // Convert x into the range [0.5, 1]
    float m = x;
    float adjustment = 0.0f;

    while (m > 1.0f) {
        m = m / 2.0f;
        adjustment += 0.69314718056f;
    }

    while (m < 0.5f) {
        m = m * 2.0f;
        adjustment -= 0.69314718056f;
    }

    // log(m) = 2 * (z + z^3/3 + z^5/5 + ...)
    float z = (m - 1.0f) / (m + 1.0f);
    float z_squared = z * z;

    float term = z;
    float sum = 0.0f;

    for (int i = 1; i <= 39; i += 2) {
        sum += term / (float)i;
        term = term * z_squared;
    }

    return 2.0f * sum + adjustment;
}


// =========================================================================
// 1. LINEAR / DENSE FORWARD
//
// y = x * W^T + b
//
// Input:  (B, in_features)
// Weight: (out_features, in_features)
// Output: (B, out_features)
// =========================================================================

void seq_linear_forward(const float* x, const float* w,
                        const float* b, float* y,
                        int B, int in_features, int out_features) {

    for (int batch = 0; batch < B; batch++) {

        for (int out = 0; out < out_features; out++) {

            float sum = 0.0f;

            // Add bias
            if (b != 0) {
                sum = b[out];
            }

            // Multiply input and weight
            for (int in = 0; in < in_features; in++) {

                int x_index = batch * in_features + in;

                int w_index = out * in_features + in;

                sum += x[x_index] * w[w_index];
            }

            int y_index = batch * out_features + out;

            y[y_index] = sum;
        }
    }
}


// =========================================================================
// 2. LINEAR / DENSE BACKWARD
//
// dx = dy * W
// dw = dy^T * x
// db = sum(dy, axis=0)
// =========================================================================

void seq_linear_backward(const float* dy, const float* x,
                         const float* w,
                         float* dx, float* dw, float* db,
                         int B, int in_features, int out_features) {


    // -----------------------------------------------------
    // Step 1: Calculate input gradient (dx)
    // dx[b][i] = sum(dy[b][j] * w[j][i])
    // -----------------------------------------------------

    if (dx != 0) {

        for (int batch = 0; batch < B; batch++) {

            for (int in = 0; in < in_features; in++) {

                float sum = 0.0f;

                for (int out = 0; out < out_features; out++) {

                    int dy_index = batch * out_features + out;

                    int w_index = out * in_features + in;

                    sum += dy[dy_index] * w[w_index];
                }

                int dx_index = batch * in_features + in;

                dx[dx_index] = sum;
            }
        }
    }


    // -----------------------------------------------------
    // Step 2: Calculate weight gradient (dw)
    // dw[j][i] = sum(dy[b][j] * x[b][i])
    // -----------------------------------------------------

    if (dw != 0) {

        for (int out = 0; out < out_features; out++) {

            for (int in = 0; in < in_features; in++) {

                float sum = 0.0f;

                for (int batch = 0; batch < B; batch++) {

                    int dy_index = batch * out_features + out;

                    int x_index = batch * in_features + in;

                    sum += dy[dy_index] * x[x_index];
                }

                int dw_index = out * in_features + in;

                dw[dw_index] = sum;
            }
        }
    }


    // -----------------------------------------------------
    // Step 3: Calculate bias gradient (db)
    // db[j] = sum(dy[b][j])
    // -----------------------------------------------------

    if (db != 0) {

        for (int out = 0; out < out_features; out++) {

            float sum = 0.0f;

            for (int batch = 0; batch < B; batch++) {

                int dy_index = batch * out_features + out;

                sum += dy[dy_index];
            }

            db[out] = sum;
        }
    }
}


// =========================================================================
// 3. SOFTMAX + CATEGORICAL CROSS-ENTROPY
//
// Softmax:
// p[c] = exp(logit[c] - max) / sum(exp(logit - max))
//
// Gradient:
// dlogits[c] = (p[c] - target_indicator) / B
//
// Loss:
// -log(probability of correct class)
// =========================================================================

void seq_softmax_cross_entropy_forward_backward(
    const float* logits,
    const int* labels,
    float* probs,
    float* dlogits,
    float* batch_loss,
    int B,
    int num_classes) {

    if (B <= 0 || num_classes <= 0) {
        return;
    }

    for (int batch = 0; batch < B; batch++) {

        int target = labels[batch];

        int row_start = batch * num_classes;

        // -------------------------------------------------
        // Step 1: Find maximum logit
        // Used for numerical stability
        // -------------------------------------------------

        float max_val = -3.402823466e+38F;

        for (int c = 0; c < num_classes; c++) {

            if (logits[row_start + c] > max_val) {
                max_val = logits[row_start + c];
            }
        }


        // -------------------------------------------------
        // Step 2: Calculate exponential values
        // -------------------------------------------------

        float sum_exp = 0.0f;

        for (int c = 0; c < num_classes; c++) {

            float value = logits[row_start + c] - max_val;

            float exp_value = seq_exp(value);

            probs[row_start + c] = exp_value;

            sum_exp += exp_value;
        }


        // -------------------------------------------------
        // Step 3: Calculate probabilities
        // -------------------------------------------------

        float inv_sum = 1.0f / (sum_exp + 1e-12f);

        float inv_B = 1.0f / (float)B;

        for (int c = 0; c < num_classes; c++) {

            float p = probs[row_start + c] * inv_sum;

            probs[row_start + c] = p;
        }


        // -------------------------------------------------
        // Step 4: Calculate gradient
        // dlogits = (p - target_indicator) / B
        // -------------------------------------------------

        for (int c = 0; c < num_classes; c++) {

            float target_indicator = 0.0f;

            if (c == target) {
                target_indicator = 1.0f;
            }

            float p = probs[row_start + c];

            dlogits[row_start + c] =
                (p - target_indicator) * inv_B;
        }


        // -------------------------------------------------
        // Step 5: Calculate cross-entropy loss
        // loss = -log(p_target)
        // -------------------------------------------------

        if (batch_loss != 0) {

            if (target >= 0 && target < num_classes) {

                float p_target = probs[row_start + target];

                if (p_target < 1e-12f) {
                    p_target = 1e-12f;
                }

                batch_loss[batch] = -seq_log(p_target);
            }
            else {

                batch_loss[batch] = 0.0f;
            }
        }
    }
}


// =========================================================================
// 4. SGD WITH MOMENTUM AND WEIGHT DECAY
//
// g = gradient + weight_decay * parameter
// v = momentum * v + g
// parameter = parameter - learning_rate * v
//
// Biases do not use weight decay.
// =========================================================================

void seq_sgd_update(float* w, float* dw, float* vw,
                    float* b, float* db, float* vb,
                    size_t w_size, size_t b_size,
                    float lr, float momentum, float weight_decay) {


    // -----------------------------------------------------
    // Step 1: Update weights
    // -----------------------------------------------------

    if (w != 0 && w_size > 0) {

        for (size_t i = 0; i < w_size; i++) {

            float p = w[i];

            float g = dw[i];

            // Apply weight decay
            if (weight_decay != 0.0f) {
                g += weight_decay * p;
            }

            // Update momentum
            float v = vw[i];

            v = momentum * v + g;

            vw[i] = v;

            // Update parameter
            w[i] = p - lr * v;
        }
    }


    // -----------------------------------------------------
    // Step 2: Update biases
    // Biases do not use weight decay.
    // -----------------------------------------------------

    if (b != 0 && b_size > 0) {

        for (size_t i = 0; i < b_size; i++) {

            float p = b[i];

            float g = db[i];

            // Update momentum
            float v = vb[i];

            v = momentum * v + g;

            vb[i] = v;

            // Update bias
            b[i] = p - lr * v;
        }
    }
}

} // namespace cutin
