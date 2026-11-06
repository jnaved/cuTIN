#include "network.hpp"
#include <cmath>
#include <cfloat>

namespace cutin {

void omp_linear_forward(const float* x, const float* w, const float* b, float* y,
                        int B, int in_features, int out_features) {
    const long long total = 1LL * B * out_features;
    #pragma omp parallel for schedule(static)
    for (long long idx = 0; idx < total; ++idx) {
        int out = idx % out_features;
        int batch = idx / out_features;
        float sum = b ? b[out] : 0.0f;
        const float* x_row = x + static_cast<size_t>(batch) * in_features;
        const float* w_row = w + static_cast<size_t>(out) * in_features;
        for (int k = 0; k < in_features; ++k) sum += x_row[k] * w_row[k];
        y[idx] = sum;
    }
}

void omp_linear_backward(const float* dy, const float* x, const float* w,
                         float* dx, float* dw, float* db,
                         int B, int in_features, int out_features) {
    if (dx) {
        const long long total = 1LL * B * in_features;
        #pragma omp parallel for schedule(static)
        for (long long idx = 0; idx < total; ++idx) {
            int in = idx % in_features;
            int batch = idx / in_features;
            float sum = 0.0f;
            for (int out = 0; out < out_features; ++out)
                sum += dy[batch * out_features + out] * w[out * in_features + in];
            dx[idx] = sum;
        }
    }
    if (dw) {
        const long long total = 1LL * out_features * in_features;
        #pragma omp parallel for schedule(static)
        for (long long idx = 0; idx < total; ++idx) {
            int in = idx % in_features;
            int out = idx / in_features;
            float sum = 0.0f;
            for (int batch = 0; batch < B; ++batch)
                sum += dy[batch * out_features + out] * x[batch * in_features + in];
            dw[idx] = sum;
        }
    }
    if (db) {
        #pragma omp parallel for schedule(static)
        for (int out = 0; out < out_features; ++out) {
            float sum = 0.0f;
            for (int batch = 0; batch < B; ++batch) sum += dy[batch * out_features + out];
            db[out] = sum;
        }
    }
}

void omp_softmax_cross_entropy_forward_backward(const float* logits, const int* labels,
                                                float* probs, float* dlogits,
                                                float* batch_loss, int B, int num_classes) {
    #pragma omp parallel for schedule(static)
    for (int batch = 0; batch < B; ++batch) {
        const float* row = logits + static_cast<size_t>(batch) * num_classes;
        float* p_row = probs + static_cast<size_t>(batch) * num_classes;
        float* grad_row = dlogits + static_cast<size_t>(batch) * num_classes;
        int target = labels[batch];
        float max_value = -FLT_MAX;
        for (int c = 0; c < num_classes; ++c)
            if (row[c] > max_value) max_value = row[c];
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            p_row[c] = std::exp(row[c] - max_value);
            sum_exp += p_row[c];
        }
        float inv_sum = 1.0f / (sum_exp + 1e-12f);
        float inv_batch = 1.0f / static_cast<float>(B);
        for (int c = 0; c < num_classes; ++c) {
            float p = p_row[c] * inv_sum;
            p_row[c] = p;
            grad_row[c] = (p - (c == target ? 1.0f : 0.0f)) * inv_batch;
        }
        if (batch_loss) {
            if (target >= 0 && target < num_classes)
                batch_loss[batch] = -std::log(std::fmax(p_row[target], 1e-12f));
            else batch_loss[batch] = 0.0f;
        }
    }
}

void omp_sgd_update(float* w, float* dw, float* vw,
                    float* b, float* db, float* vb,
                    size_t w_size, size_t b_size,
                    float lr, float momentum, float weight_decay) {
    if (w && dw && vw) {
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(w_size); ++i) {
            float grad = dw[i] + weight_decay * w[i];
            vw[i] = momentum * vw[i] + grad;
            w[i] -= lr * vw[i];
        }
    }
    if (b && db && vb) {
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(b_size); ++i) {
            vb[i] = momentum * vb[i] + db[i];
            b[i] -= lr * vb[i];
        }
    }
}

} // namespace cutin
