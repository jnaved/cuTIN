#include "network.hpp"
#include <iostream>

namespace cutin {

// =========================================================================
// SKELETON REFERENCE FOR SEQUENTIAL IMPLEMENTATION
// Teammate note: Implement the sequential CPU operations below.
// =========================================================================

void seq_linear_forward(const float* x, const float* w, const float* b, float* y,
                        int B, int in_features, int out_features) {
    // TODO: [Teammate - Sequential] Implement fully connected forward: y = x * W^T + b
    (void)x; (void)w; (void)b; (void)y;
    (void)B; (void)in_features; (void)out_features;
}

void seq_linear_backward(const float* dy, const float* x, const float* w,
                         float* dx, float* dw, float* db,
                         int B, int in_features, int out_features) {
    // TODO: [Teammate - Sequential] Implement fully connected backward:
    // dx = dy * W
    // dw = dy^T * x
    // db = sum(dy, axis=0)
    (void)dy; (void)x; (void)w; (void)dx; (void)dw; (void)db;
    (void)B; (void)in_features; (void)out_features;
}

void seq_softmax_cross_entropy_forward_backward(const float* logits, const int* labels,
                                                float* probs, float* dlogits,
                                                float* batch_loss, int B, int num_classes) {
    // TODO: [Teammate - Sequential] Implement stable Softmax + Cross-Entropy Loss & dlogits = (p - 1{y==c})/B
    (void)logits; (void)labels; (void)probs; (void)dlogits; (void)batch_loss;
    (void)B; (void)num_classes;
}

void seq_sgd_update(float* w, float* dw, float* vw,
                    float* b, float* db, float* vb,
                    size_t w_size, size_t b_size,
                    float lr, float momentum, float weight_decay) {
    // TODO: [Teammate - Sequential] Implement SGD with momentum & weight decay
    (void)w; (void)dw; (void)vw; (void)b; (void)db; (void)vb;
    (void)w_size; (void)b_size; (void)lr; (void)momentum; (void)weight_decay;
}

} // namespace cutin

