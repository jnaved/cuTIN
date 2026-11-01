#ifndef CUTIN_NETWORK_HPP
#define CUTIN_NETWORK_HPP

#include <vector>
#include <string>
#include <cstddef>

namespace cutin {

// Forward declaration to decouple network headers from dataloader headers
class TinyImageNetDataset;

// Architectural Constants
struct ArchConfig {
    static constexpr int BATCH_SIZE = 64;
    static constexpr int IN_C = 3;
    static constexpr int IN_H = 64;
    static constexpr int IN_W = 64;

    // Conv1: 3 -> 16, 3x3, pad=1, stride=1 => (16, 64, 64)
    static constexpr int C1_OUT_C = 16;
    static constexpr int C1_K = 3;
    static constexpr int C1_PAD = 1;
    static constexpr int C1_STRIDE = 1;
    static constexpr int C1_OUT_H = 64;
    static constexpr int C1_OUT_W = 64;
    static constexpr size_t C1_W_SIZE = C1_OUT_C * IN_C * C1_K * C1_K; // 16*3*3*3 = 432
    static constexpr size_t C1_B_SIZE = C1_OUT_C;                      // 16

    // Pool1: 2x2, stride=2 => (16, 32, 32)
    static constexpr int P1_K = 2;
    static constexpr int P1_STRIDE = 2;
    static constexpr int P1_OUT_H = 32;
    static constexpr int P1_OUT_W = 32;

    // Conv2: 16 -> 32, 3x3, pad=1, stride=1 => (32, 32, 32)
    static constexpr int C2_OUT_C = 32;
    static constexpr int C2_K = 3;
    static constexpr int C2_PAD = 1;
    static constexpr int C2_STRIDE = 1;
    static constexpr int C2_OUT_H = 32;
    static constexpr int C2_OUT_W = 32;
    static constexpr size_t C2_W_SIZE = C2_OUT_C * C1_OUT_C * C2_K * C2_K; // 32*16*3*3 = 4608
    static constexpr size_t C2_B_SIZE = C2_OUT_C;                          // 32

    // Pool2: 2x2, stride=2 => (32, 16, 16)
    static constexpr int P2_K = 2;
    static constexpr int P2_STRIDE = 2;
    static constexpr int P2_OUT_H = 16;
    static constexpr int P2_OUT_W = 16;

    // Flatten: 32 * 16 * 16 = 8192
    static constexpr int FLATTEN_SIZE = C2_OUT_C * P2_OUT_H * P2_OUT_W; // 8192

    // FC1: 8192 -> 512
    static constexpr int FC1_OUT = 512;
    static constexpr size_t FC1_W_SIZE = static_cast<size_t>(FC1_OUT) * FLATTEN_SIZE; // 512 * 8192 = 4,194,304
    static constexpr size_t FC1_B_SIZE = FC1_OUT;                                     // 512

    // FC2: 512 -> 200
    static constexpr int FC2_OUT = 200;
    static constexpr size_t FC2_W_SIZE = static_cast<size_t>(FC2_OUT) * FC1_OUT; // 200 * 512 = 102,400
    static constexpr size_t FC2_B_SIZE = FC2_OUT;                                 // 200

    static constexpr size_t TOTAL_WEIGHTS = C1_W_SIZE + C1_B_SIZE +
                                            C2_W_SIZE + C2_B_SIZE +
                                            FC1_W_SIZE + FC1_B_SIZE +
                                            FC2_W_SIZE + FC2_B_SIZE;
};

// Hyperparameters
struct TrainingConfig {
    int epochs = 5;
    int batch_size = 64;
    float learning_rate = 0.001f;
    float momentum = 0.9f;
    float weight_decay = 1e-4f;
    std::string data_dir = "tiny-imagenet-200";
    int max_train_samples = -1; // -1 for all
    int max_val_samples = -1;   // -1 for all
};

// Weights and biases container (CPU host or GPU device)
struct ModelParameters {
    // Conv1
    float* w_c1 = nullptr;
    float* b_c1 = nullptr;
    float* dw_c1 = nullptr;
    float* db_c1 = nullptr;
    float* vw_c1 = nullptr; // momentum buffer
    float* vb_c1 = nullptr;

    // Conv2
    float* w_c2 = nullptr;
    float* b_c2 = nullptr;
    float* dw_c2 = nullptr;
    float* db_c2 = nullptr;
    float* vw_c2 = nullptr;
    float* vb_c2 = nullptr;

    // FC1
    float* w_fc1 = nullptr;
    float* b_fc1 = nullptr;
    float* dw_fc1 = nullptr;
    float* db_fc1 = nullptr;
    float* vw_fc1 = nullptr;
    float* vb_fc1 = nullptr;

    // FC2
    float* w_fc2 = nullptr;
    float* b_fc2 = nullptr;
    float* dw_fc2 = nullptr;
    float* db_fc2 = nullptr;
    float* vw_fc2 = nullptr;
    float* vb_fc2 = nullptr;

    void allocate_host();
    void free_host();
    void init_weights(unsigned int seed = 42);
};

// Execution Mode
enum class ExecutionMode {
    SEQUENTIAL,
    OPENMP,
    CUDA
};

// =========================================================================
// CUDA Implementation Declarations (Completed by CUDA Member)
// =========================================================================
struct CudaDeviceBuffers {
    int batch_size = 0;

    // Input & Activations
    float* d_input = nullptr;        // (B, 3, 64, 64)
    int*   d_labels = nullptr;       // (B)
    float* d_c1_out = nullptr;       // (B, 16, 64, 64)
    float* d_p1_out = nullptr;       // (B, 16, 32, 32)
    int*   d_p1_mask = nullptr;      // (B, 16, 32, 32)
    float* d_c2_out = nullptr;       // (B, 32, 32, 32)
    float* d_p2_out = nullptr;       // (B, 32, 16, 16)
    int*   d_p2_mask = nullptr;      // (B, 32, 16, 16)
    float* d_fc1_out = nullptr;      // (B, 512)
    float* d_fc1_relu = nullptr;     // (B, 512)
    float* d_fc2_out = nullptr;      // (B, 200)
    float* d_probs = nullptr;        // (B, 200)
    float* d_batch_loss = nullptr;   // (B)

    // Gradients
    float* d_dlogits = nullptr;      // (B, 200)
    float* d_dfc1_relu = nullptr;    // (B, 512)
    float* d_dfc1_out = nullptr;     // (B, 512)
    float* d_dp2_out = nullptr;      // (B, 32, 16, 16)
    float* d_dc2_out = nullptr;      // (B, 32, 32, 32)
    float* d_dp1_out = nullptr;      // (B, 16, 32, 32)
    float* d_dc1_out = nullptr;      // (B, 16, 64, 64)
    float* d_dinput = nullptr;       // (B, 3, 64, 64)

    void allocate(int b_size);
    void free();
};

void cuda_allocate_model_parameters(ModelParameters& params);
void cuda_free_model_parameters(ModelParameters& params);
void cuda_copy_parameters_to_device(ModelParameters& d_params, const ModelParameters& h_params);
void cuda_copy_parameters_to_host(ModelParameters& h_params, const ModelParameters& d_params);

// CUDA Layer Forward & Backward Primitives
void cuda_conv2d_forward(const float* d_x, const float* d_w, const float* d_b, float* d_y,
                         int B, int C_in, int H_in, int W_in,
                         int C_out, int K, int pad, int stride);

void cuda_conv2d_backward(const float* d_dy, const float* d_x, const float* d_w,
                          float* d_dx, float* d_dw, float* d_db,
                          int B, int C_in, int H_in, int W_in,
                          int C_out, int K, int pad, int stride);

void cuda_relu_forward(const float* d_x, float* d_y, size_t size);
void cuda_relu_backward(const float* d_dy, const float* d_y, float* d_dx, size_t size);

void cuda_maxpool2d_forward(const float* d_x, float* d_y, int* d_mask,
                            int B, int C, int H_in, int W_in, int K, int stride);

void cuda_maxpool2d_backward(const float* d_dy, const int* d_mask, float* d_dx,
                             int B, int C, int H_out, int W_out, int H_in, int W_in, int K, int stride);

void cuda_linear_forward(const float* d_x, const float* d_w, const float* d_b, float* d_y,
                         int B, int in_features, int out_features);

void cuda_linear_backward(const float* d_dy, const float* d_x, const float* d_w,
                          float* d_dx, float* d_dw, float* d_db,
                          int B, int in_features, int out_features);

void cuda_softmax_cross_entropy_forward_backward(const float* d_logits, const int* d_labels,
                                                 float* d_probs, float* d_dlogits,
                                                 float* d_batch_loss, int B, int num_classes);

void cuda_sgd_update(float* d_w, float* d_dw, float* d_vw,
                     float* d_b, float* d_db, float* d_vb,
                     size_t w_size, size_t b_size,
                     float lr, float momentum, float weight_decay);

// High-level CUDA Network execution
void cuda_train(const TinyImageNetDataset& train_data,
                const TinyImageNetDataset& val_data,
                ModelParameters& params,
                const TrainingConfig& config);

float cuda_evaluate(const TinyImageNetDataset& val_data,
                    const ModelParameters& d_params,
                    int batch_size);

// =========================================================================
// Sequential Implementation Declarations (Skeleton for Member - Sequential)
// =========================================================================
void seq_conv2d_forward(const float* x, const float* w, const float* b, float* y,
                        int B, int C_in, int H_in, int W_in,
                        int C_out, int K, int pad, int stride);

void seq_conv2d_backward(const float* dy, const float* x, const float* w,
                         float* dx, float* dw, float* db,
                         int B, int C_in, int H_in, int W_in,
                         int C_out, int K, int pad, int stride);

void seq_relu_forward(const float* x, float* y, size_t size);
void seq_relu_backward(const float* dy, const float* y, float* dx, size_t size);

void seq_maxpool2d_forward(const float* x, float* y, int* mask,
                           int B, int C, int H_in, int W_in, int K, int stride);

void seq_maxpool2d_backward(const float* dy, const int* mask, float* dx,
                            int B, int C, int H_out, int W_out, int H_in, int W_in, int K, int stride);

void seq_linear_forward(const float* x, const float* w, const float* b, float* y,
                        int B, int in_features, int out_features);

void seq_linear_backward(const float* dy, const float* x, const float* w,
                         float* dx, float* dw, float* db,
                         int B, int in_features, int out_features);

void seq_softmax_cross_entropy_forward_backward(const float* logits, const int* labels,
                                                float* probs, float* dlogits,
                                                float* batch_loss, int B, int num_classes);

void seq_sgd_update(float* w, float* dw, float* vw,
                    float* b, float* db, float* vb,
                    size_t w_size, size_t b_size,
                    float lr, float momentum, float weight_decay);

void seq_train(const TinyImageNetDataset& train_data,
               const TinyImageNetDataset& val_data,
               ModelParameters& params,
               const TrainingConfig& config);

float seq_evaluate(const TinyImageNetDataset& val_data,
                   const ModelParameters& params,
                   int batch_size);

// =========================================================================
// OpenMP Implementation Declarations (Skeleton for Member - OpenMP)
// =========================================================================
void omp_conv2d_forward(const float* x, const float* w, const float* b, float* y,
                        int B, int C_in, int H_in, int W_in,
                        int C_out, int K, int pad, int stride);

void omp_conv2d_backward(const float* dy, const float* x, const float* w,
                         float* dx, float* dw, float* db,
                         int B, int C_in, int H_in, int W_in,
                         int C_out, int K, int pad, int stride);

void omp_relu_forward(const float* x, float* y, size_t size);
void omp_relu_backward(const float* dy, const float* y, float* dx, size_t size);

void omp_maxpool2d_forward(const float* x, float* y, int* mask,
                           int B, int C, int H_in, int W_in, int K, int stride);

void omp_maxpool2d_backward(const float* dy, const int* mask, float* dx,
                            int B, int C, int H_out, int W_out, int H_in, int W_in, int K, int stride);

void omp_linear_forward(const float* x, const float* w, const float* b, float* y,
                        int B, int in_features, int out_features);

void omp_linear_backward(const float* dy, const float* x, const float* w,
                         float* dx, float* dw, float* db,
                         int B, int in_features, int out_features);

void omp_softmax_cross_entropy_forward_backward(const float* logits, const int* labels,
                                                float* probs, float* dlogits,
                                                float* batch_loss, int B, int num_classes);

void omp_sgd_update(float* w, float* dw, float* vw,
                    float* b, float* db, float* vb,
                    size_t w_size, size_t b_size,
                    float lr, float momentum, float weight_decay);

void omp_train(const TinyImageNetDataset& train_data,
               const TinyImageNetDataset& val_data,
               ModelParameters& params,
               const TrainingConfig& config);

float omp_evaluate(const TinyImageNetDataset& val_data,
                   const ModelParameters& params,
                   int batch_size);

} // namespace cutin

#endif // CUTIN_NETWORK_HPP
