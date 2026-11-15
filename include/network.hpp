#ifndef CUTIN_NETWORK_HPP
#define CUTIN_NETWORK_HPP

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iomanip>

#ifndef CUDA_CHECK
#include <cuda_runtime.h>
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error at %s:%d - %s\n",                   \
                         __FILE__, __LINE__, cudaGetErrorString(err));           \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)
#endif

namespace cutin {

// =========================================================================
// 1. ARCHITECTURE CONFIGURATION (Matches src/main.cpp Graph)
// =========================================================================
struct ArchConfig {
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
    static constexpr size_t C1_W_SIZE = C1_OUT_C * IN_C * C1_K * C1_K; // 432
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
    static constexpr size_t C2_W_SIZE = C2_OUT_C * C1_OUT_C * C2_K * C2_K; // 4,608
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
    static constexpr size_t FC1_W_SIZE = static_cast<size_t>(FC1_OUT) * FLATTEN_SIZE; // 4,194,304
    static constexpr size_t FC1_B_SIZE = FC1_OUT;                                     // 512

    // FC2: 512 -> 200
    static constexpr int FC2_OUT = 200;
    static constexpr size_t FC2_W_SIZE = static_cast<size_t>(FC2_OUT) * FC1_OUT; // 102,400
    static constexpr size_t FC2_B_SIZE = FC2_OUT;                                 // 200

    static constexpr size_t TOTAL_WEIGHTS = C1_W_SIZE + C1_B_SIZE +
                                            C2_W_SIZE + C2_B_SIZE +
                                            FC1_W_SIZE + FC1_B_SIZE +
                                            FC2_W_SIZE + FC2_B_SIZE; // 4,302,504 floats (~16.4 MB)
};

// Hyperparameters
struct TrainingConfig {
    int epochs = 5;
    int batch_size = 64;
    float learning_rate = 0.001f;
    float momentum = 0.9f;
    float weight_decay = 1e-4f;
    std::string data_dir = "tiny-imagenet-200";
    int max_train_samples = -1;
    int max_val_samples = -1;
};

// Execution backend selector
enum class ExecutionMode {
    CUDA,
    OPENMP,
    SEQUENTIAL,
    ALL
};

// Model Weights and Biases Container
struct ModelParameters {
    float* w_c1 = nullptr;  float* b_c1 = nullptr;
    float* dw_c1 = nullptr; float* db_c1 = nullptr;
    float* vw_c1 = nullptr; float* vb_c1 = nullptr;

    float* w_c2 = nullptr;  float* b_c2 = nullptr;
    float* dw_c2 = nullptr; float* db_c2 = nullptr;
    float* vw_c2 = nullptr; float* vb_c2 = nullptr;

    float* w_fc1 = nullptr; float* b_fc1 = nullptr;
    float* dw_fc1 = nullptr; float* db_fc1 = nullptr;
    float* vw_fc1 = nullptr; float* vb_fc1 = nullptr;

    float* w_fc2 = nullptr; float* b_fc2 = nullptr;
    float* dw_fc2 = nullptr; float* db_fc2 = nullptr;
    float* vw_fc2 = nullptr; float* vb_fc2 = nullptr;

    void allocate_host();
    void free_host();
    void init_weights(unsigned int seed = 42);
};

// High-resolution CPU Timer
class CPUTimer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double stop() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start_;
        return elapsed.count(); // milliseconds
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// Numerical comparison helper for tests
inline bool check_close(const float* a, const float* b, size_t size, float atol = 1e-4f, float rtol = 1e-3f) {
    size_t errors = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float diff = std::fabs(a[i] - b[i]);
        float tol = atol + rtol * std::fabs(b[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tol) errors++;
    }
    if (errors > 0) {
        std::cerr << "Mismatch: " << errors << "/" << size << " (Max diff = " << max_diff << ")\n";
        return false;
    }
    return true;
}

// =========================================================================
// 2. DATASET LOADER INTERFACE
// =========================================================================
struct ImageSample {
    std::string filepath;
    int label; // 0 to 199
};

class TinyImageNetDataset {
public:
    TinyImageNetDataset() = default;
    bool load(const std::string& data_root, bool is_train = true, int max_samples = -1);
    size_t size() const { return samples_.size(); }
    void shuffle(unsigned int seed = 42);
    int get_batch(size_t start_idx, int batch_size, float* out_images, int* out_labels) const;

    static constexpr int CHANNELS = 3;
    static constexpr int HEIGHT = 64;
    static constexpr int WIDTH = 64;
    static constexpr int NUM_CLASSES = 200;
    static constexpr int IMAGE_SIZE = CHANNELS * HEIGHT * WIDTH; // 12288

private:
    std::vector<ImageSample> samples_;
    std::vector<std::string> wnids_;
    std::unordered_map<std::string, int> wnid_to_idx_;
    bool load_class_metadata(const std::string& data_root);
};

// =========================================================================
// 3. BACKEND API DECLARATIONS (1-to-1 Mapping across Member Implementations)
// =========================================================================

// --- Sequential (Member 1 - src/sequential.cpp) ---
void seq_relu_forward(const float* x, float* y, size_t size);
void seq_relu_backward(const float* dy, const float* y, float* dx, size_t size);
void seq_maxpool2d_forward(const float* x, float* y, int* mask, int B, int C, int H_in, int W_in, int K, int stride);
void seq_maxpool2d_backward(const float* dy, const int* mask, float* dx, int B, int C, int H_out, int W_out, int H_in, int W_in, int K, int stride);
void seq_conv2d_forward(const float* x, const float* w, const float* b, float* y, int B, int C_in, int H_in, int W_in, int C_out, int K, int pad, int stride);
void seq_conv2d_backward(const float* dy, const float* x, const float* w, float* dx, float* dw, float* db, int B, int C_in, int H_in, int W_in, int C_out, int K, int pad, int stride);
void seq_linear_forward(const float* x, const float* w, const float* b, float* y, int B, int in_f, int out_f);
void seq_linear_backward(const float* dy, const float* x, const float* w, float* dx, float* dw, float* db, int B, int in_f, int out_f);
void seq_softmax_cross_entropy_forward_backward(const float* logits, const int* labels, float* probs, float* dlogits, float* batch_loss, int B, int num_classes);
void seq_sgd_update(float* w, float* dw, float* vw, float* b, float* db, float* vb, size_t w_size, size_t b_size, float lr, float momentum, float weight_decay);
void seq_train(const TinyImageNetDataset& train_data, const TinyImageNetDataset& val_data, ModelParameters& params, const TrainingConfig& config);
float seq_evaluate(const TinyImageNetDataset& val_data, const ModelParameters& params, int batch_size);

// --- OpenMP (Member 2 - src/openmp.cpp) ---
void omp_relu_forward(const float* x, float* y, size_t size);
void omp_relu_backward(const float* dy, const float* y, float* dx, size_t size);
void omp_maxpool2d_forward(const float* x, float* y, int* mask, int B, int C, int H_in, int W_in, int K, int stride);
void omp_maxpool2d_backward(const float* dy, const int* mask, float* dx, int B, int C, int H_out, int W_out, int H_in, int W_in, int K, int stride);
void omp_conv2d_forward(const float* x, const float* w, const float* b, float* y, int B, int C_in, int H_in, int W_in, int C_out, int K, int pad, int stride);
void omp_conv2d_backward(const float* dy, const float* x, const float* w, float* dx, float* dw, float* db, int B, int C_in, int H_in, int W_in, int C_out, int K, int pad, int stride);
void omp_linear_forward(const float* x, const float* w, const float* b, float* y, int B, int in_f, int out_f);
void omp_linear_backward(const float* dy, const float* x, const float* w, float* dx, float* dw, float* db, int B, int in_f, int out_f);
void omp_softmax_cross_entropy_forward_backward(const float* logits, const int* labels, float* probs, float* dlogits, float* batch_loss, int B, int num_classes);
void omp_sgd_update(float* w, float* dw, float* vw, float* b, float* db, float* vb, size_t w_size, size_t b_size, float lr, float momentum, float weight_decay);
void omp_train(const TinyImageNetDataset& train_data, const TinyImageNetDataset& val_data, ModelParameters& params, const TrainingConfig& config);
float omp_evaluate(const TinyImageNetDataset& val_data, const ModelParameters& params, int batch_size);

// --- CUDA (Member 3 - src/cuda.cu) ---
void cuda_relu_forward(const float* d_x, float* d_y, size_t size);
void cuda_relu_backward(const float* d_dy, const float* d_y, float* d_dx, size_t size);
void cuda_maxpool2d_forward(const float* d_x, float* d_y, int* d_mask, int B, int C, int H_in, int W_in, int K, int stride);
void cuda_maxpool2d_backward(const float* d_dy, const int* d_mask, float* d_dx, int B, int C, int H_out, int W_out, int H_in, int W_in, int K, int stride);
void cuda_conv2d_forward(const float* d_x, const float* d_w, const float* d_b, float* d_y, int B, int C_in, int H_in, int W_in, int C_out, int K, int pad, int stride);
void cuda_conv2d_backward(const float* d_dy, const float* d_x, const float* d_w, float* d_dx, float* d_dw, float* d_db, int B, int C_in, int H_in, int W_in, int C_out, int K, int pad, int stride);
void cuda_linear_forward(const float* d_x, const float* d_w, const float* d_b, float* d_y, int B, int in_f, int out_f);
void cuda_linear_backward(const float* d_dy, const float* d_x, const float* d_w, float* d_dx, float* d_dw, float* d_db, int B, int in_f, int out_f);
void cuda_softmax_cross_entropy_forward_backward(const float* d_logits, const int* d_labels, float* d_probs, float* d_dlogits, float* d_batch_loss, int B, int num_classes);
void cuda_sgd_update(float* d_w, float* d_dw, float* d_vw, float* d_b, float* d_db, float* d_vb, size_t w_size, size_t b_size, float lr, float momentum, float weight_decay);
void cuda_train(const TinyImageNetDataset& train_data, const TinyImageNetDataset& val_data, ModelParameters& params, const TrainingConfig& config);
float cuda_evaluate(const TinyImageNetDataset& val_data, const ModelParameters& params, int batch_size);

} // namespace cutin

#endif // CUTIN_NETWORK_HPP
