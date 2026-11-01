#include "network.hpp"
#include "dataloader.hpp"
#include "utils.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>

namespace cutin {

void CudaDeviceBuffers::allocate(int b_size) {
    batch_size = b_size;

    size_t in_size   = static_cast<size_t>(b_size) * ArchConfig::IN_C * ArchConfig::IN_H * ArchConfig::IN_W;
    size_t c1_size   = static_cast<size_t>(b_size) * ArchConfig::C1_OUT_C * ArchConfig::C1_OUT_H * ArchConfig::C1_OUT_W;
    size_t p1_size   = static_cast<size_t>(b_size) * ArchConfig::C1_OUT_C * ArchConfig::P1_OUT_H * ArchConfig::P1_OUT_W;
    size_t c2_size   = static_cast<size_t>(b_size) * ArchConfig::C2_OUT_C * ArchConfig::C2_OUT_H * ArchConfig::C2_OUT_W;
    size_t p2_size   = static_cast<size_t>(b_size) * ArchConfig::FLATTEN_SIZE;
    size_t fc1_size  = static_cast<size_t>(b_size) * ArchConfig::FC1_OUT;
    size_t fc2_size  = static_cast<size_t>(b_size) * ArchConfig::FC2_OUT;

    CUDA_CHECK(cudaMalloc(&d_input, in_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, b_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_c1_out, c1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p1_out, p1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p1_mask, p1_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_c2_out, c2_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p2_out, p2_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p2_mask, p2_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_fc1_out, fc1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc1_relu, fc1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc2_out, fc2_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_probs, fc2_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_loss, b_size * sizeof(float)));

    // Gradients
    CUDA_CHECK(cudaMalloc(&d_dlogits, fc2_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dfc1_relu, fc1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dfc1_out, fc1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dp2_out, p2_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dc2_out, c2_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dp1_out, p1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dc1_out, c1_size * sizeof(float)));
}

void CudaDeviceBuffers::free() {
    cudaFree(d_input);
    cudaFree(d_labels);
    cudaFree(d_c1_out);
    cudaFree(d_p1_out);
    cudaFree(d_p1_mask);
    cudaFree(d_c2_out);
    cudaFree(d_p2_out);
    cudaFree(d_p2_mask);
    cudaFree(d_fc1_out);
    cudaFree(d_fc1_relu);
    cudaFree(d_fc2_out);
    cudaFree(d_probs);
    cudaFree(d_batch_loss);

    cudaFree(d_dlogits);
    cudaFree(d_dfc1_relu);
    cudaFree(d_dfc1_out);
    cudaFree(d_dp2_out);
    cudaFree(d_dc2_out);
    cudaFree(d_dp1_out);
    cudaFree(d_dc1_out);
}

void cuda_allocate_model_parameters(ModelParameters& params) {
    CUDA_CHECK(cudaMalloc(&params.w_c1, ArchConfig::C1_W_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params.b_c1, ArchConfig::C1_B_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params.dw_c1, ArchConfig::C1_W_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params.db_c1, ArchConfig::C1_B_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params.vw_c1, ArchConfig::C1_W_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params.vb_c1, ArchConfig::C1_B_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(params.vw_c1, 0, ArchConfig::C1_W_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(params.vb_c1, 0, ArchConfig::C1_B_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&params.w_c2, ArchConfig::C2_W_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params.b_c2, ArchConfig::C2_B_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params.dw_c2, ArchConfig::C2_W_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params.db_c2, ArchConfig::C2_B_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params.vw_c2, ArchConfig::C2_W_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params.vb_c2, ArchConfig::C2_B_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(params.vw_c2, 0, ArchConfig::C2_W_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(params.vb_c2, 0, ArchConfig::C2_B_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&params.w_fc1, ArchConfig::FC1_W_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params.b_fc1, ArchConfig::FC1_B_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params.dw_fc1, ArchConfig::FC1_W_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params.db_fc1, ArchConfig::FC1_B_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params.vw_fc1, ArchConfig::FC1_W_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params.vb_fc1, ArchConfig::FC1_B_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(params.vw_fc1, 0, ArchConfig::FC1_W_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(params.vb_fc1, 0, ArchConfig::FC1_B_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&params.w_fc2, ArchConfig::FC2_W_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params.b_fc2, ArchConfig::FC2_B_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params.dw_fc2, ArchConfig::FC2_W_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params.db_fc2, ArchConfig::FC2_B_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params.vw_fc2, ArchConfig::FC2_W_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params.vb_fc2, ArchConfig::FC2_B_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(params.vw_fc2, 0, ArchConfig::FC2_W_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(params.vb_fc2, 0, ArchConfig::FC2_B_SIZE * sizeof(float)));
}

void cuda_free_model_parameters(ModelParameters& params) {
    cudaFree(params.w_c1);  cudaFree(params.b_c1);  cudaFree(params.dw_c1);  cudaFree(params.db_c1);  cudaFree(params.vw_c1);  cudaFree(params.vb_c1);
    cudaFree(params.w_c2);  cudaFree(params.b_c2);  cudaFree(params.dw_c2);  cudaFree(params.db_c2);  cudaFree(params.vw_c2);  cudaFree(params.vb_c2);
    cudaFree(params.w_fc1); cudaFree(params.b_fc1); cudaFree(params.dw_fc1); cudaFree(params.db_fc1); cudaFree(params.vw_fc1); cudaFree(params.vb_fc1);
    cudaFree(params.w_fc2); cudaFree(params.b_fc2); cudaFree(params.dw_fc2); cudaFree(params.db_fc2); cudaFree(params.vw_fc2); cudaFree(params.vb_fc2);
}

void cuda_copy_parameters_to_device(ModelParameters& d_params, const ModelParameters& h_params) {
    CUDA_CHECK(cudaMemcpy(d_params.w_c1, h_params.w_c1, ArchConfig::C1_W_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_params.b_c1, h_params.b_c1, ArchConfig::C1_B_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_params.w_c2, h_params.w_c2, ArchConfig::C2_W_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_params.b_c2, h_params.b_c2, ArchConfig::C2_B_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_params.w_fc1, h_params.w_fc1, ArchConfig::FC1_W_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_params.b_fc1, h_params.b_fc1, ArchConfig::FC1_B_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_params.w_fc2, h_params.w_fc2, ArchConfig::FC2_W_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_params.b_fc2, h_params.b_fc2, ArchConfig::FC2_B_SIZE * sizeof(float), cudaMemcpyHostToDevice));
}

void cuda_copy_parameters_to_host(ModelParameters& h_params, const ModelParameters& d_params) {
    CUDA_CHECK(cudaMemcpy(h_params.w_c1, d_params.w_c1, ArchConfig::C1_W_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params.b_c1, d_params.b_c1, ArchConfig::C1_B_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(h_params.w_c2, d_params.w_c2, ArchConfig::C2_W_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params.b_c2, d_params.b_c2, ArchConfig::C2_B_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(h_params.w_fc1, d_params.w_fc1, ArchConfig::FC1_W_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params.b_fc1, d_params.b_fc1, ArchConfig::FC1_B_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(h_params.w_fc2, d_params.w_fc2, ArchConfig::FC2_W_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params.b_fc2, d_params.b_fc2, ArchConfig::FC2_B_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
}

float cuda_evaluate(const TinyImageNetDataset& val_data,
                    const ModelParameters& d_params,
                    int batch_size) {
    if (val_data.size() == 0) return 0.0f;

    CudaDeviceBuffers buffers;
    buffers.allocate(batch_size);

    std::vector<float> h_images(static_cast<size_t>(batch_size) * TinyImageNetDataset::IMAGE_SIZE);
    std::vector<int>   h_labels(batch_size);
    std::vector<float> h_probs(static_cast<size_t>(batch_size) * ArchConfig::FC2_OUT);

    int correct = 0;
    int total_evaluated = 0;

    for (size_t start = 0; start < val_data.size(); start += batch_size) {
        int actual = val_data.get_batch(start, batch_size, h_images.data(), h_labels.data());
        if (actual == 0) break;

        CUDA_CHECK(cudaMemcpy(buffers.d_input, h_images.data(),
                              static_cast<size_t>(batch_size) * TinyImageNetDataset::IMAGE_SIZE * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Forward Pass
        // 1. Conv1 + ReLU1
        cuda_conv2d_forward(buffers.d_input, d_params.w_c1, d_params.b_c1, buffers.d_c1_out,
                            batch_size, ArchConfig::IN_C, ArchConfig::IN_H, ArchConfig::IN_W,
                            ArchConfig::C1_OUT_C, ArchConfig::C1_K, ArchConfig::C1_PAD, ArchConfig::C1_STRIDE);
        cuda_relu_forward(buffers.d_c1_out, buffers.d_c1_out,
                          static_cast<size_t>(batch_size) * ArchConfig::C1_OUT_C * ArchConfig::C1_OUT_H * ArchConfig::C1_OUT_W);

        // 2. MaxPool1
        cuda_maxpool2d_forward(buffers.d_c1_out, buffers.d_p1_out, buffers.d_p1_mask,
                               batch_size, ArchConfig::C1_OUT_C, ArchConfig::C1_OUT_H, ArchConfig::C1_OUT_W,
                               ArchConfig::P1_K, ArchConfig::P1_STRIDE);

        // 3. Conv2 + ReLU2
        cuda_conv2d_forward(buffers.d_p1_out, d_params.w_c2, d_params.b_c2, buffers.d_c2_out,
                            batch_size, ArchConfig::C1_OUT_C, ArchConfig::P1_OUT_H, ArchConfig::P1_OUT_W,
                            ArchConfig::C2_OUT_C, ArchConfig::C2_K, ArchConfig::C2_PAD, ArchConfig::C2_STRIDE);
        cuda_relu_forward(buffers.d_c2_out, buffers.d_c2_out,
                          static_cast<size_t>(batch_size) * ArchConfig::C2_OUT_C * ArchConfig::C2_OUT_H * ArchConfig::C2_OUT_W);

        // 4. MaxPool2
        cuda_maxpool2d_forward(buffers.d_c2_out, buffers.d_p2_out, buffers.d_p2_mask,
                               batch_size, ArchConfig::C2_OUT_C, ArchConfig::C2_OUT_H, ArchConfig::C2_OUT_W,
                               ArchConfig::P2_K, ArchConfig::P2_STRIDE);

        // 5. FC1 + ReLU3
        cuda_linear_forward(buffers.d_p2_out, d_params.w_fc1, d_params.b_fc1, buffers.d_fc1_out,
                            batch_size, ArchConfig::FLATTEN_SIZE, ArchConfig::FC1_OUT);
        cuda_relu_forward(buffers.d_fc1_out, buffers.d_fc1_relu,
                          static_cast<size_t>(batch_size) * ArchConfig::FC1_OUT);

        // 6. FC2
        cuda_linear_forward(buffers.d_fc1_relu, d_params.w_fc2, d_params.b_fc2, buffers.d_fc2_out,
                            batch_size, ArchConfig::FC1_OUT, ArchConfig::FC2_OUT);

        // Softmax probabilities
        cuda_softmax_cross_entropy_forward_backward(buffers.d_fc2_out, buffers.d_labels,
                                                     buffers.d_probs, buffers.d_dlogits,
                                                     nullptr, batch_size, ArchConfig::FC2_OUT);

        CUDA_CHECK(cudaMemcpy(h_probs.data(), buffers.d_probs,
                              static_cast<size_t>(batch_size) * ArchConfig::FC2_OUT * sizeof(float),
                              cudaMemcpyDeviceToHost));

        for (int b = 0; b < actual; ++b) {
            float max_p = -1.0f;
            int pred = 0;
            for (int c = 0; c < ArchConfig::FC2_OUT; ++c) {
                float p = h_probs[b * ArchConfig::FC2_OUT + c];
                if (p > max_p) {
                    max_p = p;
                    pred = c;
                }
            }
            if (pred == h_labels[b]) correct++;
            total_evaluated++;
        }
    }

    buffers.free();
    return total_evaluated > 0 ? (100.0f * static_cast<float>(correct) / static_cast<float>(total_evaluated)) : 0.0f;
}

void cuda_train(const TinyImageNetDataset& train_data,
                const TinyImageNetDataset& val_data,
                ModelParameters& h_params,
                const TrainingConfig& config) {
    std::cout << "\n=======================================================\n";
    std::cout << " Starting Deep CNN Training using CUDA C / C++\n";
    std::cout << " Architecture: 2 Conv + 2 MaxPool + 2 Linear + Softmax\n";
    std::cout << " Classes: " << ArchConfig::FC2_OUT << " | Batch Size: " << config.batch_size
              << " | Epochs: " << config.epochs << " | LR: " << config.learning_rate << "\n";
    std::cout << "=======================================================\n";

    ModelParameters d_params;
    cuda_allocate_model_parameters(d_params);
    cuda_copy_parameters_to_device(d_params, h_params);

    CudaDeviceBuffers buffers;
    buffers.allocate(config.batch_size);

    std::vector<float> h_images(static_cast<size_t>(config.batch_size) * TinyImageNetDataset::IMAGE_SIZE);
    std::vector<int>   h_labels(config.batch_size);
    std::vector<float> h_batch_loss(config.batch_size);
    std::vector<float> h_probs(static_cast<size_t>(config.batch_size) * ArchConfig::FC2_OUT);

    CPUTimer epoch_timer;
    CPUTimer batch_timer;

    size_t num_train = train_data.size();
    size_t total_batches = (num_train + config.batch_size - 1) / config.batch_size;

    for (int epoch = 1; epoch <= config.epochs; ++epoch) {
        epoch_timer.start();
        float epoch_loss = 0.0f;
        int epoch_correct = 0;
        int total_samples_processed = 0;

        // Shuffle dataset every epoch
        const_cast<TinyImageNetDataset&>(train_data).shuffle(epoch * 1000 + 42);

        for (size_t b_idx = 0; b_idx < total_batches; ++b_idx) {
            batch_timer.start();
            size_t start = b_idx * config.batch_size;
            int actual = train_data.get_batch(start, config.batch_size, h_images.data(), h_labels.data());
            if (actual == 0) break;

            // Copy batch to GPU
            CUDA_CHECK(cudaMemcpy(buffers.d_input, h_images.data(),
                                  static_cast<size_t>(config.batch_size) * TinyImageNetDataset::IMAGE_SIZE * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(buffers.d_labels, h_labels.data(),
                                  config.batch_size * sizeof(int),
                                  cudaMemcpyHostToDevice));

            // -------------------------------------------------------------
            // FORWARD PASS
            // -------------------------------------------------------------
            // Conv1: (B, 3, 64, 64) -> (B, 16, 64, 64)
            cuda_conv2d_forward(buffers.d_input, d_params.w_c1, d_params.b_c1, buffers.d_c1_out,
                                config.batch_size, ArchConfig::IN_C, ArchConfig::IN_H, ArchConfig::IN_W,
                                ArchConfig::C1_OUT_C, ArchConfig::C1_K, ArchConfig::C1_PAD, ArchConfig::C1_STRIDE);
            // ReLU1
            cuda_relu_forward(buffers.d_c1_out, buffers.d_c1_out,
                              static_cast<size_t>(config.batch_size) * ArchConfig::C1_OUT_C * ArchConfig::C1_OUT_H * ArchConfig::C1_OUT_W);
            // MaxPool1: (B, 16, 64, 64) -> (B, 16, 32, 32)
            cuda_maxpool2d_forward(buffers.d_c1_out, buffers.d_p1_out, buffers.d_p1_mask,
                                   config.batch_size, ArchConfig::C1_OUT_C, ArchConfig::C1_OUT_H, ArchConfig::C1_OUT_W,
                                   ArchConfig::P1_K, ArchConfig::P1_STRIDE);

            // Conv2: (B, 16, 32, 32) -> (B, 32, 32, 32)
            cuda_conv2d_forward(buffers.d_p1_out, d_params.w_c2, d_params.b_c2, buffers.d_c2_out,
                                config.batch_size, ArchConfig::C1_OUT_C, ArchConfig::P1_OUT_H, ArchConfig::P1_OUT_W,
                                ArchConfig::C2_OUT_C, ArchConfig::C2_K, ArchConfig::C2_PAD, ArchConfig::C2_STRIDE);
            // ReLU2
            cuda_relu_forward(buffers.d_c2_out, buffers.d_c2_out,
                              static_cast<size_t>(config.batch_size) * ArchConfig::C2_OUT_C * ArchConfig::C2_OUT_H * ArchConfig::C2_OUT_W);
            // MaxPool2: (B, 32, 32, 32) -> (B, 32, 16, 16)
            cuda_maxpool2d_forward(buffers.d_c2_out, buffers.d_p2_out, buffers.d_p2_mask,
                                   config.batch_size, ArchConfig::C2_OUT_C, ArchConfig::C2_OUT_H, ArchConfig::C2_OUT_W,
                                   ArchConfig::P2_K, ArchConfig::P2_STRIDE);

            // FC1 (Flattened: 8192 -> 512)
            cuda_linear_forward(buffers.d_p2_out, d_params.w_fc1, d_params.b_fc1, buffers.d_fc1_out,
                                config.batch_size, ArchConfig::FLATTEN_SIZE, ArchConfig::FC1_OUT);
            // ReLU3
            cuda_relu_forward(buffers.d_fc1_out, buffers.d_fc1_relu,
                              static_cast<size_t>(config.batch_size) * ArchConfig::FC1_OUT);

            // FC2 (512 -> 200)
            cuda_linear_forward(buffers.d_fc1_relu, d_params.w_fc2, d_params.b_fc2, buffers.d_fc2_out,
                                config.batch_size, ArchConfig::FC1_OUT, ArchConfig::FC2_OUT);

            // Softmax + Categorical Cross-Entropy Loss
            cuda_softmax_cross_entropy_forward_backward(buffers.d_fc2_out, buffers.d_labels,
                                                         buffers.d_probs, buffers.d_dlogits,
                                                         buffers.d_batch_loss, config.batch_size, ArchConfig::FC2_OUT);

            // -------------------------------------------------------------
            // BACKWARD PASS
            // -------------------------------------------------------------
            // FC2 backward
            cuda_linear_backward(buffers.d_dlogits, buffers.d_fc1_relu, d_params.w_fc2,
                                buffers.d_dfc1_relu, d_params.dw_fc2, d_params.db_fc2,
                                config.batch_size, ArchConfig::FC1_OUT, ArchConfig::FC2_OUT);

            // ReLU3 backward
            cuda_relu_backward(buffers.d_dfc1_relu, buffers.d_fc1_relu, buffers.d_dfc1_out,
                               static_cast<size_t>(config.batch_size) * ArchConfig::FC1_OUT);

            // FC1 backward
            cuda_linear_backward(buffers.d_dfc1_out, buffers.d_p2_out, d_params.w_fc1,
                                buffers.d_dp2_out, d_params.dw_fc1, d_params.db_fc1,
                                config.batch_size, ArchConfig::FLATTEN_SIZE, ArchConfig::FC1_OUT);

            // MaxPool2 backward
            cuda_maxpool2d_backward(buffers.d_dp2_out, buffers.d_p2_mask, buffers.d_dc2_out,
                                    config.batch_size, ArchConfig::C2_OUT_C, ArchConfig::P2_OUT_H, ArchConfig::P2_OUT_W,
                                    ArchConfig::C2_OUT_H, ArchConfig::C2_OUT_W,
                                    ArchConfig::P2_K, ArchConfig::P2_STRIDE);

            // ReLU2 backward
            cuda_relu_backward(buffers.d_dc2_out, buffers.d_c2_out, buffers.d_dc2_out,
                               static_cast<size_t>(config.batch_size) * ArchConfig::C2_OUT_C * ArchConfig::C2_OUT_H * ArchConfig::C2_OUT_W);

            // Conv2 backward
            cuda_conv2d_backward(buffers.d_dc2_out, buffers.d_p1_out, d_params.w_c2,
                                 buffers.d_dp1_out, d_params.dw_c2, d_params.db_c2,
                                 config.batch_size, ArchConfig::C1_OUT_C, ArchConfig::P1_OUT_H, ArchConfig::P1_OUT_W,
                                 ArchConfig::C2_OUT_C, ArchConfig::C2_K, ArchConfig::C2_PAD, ArchConfig::C2_STRIDE);

            // MaxPool1 backward
            cuda_maxpool2d_backward(buffers.d_dp1_out, buffers.d_p1_mask, buffers.d_dc1_out,
                                    config.batch_size, ArchConfig::C1_OUT_C, ArchConfig::P1_OUT_H, ArchConfig::P1_OUT_W,
                                    ArchConfig::C1_OUT_H, ArchConfig::C1_OUT_W,
                                    ArchConfig::P1_K, ArchConfig::P1_STRIDE);

            // ReLU1 backward
            cuda_relu_backward(buffers.d_dc1_out, buffers.d_c1_out, buffers.d_dc1_out,
                               static_cast<size_t>(config.batch_size) * ArchConfig::C1_OUT_C * ArchConfig::C1_OUT_H * ArchConfig::C1_OUT_W);

            // Conv1 backward
            cuda_conv2d_backward(buffers.d_dc1_out, buffers.d_input, d_params.w_c1,
                                 nullptr, d_params.dw_c1, d_params.db_c1,
                                 config.batch_size, ArchConfig::IN_C, ArchConfig::IN_H, ArchConfig::IN_W,
                                 ArchConfig::C1_OUT_C, ArchConfig::C1_K, ArchConfig::C1_PAD, ArchConfig::C1_STRIDE);

            // -------------------------------------------------------------
            // OPTIMIZER STEP (SGD with Momentum)
            // -------------------------------------------------------------
            cuda_sgd_update(d_params.w_c1, d_params.dw_c1, d_params.vw_c1,
                            d_params.b_c1, d_params.db_c1, d_params.vb_c1,
                            ArchConfig::C1_W_SIZE, ArchConfig::C1_B_SIZE,
                            config.learning_rate, config.momentum, config.weight_decay);

            cuda_sgd_update(d_params.w_c2, d_params.dw_c2, d_params.vw_c2,
                            d_params.b_c2, d_params.db_c2, d_params.vb_c2,
                            ArchConfig::C2_W_SIZE, ArchConfig::C2_B_SIZE,
                            config.learning_rate, config.momentum, config.weight_decay);

            cuda_sgd_update(d_params.w_fc1, d_params.dw_fc1, d_params.vw_fc1,
                            d_params.b_fc1, d_params.db_fc1, d_params.vb_fc1,
                            ArchConfig::FC1_W_SIZE, ArchConfig::FC1_B_SIZE,
                            config.learning_rate, config.momentum, config.weight_decay);

            cuda_sgd_update(d_params.w_fc2, d_params.dw_fc2, d_params.vw_fc2,
                            d_params.b_fc2, d_params.db_fc2, d_params.vb_fc2,
                            ArchConfig::FC2_W_SIZE, ArchConfig::FC2_B_SIZE,
                            config.learning_rate, config.momentum, config.weight_decay);

            // Track Batch Metrics
            CUDA_CHECK(cudaMemcpy(h_batch_loss.data(), buffers.d_batch_loss,
                                  config.batch_size * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_probs.data(), buffers.d_probs,
                                  static_cast<size_t>(config.batch_size) * ArchConfig::FC2_OUT * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            float b_loss_sum = 0.0f;
            int b_correct = 0;
            for (int i = 0; i < actual; ++i) {
                b_loss_sum += h_batch_loss[i];
                float max_p = -1.0f;
                int pred = 0;
                for (int c = 0; c < ArchConfig::FC2_OUT; ++c) {
                    float p = h_probs[i * ArchConfig::FC2_OUT + c];
                    if (p > max_p) {
                        max_p = p;
                        pred = c;
                    }
                }
                if (pred == h_labels[i]) b_correct++;
            }

            epoch_loss += b_loss_sum;
            epoch_correct += b_correct;
            total_samples_processed += actual;

            if (b_idx % 10 == 0 || b_idx == total_batches - 1) {
                double b_time = batch_timer.stop();
                float cur_loss = b_loss_sum / static_cast<float>(actual);
                float cur_acc = 100.0f * static_cast<float>(b_correct) / static_cast<float>(actual);
                float throughput = (static_cast<float>(actual) / (static_cast<float>(b_time) / 1000.0f));

                std::cout << "\r[Epoch " << epoch << "/" << config.epochs << "] "
                          << "Batch " << (b_idx + 1) << "/" << total_batches
                          << " | Loss: " << std::fixed << std::setprecision(4) << cur_loss
                          << " | Acc: " << std::setprecision(2) << cur_acc << "%"
                          << " | " << std::setprecision(1) << throughput << " img/s"
                          << std::flush;
            }
        }

        double epoch_ms = epoch_timer.stop();
        float avg_loss = epoch_loss / static_cast<float>(total_samples_processed);
        float avg_acc = 100.0f * static_cast<float>(epoch_correct) / static_cast<float>(total_samples_processed);

        std::cout << "\n--> Epoch " << epoch << " Finished in "
                  << std::fixed << std::setprecision(2) << (epoch_ms / 1000.0) << "s"
                  << " | Train Loss: " << avg_loss
                  << " | Train Acc: " << avg_acc << "%\n";

        // Validation Step
        if (val_data.size() > 0) {
            float val_acc = cuda_evaluate(val_data, d_params, config.batch_size);
            std::cout << "--> Validation Accuracy: " << std::setprecision(2) << val_acc << "%\n";
        }
    }

    // Copy trained weights back to host
    cuda_copy_parameters_to_host(h_params, d_params);

    buffers.free();
    cuda_free_model_parameters(d_params);
    std::cout << "=======================================================\n";
    std::cout << " CUDA Training Completed Successfully!\n";
    std::cout << "=======================================================\n";
}

} // namespace cutin
