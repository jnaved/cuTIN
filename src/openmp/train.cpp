#include "network.hpp"
#include "dataloader.hpp"
#include "utils.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace cutin {
namespace {

struct OpenMPBuffers {
    int batch;
    std::vector<float> input, c1, p1, c2, p2, fc1, fc1_relu, fc2, probs, loss;
    std::vector<int> labels, p1_mask, p2_mask;
    std::vector<float> dlogits, dfc1_relu, dfc1, dp2, dc2, dp1, dc1;

    explicit OpenMPBuffers(int b) : batch(b),
        input(static_cast<size_t>(b) * ArchConfig::IN_C * ArchConfig::IN_H * ArchConfig::IN_W),
        c1(static_cast<size_t>(b) * ArchConfig::C1_OUT_C * ArchConfig::C1_OUT_H * ArchConfig::C1_OUT_W),
        p1(static_cast<size_t>(b) * ArchConfig::C1_OUT_C * ArchConfig::P1_OUT_H * ArchConfig::P1_OUT_W),
        c2(static_cast<size_t>(b) * ArchConfig::C2_OUT_C * ArchConfig::C2_OUT_H * ArchConfig::C2_OUT_W),
        p2(static_cast<size_t>(b) * ArchConfig::FLATTEN_SIZE),
        fc1(static_cast<size_t>(b) * ArchConfig::FC1_OUT),
        fc1_relu(static_cast<size_t>(b) * ArchConfig::FC1_OUT),
        fc2(static_cast<size_t>(b) * ArchConfig::FC2_OUT),
        probs(static_cast<size_t>(b) * ArchConfig::FC2_OUT), loss(b), labels(b),
        p1_mask(static_cast<size_t>(b) * ArchConfig::C1_OUT_C * ArchConfig::P1_OUT_H * ArchConfig::P1_OUT_W),
        p2_mask(static_cast<size_t>(b) * ArchConfig::FLATTEN_SIZE),
        dlogits(static_cast<size_t>(b) * ArchConfig::FC2_OUT),
        dfc1_relu(static_cast<size_t>(b) * ArchConfig::FC1_OUT),
        dfc1(static_cast<size_t>(b) * ArchConfig::FC1_OUT),
        dp2(static_cast<size_t>(b) * ArchConfig::FLATTEN_SIZE),
        dc2(static_cast<size_t>(b) * ArchConfig::C2_OUT_C * ArchConfig::C2_OUT_H * ArchConfig::C2_OUT_W),
        dp1(static_cast<size_t>(b) * ArchConfig::C1_OUT_C * ArchConfig::P1_OUT_H * ArchConfig::P1_OUT_W),
        dc1(static_cast<size_t>(b) * ArchConfig::C1_OUT_C * ArchConfig::C1_OUT_H * ArchConfig::C1_OUT_W) {}
};

void forward_batch(const float* input, const ModelParameters& p,
                   OpenMPBuffers& a, bool calculate_loss, const int* labels) {
    const int B = a.batch;
    omp_conv2d_forward(input, p.w_c1, p.b_c1, a.c1.data(), B,
        ArchConfig::IN_C, ArchConfig::IN_H, ArchConfig::IN_W,
        ArchConfig::C1_OUT_C, ArchConfig::C1_K, ArchConfig::C1_PAD, ArchConfig::C1_STRIDE);
    omp_relu_forward(a.c1.data(), a.c1.data(), a.c1.size());
    omp_maxpool2d_forward(a.c1.data(), a.p1.data(), a.p1_mask.data(), B,
        ArchConfig::C1_OUT_C, ArchConfig::C1_OUT_H, ArchConfig::C1_OUT_W,
        ArchConfig::P1_K, ArchConfig::P1_STRIDE);

    omp_conv2d_forward(a.p1.data(), p.w_c2, p.b_c2, a.c2.data(), B,
        ArchConfig::C1_OUT_C, ArchConfig::P1_OUT_H, ArchConfig::P1_OUT_W,
        ArchConfig::C2_OUT_C, ArchConfig::C2_K, ArchConfig::C2_PAD, ArchConfig::C2_STRIDE);
    omp_relu_forward(a.c2.data(), a.c2.data(), a.c2.size());
    omp_maxpool2d_forward(a.c2.data(), a.p2.data(), a.p2_mask.data(), B,
        ArchConfig::C2_OUT_C, ArchConfig::C2_OUT_H, ArchConfig::C2_OUT_W,
        ArchConfig::P2_K, ArchConfig::P2_STRIDE);

    omp_linear_forward(a.p2.data(), p.w_fc1, p.b_fc1, a.fc1.data(),
        B, ArchConfig::FLATTEN_SIZE, ArchConfig::FC1_OUT);
    omp_relu_forward(a.fc1.data(), a.fc1_relu.data(), a.fc1.size());
    omp_linear_forward(a.fc1_relu.data(), p.w_fc2, p.b_fc2, a.fc2.data(),
        B, ArchConfig::FC1_OUT, ArchConfig::FC2_OUT);

    omp_softmax_cross_entropy_forward_backward(a.fc2.data(), labels,
        a.probs.data(), a.dlogits.data(), calculate_loss ? a.loss.data() : nullptr,
        B, ArchConfig::FC2_OUT);
}

void backward_and_update(OpenMPBuffers& a, ModelParameters& p,
                         const TrainingConfig& config) {
    const int B = a.batch;
    omp_linear_backward(a.dlogits.data(), a.fc1_relu.data(), p.w_fc2,
        a.dfc1_relu.data(), p.dw_fc2, p.db_fc2, B, ArchConfig::FC1_OUT, ArchConfig::FC2_OUT);
    omp_relu_backward(a.dfc1_relu.data(), a.fc1_relu.data(), a.dfc1.data(), a.fc1_relu.size());
    omp_linear_backward(a.dfc1.data(), a.p2.data(), p.w_fc1,
        a.dp2.data(), p.dw_fc1, p.db_fc1, B, ArchConfig::FLATTEN_SIZE, ArchConfig::FC1_OUT);

    omp_maxpool2d_backward(a.dp2.data(), a.p2_mask.data(), a.dc2.data(), B,
        ArchConfig::C2_OUT_C, ArchConfig::P2_OUT_H, ArchConfig::P2_OUT_W,
        ArchConfig::C2_OUT_H, ArchConfig::C2_OUT_W, ArchConfig::P2_K, ArchConfig::P2_STRIDE);
    omp_relu_backward(a.dc2.data(), a.c2.data(), a.dc2.data(), a.c2.size());
    omp_conv2d_backward(a.dc2.data(), a.p1.data(), p.w_c2,
        a.dp1.data(), p.dw_c2, p.db_c2, B,
        ArchConfig::C1_OUT_C, ArchConfig::P1_OUT_H, ArchConfig::P1_OUT_W,
        ArchConfig::C2_OUT_C, ArchConfig::C2_K, ArchConfig::C2_PAD, ArchConfig::C2_STRIDE);

    omp_maxpool2d_backward(a.dp1.data(), a.p1_mask.data(), a.dc1.data(), B,
        ArchConfig::C1_OUT_C, ArchConfig::P1_OUT_H, ArchConfig::P1_OUT_W,
        ArchConfig::C1_OUT_H, ArchConfig::C1_OUT_W, ArchConfig::P1_K, ArchConfig::P1_STRIDE);
    omp_relu_backward(a.dc1.data(), a.c1.data(), a.dc1.data(), a.c1.size());
    omp_conv2d_backward(a.dc1.data(), a.input.data(), p.w_c1,
        nullptr, p.dw_c1, p.db_c1, B,
        ArchConfig::IN_C, ArchConfig::IN_H, ArchConfig::IN_W,
        ArchConfig::C1_OUT_C, ArchConfig::C1_K, ArchConfig::C1_PAD, ArchConfig::C1_STRIDE);

    omp_sgd_update(p.w_c1, p.dw_c1, p.vw_c1, p.b_c1, p.db_c1, p.vb_c1,
        ArchConfig::C1_W_SIZE, ArchConfig::C1_B_SIZE, config.learning_rate, config.momentum, config.weight_decay);
    omp_sgd_update(p.w_c2, p.dw_c2, p.vw_c2, p.b_c2, p.db_c2, p.vb_c2,
        ArchConfig::C2_W_SIZE, ArchConfig::C2_B_SIZE, config.learning_rate, config.momentum, config.weight_decay);
    omp_sgd_update(p.w_fc1, p.dw_fc1, p.vw_fc1, p.b_fc1, p.db_fc1, p.vb_fc1,
        ArchConfig::FC1_W_SIZE, ArchConfig::FC1_B_SIZE, config.learning_rate, config.momentum, config.weight_decay);
    omp_sgd_update(p.w_fc2, p.dw_fc2, p.vw_fc2, p.b_fc2, p.db_fc2, p.vb_fc2,
        ArchConfig::FC2_W_SIZE, ArchConfig::FC2_B_SIZE, config.learning_rate, config.momentum, config.weight_decay);
}

int batch_correct(const float* probs, const int* labels, int count) {
    int correct = 0;
    #pragma omp parallel for reduction(+:correct) schedule(static)
    for (int b = 0; b < count; ++b) {
        int pred = 0;
        float best = -1.0f;
        for (int c = 0; c < ArchConfig::FC2_OUT; ++c) {
            float value = probs[b * ArchConfig::FC2_OUT + c];
            if (value > best) { best = value; pred = c; }
        }
        if (pred == labels[b]) ++correct;
    }
    return correct;
}

} // anonymous namespace

float omp_evaluate(const TinyImageNetDataset& val_data,
                   const ModelParameters& params, int batch_size) {
    if (val_data.size() == 0 || batch_size <= 0) return 0.0f;
    OpenMPBuffers a(batch_size);
    int correct = 0;
    int evaluated = 0;

    for (size_t start = 0; start < val_data.size(); start += batch_size) {
        int actual = val_data.get_batch(start, batch_size, a.input.data(), a.labels.data());
        if (actual == 0) break;
        forward_batch(a.input.data(), params, a, false, a.labels.data());
        correct += batch_correct(a.probs.data(), a.labels.data(), actual);
        evaluated += actual;
    }
    return evaluated ? 100.0f * static_cast<float>(correct) / evaluated : 0.0f;
}

void omp_train(const TinyImageNetDataset& train_data,
               const TinyImageNetDataset& val_data,
               ModelParameters& params,
               const TrainingConfig& config) {
    if (train_data.size() == 0 || config.batch_size <= 0 || config.epochs <= 0) {
        std::cerr << "[OpenMP] Invalid training data or configuration.\n";
        return;
    }

    std::cout << "\n=======================================================\n";
    std::cout << " Starting Deep CNN Training using OpenMP\n";
    std::cout << " Architecture: 2 Conv + 2 MaxPool + 2 Linear + Softmax\n";
    std::cout << " Classes: " << ArchConfig::FC2_OUT << " | Batch Size: " << config.batch_size
              << " | Epochs: " << config.epochs << " | LR: " << config.learning_rate << "\n";
#ifdef _OPENMP
    std::cout << " OpenMP Threads: " << omp_get_max_threads() << "\n";
#endif
    std::cout << "=======================================================\n";

    OpenMPBuffers a(config.batch_size);
    CPUTimer epoch_timer;
    CPUTimer batch_timer;
    const size_t total_batches = (train_data.size() + config.batch_size - 1) / config.batch_size;

    for (int epoch = 1; epoch <= config.epochs; ++epoch) {
        epoch_timer.start();
        float epoch_loss = 0.0f;
        int epoch_correct = 0;
        int samples_processed = 0;
        const_cast<TinyImageNetDataset&>(train_data).shuffle(epoch * 1000 + 42);

        for (size_t batch_idx = 0; batch_idx < total_batches; ++batch_idx) {
            batch_timer.start();
            size_t start = batch_idx * config.batch_size;
            int actual = train_data.get_batch(start, config.batch_size, a.input.data(), a.labels.data());
            if (actual == 0) break;

            // Same full configured batch processing as the CUDA backend.
            forward_batch(a.input.data(), params, a, true, a.labels.data());
            backward_and_update(a, params, config);

            float batch_loss_sum = 0.0f;
            for (int i = 0; i < actual; ++i) batch_loss_sum += a.loss[i];
            int batch_hits = batch_correct(a.probs.data(), a.labels.data(), actual);
            epoch_loss += batch_loss_sum;
            epoch_correct += batch_hits;
            samples_processed += actual;

            if (batch_idx % 10 == 0 || batch_idx == total_batches - 1) {
                double batch_ms = batch_timer.stop();
                float avg_batch_loss = batch_loss_sum / static_cast<float>(actual);
                float batch_acc = 100.0f * static_cast<float>(batch_hits) / actual;
                float throughput = static_cast<float>(actual) / (static_cast<float>(batch_ms) / 1000.0f);
                std::cout << "\r[Epoch " << epoch << "/" << config.epochs << "] "
                          << "Batch " << (batch_idx + 1) << "/" << total_batches
                          << " | Loss: " << std::fixed << std::setprecision(4) << avg_batch_loss
                          << " | Acc: " << std::setprecision(2) << batch_acc << "%"
                          << " | " << std::setprecision(1) << throughput << " img/s" << std::flush;
            }
        }

        double epoch_ms = epoch_timer.stop();
        if (samples_processed == 0) break;
        float avg_loss = epoch_loss / static_cast<float>(samples_processed);
        float avg_acc = 100.0f * static_cast<float>(epoch_correct) / samples_processed;
        std::cout << "\n--> Epoch " << epoch << " Finished in "
                  << std::fixed << std::setprecision(2) << epoch_ms / 1000.0 << "s"
                  << " | Train Loss: " << avg_loss
                  << " | Train Acc: " << avg_acc << "%\n";

        if (val_data.size() > 0) {
            float val_acc = omp_evaluate(val_data, params, config.batch_size);
            std::cout << "--> Validation Accuracy: " << std::setprecision(2) << val_acc << "%\n";
        }
    }

    std::cout << "=======================================================\n";
    std::cout << " OpenMP Training Completed Successfully!\n";
    std::cout << "=======================================================\n";
}

} // namespace cutin
