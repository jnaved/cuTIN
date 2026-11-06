
#include "dataloader.hpp"
#include "network.hpp"
#include <ctime>
#include <iomanip>
#include <iostream>

namespace cutin {

// =========================================================================
// SEQUENTIAL CPU WORKSPACE
// Stores intermediate tensors and gradients in host memory.
// =========================================================================

struct SeqWorkspace {

  int batch_size;

  float *input;
  int *labels;

  float *c1_out;
  float *p1_out;
  int *p1_mask;

  float *c2_out;
  float *p2_out;
  int *p2_mask;

  float *fc1_out;
  float *fc1_relu;
  float *fc2_out;

  float *probs;
  float *batch_loss;

  // Backward gradients
  float *dlogits;
  float *dfc1_relu;
  float *dfc1_out;
  float *dp2_out;
  float *dc2_out;
  float *dp1_out;
  float *dc1_out;

  // Constructor: allocate all arrays
  SeqWorkspace(int B) {

    batch_size = B;

    size_t in_size =
        (size_t)B * ArchConfig::IN_C * ArchConfig::IN_H * ArchConfig::IN_W;

    size_t c1_size = (size_t)B * ArchConfig::C1_OUT_C * ArchConfig::C1_OUT_H *
                     ArchConfig::C1_OUT_W;

    size_t p1_size = (size_t)B * ArchConfig::C1_OUT_C * ArchConfig::P1_OUT_H *
                     ArchConfig::P1_OUT_W;

    size_t c2_size = (size_t)B * ArchConfig::C2_OUT_C * ArchConfig::C2_OUT_H *
                     ArchConfig::C2_OUT_W;

    size_t p2_size = (size_t)B * ArchConfig::FLATTEN_SIZE;

    size_t fc1_size = (size_t)B * ArchConfig::FC1_OUT;

    size_t fc2_size = (size_t)B * ArchConfig::FC2_OUT;

    input = new float[in_size];
    labels = new int[B];

    c1_out = new float[c1_size];
    p1_out = new float[p1_size];
    p1_mask = new int[p1_size];

    c2_out = new float[c2_size];
    p2_out = new float[p2_size];
    p2_mask = new int[p2_size];

    fc1_out = new float[fc1_size];
    fc1_relu = new float[fc1_size];
    fc2_out = new float[fc2_size];

    probs = new float[fc2_size];
    batch_loss = new float[B];

    dlogits = new float[fc2_size];
    dfc1_relu = new float[fc1_size];
    dfc1_out = new float[fc1_size];

    dp2_out = new float[p2_size];
    dc2_out = new float[c2_size];
    dp1_out = new float[p1_size];
    dc1_out = new float[c1_size];
  }

  // Destructor: release all arrays
  ~SeqWorkspace() {

    delete[] input;
    delete[] labels;

    delete[] c1_out;
    delete[] p1_out;
    delete[] p1_mask;

    delete[] c2_out;
    delete[] p2_out;
    delete[] p2_mask;

    delete[] fc1_out;
    delete[] fc1_relu;
    delete[] fc2_out;

    delete[] probs;
    delete[] batch_loss;

    delete[] dlogits;
    delete[] dfc1_relu;
    delete[] dfc1_out;

    delete[] dp2_out;
    delete[] dc2_out;
    delete[] dp1_out;
    delete[] dc1_out;
  }

  // Prevent accidental copying of pointer-owning workspace
  SeqWorkspace(const SeqWorkspace &) = delete;
  SeqWorkspace &operator=(const SeqWorkspace &) = delete;
};

// =========================================================================
// SEQUENTIAL FORWARD PASS
//
// Conv1 -> ReLU -> Pool1
// Conv2 -> ReLU -> Pool2
// FC1 -> ReLU
// FC2
// =========================================================================

static void seq_forward_pass(const float *input, const ModelParameters &params,
                             SeqWorkspace &buffers, int B) {

  // -----------------------------------------------------
  // 1. Conv1 + ReLU1
  // -----------------------------------------------------

  seq_conv2d_forward(input, params.w_c1, params.b_c1, buffers.c1_out, B,
                     ArchConfig::IN_C, ArchConfig::IN_H, ArchConfig::IN_W,
                     ArchConfig::C1_OUT_C, ArchConfig::C1_K, ArchConfig::C1_PAD,
                     ArchConfig::C1_STRIDE);

  size_t c1_size = (size_t)B * ArchConfig::C1_OUT_C * ArchConfig::C1_OUT_H *
                   ArchConfig::C1_OUT_W;

  seq_relu_forward(buffers.c1_out, buffers.c1_out, c1_size);

  // -----------------------------------------------------
  // 2. MaxPool1
  // -----------------------------------------------------

  seq_maxpool2d_forward(buffers.c1_out, buffers.p1_out, buffers.p1_mask, B,
                        ArchConfig::C1_OUT_C, ArchConfig::C1_OUT_H,
                        ArchConfig::C1_OUT_W, ArchConfig::P1_K,
                        ArchConfig::P1_STRIDE);

  // -----------------------------------------------------
  // 3. Conv2 + ReLU2
  // -----------------------------------------------------

  seq_conv2d_forward(buffers.p1_out, params.w_c2, params.b_c2, buffers.c2_out,
                     B, ArchConfig::C1_OUT_C, ArchConfig::P1_OUT_H,
                     ArchConfig::P1_OUT_W, ArchConfig::C2_OUT_C,
                     ArchConfig::C2_K, ArchConfig::C2_PAD,
                     ArchConfig::C2_STRIDE);

  size_t c2_size = (size_t)B * ArchConfig::C2_OUT_C * ArchConfig::C2_OUT_H *
                   ArchConfig::C2_OUT_W;

  seq_relu_forward(buffers.c2_out, buffers.c2_out, c2_size);

  // -----------------------------------------------------
  // 4. MaxPool2
  // -----------------------------------------------------

  seq_maxpool2d_forward(buffers.c2_out, buffers.p2_out, buffers.p2_mask, B,
                        ArchConfig::C2_OUT_C, ArchConfig::C2_OUT_H,
                        ArchConfig::C2_OUT_W, ArchConfig::P2_K,
                        ArchConfig::P2_STRIDE);

  // -----------------------------------------------------
  // 5. FC1 + ReLU3
  // -----------------------------------------------------

  seq_linear_forward(buffers.p2_out, params.w_fc1, params.b_fc1,
                     buffers.fc1_out, B, ArchConfig::FLATTEN_SIZE,
                     ArchConfig::FC1_OUT);

  size_t fc1_size = (size_t)B * ArchConfig::FC1_OUT;

  seq_relu_forward(buffers.fc1_out, buffers.fc1_relu, fc1_size);

  // -----------------------------------------------------
  // 6. FC2
  // -----------------------------------------------------

  seq_linear_forward(buffers.fc1_relu, params.w_fc2, params.b_fc2,
                     buffers.fc2_out, B, ArchConfig::FC1_OUT,
                     ArchConfig::FC2_OUT);
}

// =========================================================================
// 1. SEQUENTIAL EVALUATION
//
// Forward pass only.
// Calculate classification accuracy on validation dataset.
// =========================================================================

float seq_evaluate(const TinyImageNetDataset &val_data,
                   const ModelParameters &params, int batch_size) {

  if (val_data.size() == 0 || batch_size <= 0) {
    return 0.0f;
  }

  SeqWorkspace buffers(batch_size);

  int correct = 0;
  int total_evaluated = 0;

  for (size_t start = 0; start < val_data.size(); start += batch_size) {

    // Load one batch
    int actual =
        val_data.get_batch(start, batch_size, buffers.input, buffers.labels);

    if (actual <= 0) {
      break;
    }

    // Forward pass
    seq_forward_pass(buffers.input, params, buffers, actual);

    // Calculate probabilities
    seq_softmax_cross_entropy_forward_backward(
        buffers.fc2_out, buffers.labels, buffers.probs, buffers.dlogits,
        nullptr, actual, ArchConfig::FC2_OUT);

    // Find predicted class
    for (int b = 0; b < actual; b++) {

      float max_probability = -1.0f;
      int prediction = 0;

      for (int c = 0; c < ArchConfig::FC2_OUT; c++) {

        float p = buffers.probs[b * ArchConfig::FC2_OUT + c];

        if (p > max_probability) {
          max_probability = p;
          prediction = c;
        }
      }

      if (prediction == buffers.labels[b]) {
        correct++;
      }

      total_evaluated++;
    }
  }

  if (total_evaluated == 0) {
    return 0.0f;
  }

  float accuracy = 100.0f * (float)correct / (float)total_evaluated;

  return accuracy;
}

// =========================================================================
// 2. SEQUENTIAL TRAINING
//
// Forward -> Loss -> Backward -> SGD -> Metrics
// =========================================================================

void seq_train(const TinyImageNetDataset &train_data,
               const TinyImageNetDataset &val_data, ModelParameters &params,
               const TrainingConfig &config) {

  if (config.batch_size <= 0 || config.epochs <= 0 || train_data.size() == 0) {

    std::cout << "[Sequential] Invalid training configuration.\n";
    return;
  }

  std::cout << "\n=======================================================\n";
  std::cout << " Starting Deep CNN Training using Sequential C++\n";
  std::cout << " Architecture: 2 Conv + 2 MaxPool + 2 Linear + Softmax\n";
  std::cout << " Classes: " << ArchConfig::FC2_OUT
            << " | Batch Size: " << config.batch_size
            << " | Epochs: " << config.epochs
            << " | LR: " << config.learning_rate << "\n";
  std::cout << "=======================================================\n";

  SeqWorkspace buffers(config.batch_size);

  size_t num_train = train_data.size();

  size_t total_batches =
      (num_train + config.batch_size - 1) / config.batch_size;

  // =====================================================
  // EPOCH LOOP
  // =====================================================

  for (int epoch = 1; epoch <= config.epochs; epoch++) {

    clock_t epoch_start = clock();

    float epoch_loss = 0.0f;

    int epoch_correct = 0;
    int total_samples_processed = 0;

    // Shuffle dataset for each epoch
    const_cast<TinyImageNetDataset &>(train_data).shuffle(epoch * 1000 + 42);

    // =================================================
    // BATCH LOOP
    // =================================================

    for (size_t b_idx = 0; b_idx < total_batches; b_idx++) {

      clock_t batch_start = clock();

      size_t start = b_idx * config.batch_size;

      // Load batch
      int actual = train_data.get_batch(start, config.batch_size, buffers.input,
                                        buffers.labels);

      if (actual <= 0) {
        break;
      }

      // =================================================
      // FORWARD PASS
      // =================================================

      seq_forward_pass(buffers.input, params, buffers, actual);

      // =================================================
      // SOFTMAX + CROSS-ENTROPY
      // =================================================

      seq_softmax_cross_entropy_forward_backward(
          buffers.fc2_out, buffers.labels, buffers.probs, buffers.dlogits,
          buffers.batch_loss, actual, ArchConfig::FC2_OUT);

      // =================================================
      // BACKWARD PASS
      // =================================================

      // -------------------------------------------------
      // 1. FC2 backward
      // -------------------------------------------------

      seq_linear_backward(buffers.dlogits, buffers.fc1_relu, params.w_fc2,
                          buffers.dfc1_relu, params.dw_fc2, params.db_fc2,
                          actual, ArchConfig::FC1_OUT, ArchConfig::FC2_OUT);

      // -------------------------------------------------
      // 2. ReLU3 backward
      // -------------------------------------------------

      seq_relu_backward(buffers.dfc1_relu, buffers.fc1_relu, buffers.dfc1_out,
                        (size_t)actual * ArchConfig::FC1_OUT);

      // -------------------------------------------------
      // 3. FC1 backward
      // -------------------------------------------------

      seq_linear_backward(buffers.dfc1_out, buffers.p2_out, params.w_fc1,
                          buffers.dp2_out, params.dw_fc1, params.db_fc1, actual,
                          ArchConfig::FLATTEN_SIZE, ArchConfig::FC1_OUT);

      // -------------------------------------------------
      // 4. MaxPool2 backward
      // -------------------------------------------------

      seq_maxpool2d_backward(buffers.dp2_out, buffers.p2_mask, buffers.dc2_out,
                             actual, ArchConfig::C2_OUT_C, ArchConfig::P2_OUT_H,
                             ArchConfig::P2_OUT_W, ArchConfig::C2_OUT_H,
                             ArchConfig::C2_OUT_W, ArchConfig::P2_K,
                             ArchConfig::P2_STRIDE);

      // -------------------------------------------------
      // 5. ReLU2 backward
      // -------------------------------------------------

      seq_relu_backward(buffers.dc2_out, buffers.c2_out, buffers.dc2_out,
                        (size_t)actual * ArchConfig::C2_OUT_C *
                            ArchConfig::C2_OUT_H * ArchConfig::C2_OUT_W);

      // -------------------------------------------------
      // 6. Conv2 backward
      // -------------------------------------------------

      seq_conv2d_backward(
          buffers.dc2_out, buffers.p1_out, params.w_c2, buffers.dp1_out,
          params.dw_c2, params.db_c2, actual, ArchConfig::C1_OUT_C,
          ArchConfig::P1_OUT_H, ArchConfig::P1_OUT_W, ArchConfig::C2_OUT_C,
          ArchConfig::C2_K, ArchConfig::C2_PAD, ArchConfig::C2_STRIDE);

      // -------------------------------------------------
      // 7. MaxPool1 backward
      // -------------------------------------------------

      seq_maxpool2d_backward(buffers.dp1_out, buffers.p1_mask, buffers.dc1_out,
                             actual, ArchConfig::C1_OUT_C, ArchConfig::P1_OUT_H,
                             ArchConfig::P1_OUT_W, ArchConfig::C1_OUT_H,
                             ArchConfig::C1_OUT_W, ArchConfig::P1_K,
                             ArchConfig::P1_STRIDE);

      // -------------------------------------------------
      // 8. ReLU1 backward
      // -------------------------------------------------

      seq_relu_backward(buffers.dc1_out, buffers.c1_out, buffers.dc1_out,
                        (size_t)actual * ArchConfig::C1_OUT_C *
                            ArchConfig::C1_OUT_H * ArchConfig::C1_OUT_W);

      // -------------------------------------------------
      // 9. Conv1 backward
      // Input gradient is not required, so dx = nullptr.
      // -------------------------------------------------

      seq_conv2d_backward(buffers.dc1_out, buffers.input, params.w_c1, nullptr,
                          params.dw_c1, params.db_c1, actual, ArchConfig::IN_C,
                          ArchConfig::IN_H, ArchConfig::IN_W,
                          ArchConfig::C1_OUT_C, ArchConfig::C1_K,
                          ArchConfig::C1_PAD, ArchConfig::C1_STRIDE);

      // =================================================
      // OPTIMIZER STEP: SGD WITH MOMENTUM
      // =================================================

      // -------------------------------------------------
      // Update Conv1 parameters
      // -------------------------------------------------

      seq_sgd_update(params.w_c1, params.dw_c1, params.vw_c1, params.b_c1,
                     params.db_c1, params.vb_c1, ArchConfig::C1_W_SIZE,
                     ArchConfig::C1_B_SIZE, config.learning_rate,
                     config.momentum, config.weight_decay);

      // -------------------------------------------------
      // Update Conv2 parameters
      // -------------------------------------------------

      seq_sgd_update(params.w_c2, params.dw_c2, params.vw_c2, params.b_c2,
                     params.db_c2, params.vb_c2, ArchConfig::C2_W_SIZE,
                     ArchConfig::C2_B_SIZE, config.learning_rate,
                     config.momentum, config.weight_decay);

      // -------------------------------------------------
      // Update FC1 parameters
      // -------------------------------------------------

      seq_sgd_update(params.w_fc1, params.dw_fc1, params.vw_fc1, params.b_fc1,
                     params.db_fc1, params.vb_fc1, ArchConfig::FC1_W_SIZE,
                     ArchConfig::FC1_B_SIZE, config.learning_rate,
                     config.momentum, config.weight_decay);

      // -------------------------------------------------
      // Update FC2 parameters
      // -------------------------------------------------

      seq_sgd_update(params.w_fc2, params.dw_fc2, params.vw_fc2, params.b_fc2,
                     params.db_fc2, params.vb_fc2, ArchConfig::FC2_W_SIZE,
                     ArchConfig::FC2_B_SIZE, config.learning_rate,
                     config.momentum, config.weight_decay);

      // =================================================
      // BATCH METRICS
      // =================================================

      float batch_loss_sum = 0.0f;

      int batch_correct = 0;

      for (int i = 0; i < actual; i++) {

        batch_loss_sum += buffers.batch_loss[i];

        float max_p = -1.0f;
        int prediction = 0;

        for (int c = 0; c < ArchConfig::FC2_OUT; c++) {

          float p = buffers.probs[i * ArchConfig::FC2_OUT + c];

          if (p > max_p) {
            max_p = p;
            prediction = c;
          }
        }

        if (prediction == buffers.labels[i]) {
          batch_correct++;
        }
      }

      epoch_loss += batch_loss_sum;
      epoch_correct += batch_correct;
      total_samples_processed += actual;

      // =================================================
      // PRINT BATCH PROGRESS
      // =================================================

      if (b_idx % 10 == 0 || b_idx == total_batches - 1) {

        clock_t batch_end = clock();

        double batch_seconds =
            (double)(batch_end - batch_start) / CLOCKS_PER_SEC;

        float current_loss = batch_loss_sum / (float)actual;

        float current_accuracy = 100.0f * (float)batch_correct / (float)actual;

        float throughput = 0.0f;

        if (batch_seconds > 0.0) {
          throughput = (float)actual / (float)batch_seconds;
        }

        std::cout << "\r[Epoch " << epoch << "/" << config.epochs << "] "
                  << "Batch " << (b_idx + 1) << "/" << total_batches
                  << " | Loss: " << std::fixed << std::setprecision(4)
                  << current_loss << " | Acc: " << std::setprecision(2)
                  << current_accuracy << "%"
                  << " | " << std::setprecision(1) << throughput << " img/s"
                  << std::flush;
      }
    }

    // =================================================
    // EPOCH METRICS
    // =================================================

    clock_t epoch_end = clock();

    double epoch_seconds = (double)(epoch_end - epoch_start) / CLOCKS_PER_SEC;

    float avg_loss = 0.0f;
    float avg_accuracy = 0.0f;

    if (total_samples_processed > 0) {

      avg_loss = epoch_loss / (float)total_samples_processed;

      avg_accuracy =
          100.0f * (float)epoch_correct / (float)total_samples_processed;
    }

    std::cout << "\n--> Epoch " << epoch << " Finished in " << std::fixed
              << std::setprecision(2) << epoch_seconds << "s"
              << " | Train Loss: " << avg_loss
              << " | Train Acc: " << avg_accuracy << "%\n";

    // =================================================
    // VALIDATION
    // =================================================

    if (val_data.size() > 0) {

      float val_accuracy = seq_evaluate(val_data, params, config.batch_size);

      std::cout << "--> Validation Accuracy: " << std::setprecision(2)
                << val_accuracy << "%\n";
    }
  }

  std::cout << "=======================================================\n";
  std::cout << " Sequential Training Completed Successfully!\n";
  std::cout << "=======================================================\n";
}

} // namespace cutin
