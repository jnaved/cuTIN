#include "network.hpp"
#include <iostream>

namespace cutin {

// =========================================================================
// SKELETON REFERENCE FOR SEQUENTIAL IMPLEMENTATION
// Teammate note: Implement the sequential training loop and evaluation.
// =========================================================================

float seq_evaluate(const TinyImageNetDataset& val_data,
                   const ModelParameters& params,
                   int batch_size) {
    // TODO: [Teammate - Sequential] Implement forward pass evaluation loop over validation data
    std::cout << "[Sequential] seq_evaluate() is a skeleton. Implement in src/sequential/train.cpp\n";
    (void)val_data;
    (void)params;
    (void)batch_size;
    return 0.0f;
}

void seq_train(const TinyImageNetDataset& train_data,
               const TinyImageNetDataset& val_data,
               ModelParameters& params,
               const TrainingConfig& config) {
    // TODO: [Teammate - Sequential] Implement full sequential training epoch loop:
    // Forward -> Loss -> Backward -> SGD Step -> Metrics
    std::cout << "[Sequential] seq_train() is a skeleton. Implement in src/sequential/train.cpp\n";
    (void)train_data;
    (void)val_data;
    (void)params;
    (void)config;
}

} // namespace cutin

