#include "network.hpp"
#include <iostream>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace cutin {

// =========================================================================
// SKELETON REFERENCE FOR OPENMP IMPLEMENTATION
// Teammate note: Implement the OpenMP-accelerated training loop and evaluation.
// =========================================================================

float omp_evaluate(const TinyImageNetDataset& val_data,
                   const ModelParameters& params,
                   int batch_size) {
    // TODO: [Teammate - OpenMP] Implement OpenMP-accelerated evaluation loop
    std::cout << "[OpenMP] omp_evaluate() is a skeleton. Implement in src/openmp/train.cpp\n";
    (void)val_data;
    (void)params;
    (void)batch_size;
    return 0.0f;
}

void omp_train(const TinyImageNetDataset& train_data,
               const TinyImageNetDataset& val_data,
               ModelParameters& params,
               const TrainingConfig& config) {
    // TODO: [Teammate - OpenMP] Implement full OpenMP training epoch loop
    std::cout << "[OpenMP] omp_train() is a skeleton. Implement in src/openmp/train.cpp\n";
    (void)train_data;
    (void)val_data;
    (void)params;
    (void)config;
}

} // namespace cutin

