/*
Input (B, 3, 64, 64)
  │
  ▼
Conv2D (16 filters, 3x3, pad=1, stride=1) ──> Output: (B, 16, 64, 64)
  │
  ▼
ReLU + MaxPool (2x2, stride=2)            ──> Output: (B, 16, 32, 32)
  │
  ▼
Conv2D (32 filters, 3x3, pad=1, stride=1) ──> Output: (B, 32, 32, 32)
  │
  ▼
ReLU + MaxPool (2x2, stride=2)            ──> Output: (B, 32, 16, 16)
  │
  ▼
Flatten                                   ──> Output: (B, 32 * 16 * 16 = 8192)
  │
  ▼
Fully Connected (Linear: 8192 -> 512)     ──> Output: (B, 512)
  │
  ▼
ReLU
  │
  ▼
Fully Connected (Linear: 512 -> 200)      ──> Output: (B, 200)
  │
  ▼
Softmax + Categorical Cross-Entropy Loss
*/

#include "network.hpp"
#include "dataloader.hpp"
#include "utils.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

using namespace cutin;

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --mode <cuda|openmp|sequential>  Backend execution mode (default: cuda)\n"
              << "  --epochs <int>                   Number of training epochs (default: 5)\n"
              << "  --batch-size <int>               Mini-batch size (default: 64)\n"
              << "  --lr <float>                     Learning rate for SGD (default: 0.001)\n"
              << "  --momentum <float>               Momentum for SGD (default: 0.9)\n"
              << "  --weight-decay <float>           L2 regularization / weight decay (default: 1e-4)\n"
              << "  --data <dir>                     Path to tiny-imagenet-200 (default: tiny-imagenet-200)\n"
              << "  --max-samples <int>              Limit training/val samples (-1 for full dataset)\n"
              << "  --max-val-samples <int>          Limit validation samples (-1 for full val dataset)\n"
              << "  --help                           Display this help message\n";
}

void print_cuda_device_info() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "[CUDA Warning] No CUDA capable device found!\n";
        return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "-------------------------------------------------------\n";
    std::cout << " GPU Device: " << prop.name << "\n";
    std::cout << " Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << " Total Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
    std::cout << " Multiprocessors (SMs): " << prop.multiProcessorCount << "\n";
    std::cout << " Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "-------------------------------------------------------\n";
}

int main(int argc, char* argv[]) {
    TrainingConfig config;
    ExecutionMode mode = ExecutionMode::CUDA;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "cuda") mode = ExecutionMode::CUDA;
            else if (m == "openmp") mode = ExecutionMode::OPENMP;
            else if (m == "sequential") mode = ExecutionMode::SEQUENTIAL;
            else {
                std::cerr << "Unknown mode: " << m << "\n";
                return 1;
            }
        } else if (arg == "--epochs" && i + 1 < argc) {
            config.epochs = std::atoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            config.batch_size = std::atoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            config.learning_rate = std::atof(argv[++i]);
        } else if (arg == "--momentum" && i + 1 < argc) {
            config.momentum = std::atof(argv[++i]);
        } else if (arg == "--weight-decay" && i + 1 < argc) {
            config.weight_decay = std::atof(argv[++i]);
        } else if (arg == "--data" && i + 1 < argc) {
            config.data_dir = argv[++i];
        } else if (arg == "--max-samples" && i + 1 < argc) {
            config.max_train_samples = std::atoi(argv[++i]);
            config.max_val_samples = config.max_train_samples;
        } else if (arg == "--max-val-samples" && i + 1 < argc) {
            config.max_val_samples = std::atoi(argv[++i]);
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    std::cout << "#######################################################\n";
    std::cout << "#  cuTIN: Convolutional Neural Network on Tiny ImageNet #\n";
    std::cout << "#######################################################\n";

    if (mode == ExecutionMode::CUDA) {
        print_cuda_device_info();
    }

    // 1. Load Dataset
    std::cout << "\n[1/3] Loading Tiny ImageNet Dataset from: " << config.data_dir << "\n";
    TinyImageNetDataset train_dataset;
    TinyImageNetDataset val_dataset;

    if (!train_dataset.load(config.data_dir, true, config.max_train_samples)) {
        std::cerr << "Failed to load training dataset from " << config.data_dir << "\n";
        return 1;
    }

    if (!val_dataset.load(config.data_dir, false, config.max_val_samples)) {
        std::cerr << "Warning: Failed to load validation dataset. Proceeding with training only.\n";
    }

    // 2. Initialize Model Parameters
    std::cout << "\n[2/3] Allocating and Initializing Model Parameters (Kaiming Normal)...\n";
    ModelParameters params;
    params.allocate_host();
    params.init_weights(42);

    std::cout << "Total Network Parameters: " << ArchConfig::TOTAL_WEIGHTS
              << " (" << (ArchConfig::TOTAL_WEIGHTS * sizeof(float) / (1024 * 1024.0f)) << " MB)\n";

    // 3. Dispatch Training
    std::cout << "\n[3/3] Launching Training Pipeline...\n";
    if (mode == ExecutionMode::CUDA) {
        cuda_train(train_dataset, val_dataset, params, config);
    } else if (mode == ExecutionMode::OPENMP) {
        std::cout << "[Notice] Dispatching to OpenMP Backend...\n";
        omp_train(train_dataset, val_dataset, params, config);
    } else if (mode == ExecutionMode::SEQUENTIAL) {
        std::cout << "[Notice] Dispatching to Sequential Backend...\n";
        seq_train(train_dataset, val_dataset, params, config);
    }

    // Free Host Memory
    params.free_host();
    std::cout << "\n[Done] Process exited cleanly.\n";
    return 0;
}
