/*
=============================================================================
 cuTIN: Deep Convolutional Neural Network from Scratch
 Backends: Sequential CPU, OpenMP Multi-Core, CUDA GPU
 Dataset: Tiny ImageNet (200 classes, 64x64 RGB)
=============================================================================
 Architecture Pipeline:
   Input (B, 3, 64, 64)
     │
     ▼
   Conv2D (16 filters, 3x3, pad=1, stride=1)  ──> (B, 16, 64, 64)
     │
     ▼
   ReLU + MaxPool (2x2, stride=2)              ──> (B, 16, 32, 32)
     │
     ▼
   Conv2D (32 filters, 3x3, pad=1, stride=1)  ──> (B, 32, 32, 32)
     │
     ▼
   ReLU + MaxPool (2x2, stride=2)              ──> (B, 32, 16, 16)
     │
     ▼
   Flatten                                     ──> (B, 8192)
     │
     ▼
   Linear (8192 -> 512) + ReLU                 ──> (B, 512)
     │
     ▼
   Linear (512 -> 200)                         ──> (B, 200)
     │
     ▼
   Softmax + Categorical Cross-Entropy Loss
=============================================================================
*/

#include "network.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <cuda_runtime.h>

using namespace cutin;

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --mode <cuda|openmp|sequential|all> Backend execution mode (default: cuda)\n"
              << "  --epochs <int>                      Number of training epochs (default: 5)\n"
              << "  --batch-size <int>                  Mini-batch size (default: 64)\n"
              << "  --lr <float>                        Learning rate for SGD (default: 0.001)\n"
              << "  --momentum <float>                  Momentum for SGD (default: 0.9)\n"
              << "  --weight-decay <float>              L2 weight decay (default: 1e-4)\n"
              << "  --data <dir>                        Path to tiny-imagenet-200 (default: tiny-imagenet-200)\n"
              << "  --max-samples <int>                 Limit train samples (-1 for full dataset)\n"
              << "  --max-val-samples <int>             Limit validation samples (-1 for full val dataset)\n"
              << "  --help                              Display this help message\n";
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

void run_all_benchmark(const TinyImageNetDataset& train_dataset,
                       const TinyImageNetDataset& val_dataset,
                       const TrainingConfig& config) {
    std::cout << "\n=======================================================\n";
    std::cout << "      RUNNING FULL 3-WAY PERFORMANCE BENCHMARK         \n";
    std::cout << "=======================================================\n";
    std::cout << "Training Samples: " << train_dataset.size()
              << " | Epochs: " << config.epochs
              << " | Batch Size: " << config.batch_size << "\n\n";

    CPUTimer timer;
    double t_seq = 0.0, t_omp = 0.0, t_cuda = 0.0;

    // 1. Sequential Benchmark
    {
        std::cout << ">>> [1/3] Benchmarking Sequential CPU Backend...\n";
        ModelParameters p_seq;
        p_seq.allocate_host();
        p_seq.init_weights(42);

        timer.start();
        seq_train(train_dataset, val_dataset, p_seq, config);
        t_seq = timer.stop() / 1000.0; // in seconds
        p_seq.free_host();
    }

    // 2. OpenMP Benchmark
    {
        std::cout << "\n>>> [2/3] Benchmarking OpenMP Multi-Core Backend...\n";
        ModelParameters p_omp;
        p_omp.allocate_host();
        p_omp.init_weights(42);

        timer.start();
        omp_train(train_dataset, val_dataset, p_omp, config);
        t_omp = timer.stop() / 1000.0; // in seconds
        p_omp.free_host();
    }

    // 3. CUDA GPU Benchmark
    {
        std::cout << "\n>>> [3/3] Benchmarking CUDA GPU Backend...\n";
        ModelParameters p_cuda;
        p_cuda.allocate_host();
        p_cuda.init_weights(42);

        timer.start();
        cuda_train(train_dataset, val_dataset, p_cuda, config);
        t_cuda = timer.stop() / 1000.0; // in seconds
        p_cuda.free_host();
    }

    // Performance Summary Table
    size_t total_processed = train_dataset.size() * config.epochs;
    double seq_throughput  = (t_seq > 0.0)  ? (total_processed / t_seq)  : 0.0;
    double omp_throughput  = (t_omp > 0.0)  ? (total_processed / t_omp)  : 0.0;
    double cuda_throughput = (t_cuda > 0.0) ? (total_processed / t_cuda) : 0.0;

    double omp_speedup  = (t_omp > 0.0)  ? (t_seq / t_omp)  : 1.0;
    double cuda_speedup = (t_cuda > 0.0) ? (t_seq / t_cuda) : 1.0;

    std::cout << "\n";
    std::cout << "===============================================================================\n";
    std::cout << "                      COMPARATIVE PERFORMANCE SUMMARY                          \n";
    std::cout << "===============================================================================\n";
    std::cout << std::left
              << std::setw(18) << " Backend"
              << std::setw(16) << " Time (s)"
              << std::setw(22) << " Throughput (img/s)"
              << std::setw(18) << " Speedup vs Seq"
              << "\n";
    std::cout << "-------------------------------------------------------------------------------\n";
    std::cout << std::left
              << std::setw(18) << " Sequential CPU"
              << std::setw(16) << std::fixed << std::setprecision(2) << t_seq
              << std::setw(22) << std::fixed << std::setprecision(1) << seq_throughput
              << std::setw(18) << "1.00x (Baseline)"
              << "\n";
    std::cout << std::left
              << std::setw(18) << " OpenMP CPU"
              << std::setw(16) << std::fixed << std::setprecision(2) << t_omp
              << std::setw(22) << std::fixed << std::setprecision(1) << omp_throughput
              << std::setw(18) << (std::to_string(omp_speedup).substr(0, 4) + "x")
              << "\n";
    std::cout << std::left
              << std::setw(18) << " CUDA GPU"
              << std::setw(16) << std::fixed << std::setprecision(2) << t_cuda
              << std::setw(22) << std::fixed << std::setprecision(1) << cuda_throughput
              << std::setw(18) << (std::to_string(cuda_speedup).substr(0, 5) + "x")
              << "\n";
    std::cout << "===============================================================================\n";
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
            else if (m == "all") mode = ExecutionMode::ALL;
            else {
                std::cerr << "Unknown mode: " << m << "\n";
                print_usage(argv[0]);
                return 1;
            }
        } else if (arg == "--epochs" && i + 1 < argc) {
            config.epochs = std::atoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            config.batch_size = std::atoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            config.learning_rate = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--momentum" && i + 1 < argc) {
            config.momentum = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--weight-decay" && i + 1 < argc) {
            config.weight_decay = static_cast<float>(std::atof(argv[++i]));
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

    if (mode == ExecutionMode::CUDA || mode == ExecutionMode::ALL) {
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

    // 2. Report Model Parameter Size
    std::cout << "\n[2/3] CNN Model Statistics:\n";
    std::cout << "  Input Dimensions:  " << ArchConfig::IN_C << " x " << ArchConfig::IN_H << " x " << ArchConfig::IN_W << "\n";
    std::cout << "  Conv1 (3x3):       3 -> " << ArchConfig::C1_OUT_C << " channels\n";
    std::cout << "  Conv2 (3x3):       " << ArchConfig::C1_OUT_C << " -> " << ArchConfig::C2_OUT_C << " channels\n";
    std::cout << "  FC1 (Dense):       " << ArchConfig::FLATTEN_SIZE << " -> " << ArchConfig::FC1_OUT << "\n";
    std::cout << "  FC2 (Output):      " << ArchConfig::FC1_OUT << " -> " << ArchConfig::FC2_OUT << " classes\n";
    std::cout << "  Total Parameters:  " << ArchConfig::TOTAL_WEIGHTS
              << " (" << (ArchConfig::TOTAL_WEIGHTS * sizeof(float) / (1024 * 1024.0f)) << " MB)\n";

    // 3. Dispatch Training
    std::cout << "\n[3/3] Launching Execution...\n";
    if (mode == ExecutionMode::ALL) {
        run_all_benchmark(train_dataset, val_dataset, config);
    } else {
        ModelParameters params;
        params.allocate_host();
        params.init_weights(42);

        if (mode == ExecutionMode::CUDA) {
            cuda_train(train_dataset, val_dataset, params, config);
        } else if (mode == ExecutionMode::OPENMP) {
            omp_train(train_dataset, val_dataset, params, config);
        } else if (mode == ExecutionMode::SEQUENTIAL) {
            seq_train(train_dataset, val_dataset, params, config);
        }
        params.free_host();
    }

    std::cout << "\n[Done] Process exited cleanly.\n";
    return 0;
}
