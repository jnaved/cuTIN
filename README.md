# cuTIN: High-Performance CNN from Scratch with CUDA, OpenMP & C++

**Design and Performance Analysis of a CNN-Based Image Classification System Using Tiny ImageNet with C++, OpenMP, and CUDA**

---

## 1. Project Overview

**cuTIN** is an end-to-end Convolutional Neural Network (CNN) built from mathematical first principles in C++ and CUDA. It implements complete forward propagation, analytical backward gradient propagation, loss calculation, and stochastic gradient descent (SGD) optimization without relying on high-level deep learning frameworks (no PyTorch, TensorFlow, cuDNN, or BLAS).

The system classifies images from the **Tiny ImageNet** benchmark (200 visual categories, $64 \times 64$ RGB images) and provides three distinct execution paradigms to study high-performance computing trade-offs:

1. **Sequential C++ (`src/sequential.cpp`)**: Clean single-threaded CPU baseline serving as the golden mathematical ground-truth.
2. **OpenMP Multi-Core C++ (`src/openmp.cpp`)**: Shared-memory multi-threaded parallelism utilizing 1D loop flattening and race-condition-free transposed gather convolutions.
3. **CUDA C/C++ GPU (`src/cuda.cu`)**: Massively parallel GPU acceleration executing custom kernels on NVIDIA streaming multiprocessors (SMs) with zero-overhead device memory management.

---

## 2. CNN Architecture & Parameter Specifications

The network topology is specifically configured for Tiny ImageNet ($3 \times 64 \times 64$) and consists of **4,302,504 trainable parameters** (~16.41 MB memory footprint):

```
Input (B, 3, 64, 64)
  │
  ▼
Conv2D (16 filters, 3x3, pad=1, stride=1) ──> (B, 16, 64, 64)
  │
  ▼
ReLU + MaxPool2D (2x2, stride=2)           ──> (B, 16, 32, 32)
  │
  ▼
Conv2D (32 filters, 3x3, pad=1, stride=1) ──> (B, 32, 32, 32)
  │
  ▼
ReLU + MaxPool2D (2x2, stride=2)           ──> (B, 32, 16, 16)
  │
  ▼
Flatten (32 * 16 * 16)                     ──> (B, 8192)
  │
  ▼
Linear / Dense (8192 -> 512) + ReLU        ──> (B, 512)
  │
  ▼
Linear / Dense (512 -> 200)                ──> (B, 200)
  │
  ▼
Softmax + Categorical Cross-Entropy Loss
```

### Layer-by-Layer Parameter Breakdown

| Layer | Input Shape | Output Shape | Filter / Stride | Weight Dimension | Bias Dimension | Parameter Count |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | — | `(B, 3, 64, 64)` | — | — | — | 0 |
| **Conv1** | `(B, 3, 64, 64)` | `(B, 16, 64, 64)` | $3\times 3$, $s=1, p=1$ | $16 \times 3 \times 3 \times 3$ | 16 | 448 |
| **Pool1** | `(B, 16, 64, 64)` | `(B, 16, 32, 32)` | $2\times 2$, $s=2$ | — | — | 0 |
| **Conv2** | `(B, 16, 32, 32)` | `(B, 32, 32, 32)` | $3\times 3$, $s=1, p=1$ | $32 \times 16 \times 3 \times 3$ | 32 | 4,640 |
| **Pool2** | `(B, 32, 32, 32)` | `(B, 32, 16, 16)` | $2\times 2$, $s=2$ | — | — | 0 |
| **Flatten** | `(B, 32, 16, 16)` | `(B, 8192)` | Reshape | — | — | 0 |
| **FC1** | `(B, 8192)` | `(B, 512)` | Dense + ReLU | $512 \times 8192$ | 512 | 4,194,816 |
| **FC2** | `(B, 512)` | `(B, 200)` | Dense (Logits) | $200 \times 512$ | 200 | 102,600 |
| **Total** | | | | | | **4,302,504 floats (~16.41 MB)** |

---

## 3. Key Features

- **1:1 Mathematical Authenticity**: Every layer (spatial convolutions, argmax pooling masks, fused log-sum-exp softmax cross-entropy, and momentum SGD updates) follows exact deep learning calculus without approximations.
- **Automated Numerical Equivalence Suite**: `test_equivalence` validates layer-by-layer floating-point outputs across Sequential, OpenMP, and CUDA within $\text{atol} = 10^{-4}$.
- **Automatic Model Checkpointing**: Training automatically saves the model weights into binary checkpoint files (`cuda_model.bin`, `openmp_model.bin`, `seq_model.bin`).
- **Comprehensive Epoch Analytics**: Detailed CSV logs (`cuda_stats.csv`, `openmp_stats.csv`, `seq_stats.csv`) record training loss, training accuracy, training error, validation accuracy, testing error, epoch duration, and image throughput.
- **Custom Image Inference**: Test any random JPEG/PNG image on-demand (`--predict <image>`), with automatic bilinear resizing to $64 \times 64$ and human-readable top-5 WordNet predictions.

---

## 4. Prerequisites & Build Instructions

### Prerequisites
- **Linux** or **Windows Subsystem for Linux (WSL2)** (Ubuntu 20.04 / 22.04 recommended)
- **CMake** 3.18 or higher
- **GCC / G++** 9.0 or higher (with OpenMP support)
- **NVIDIA CUDA Toolkit** 11.0 or higher (with `nvcc`)

### Building the Project
From the project root:

```bash
# 1. Create and navigate to the build directory
mkdir -p build
cd build

# 2. Configure the build with CMake
cmake ..

# 3. Compile binaries in parallel
make -j$(nproc)

# 4. Return to project root
cd ..
```

This compiles two executables inside `build/`:
- `build/cuTIN`: Main training, benchmark, and inference executable.
- `build/test_equivalence`: Numerical verification test suite.

---

## 5. How to Run

### Step 1: Verify All Backends (Unit & Equivalence Tests)
Run this first to prove to your examiner that CUDA, OpenMP, and Sequential produce identical math:

```bash
./build/test_equivalence
```

*Expected output:*
```
=======================================================
 cuTIN Unit & Backend Equivalence Verification Suite
=======================================================
[Test] Running ReLU Backend Equivalence Tests:
  - CUDA vs Ref:       PASSED
  - OpenMP vs Ref:     PASSED
  - Sequential vs Ref: PASSED
...
 ALL BACKENDS (CUDA, OpenMP, Sequential) PASSED EQUIVALENCE!
=======================================================
```

---

### Step 2: Training & Benchmarks

#### A. Fast CUDA GPU Training (Recommended Demo)
Runs massively parallel GPU training on your NVIDIA GPU:
```bash
./build/cuTIN --mode cuda --max-samples 1000 --epochs 2 --batch-size 64
```
*Outputs:*
- Live loss, accuracy, and throughput in images/second.
- Model saved to: `cuda_model.bin`.
- Epoch statistics saved to: `cuda_stats.csv`.

#### B. OpenMP Multi-Core CPU Training
Spawns threads across all available CPU cores:
```bash
./build/cuTIN --mode openmp --max-samples 1000 --epochs 2 --batch-size 64
```
*Outputs:*
- Model saved to: `openmp_model.bin`.
- Epoch statistics saved to: `openmp_stats.csv`.

#### C. Sequential Single-Threaded CPU Training
Runs single-core baseline:
```bash
./build/cuTIN --mode sequential --max-samples 200 --epochs 1 --batch-size 64
```
*Outputs:*
- Model saved to: `seq_model.bin`.
- Epoch statistics saved to: `seq_stats.csv`.

#### D. Full 3-Way Comparative Benchmark
Trains identical initialized weights across all three engines on the same batch data, printing an automated comparison summary:
```bash
./build/cuTIN --mode all --max-samples 500 --epochs 1 --batch-size 64
```

---

### Step 3: Evaluating Model on Test / Validation Data

To evaluate your saved model weights on the 10,000 test images without retraining:

```bash
./build/cuTIN --eval --mode cuda
```
*(Optionally specify a custom model: `./build/cuTIN --eval --model cuda_model.bin`)*

---

### Step 4: Classifying Custom Images (Inference)

To test any arbitrary JPEG or PNG image from your computer:

```bash
./build/cuTIN --predict path/to/my_image.jpg
```

You can optionally specify which trained model weights to load:
```bash
./build/cuTIN --predict path/to/my_image.jpg --model cuda_model.bin
```

*Sample Prediction Output:*
```
#######################################################
#  cuTIN: Custom Image Classification & Prediction     #
#######################################################
[1/4] Loading class metadata from: tiny-imagenet-200...
[2/4] Loading and preprocessing image: val_0.JPEG...
[3/4] Loading model weights from: cuda_model.bin...
[4/4] Executing forward inference...

=======================================================
               CLASSIFICATION RESULTS                  
=======================================================
 Input Image: val_0.JPEG
 Model Used:  cuda_model.bin
-------------------------------------------------------
 Rank  Confidence   Class Description
-------------------------------------------------------
 #1     88.42%     n03444034 (go-kart)
 #2      4.12%     n04067472 (remote control, remote)
 #3      2.05%     n02808440 (bathtub, bathing tub)
 #4      1.10%     n04487081 (trolleybus, trolley coach)
 #5      0.54%     n02124075 (Egyptian cat)
=======================================================
```

---

## 6. Command-Line Reference

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--mode` | string | `cuda` | Execution backend: `cuda`, `openmp`, `sequential`, `all` |
| `--epochs` | int | `5` | Number of training epochs |
| `--batch-size` | int | `64` | Mini-batch size for training and evaluation |
| `--lr` | float | `0.001` | Learning rate for SGD |
| `--momentum` | float | `0.9` | Momentum factor for SGD |
| `--weight-decay` | float | `1e-4` | L2 weight decay regularization |
| `--data` | string | `tiny-imagenet-200` | Path to Tiny ImageNet root directory |
| `--max-samples` | int | `-1` | Maximum training samples (`-1` = full dataset) |
| `--max-val-samples` | int | `-1` | Maximum validation samples (`-1` = full dataset) |
| `--predict` | string | `""` | Path to a custom test image for classification |
| `--eval` | flag | `false` | Evaluate saved model on validation/test set without training |
| `--model` | string | `""` | Path to `.bin` model weights file for inference |
| `--save-model` | string | `""` | Custom output file for saving trained model weights |
| `--stats-file` | string | `""` | Custom output file for epoch statistics CSV |
| `--help` | — | — | Print command-line options and usage |

---

## 7. Model Statistics CSV Format

Every training run exports per-epoch telemetry in standard CSV format:

```csv
epoch,train_loss,train_acc_pct,train_error_pct,val_acc_pct,val_error_pct,epoch_time_sec,throughput_img_sec
1,4.1821,31.25,68.75,0.78,99.22,0.63,202.51
2,0.9695,100.00,0.00,0.78,99.22,0.61,210.16
```

This file can be directly imported into Python (Matplotlib, Pandas), Excel, or Google Sheets to generate loss curves and performance charts for reports and presentations.

---

## 8. Typical Performance Comparison

Measured on an **NVIDIA GeForce RTX 4050 Laptop GPU** and an **AMD/Intel 16-Thread Multi-Core CPU**:

| Backend | Hardware Utilized | Throughput | Relative Speedup | Key Architectural Driver |
| :--- | :--- | :--- | :--- | :--- |
| **Sequential C++** | 1 CPU Core | ~2.6 img/s | **1.00x (Baseline)** | Serial nested spatial loops |
| **OpenMP CPU** | 16 CPU Threads | ~65–75 img/s | **~25x–28x** | 1D loop flattening & multi-core scheduling |
| **CUDA GPU** | NVIDIA RTX 4050 | ~210–240 img/s | **~80x–92x** | Thousands of GPU cores & persistent VRAM |

---

## 9. Repository Structure

```
cuTIN/
├── CMakeLists.txt              # Cross-platform build configuration (C++17, CUDA, OpenMP)
├── README.md                   # Project documentation and user manual
├── include/
│   ├── network.hpp             # Architecture constants, structs, timer, and backend APIs
│   └── stb_image.h             # Header-only image loader and decoder
├── src/
│   ├── dataloader.cpp          # Tiny ImageNet parser, words mapping, and image normalizer
│   ├── sequential.cpp          # Member 1: Sequential CPU implementation
│   ├── openmp.cpp              # Member 2: OpenMP multi-core CPU implementation
│   ├── cuda.cu                 # Member 3: CUDA GPU kernels and device pipelines
│   └── main.cpp                # CLI parser, benchmark runner, and inference engine
├── tests/
│   └── test_equivalence.cpp    # Unit and numerical equivalence verification test suite
├── tiny-imagenet-200/          # Dataset directory (wnids.txt, words.txt, train/, val/)
├── *.docx                      # Detailed architectural and code walkthrough documents:
│   ├── seq_walkthrough.docx
│   ├── openmp_walkthrough.docx
│   ├── cuda_walkthrough.docx
│   └── general_walkthrough.docx
└── build/                      # Compiled binary targets (cuTIN, test_equivalence)
```

---

## 10. Authors & Acknowledgments

Academic laboratory project developed to benchmark parallel computing paradigms (Sequential CPU, OpenMP Multi-Core, and NVIDIA CUDA) for deep convolutional neural networks from scratch.
