#include "network.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace cutin {

// =========================================================================
// OPENMP PARALLEL BACKEND (Member 2)
// Multi-core CPU acceleration using OpenMP parallel loops and reductions.
// Tensor Layout: NCHW | Filter Layout: OIHW
// =========================================================================

void omp_relu_forward(const float* x, float* y, size_t size) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(size); ++i)
        y[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
}

void omp_relu_backward(const float* dy, const float* y, float* dx, size_t size) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(size); ++i)
        dx[i] = (y[i] > 0.0f) ? dy[i] : 0.0f;
}

void omp_maxpool2d_forward(const float* x, float* y, int* mask,
                           int B, int C, int H_in, int W_in, int K, int stride) {
    int H_out = (H_in - K) / stride + 1;
    int W_out = (W_in - K) / stride + 1;
    long long total = 1LL * B * C * H_out * W_out;

    #pragma omp parallel for schedule(static)
    for (long long idx = 0; idx < total; ++idx) {
        int w_out = idx % W_out;
        long long tmp = idx / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int b = tmp / C;
        int h_start = h_out * stride;
        int w_start = w_out * stride;
        int in_offset = (b * C + c) * H_in * W_in;

        float max_val = -1e30f;
        int max_idx = -1;

        for (int kh = 0; kh < K; ++kh) {
            int h = h_start + kh;
            if (h >= H_in) continue;
            for (int kw = 0; kw < K; ++kw) {
                int w = w_start + kw;
                if (w >= W_in) continue;
                int in_idx = in_offset + h * W_in + w;
                if (x[in_idx] > max_val) {
                    max_val = x[in_idx];
                    max_idx = in_idx;
                }
            }
        }
        y[idx] = max_val;
        mask[idx] = max_idx;
    }
}

void omp_maxpool2d_backward(const float* dy, const int* mask, float* dx,
                            int B, int C, int H_out, int W_out, int H_in, int W_in, int K, int stride) {
    (void)K; (void)stride;
    long long total_in = 1LL * B * C * H_in * W_in;
    long long total_out = 1LL * B * C * H_out * W_out;

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < total_in; ++i) dx[i] = 0.0f;

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < total_out; ++i) {
        int target = mask[i];
        if (target >= 0 && target < total_in) {
            #pragma omp atomic update
            dx[target] += dy[i];
        }
    }
}

void omp_conv2d_forward(const float* x, const float* w, const float* b, float* y,
                        int B, int C_in, int H_in, int W_in,
                        int C_out, int K, int pad, int stride) {
    int H_out = (H_in + 2 * pad - K) / stride + 1;
    int W_out = (W_in + 2 * pad - K) / stride + 1;
    long long total = 1LL * B * C_out * H_out * W_out;

    #pragma omp parallel for schedule(static)
    for (long long idx = 0; idx < total; ++idx) {
        int w_out = idx % W_out;
        long long tmp = idx / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int c_out = tmp % C_out;
        int b_idx = tmp / C_out;

        float sum = b ? b[c_out] : 0.0f;
        int h_base = h_out * stride - pad;
        int w_base = w_out * stride - pad;

        for (int c = 0; c < C_in; ++c) {
            int x_ch = (b_idx * C_in + c) * H_in * W_in;
            int w_ch = (c_out * C_in + c) * K * K;
            for (int kh = 0; kh < K; ++kh) {
                int h = h_base + kh;
                if (h < 0 || h >= H_in) continue;
                for (int kw = 0; kw < K; ++kw) {
                    int wi = w_base + kw;
                    if (wi >= 0 && wi < W_in) {
                        sum += x[x_ch + h * W_in + wi] * w[w_ch + kh * K + kw];
                    }
                }
            }
        }
        y[idx] = sum;
    }
}

void omp_conv2d_backward(const float* dy, const float* x, const float* w,
                         float* dx, float* dw, float* db,
                         int B, int C_in, int H_in, int W_in,
                         int C_out, int K, int pad, int stride) {
    int H_out = (H_in + 2 * pad - K) / stride + 1;
    int W_out = (W_in + 2 * pad - K) / stride + 1;

    if (dx) {
        long long total_dx = 1LL * B * C_in * H_in * W_in;
        #pragma omp parallel for schedule(static)
        for (long long idx = 0; idx < total_dx; ++idx) {
            int wi = idx % W_in;
            long long tmp = idx / W_in;
            int hi = tmp % H_in;
            tmp /= H_in;
            int ci = tmp % C_in;
            int b_idx = tmp / C_in;

            float grad = 0.0f;
            for (int co = 0; co < C_out; ++co) {
                int dy_ch = (b_idx * C_out + co) * H_out * W_out;
                int w_ch  = (co * C_in + ci) * K * K;
                for (int kh = 0; kh < K; ++kh) {
                    int h_c = hi + pad - kh;
                    if (h_c % stride != 0) continue;
                    int ho = h_c / stride;
                    if (ho < 0 || ho >= H_out) continue;
                    for (int kw = 0; kw < K; ++kw) {
                        int w_c = wi + pad - kw;
                        if (w_c % stride != 0) continue;
                        int wo = w_c / stride;
                        if (wo >= 0 && wo < W_out) {
                            grad += dy[dy_ch + ho * W_out + wo] * w[w_ch + kh * K + kw];
                        }
                    }
                }
            }
            dx[idx] = grad;
        }
    }

    if (dw) {
        long long total_dw = 1LL * C_out * C_in * K * K;
        #pragma omp parallel for schedule(static)
        for (long long idx = 0; idx < total_dw; ++idx) {
            int kw = idx % K;
            long long tmp = idx / K;
            int kh = tmp % K;
            tmp /= K;
            int ci = tmp % C_in;
            int co = tmp / C_in;

            float grad = 0.0f;
            for (int b = 0; b < B; ++b) {
                int dy_ch = (b * C_out + co) * H_out * W_out;
                int x_ch  = (b * C_in + ci) * H_in * W_in;
                for (int ho = 0; ho < H_out; ++ho) {
                    int hi = ho * stride - pad + kh;
                    if (hi < 0 || hi >= H_in) continue;
                    for (int wo = 0; wo < W_out; ++wo) {
                        int wi = wo * stride - pad + kw;
                        if (wi >= 0 && wi < W_in) {
                            grad += dy[dy_ch + ho * W_out + wo] * x[x_ch + hi * W_in + wi];
                        }
                    }
                }
            }
            dw[idx] = grad;
        }
    }

    if (db) {
        #pragma omp parallel for schedule(static)
        for (int co = 0; co < C_out; ++co) {
            float grad = 0.0f;
            for (int b = 0; b < B; ++b) {
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        grad += dy[((b * C_out + co) * H_out + h) * W_out + w];
                    }
                }
            }
            db[co] = grad;
        }
    }
}

void omp_linear_forward(const float* x, const float* w, const float* b, float* y,
                        int B, int in_f, int out_f) {
    long long total = 1LL * B * out_f;
    #pragma omp parallel for schedule(static)
    for (long long idx = 0; idx < total; ++idx) {
        int out = idx % out_f;
        int b_idx = idx / out_f;
        float sum = b ? b[out] : 0.0f;
        const float* x_row = x + static_cast<size_t>(b_idx) * in_f;
        const float* w_row = w + static_cast<size_t>(out) * in_f;
        for (int k = 0; k < in_f; ++k) sum += x_row[k] * w_row[k];
        y[idx] = sum;
    }
}

void omp_linear_backward(const float* dy, const float* x, const float* w,
                         float* dx, float* dw, float* db,
                         int B, int in_f, int out_f) {
    if (dx) {
        long long total = 1LL * B * in_f;
        #pragma omp parallel for schedule(static)
        for (long long idx = 0; idx < total; ++idx) {
            int in = idx % in_f;
            int b_idx = idx / in_f;
            float sum = 0.0f;
            for (int out = 0; out < out_f; ++out) {
                sum += dy[b_idx * out_f + out] * w[out * in_f + in];
            }
            dx[idx] = sum;
        }
    }
    if (dw) {
        long long total = 1LL * out_f * in_f;
        #pragma omp parallel for schedule(static)
        for (long long idx = 0; idx < total; ++idx) {
            int in = idx % in_f;
            int out = idx / in_f;
            float sum = 0.0f;
            for (int b = 0; b < B; ++b) {
                sum += dy[b * out_f + out] * x[b * in_f + in];
            }
            dw[idx] = sum;
        }
    }
    if (db) {
        #pragma omp parallel for schedule(static)
        for (int out = 0; out < out_f; ++out) {
            float sum = 0.0f;
            for (int b = 0; b < B; ++b) sum += dy[b * out_f + out];
            db[out] = sum;
        }
    }
}

void omp_softmax_cross_entropy_forward_backward(const float* logits, const int* labels,
                                                float* probs, float* dlogits,
                                                float* batch_loss, int B, int num_classes) {
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; ++b) {
        const float* row = logits + static_cast<size_t>(b) * num_classes;
        float* p_row = probs + static_cast<size_t>(b) * num_classes;
        float* dl_row = dlogits + static_cast<size_t>(b) * num_classes;
        int target = labels[b];

        float max_val = -1e30f;
        for (int c = 0; c < num_classes; ++c) if (row[c] > max_val) max_val = row[c];

        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            p_row[c] = std::exp(row[c] - max_val);
            sum_exp += p_row[c];
        }

        float inv_sum = 1.0f / (sum_exp + 1e-12f);
        float inv_B = 1.0f / static_cast<float>(B);

        for (int c = 0; c < num_classes; ++c) {
            float p = p_row[c] * inv_sum;
            p_row[c] = p;
            dl_row[c] = (p - (c == target ? 1.0f : 0.0f)) * inv_B;
        }

        if (batch_loss) {
            if (target >= 0 && target < num_classes) {
                batch_loss[b] = -std::log(std::max(p_row[target], 1e-12f));
            } else {
                batch_loss[b] = 0.0f;
            }
        }
    }
}

void omp_sgd_update(float* w, float* dw, float* vw,
                    float* b, float* db, float* vb,
                    size_t w_size, size_t b_size,
                    float lr, float momentum, float weight_decay) {
    if (w && dw && vw) {
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(w_size); ++i) {
            float g = dw[i] + weight_decay * w[i];
            vw[i] = momentum * vw[i] + g;
            w[i] -= lr * vw[i];
        }
    }
    if (b && db && vb) {
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(b_size); ++i) {
            vb[i] = momentum * vb[i] + db[i];
            b[i] -= lr * vb[i];
        }
    }
}

// =========================================================================
// OPENMP WORKSPACE & HIGH-LEVEL PIPELINE
// =========================================================================
namespace {
struct OmpBuffers {
    int B;
    std::vector<float> input, c1, p1, c2, p2, fc1, fc1_relu, fc2, probs, loss;
    std::vector<int> labels, p1_mask, p2_mask;
    std::vector<float> dlogits, dfc1_relu, dfc1, dp2, dc2, dp1, dc1;

    explicit OmpBuffers(int b) : B(b),
        input(b * ArchConfig::IN_C * ArchConfig::IN_H * ArchConfig::IN_W),
        c1(b * ArchConfig::C1_OUT_C * ArchConfig::C1_OUT_H * ArchConfig::C1_OUT_W),
        p1(b * ArchConfig::C1_OUT_C * ArchConfig::P1_OUT_H * ArchConfig::P1_OUT_W),
        c2(b * ArchConfig::C2_OUT_C * ArchConfig::C2_OUT_H * ArchConfig::C2_OUT_W),
        p2(b * ArchConfig::FLATTEN_SIZE),
        fc1(b * ArchConfig::FC1_OUT), fc1_relu(b * ArchConfig::FC1_OUT),
        fc2(b * ArchConfig::FC2_OUT), probs(b * ArchConfig::FC2_OUT),
        loss(b), labels(b),
        p1_mask(b * ArchConfig::C1_OUT_C * ArchConfig::P1_OUT_H * ArchConfig::P1_OUT_W),
        p2_mask(b * ArchConfig::FLATTEN_SIZE),
        dlogits(b * ArchConfig::FC2_OUT), dfc1_relu(b * ArchConfig::FC1_OUT),
        dfc1(b * ArchConfig::FC1_OUT), dp2(b * ArchConfig::FLATTEN_SIZE),
        dc2(b * ArchConfig::C2_OUT_C * ArchConfig::C2_OUT_H * ArchConfig::C2_OUT_W),
        dp1(b * ArchConfig::C1_OUT_C * ArchConfig::P1_OUT_H * ArchConfig::P1_OUT_W),
        dc1(b * ArchConfig::C1_OUT_C * ArchConfig::C1_OUT_H * ArchConfig::C1_OUT_W) {}
};

void forward_batch(const float* x, const ModelParameters& p, OmpBuffers& b, bool compute_loss, const int* labels) {
    int B = b.B;
    omp_conv2d_forward(x, p.w_c1, p.b_c1, b.c1.data(), B, ArchConfig::IN_C, ArchConfig::IN_H, ArchConfig::IN_W,
                       ArchConfig::C1_OUT_C, ArchConfig::C1_K, ArchConfig::C1_PAD, ArchConfig::C1_STRIDE);
    omp_relu_forward(b.c1.data(), b.c1.data(), b.c1.size());
    omp_maxpool2d_forward(b.c1.data(), b.p1.data(), b.p1_mask.data(), B, ArchConfig::C1_OUT_C, ArchConfig::C1_OUT_H, ArchConfig::C1_OUT_W, ArchConfig::P1_K, ArchConfig::P1_STRIDE);

    omp_conv2d_forward(b.p1.data(), p.w_c2, p.b_c2, b.c2.data(), B, ArchConfig::C1_OUT_C, ArchConfig::P1_OUT_H, ArchConfig::P1_OUT_W,
                       ArchConfig::C2_OUT_C, ArchConfig::C2_K, ArchConfig::C2_PAD, ArchConfig::C2_STRIDE);
    omp_relu_forward(b.c2.data(), b.c2.data(), b.c2.size());
    omp_maxpool2d_forward(b.c2.data(), b.p2.data(), b.p2_mask.data(), B, ArchConfig::C2_OUT_C, ArchConfig::C2_OUT_H, ArchConfig::C2_OUT_W, ArchConfig::P2_K, ArchConfig::P2_STRIDE);

    omp_linear_forward(b.p2.data(), p.w_fc1, p.b_fc1, b.fc1.data(), B, ArchConfig::FLATTEN_SIZE, ArchConfig::FC1_OUT);
    omp_relu_forward(b.fc1.data(), b.fc1_relu.data(), b.fc1.size());
    omp_linear_forward(b.fc1_relu.data(), p.w_fc2, p.b_fc2, b.fc2.data(), B, ArchConfig::FC1_OUT, ArchConfig::FC2_OUT);

    omp_softmax_cross_entropy_forward_backward(b.fc2.data(), labels, b.probs.data(), b.dlogits.data(),
                                               compute_loss ? b.loss.data() : nullptr, B, ArchConfig::FC2_OUT);
}

void backward_and_update(OmpBuffers& b, ModelParameters& p, const TrainingConfig& cfg) {
    int B = b.B;
    omp_linear_backward(b.dlogits.data(), b.fc1_relu.data(), p.w_fc2, b.dfc1_relu.data(), p.dw_fc2, p.db_fc2, B, ArchConfig::FC1_OUT, ArchConfig::FC2_OUT);
    omp_relu_backward(b.dfc1_relu.data(), b.fc1_relu.data(), b.dfc1.data(), b.fc1_relu.size());
    omp_linear_backward(b.dfc1.data(), b.p2.data(), p.w_fc1, b.dp2.data(), p.dw_fc1, p.db_fc1, B, ArchConfig::FLATTEN_SIZE, ArchConfig::FC1_OUT);

    omp_maxpool2d_backward(b.dp2.data(), b.p2_mask.data(), b.dc2.data(), B, ArchConfig::C2_OUT_C, ArchConfig::P2_OUT_H, ArchConfig::P2_OUT_W, ArchConfig::C2_OUT_H, ArchConfig::C2_OUT_W, ArchConfig::P2_K, ArchConfig::P2_STRIDE);
    omp_relu_backward(b.dc2.data(), b.c2.data(), b.dc2.data(), b.c2.size());
    omp_conv2d_backward(b.dc2.data(), b.p1.data(), p.w_c2, b.dp1.data(), p.dw_c2, p.db_c2, B, ArchConfig::C1_OUT_C, ArchConfig::P1_OUT_H, ArchConfig::P1_OUT_W, ArchConfig::C2_OUT_C, ArchConfig::C2_K, ArchConfig::C2_PAD, ArchConfig::C2_STRIDE);

    omp_maxpool2d_backward(b.dp1.data(), b.p1_mask.data(), b.dc1.data(), B, ArchConfig::C1_OUT_C, ArchConfig::P1_OUT_H, ArchConfig::P1_OUT_W, ArchConfig::C1_OUT_H, ArchConfig::C1_OUT_W, ArchConfig::P1_K, ArchConfig::P1_STRIDE);
    omp_relu_backward(b.dc1.data(), b.c1.data(), b.dc1.data(), b.c1.size());
    omp_conv2d_backward(b.dc1.data(), b.input.data(), p.w_c1, nullptr, p.dw_c1, p.db_c1, B, ArchConfig::IN_C, ArchConfig::IN_H, ArchConfig::IN_W, ArchConfig::C1_OUT_C, ArchConfig::C1_K, ArchConfig::C1_PAD, ArchConfig::C1_STRIDE);

    omp_sgd_update(p.w_c1, p.dw_c1, p.vw_c1, p.b_c1, p.db_c1, p.vb_c1, ArchConfig::C1_W_SIZE, ArchConfig::C1_B_SIZE, cfg.learning_rate, cfg.momentum, cfg.weight_decay);
    omp_sgd_update(p.w_c2, p.dw_c2, p.vw_c2, p.b_c2, p.db_c2, p.vb_c2, ArchConfig::C2_W_SIZE, ArchConfig::C2_B_SIZE, cfg.learning_rate, cfg.momentum, cfg.weight_decay);
    omp_sgd_update(p.w_fc1, p.dw_fc1, p.vw_fc1, p.b_fc1, p.db_fc1, p.vb_fc1, ArchConfig::FC1_W_SIZE, ArchConfig::FC1_B_SIZE, cfg.learning_rate, cfg.momentum, cfg.weight_decay);
    omp_sgd_update(p.w_fc2, p.dw_fc2, p.vw_fc2, p.b_fc2, p.db_fc2, p.vb_fc2, ArchConfig::FC2_W_SIZE, ArchConfig::FC2_B_SIZE, cfg.learning_rate, cfg.momentum, cfg.weight_decay);
}
} // namespace

float omp_evaluate(const TinyImageNetDataset& val_data, const ModelParameters& params, int batch_size) {
    if (val_data.size() == 0 || batch_size <= 0) return 0.0f;
    OmpBuffers b(batch_size);
    int correct = 0, evaluated = 0;

    for (size_t start = 0; start < val_data.size(); start += batch_size) {
        int actual = val_data.get_batch(start, batch_size, b.input.data(), b.labels.data());
        if (actual == 0) break;
        forward_batch(b.input.data(), params, b, false, b.labels.data());

        int b_hits = 0;
        #pragma omp parallel for reduction(+:b_hits) schedule(static)
        for (int i = 0; i < actual; ++i) {
            int pred = 0; float best = -1.0f;
            for (int c = 0; c < ArchConfig::FC2_OUT; ++c) {
                float v = b.probs[i * ArchConfig::FC2_OUT + c];
                if (v > best) { best = v; pred = c; }
            }
            if (pred == b.labels[i]) b_hits++;
        }
        correct += b_hits;
        evaluated += actual;
    }
    return evaluated ? (100.0f * correct / evaluated) : 0.0f;
}

void omp_train(const TinyImageNetDataset& train_data, const TinyImageNetDataset& val_data,
               ModelParameters& params, const TrainingConfig& config) {
    std::cout << "\n=======================================================\n"
              << " Starting Deep CNN Training using OpenMP\n"
              << " Classes: " << ArchConfig::FC2_OUT << " | Batch: " << config.batch_size
              << " | Epochs: " << config.epochs << " | LR: " << config.learning_rate << "\n";
#ifdef _OPENMP
    std::cout << " OpenMP Threads: " << omp_get_max_threads() << "\n";
#endif
    std::cout << "=======================================================\n";

    OmpBuffers b(config.batch_size);
    CPUTimer epoch_timer, batch_timer;
    size_t total_batches = (train_data.size() + config.batch_size - 1) / config.batch_size;

    std::string stats_file = config.stats_file.empty() ? "openmp_stats.csv" : config.stats_file;
    std::ofstream csv(stats_file);
    if (csv.is_open()) {
        csv << "epoch,train_loss,train_acc_pct,train_error_pct,val_acc_pct,val_error_pct,epoch_time_sec,throughput_img_sec\n";
    }

    for (int epoch = 1; epoch <= config.epochs; ++epoch) {
        epoch_timer.start();
        float epoch_loss = 0.0f;
        int epoch_correct = 0, samples_processed = 0;
        const_cast<TinyImageNetDataset&>(train_data).shuffle(epoch * 1000 + 42);

        for (size_t b_idx = 0; b_idx < total_batches; ++b_idx) {
            batch_timer.start();
            size_t start = b_idx * config.batch_size;
            int actual = train_data.get_batch(start, config.batch_size, b.input.data(), b.labels.data());
            if (actual == 0) break;

            forward_batch(b.input.data(), params, b, true, b.labels.data());
            backward_and_update(b, params, config);

            float b_loss = 0.0f;
            for (int i = 0; i < actual; ++i) b_loss += b.loss[i];

            int b_hits = 0;
            #pragma omp parallel for reduction(+:b_hits) schedule(static)
            for (int i = 0; i < actual; ++i) {
                int pred = 0; float best = -1.0f;
                for (int c = 0; c < ArchConfig::FC2_OUT; ++c) {
                    float v = b.probs[i * ArchConfig::FC2_OUT + c];
                    if (v > best) { best = v; pred = c; }
                }
                if (pred == b.labels[i]) b_hits++;
            }

            epoch_loss += b_loss;
            epoch_correct += b_hits;
            samples_processed += actual;

            if (b_idx % 10 == 0 || b_idx == total_batches - 1) {
                double ms = batch_timer.stop();
                float avg_l = b_loss / actual;
                float acc = 100.0f * b_hits / actual;
                float throughput = actual / (ms / 1000.0f);
                std::cout << "\r[Epoch " << epoch << "/" << config.epochs << "] Batch " << (b_idx + 1) << "/" << total_batches
                          << " | Loss: " << std::fixed << std::setprecision(4) << avg_l
                          << " | Acc: " << std::setprecision(2) << acc << "%"
                          << " | " << std::setprecision(1) << throughput << " img/s" << std::flush;
            }
        }

        double ep_ms = epoch_timer.stop();
        if (samples_processed == 0) break;
        float tr_loss = epoch_loss / samples_processed;
        float tr_acc = 100.0f * epoch_correct / samples_processed;
        float tr_err = 100.0f - tr_acc;
        float val_acc = 0.0f;
        float val_err = 0.0f;

        std::cout << "\n--> Epoch " << epoch << " Finished in " << std::fixed << std::setprecision(2) << ep_ms / 1000.0 << "s"
                  << " | Train Loss: " << tr_loss
                  << " | Train Acc: " << tr_acc << "%\n";

        if (val_data.size() > 0) {
            val_acc = omp_evaluate(val_data, params, config.batch_size);
            val_err = 100.0f - val_acc;
            std::cout << "--> Validation Accuracy: " << std::setprecision(2) << val_acc << "%\n";
        }

        double ep_sec = ep_ms / 1000.0;
        double throughput = (ep_sec > 0.0) ? (samples_processed / ep_sec) : 0.0;
        if (csv.is_open()) {
            csv << epoch << "," << std::fixed << std::setprecision(4) << tr_loss << ","
                << std::setprecision(2) << tr_acc << "," << tr_err << ","
                << val_acc << "," << val_err << ","
                << ep_sec << "," << throughput << "\n";
            csv.flush();
        }
    }

    std::string model_file = config.model_save_path.empty() ? "openmp_model.bin" : config.model_save_path;
    if (params.save_to_file(model_file)) {
        std::cout << "--> Model checkpoint successfully saved to: " << model_file << "\n";
    }
    if (csv.is_open()) {
        std::cout << "--> Epoch training & testing statistics saved to: " << stats_file << "\n";
    }

    std::cout << "=======================================================\n"
              << " OpenMP Training Completed Successfully!\n"
              << "=======================================================\n";
}

} // namespace cutin
