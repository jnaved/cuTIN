#include "network.hpp"

namespace cutin {

// =========================================================================
// SEQUENTIAL BACKEND (Member 1)
// Clean CPU implementation using standard serial nested loops.
// Tensor Layout: NCHW | Filter Layout: OIHW
// =========================================================================

void seq_relu_forward(const float* x, float* y, size_t size) {
    for (size_t i = 0; i < size; ++i) y[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
}

void seq_relu_backward(const float* dy, const float* y, float* dx, size_t size) {
    for (size_t i = 0; i < size; ++i) dx[i] = (y[i] > 0.0f) ? dy[i] : 0.0f;
}

void seq_maxpool2d_forward(const float* x, float* y, int* mask,
                           int B, int C, int H_in, int W_in, int K, int stride) {
    int H_out = (H_in - K) / stride + 1;
    int W_out = (W_in - K) / stride + 1;

    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            int in_ch_offset = (b * C + c) * H_in * W_in;
            for (int ho = 0; ho < H_out; ++ho) {
                for (int wo = 0; wo < W_out; ++wo) {
                    float max_val = -1e30f;
                    int max_idx = -1;
                    int h_start = ho * stride;
                    int w_start = wo * stride;

                    for (int kh = 0; kh < K; ++kh) {
                        for (int kw = 0; kw < K; ++kw) {
                            int hi = h_start + kh;
                            int wi = w_start + kw;
                            int in_idx = in_ch_offset + hi * W_in + wi;
                            if (x[in_idx] > max_val) {
                                max_val = x[in_idx];
                                max_idx = in_idx;
                            }
                        }
                    }
                    int out_idx = ((b * C + c) * H_out + ho) * W_out + wo;
                    y[out_idx] = max_val;
                    mask[out_idx] = max_idx;
                }
            }
        }
    }
}

void seq_maxpool2d_backward(const float* dy, const int* mask, float* dx,
                            int B, int C, int H_out, int W_out, int H_in, int W_in, int K, int stride) {
    (void)K; (void)stride;
    int total_in = B * C * H_in * W_in;
    int total_out = B * C * H_out * W_out;
    for (int i = 0; i < total_in; ++i) dx[i] = 0.0f;
    for (int i = 0; i < total_out; ++i) {
        int idx = mask[i];
        if (idx >= 0 && idx < total_in) dx[idx] += dy[i];
    }
}

void seq_conv2d_forward(const float* x, const float* w, const float* b, float* y,
                        int B, int C_in, int H_in, int W_in,
                        int C_out, int K, int pad, int stride) {
    int H_out = (H_in + 2 * pad - K) / stride + 1;
    int W_out = (W_in + 2 * pad - K) / stride + 1;

    for (int b_idx = 0; b_idx < B; ++b_idx) {
        for (int co = 0; co < C_out; ++co) {
            float bias_val = b ? b[co] : 0.0f;
            for (int ho = 0; ho < H_out; ++ho) {
                for (int wo = 0; wo < W_out; ++wo) {
                    float sum = bias_val;
                    for (int ci = 0; ci < C_in; ++ci) {
                        for (int kh = 0; kh < K; ++kh) {
                            int hi = ho * stride + kh - pad;
                            if (hi < 0 || hi >= H_in) continue;
                            for (int kw = 0; kw < K; ++kw) {
                                int wi = wo * stride + kw - pad;
                                if (wi >= 0 && wi < W_in) {
                                    sum += x[((b_idx * C_in + ci) * H_in + hi) * W_in + wi]
                                         * w[((co * C_in + ci) * K + kh) * K + kw];
                                }
                            }
                        }
                    }
                    y[((b_idx * C_out + co) * H_out + ho) * W_out + wo] = sum;
                }
            }
        }
    }
}

void seq_conv2d_backward(const float* dy, const float* x, const float* w,
                         float* dx, float* dw, float* db,
                         int B, int C_in, int H_in, int W_in,
                         int C_out, int K, int pad, int stride) {
    int H_out = (H_in + 2 * pad - K) / stride + 1;
    int W_out = (W_in + 2 * pad - K) / stride + 1;

    if (dx) std::fill(dx, dx + B * C_in * H_in * W_in, 0.0f);
    if (dw) std::fill(dw, dw + C_out * C_in * K * K, 0.0f);
    if (db) std::fill(db, db + C_out, 0.0f);

    for (int b_idx = 0; b_idx < B; ++b_idx) {
        for (int co = 0; co < C_out; ++co) {
            for (int ho = 0; ho < H_out; ++ho) {
                for (int wo = 0; wo < W_out; ++wo) {
                    float grad = dy[((b_idx * C_out + co) * H_out + ho) * W_out + wo];
                    if (db) db[co] += grad;

                    for (int ci = 0; ci < C_in; ++ci) {
                        for (int kh = 0; kh < K; ++kh) {
                            int hi = ho * stride + kh - pad;
                            if (hi < 0 || hi >= H_in) continue;
                            for (int kw = 0; kw < K; ++kw) {
                                int wi = wo * stride + kw - pad;
                                if (wi >= 0 && wi < W_in) {
                                    int in_idx = ((b_idx * C_in + ci) * H_in + hi) * W_in + wi;
                                    int w_idx  = ((co * C_in + ci) * K + kh) * K + kw;
                                    if (dx) dx[in_idx] += grad * w[w_idx];
                                    if (dw) dw[w_idx]  += grad * x[in_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void seq_linear_forward(const float* x, const float* w, const float* b, float* y,
                        int B, int in_f, int out_f) {
    for (int b_idx = 0; b_idx < B; ++b_idx) {
        for (int j = 0; j < out_f; ++j) {
            float sum = b ? b[j] : 0.0f;
            for (int k = 0; k < in_f; ++k) sum += x[b_idx * in_f + k] * w[j * in_f + k];
            y[b_idx * out_f + j] = sum;
        }
    }
}

void seq_linear_backward(const float* dy, const float* x, const float* w,
                         float* dx, float* dw, float* db,
                         int B, int in_f, int out_f) {
    if (dx) {
        for (int b_idx = 0; b_idx < B; ++b_idx) {
            for (int k = 0; k < in_f; ++k) {
                float sum = 0.0f;
                for (int j = 0; j < out_f; ++j) sum += dy[b_idx * out_f + j] * w[j * in_f + k];
                dx[b_idx * in_f + k] = sum;
            }
        }
    }
    if (dw) {
        for (int j = 0; j < out_f; ++j) {
            for (int k = 0; k < in_f; ++k) {
                float sum = 0.0f;
                for (int b_idx = 0; b_idx < B; ++b_idx) sum += dy[b_idx * out_f + j] * x[b_idx * in_f + k];
                dw[j * in_f + k] = sum;
            }
        }
    }
    if (db) {
        for (int j = 0; j < out_f; ++j) {
            float sum = 0.0f;
            for (int b_idx = 0; b_idx < B; ++b_idx) sum += dy[b_idx * out_f + j];
            db[j] = sum;
        }
    }
}

void seq_softmax_cross_entropy_forward_backward(const float* logits, const int* labels,
                                                float* probs, float* dlogits,
                                                float* batch_loss, int B, int num_classes) {
    for (int b = 0; b < B; ++b) {
        const float* row = logits + b * num_classes;
        float* p_row = probs + b * num_classes;
        float* dl_row = dlogits + b * num_classes;
        int target = labels[b];

        float max_val = -1e30f;
        for (int c = 0; c < num_classes; ++c) if (row[c] > max_val) max_val = row[c];

        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            float val = std::exp(row[c] - max_val);
            p_row[c] = val;
            sum_exp += val;
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

void seq_sgd_update(float* w, float* dw, float* vw,
                    float* b, float* db, float* vb,
                    size_t w_size, size_t b_size,
                    float lr, float momentum, float weight_decay) {
    if (w && dw && vw) {
        for (size_t i = 0; i < w_size; ++i) {
            float g = dw[i] + weight_decay * w[i];
            vw[i] = momentum * vw[i] + g;
            w[i] -= lr * vw[i];
        }
    }
    if (b && db && vb) {
        for (size_t i = 0; i < b_size; ++i) {
            vb[i] = momentum * vb[i] + db[i];
            b[i] -= lr * vb[i];
        }
    }
}

// =========================================================================
// SEQUENTIAL WORKSPACE & HIGH-LEVEL PIPELINE
// =========================================================================
namespace {
struct SeqBuffers {
    int B;
    std::vector<float> input, c1, p1, c2, p2, fc1, fc1_relu, fc2, probs, loss;
    std::vector<int> labels, p1_mask, p2_mask;
    std::vector<float> dlogits, dfc1_relu, dfc1, dp2, dc2, dp1, dc1;

    explicit SeqBuffers(int b) : B(b),
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

void forward_batch(const float* x, const ModelParameters& p, SeqBuffers& b, bool compute_loss, const int* labels) {
    int B = b.B;
    seq_conv2d_forward(x, p.w_c1, p.b_c1, b.c1.data(), B, ArchConfig::IN_C, ArchConfig::IN_H, ArchConfig::IN_W,
                       ArchConfig::C1_OUT_C, ArchConfig::C1_K, ArchConfig::C1_PAD, ArchConfig::C1_STRIDE);
    seq_relu_forward(b.c1.data(), b.c1.data(), b.c1.size());
    seq_maxpool2d_forward(b.c1.data(), b.p1.data(), b.p1_mask.data(), B, ArchConfig::C1_OUT_C, ArchConfig::C1_OUT_H, ArchConfig::C1_OUT_W, ArchConfig::P1_K, ArchConfig::P1_STRIDE);

    seq_conv2d_forward(b.p1.data(), p.w_c2, p.b_c2, b.c2.data(), B, ArchConfig::C1_OUT_C, ArchConfig::P1_OUT_H, ArchConfig::P1_OUT_W,
                       ArchConfig::C2_OUT_C, ArchConfig::C2_K, ArchConfig::C2_PAD, ArchConfig::C2_STRIDE);
    seq_relu_forward(b.c2.data(), b.c2.data(), b.c2.size());
    seq_maxpool2d_forward(b.c2.data(), b.p2.data(), b.p2_mask.data(), B, ArchConfig::C2_OUT_C, ArchConfig::C2_OUT_H, ArchConfig::C2_OUT_W, ArchConfig::P2_K, ArchConfig::P2_STRIDE);

    seq_linear_forward(b.p2.data(), p.w_fc1, p.b_fc1, b.fc1.data(), B, ArchConfig::FLATTEN_SIZE, ArchConfig::FC1_OUT);
    seq_relu_forward(b.fc1.data(), b.fc1_relu.data(), b.fc1.size());
    seq_linear_forward(b.fc1_relu.data(), p.w_fc2, p.b_fc2, b.fc2.data(), B, ArchConfig::FC1_OUT, ArchConfig::FC2_OUT);

    seq_softmax_cross_entropy_forward_backward(b.fc2.data(), labels, b.probs.data(), b.dlogits.data(),
                                               compute_loss ? b.loss.data() : nullptr, B, ArchConfig::FC2_OUT);
}

void backward_and_update(SeqBuffers& b, ModelParameters& p, const TrainingConfig& cfg) {
    int B = b.B;
    seq_linear_backward(b.dlogits.data(), b.fc1_relu.data(), p.w_fc2, b.dfc1_relu.data(), p.dw_fc2, p.db_fc2, B, ArchConfig::FC1_OUT, ArchConfig::FC2_OUT);
    seq_relu_backward(b.dfc1_relu.data(), b.fc1_relu.data(), b.dfc1.data(), b.fc1_relu.size());
    seq_linear_backward(b.dfc1.data(), b.p2.data(), p.w_fc1, b.dp2.data(), p.dw_fc1, p.db_fc1, B, ArchConfig::FLATTEN_SIZE, ArchConfig::FC1_OUT);

    seq_maxpool2d_backward(b.dp2.data(), b.p2_mask.data(), b.dc2.data(), B, ArchConfig::C2_OUT_C, ArchConfig::P2_OUT_H, ArchConfig::P2_OUT_W, ArchConfig::C2_OUT_H, ArchConfig::C2_OUT_W, ArchConfig::P2_K, ArchConfig::P2_STRIDE);
    seq_relu_backward(b.dc2.data(), b.c2.data(), b.dc2.data(), b.c2.size());
    seq_conv2d_backward(b.dc2.data(), b.p1.data(), p.w_c2, b.dp1.data(), p.dw_c2, p.db_c2, B, ArchConfig::C1_OUT_C, ArchConfig::P1_OUT_H, ArchConfig::P1_OUT_W, ArchConfig::C2_OUT_C, ArchConfig::C2_K, ArchConfig::C2_PAD, ArchConfig::C2_STRIDE);

    seq_maxpool2d_backward(b.dp1.data(), b.p1_mask.data(), b.dc1.data(), B, ArchConfig::C1_OUT_C, ArchConfig::P1_OUT_H, ArchConfig::P1_OUT_W, ArchConfig::C1_OUT_H, ArchConfig::C1_OUT_W, ArchConfig::P1_K, ArchConfig::P1_STRIDE);
    seq_relu_backward(b.dc1.data(), b.c1.data(), b.dc1.data(), b.c1.size());
    seq_conv2d_backward(b.dc1.data(), b.input.data(), p.w_c1, nullptr, p.dw_c1, p.db_c1, B, ArchConfig::IN_C, ArchConfig::IN_H, ArchConfig::IN_W, ArchConfig::C1_OUT_C, ArchConfig::C1_K, ArchConfig::C1_PAD, ArchConfig::C1_STRIDE);

    seq_sgd_update(p.w_c1, p.dw_c1, p.vw_c1, p.b_c1, p.db_c1, p.vb_c1, ArchConfig::C1_W_SIZE, ArchConfig::C1_B_SIZE, cfg.learning_rate, cfg.momentum, cfg.weight_decay);
    seq_sgd_update(p.w_c2, p.dw_c2, p.vw_c2, p.b_c2, p.db_c2, p.vb_c2, ArchConfig::C2_W_SIZE, ArchConfig::C2_B_SIZE, cfg.learning_rate, cfg.momentum, cfg.weight_decay);
    seq_sgd_update(p.w_fc1, p.dw_fc1, p.vw_fc1, p.b_fc1, p.db_fc1, p.vb_fc1, ArchConfig::FC1_W_SIZE, ArchConfig::FC1_B_SIZE, cfg.learning_rate, cfg.momentum, cfg.weight_decay);
    seq_sgd_update(p.w_fc2, p.dw_fc2, p.vw_fc2, p.b_fc2, p.db_fc2, p.vb_fc2, ArchConfig::FC2_W_SIZE, ArchConfig::FC2_B_SIZE, cfg.learning_rate, cfg.momentum, cfg.weight_decay);
}
} // namespace

float seq_evaluate(const TinyImageNetDataset& val_data, const ModelParameters& params, int batch_size) {
    if (val_data.size() == 0 || batch_size <= 0) return 0.0f;
    SeqBuffers b(batch_size);
    int correct = 0, evaluated = 0;

    for (size_t start = 0; start < val_data.size(); start += batch_size) {
        int actual = val_data.get_batch(start, batch_size, b.input.data(), b.labels.data());
        if (actual == 0) break;
        forward_batch(b.input.data(), params, b, false, b.labels.data());

        for (int i = 0; i < actual; ++i) {
            int pred = 0; float best = -1.0f;
            for (int c = 0; c < ArchConfig::FC2_OUT; ++c) {
                float v = b.probs[i * ArchConfig::FC2_OUT + c];
                if (v > best) { best = v; pred = c; }
            }
            if (pred == b.labels[i]) correct++;
        }
        evaluated += actual;
    }
    return evaluated ? (100.0f * correct / evaluated) : 0.0f;
}

void seq_train(const TinyImageNetDataset& train_data, const TinyImageNetDataset& val_data,
               ModelParameters& params, const TrainingConfig& config) {
    std::cout << "\n=======================================================\n"
              << " Starting Deep CNN Training using Sequential C++\n"
              << " Classes: " << ArchConfig::FC2_OUT << " | Batch: " << config.batch_size
              << " | Epochs: " << config.epochs << " | LR: " << config.learning_rate << "\n"
              << "=======================================================\n";

    SeqBuffers b(config.batch_size);
    CPUTimer epoch_timer, batch_timer;
    size_t total_batches = (train_data.size() + config.batch_size - 1) / config.batch_size;

    std::string stats_file = config.stats_file.empty() ? "seq_stats.csv" : config.stats_file;
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
            int b_hits = 0;
            for (int i = 0; i < actual; ++i) {
                b_loss += b.loss[i];
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
            val_acc = seq_evaluate(val_data, params, config.batch_size);
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

    std::string model_file = config.model_save_path.empty() ? "seq_model.bin" : config.model_save_path;
    if (params.save_to_file(model_file)) {
        std::cout << "--> Model checkpoint successfully saved to: " << model_file << "\n";
    }
    if (csv.is_open()) {
        std::cout << "--> Epoch training & testing statistics saved to: " << stats_file << "\n";
    }

    std::cout << "=======================================================\n"
              << " Sequential Training Completed Successfully!\n"
              << "=======================================================\n";
}

} // namespace cutin
