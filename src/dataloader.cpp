#include "network.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <random>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#include "stb_image.h"

namespace fs = std::filesystem;

namespace cutin {

bool TinyImageNetDataset::load_class_metadata(const std::string& data_root) {
    std::string wnids_path = data_root + "/wnids.txt";
    std::ifstream wnids_file(wnids_path);
    if (!wnids_file.is_open()) return false;

    wnids_.clear();
    wnid_to_idx_.clear();
    std::string line;
    int idx = 0;
    while (std::getline(wnids_file, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (!line.empty()) {
            wnids_.push_back(line);
            wnid_to_idx_[line] = idx++;
        }
    }
    return !wnids_.empty();
}

bool TinyImageNetDataset::load(const std::string& data_root, bool is_train, int max_samples) {
    if (!load_class_metadata(data_root)) return false;

    samples_.clear();
    if (is_train) {
        std::string train_dir = data_root + "/train";
        for (const auto& wnid : wnids_) {
            int label = wnid_to_idx_[wnid];
            std::string img_dir = train_dir + "/" + wnid + "/images";
            if (fs::exists(img_dir) && fs::is_directory(img_dir)) {
                for (const auto& entry : fs::directory_iterator(img_dir)) {
                    std::string p = entry.path().string();
                    if (p.find(".JPEG") != std::string::npos || p.find(".jpg") != std::string::npos) {
                        samples_.push_back({p, label});
                    }
                }
            }
        }
    } else {
        std::string val_annot = data_root + "/val/val_annotations.txt";
        std::ifstream f(val_annot);
        if (!f.is_open()) return false;
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream iss(line);
            std::string filename, wnid;
            if (iss >> filename >> wnid && wnid_to_idx_.count(wnid)) {
                samples_.push_back({data_root + "/val/images/" + filename, wnid_to_idx_[wnid]});
            }
        }
    }

    if (max_samples > 0 && static_cast<size_t>(max_samples) < samples_.size()) {
        samples_.resize(max_samples);
    }

    std::cout << "[DataLoader] Loaded " << samples_.size()
              << (is_train ? " training" : " validation") << " samples ("
              << wnids_.size() << " classes).\n";
    return !samples_.empty();
}

void TinyImageNetDataset::shuffle(unsigned int seed) {
    std::mt19937 g(seed);
    std::shuffle(samples_.begin(), samples_.end(), g);
}

int TinyImageNetDataset::get_batch(size_t start_idx, int batch_size,
                                   float* out_images, int* out_labels) const {
    if (start_idx >= samples_.size()) return 0;
    size_t end_idx = std::min(start_idx + batch_size, samples_.size());
    int actual = static_cast<int>(end_idx - start_idx);

    for (int b = 0; b < actual; ++b) {
        int w = 0, h = 0, comp = 0;
        unsigned char* data = stbi_load(samples_[start_idx + b].filepath.c_str(), &w, &h, &comp, CHANNELS);
        float* dst = out_images + b * IMAGE_SIZE;

        if (data && w == WIDTH && h == HEIGHT) {
            for (int c = 0; c < CHANNELS; ++c) {
                for (int y = 0; y < HEIGHT; ++y) {
                    for (int x = 0; x < WIDTH; ++x) {
                        dst[(c * HEIGHT + y) * WIDTH + x] = static_cast<float>(data[(y * WIDTH + x) * CHANNELS + c]) / 255.0f;
                    }
                }
            }
            stbi_image_free(data);
        } else {
            if (data) stbi_image_free(data);
            std::memset(dst, 0, IMAGE_SIZE * sizeof(float));
        }
        out_labels[b] = samples_[start_idx + b].label;
    }

    // Zero-pad remainder of buffer if last batch is smaller than batch_size
    if (actual < batch_size) {
        std::memset(out_images + actual * IMAGE_SIZE, 0, (batch_size - actual) * IMAGE_SIZE * sizeof(float));
        for (int b = actual; b < batch_size; ++b) out_labels[b] = 0;
    }
    return actual;
}

// Model Parameters Host Memory
void ModelParameters::allocate_host() {
    w_c1  = new float[ArchConfig::C1_W_SIZE];  b_c1  = new float[ArchConfig::C1_B_SIZE];
    dw_c1 = new float[ArchConfig::C1_W_SIZE]; db_c1 = new float[ArchConfig::C1_B_SIZE];
    vw_c1 = new float[ArchConfig::C1_W_SIZE](); vb_c1 = new float[ArchConfig::C1_B_SIZE]();

    w_c2  = new float[ArchConfig::C2_W_SIZE];  b_c2  = new float[ArchConfig::C2_B_SIZE];
    dw_c2 = new float[ArchConfig::C2_W_SIZE]; db_c2 = new float[ArchConfig::C2_B_SIZE];
    vw_c2 = new float[ArchConfig::C2_W_SIZE](); vb_c2 = new float[ArchConfig::C2_B_SIZE]();

    w_fc1  = new float[ArchConfig::FC1_W_SIZE];  b_fc1  = new float[ArchConfig::FC1_B_SIZE];
    dw_fc1 = new float[ArchConfig::FC1_W_SIZE]; db_fc1 = new float[ArchConfig::FC1_B_SIZE];
    vw_fc1 = new float[ArchConfig::FC1_W_SIZE](); vb_fc1 = new float[ArchConfig::FC1_B_SIZE]();

    w_fc2  = new float[ArchConfig::FC2_W_SIZE];  b_fc2  = new float[ArchConfig::FC2_B_SIZE];
    dw_fc2 = new float[ArchConfig::FC2_W_SIZE]; db_fc2 = new float[ArchConfig::FC2_B_SIZE];
    vw_fc2 = new float[ArchConfig::FC2_W_SIZE](); vb_fc2 = new float[ArchConfig::FC2_B_SIZE]();
}

void ModelParameters::free_host() {
    delete[] w_c1;  delete[] b_c1;  delete[] dw_c1;  delete[] db_c1;  delete[] vw_c1;  delete[] vb_c1;
    delete[] w_c2;  delete[] b_c2;  delete[] dw_c2;  delete[] db_c2;  delete[] vw_c2;  delete[] vb_c2;
    delete[] w_fc1; delete[] b_fc1; delete[] dw_fc1; delete[] db_fc1; delete[] vw_fc1; delete[] vb_fc1;
    delete[] w_fc2; delete[] b_fc2; delete[] dw_fc2; delete[] db_fc2; delete[] vw_fc2; delete[] vb_fc2;
}

void ModelParameters::init_weights(unsigned int seed) {
    auto kaiming = [](float* w, size_t sz, size_t fan_in, unsigned int s) {
        std::mt19937 gen(s);
        std::normal_distribution<float> d(0.0f, std::sqrt(2.0f / static_cast<float>(fan_in)));
        for (size_t i = 0; i < sz; ++i) w[i] = d(gen);
    };
    kaiming(w_c1, ArchConfig::C1_W_SIZE, 3 * 3 * 3, seed);
    std::fill(b_c1, b_c1 + ArchConfig::C1_B_SIZE, 0.0f);

    kaiming(w_c2, ArchConfig::C2_W_SIZE, 16 * 3 * 3, seed + 1);
    std::fill(b_c2, b_c2 + ArchConfig::C2_B_SIZE, 0.0f);

    kaiming(w_fc1, ArchConfig::FC1_W_SIZE, ArchConfig::FLATTEN_SIZE, seed + 2);
    std::fill(b_fc1, b_fc1 + ArchConfig::FC1_B_SIZE, 0.0f);

    kaiming(w_fc2, ArchConfig::FC2_W_SIZE, ArchConfig::FC1_OUT, seed + 3);
    std::fill(b_fc2, b_fc2 + ArchConfig::FC2_B_SIZE, 0.0f);
}

} // namespace cutin
