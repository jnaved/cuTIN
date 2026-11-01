#include "dataloader.hpp"
#include "network.hpp"
#include "utils.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <filesystem>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#include "stb_image.h"

namespace fs = std::filesystem;

namespace cutin {

bool TinyImageNetDataset::load_class_metadata(const std::string& data_root) {
    std::string wnids_path = data_root + "/wnids.txt";
    std::ifstream wnids_file(wnids_path);
    if (!wnids_file.is_open()) {
        std::cerr << "[DataLoader Error] Failed to open " << wnids_path << "\n";
        return false;
    }

    wnids_.clear();
    wnid_to_idx_.clear();
    std::string line;
    int idx = 0;
    while (std::getline(wnids_file, line)) {
        // Strip carriage returns and whitespace
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }
        if (!line.empty()) {
            wnids_.push_back(line);
            wnid_to_idx_[line] = idx++;
        }
    }
    wnids_file.close();

    std::string words_path = data_root + "/words.txt";
    std::ifstream words_file(words_path);
    if (words_file.is_open()) {
        while (std::getline(words_file, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            if (line.empty()) continue;
            std::istringstream iss(line);
            std::string wnid, name;
            if (std::getline(iss, wnid, '\t') && std::getline(iss, name)) {
                wnid_to_name_[wnid] = name;
            }
        }
        words_file.close();
    }

    return !wnids_.empty();
}

bool TinyImageNetDataset::load(const std::string& data_root, bool is_train, int max_samples) {
    if (!load_class_metadata(data_root)) {
        return false;
    }

    samples_.clear();

    if (is_train) {
        std::string train_dir = data_root + "/train";
        for (const auto& wnid : wnids_) {
            int label = wnid_to_idx_[wnid];
            std::string img_dir = train_dir + "/" + wnid + "/images";

            if (fs::exists(img_dir) && fs::is_directory(img_dir)) {
                for (const auto& entry : fs::directory_iterator(img_dir)) {
                    if (entry.is_regular_file()) {
                        std::string path_str = entry.path().string();
                        // Filter for JPEG files
                        if (path_str.find(".JPEG") != std::string::npos ||
                            path_str.find(".jpg") != std::string::npos ||
                            path_str.find(".jpeg") != std::string::npos) {
                            samples_.push_back({path_str, label});
                        }
                    }
                }
            }
        }
    } else {
        std::string val_annot_path = data_root + "/val/val_annotations.txt";
        std::ifstream val_file(val_annot_path);
        if (!val_file.is_open()) {
            std::cerr << "[DataLoader Error] Failed to open " << val_annot_path << "\n";
            return false;
        }

        std::string line;
        while (std::getline(val_file, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            if (line.empty()) continue;

            std::istringstream iss(line);
            std::string filename, wnid;
            if (iss >> filename >> wnid) {
                if (wnid_to_idx_.find(wnid) != wnid_to_idx_.end()) {
                    int label = wnid_to_idx_[wnid];
                    std::string img_path = data_root + "/val/images/" + filename;
                    samples_.push_back({img_path, label});
                }
            }
        }
        val_file.close();
    }

    if (max_samples > 0 && static_cast<size_t>(max_samples) < samples_.size()) {
        samples_.resize(max_samples);
    }

    std::cout << "[DataLoader] Loaded " << samples_.size()
              << (is_train ? " training" : " validation") << " samples across "
              << wnids_.size() << " classes.\n";

    return !samples_.empty();
}

void TinyImageNetDataset::shuffle(unsigned int seed) {
    std::mt19937 g(seed);
    std::shuffle(samples_.begin(), samples_.end(), g);
}

int TinyImageNetDataset::get_batch(size_t start_idx, int batch_size,
                                   float* out_images, int* out_labels) const {
    if (start_idx >= samples_.size()) return 0;

    int count = 0;
    size_t end_idx = std::min(start_idx + batch_size, samples_.size());
    int actual_batch = static_cast<int>(end_idx - start_idx);

    for (int b = 0; b < actual_batch; ++b) {
        const auto& sample = samples_[start_idx + b];
        int w = 0, h = 0, comp = 0;
        unsigned char* data = stbi_load(sample.filepath.c_str(), &w, &h, &comp, CHANNELS);

        float* img_batch_ptr = out_images + b * IMAGE_SIZE;

        if (data && w == WIDTH && h == HEIGHT) {
            // Convert HWC uint8 [0, 255] to CHW float [0, 1]
            for (int c = 0; c < CHANNELS; ++c) {
                for (int y = 0; y < HEIGHT; ++y) {
                    for (int x = 0; x < WIDTH; ++x) {
                        int hwc_idx = (y * WIDTH + x) * CHANNELS + c;
                        int chw_idx = (c * HEIGHT + y) * WIDTH + x;
                        img_batch_ptr[chw_idx] = static_cast<float>(data[hwc_idx]) / 255.0f;
                    }
                }
            }
            stbi_image_free(data);
        } else {
            if (data) stbi_image_free(data);
            // Fallback: fill zeros if image load failed or dimensions mismatch
            std::memset(img_batch_ptr, 0, IMAGE_SIZE * sizeof(float));
        }

        out_labels[b] = sample.label;
        count++;
    }

    // Zero-fill remainder of the buffer if actual_batch < batch_size
    if (actual_batch < batch_size) {
        size_t offset = actual_batch * IMAGE_SIZE;
        size_t remaining = (batch_size - actual_batch) * IMAGE_SIZE;
        std::memset(out_images + offset, 0, remaining * sizeof(float));
        for (int b = actual_batch; b < batch_size; ++b) {
            out_labels[b] = 0;
        }
    }

    return count;
}

std::string TinyImageNetDataset::get_wnid(int label_idx) const {
    if (label_idx >= 0 && label_idx < static_cast<int>(wnids_.size())) {
        return wnids_[label_idx];
    }
    return "unknown";
}

std::string TinyImageNetDataset::get_class_name(int label_idx) const {
    std::string wnid = get_wnid(label_idx);
    auto it = wnid_to_name_.find(wnid);
    if (it != wnid_to_name_.end()) {
        return it->second;
    }
    return wnid;
}

// ModelParameters host allocation implementations
void ModelParameters::allocate_host() {
    w_c1 = new float[ArchConfig::C1_W_SIZE];
    b_c1 = new float[ArchConfig::C1_B_SIZE];
    dw_c1 = new float[ArchConfig::C1_W_SIZE];
    db_c1 = new float[ArchConfig::C1_B_SIZE];
    vw_c1 = new float[ArchConfig::C1_W_SIZE]();
    vb_c1 = new float[ArchConfig::C1_B_SIZE]();

    w_c2 = new float[ArchConfig::C2_W_SIZE];
    b_c2 = new float[ArchConfig::C2_B_SIZE];
    dw_c2 = new float[ArchConfig::C2_W_SIZE];
    db_c2 = new float[ArchConfig::C2_B_SIZE];
    vw_c2 = new float[ArchConfig::C2_W_SIZE]();
    vb_c2 = new float[ArchConfig::C2_B_SIZE]();

    w_fc1 = new float[ArchConfig::FC1_W_SIZE];
    b_fc1 = new float[ArchConfig::FC1_B_SIZE];
    dw_fc1 = new float[ArchConfig::FC1_W_SIZE];
    db_fc1 = new float[ArchConfig::FC1_B_SIZE];
    vw_fc1 = new float[ArchConfig::FC1_W_SIZE]();
    vb_fc1 = new float[ArchConfig::FC1_B_SIZE]();

    w_fc2 = new float[ArchConfig::FC2_W_SIZE];
    b_fc2 = new float[ArchConfig::FC2_B_SIZE];
    dw_fc2 = new float[ArchConfig::FC2_W_SIZE];
    db_fc2 = new float[ArchConfig::FC2_B_SIZE];
    vw_fc2 = new float[ArchConfig::FC2_W_SIZE]();
    vb_fc2 = new float[ArchConfig::FC2_B_SIZE]();
}

void ModelParameters::free_host() {
    delete[] w_c1; delete[] b_c1; delete[] dw_c1; delete[] db_c1; delete[] vw_c1; delete[] vb_c1;
    delete[] w_c2; delete[] b_c2; delete[] dw_c2; delete[] db_c2; delete[] vw_c2; delete[] vb_c2;
    delete[] w_fc1; delete[] b_fc1; delete[] dw_fc1; delete[] db_fc1; delete[] vw_fc1; delete[] vb_fc1;
    delete[] w_fc2; delete[] b_fc2; delete[] dw_fc2; delete[] db_fc2; delete[] vw_fc2; delete[] vb_fc2;
}

void ModelParameters::init_weights(unsigned int seed) {
    // Conv1: fan_in = IN_C * K * K = 3 * 3 * 3 = 27
    kaiming_normal_init(w_c1, ArchConfig::C1_W_SIZE, 27, seed);
    zero_init(b_c1, ArchConfig::C1_B_SIZE);

    // Conv2: fan_in = C1_OUT_C * K * K = 16 * 3 * 3 = 144
    kaiming_normal_init(w_c2, ArchConfig::C2_W_SIZE, 144, seed + 1);
    zero_init(b_c2, ArchConfig::C2_B_SIZE);

    // FC1: fan_in = 8192
    kaiming_normal_init(w_fc1, ArchConfig::FC1_W_SIZE, ArchConfig::FLATTEN_SIZE, seed + 2);
    zero_init(b_fc1, ArchConfig::FC1_B_SIZE);

    // FC2: fan_in = 512
    kaiming_normal_init(w_fc2, ArchConfig::FC2_W_SIZE, ArchConfig::FC1_OUT, seed + 3);
    zero_init(b_fc2, ArchConfig::FC2_B_SIZE);
}

} // namespace cutin

