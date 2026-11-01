#ifndef CUTIN_DATALOADER_HPP
#define CUTIN_DATALOADER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <cstddef>

namespace cutin {

struct ImageSample {
    std::string filepath;
    int label; // 0 to 199
};

class TinyImageNetDataset {
public:
    TinyImageNetDataset() = default;

    // Load dataset from directory (path to tiny-imagenet-200 folder)
    // is_train: true for train set, false for val set
    // max_samples: if > 0, limit loaded samples (helpful for quick debugging/testing)
    bool load(const std::string& data_root, bool is_train = true, int max_samples = -1);

    // Get number of loaded samples
    size_t size() const { return samples_.size(); }

    // Shuffle the dataset
    void shuffle(unsigned int seed = 42);

    // Fetch a mini-batch of images and labels
    // out_images: pre-allocated float buffer of size (batch_size * 3 * 64 * 64) in NCHW format, normalized to [0, 1]
    // out_labels: pre-allocated int buffer of size (batch_size)
    // Returns actual number of samples loaded into the batch (handles last partial batch)
    int get_batch(size_t start_idx, int batch_size, float* out_images, int* out_labels) const;

    // Class name lookup
    std::string get_wnid(int label_idx) const;
    std::string get_class_name(int label_idx) const;

    static constexpr int CHANNELS = 3;
    static constexpr int HEIGHT = 64;
    static constexpr int WIDTH = 64;
    static constexpr int NUM_CLASSES = 200;
    static constexpr int IMAGE_SIZE = CHANNELS * HEIGHT * WIDTH; // 12288 floats

private:
    std::vector<ImageSample> samples_;
    std::vector<std::string> wnids_; // 200 class ids
    std::unordered_map<std::string, int> wnid_to_idx_;
    std::unordered_map<std::string, std::string> wnid_to_name_;

    bool load_class_metadata(const std::string& data_root);
};

} // namespace cutin

#endif // CUTIN_DATALOADER_HPP

