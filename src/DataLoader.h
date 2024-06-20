#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <cstdint>
#include <string>
#include <vector>

class DataLoader {
public:
    DataLoader(const std::string& images_path, const std::string& labels_path);
    std::vector<std::vector<float>> getImages();
    std::vector<uint8_t> getLabels();
private:
    std::vector<std::vector<float>> images;
    std::vector<uint8_t> labels;
};

#endif // DATA_LOADER_H
