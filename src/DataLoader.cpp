#include "DataLoader.h"
#include <iostream>
#include <fstream>
#include <cstdint>

// Function to read images
std::vector<std::vector<uint8_t>> readImages(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open Image file" << std::endl;
        return {};
    }

    // Read the IDX3-UBYTE file header
    char magicNumber[4];
    char numImagesBytes[4];
    char numRowBytes[4];
    char numColBytes[4];

    file.read(magicNumber, 4);
    file.read(numImagesBytes, 4);
    file.read(numRowBytes, 4);
    file.read(numColBytes, 4);

    // Convert the header information from big-endian to native endianness
    int numImages = (static_cast<uint8_t>(numImagesBytes[0]) << 24) |
                    (static_cast<uint8_t>(numImagesBytes[1]) << 16) |
                    (static_cast<uint8_t>(numImagesBytes[2]) << 8) |
                    static_cast<uint8_t>(numImagesBytes[3]);

    int numRows = (static_cast<uint8_t>(numRowBytes[0]) << 24) |
                  (static_cast<uint8_t>(numRowBytes[1]) << 16) |
                  (static_cast<uint8_t>(numRowBytes[2]) << 8) |
                  static_cast<uint8_t>(numRowBytes[3]);

    int numCols = (static_cast<uint8_t>(numColBytes[0]) << 24) |
                  (static_cast<uint8_t>(numColBytes[1]) << 16) |
                  (static_cast<uint8_t>(numColBytes[2]) << 8) |
                  static_cast<uint8_t>(numColBytes[3]);

    // Initialize a vector to store the images
    std::vector<std::vector<uint8_t>> images;

    for (int i = 0; i < numImages; i++) {
        // Read each image as a vector of uint8_t values
        std::vector<uint8_t> image(numRows * numCols);
        file.read(reinterpret_cast<char*>(image.data()), numRows * numCols);

        images.push_back(image);
    }

    file.close();

    return images;
}

// Function to read labels
std::vector<uint8_t> readLabels(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open the IDX3-UBYTE file." << std::endl;
        return {};
    }

    // Read the IDX3-UBYTE file header
    char magicNumber[4];
    char numImagesBytes[4];

    file.read(magicNumber, 4);
    file.read(numImagesBytes, 4);

    // Convert the header information from big-endian to native endianness
    int numImages = (static_cast<unsigned char>(numImagesBytes[0]) << 24) | (
                        static_cast<unsigned char>(numImagesBytes[1]) << 16) | (
                        static_cast<unsigned char>(numImagesBytes[2]) << 8) | static_cast<unsigned char>(numImagesBytes[
                        3]);

    // Initialize a vector to store the images
    std::vector<std::vector<unsigned char> > images;

    std::vector<uint8_t> labels(numImages);
    file.read(reinterpret_cast<char *>(labels.data()), numImages);
    file.close();

    return labels;
}

// DataLoader constructor
DataLoader::DataLoader(const std::string& images_path, const std::string& labels_path) {

    // Get raw images
    std::vector<std::vector<uint8_t>> raw_images = readImages(images_path);
    images.resize(raw_images.size());

    // Normalize and convert raw_images to float
    for (size_t i = 0; i < raw_images.size(); ++i) {
        images[i].resize(raw_images[i].size());
        for (size_t j = 0; j < raw_images[i].size(); ++j) {
            images[i][j] = static_cast<float>(raw_images[i][j]) / 255.0f; // Normalize and convert to float
        }
    }

    // Get labels
    std::vector<uint8_t> label_data = readLabels(labels_path);

}

// Get images
std::vector<std::vector<float> > DataLoader::getImages() {
    return images;
}

// Get Labels
std::vector<uint8_t> DataLoader::getLabels() {
    return labels;
}
