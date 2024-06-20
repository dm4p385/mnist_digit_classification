#include <iostream>
#include "ModelLoader.h"
#include "DataLoader.h"

// Use the defined model path
#ifndef DATA_DIR
#define DATA_DIR "../data"
#endif

int main() {
    std::string model_path = std::string(DATA_DIR) + "/model.tflite";
    std::string images_path = std::string(DATA_DIR) + "/t10k-images.idx3-ubyte";
    std::string labels_path = std::string(DATA_DIR) + "/t10k-labels.idx1-ubyte";

    std::cout << "hi" <<std::endl;

    // Create a DataLoader instance to load images and labels
    DataLoader data_loader(images_path, labels_path);

    // Create a ModelLoader instance to load your TensorFlow Lite model
    ModelLoader model_loader(model_path);

    // Example: Get the first image from DataLoader and predict using ModelLoader
    std::vector<std::vector<float>> images = data_loader.getImages();
    std::vector<uint8_t> labels = data_loader.getLabels();

    if (!images.empty()) {
        std::vector<float> input_image = images[0]; // Assuming the first image
        int predicted_label = model_loader.predict(input_image);

        std::cout << "Predicted Label: " << static_cast<int>(predicted_label) << std::endl;
        std::cout << "True Label: " << static_cast<int>(labels[0]) << std::endl;
    }
    return 0;
}
