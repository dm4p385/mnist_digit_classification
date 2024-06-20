#ifndef MODELLOADER_H
#define MODELLOADER_H

#include <string>
#include <vector>
#include "tensorflow/lite/model.h"

class ModelLoader {
public:
    ModelLoader(const std::string& model_path);
    int predict(const std::vector<float>& input);

private:
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    int input_size;
    int num_classes;

    void loadModel(const std::string& model_path);
};

#endif // MODELLOADER_H
