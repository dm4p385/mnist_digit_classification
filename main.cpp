#include <iostream>
#include <fstream>
#include <vector>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/stderr_reporter.h"

// Function to read the model file into a buffer
std::vector<char> read_model(const std::string& model_path) {
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (file.read(buffer.data(), size)) {
        return buffer;
    }
    throw std::runtime_error("Failed to read model file");
}

int main() {
    const std::string model_path = "path/to/your/model.tflite";

    // Read the model
    std::vector<char> model_buffer = read_model(model_path);

    // Initialize TensorFlow Lite interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::StderrReporter error_reporter;
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromBuffer(model_buffer.data(), model_buffer.size(), &error_reporter);
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    std::unique_ptr<tflite::Interpreter> interpreter;
    if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
        std::cerr << "Failed to build interpreter" << std::endl;
        return 1;
    }

    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors" << std::endl;
        return 1;
    }

    // Fill input tensor with data
    float* input = interpreter->typed_input_tensor<float>(0);
    // Fill your input data here

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke interpreter" << std::endl;
        return 1;
    }

    // Read output tensor
    float* output = interpreter->typed_output_tensor<float>(0);
    // Process your output data here

    std::cout << "Inference completed successfully" << std::endl;

    return 0;
}
