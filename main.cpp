#include <iostream>
#include <fstream>
#include <stdlib>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/stderr_reporter.h"

int main() {
    const string model_path = "./model.tflite";

    // Load the model
    std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(model_path);
    if (!model) {
        cerr << "Failed to load model" << std::endl;
        return 1;
    }

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
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
