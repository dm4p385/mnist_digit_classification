#include "ModelLoader.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/delegates/flex/delegate.h"

ModelLoader::ModelLoader(const std::string& model_path) {
    loadModel(model_path);

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        throw std::runtime_error("Failed to build interpreter");
    }

    interpreter->AllocateTensors();

    // Assuming the input tensor is flattened and has size 784 (28 * 28)
    input_size = interpreter->input_tensor(0)->bytes / sizeof(float);

    // Assuming output tensor shape is [1, num_classes]
    num_classes = interpreter->output_tensor(0)->dims->data[1];
}

void ModelLoader::loadModel(const std::string& model_path) {
    model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) {
        throw std::runtime_error("Failed to load model");
    }
}

int ModelLoader::predict(const std::vector<float>& input) {
    if (input.size() != input_size) {
        throw std::runtime_error("Input size does not match expected input shape");
    }

    // Get pointer to input tensor
    float* input_tensor = interpreter->typed_input_tensor<float>(0);
    // Copy input data into TensorFlow Lite input tensor
    std::memcpy(input_tensor, input.data(), input_size * sizeof(float));

    if (interpreter->Invoke() != kTfLiteOk) {
        throw std::runtime_error("Failed to invoke interpreter");
    }

    float* output_tensor = interpreter->typed_output_tensor<float>(0);

    // Assuming output_tensor contains multiple elements representing predictions
    // For classification, find the index of the maximum element
    int max_index = 0;
    float max_value = output_tensor[0];
    for (int i = 1; i < num_classes; ++i) {
        if (output_tensor[i] > max_value) {
            max_value = output_tensor[i];
            max_index = i;
        }
    }
    return max_index;
}
