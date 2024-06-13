cmake_minimum_required(VERSION 3.10)

project(TFLiteDemo)

# Set the path to the TensorFlow Lite library and include directories
set(TFLITE_DIR "path/to/tflite")

include_directories(${TFLITE_DIR}/include)
link_directories(${TFLITE_DIR}/lib)

# Add your source files
add_executable(TFLiteDemo main.cpp)

# Link TensorFlow Lite library
target_link_libraries(TFLiteDemo tensorflowlite)
