cmake_minimum_required(VERSION 3.10)
project(main)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the path to TensorFlow Lite headers and libraries
set(DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow/include/")
set(LIBRARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow/lib/")
#set(FLATBUFFERS_INCLUDE_DIR "third_party/flatbuffers/include/")

# Include TensorFlow Lite headers
include_directories(${INCLUDE_DIR})
#include_directories(${FLATBUFFERS_INCLUDE_DIR})

set(SOURCES
        src/main.cpp
        src/DataLoader.cpp
        src/ModelLoader.cpp
)

# Find TensorFlow Lite library (assuming tensorflowlite.dll is in TFLITE_LIBRARY_DIR)
find_library(TFLITE tensorflowlite.dll.if HINTS ${LIBRARY_DIR})
find_library(TFLITE_FLEX tensorflowlite_flex.dll.if HINTS ${LIBRARY_DIR})
find_library(FLATBUFFERS flatbuffers HINTS ${LIBRARY_DIR})

add_executable(main ${SOURCES})
add_compile_definitions(DATA_DIR="${DATA_DIR}")
# Link against the TensorFlow Lite library

target_link_libraries(main PRIVATE ${TFLITE_FLEX} ${TFLITE} ${FLATBUFFERS})

add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LIBRARY_DIR}tensorflowlite.dll" "${CMAKE_BINARY_DIR}/Debug/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LIBRARY_DIR}tensorflowlite_flex.dll" "${CMAKE_BINARY_DIR}/Debug/"
#        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LIBRARY_DIR}flatbuffers.dll" "${CMAKE_BINARY_DIR}/Debug/"
)