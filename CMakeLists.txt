cmake_minimum_required(VERSION 3.22)

project(InferenceONNXRT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to the ONNX RT install DIR")

set(ONNXRUNTIME_LIBS
                    onnxruntime.lib
                    onnxruntime_providers_shared.lib
)

include_directories(include)

link_directories(${ONNXRUNTIME_ROOT}/lib)

set(SOURCES 
            src/ONNXModel.cpp
)

add_executable(main main.cpp ${SOURCES})

target_link_libraries(main ${OpenCV_LIBS} ${ONNXRUNTIME_LIBS})
