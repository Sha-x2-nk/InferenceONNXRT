#include "ONNXHelpers/ONNXModel.hpp"

#include <iostream>
#include <chrono>
int main() {
    const wchar_t* detector_path = L"C:/Users/shash/Documents/FaceRecog/models/yolov8n-face-lindevs.onnx";
    std::vector<int64_t> inp_shape = {3, 640, 640};
    std::vector<int64_t> out_shape = {5, 8400};

    ONNXModel model(detector_path, "CPU", "images", inp_shape, "output0", out_shape);


    float* inp = new float[3 * 640 * 640];
    
    // Capture the start time
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<float> out = model.forward(inp);

    // Capture the end time
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = end - start;

    // Print the duration
    std::cout << "\nFunction runtime: " << duration.count() << " seconds" << std::endl;


    std::cout<<"\n out.size(): "<<out.size();
    return 0;

}