#include "ONNXModel.hpp"

#include <iostream>
#include <chrono>
int main() {
    /* 
        The below constants are for yolov8 face detector model.
        Use "netron" for seeing various properties of ONNX Model.
    */
    const wchar_t* detector_path = L"C:/Users/shash/Documents/FaceRecog/models/yolov8n-face-lindevs.onnx";

    // Execution Provider: "CPU", "CUDA", "OpenVINO", "TensorRT"
    std::string execution_provider = "CPU";

    // Input and output node names and shapes
    const char* input_node_name = "images";
    std::vector<int64_t> input_shape = {3, 640, 640}; // [C, H, W]

    const char* output_node_name = "output0";
    std::vector<int64_t> output_shape = {5, 8400}; 


    ONNXModel model(detector_path, execution_provider, input_node_name, input_shape, output_node_name, output_shape);

    /* 
        Demo data 
        [N C H W] = [20 3 640 640]
    */
    float* inp = new float[20 * 3 * 640 * 640];
    
    // Capture the start time
    auto start_time = std::chrono::high_resolution_clock::now();

    /* out.size : 20 x 5 x 8400 */
    std::vector<float> out = model.forward(inp, /* batch_size = */ 20); 
    
    // Capture the end time
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Print the duration
    std::cout << "\nForward pass runtime: " << duration.count() << " milliseconds." << std::endl;


    std::cout<<"\n out.size(): "<<out.size();
    return 0;
}