# ONNX Model Inference API
This project provides a simple C++ API for running inference on ONNX models using ONNXRuntime. The API supports models with a single input node and a single output node and allows for execution using various providers like CPU, CUDA, OpenVINO, and TensorRT.

## Features
* Load and run inference on ONNX models.
* Supports different execution providers: CPU (default), CUDA, OpenVINO, and TensorRT.
* Easy integration with models having one input and one output node.

## Requirements 
* <b>Install/build ONNXRuntime</b>: Build ONNXRuntime from source and configure it according to the EPs you need, or install an existing build.

## Usage
C++ inference example.
```cpp
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
```

## API Reference
### `ONNXModel`

#### Constructor
```cpp
ONNXModel(
    const wchar_t* model_path,
    const std::string &EP,
    const char* inp_node_name,
    std::vector<int64_t> &inp_shape,
    const char* out_node_name,
    std::vector<int64_t> &out_shape
);
```
* <b>model_path</b>: Path to the ONNX model.
* <b>EP</b>: Execution Provider ("Default", "CUDA", "OpenVINO", "TensorRT").
* <b>inp_node_name</b>: Name of the input node.
* <b>inp_shape</b>: Shape of the input tensor (excluding batch size).
* <b>out_node_name</b>: Name of the output node.
* <b>out_shape</b>: Shape of the output tensor (excluding batch size).

#### `forward`
```cpp
std::vector<float> forward(float* data, int batch_size = 1);
```
* <b>data</b>: Pointer to the input data.
* <b>batch_size</b>: Batch size for the input data (default is 1).

#### Destructor
```cpp
~ONNXModel();
```

## Contributing
Contributions are welcome! Upcoming updates will be adding automatic deduction of input names and input shapes.

## Acknowledgements
This project uses the <a href = "https://github.com/microsoft/onnxruntime">ONNXRuntime</a> library for running ONNX model inference.
