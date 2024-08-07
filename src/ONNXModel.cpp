#include "ONNXModel.hpp"

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <iostream>
#include <string>
#include <vector>
#include <numeric>


ONNXModel::ONNXModel(const wchar_t* model_path, 
                    const std::string &EP, 
                    const char* inp_node_name, std::vector<int64_t> &inp_shape, 
                    const char* out_node_name, std::vector<int64_t> &out_shape) {
    // Step 1. Create env
    this->ort_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Default");

    // Step 2. Set session options
    Ort::SessionOptions sess_opt;

    // We will now configure different EPs
    // OpenVINO EP
    if(EP == "OpenVINO") {
        sess_opt.SetIntraOpNumThreads(1);
        OrtOpenVINOProviderOptions openvino_opt;
        openvino_opt.device_type = "CPU";

        sess_opt.AppendExecutionProvider_OpenVINO(openvino_opt);
        std::cout<<"\n[+] Configured OpenVINO EP.";
    }

    // CUDA EP
    else if (EP == "CUDA") {
        OrtSessionOptionsAppendExecutionProvider_CUDA(sess_opt, /* device_id = */ 0);
        std::cout<<"\n[+] Configured CUDA EP.";
    }

    // TensorRT EP
    else if (EP == "TensorRT") {
        OrtSessionOptionsAppendExecutionProvider_Tensorrt(sess_opt, /* device_id = */ 0);
        std::cout<<"\n[+] Configured TensorRT EP.";
    }

    // Default Backend
    else {
        std::cout<<"\n[+] No EP provided in YOLO constructor... Defaulting to CPU session.";
    }

    sess_opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    std::cout<<"\n[+] GraphOptimizationLevel = ORT_ENABLE_ALL";

    // Step 3. Create session
    try {
        this->ort_sess = new Ort::Session(*(this->ort_env), model_path, sess_opt);
    } 
    catch (Ort::Exception oe) {
        std::cout<<"[-] ONNX RT exception caught: "<<oe.what() << ". Code: "<<oe.GetOrtErrorCode()<<".\n";
        exit(-1);
    }

    // Step 4. Input preparation
    try {
        this->ort_mem_info = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault));
    }
    catch (Ort::Exception oe) {
        std::cout<<"[-] ONNX RT exception caught: "<<oe.what() << ". Code: "<<oe.GetOrtErrorCode()<<".\n";
        exit(-1);
    }

    this->inp_node_names.push_back(inp_node_name);
    this->out_node_names.push_back(out_node_name);

    if(inp_shape.size() == 3) {         // CHW 
        this->inp_shape.push_back(1);   // converting to NCHW
        this->out_shape.push_back(1);   
    }

    for(int i = 0; i< inp_shape.size(); ++i)
        this->inp_shape.push_back(inp_shape[i]);
    
    for(int i = 0; i< out_shape.size(); ++i)
        this->out_shape.push_back(out_shape[i]);
}

std::vector<float> ONNXModel::forward(float* input, int batch_size) {
    this->inp_shape[0] = batch_size;
    this->inp_shape[0] = batch_size;

    auto model_inp_tensor = Ort::Value::CreateTensor<float>(*this->ort_mem_info, reinterpret_cast<float*>(input), std::accumulate(this->inp_shape.begin(), this->inp_shape.end(), 1, std::multiplies<int64_t>()) /* NxCxHxW */, inp_shape.data(), inp_shape.size());

    auto model_out_tensor = this->ort_sess->Run(Ort::RunOptions{nullptr}, this->inp_node_names.data(), &model_inp_tensor, 1, this->out_node_names.data(), 1);

    /* We assume only 1 output */
    float* out_float_arr = model_out_tensor[0].GetTensorMutableData<float>();
    size_t total_elements = std::accumulate(this->out_shape.begin(), this->out_shape.end(), 1, std::multiplies<int64_t>());

    // Copy the data to a std::vector
    std::vector<float> out_float_vec(out_float_arr, out_float_arr + total_elements);

    return out_float_vec;
}


ONNXModel::~ONNXModel() {
    delete this->ort_env;
    delete this->ort_sess;
    delete this->ort_mem_info;
}