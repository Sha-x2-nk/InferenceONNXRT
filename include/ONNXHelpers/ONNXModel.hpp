#include <onnxruntime/onnxruntime_cxx_api.h>

#include <string>
#include <vector>

/*
    Our API will only target models with 1 input node and 1 output node.
*/
class ONNXModel {
    private:
    Ort::Env* ort_env = nullptr;
    Ort::Session* ort_sess = nullptr;
    Ort::MemoryInfo* ort_mem_info = nullptr;

    std::vector<const char*>* input_node_names = nullptr;
    std::vector<std::vector<int64_t>> input_node_dims;

    std::vector<const char*> inp_node_names, out_node_names;

    std::vector<int64_t> inp_shape, out_shape;


    public:
    
    /*
        model_path  wchar_t     path of the ONNX model.
        EP:         string      Execution Provider          [Default, CUDA, OpenVINO, TensorRT]
    */
    ONNXModel(const wchar_t* model_path, 
            const std::string &EP, 
            const char* inp_node_name, std::vector<int64_t> &inp_shape,     /* inp_shape = [C, H, W]. Batch size given in forward pass. N, if given, will be ignored. */ 
            const char* out_node_name, std::vector<int64_t> &out_shape);    /* same as inp_shape */
    
    /*
        Forward pass
    */
    std::vector<float> forward (float* data, int batch_size = 1);

    /*
        Destructor
    */
    ~ONNXModel ();
};