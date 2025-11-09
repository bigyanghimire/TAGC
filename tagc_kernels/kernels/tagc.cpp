#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_device_runtime_api.h>
#include "../include/tagc.h"

void torch_launch_sparsify_init(torch::Tensor &grad, torch::Tensor &remainder, torch::Tensor &out_abs, int grid_size, int block_size, long stream){
    launch_sparsify_init((__nv_bfloat16*)grad.data_ptr(), (__nv_bfloat16*)remainder.data_ptr(), (__nv_bfloat16*)out_abs.data_ptr(), grid_size, block_size, stream);
}

void torch_launch_sparsify_set(torch::Tensor &data, torch::Tensor &in_abs_out_result, int grid_size, int block_size, float sparsify_step, long stream){
    __nv_bfloat16 sparsify_step_bf16 = __float2bfloat16(sparsify_step);
    launch_sparsify_set((__nv_bfloat16*)data.data_ptr(), (__nv_bfloat16*)in_abs_out_result.data_ptr(), grid_size, block_size, sparsify_step_bf16, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_sparsify_init", &torch_launch_sparsify_init, "launch_sparsify_init");
    m.def("torch_launch_sparsify_set", &torch_launch_sparsify_set, "launch_sparsify_set");
}

