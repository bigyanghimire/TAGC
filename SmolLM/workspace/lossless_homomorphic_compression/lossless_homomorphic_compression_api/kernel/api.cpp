#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_device_runtime_api.h>
#include "../include/api.h"

void torch_launch_create_index_4_bit(torch::Tensor &data, torch::Tensor &index, int grid_size, int block_size, torch::Tensor & counter, long stream){
    launch_create_index_4_bit((float*)data.data_ptr(), (char*)index.data_ptr(), grid_size, block_size, (unsigned long long *)counter.data_ptr(), stream);
}

void torch_launch_create_index_4_bit_bf16(torch::Tensor &data, torch::Tensor &index, int grid_size, int block_size, torch::Tensor & counter, long stream){
    launch_create_index_4_bit_bf16((__nv_bfloat16*)data.data_ptr(), (char*)index.data_ptr(), grid_size, block_size, (unsigned long long *)counter.data_ptr(), stream);
}

void torch_launch_read_index_4_bit(torch::Tensor &data, torch::Tensor &index, int grid_size, int block_size, long stream){
    launch_read_index_4_bit((int*)data.data_ptr(), (char*)index.data_ptr(), grid_size, block_size, stream);
}

void torch_launch_read_index_4_bit_bf16(torch::Tensor &data, torch::Tensor &index, int grid_size, int block_size, long stream){
    launch_read_index_4_bit_bf16((short*)data.data_ptr(), (char*)index.data_ptr(), grid_size, block_size, stream);
}

void torch_launch_create_index_1_bit(torch::Tensor &data, torch::Tensor &index, int grid_size, int block_size, torch::Tensor & counter, long stream){
    launch_create_index_1_bit((float*)data.data_ptr(), (char*)index.data_ptr(), grid_size, block_size, (unsigned long long *)counter.data_ptr(), stream);
}

void torch_launch_create_index_1_bit_bf16(torch::Tensor &data, torch::Tensor &index, int grid_size, int block_size, torch::Tensor & counter, long stream){
    launch_create_index_1_bit_bf16((__nv_bfloat16*)data.data_ptr(), (char*)index.data_ptr(), grid_size, block_size, (unsigned long long *)counter.data_ptr(), stream);
}

void torch_launch_read_index_1_bit(torch::Tensor &data, torch::Tensor &index, int grid_size, int block_size, long stream){
    launch_read_index_1_bit((int*)data.data_ptr(), (char*)index.data_ptr(), grid_size, block_size, stream);
}

void torch_launch_read_index_1_bit_bf16(torch::Tensor &data, torch::Tensor &index, int grid_size, int block_size, long stream){
    launch_read_index_1_bit_bf16((short*)data.data_ptr(), (char*)index.data_ptr(), grid_size, block_size, stream);
}

void torch_launch_compress_float_32(torch::Tensor &data, torch::Tensor &count_sketch, torch::Tensor &count_mapping, int compressed_r, int grid_size, int block_size, long stream){
    launch_compress_float_32((float*)data.data_ptr(), (float*)count_sketch.data_ptr(), (int*)count_mapping.data_ptr(), compressed_r, grid_size, block_size, stream);
}

void torch_launch_compress_bfloat_16(torch::Tensor &data, torch::Tensor &count_sketch, torch::Tensor &count_mapping, int compressed_r, int grid_size, int block_size, long stream){
    launch_compress_bfloat_16((__nv_bfloat16*)data.data_ptr(), (__nv_bfloat16*)count_sketch.data_ptr(), (int*)count_mapping.data_ptr(), compressed_r, grid_size, block_size, stream);
}

void torch_launch_decompress_float_32(torch::Tensor &data, torch::Tensor &count_sketch, torch::Tensor &count_mapping, int compressed_r, int grid_size, int block_size, torch::Tensor & flag, long stream){
    launch_decompress_float_32((float*)data.data_ptr(), (float*)count_sketch.data_ptr(), (int*)count_mapping.data_ptr(), compressed_r, grid_size, block_size, (int*)flag.data_ptr(), stream);
}

void torch_launch_decompress_bfloat_16(torch::Tensor &data, torch::Tensor &count_sketch, torch::Tensor &count_mapping, int compressed_r, int grid_size, int block_size, torch::Tensor & flag, long stream){
    launch_decompress_bfloat_16((__nv_bfloat16*)data.data_ptr(), (__nv_bfloat16*)count_sketch.data_ptr(), (int*)count_mapping.data_ptr(), compressed_r, grid_size, block_size, (int*)flag.data_ptr(), stream);
}

void torch_launch_estimate_float_32(torch::Tensor &data, torch::Tensor &count_sketch, int compressed_r, int grid_size, int block_size, torch::Tensor & counter, long stream){
    launch_estimate_float_32((float*)data.data_ptr(), (float*)count_sketch.data_ptr(), compressed_r, grid_size, block_size, (unsigned long long*)counter.data_ptr(), stream);
}

void torch_launch_estimate_bfloat_16(torch::Tensor &data, torch::Tensor &count_sketch, int compressed_r, int grid_size, int block_size, torch::Tensor & counter, long stream){
    launch_estimate_bfloat_16((__nv_bfloat16*)data.data_ptr(), (__nv_bfloat16*)count_sketch.data_ptr(), compressed_r, grid_size, block_size, (unsigned long long*)counter.data_ptr(), stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_create_index_4_bit", &torch_launch_create_index_4_bit, "launch_create_index_4_bit");
    m.def("torch_launch_create_index_4_bit_bf16", &torch_launch_create_index_4_bit_bf16, "launch_create_index_4_bit_bf16");
    m.def("torch_launch_read_index_4_bit", &torch_launch_read_index_4_bit, "launch_read_index_4_bit");
    m.def("torch_launch_read_index_4_bit_bf16", &torch_launch_read_index_4_bit_bf16, "launch_read_index_4_bit_bf16");
    m.def("torch_launch_create_index_1_bit", &torch_launch_create_index_1_bit, "launch_create_index_1_bit");
    m.def("torch_launch_create_index_1_bit_bf16", &torch_launch_create_index_1_bit_bf16, "launch_create_index_1_bit_bf16");
    m.def("torch_launch_read_index_1_bit", &torch_launch_read_index_1_bit, "launch_read_index_1_bit");
    m.def("torch_launch_read_index_1_bit_bf16", &torch_launch_read_index_1_bit_bf16, "launch_read_index_1_bit_bf16");
    m.def("torch_launch_compress_float_32", &torch_launch_compress_float_32, "launch_compress_float_32");
    m.def("torch_launch_compress_bfloat_16", &torch_launch_compress_bfloat_16, "launch_compress_bfloat_16");
    m.def("torch_launch_decompress_float_32", &torch_launch_decompress_float_32, "launch_decompress_float_32");
    m.def("torch_launch_decompress_bfloat_16", &torch_launch_decompress_bfloat_16, "launch_decompress_bfloat_16");
    m.def("torch_launch_estimate_float_32", &torch_launch_estimate_float_32, "launch_estimate_float_32");
    m.def("torch_launch_estimate_bfloat_16", &torch_launch_estimate_bfloat_16, "launch_estimate_bfloat_16");
}

