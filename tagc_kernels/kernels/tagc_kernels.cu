#include <cuda_bf16.h>

__global__ void sparsify_init(__nv_bfloat16* grad, __nv_bfloat16* remainder, __nv_bfloat16* out_abs) {
    // Accept abs(grad) as a separate argument?
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    __nv_bfloat16 value = grad[index] + remainder[index];
    grad[index] = value;
    if (value < CUDART_ZERO_BF16) {
        out_abs[index] = -value;
    } else {
        out_abs[index] = value;
    }
}

__global__ void sparsify_set(__nv_bfloat16* data, __nv_bfloat16* in_abs_out_result, __nv_bfloat16 sparsify_step) {
    // Accept abs(grad) as a separate argument?
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (in_abs_out_result[index] < sparsify_step) {
        in_abs_out_result[index] = CUDART_ZERO_BF16;
    } else {
        in_abs_out_result[index] =  data[index];
    }
}

/*
    The following functions are the launch functions of the above kernels. Users can call these functions to launch the
    corresponding kernels.

    The parameters of the launch functions are the same as the parameters of the corresponding kernels. Users can refer to
    the comments of the corresponding kernels for the detailed explanation of the parameters.
*/

void launch_sparsify_init(__nv_bfloat16* grad, __nv_bfloat16* remainder, __nv_bfloat16* out_abs, int grid_size, int block_size, long stream) {
    sparsify_init<<<grid_size, block_size, 0, (cudaStream_t)stream>>>(grad, remainder, out_abs);
}

void launch_sparsify_set(__nv_bfloat16* data, __nv_bfloat16* in_abs_out_result, int grid_size, int block_size, __nv_bfloat16 sparsify_step, long stream) {
    sparsify_set<<<grid_size, block_size, 0, (cudaStream_t)stream>>>(data, in_abs_out_result, sparsify_step);
}

