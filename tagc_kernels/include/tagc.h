void launch_sparsify_init(__nv_bfloat16* grad, __nv_bfloat16* remainder, __nv_bfloat16* out_abs, int grid_size, int block_size, long stream);
void launch_sparsify_set(__nv_bfloat16* data, __nv_bfloat16* in_abs_out_result, int grid_size, int block_size, __nv_bfloat16 sparsify_step, long stream);
