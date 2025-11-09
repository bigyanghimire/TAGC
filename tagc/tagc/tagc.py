import bisect
from   enum import IntEnum
import functools
import math
import numpy as np
from   typing import Callable, List, Optional

import torch
import torch.distributed as dist
from   torch.distributed.algorithms._comm_hooks import default_hooks

import api
import tagc_kernels

# The block size of the data. We set the block size to 1024, which is the max number of thread
# in one GPU block. TODO does it depend on GPU?
BLOCK_SIZE = 256


# TODO consider decreasing number of sparsification steps or getting rid of table by limiting
# number of bisection times
SPARSIFY_STEPS = [1e-20, 1e-15, 1e-10, 1e-8, 1e-7, 2e-7, 5e-7,
                  1e-6, 1.09e-6, 1.19e-6, 1.30e-6, 1.41e-6, 1.54e-6, 1.68e-6, 1.83e-6, 2e-6, 2.18e-6, 2.38e-6, 2.59e-6, 3.16e-6, 3.5e-6, 3.98e-6, 4.4e-6, 5e-6, 5.45e-6, 5.95e-6, 6.49e-6, 7.07e-6, 7.71e-6, 8.41e-6,
                  1e-5, 1.09e-5, 1.19e-5, 1.30e-5, 1.41e-5, 1.54e-5, 1.68e-5, 1.83e-5, 2e-5, 2.18e-5, 2.38e-5, 2.59e-5, 3.16e-5, 3.5e-5, 3.98e-5, 4.4e-5, 5e-5, 5.45e-5, 5.95e-5, 6.49e-5, 7.07e-5, 7.71e-5, 8.41e-5,
                  1e-4, 1.09e-4, 1.19e-4, 1.30e-4, 1.41e-4, 1.54e-4, 1.68e-4, 1.83e-4, 2e-4, 2.18e-4, 2.38e-4, 2.59e-4, 3.16e-4, 3.5e-4, 3.98e-4, 4.4e-4, 5e-4, 5.45e-4, 5.95e-4, 6.49e-4, 7.07e-4, 7.71e-4, 8.41e-4,
                  1e-3, 1.09e-3, 1.19e-3, 1.30e-3, 1.41e-3, 1.54e-3, 1.68e-3, 1.83e-3, 2e-3, 2.18e-3, 2.38e-3, 2.59e-3, 3.16e-3, 3.5e-3, 3.98e-3, 4.4e-3, 5e-3, 5.45e-3, 5.95e-3, 6.49e-3, 7.07e-3, 7.71e-3, 8.41e-3,
                  1e-2, 1.09e-2, 1.19e-2, 1.30e-2, 1.41e-2, 1.54e-2, 1.68e-2, 1.83e-2, 2e-2, 2.18e-2, 2.38e-2, 2.59e-2, 3.16e-2, 3.5e-2, 3.98e-2, 4.4e-2, 5e-2, 5.45e-2, 5.95e-2, 6.49e-2, 7.07e-2, 7.71e-2, 8.41e-2,
                  1e-1, 1.41e-1, 2e-1, 5e-1]


class IndexSize(IntEnum):
    ONE_BIT = 1
    FOUR_BITS = 4


def fp32_compress_hook(state: default_hooks.LowPrecisionState, grad: torch.Tensor, output: Optional[torch.Tensor] = None):
    with torch.profiler.record_function(
        "tagc.fp32_compress_hook"
    ):
        fp32_hook = functools.partial(default_hooks._low_precision_hook, torch.float32)
        return fp32_hook(state, grad, output)


class HomomorphicCompressState(default_hooks.LowPrecisionState):

    __slots__ = [
        "num_processes",
        "process_index",
        "cuda_stream_nccl",
        "cuda_stream_comp",
        "cuda_streams_sparse",
        "grad_num",
        "grad_remainders",
        "iter_num",
        "sparsify_fraction",
        "estimate_counter",
        "total_counter",
        "nonzero_counter",
        # Everything below is reassigned for each gradient
        "aligned_sparsified_grad_shards",
        "aligned_output",
        "sparse_readys",
        "indexes",
        "local_index_ready",
        "count_sketches",
        "count_mappings",
        "global_index_ready",
        "local_count_sketch_ready",
        "compress_ratio",
        "index_size",
        "flag_zero",
        "decompress_unfinished",
        "estimate_finished",
        "grid_size",
        "compressed_r",
    ]

    def __init__(
        self,
        process_group: int,
        num_processes: int,
        process_index: int,
        sparsify_fraction: float,
        index_size: IndexSize,
        compress_ratio: float,
        device: str,
        parameter_type: torch.dtype
    ):
        super().__init__(process_group, parameter_type=parameter_type)
        self.num_processes = num_processes
        self.process_index = process_index
        self.sparsify_fraction = sparsify_fraction
        self.index_size = index_size
        self.compress_ratio = compress_ratio

        self.cuda_stream_nccl = torch.cuda.Stream(priority=torch.cuda.default_stream().priority - 5)
        self.cuda_stream_comp = torch.cuda.Stream()
        self.cuda_streams_sparse = [torch.cuda.Stream() for _ in range(self.num_processes)]
        self.grad_num = 0
        # FIXME Are accumulated locally for all gradients, not just for shards, this
        # is a memory issue contradicting sharded strategy
        # Plan: 1) hope that variance precomputed during heat-up will be anough to compensate
        # error as we won't be able to store momentum for all gradients
        # 2) move to pipeline parallelism
        # 3) move to data parallelism
        self.grad_remainders = {}

        self.iter_num = 0

        self.estimate_counter = torch.zeros(1, dtype=torch.int64, device=device)
        self.total_counter = 0
        self.nonzero_counter = torch.zeros(1, dtype=torch.int64, device=device)

        # Everything below is reassigned for each shard
        self.aligned_sparsified_grad_shards = {}
        self.aligned_output = None
        self.sparse_readys = {}

        self.indexes = {}
        self.local_index_ready = {}
        self.count_sketches = {}
        self.count_mappings = {}

        self.global_index_ready = {}

        self.flag_zero = torch.zeros(1, dtype=torch.int32, device=device)
        self.decompress_unfinished = {}
        self.estimate_finished = {}

        self.local_count_sketch_ready = {}

    def next_grad(self):
        self.grad_num += 1

#    def add_grad_remainder(self, grad: torch.Tensor, shard: int):
#        index = self.grad_num * self.num_processes + shard
#        if index in self.grad_remainders:
#            grad += self.grad_remainders[index]

    def replace_grad_remainder(self, grad_remainder: torch.Tensor, shard: int):
        index = self.grad_num * self.num_processes + shard
        self.grad_remainders[index] = grad_remainder

    def end_iteration(self):
        self.grad_num = 0
        self.iter_num += 1
        for shard in self.aligned_sparsified_grad_shards:
            self.aligned_sparsified_grad_shards[shard] = None
        self.aligned_output = None

        for shard in self.indexes:
            self.indexes[shard] = None
        for shard in self.count_sketches:
            self.count_sketches[shard] = None
        for shard in self.count_mappings:
            self.count_mappings[shard] = None

    def finish_decompress_estimate(self, shard: int):
        if shard != self.process_index:
            return
        if self.estimate_finished[shard]:
            return
        grad_shard = self.aligned_sparsified_grad_shards[shard]
        output = self.aligned_output
        with torch.cuda.stream(self.cuda_stream_comp):
            while self.decompress_unfinished[shard][0] != self.flag_zero[0]:
                self.decompress_unfinished[shard][0] = self.flag_zero[0]
                # As output is size of shard and count_sketch is created for shard, they should match in sizes
                api.torch_launch_decompress_bfloat_16(
                    output, self.count_sketches[shard], self.count_mappings[shard],
                    self.compressed_r, self.grid_size, BLOCK_SIZE, self.decompress_unfinished[shard],
                    self.cuda_stream_comp.cuda_stream)

            api.torch_launch_estimate_bfloat_16(
                output, self.count_sketches[shard], self.compressed_r,
                self.grid_size, BLOCK_SIZE, self.estimate_counter,
                self.cuda_stream_comp.cuda_stream)
            self.total_counter += grad_shard.numel()

    def read_compress_index(self, shard: int,
                        grad_shard: torch.Tensor, output: torch.Tensor):
        if shard in self.local_count_sketch_ready:
            return
        with torch.cuda.stream(self.cuda_stream_comp):
            self.global_index_ready[shard].wait()

            # Artificial op in comp stream so that reduce begins before read_index
            self.flag_zero.zero_()

            # sets 0/1 to lowest bits of gradient -- this is used by compress
            if self.index_size == 1:
                api.torch_launch_read_index_1_bit_bf16(grad_shard, self.indexes[shard], self.grid_size, BLOCK_SIZE,
                                                       self.cuda_stream_comp.cuda_stream)
            else:
                api.torch_launch_read_index_4_bit_bf16(grad_shard, self.indexes[shard], self.grid_size, BLOCK_SIZE,
                                                       self.cuda_stream_comp.cuda_stream)

            api.torch_launch_compress_bfloat_16(grad_shard, self.count_sketches[shard],
                                            self.count_mappings[shard], self.compressed_r, self.grid_size, BLOCK_SIZE,
                                            self.cuda_stream_comp.cuda_stream)
            self.local_count_sketch_ready[shard] = torch.cuda.Event()
            self.local_count_sketch_ready[shard].record()


def reduce_hook(state: default_hooks.DefaultState, grad: torch.Tensor, dst: int, skip_divide=False):
    r"""
    Implement the  FSDP communication hook for ``reduce`` algorithm and a necessary pre- and post-division of gradients.

    Args:
        state (DefaultState): State information, configures pre- and post-division factors.
        grad (torch.Tensor): A gradient for the local batch that needs to be communicated across ranks.
    """
    # Average grad by pre-division factor. Together pre- and post-division factors
    # lead to an overall averaging by world_size, required for consistency with PyTorch DDP.
    # This is a two-step process to avoid potential underflow and overflow.
    if state.gradient_predivide_factor > 1 and not skip_divide:
        grad.div_(state.gradient_predivide_factor)
    dist.reduce(grad, dst, group=state.process_group)
    # Average grad by post-division factor.
    if state.gradient_postdivide_factor > 1 and not skip_divide:
        grad.div_(state.gradient_postdivide_factor)


def allreduce_hook(state: default_hooks.DefaultState, grad: torch.Tensor, skip_divide=False):
    r"""
    Implement the  FSDP communication hook for ``all_reduce`` algorithm and a necessary pre- and post-division of gradients.

    Args:
        state (DefaultState): State information, configures pre- and post-division factors.
        grad (torch.Tensor): A gradient for the local batch that needs to be communicated across ranks.
    """
    # Average grad by pre-division factor. Together pre- and post-division factors
    # lead to an overall averaging by world_size, required for consistency with PyTorch DDP.
    # This is a two-step process to avoid potential underflow and overflow.
    if state.gradient_predivide_factor > 1 and not skip_divide:
        grad.div_(state.gradient_predivide_factor)
    dist.all_reduce(grad, group=state.process_group)
    # Average grad by post-division factor.
    if state.gradient_postdivide_factor > 1 and not skip_divide:
        grad.div_(state.gradient_postdivide_factor)


SPARSIFY_SAMPLE_STEP = 100


def sparsify(state: HomomorphicCompressState, grad: torch.Tensor, shard: int):
    assert grad.ndim == 1 # Expecting flatten tensor

#    state.add_grad_remainder(grad, shard)
#    in_abs_out_result = torch.abs(grad)
    index = state.grad_num * state.num_processes + shard
    if not index in state.grad_remainders:
        state.grad_remainders [index]= torch.zeros_like(grad)
    in_abs_out_result = torch.empty_like(grad)
    tagc_kernels.torch_launch_sparsify_init(
        grad, state.grad_remainders[index], in_abs_out_result,
        state.grid_size, BLOCK_SIZE,
        state.cuda_streams_sparse[shard].cuda_stream)

    if SPARSIFY_SAMPLE_STEP == 0:
        grad_sample = in_abs_out_result
    else:
        grad_sample = in_abs_out_result[torch.randint(grad.numel(), (grad.numel() // SPARSIFY_SAMPLE_STEP,), device=grad.device)]

    def fraction_from_step(step: float):
        result = torch.where(grad_sample < step, 0., grad_sample)
        result_fraction = 1 - result.count_nonzero().item() / result.numel()
        return result_fraction

##    print(torch.cuda.memory_stats())
#    print(torch.cuda.memory_summary())

    sparsify_step_idx = bisect.bisect_left(SPARSIFY_STEPS, state.sparsify_fraction, key=fraction_from_step)
    if sparsify_step_idx == 0:
        return grad


    if sparsify_step_idx < len(SPARSIFY_STEPS):
        step = SPARSIFY_STEPS[sparsify_step_idx - 1]
    else:
        step = SPARSIFY_STEPS[len(SPARSIFY_STEPS) - 1]

    assert grad.numel() % (state.grid_size * BLOCK_SIZE) == 0
    tagc_kernels.torch_launch_sparsify_set(
        grad, in_abs_out_result,
        state.grid_size, BLOCK_SIZE, step,
        state.cuda_streams_sparse[shard].cuda_stream)
    return in_abs_out_result


def save_remainder_after_sparse(state: HomomorphicCompressState, full_grad: torch.Tensor, shard: int):
    with torch.cuda.stream(state.cuda_stream_comp):
        # Artificial op in comp stream so that reduce begins before replace_grad_remainder
        state.flag_zero.zero_()

        grad_shard = full_grad[shard * full_grad.numel() // state.num_processes:(shard + 1) * full_grad.numel() // state.num_processes]
        aligned_grad_shard = grad_shard[:grad_shard.numel() // BLOCK_SIZE * BLOCK_SIZE]
        state.replace_grad_remainder(aligned_grad_shard - state.aligned_sparsified_grad_shards[shard], shard)


def align_sparsify_and_create_index(state: HomomorphicCompressState, shard : int,
                       full_grad: torch.Tensor, output: torch.Tensor):
    if shard in state.local_index_ready:
        return
    with torch.cuda.stream(state.cuda_streams_sparse[shard]):
        # For the critical path on shard 0, especially all-reduce to finish soon
        # we don't start second sparse before the 1st sparse+create_index is finished
        if shard > 0:
            state.local_index_ready[shard-1].wait()
        grad_shard = full_grad[shard * full_grad.numel() // state.num_processes:(shard + 1) * full_grad.numel() // state.num_processes]
        assert grad_shard.numel() == output.numel()

        aligned_grad_shard = grad_shard[:grad_shard.numel() // BLOCK_SIZE * BLOCK_SIZE]
        state.grid_size = (aligned_grad_shard.numel() + BLOCK_SIZE - 1)//BLOCK_SIZE
        aligned_output = output[:grad_shard.numel() // BLOCK_SIZE * BLOCK_SIZE]

        aligned_sparsified_grad_shard = sparsify(state, aligned_grad_shard, shard)
#       Now working with bfloat16
#        aligned_sparsified_grad_shard = aligned_sparsified_grad_shard.to(torch.float32)
        state.aligned_sparsified_grad_shards[shard] = aligned_sparsified_grad_shard
        state.aligned_output = aligned_output
        sparse_ready = torch.cuda.Event()
        sparse_ready.record()
        state.sparse_readys[shard] = sparse_ready

    create_index(state, shard, aligned_sparsified_grad_shard, aligned_output)


def compress_aligned_sparsified(
        state: HomomorphicCompressState, shard : int,
        full_grad: torch.Tensor, output: torch.Tensor):

    def pre_compute_next_shard():
        if shard < state.num_processes - 1:
            align_sparsify_and_create_index(state, shard + 1, full_grad, output)
        if shard < state.num_processes - 1:
            all_reduce_index(state, shard + 1, full_grad, output)

    def pre_communicate_next_shard_and_finish_compute_prev_shard():
        if shard < state.num_processes - 1:
            all_reduce_index(state, shard + 1, full_grad, output)
        if shard < state.num_processes - 2:
            all_reduce_index(state, shard + 2, full_grad, output)
        if not state.cuda_stream_nccl.query() and shard < state.num_processes - 1:
            state.read_compress_index(shard + 1, full_grad, output)

        if shard > 0:
            save_remainder_after_sparse(state, full_grad, shard - 1)
        if shard == state.num_processes - 1:
            save_remainder_after_sparse(state, full_grad, shard)

        if shard > 0:
            state.finish_decompress_estimate(shard - 1)


    homomorphic_compress_shard_aligned_sparsified(state, shard, pre_compute_next_shard,
                                        pre_communicate_next_shard_and_finish_compute_prev_shard)


def reduce_remainder(state: HomomorphicCompressState, shard : int,
        full_grad: torch.Tensor, output: torch.Tensor):
    if full_grad.numel() % (state.num_processes * BLOCK_SIZE) == 0:
        return

    with torch.cuda.stream(state.cuda_stream_comp):
        grad_shard = full_grad[shard * full_grad.numel() // state.num_processes:(shard + 1) * full_grad.numel() // state.num_processes]
        grad_shard_remainder = grad_shard[grad_shard.numel() // BLOCK_SIZE * BLOCK_SIZE:]
#       Now working with bfloat16
#        grad_shard_remainder = grad_shard_remainder.to(torch.float32)
        # Assuming slice provides view for actual tensor, not copying it
        # https://stackoverflow.com/questions/61964164/pytorch-tensor-slice-and-memory-usage
        output_remainder = output[output.numel() // BLOCK_SIZE * BLOCK_SIZE:]
        if shard == state.process_index:
            output_remainder.copy_(grad_shard_remainder)
        remainder_ready = torch.cuda.Event()
        remainder_ready.record()

    with torch.cuda.stream(state.cuda_stream_nccl):
        remainder_ready.wait()

        # TODO if quality drops set skip_divide=False
        # it is set to True to avoid postpoining this reduce too much
        # as ops to nccl stream are added after 1 more operation in default stream
        # and division postpones this even by 1 more operation
        reduce_hook(state, output_remainder if shard == state.process_index else grad_shard_remainder, shard,
                    skip_divide=True)


def homomorphic_compress_shard_not_aligned(state: HomomorphicCompressState, shard : int,
                                           full_grad: torch.Tensor, output: torch.Tensor):
    align_sparsify_and_create_index(state, shard, full_grad, output)
    compress_aligned_sparsified(state, shard, full_grad, output)


def create_index(state: HomomorphicCompressState, shard : int,
                 grad_shard: torch.Tensor, output: torch.Tensor):
    if shard in state.local_index_ready:
        return
    with torch.cuda.stream(state.cuda_stream_comp):
        state.compressed_r = math.ceil(state.grid_size / state.compress_ratio)
        if state.compress_ratio <= 1:
            state.local_index_ready[shard] = None
            return

        if state.index_size == 1:
            state.indexes[shard] = torch.zeros(grad_shard.numel()//8, dtype=torch.uint8, device=grad_shard.device)
        else:
            state.indexes[shard] = torch.zeros(grad_shard.numel()//2, dtype=torch.uint8, device=grad_shard.device)
        state.count_sketches[shard] = torch.zeros(state.compressed_r * BLOCK_SIZE, dtype=torch.bfloat16, device=grad_shard.device)
        state.count_mappings[shard] = torch.zeros(state.compressed_r * BLOCK_SIZE, dtype=torch.uint8, device=grad_shard.device)
        state.decompress_unfinished[shard] = torch.ones(1, dtype=torch.int32, device=grad_shard.device)

        state.sparse_readys[shard].wait()
        if state.index_size == 1:
            api.torch_launch_create_index_1_bit_bf16(grad_shard, state.indexes[shard], state.grid_size, BLOCK_SIZE,
                                                     state.nonzero_counter, state.cuda_stream_comp.cuda_stream)
        else:
            api.torch_launch_create_index_4_bit_bf16(grad_shard, state.indexes[shard], state.grid_size, BLOCK_SIZE,
                                                     state.nonzero_counter, state.cuda_stream_comp.cuda_stream)
        state.local_index_ready[shard] = torch.cuda.Event()
        state.local_index_ready[shard].record()


def all_reduce_index(state: HomomorphicCompressState, shard: int,
                     grad_shard: torch.Tensor, output: torch.Tensor):
    if shard in state.global_index_ready:
        return
    if state.compress_ratio <= 1:
        if shard == state.process_index:
            output.copy_(grad_shard)
        reduce_hook(state, output if shard == state.process_index else grad_shard, shard)
        state.global_index_ready[shard] = None
        return

    with torch.cuda.stream(state.cuda_stream_nccl):
        state.local_index_ready[shard].wait()
        allreduce_hook(state, state.indexes[shard], skip_divide=True)
        state.global_index_ready[shard] = torch.cuda.Event()
        state.global_index_ready[shard].record()


def homomorphic_compress_shard_aligned_sparsified(state: HomomorphicCompressState, shard : int,
                                       network_time_callback: Callable[[],None],
                                       network_time_2_callback: Callable[[],None]):
    grad_shard = state.aligned_sparsified_grad_shards[shard]
    output = state.aligned_output
    create_index(state, shard, grad_shard, output)

    all_reduce_index(state, shard, grad_shard, output)

    network_time_callback()

    state.read_compress_index(shard, grad_shard, output)

    # TODO add saving diff in gradient and accumulating it
    with torch.cuda.stream(state.cuda_stream_nccl):
        state.local_count_sketch_ready[shard].wait()
        # skip_divide=True allows reduce not to wait for GPU compute (e.g. default_stream) availablilty
        reduce_hook(state, state.count_sketches[shard], shard, skip_divide=True)
        global_count_sketch_ready = torch.cuda.Event()
        global_count_sketch_ready.record()

    network_time_2_callback()

    if shard != state.process_index:
        return

#    FIXME TODO what is better, copy or zero? What is the difference in error/loss?
#    output.copy_(grad_shard)
    with torch.cuda.stream(state.cuda_stream_comp):
        output.zero_()
        if state.index_size == 1:
            api.torch_launch_read_index_1_bit_bf16(output, state.indexes[shard], state.grid_size,
                                                   BLOCK_SIZE, state.cuda_stream_comp.cuda_stream)
        else:
            api.torch_launch_read_index_4_bit_bf16(output, state.indexes[shard], state.grid_size,
                                                   BLOCK_SIZE, state.cuda_stream_comp.cuda_stream)

        global_count_sketch_ready.wait()

        finished_decompress_estimate = True
        state.estimate_finished[shard] = False

        while state.decompress_unfinished[shard][0] != state.flag_zero[0]:
            if shard < state.num_processes - 1 and state.cuda_stream_nccl.query():
                finished_decompress_estimate = False
                break

            state.decompress_unfinished[shard][0] = state.flag_zero[0]
            # As output is size of shard and count_sketch is created for shard, they should match in sizes
            api.torch_launch_decompress_bfloat_16(
                output, state.count_sketches[shard], state.count_mappings[shard],
                state.compressed_r, state.grid_size, BLOCK_SIZE, state.decompress_unfinished[shard],
                state.cuda_stream_comp.cuda_stream)

        if finished_decompress_estimate:
            if shard == state.num_processes - 1 or not state.cuda_stream_nccl.query():
                api.torch_launch_estimate_bfloat_16(output, state.count_sketches[shard], state.compressed_r, state.grid_size,
                                                    BLOCK_SIZE, state.estimate_counter, state.cuda_stream_comp.cuda_stream)
                state.total_counter += grad_shard.numel()
                state.estimate_finished[shard] = True


class TAGCState(HomomorphicCompressState):
    __slots__ = [
        "is_transformer_hook",
    ]

    def __init__(
        self,
        process_group: int,
        num_processes: int,
        process_index: int,
        sparsify_fraction: float,
        index_size: IndexSize,
        compress_ratio: float,
        device: str,
        is_transformer_hook: Callable[[int], bool],
        parameter_type: torch.dtype
    ):
        super().__init__(process_group, num_processes, process_index,
                         sparsify_fraction, index_size, compress_ratio, device, parameter_type)
        self.is_transformer_hook = is_transformer_hook


def transformer_compress_hook(state: TAGCState, grad: torch.Tensor, output: Optional[torch.Tensor] = None):
    with torch.profiler.record_function(
        "tagc.transformer_compress_hook"
    ):
        if output is None:
            default_hooks.allreduce_hook(state, grad)

#       Now working with bfloat16
#        if output.dtype != torch.float32:
#            output.data = output.data.to(torch.float32)

        if not state.is_transformer_hook(grad.numel()):
            homomorphic_compress_hook(state, grad, output)
            default_hooks._decompress(state, output)
        else:
#            fp32_compress_hook(state, grad, output)
            default_hooks.bf16_compress_hook(state, grad, output)

def homomorphic_compress_hook(state: HomomorphicCompressState, grad: torch.Tensor, output: Optional[torch.Tensor] = None):
    if output is None:
        return default_hooks.allreduce_hook(state, grad)

    # For easy handling in CUDA of non-adjusted grads
    # padding
    assert len(grad.shape) == 1 # gradient should be flattened

    comm_hook_gradient_ready = torch.cuda.Event()
    comm_hook_gradient_ready.record()

    state.local_index_ready = {}
    state.global_index_ready = {}
    state.local_count_sketch_ready = {}

    for i in range(state.num_processes):
        with torch.cuda.stream(state.cuda_streams_sparse[i]):
            comm_hook_gradient_ready.wait()

    for i in range(state.num_processes):
        homomorphic_compress_shard_not_aligned(state, i, grad, output)

    for i in range(state.num_processes):
        reduce_remainder(state, i, grad, output)

    state.next_grad()
