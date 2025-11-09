# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# regular:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \

# peft:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16
"""
import logging
import os
from contextlib import nullcontext
import timeit

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import init_zero_verbose, SftScriptArguments, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset

from tqdm.rich import tqdm
from transformers import AutoTokenizer, TrainingArguments

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)

import functools
from accelerate import init_empty_weights
from transformers.utils import ContextManagers
from transformers.modeling_utils import is_local_dist_rank_0
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from accelerate.utils.dataclasses import FullyShardedDataParallelPlugin
from transformers import TrainerCallback

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


if __name__ == "__main__":
    parser = TrlParser((SftScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        return_dict=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    raw_datasets = load_dataset(args.dataset_name)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training
    ################
    training_args.accelerator_config.fsdp_plugin = FullyShardedDataParallelPlugin(
        mixed_precision_policy=torch.distributed.fsdp.MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=None,
        ),
        auto_wrap_policy=lambda model: functools.partial(
            torch.distributed.fsdp.wrap.lambda_auto_wrap_policy,
            lambda_fn=lambda m: (
                m is model.model.embed_tokens
                or m in model.model.layers
                or m is model.lm_head
            ),
        ),
        activation_checkpointing_auto_wrap_policy=lambda model: functools.partial(
            torch.distributed.fsdp.wrap.lambda_auto_wrap_policy,
            lambda_fn=lambda m: (
                m in (
                    next(layer.children()) for layer in model.model.layers[:(
                        len(model.model.layers)
                        if training_args.accelerator_config.fsdp_plugin.num_layers_to_checkpoint is None
                        else training_args.accelerator_config.fsdp_plugin.num_layers_to_checkpoint
                    )]
                )
            )
        ),
        ya_fsdp_modules_to_wrap_with_names=lambda model: (
            [(model.model.embed_tokens, "model.embed_tokens")]
            + [
                (
                    m,
                    f"model.layers.{i}",
                )
                for i, m in enumerate(model.model.layers)
            ]
            + [(model.lm_head, "lm_head")]
        ),
        ya_fsdp_rogue_layer_norm_modules_with_names=lambda model: {model.model.norm: "model.norm"},
        ya_fsdp_layer_norm_module_cls=LlamaRMSNorm,
    )

    class ProfCallback(TrainerCallback):
        def __init__(self, prof):
            self.prof = prof

        def on_step_end(self, args, state, control, **kwargs):
            self.prof.step()

    def trace_handler(prof):
        if not trainer.is_world_process_zero():
            return
        prof.export_chrome_trace(
            f"{training_args.output_dir}/trace_{prof.step_num}.json"
        )

    class IterTimeCallback(TrainerCallback):
        def __init__(self):
            self.start_time = None
            self.iter_time = None

        def on_step_begin(self, args, state, control, **kwargs):
            assert self.start_time is None
            self.start_time = timeit.default_timer()

        def on_step_end(self, args, state, control, **kwargs):
            assert self.iter_time is None
            self.iter_time = timeit.default_timer() - self.start_time
            self.start_time = None

        def on_log(self, args, state, control, **kwargs):
            if self.iter_time is not None:
                kwargs["logs"]["iter_time"] = self.iter_time
            self.iter_time = None

    class MemorySnapshotCallback(TrainerCallback):
        def __init__(self):
            torch.cuda.memory._record_memory_history()

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 10 and trainer.args.process_index in range(8):
                torch.cuda.memory._dump_snapshot(
                    f"{training_args.output_dir}/memory_snapshot_{trainer.args.process_index}.pickle"
                )

    with ContextManagers([init_context] + ([] if is_local_dist_rank_0() else [init_empty_weights()])):
        trainer = SFTTrainer(
            model=model_config.model_name_or_path,
            model_init_kwargs=model_kwargs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field=args.dataset_text_field,
            max_seq_length=args.max_seq_length,
            tokenizer=tokenizer,
            packing=args.packing,
            peft_config=get_peft_config(model_config),
            callbacks=(([RichProgressCallback] if TRL_USE_RICH else []) + [IterTimeCallback, MemorySnapshotCallback])
        )

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=10, warmup=10, active=1, repeat=1),
            on_trace_ready=trace_handler,
    ) as prof:
        trainer.add_callback(ProfCallback(prof=prof))
        train_result = trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)

        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
