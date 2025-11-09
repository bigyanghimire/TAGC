#!/bin/bash

set -o errexit

NUM_TRAIN_EPOCHS=$1

epoch=1
while [ $epoch -le $NUM_TRAIN_EPOCHS ] ; do

  echo ================================== Training epoch $epoch ======================================

  if [ $epoch -eq 1 ] ; then
    CHECKPOINTING_OPTIONS="--overwrite_output_dir"
  else
    CHECKPOINTING_OPTIONS="--resume_from_checkpoint 1"
  fi

  accelerate launch --config_file ya-fsdp/examples/fsdp_config.yaml --fsdp_ya_fsdp_enabled false \
    transformers/examples/pytorch/language-modeling/run_clm.py --do_train --do_eval --config_name ./config.json \
    --tokenizer_name HuggingFaceTB/SmolLM-360M-Instruct --block_size 2048 --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
    --save_strategy no --logging_steps 1 --report_to tensorboard --output_dir clm $CHECKPOINTING_OPTIONS \
    --token $HUGGINGFACE_TOKEN \
    --num_train_epochs $epoch --save_strategy epoch

  epoch=$((epoch+1))
done
