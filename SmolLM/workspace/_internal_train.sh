#!/bin/bash

export WANDB_PROJECT="owt"
MAX_RESTARTS=10000

restart=0

if [[ "$1" == "resume" ]]; then
    RESUME_FROM_CHECKPOINT="--resume_from_checkpoint 1"
else
    RESUME_FROM_CHECKPOINT="--overwrite_output_dir"
fi

./_oom_killer.sh &

#ulimit -m 30720000
#ulimit -m 512000 # Seemingly does not work
#ulimit -v 307200000

#cgcreate -g memory:trainGroup
#echo 5120M > /sys/fs/cgroup/memory/trainGroup/memory.limit_in_bytes

while [ $restart -le $MAX_RESTARTS ] ; do
    # prlimit -m=5120000 does not work as well
    # cgexec -g memory:trainGroup requires kernel kernel booting parameter modification ! may try with it
    # systemd-run --scope -p MemoryMax=512M --user does not work
    # ./timeout --memlimit-rss 512000 # hangs
    ./_accelerate.py launch \
        --config_file ya-fsdp/examples/fsdp_config.yaml --fsdp_ya_fsdp_enabled false \
        --main_process_ip $MASTER_ADDR --machine_rank $MACHINE_RANK \
        transformers/examples/pytorch/language-modeling/run_clm.py --do_train \
        --config_name ./config.json \
        --tokenizer_name HuggingFaceTB/SmolLM-360M-Instruct --block_size 2048 --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 --dataset_name openwebtext \
        --output_dir out/clm \
        --logging_steps 1  \
        --report_to tensorboard \
        --token $HUGGINGFACE_TOKEN \
        --save_strategy steps --save_steps 200 --save_on_each_node true \
        --ddp_timeout 60 \
        --evaluation_strategy steps --eval_steps 100000 \
        ${RESUME_FROM_CHECKPOINT}

    if [ $? -eq 0 ] ; then
        echo "Successfully finished accelerate launch."
        break
    fi

    echo "Unsuccessful accelerate launch, restarting from checkpoint."
    RESUME_FROM_CHECKPOINT="--resume_from_checkpoint 1"
    restart=$((restart+1))
done
