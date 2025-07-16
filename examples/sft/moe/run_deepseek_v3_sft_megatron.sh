#!/usr/bin/env bash
set -xeuo pipefail

# DeepSeek-V3 SFT with Megatron backend
# This script demonstrates how to run SFT on DeepSeek-V3 671B MoE model
# Requires distributed checkpointing format model

project_name='MoE-SFT'
exp_name='DeepSeek-V3-SFT-megatron'

max_length=2048
train_batch_size=128
micro_batch_size_per_gpu=1

# Number of nodes (adjust based on your setup)
NNODES=${NNODES:-8}

# Paths - update these to your actual paths
MODEL_PATH="<path_to_dsv3_config>"  # Path to DeepSeek-V3 config
MCORE_MODEL_PATH="<path_to_dpsk-v3-671B-BF16-dist_ckpt>"  # Path to distributed checkpoint
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/sft_train.parquet"}
VAL_FILE=${VAL_FILE:-"${RAY_DATA_HOME}/data/sft_val.parquet"}

# Performance Related Parameters
offload=True
train_tp=1      # Tensor parallelism
train_ep=32     # Expert parallelism (important for MoE)
train_pp=16     # Pipeline parallelism

# Learning rate and training steps
lr=1e-6
total_training_steps=1000
save_freq=100
test_freq=50

torchrun --standalone --nnodes=${NNODES} --nproc_per_node=8 \
    -m verl.trainer.megatron_sft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=${max_length} \
    data.train_batch_size=${train_batch_size} \
    data.micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    model.partial_pretrain="${MODEL_PATH}" \
    model.enable_gradient_checkpointing=True \
    model.model_dtype=bf16 \
    model.load_weight=True \
    model.override_config.moe_config.freeze_moe_router=False \
    megatron.param_offload=${offload} \
    megatron.optimizer_offload=${offload} \
    megatron.grad_offload=${offload} \
    megatron.pipeline_model_parallel_size=${train_pp} \
    megatron.tensor_model_parallel_size=${train_tp} \
    megatron.expert_model_parallel_size=${train_ep} \
    megatron.dist_checkpointing_path=${MCORE_MODEL_PATH} \
    megatron.use_dist_checkpointing=True \
    +megatron.override_transformer_config.num_layers_in_first_pipeline_stage=3 \
    +megatron.override_transformer_config.num_layers_in_last_pipeline_stage=2 \
    optim.lr=${lr} \
    optim.clip_grad=1.0 \
    optim.lr_scheduler=cosine \
    optim.warmup_steps_ratio=0.1 \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_training_steps=${total_training_steps} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.logger='["console","wandb"]'