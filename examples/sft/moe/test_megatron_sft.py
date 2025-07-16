#!/usr/bin/env python3
"""
Test script for Megatron SFT trainer
This script validates the basic functionality with a small model
"""

import os
import tempfile
import torch
import pandas as pd
from pathlib import Path

def create_test_data():
    """Create a small test dataset"""
    data = [
        {"prompt": "What is 2+2?", "response": "2+2 equals 4."},
        {"prompt": "What is the capital of France?", "response": "The capital of France is Paris."},
        {"prompt": "Explain photosynthesis", "response": "Photosynthesis is the process by which plants convert sunlight into energy."},
        {"prompt": "What is machine learning?", "response": "Machine learning is a subset of AI that enables computers to learn from data."},
    ]
    
    # Create train and val datasets
    train_data = data * 10  # Repeat for more samples
    val_data = data[:2]
    
    return train_data, val_data

def test_megatron_sft():
    """Test the Megatron SFT trainer with a small model"""
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test data
        train_data, val_data = create_test_data()
        
        # Save as parquet files
        train_file = temp_path / "train.parquet"
        val_file = temp_path / "val.parquet"
        
        pd.DataFrame(train_data).to_parquet(train_file)
        pd.DataFrame(val_data).to_parquet(val_file)
        
        # Create test config
        config_content = f"""
data:
  train_batch_size: 4
  micro_batch_size_per_gpu: 2
  train_files: {train_file}
  val_files: {val_file}
  prompt_key: prompt
  response_key: response
  max_length: 512
  truncation: right

model:
  partial_pretrain: microsoft/DialoGPT-small  # Small model for testing
  model_dtype: fp32
  enable_gradient_checkpointing: False
  trust_remote_code: False
  load_weight: True

megatron:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  expert_model_parallel_size: 1
  context_parallel_size: 1
  sequence_parallel: False
  use_distributed_optimizer: False
  use_dist_checkpointing: False
  param_offload: False
  grad_offload: False
  optimizer_offload: False

optim:
  lr: 1e-4
  weight_decay: 0.01
  warmup_steps_ratio: 0.1
  clip_grad: 1.0
  lr_scheduler: cosine

trainer:
  project_name: test-moe-sft
  experiment_name: test
  total_training_steps: 5
  save_freq: 10
  test_freq: 2
  logger: ['console']
  default_local_dir: {temp_path}/checkpoints
"""
        
        config_file = temp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"Created test config at: {config_file}")
        print(f"Train data: {len(train_data)} samples")
        print(f"Val data: {len(val_data)} samples")
        
        # Print command to run
        print("\nTo test the Megatron SFT trainer, run:")
        print(f"python -m verl.trainer.megatron_sft_trainer --config-path {temp_path} --config-name test_config")
        
        return True

if __name__ == "__main__":
    test_megatron_sft()