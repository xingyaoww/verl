# Megatron SFT Training for MoE Models

This directory contains scripts and configurations for running Supervised Fine-Tuning (SFT) on Mixture of Experts (MoE) models using the Megatron backend. This enables efficient training of large MoE models like DeepSeek-V3, Qwen3-236B, and Mixtral with expert parallelism.

## Features

- **Expert Parallelism**: Distribute MoE experts across multiple GPUs
- **Pipeline Parallelism**: Split model layers across pipeline stages
- **Tensor Parallelism**: Distribute attention and MLP computations
- **Context Parallelism**: Handle long sequences efficiently
- **Distributed Checkpointing**: Save and load large model checkpoints
- **Memory Offloading**: Offload parameters, gradients, and optimizer states
- **Gradient Checkpointing**: Reduce memory usage during training

## Supported Models

- **DeepSeek-V3** (671B parameters): Ultra-large MoE model
- **Qwen3-236B-A22B**: Large MoE model with 22B active parameters
- **Mixtral 8x7B**: Popular open-source MoE model
- **Qwen2 MoE**: Various sizes of Qwen2 MoE models

## Prerequisites

1. **Install VERL with Megatron support**:
   ```bash
   # Use the recommended Docker image
   docker pull whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6.post5-mcore0.12.1-te2.3-deepseekv3
   ```

2. **Convert HuggingFace models to Megatron format**:
   ```bash
   # For DeepSeek-V3 (distributed conversion recommended)
   torchrun --nproc_per_node 1 --nnodes 4 --node_rank ${RANK} \
     scripts/converter_hf_to_mcore.py \
     --hf_model_path /path/to/deepseek-v3 \
     --output_path /path/to/deepseek-v3-mcore \
     --use_cpu_initialization

   # For smaller models (single node conversion)
   python scripts/converter_hf_to_mcore.py \
     --hf_model_path /path/to/mixtral-8x7b \
     --output_path /path/to/mixtral-8x7b-mcore \
     --use_cpu_initialization
   ```

3. **Prepare your dataset**:
   - Format: Parquet files with `prompt` and `response` columns
   - Or use multi-turn format with `messages` column

## Usage

### 1. DeepSeek-V3 (671B)

```bash
# Update paths in the script
vim run_deepseek_v3_sft_megatron.sh

# Run on 8 nodes with 8 GPUs each (64 GPUs total)
NNODES=8 bash run_deepseek_v3_sft_megatron.sh
```

**Key configurations for DeepSeek-V3**:
- Expert Parallelism: 32
- Pipeline Parallelism: 16
- Tensor Parallelism: 1
- Memory offloading enabled
- Distributed checkpointing required

### 2. Qwen3-236B

```bash
# Update paths in the script
vim run_qwen3_236b_sft_megatron.sh

# Run on 4 nodes with 8 GPUs each (32 GPUs total)
NNODES=4 bash run_qwen3_236b_sft_megatron.sh
```

**Key configurations for Qwen3-236B**:
- Expert Parallelism: 4
- Pipeline Parallelism: 8
- Tensor Parallelism: 4
- Memory offloading enabled

### 3. Mixtral 8x7B

```bash
# Update paths in the script
vim run_mixtral_8x7b_sft_megatron.sh

# Run on 2 nodes with 8 GPUs each (16 GPUs total)
NNODES=2 bash run_mixtral_8x7b_sft_megatron.sh
```

**Key configurations for Mixtral**:
- Expert Parallelism: 4
- Pipeline Parallelism: 2
- Tensor Parallelism: 2
- Memory offloading optional

## Configuration Guide

### Parallelism Strategy

The total number of GPUs must equal: `TP × PP × EP × DP`

Where:
- **TP** (Tensor Parallelism): Splits attention/MLP within layers
- **PP** (Pipeline Parallelism): Splits layers across pipeline stages
- **EP** (Expert Parallelism): Splits MoE experts across GPUs
- **DP** (Data Parallelism): Automatically calculated

### Memory Optimization

For large models, enable offloading:
```yaml
megatron:
  param_offload: True      # Offload parameters to CPU
  grad_offload: True       # Offload gradients to CPU
  optimizer_offload: True  # Offload optimizer states to CPU
```

### MoE-Specific Settings

```yaml
model:
  override_config:
    moe_config:
      freeze_moe_router: False  # Set to True to freeze router during training

megatron:
  expert_model_parallel_size: 8  # Number of expert parallel groups
```

### Gradient Checkpointing

For memory efficiency:
```yaml
model:
  enable_gradient_checkpointing: True
```

## Data Format

### Single-turn SFT
```json
{
  "prompt": "What is the capital of France?",
  "response": "The capital of France is Paris."
}
```

### Multi-turn SFT
```json
{
  "messages": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What about Germany?"},
    {"role": "assistant", "content": "The capital of Germany is Berlin."}
  ]
}
```

## Performance Tips

1. **Batch Size Tuning**: Start with small micro batch sizes and increase gradually
2. **Sequence Length**: Use appropriate max_length for your use case
3. **Learning Rate**: MoE models often need lower learning rates (1e-6 to 2e-5)
4. **Expert Load Balancing**: Monitor expert utilization during training
5. **Memory Monitoring**: Use `nvidia-smi` to monitor GPU memory usage

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce micro batch size or enable more offloading
2. **Slow Training**: Check expert load balancing and reduce sequence length
3. **Convergence Issues**: Adjust learning rate and warmup steps
4. **Checkpoint Loading**: Ensure distributed checkpoint format matches

### Debugging

Enable detailed logging:
```bash
export VERL_SFT_LOGGING_LEVEL=INFO
export NCCL_DEBUG=INFO
```

## Comparison with FSDP SFT

| Feature | FSDP SFT | Megatron SFT |
|---------|----------|--------------|
| Model Support | Dense models | MoE + Dense models |
| Expert Parallelism | ❌ | ✅ |
| Pipeline Parallelism | ❌ | ✅ |
| Memory Efficiency | Good | Excellent |
| Scalability | Limited | Excellent |
| Setup Complexity | Simple | Moderate |

## Next Steps

After SFT training, you can:
1. **Continue with RL training** using the same Megatron backend
2. **Convert back to HuggingFace format** for inference
3. **Merge LoRA adapters** if using LoRA fine-tuning
4. **Deploy with vLLM** for efficient inference

## References

- [VERL Documentation](https://verl.readthedocs.io/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [DeepSeek-V3 Paper](https://arxiv.org/abs/2412.19437)
- [Mixtral Paper](https://arxiv.org/abs/2401.04088)