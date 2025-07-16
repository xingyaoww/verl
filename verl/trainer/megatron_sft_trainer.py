# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
A Megatron-based SFT Trainer for MoE models
This trainer supports:
- Expert parallelism for MoE models
- Pipeline parallelism
- Tensor parallelism
- Context parallelism
- Distributed checkpointing
- Various offloading strategies
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
from contextlib import nullcontext
from typing import Dict, Any, Optional

import hydra
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, PreTrainedModel

# Megatron imports
from megatron.core import parallel_state as mpu
from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.models.gpt.gpt_model import ModelType

import verl.utils.hdfs_io as hdfs_io
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.tracking import Tracking
from verl.utils.megatron_utils import (
    get_model,
    get_megatron_optimizer,
    get_megatron_optimizer_param_scheduler,
    init_megatron_optim_config,
    load_mcore_dist_weights,
    load_megatron_gptmodel_weights,
)
from verl.utils.model import get_hf_model_path
from verl.models.mcore import hf_to_mcore_config, init_mcore_model
from verl.utils.megatron.pipeline_parallel import make_batch_generator

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


class MegatronSFTTrainer:
    def __init__(
        self,
        config: DictConfig,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        self.config = config
        self.tokenizer = tokenizer
        
        # Initialize Megatron distributed environment
        self._init_megatron_distributed()
        
        # normalize dp size
        self._normalize_config_bsz()
        
        self._build_dataloader(train_dataset, val_dataset)
        # build model
        self._build_model_optimizer()
        
        if mpu.get_data_parallel_rank() == 0:
            print(self.config)
        self.device_name = get_device_name()

    def _init_megatron_distributed(self):
        """Initialize Megatron distributed environment"""
        if not torch.distributed.is_initialized():
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=int(os.environ["WORLD_SIZE"]),
            )
        
        # Initialize Megatron parallel state
        from megatron.core import parallel_state
        
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
            expert_model_parallel_size=self.config.megatron.expert_model_parallel_size,
            context_parallel_size=self.config.megatron.get("context_parallel_size", 1),
            virtual_pipeline_model_parallel_size=self.config.megatron.get("virtual_pipeline_model_parallel_size", None),
        )

    def _normalize_config_bsz(self):
        dp_size = mpu.get_data_parallel_world_size()
        if mpu.get_data_parallel_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, (
            f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        )

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        # build dataset
        config = self.config
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # build dataloader
        # Use data parallel rank and size
        rank = mpu.get_data_parallel_rank()
        world_size = mpu.get_data_parallel_world_size()
        
        if mpu.get_data_parallel_rank() == 0:
            print(f"Using Megatron DP rank {rank} and size {world_size} for data distribution")

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        self.val_sampler = DistributedSampler(
            self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def _build_model_optimizer(self):
        """Build Megatron model and optimizer"""
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            import importlib
            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = self.config.model.get("model_dtype", "bf16")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        
        # Load HF config
        hf_config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.hf_config = hf_config
        
        # Convert to Megatron config
        override_transformer_config = OmegaConf.to_container(
            self.config.megatron.get("override_transformer_config", OmegaConf.create()), resolve=True
        )
        self.tf_config = hf_to_mcore_config(hf_config, torch_dtype, **override_transformer_config)

        # Model provider function for Megatron
        def megatron_model_provider(pre_process=True, post_process=True):
            """Model provider function for Megatron"""
            override_model_config = OmegaConf.to_container(
                self.config.model.get("override_config", OmegaConf.create()), resolve=True
            )
            
            model = init_mcore_model(
                self.tf_config,
                hf_config,
                pre_process=pre_process,
                post_process=post_process,
                share_embeddings_and_output_weights=False,
                value=False,
                freeze_moe_router=override_model_config.get("moe_config", {}).get("freeze_moe_router", False),
            )
            model.to(get_device_name())
            return model

        # Build model using Megatron utilities
        override_ddp_config = OmegaConf.to_container(
            self.config.megatron.get("override_ddp_config", OmegaConf.create()), resolve=True
        )
        
        self.model = get_model(
            megatron_model_provider,
            model_type=ModelType.encoder_or_decoder,
            wrap_with_ddp=True,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
            override_ddp_config=override_ddp_config,
            transformer_config=self.tf_config,
        )

        # Load weights
        if self.config.model.get("load_weight", True):
            if self.config.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(
                    self.model, self.config.megatron.dist_checkpointing_path, is_value_model=False
                )
            else:
                load_megatron_gptmodel_weights(
                    self.config, self.hf_config, self.model, params_dtype=torch_dtype, is_value_model=False
                )

        log_gpu_memory_usage("After model allocation", logger=logger)

        # Build optimizer
        optim_config_megatron = init_megatron_optim_config(self.config.optim)
        self.optimizer = get_megatron_optimizer(model=self.model, config=optim_config_megatron)
        self.optimizer_scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=self.optimizer, config=self.config.optim
        )

        log_gpu_memory_usage("After optimizer init", logger=logger)

    def compute_loss(self, batch):
        """Compute SFT loss using Megatron pipeline parallel"""
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        labels = batch["labels"]
        
        # Forward step function for pipeline parallel
        def forward_step_func(data_iterator, model):
            """Forward step function for Megatron pipeline parallel"""
            data = next(data_iterator)
            input_ids = data["input_ids"]
            attention_mask = data.get("attention_mask", None)
            
            # Forward pass through model
            if attention_mask is not None:
                output = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                output = model(input_ids=input_ids)
            
            return output.logits if hasattr(output, 'logits') else output

        # Loss function for pipeline parallel
        def loss_func(loss_mask, output_tensor):
            """Loss function for pipeline parallel"""
            if output_tensor is None:
                return None, {}
            
            logits = output_tensor
            # Shift labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross entropy
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Compute loss only on non-ignored tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits, shift_labels)
            
            # Reduce loss across data parallel groups
            if mpu.get_data_parallel_world_size() > 1:
                torch.distributed.all_reduce(loss, group=mpu.get_data_parallel_group())
                loss = loss / mpu.get_data_parallel_world_size()
            
            return loss, {"loss": loss.item()}

        # Use Megatron's forward-backward function
        forward_backward_func = get_forward_backward_func()
        
        # Create data iterator for pipeline parallel
        def data_iterator():
            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        
        # Create loss mask (not used in this case but required by Megatron)
        loss_mask = torch.ones_like(labels, dtype=torch.float)
        
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator(),
            model=self.model,
            num_microbatches=1,
            seq_length=input_ids.size(1),
            micro_batch_size=input_ids.size(0),
            decoder_seq_length=input_ids.size(1),
            forward_only=False,
            loss_func=loss_func,
        )
        
        return losses_reduced

    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Compute loss (backward pass is handled by forward_backward_func)
        losses_reduced = self.compute_loss(batch)
        
        # Finalize model gradients (required by Megatron)
        finalize_model_grads(self.model)
        
        # Clip gradients
        grad_norm = None
        if self.config.optim.get("clip_grad", 0) > 0:
            if hasattr(self.optimizer, 'clip_grad_norm'):
                # Use Megatron's gradient clipping
                grad_norm = self.optimizer.clip_grad_norm(self.config.optim.clip_grad)
            else:
                # Fallback to PyTorch's gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for model in self.model for p in model.parameters()], 
                    self.config.optim.clip_grad
                )
        
        # Optimizer step
        self.optimizer.step()
        if self.optimizer_scheduler is not None:
            self.optimizer_scheduler.step()
        
        # Extract loss value
        loss_value = 0.0
        if losses_reduced and len(losses_reduced) > 0:
            if isinstance(losses_reduced[0], dict):
                loss_value = losses_reduced[0].get("loss", 0.0)
            else:
                loss_value = float(losses_reduced[0])
        
        metrics = {
            "loss": loss_value,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        
        if grad_norm is not None:
            metrics["grad_norm"] = float(grad_norm)
        
        return metrics

    def train(self):
        """Main training loop"""
        total_steps = self.config.trainer.get("total_training_steps", None)
        total_epochs = self.config.trainer.get("total_epochs", 1)
        
        if total_steps is None:
            total_steps = len(self.train_dataloader) * total_epochs
        
        step = 0
        epoch = 0
        
        # Initialize tracking
        if mpu.get_data_parallel_rank() == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                logger=self.config.trainer.get("logger", ["console"]),
            )
        
        while step < total_steps and epoch < total_epochs:
            self.train_sampler.set_epoch(epoch)
            
            for batch in tqdm(self.train_dataloader, disable=mpu.get_data_parallel_rank() != 0):
                if step >= total_steps:
                    break
                
                # Move batch to device
                batch = {k: v.to(get_device_name()) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Training step
                metrics = self.train_step(batch)
                
                # Log metrics
                if mpu.get_data_parallel_rank() == 0:
                    tracking.log(metrics, step=step)
                
                # Validation
                if (step + 1) % self.config.trainer.get("test_freq", 100) == 0:
                    val_metrics = self.validate()
                    if mpu.get_data_parallel_rank() == 0:
                        tracking.log(val_metrics, step=step)
                
                # Save checkpoint
                if (step + 1) % self.config.trainer.get("save_freq", 1000) == 0:
                    self.save_checkpoint(step)
                
                step += 1
            
            epoch += 1
        
        # Final save
        self.save_checkpoint(step)

    def validate(self):
        """Validation loop"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move batch to device
                batch = {k: v.to(get_device_name()) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Compute loss
                losses = self.compute_loss(batch)
                total_loss += losses[0] if losses else 0.0
                num_batches += 1
                
                # Limit validation batches
                if num_batches >= 10:
                    break
        
        avg_loss = total_loss / max(num_batches, 1)
        return {"val_loss": avg_loss}

    def save_checkpoint(self, step):
        """Save model checkpoint"""
        if mpu.get_data_parallel_rank() == 0:
            print(f"Saving checkpoint at step {step}")
        
        # Use Megatron's distributed checkpointing if enabled
        if self.config.megatron.use_dist_checkpointing:
            from megatron.core import dist_checkpointing
            
            checkpoint_dir = os.path.join(
                self.config.trainer.default_local_dir, f"global_step_{step}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            state_dict = self.model[0].module.sharded_state_dict()
            dist_checkpointing.save(state_dict, checkpoint_dir)
        else:
            # Regular checkpoint saving
            if mpu.get_data_parallel_rank() == 0:
                checkpoint_dir = os.path.join(
                    self.config.trainer.default_local_dir, f"global_step_{step}"
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                torch.save({
                    "model": self.model[0].module.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "step": step,
                    "config": self.config,
                }, os.path.join(checkpoint_dir, "checkpoint.pt"))


@hydra.main(config_path="config", config_name="megatron_sft_trainer", version_base=None)
def main(config):
    """Main entry point for Megatron SFT training"""
    
    # Initialize distributed environment
    initialize_global_process_group()
    
    # Load tokenizer
    from transformers import AutoTokenizer
    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path, 
        trust_remote_code=config.model.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    if config.data.multiturn.enable:
        train_dataset = MultiTurnSFTDataset(
            parquet_files=config.data.train_files,
            tokenizer=tokenizer,
            max_length=config.data.max_length,
            messages_key=config.data.multiturn.messages_key,
            tools_key=config.data.multiturn.tools_key,
            enable_thinking_key=config.data.multiturn.enable_thinking_key,
        )
        val_dataset = MultiTurnSFTDataset(
            parquet_files=config.data.val_files,
            tokenizer=tokenizer,
            max_length=config.data.max_length,
            messages_key=config.data.multiturn.messages_key,
            tools_key=config.data.multiturn.tools_key,
            enable_thinking_key=config.data.multiturn.enable_thinking_key,
        )
    else:
        train_dataset = SFTDataset(
            parquet_files=config.data.train_files,
            tokenizer=tokenizer,
            prompt_key=config.data.prompt_key,
            response_key=config.data.response_key,
            prompt_dict_keys=config.data.prompt_dict_keys,
            response_dict_keys=config.data.response_dict_keys,
            max_length=config.data.max_length,
            truncation=config.data.truncation,
        )
        val_dataset = SFTDataset(
            parquet_files=config.data.val_files,
            tokenizer=tokenizer,
            prompt_key=config.data.prompt_key,
            response_key=config.data.response_key,
            prompt_dict_keys=config.data.prompt_dict_keys,
            response_dict_keys=config.data.response_dict_keys,
            max_length=config.data.max_length,
            truncation=config.data.truncation,
        )

    # Create trainer
    trainer = MegatronSFTTrainer(
        config=config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    
    # Start training
    trainer.train()
    
    # Cleanup
    destroy_global_process_group()


if __name__ == "__main__":
    main()