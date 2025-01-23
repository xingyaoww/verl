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
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import logging
import re
import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, CPUOffload
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, AutoConfig
from verl.utils.torch_functional import get_cosine_schedule_with_warmup
from tensordict import TensorDict
from torch.utils.data import DataLoader, DistributedSampler
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.dataset import SFTDataset
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.tracking import Tracking
from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, set_ulysses_sequence_parallel_group
from torch.distributed.device_mesh import DeviceMesh

import verl.utils.hdfs_io as hdfs_io
from verl.utils.debug import log_gpu_memory_usage
from peft import LoraConfig, TaskType, get_peft_model

from verl.workers.sharding_manager import FSDPUlyssesShardingManager
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))

def debug_print(msg, rank=None):
    from termcolor import colored
    prefix = f'rank={torch.distributed.get_rank()}'
    if rank is not None and torch.distributed.get_rank() == rank:
        print(colored(f'{prefix}: {msg}', "magenta"))
    elif rank is None:
        if torch.distributed.get_rank() == 0:
            print(colored(f'{prefix}: {msg}', "magenta")) 
        else:
            print(colored(f'{prefix}: {msg}', "green"))

def extract_step(path):
    match = re.search(r'global_step_(\d+)', path)
    if match:
        return int(match.group(1))
    return None


def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import ListConfig, DictConfig
    if isinstance(obj, (ListConfig, DictConfig)):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj


class FSDPSFTTrainer(object):

    def __init__(
            self,
            config,
            device_mesh: DeviceMesh,
            ulysses_device_mesh: DeviceMesh
        ):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        # build tokenizer first
        local_model_path = copy_local_path_from_hdfs(src=self.config.model.partial_pretrain, verbose=True)
        from verl.utils import hf_tokenizer
        self.tokenizer = hf_tokenizer(local_model_path, trust_remote_code=self.config.model.trust_remote_code)
        if self.config.data.chat_template is not None:
            raise ValueError('Apply Chat template from config is not supported yet.')

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, 'ulysses_sequence_parallel_size', 1)
        self.use_remove_padding = getattr(self.config, 'use_remove_padding', False)
        if self.device_mesh.get_rank() == 0:
            print(f'Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}')
            print(f'Using remove padding: {self.use_remove_padding}')

        self._build_dataloader()
        # build model
        self._build_model_optimizer()

        # TODO: add checkpoint manager
        if self.device_mesh.get_rank() == 0:
            print(self.config)

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f'Normalize batch size by dp {dp_size}')

        assert self.config.data.train_batch_size % dp_size == 0, f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        assert self.config.data.micro_batch_size % dp_size == 0, f"Micro batch size {self.config.data.micro_batch_size} is not divisible by dp size {dp_size}"

        self.config.data.train_batch_size //= dp_size
        self.config.data.micro_batch_size //= dp_size

    def _build_dataloader(self):
        config = self.config
        # build dataset
        self.train_dataset = SFTDataset(parquet_files=config.data.train_files,
                                        tokenizer=self.tokenizer,
                                        prompt_key=config.data.prompt_key,
                                        prompt_dict_keys=config.data.get('prompt_dict_keys', None),
                                        response_key=config.data.response_key,
                                        response_dict_keys=config.data.get('response_dict_keys', None),
                                        max_length=config.data.max_length,
                                        truncation=config.data.truncation,
                                        skip_template_apply=config.data.skip_template_apply)
        self.val_dataset = SFTDataset(parquet_files=config.data.val_files,
                                      tokenizer=self.tokenizer,
                                      prompt_key=config.data.prompt_key,
                                      prompt_dict_keys=config.data.get('prompt_dict_keys', None),
                                      response_key=config.data.response_key,
                                      response_dict_keys=config.data.get('response_dict_keys', None),
                                      max_length=config.data.max_length,
                                      truncation=config.data.truncation,
                                      skip_template_apply=config.data.skip_template_apply)

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank('dp')
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f'Using SP rank {rank} and size {world_size} for data distribution')
                print(f'Each SP rank gets different data, but the same data WITHIN the same rank')
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f'Using FSDP rank {rank} and size {world_size} for data distribution')

        self.train_sampler = DistributedSampler(self.train_dataset,
                                                shuffle=True,
                                                num_replicas=world_size,
                                                rank=rank,
                                                drop_last=True)
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=config.data.train_batch_size,
                                           sampler=self.train_sampler,
                                           num_workers=8,
                                           pin_memory=True,
                                           drop_last=True)
        debug_print(f'sampler initialized with num_replicas: {world_size}, rank: {rank}, drop_last: {True}')
        self.val_sampler = DistributedSampler(self.val_dataset,
                                              shuffle=True,
                                              num_replicas=world_size,
                                              rank=rank,
                                              drop_last=True)
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=config.data.micro_batch_size,
                                         sampler=self.val_sampler,
                                         num_workers=8,
                                         pin_memory=True,
                                         drop_last=True)

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_local_path_from_hdfs(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get('external_lib', None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib
            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage('Before model allocation', logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        if self.use_remove_padding:
            assert self.config.ulysses_sequence_parallel_size > 1, "Remove padding is only supported with sequence parallel"
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(config.model_type)

        if self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(config, verbose=True)
            debug_print(f'Model config after monkey patch: {config}')

        # This may be very large
        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings)

        # Perform RoPE scaling when self.config.model.rope_scaling is not None
        from verl.utils.model import print_model_size, update_model_config
        override_config_kwargs = {}
        if 'rope_scaling' in self.config.model and self.config.model.rope_scaling is not None:
            override_config_kwargs['rope_scaling'] = dict(self.config.model.rope_scaling)
            print(f'rope_scaling setted. rope_scaling={override_config_kwargs["rope_scaling"]}')
        update_model_config(config, override_config_kwargs=override_config_kwargs)
        debug_print(f'Model config after override: {config}', rank=0)


        with init_context():
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(local_model_path,
                                                                               config=config,
                                                                               torch_dtype=torch.float32,
                                                                               attn_implementation='flash_attention_2',
                                                                               trust_remote_code=trust_remote_code)
            
            if self.config.model.get('lora_rank', 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    'task_type': TaskType.CAUSAL_LM,
                    'r': self.config.model.lora_rank,
                    'lora_alpha': self.config.model.lora_alpha,
                    'target_modules': convert_to_regular_types(self.config.model.target_modules),
                    'bias': "none"
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

        log_gpu_memory_usage('After model allocation', logger=logger)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16,
                                         reduce_dtype=torch.float32,
                                         buffer_dtype=torch.float32)

        auto_wrap_policy = get_fsdp_wrap_policy(self.model, config=self.config.model.fsdp_config.wrap_policy)
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        self.fsdp_model = FSDP(module=self.model,
                               auto_wrap_policy=auto_wrap_policy,
                               param_init_fn=init_fn,
                               sharding_strategy=ShardingStrategy.FULL_SHARD,
                               mixed_precision=mixed_precision,
                               device_mesh=self.device_mesh,
                               sync_module_states=True,
                               device_id=torch.cuda.current_device(),
                               cpu_offload=cpu_offload,
                               use_orig_params=False)

        log_gpu_memory_usage('After FSDP wrapping', logger=logger)

        self.optimizer = optim.AdamW(self.fsdp_model.parameters(),
                                     lr=self.config.optim.lr,
                                     betas=self.config.optim.betas,
                                     weight_decay=self.config.optim.weight_decay)

        log_gpu_memory_usage('After initialize optimizer', logger=logger)

        steps_per_epoch = len(self.train_dataloader)
        total_steps = steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(
                f'Number of steps/epoch {steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {total_steps}'
            )

        num_warmup_steps = int(total_steps * self.config.optim.warmup_steps_ratio)

        self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=total_steps)

    def _compute_loss_and_backward(self, batch, do_backward=True):
        loss_mask = batch.pop('loss_mask')[:, :-1].reshape(-1).cuda()
        labels = batch['input_ids'][:, 1:].cuda()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = self.fsdp_model(input_ids=batch['input_ids'],
                                     attention_mask=batch['attention_mask'],
                                     position_ids=batch['position_ids'],
                                     use_cache=False)  # prevent model thinks it it generating

        logits = output.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels.contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(
            reduction='none',
            label_smoothing=self.config.optim.label_smoothing,
        )
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss * loss_mask

        valid_token_this_rank = torch.sum(loss_mask)

        if self.config.data.balance_dp_token:
            torch.distributed.all_reduce(valid_token_this_rank)  # becomes total valid tokens in all ranks
            dp_size = torch.distributed.get_world_size()
        else:
            dp_size = 1

        loss = torch.sum(loss) / valid_token_this_rank * dp_size  # possible bugs here for dp
        if do_backward:
            loss.backward()
        return loss

    def _compute_loss_and_backward_sp(self, batch, do_backward=True):
        """Compute loss with ulysses sequence parallelism and remove padding features enabled"""
        with self.sharding_manager:
            # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
            # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
            # 1. All SP ranks will receive the *SAME* batch
            # 2. Different SP groups will receive *DIFFERENT* batches
            # This is implemented by the DistributedSampler
            input_ids = batch['input_ids'].cuda()
            batch_size, seqlen = input_ids.shape
            attention_mask = batch['attention_mask'].cuda()
            position_ids = batch['position_ids'].cuda()
            loss_mask = batch['loss_mask'][:, :-1].reshape(-1).cuda()
            # debug_print(f'input_ids: {input_ids.shape}. Middle 10: {input_ids[:, 1000:1010]}, device: {input_ids.device}')
            # debug_print(f'attention_mask: {attention_mask.shape}, device: {attention_mask.device}')
            # debug_print(f'position_ids: {position_ids.shape}. Middle 10: {position_ids[:, 1000:1010]}, device: {position_ids.device}')
            # debug_print(f'loss_mask: {loss_mask.shape}, device: {loss_mask.device}')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

                # Remove padding
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # Unpad position_ids to align rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                  indices).transpose(0, 1)



                # Pad and slice inputs for sequence parallelism
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size()
                    # self.config.ulysses_sequence_parallel_size
                )
                # debug_print(f'input_ids_rmpad: {input_ids_rmpad.shape}; input_ids_rmpad_sliced: {input_ids_rmpad_sliced.shape}. Middle 10: {input_ids_rmpad_sliced[:, 1000:1010]}; position_ids_rmpad_padded: {position_ids_rmpad_padded.shape}. Middle 10: {position_ids_rmpad_padded[:, 1000:1010]}')
                # For computing loss
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size()
                )
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # debug_print(f'device_mesh: {self.device_mesh.get_rank()}, self.ulysses_device_mesh: {self.ulysses_device_mesh.get_rank()}, SP input_ids_rmpad_sliced: {input_ids_rmpad_sliced.shape} (device: {input_ids_rmpad_sliced.device}), position_ids_rmpad_padded: {position_ids_rmpad_padded.shape} (device: {position_ids_rmpad_padded.device})')
                
                # Forward pass
                debug_print(f'BEFORE forward. input_ids_rmpad_sliced: {input_ids_rmpad_sliced.shape} (device: {input_ids_rmpad_sliced.device}): FIRST 10: {input_ids_rmpad_sliced[:, :10]}')
                output = self.fsdp_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Not needed with flash attention varlen
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False
                )
                debug_print(f'AFTER forward. output: {output.logits.shape}')
                
                # Compute loss
                loss_fct = nn.CrossEntropyLoss(
                    reduction='none',
                    label_smoothing=self.config.optim.label_smoothing
                )
                
                # # Approach 1: Gather COMPLETE logprobs first
                # logits_split_in_seq = output.logits
                # logits_full = gather_outpus_and_unpad(logits_split_in_seq, gather_dim=1, unpad_dim=1, padding_size=pad_size)
                # debug_print(f'logits_full: {logits_full.shape}, input_ids_rmpad: {input_ids_rmpad.shape}, input_ids_rmpad_rolled: {input_ids_rmpad_rolled.shape}')
                # loss = loss_fct(logits_full.squeeze(0), input_ids_rmpad.squeeze(0))
                # debug_print(f'loss: {loss.shape}, loss_mask: {loss_mask.shape}')

                # They indeed match!
                # basically computing the loss on the current slice of sequence/labels

                # Approach 2: Calculate locally then aggregate
                logits_rmpad = output.logits.squeeze(0)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                debug_print(f'logits_rmpad: {logits_rmpad.shape}, input_ids_rmpad_rolled: {input_ids_rmpad_rolled.shape}')
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                # loss = loss_fct(logits_full, input_ids_rmpad)
                debug_print(f'loss: {loss.shape}')
                # Gather and unpad for sequence parallelism
                loss = gather_outpus_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                debug_print(f'loss (gathered): {loss.shape}, loss_mask: {loss_mask.shape}')
                
                # This is the loss collected from all ulysses ranks
                full_loss = pad_input(hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                full_loss = full_loss.reshape(-1)
                # debug_print(f'full_loss: {full_loss.shape} (device: {full_loss.device}), loss_mask: {loss_mask.shape} (device: {loss_mask.device})')
                loss_mask = loss_mask.to(full_loss.device)
                loss = full_loss * loss_mask
                valid_token_this_rank = torch.sum(loss_mask)

                if self.config.data.balance_dp_token:
                    debug_print("Balance DP token is on")
                    torch.distributed.all_reduce(valid_token_this_rank)  # becomes total valid tokens in all ranks
                    assert self.ulysses_device_mesh is not None
                    dp_size = self.ulysses_device_mesh.size('dp')
                else:
                    dp_size = 1

                loss = torch.sum(loss) / valid_token_this_rank * dp_size
                debug_print(f'loss (after reduction): {loss.shape}, {loss}')
                
                # IMPORTANT: WE need to DO BACKWARD inside _compute_loss_and_backward_sp (within the sharding manager)
                # Otherwise, the backward+grad checkpoint won't use grad checkpoint properly
                if do_backward:
                    loss.backward()
            return loss

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()

        log_gpu_memory_usage('Before optimizer zero_grad', logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage('After optimizer zero_grad', logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size)
        # debug_print(f'Micro batches: {len(micro_batches[0])}')
        # debug_print(f'Micro batches: {micro_batches}')
        n_micro_batches = len(micro_batches)
        step_loss = 0
        for micro_batch in micro_batches:
            if self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1:
                # micro_batch = micro_batch.to('cuda')
                loss = self._compute_loss_and_backward_sp(batch=micro_batch) / n_micro_batches
            else:
                assert self.use_remove_padding == False and self.config.ulysses_sequence_parallel_size == 1
                loss = self._compute_loss_and_backward(batch=micro_batch) / n_micro_batches
            step_loss += loss.item()
            debug_print(f'step_loss: {step_loss}')

        self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)

        log_gpu_memory_usage('Before optimizer step', logger=logger)

        self.optimizer.step()

        log_gpu_memory_usage('After optimizer step', logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage('After offload weights', logger=logger)

        step_loss = torch.tensor(step_loss).cuda()
        torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        return {'train/loss': step_loss.detach().item(), 'train/lr(1e-3)': lr * 1e3}

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            if self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1:
                loss = self._compute_loss_and_backward_sp(batch, do_backward=False)
            else:
                assert self.use_remove_padding == False and self.config.ulysses_sequence_parallel_size == 1
                loss = self._compute_loss_and_backward(batch, do_backward=False)
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        return loss

    def save_checkpoint(self, step):
        # save checkpoint
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.fsdp_model.state_dict()

        path = os.path.join(self.config.trainer.default_local_dir, f'global_step_{step}')
        # save huggingface model
        if self.device_mesh.get_rank() == 0:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path, state_dict=state_dict)
            self.tokenizer.save_pretrained(path)
            if self.config.trainer.default_hdfs_dir:
                hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
                hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)
        torch.distributed.barrier()
    
    def debug(self):
        """Debug function to compare original forward pass with ulysses_sp and use_remove_padding features"""
        if self.device_mesh.get_rank() == 0:
            print("\nStarting debug comparison between original and SP+rmpad forward passes...")
            print(f"Sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            print(f"Remove padding: {self.use_remove_padding}\n")

        total_steps = 4

        for epoch in range(1):  # Just one epoch for debugging
            self.train_sampler.set_epoch(epoch=epoch)
            for data in self.train_dataloader:
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).cuda()
                self.fsdp_model.train()
                micro_batches = data.split(self.config.data.micro_batch_size)
                
                for idx, micro_batch in enumerate(micro_batches):
                    if self.device_mesh.get_rank() == 0:
                        print(f"\nProcessing micro batch {idx + 1}/{len(micro_batches)}")
                    # Compute losses using both methods
                    loss_ref = self._compute_loss_and_backward(micro_batch.copy())
                    loss_sp = self._compute_loss_and_backward_sp(micro_batch.copy())
                    
                    # Collect losses across all ranks
                    loss_ref_all = loss_ref.clone()
                    loss_sp_all = loss_sp.clone()
                    torch.distributed.all_reduce(loss_ref_all, op=torch.distributed.ReduceOp.AVG)
                    torch.distributed.all_reduce(loss_sp_all, op=torch.distributed.ReduceOp.AVG)
                    
                    # Calculate relative difference of averaged losses
                    rel_diff = torch.abs(loss_ref_all - loss_sp_all) / (torch.abs(loss_ref_all) + 1e-8)
                    
                    if self.device_mesh.get_rank() == 0:
                        print("\nComparison Results (Averaged across ranks):")
                        print(f"Reference Loss: {loss_ref_all.item():.6f}")
                        print(f"SP+rmpad Loss: {loss_sp_all.item():.6f}") 
                        print(f"Relative Difference: {rel_diff.item():.6f}")
                        
                        if rel_diff.item() > 1e-5:
                            print("\nWARNING: Significant difference detected between averaged losses!")
                        else:
                            print("\nAveraged losses match within tolerance.")
                    
                    total_steps -= 1
                    if total_steps == 0:
                        break
                if total_steps == 0:
                    break
            break

        if self.device_mesh.get_rank() == 0:
            print("\nDebug comparison completed.")

    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(project_name=self.config.trainer.project_name,
                                experiment_name=self.config.trainer.experiment_name,
                                default_backend=self.config.trainer.logger)

        global_step = 0

        # TODO (zhangchi.usc1992) add back checkpoint manager. Currently, it blocks when uploading to hdfs. So very slow.

        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in self.train_dataloader:
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).cuda()
                metric = self.training_step(data)
                if rank == 0:
                    tracking.log(data=metric, step=global_step)
                global_step += 1

            # validation
            val_losses = []
            for data in self.val_dataloader:
                data = TensorDict(data, batch_size=self.config.data.micro_batch_size).cuda()
                val_loss = self.validation_step(data)
                val_losses.append(val_loss)
            if rank == 0:
                val_loss = torch.mean(torch.stack(val_losses))
                metric = {'val/loss': val_loss.detach().item()}
                tracking.log(data=metric, step=global_step)
            torch.distributed.barrier()

            # save checkpoint
            self.save_checkpoint(step=global_step)


from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
import hydra

from torch.distributed.device_mesh import init_device_mesh

from verl.utils.distributed import initialize_global_process_group


@hydra.main(config_path='config', config_name='sft_trainer', version_base=None)
def main(config):
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type='cuda', mesh_shape=(world_size,), mesh_dim_names=('fsdp',))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(device_type='cuda',
                                           mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
                                           mesh_dim_names=('dp', 'sp'))
    trainer = FSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh
    )
    if config.debug:
        trainer.debug()
    else:
        trainer.fit()


if __name__ == '__main__':
    main()
