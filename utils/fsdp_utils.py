# Copyright 2024 Bytedance Ltd. and/or its affiliates

import functools
from typing import Any, Dict, Set, Type, Union

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)

def get_module_class_from_name(class_name: Union[str, Any]) -> Any:
    """
    Gets a class from a module by its name.

    Args:
        class_name (`str` or `type`):
            The name of the class to get or the class itself.

    Returns:
        `type`: The class.
    """
    if isinstance(class_name, str):
        import importlib

        class_name = class_name.lower()
        if "llama" in class_name:
            module_name = "transformers.models.llama.modeling_llama"
        elif "mistral" in class_name:
            module_name = "transformers.models.mistral.modeling_mistral"
        elif "qwen" in class_name:
            module_name = "transformers.models.qwen2.modeling_qwen2"
        else:
            raise ValueError(f"Unknown class name: {class_name}")

        module = importlib.import_module(module_name)
        class_name = class_name.split(".")[-1]
        for name, cls in module.__dict__.items():
            if name.lower() == class_name:
                return cls
    return class_name

def get_init_weight_context_manager(use_meta_tensor=True):
    if use_meta_tensor:
        return torch.device('meta')
    else:
        return torch.device('cuda')

def get_fsdp_wrap_policy(module, config=None):
    # should work for both transformers lib models and peft lib models
    if config is None:
        config = {}

    # Get the transformer layer class to wrap
    fsdp_transformer_layer_cls_to_wrap = config.get('fsdp_transformer_layer_cls_to_wrap', None)
    default_transformer_cls_names_to_wrap = ['LlamaDecoderLayer', 'MistralDecoderLayer', 'Qwen2DecoderLayer']
    if fsdp_transformer_layer_cls_to_wrap is None:
        fsdp_transformer_layer_cls_to_wrap = getattr(module, 'fsdp_transformer_layer_cls_to_wrap',
                                                    default_transformer_cls_names_to_wrap)
    min_num_params = config.get('min_num_params', 0)
    auto_wrap_policy = None

    policies = []

    # Add policy for PEFT modules
    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False
    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    policies.append(lambda_policy)

    # Add size-based policy if specified
    if min_num_params > 0:
        size_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
        policies.append(size_policy)
    # Add transformer-based policy if specified
    elif fsdp_transformer_layer_cls_to_wrap is not None:
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(layer_class)
            if transformer_cls is None:
                print(f"Could not find the transformer layer class to wrap from the name {layer_class}.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)

        transformer_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_cls_to_wrap,
        )
        policies.append(transformer_policy)

    if len(policies) > 0:
        auto_wrap_policy = functools.partial(_or_policy, policies=policies)

    return auto_wrap_policy

def offload_fsdp_grad(module):
    for _, param in module.named_parameters():
        if param.grad is not None:
            param.grad = param.grad.cpu()