"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer


class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer,
                 messages_key='messages',  # Key for the messages list in the parquet file
                 max_length=1024,
                 truncation='error'):
        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.messages_key = messages_key
        self.max_length = max_length

        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        def series_to_item(ls):
            import pandas, numpy
            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        
        # Extract messages list from dataframe
        self.messages = self.dataframe[self.messages_key].apply(series_to_item).tolist()

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        messages = self.messages[item]

        # Process each message and concatenate with special tokens
        all_tokens = []
        loss_mask_parts = []
        
        for msg in messages:
            # Add role prefix
            if msg['role'] == 'system':
                prefix = "<|system|>\n"
            elif msg['role'] == 'user':
                prefix = "<|user|>\n"
            elif msg['role'] == 'assistant':
                prefix = "<|assistant|>\n"
            else:
                raise ValueError(f"Unknown role: {msg['role']}")
            
            # Tokenize the message
            msg_str = prefix + msg['content'] + "\n"
            msg_tokens = tokenizer(msg_str, return_tensors='pt', add_special_tokens=False)
            msg_ids = msg_tokens['input_ids'][0]
            
            # Create loss mask for this message (1 for assistant responses, 0 for others)
            msg_mask = torch.zeros_like(msg_ids, dtype=torch.long)
            if msg['role'] == 'assistant':
                msg_mask = torch.ones_like(msg_ids, dtype=torch.long)
            
            all_tokens.append(msg_ids)
            loss_mask_parts.append(msg_mask)
        
        # Concatenate all tokens and masks
        input_ids = torch.cat(all_tokens)
        loss_mask = torch.cat(loss_mask_parts)
        attention_mask = torch.ones_like(input_ids)

        # Handle sequence length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            # Pad sequences
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                        dtype=input_ids.dtype) * pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), 
                                              dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros(size=(self.max_length - sequence_length,), 
                                         dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))
        elif sequence_length > self.max_length:
            if self.truncation == 'left':
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]
            elif self.truncation == 'right':
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
            elif self.truncation == 'error':
                raise ValueError(f'{sequence_length=} is larger than {self.max_length=}')
            else:
                raise ValueError(f'Unknown truncation method {self.truncation}')

        position_ids = compute_position_id_with_mask(attention_mask)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask
        }