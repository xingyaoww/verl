"""
Example script for preprocessing multi-turn conversation data into parquet format
"""

import os
import argparse
import datasets
from verl.utils.hdfs_io import copy, makedirs


def process_conversation(example, idx, split):
    """Convert a conversation into the expected format"""
    messages = []
    
    # Add system message if present
    if example.get('system', ''):
        messages.append({
            "role": "system",
            "content": example['system']
        })
    
    # Add conversation turns
    for turn in example['conversation']:
        messages.append({
            "role": turn['role'],
            "content": turn['content']
        })
    
    # Return the processed data
    return {
        "data_source": "multiturn_example",
        "messages": messages,
        "extra_info": {
            'split': split,
            'index': idx,
            'original': example
        }
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/multiturn')
    parser.add_argument('--hdfs_dir', default=None)
    
    args = parser.parse_args()
    
    # Load your dataset here
    # This is just an example - replace with your actual data loading
    dataset = datasets.load_dataset('your_dataset_name')
    
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    # Process the datasets
    train_dataset = train_dataset.map(
        function=lambda x, i: process_conversation(x, i, 'train'),
        with_indices=True
    )
    test_dataset = test_dataset.map(
        function=lambda x, i: process_conversation(x, i, 'test'),
        with_indices=True
    )
    
    # Save to parquet files
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)