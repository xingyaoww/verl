"""
Test the MultiTurnSFTDataset implementation
"""
import os
import pandas as pd
import torch
from transformers import AutoTokenizer
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


def test_multiturn_sft_dataset():
    # Create a temporary parquet file with test data
    test_data = {
        'messages': [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
                {"role": "user", "content": "And what is 4+4?"},
                {"role": "assistant", "content": "4+4 equals 8."}
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "Why did the chicken cross the road?"},
                {"role": "user", "content": "Why?"},
                {"role": "assistant", "content": "To get to the other side!"}
            ]
        ]
    }
    
    # Create test directory if it doesn't exist
    os.makedirs('test_data', exist_ok=True)
    test_file = 'test_data/test.parquet'
    
    # Save test data to parquet
    df = pd.DataFrame(test_data)
    df.to_parquet(test_file)
    
    # Initialize tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct')
    dataset = MultiTurnSFTDataset(
        parquet_files=test_file,
        tokenizer=tokenizer,
        max_length=512
    )
    
    # Test dataset length
    assert len(dataset) == 2, f"Expected dataset length 2, got {len(dataset)}"
    
    # Get first item
    item = dataset[0]
    
    # Check that all required keys are present
    required_keys = ['input_ids', 'attention_mask', 'position_ids', 'loss_mask']
    for key in required_keys:
        assert key in item, f"Missing key {key} in dataset item"
        assert isinstance(item[key], torch.Tensor), f"Expected torch.Tensor for {key}"
    
    # Verify loss mask shape matches input_ids
    assert item['loss_mask'].shape == item['input_ids'].shape, \
        "Loss mask shape doesn't match input_ids shape"
    
    # Decode the tokens where loss_mask is 1 to verify they correspond to assistant messages
    loss_mask = item['loss_mask']
    input_ids = item['input_ids']
    
    # Get positions where loss_mask is 1
    assistant_positions = torch.where(loss_mask == 1)[0]
    
    # Verify that we have assistant positions with loss_mask=1
    assert len(assistant_positions) > 0, "No positions found with loss_mask=1"
    
    # Get all text from positions where loss_mask=1
    assistant_text = tokenizer.decode(input_ids[loss_mask == 1])
    print(f"Assistant text: {assistant_text}")
    
    # Verify it contains our expected assistant responses
    assert any(x in assistant_text.lower() for x in ['equals', 'get to the other side']), \
        f"Expected assistant response content, got: {assistant_text}"
    
    print("All tests passed!")


if __name__ == "__main__":
    test_multiturn_sft_dataset()