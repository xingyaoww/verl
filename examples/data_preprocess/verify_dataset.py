"""
Verify the format of the multi-turn dataset
"""

import os
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/tmp/multiturn_test')
    args = parser.parse_args()
    
    # Load the parquet files
    train_path = os.path.join(args.local_dir, 'train.parquet')
    test_path = os.path.join(args.local_dir, 'test.parquet')
    
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Print statistics
    print(f"Train dataset size: {len(train_df)}")
    print(f"Test dataset size: {len(test_df)}")
    
    # Check the structure of the first conversation
    first_conversation = train_df['messages'][0]
    print("\nFirst conversation structure:")
    for i, message in enumerate(first_conversation):
        print(f"Message {i+1}:")
        print(f"  Role: {message['role']}")
        print(f"  Content: {message['content'][:50]}...")
    
    # Verify that the dataset format is compatible with the MultiTurnSFTDataset
    print("\nVerifying dataset format compatibility...")
    
    # Check if 'messages' column exists
    if 'messages' in train_df.columns:
        print("✓ 'messages' column exists")
    else:
        print("✗ 'messages' column is missing")
    
    # Check if messages have 'role' and 'content' fields
    first_message = train_df['messages'][0][0]
    if 'role' in first_message and 'content' in first_message:
        print("✓ Messages have 'role' and 'content' fields")
    else:
        print("✗ Messages are missing 'role' or 'content' fields")
    
    # Check if roles are valid
    roles = set()
    for conversation in train_df['messages']:
        for message in conversation:
            roles.add(message['role'])
    
    print(f"✓ Roles found: {', '.join(roles)}")
    
    if 'system' in roles and 'user' in roles and 'assistant' in roles:
        print("✓ All required roles (system, user, assistant) are present")
    else:
        print("✗ Some required roles are missing")
    
    print("\nDataset verification complete.")

if __name__ == '__main__':
    main()