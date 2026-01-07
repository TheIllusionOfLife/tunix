#!/usr/bin/env python3
"""
Pre-sample CoT-Collection for Kaggle SFT Training

This script downloads CoT-Collection from HuggingFace, samples 10K rows,
and saves as a parquet file for efficient loading in the notebook.

Usage:
    python scripts/sample_cot_collection.py

Output:
    data/cot_collection_10k.parquet
"""

import datasets
import random
import os

# Configuration
SAMPLE_SIZE = 10000
SEED = 42
OUTPUT_PATH = "data/cot_collection_10k.parquet"

def main():
    print("=" * 60)
    print("CoT-Collection Pre-Sampler")
    print("=" * 60)
    
    # Set seed for reproducibility
    random.seed(SEED)
    
    # Load dataset (streaming to avoid downloading full 3.5GB at once)
    print("\nLoading CoT-Collection from HuggingFace (streaming)...")
    print("This may take a few minutes...")
    
    try:
        # Stream the dataset to avoid memory issues
        ds = datasets.load_dataset(
            "pharaouk/CoT-Collection",
            split="train",
            streaming=True
        )
    except Exception as e:
        print(f"Streaming failed: {e}")
        print("Trying full download...")
        ds = datasets.load_dataset("pharaouk/CoT-Collection", split="train")
    
    # Collect samples using reservoir sampling for streaming
    print(f"\nSampling {SAMPLE_SIZE} rows using reservoir sampling...")
    
    samples = []
    count = 0
    
    for item in ds:
        count += 1
        
        if len(samples) < SAMPLE_SIZE:
            samples.append(item)
        else:
            # Reservoir sampling: replace with probability SAMPLE_SIZE/count
            j = random.randint(0, count - 1)
            if j < SAMPLE_SIZE:
                samples[j] = item
        
        if count % 100000 == 0:
            print(f"  Processed {count:,} samples...")
    
    print(f"\nTotal samples in dataset: {count:,}")
    print(f"Sampled: {len(samples)} rows")
    
    # Convert to Dataset and save
    print(f"\nSaving to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    sampled_ds = datasets.Dataset.from_list(samples)
    sampled_ds.to_parquet(OUTPUT_PATH)
    
    # Verify
    file_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\n✅ Saved successfully!")
    print(f"   File size: {file_size:.1f} MB")
    print(f"   Rows: {len(sampled_ds)}")
    print(f"   Columns: {sampled_ds.column_names}")
    
    # Show sample
    print("\n📋 Sample row:")
    sample = sampled_ds[0]
    print(f"   source: {sample['source'][:100]}...")
    print(f"   rationale: {sample['rationale'][:100]}...")
    print(f"   target: {sample['target'][:100]}...")
    print(f"   task: {sample.get('task', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("Done! Upload data/cot_collection_10k.parquet to your Kaggle dataset.")
    print("=" * 60)

if __name__ == "__main__":
    main()
