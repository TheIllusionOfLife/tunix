#!/usr/bin/env python3
"""
Pre-sample OpenO1-SFT (English only) for Kaggle SFT Training

This script:
1. Downloads OpenO1-SFT from HuggingFace
2. Filters out Chinese samples
3. Randomly samples 20K English samples
4. Saves as parquet

Usage:
    python scripts/sample_openo1_english.py

Output:
    data/openo1_sft_english_20k.parquet
"""

import datasets
import random
import os
import re

# Configuration
SAMPLE_SIZE = 20000
SEED = 42
OUTPUT_PATH = "data/openo1_sft_english_20k.parquet"

def contains_chinese(text: str) -> bool:
    """Check if text contains any Chinese characters."""
    if not text:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def main():
    print("=" * 60)
    print("OpenO1-SFT English Sampler")
    print("=" * 60)
    
    # Set seed for reproducibility
    random.seed(SEED)
    
    # Load full dataset
    print("\nLoading OpenO1-SFT from HuggingFace...")
    ds = datasets.load_dataset("O1-OPEN/OpenO1-SFT", split="train")
    print(f"Total samples in dataset: {len(ds):,}")
    
    # Filter English only
    print("\nFiltering English-only samples...")
    english_samples = []
    chinese_count = 0
    
    for i, sample in enumerate(ds):
        instruction = sample.get("instruction", "")
        output = sample.get("output", "")
        
        if contains_chinese(instruction) or contains_chinese(output):
            chinese_count += 1
        else:
            english_samples.append(sample)
        
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,} samples...")
    
    print(f"\nFiltering results:")
    print(f"  Chinese samples removed: {chinese_count:,}")
    print(f"  English samples remaining: {len(english_samples):,}")
    
    # Random sample
    if len(english_samples) > SAMPLE_SIZE:
        print(f"\nRandomly sampling {SAMPLE_SIZE:,} from {len(english_samples):,} English samples...")
        random.shuffle(english_samples)
        english_samples = english_samples[:SAMPLE_SIZE]
    else:
        print(f"\nUsing all {len(english_samples):,} English samples (less than target {SAMPLE_SIZE:,})")
    
    # Save to parquet
    print(f"\nSaving to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    sampled_ds = datasets.Dataset.from_list(english_samples)
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
    print(f"   instruction: {sample['instruction'][:100]}...")
    print(f"   output: {sample['output'][:100]}...")
    
    print("\n" + "=" * 60)
    print("Done! Upload data/openo1_sft_english_20k.parquet to your Kaggle dataset.")
    print("=" * 60)

if __name__ == "__main__":
    main()
