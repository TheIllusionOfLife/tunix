#!/usr/bin/env python3
"""
Sample fresh GlaiveAI data for Continuation Training (Unrestricted Mode)

This script:
1. Downloads GlaiveAI reasoning-v1-20m from HuggingFace
2. SKIPS the first 30K samples (used in single session)
3. Samples 100K fresh samples from the remaining data
4. Saves as parquet for a separate Kaggle dataset

Usage:
    python scripts/sample_glaiveai_continuation.py

Output:
    data/glaiveai_continuation_100k.parquet
"""

import datasets
import random
import os

# Configuration
SKIP_FIRST = 30000  # Already used in single session
SAMPLE_SIZE = 100000
SEED = 42
OUTPUT_PATH = "data/glaiveai_continuation_100k.parquet"

def main():
    print("=" * 60)
    print("GlaiveAI Continuation Sampler")
    print("=" * 60)
    
    # Set seed for reproducibility
    random.seed(SEED)
    
    # Load dataset (streaming to avoid downloading full 87GB)
    print("\nLoading GlaiveAI reasoning-v1-20m from HuggingFace (streaming)...")
    print("This will take a while...")
    
    ds = datasets.load_dataset(
        "glaiveai/reasoning-v1-20m",
        split="train",
        streaming=True
    )
    
    # Skip first 30K and collect next samples
    print(f"\nSkipping first {SKIP_FIRST:,} samples (used in single session)...")
    print(f"Then sampling {SAMPLE_SIZE:,} fresh samples...")
    
    samples = []
    count = 0
    skipped = 0
    
    for item in ds:
        count += 1
        
        # Skip first 30K
        if skipped < SKIP_FIRST:
            skipped += 1
            if skipped % 10000 == 0:
                print(f"  Skipped {skipped:,} samples...")
            continue
        
        # Collect samples
        samples.append({
            "prompt": item.get("prompt", ""),
            "response": item.get("response", ""),
        })
        
        if len(samples) % 10000 == 0:
            print(f"  Collected {len(samples):,} samples...")
        
        if len(samples) >= SAMPLE_SIZE:
            break
    
    print(f"\nTotal processed: {count:,}")
    print(f"Skipped: {skipped:,}")
    print(f"Collected: {len(samples):,}")
    
    # Shuffle for good measure
    print("\nShuffling samples...")
    random.shuffle(samples)
    
    # Save to parquet
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
    print(f"   prompt: {sample['prompt'][:100]}...")
    print(f"   response: {sample['response'][:100]}...")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("Upload data/glaiveai_continuation_100k.parquet as a NEW Kaggle dataset.")
    print("Suggested dataset name: tunix-sft-continuation-data")
    print("=" * 60)

if __name__ == "__main__":
    main()
