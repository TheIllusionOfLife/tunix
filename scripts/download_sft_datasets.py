#!/usr/bin/env python3
"""
Download GlaiveAI dataset for Tunix SFT training.

Strategy: GlaiveAI-only (other datasets dropped due to quality/alignment issues)

Dataset:
- glaiveai/reasoning-v1-20m (180K samples for single session)

Usage:
    python scripts/download_sft_datasets.py

Output:
    data/glaiveai_180k.parquet
"""

import os
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_SIZE = 180000
SEED = 42


def download_glaiveai(sample_size: int):
    """Download GlaiveAI samples via streaming"""
    print("=" * 60)
    print(f"Downloading GlaiveAI reasoning-v1-20m ({sample_size:,} samples)")
    print("=" * 60)
    
    ds = load_dataset(
        "glaiveai/reasoning-v1-20m",
        split="train",
        streaming=True
    )
    
    samples = []
    for item in tqdm(ds, total=sample_size, desc="Downloading"):
        samples.append({
            "prompt": item.get("prompt", ""),
            "response": item.get("response", ""),
        })
        
        if len(samples) >= sample_size:
            break
    
    print(f"\nCollected {len(samples):,} samples")
    
    # Save as parquet
    output_path = os.path.join(OUTPUT_DIR, f"glaiveai_{sample_size // 1000}k.parquet")
    
    from datasets import Dataset
    ds_final = Dataset.from_list(samples)
    ds_final.to_parquet(output_path)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ Saved to {output_path}")
    print(f"   File size: {file_size:.1f} MB")
    print(f"   Rows: {len(ds_final)}")
    
    # Show sample
    print("\n📋 Sample row:")
    sample = ds_final[0]
    print(f"   prompt: {sample['prompt'][:100]}...")
    print(f"   response: {sample['response'][:100]}...")
    
    return len(samples)


def main():
    print("=" * 60)
    print("GlaiveAI-Only SFT Dataset Downloader")
    print("=" * 60)
    print("\nStrategy: Using only GlaiveAI dataset (others dropped)")
    print("Reason: 2025 quality, non-math/code focus, consistent format\n")
    
    total = download_glaiveai(SAMPLE_SIZE)
    
    print("\n" + "=" * 60)
    print(f"TOTAL SAMPLES: {total:,}")
    print("=" * 60)
    print("\nUpload data/glaiveai_180k.parquet to Kaggle dataset: tunix-sft-data")


if __name__ == "__main__":
    main()
