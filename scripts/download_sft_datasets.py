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

# Split into two files to prevent OOM on Kaggle
TOTAL_SAMPLES = 180000
SAMPLES_PER_FILE = 90000
SEED = 42


def download_glaiveai(total_samples: int, samples_per_file: int):
    """Download GlaiveAI samples via streaming and split into files"""
    print("=" * 60)
    print(f"Downloading GlaiveAI reasoning-v1-20m ({total_samples:,} samples)")
    print(f"Splitting into {total_samples // samples_per_file} files of {samples_per_file:,} each")
    print("=" * 60)
    
    ds = load_dataset(
        "glaiveai/reasoning-v1-20m",
        split="train",
        streaming=True
    )
    
    samples = []
    for item in tqdm(ds, total=total_samples, desc="Downloading"):
        samples.append({
            "prompt": item.get("prompt", ""),
            "response": item.get("response", ""),
        })
        
        if len(samples) >= total_samples:
            break
    
    print(f"\nCollected {len(samples):,} samples")
    
    # Split and save
    from datasets import Dataset
    
    num_files = total_samples // samples_per_file
    for i in range(num_files):
        start = i * samples_per_file
        end = start + samples_per_file
        part_samples = samples[start:end]
        
        output_path = os.path.join(OUTPUT_DIR, f"glaiveai_90k_part{i+1}.parquet")
        ds_part = Dataset.from_list(part_samples)
        ds_part.to_parquet(output_path)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Part {i+1}: {output_path} ({file_size:.1f} MB, {len(ds_part)} rows)")
    
    # Show sample
    print("\n📋 Sample row:")
    sample = samples[0]
    print(f"   prompt: {sample['prompt'][:100]}...")
    print(f"   response: {sample['response'][:100]}...")
    
    return len(samples)


def main():
    print("=" * 60)
    print("GlaiveAI-Only SFT Dataset Downloader (Split Mode)")
    print("=" * 60)
    print("\nStrategy: Using only GlaiveAI dataset (others dropped)")
    print("Reason: 2025 quality, non-math/code focus, consistent format")
    print("Note: Split into 2 files to prevent OOM on Kaggle\n")
    
    total = download_glaiveai(TOTAL_SAMPLES, SAMPLES_PER_FILE)
    
    print("\n" + "=" * 60)
    print(f"TOTAL SAMPLES: {total:,}")
    print("=" * 60)
    print("\nUpload to Kaggle dataset: tunix-sft-data")
    print("  - data/glaiveai_90k_part1.parquet")
    print("  - data/glaiveai_90k_part2.parquet")


if __name__ == "__main__":
    main()
