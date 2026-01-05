#!/usr/bin/env python3
"""
Download SFT datasets from HuggingFace for Tunix competition.
Downloads RAW data without preprocessing - formatting done in notebook.

Datasets:
1. Raiden-DeepSeek-R1 (62.9K creative/analytical)
2. OpenO1-SFT (77.7K general reasoning)
3. General_Inquiry_Thinking (6K philosophical) which turns out to be gated, so we can't use this.
4. CoT-Collection (sampled, commonsense/ethics)
5. glaiveai/reasoning-v1-20m (sampled for extended training)
"""

import os
from datasets import load_dataset

OUTPUT_DIR = "data/sft_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_dataset(name, hf_path, split="train", max_samples=None, output_name=None):
    """Download dataset from HuggingFace and save as parquet/jsonl"""
    print(f"\n{'='*60}")
    print(f"Downloading: {hf_path}")
    print(f"{'='*60}")
    
    try:
        if max_samples:
            ds = load_dataset(hf_path, split=f"{split}[:{max_samples}]")
        else:
            ds = load_dataset(hf_path, split=split)
        
        print(f"Downloaded {len(ds)} samples")
        print(f"Columns: {ds.column_names}")
        
        # Save as parquet (efficient) and jsonl (readable)
        base_name = output_name or name
        parquet_path = os.path.join(OUTPUT_DIR, f"{base_name}.parquet")
        jsonl_path = os.path.join(OUTPUT_DIR, f"{base_name}.jsonl")
        
        ds.to_parquet(parquet_path)
        ds.to_json(jsonl_path)
        
        print(f"Saved to: {parquet_path}")
        print(f"Saved to: {jsonl_path}")
        
        # Show sample
        print(f"\nSample (first row):")
        print(ds[0])
        
        return len(ds)
        
    except Exception as e:
        print(f"FAILED: {e}")
        return 0


def main():
    print("="*60)
    print("SFT Dataset Downloader - RAW (No preprocessing)")
    print("="*60)
    
    total = 0
    
    # 1. Raiden-DeepSeek-R1 (Creative/Analytical) - Full dataset
    total += download_dataset(
        "raiden_deepseek_r1",
        "sequelbox/Raiden-DeepSeek-R1"
    )
    
    # 2. OpenO1-SFT (General reasoning) - Sample 20K
    total += download_dataset(
        "openo1_sft",
        "O1-OPEN/OpenO1-SFT",
        max_samples=20000
    )
    
    # 3. General_Inquiry_Thinking (Philosophical) - Full dataset
    total += download_dataset(
        "general_inquiry_thinking",
        "moremilk/General_Inquiry_Thinking-Chain-Of-Thought"
    )
    
    # 4. CoT-Collection (Commonsense/Ethics) - Sample 10K
    total += download_dataset(
        "cot_collection",
        "pharaouk/CoT-Collection",
        max_samples=10000
    )
    
    # 5. glaiveai/reasoning-v1-20m (Extended training) - Sample 50K for now
    total += download_dataset(
        "glaiveai_reasoning",
        "glaiveai/reasoning-v1-20m",
        max_samples=50000
    )
    
    print("\n" + "="*60)
    print(f"TOTAL SAMPLES DOWNLOADED: {total}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)
    
    # List output files
    print("\nFiles created:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        path = os.path.join(OUTPUT_DIR, f)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {f}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
