#!/usr/bin/env python3
"""
Smoke Test for GlaiveAI-Only Dataset
Tests that the parquet files load correctly and format properly.
"""

import os
import re
import datasets

# Mock Constants
SYSTEM_PROMPT = "You are a deep thinking AI. Think step by step about the problem and provide your reasoning between <reasoning> and </reasoning> tags. Then, provide the final answer between <answer> and </answer> tags."

# Use local pre-sampled parquet files
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def standardize_glaive_format(prompt, response):
    """Convert GlaiveAI <think> format to <reasoning>/<answer> tags"""
    text = response
    
    # Replace think tags with reasoning tags
    text = re.sub(r"<think>", "<reasoning>", text, flags=re.IGNORECASE)
    text = re.sub(r"</think>", "</reasoning>", text, flags=re.IGNORECASE)
    
    # Extract reasoning and answer parts
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL | re.IGNORECASE)
    
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
        # Get content after </reasoning> as answer
        remaining = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        
        if remaining:
            answer = remaining
        else:
            # No content after reasoning - use summary
            sentences = reasoning.split(".")
            answer = sentences[-1].strip() if sentences else reasoning[:200]
    else:
        # No think tags - use whole response
        reasoning = text[:500] if len(text) > 500 else text
        answer = text
    
    # Format for Gemma
    formatted = f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n<reasoning>{reasoning}</reasoning>\n<answer>{answer}</answer>"
    return formatted


print("=" * 60)
print("Smoke Test: GlaiveAI-Only Dataset")
print("=" * 60)

all_texts = []
files_found = 0

try:
    # Test Part 1
    print("\n1. Testing glaiveai_90k_part1.parquet...")
    part1_path = os.path.join(DATA_DIR, "glaiveai_90k_part1.parquet")
    if os.path.exists(part1_path):
        files_found += 1
        ds = datasets.load_dataset("parquet", data_files=part1_path, split="train[:50]")
        before_count = len(all_texts)
        for sample in ds:
            prompt = sample.get("prompt", "")
            response = sample.get("response", "")
            if len(response) > 4000 or len(response) < 50:
                continue
            formatted = standardize_glaive_format(prompt, response)
            all_texts.append({"text": formatted})
        print(f"   ✅ Part 1: Loaded {len(all_texts) - before_count} samples")
        
        # Check file size
        size_mb = os.path.getsize(part1_path) / (1024 * 1024)
        print(f"   📁 File size: {size_mb:.1f} MB")
    else:
        print(f"   ❌ Part 1 NOT FOUND: {part1_path}")

    # Test Part 2
    print("\n2. Testing glaiveai_90k_part2.parquet...")
    part2_path = os.path.join(DATA_DIR, "glaiveai_90k_part2.parquet")
    if os.path.exists(part2_path):
        files_found += 1
        ds = datasets.load_dataset("parquet", data_files=part2_path, split="train[:50]")
        before_count = len(all_texts)
        for sample in ds:
            prompt = sample.get("prompt", "")
            response = sample.get("response", "")
            if len(response) > 4000 or len(response) < 50:
                continue
            formatted = standardize_glaive_format(prompt, response)
            all_texts.append({"text": formatted})
        print(f"   ✅ Part 2: Loaded {len(all_texts) - before_count} samples")
        
        # Check file size
        size_mb = os.path.getsize(part2_path) / (1024 * 1024)
        print(f"   📁 File size: {size_mb:.1f} MB")
    else:
        print(f"   ❌ Part 2 NOT FOUND: {part2_path}")

    # Test Continuation (optional)
    print("\n3. Testing glaiveai_continuation_100k.parquet (optional)...")
    cont_path = os.path.join(DATA_DIR, "glaiveai_continuation_100k.parquet")
    if os.path.exists(cont_path):
        ds = datasets.load_dataset("parquet", data_files=cont_path, split="train[:10]")
        print(f"   ✅ Continuation: {len(ds)} test samples loaded")
        size_mb = os.path.getsize(cont_path) / (1024 * 1024)
        print(f"   📁 File size: {size_mb:.1f} MB")
    else:
        print(f"   ⚠️ Continuation not found (optional for unrestricted mode)")

    # Summary
    print("\n" + "=" * 60)
    if files_found >= 2:
        print(f"✅ SMOKE TEST PASSED")
        print(f"   Files: {files_found}/2 required files found")
        print(f"   Samples tested: {len(all_texts)}")
        
        # Show sample
        if all_texts:
            print(f"\n📋 Sample formatted text:")
            print(all_texts[0]["text"][:500] + "...")
    else:
        print(f"❌ SMOKE TEST FAILED")
        print(f"   Missing files: {2 - files_found} required files not found")
        print(f"   Run: python scripts/download_sft_datasets.py")
        exit(1)
    print("=" * 60)

except Exception as e:
    print(f"\n❌ Smoke Test Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
