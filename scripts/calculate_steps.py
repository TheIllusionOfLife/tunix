import os
import glob
import datasets
import math

DATA_DIR = "data"
parquet_files = glob.glob(os.path.join(DATA_DIR, "glaiveai_90k_part*.parquet"))

if not parquet_files:
    print("No data files found.")
    exit()

total_samples = 0
threshold_counts = {10400: 0, 12500: 0, 15000: 0, None: 0}

print("Scanning datasets (Pass 1 - Counting)...")
for pf in parquet_files:
    # Streaming load for memory efficiency in utility script too
    ds = datasets.load_dataset("parquet", data_files=pf, split="train") # Local load is usually fine, but consistency is good
    for sample in ds:
        length = len(sample.get("response", ""))
        if length < 50: continue
        
        total_samples += 1
        for t in threshold_counts:
            if t is None or length <= t:
                threshold_counts[t] += 1

print(f"Total valid samples: {total_samples}")

selected_count = 0
selected_threshold = 0

for t in [10400, 12500, 15000, None]:
    count = threshold_counts[t]
    ratio = count / total_samples
    print(f"Threshold {t}: {count} ({ratio:.1%})")
    
    if ratio >= 0.8:
        selected_count = count
        selected_threshold = t
        break
else:
    selected_count = threshold_counts[None]
    selected_threshold = "None"

print(f"\nSelected Threshold: {selected_threshold}")
print(f"Selected Count: {selected_count}")

# Formula: Steps = ceil((Samples * Epochs) / Batch)
# Epochs = 4
# Batch = 32
TARGET_EPOCHS = 4
EFFECTIVE_BATCH = 32

steps = max(1, math.ceil((selected_count * TARGET_EPOCHS) / EFFECTIVE_BATCH))
print(f"Calculated SFT_STEPS: {steps} (using math.ceil)")
