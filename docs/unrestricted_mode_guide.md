# Unrestricted Mode Guide (SFT Strategy)

This guide explains how to extend training across multiple sessions for the 15 bonus points.

---

## Overview

| Session | Method | Data | Output |
|:---|:---|:---|:---|
| 1 | SFT | Core datasets (~123K) | Checkpoint |
| 2 | SFT | glaiveai continuation (100K fresh) | Checkpoint |
| 3 | Optional polish | If time permits | Final Model |

---

## Kaggle Dataset Structure

### Dataset 1: `tunix-sft-data` (Single Session)

| File | Samples | Source |
|:---|---:|:---|
| `raiden_deepseek_r1.parquet` | 62,925 | Full dataset |
| `openo1_sft_english_20k.parquet` | 20,000 | English-only, random |
| `cot_collection_10k.parquet` | 10,000 | Reservoir sampled |
| `glaiveai_30k.parquet` | 30,000 | First N |
| **Total** | **~123K** | |

### Dataset 2: `tunix-sft-continuation-data` (Unrestricted)

| File | Samples | Source |
|:---|---:|:---|
| `glaiveai_continuation_100k.parquet` | 100,000 | Samples 30,001-130,000 |

> **Important**: Continuation data does NOT overlap with session 1.

---

## Session 1: Single Session Submission

**Notebook**: `tunix_sft_train.ipynb`

1. Attach Kaggle dataset: `tunix-sft-data`
2. Run SFT training (~123K samples)
3. Save checkpoint to output
4. Upload output as Kaggle Dataset: `tunix-session1-checkpoint`

---

## Session 2: Extended SFT

**Notebook**: `tunix_sft_continuation.ipynb`

1. Attach datasets:
   - `tunix-session1-checkpoint` (previous checkpoint)
   - `tunix-sft-continuation-data` (100K fresh GlaiveAI)

2. Load checkpoint from session 1
3. Continue SFT on fresh 100K samples
4. Save new checkpoint

```python
# Load previous checkpoint
prev_checkpointer.restore(PREV_CHECKPOINT_PATH, target=abs_lora_params)
nnx.update(lora_model, restored_lora_params)

# Continue SFT on new data (loaded from parquet)
```

---

## Session 3: Optional Polish

Options:
- **More SFT**: Sample another 100K from remaining GlaiveAI (22M total)
- **GRPO**: Light RL polish on verifiable tasks (if time permits)

---

## Final Upload

1. Download final checkpoint
2. Create Kaggle Model:
   - Structure: `jax/size/` folder containing checkpoint files
   - Visibility: Public
3. Update notebook with Model ID:
   ```python
   unrestricted_kaggle_model = "username/tunix-gemma2-sft"
   ```

---

## Summary

| Session | Dataset | Samples | Cumulative |
|:---|:---|:---:|:---:|
| 1 | tunix-sft-data | ~123K | 123K |
| 2 | tunix-sft-continuation-data | 100K | 223K |
| 3 | (Optional) | Variable | 223K+ |
