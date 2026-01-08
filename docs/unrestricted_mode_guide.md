# Unrestricted Mode Guide (GlaiveAI-Only Strategy)

This guide explains how to extend training across multiple sessions for the 15 bonus points.

---

## Overview

| Session | Method | Data | Output |
|:---|:---|:---|:---|
| 1 | SFT | GlaiveAI (180K) | Checkpoint |
| 2 | SFT | GlaiveAI continuation (100K fresh) | Final Model |

---

## Kaggle Dataset Structure

### Dataset 1: `tunix-sft-data` (Single Session)

| File | Samples | Source |
|:---|---:|:---|
| `glaiveai_90k_part*.parquet` | ~180,000 | `train[:180000]` |

### Dataset 2: `tunix-sft-continuation-data` (Unrestricted)

| File | Samples | Source |
|:---|---:|:---|
| `glaiveai_continuation_*.parquet` | ~100,000 | `train[180000:280000]` |

> **Important**: Continuation data does NOT overlap with session 1.

---

## Session 1: Single Session Submission

**Notebook**: `tunix_sft_train.ipynb`

1. Attach Kaggle dataset: `tunix-sft-data`
2. Run SFT training (Target: 4 epochs, dynamic steps)
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
| 1 | tunix-sft-data | 180K | 180K |
| 2 | tunix-sft-continuation-data | 100K | 280K |
