# Unrestricted Mode Guide (SFT Strategy)

This guide explains how to extend training across multiple sessions for the 15 bonus points.

---

## Overview

| Session | Method | Data | Output |
|:---|:---|:---|:---|
| 1 | SFT | Core datasets (~100K) | Checkpoint |
| 2 | SFT | glaiveai extended (~100K more) | Checkpoint |
| 3 | Optional GRPO | Subset with answers | Final Model |

---

## Session 1: Single Session Submission

**Notebook**: `tunix_sft_train.ipynb`

1. Attach Kaggle dataset with raw data
2. Run preprocessing + SFT training
3. Save checkpoint to output
4. Upload output as Kaggle Dataset: `tunix-session1-checkpoint`

---

## Session 2: Extended SFT

**Notebook**: `tunix_sft_continuation.ipynb`

1. Attach datasets:
   - `tunix-session1-checkpoint`
   - `tunix-glaiveai-data` (or download in notebook)

2. Load checkpoint from session 1
3. Continue SFT on glaiveai samples
4. Save new checkpoint

```python
# Load previous checkpoint
checkpointer.restore(PREV_CHECKPOINT_PATH, target=model_state)

# Continue SFT on new data
trainer.train(glaiveai_iterator)
```

---

## Session 3: Optional Polish

Two options:
- **More SFT**: Continue with specialized data
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

## Data for Each Session

| Session | Dataset | Samples |
|:---|:---|:---:|
| 1 | Raiden + OpenO1 + General + CoT | ~100K |
| 2 | glaiveai/reasoning-v1-20m | 100K-500K |
| 3 | (Depends on session 2 results) | Variable |
