# Unrestricted Mode Guide (Optional 15 Points)

This guide explains how to win the extra 15 points for "Model quality across multiple Kaggle sessions".

**Goal**: Train a stronger model by chaining sessions and using your private "Hard Reasoning" dataset.

---

## Prerequisites

| Asset | Status | Location |
|:---|:---:|:---|
| `tunix_zero_cost_train.ipynb` | ✅ | Main track notebook (Session 1) |
| `tunix_continuation.ipynb` | ✅ | Continuation notebook (Session 2+) |
| `private_hard_reasoning.jsonl` | ✅ | `data/` folder (upload separately) |

---

## Step 1: Upload Datasets to Kaggle

### Public Data (for Session 1):
**Dataset Name**: `tunix-public-data`
**Contents**:
- `sft_magpie.jsonl`
- `sft_ultrafeedback.jsonl`
- `grpo_gsm8k_train.jsonl`
- `grpo_mbpp_train.jsonl`

### Private Data (for Session 2+):
**Dataset Name**: `tunix-private-hard-reasoning`
**Contents**:
- `private_hard_reasoning.jsonl`
**Visibility**: **Private**

---

## Step 2: Session 1 - Base Training

1. Open `tunix_zero_cost_train.ipynb` on Kaggle.
2. Attach dataset: `tunix-public-data`.
3. Run the notebook (9h limit).
4. After completion, download `final_submission_model/` from Output.
5. Upload as Kaggle Dataset: `tunix-session1-checkpoint`.

---

## Step 3: Session 2 - Continuation Training

1. Open `tunix_continuation.ipynb` on Kaggle.
2. Attach datasets:
   - `tunix-session1-checkpoint`
   - `tunix-private-hard-reasoning`
3. Update config cell:
   ```python
   PREV_CHECKPOINT_DATASET = "/kaggle/input/tunix-session1-checkpoint/checkpoint"
   DATA_DATASET = "/kaggle/input/tunix-private-hard-reasoning"
   ```
4. Run the notebook.
5. Download output → Upload as `tunix-session2-checkpoint`.

---

## Step 4: Session 3+ (Optional)

Repeat Session 2 with updated checkpoint path:

```python
PREV_CHECKPOINT_DATASET = "/kaggle/input/tunix-session2-checkpoint/checkpoint"
```

---

## Step 5: Final Submission

1. After final session, **upload output as Kaggle Model**:
   - Go to Kaggle → Models → New Model
   - Upload `final_continuation_model/` contents
   - Set visibility: Public
   - Note the Model ID (e.g., `yuyamukai/tunix-gemma2-2b-unrestricted`)

2. Update `unrestricted_kaggle_model` in your submission notebook:
   ```python
   unrestricted_kaggle_model = "yuyamukai/tunix-gemma2-2b-unrestricted"
   ```

---

## Summary

| Session | Notebook | Input Data | Output |
|:---|:---|:---|:---|
| 1 | `tunix_zero_cost_train.ipynb` | Public data | `tunix-session1-checkpoint` |
| 2 | `tunix_continuation.ipynb` | Private hard data | `tunix-session2-checkpoint` |
| 3+ | `tunix_continuation.ipynb` | Private hard data | Final Kaggle Model |
