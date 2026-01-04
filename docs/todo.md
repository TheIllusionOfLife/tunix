# Tunix Zero-Cost Challenge: Master TODO

This file orchestrates the execution plan across all specific documentation.

## 📚 Documentation Index
*   **Strategy**: [Scoring Breakdown](scoring_breakdown.md) | [Advanced Strategies](advanced_strategies.md)
*   **Execution**: [Kaggle Memo](kaggle_memo.md) | [TPU Resource Plan](tpu_resource_plan.md)
*   **Submission**: [Unrestricted Guide](unrestricted_mode_guide.md) | [Writeup Content](writeup_content.md) | [Video Script](video_script.md)

---

## 🗂️ Pre-Run: Dataset Setup

- [ ] **Delete old `tunix-public-data`** on Kaggle (if exists).
- [ ] **Create new `tunix-public-data`** dataset with:
    - `sft_magpie.jsonl`
    - `sft_ultrafeedback.jsonl`
    - `grpo_gsm8k_train.jsonl`
    - `grpo_mbpp_train.jsonl`
    - ⚠️ **DO NOT include** `private_hard_reasoning.jsonl`
- [ ] **Create `tunix-private-hard-reasoning`** dataset (Private) with:
    - `private_hard_reasoning.jsonl`

---

## ✅ Phase 1: Main Track Submission (45 pts)

- [ ] **Run `tunix_zero_cost_train.ipynb`** on Kaggle TPU.
- [ ] **Verify Output**: Check `<reasoning>` tags in generated samples.
- [ ] **Save Checkpoint**: Download `final_submission_model/` from Output.
- [ ] **Upload as Dataset**: Name it `tunix-session1-checkpoint` (for Unrestricted Mode).

---

## 🚀 Phase 3: Unrestricted Mode (15 bonus pts)

**Notebook**: `tunix_continuation.ipynb`

### Session 2:
- [ ] Update config: `PREV_CHECKPOINT_DATASET = "/kaggle/input/tunix-session1-checkpoint/checkpoint"`
- [ ] Run notebook on Kaggle TPU.
- [ ] Download output → Upload as `tunix-session2-checkpoint`.

### Session 3 (if needed):
- [ ] Update config: `PREV_CHECKPOINT_DATASET = "/kaggle/input/tunix-session2-checkpoint/checkpoint"`
- [ ] Run notebook.
- [ ] Download output → Upload as **Kaggle Model**: `yuyamukai/tunix-gemma2-2b-unrestricted`

### Final Step:
- [ ] Update `unrestricted_kaggle_model` in `tunix_zero_cost_train.ipynb` with final Model ID.

---

## 🎬 Phase 4: Final Submission Bundle

- [ ] **Video**: Record using `docs/video_script.md` (< 3 min).
- [ ] **Writeup**: Submit on Kaggle using `docs/writeup_content.md`.
- [ ] **Attach**: Notebook + Video to Writeup.
