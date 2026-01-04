# Tunix Zero-Cost Challenge: Master TODO

This file orchestrates the execution plan across all specific documentation.

## 📚 Documentation Index
*   **Strategy**: [Scoring Breakdown](scoring_breakdown.md) | [Advanced Strategies](advanced_strategies.md)
*   **Execution**: [Kaggle Memo](kaggle_memo.md) | [TPU Resource Plan](tpu_resource_plan.md)
*   **Submission**: [Unrestricted Guide](unrestricted_mode_guide.md) | [Writeup Content](writeup_content.md) | [Video Script](video_script.md)

---

## ✅ Code Quality Fixes (Completed)

- [x] Fixed `algo_config` parameter in GRPOLearner (v0.1.5 API)
- [x] Fixed import typo `rollouts` → `rollout`
- [x] Increased GRPO_STEPS from 600 to 1500
- [x] Fixed TEMPLATE definition order (now before baseline eval)
- [x] Updated dataset description to match reality (GSM8K/MBPP only)
- [x] Removed unused `SFT_STEPS` and `PROMPT_TEMPLATE`
- [x] Updated Learnings section (no-SFT approach)
- [x] Changed fallback to `raise RuntimeError` (fail-fast)
- [x] Fixed `code_correctness_reward` to handle `<answer>` tags

---

## 🗂️ Pre-Run: Dataset Setup

- [x] **Delete old `tunix-public-data`** on Kaggle.
- [x] **Create new `tunix-public-data`** dataset with:
    - `sft_magpie.jsonl` (for potential future SFT)
    - `sft_ultrafeedback.jsonl` (for potential future SFT)
    - `grpo_gsm8k_train.jsonl` ← **Used by GRPO**
    - `grpo_mbpp_train.jsonl` ← **Used by GRPO**
- [ ] **Create `tunix-private-hard-reasoning`** dataset (Private) with:
    - `private_hard_reasoning.jsonl`

---

## ✅ Phase 1: Main Track Submission (45 pts)

- [ ] **Run `tunix_zero_cost_train.ipynb`** on Kaggle TPU v5e-8
- [ ] **Verify Output**: Check `<reasoning>` tags in generated samples (~99% expected)
- [ ] **Save Checkpoint**: Download `final_submission_model/` from Output
- [ ] **Upload as Dataset**: Name it `tunix-session1-checkpoint`

---

## 🚀 Phase 2: Unrestricted Mode (15 bonus pts)

**Notebook**: `tunix_continuation.ipynb`

### Session 2:
- [ ] Attach datasets: `tunix-session1-checkpoint` + `tunix-private-hard-reasoning`
- [ ] Update config: `PREV_CHECKPOINT_DATASET = "/kaggle/input/tunix-session1-checkpoint/checkpoint"`
- [ ] Run notebook on Kaggle TPU
- [ ] Download output → Upload as `tunix-session2-checkpoint`

### Session 3 (if needed):
- [ ] Update config: `PREV_CHECKPOINT_DATASET = "/kaggle/input/tunix-session2-checkpoint/checkpoint"`
- [ ] Run notebook
- [ ] Download output → Upload as **Kaggle Model** with path `/jax/size/`

### Final Step:
- [ ] Update `unrestricted_kaggle_model` in submission notebook with Model ID

**Note**: When uploading to Kaggle Models, create folder structure `jax/size/` and place checkpoint files inside.

---

## 🎬 Phase 3: Final Submission Bundle

- [ ] **Video**: Record using `docs/video_script.md` (< 3 min)
- [ ] **Writeup**: Submit on Kaggle using `docs/writeup_content.md`
- [ ] **Attach**: Notebook + Video to Writeup
