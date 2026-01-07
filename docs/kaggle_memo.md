# Kaggle Submission Memo (GlaiveAI-Only Strategy)

## 1. Prepare Data

### Kaggle Dataset: `tunix-sft-data`

Upload from `data/`:
- `glaiveai_180k.parquet` (180K samples)

### Data Source

| Dataset | Samples | Slice |
|---------|---------|-------|
| GlaiveAI | 180,000 | `train[:180000]` |

---

## 2. Prepare Notebook

### Single Session
1. Upload `tunix_sft_train.ipynb`
2. Configure accelerator: **TPU VM v5e-8**
3. Attach: `tunix-sft-data`
4. Enable persistence for checkpoints

### Unrestricted Mode
1. Upload `tunix_sft_continuation.ipynb`
2. Configure accelerator: **TPU VM v5e-8**
3. Attach:
   - `tunix-session1-checkpoint` (output from session 1)
   - `tunix-sft-continuation-data`

---

## 3. Run Training

Notebook will:
1. Install Tunix and dependencies
2. Load GlaiveAI parquet file
3. Initialize Gemma 2B-IT with LoRA
4. Run SFT for ~22,500 steps (~7 hours)
5. Save final checkpoint

---

## 4. Pre-flight Checklist

Before submitting:
- [ ] Run `python scripts/smoke_test_notebook.py` (must PASS)
- [ ] Verify `glaiveai_180k.parquet` is in Kaggle dataset
- [ ] Check DATA_SOURCES.md is included
- [ ] Test on CPU runtime first (quick syntax check)

---

## 5. Submission

1. **Save Version**: "Save & Run All (Commit)"
2. **Verify**: Check output has checkpoint files
3. **Submit**: Link notebook on competition page

---

## Debugging Tips

- **Smoke test locally**: `python tests/smoke_test_notebook.py`
- **Debug mode**: Set `SFT_STEPS=100` for quick test
- **Memory issues**: Reduce batch size or `EVAL_MAX_TOKENS`
- **Data errors**: Check preprocessing logs carefully
