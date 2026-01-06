# Kaggle Submission Memo (SFT Strategy)

## 1. Prepare Data

### Kaggle Datasets to Create

**Dataset 1: `tunix-sft-data`** (Single Session)

Upload these pre-sampled parquet files from `data/`:
- `raiden_deepseek_r1.parquet` (62.9K)
- `openo1_sft_english_20k.parquet` (20K)
- `cot_collection_10k.parquet` (10K)
- `glaiveai_30k.parquet` (30K)

**Dataset 2: `tunix-sft-continuation-data`** (Unrestricted)

Upload:
- `glaiveai_continuation_100k.parquet` (100K)

### Data Sources Documentation

Include `DATA_SOURCES.md` (already in `data/` folder):

| Dataset | Full Size | Included | Method |
|:---|---:|---:|:---|
| Raiden-DeepSeek-R1 | 62,925 | 62,925 | Full |
| OpenO1-SFT | 77,685 | 20,000 | English filter + Random |
| CoT-Collection | 1,837,928 | 10,000 | Reservoir |
| GlaiveAI | 22,200,000+ | 30,000 | First N |

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
2. Load pre-sampled parquet files
3. Initialize Gemma 2B-IT with LoRA
4. Run SFT for ~3000 steps
5. Save final checkpoint

---

## 4. Pre-flight Checklist

Before submitting:
- [ ] Run `python scripts/smoke_test_notebook.py` (must PASS)
- [ ] Verify parquet files are in your Kaggle dataset
- [ ] Check DATA_SOURCES.md is included
- [ ] Test on CPU runtime first (quick syntax check)

---

## 5. Submission

1. **Save Version**: "Save & Run All (Commit)"
2. **Verify**: Check output has checkpoint files
3. **Submit**: Link notebook on competition page

---

## Debugging Tips

- **Smoke test locally**: `python scripts/smoke_test_notebook.py`
- **Debug mode**: Set `MAX_STEPS=100` to quick test
- **Memory issues**: Reduce batch size or sequence length
- **Data errors**: Check preprocessing logs carefully
- **Thrift errors**: Pre-sampled parquets should eliminate this
