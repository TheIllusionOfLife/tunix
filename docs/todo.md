# Tunix SFT Strategy: Master TODO

Strategy pivot: From GRPO on math/code to **SFT on diverse domains** with reasoning traces.

## 📚 Documentation Index
- **Strategy**: [Scoring Breakdown](scoring_breakdown.md) | [Advanced Strategies](advanced_strategies.md)
- **Execution**: [Kaggle Memo](kaggle_memo.md) | [TPU Resource Plan](tpu_resource_plan.md)
- **Submission**: [Unrestricted Guide](unrestricted_mode_guide.md) | [Writeup Content](writeup_content.md) | [Video Script](video_script.md)
- **Data**: [DATA_SOURCES.md](../data/DATA_SOURCES.md)
- **Archive**: [GRPO Strategy Docs](../archive/) (backup if we revert)

---

## 🎯 Core Strategy

**Method**: Supervised Fine-Tuning (SFT) on high-quality reasoning traces
**Focus**: Non-verifiable domains (creative, analytical, philosophical, commonsense)
**Rationale**: FAQ states verifiable tasks (math/code) have "much lower weights"

---

## 📦 Datasets (Pre-sampled)

| Parquet File | Samples | Source | Method |
|:---|:---:|:---|:---|
| `raiden_deepseek_r1.parquet` | 62,925 | sequelbox/Raiden-DeepSeek-R1 | Full |
| `openo1_sft_english_20k.parquet` | 20,000 | O1-OPEN/OpenO1-SFT | English + Random |
| `cot_collection_10k.parquet` | 10,000 | pharaouk/CoT-Collection | Reservoir |
| `glaiveai_30k.parquet` | 30,000 | glaiveai/reasoning-v1-20m | First N |
| **Total (Session 1)** | **~123K** | | |
| `glaiveai_continuation_100k.parquet` | 100,000 | glaiveai/reasoning-v1-20m | Samples 30K-130K |

---

## 🗂️ Data Preparation

- [x] Create pre-sampling scripts for each dataset
- [x] Generate `cot_collection_10k.parquet` (reservoir sampling)
- [x] Generate `openo1_sft_english_20k.parquet` (English filter + random)
- [x] Generate `glaiveai_continuation_100k.parquet` (fresh 100K)
- [x] Update `DATA_SOURCES.md` with full/included/method
- [ ] Upload parquets to Kaggle dataset: `tunix-sft-data`
- [ ] Upload continuation parquet to: `tunix-sft-continuation-data`

### Format Standardization
All datasets converted to:
```
<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model
<reasoning>{trace}</reasoning>
<answer>{answer}</answer>
```

---

## ✅ Phase 1: Single Session (45 pts)

- [x] **Notebook**: Create `tunix_sft_train.ipynb`
- [x] **Data Loading**: Pre-sampled parquets (~123K samples)
- [x] **Smoke Test**: Syntax validation passing
- [x] **Upload**: Parquets to Kaggle dataset
- [x] **Training**: Run on Kaggle TPU
- [x] **Verify**: Check `<reasoning>` tags in outputs
- [x] **Save**: Checkpoint for unrestricted mode

---

## 🚀 Phase 2: Unrestricted Mode (+15 pts)

- [x] **Data**: Generate 100K fresh GlaiveAI samples
- [x] **Notebook**: Update `tunix_sft_continuation.ipynb`
- [ ] **Session 1 Output**: Upload as `tunix-session1-checkpoint`
- [ ] **Session 2**: Run continuation training
- [ ] **Session 3**: Optional polish
- [x] **Upload**: Final model to Kaggle Models

---

## 🎬 Phase 3: Final Submission

- [ ] **Video**: Record < 3 min demo
- [ ] **Writeup**: Submit with notebook
- [ ] **Attach**: Video + notebook to writeup

---

## 🛠️ Tooling

- [x] `scripts/generate_sft_submission.py` - Main notebook generator
- [x] `scripts/generate_continuation_notebook.py` - Continuation generator
- [x] `scripts/smoke_test_notebook.py` - Pre-flight syntax check
- [x] `scripts/sample_cot_collection.py` - CoT reservoir sampler
- [x] `scripts/sample_openo1_english.py` - OpenO1 English filter
- [x] `scripts/sample_glaiveai_continuation.py` - Continuation data sampler
