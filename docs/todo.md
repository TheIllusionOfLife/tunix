# Tunix SFT Strategy: Master TODO

Strategy pivot: From GRPO on math/code to **SFT on diverse domains** with reasoning traces.

## 📚 Documentation Index
- **Strategy**: [Scoring Breakdown](scoring_breakdown.md) | [Advanced Strategies](advanced_strategies.md)
- **Execution**: [Kaggle Memo](kaggle_memo.md) | [TPU Resource Plan](tpu_resource_plan.md)
- **Submission**: [Unrestricted Guide](unrestricted_mode_guide.md) | [Writeup Content](writeup_content.md) | [Video Script](video_script.md)
- **Archive**: [GRPO Strategy Docs](../archive/) (backup if we revert)

---

## 🎯 Core Strategy

**Method**: Supervised Fine-Tuning (SFT) on high-quality reasoning traces
**Focus**: Non-verifiable domains (creative, analytical, philosophical, commonsense)
**Rationale**: FAQ states verifiable tasks (math/code) have "much lower weights"

---

## 📦 Datasets to Use

| Dataset | Samples | Domain Focus | License |
|:---|:---:|:---|:---|
| sequelbox/Raiden-DeepSeek-R1 | 62.9K | Creative/analytical | Apache 2.0 |
| O1-OPEN/OpenO1-SFT | 20K (sampled) | General reasoning | Apache 2.0 |
| moremilk/General_Inquiry_Thinking | 6K | Philosophical/everyday | MIT |
| pharaouk/CoT-Collection | 10K (filtered) | Commonsense/ethics | CC-BY-4.0 |
| glaiveai/reasoning-v1-20m | Unlimited | Extended training | Apache 2.0 |

**Note**: Download raw from HuggingFace, process in-notebook to prove public data usage.

---

## 🗂️ Data Preparation

- [ ] Create Kaggle dataset with raw data downloads
- [ ] Write preprocessing code for each dataset format
- [ ] Add source documentation for judge verification

### Format Standardization
All datasets converted to:
```
<start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
<reasoning>{trace}</reasoning>
<answer>{answer}</answer>
```

---

## ✅ Phase 1: Single Session (45 pts)

- [ ] **Notebook**: Create `tunix_sft_train.ipynb`
- [ ] **Data**: Raiden + OpenO1 + General + CoT (~100K samples)
- [ ] **Training**: SFT with 2-3 epochs
- [ ] **Verify**: Check `<reasoning>` tags in outputs
- [ ] **Save**: Checkpoint for unrestricted mode

---

## 🚀 Phase 2: Unrestricted Mode (+15 pts)

- [ ] **Session 2**: Continue SFT on glaiveai dataset
- [ ] **Session 3**: More SFT or optional GRPO polish
- [ ] **Upload**: Final model to Kaggle Models

---

## 🎬 Phase 3: Final Submission

- [ ] **Video**: Record < 3 min demo
- [ ] **Writeup**: Submit with notebook
- [ ] **Attach**: Video + notebook to writeup
