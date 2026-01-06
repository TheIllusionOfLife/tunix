# Kaggle Writeup Content (SFT Strategy)

## 1. Basic Details

**Title**: 
`Tunix Zero-Cost: Teaching Reasoning Through Demonstration`

**Subtitle**: 
`SFT on high-quality reasoning traces across diverse domains using only public data.`

**Card Image**: 
Diagram showing: `Gemma 2B-IT` → `[SFT on Diverse Reasoning]` → `Thinking Model`

---

## 2. Project Description

### Supervised Fine-Tuning on Diverse Domains

We prioritized non-verifiable domains (creative, analytical, philosophical) over math/code because:
1. Competition evaluation emphasizes diverse reasoning quality
2. Smaller models benefit more from demonstration than exploration
3. SFT is more efficient, allowing 10x more training samples

#### Datasets Used (Pre-sampled Parquet Files)

| Dataset | Samples | Sampling Method |
|:---|:---:|:---|
| Raiden-DeepSeek-R1 | 62.9K | Full dataset |
| OpenO1-SFT | 20K | English-only, random sample |
| CoT-Collection | 10K | Reservoir sampling (seed=42) |
| GlaiveAI-Reasoning | 30K | First N |
| **Total** | **~123K** | |

All datasets feature explicit reasoning traces (`<think>` tags) distilled from frontier models.

### Implementation Details

- **Library**: `google-tunix`, `flax`, `jax`
- **Hardware**: Kaggle TPU VM v5e-8
- **Method**: LoRA (Low-Rank Adaptation) for efficient fine-tuning
- **Data**: Pre-processed parquet files for reproducibility
- **Runtime**: ~8 hours processing 120K+ samples

### Key Insight

> "For 2B parameter models, learning from demonstrations is more effective than reinforcement learning. SFT provides dense supervision at every token, while RL provides sparse rewards only at sequence end."

### Unrestricted Mode

For bonus points, we continue training using **100K fresh samples** from glaiveai/reasoning-v1-20m (samples 30,001-130,000, not overlapping with session 1).

---

## 3. Learnings

- **Domain Matters More Than Method**: Training on diverse, high-weight domains (creative, analytical) outweighs technique sophistication
- **SFT Efficiency**: Processed 120K samples vs ~1,500 GRPO steps in same time
- **Pre-sampling Saves Runtime**: Pre-processed parquet files eliminate streaming/sampling overhead on Kaggle
- **Reasoning Traces Are Key**: Explicit `<think>` traces teach structured problem-solving

---

## 4. Attachments

- **Video**: YouTube link (< 3 min)
- **Notebook**: `tunix_sft_train.ipynb`
