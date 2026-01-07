# Kaggle Writeup Content (SFT Strategy)

## 1. Basic Details

**Title**: 
`Tunix Zero-Cost: Teaching Reasoning Through Demonstration`

**Subtitle**: 
`SFT on high-quality GlaiveAI reasoning traces using only public data.`

**Card Image**: 
Diagram showing: `Gemma 2B-IT` → `[SFT on GlaiveAI]` → `Thinking Model`

---

## 2. Project Description

### Supervised Fine-Tuning on GlaiveAI

We chose **GlaiveAI-only** training after evaluating multiple datasets:

| Dataset | Decision | Reason |
|---------|----------|--------|
| GlaiveAI | ✅ **Use** | 2025 model, non-math/code focus |
| CoT-Collection | ❌ Drop | 2023 models, outdated |
| Raiden-DeepSeek-R1 | ❌ Drop | Unfiltered, infinite loops |
| OpenO1-SFT | ❌ Drop | Math/code focus (deprioritized) |

### Why GlaiveAI?

1. **Competition-aligned**: FAQ says math/code have "much lower weights"
2. **2025 Quality**: DeepSeek-R1-Distill-70B reasoning
3. **Focus**: Social science, creative writing, analytical domains
4. **Scale**: 22M+ samples available

### Training Configuration

| Setting | Value |
|---------|-------|
| Samples | 180K |
| Epochs | 4 |
| Steps | ~22,500 |
| Runtime | ~7 hours |

### Implementation Details

- **Library**: `google-tunix`, `flax`, `jax`
- **Hardware**: Kaggle TPU VM v5e-8
- **Method**: LoRA (Low-Rank Adaptation)
- **Format**: `<reasoning>` / `<answer>` tags

### Key Insight

> "Quality over quantity: One 2025 dataset aligned with competition goals outperforms four mixed-quality datasets."

### Unrestricted Mode

For bonus points, we continue training using **100K fresh samples** from GlaiveAI (`train[180000:280000]`, non-overlapping with session 1).

---

## 3. Learnings

- **Dataset Quality Matters**: 2023 datasets can't compete with 2025 reasoning quality
- **Alignment > Diversity**: Better to focus on competition-aligned domains
- **Single Source Benefits**: No format standardization issues
- **GlaiveAI is Underrated**: Massive, high-quality, perfect for this competition

---

## 4. Attachments

- **Video**: YouTube link (< 3 min)
- **Notebook**: `tunix_sft_train.ipynb`
