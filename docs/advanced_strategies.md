# Advanced Strategies (SFT Approach)

## Core Philosophy

> "For 2B parameter models, demonstration is more effective than exploration."

Instead of reinforcement learning on verifiable tasks, we teach reasoning through high-quality examples.

---

## Dataset Selection Strategy

### Primary Criteria
1. **Domain**: Non-math, non-code (high competition weight)
2. **Format**: Explicit reasoning traces (`<think>`, `<Thought>`, etc.)
3. **Quality**: Distilled from frontier models (R1, O1, etc.)
4. **License**: Apache 2.0, MIT, or CC-BY
5. **Language**: English-only (Chinese filtered out)

### Dataset Mix (~123K samples)

| Dataset | Samples | Method | Strength |
|:---|:---:|:---|:---|
| Raiden-DeepSeek-R1 | 62.9K | Full | Creative/analytical |
| OpenO1-SFT | 20K | English + Random | General reasoning |
| CoT-Collection | 10K | Reservoir sampling | Commonsense/ethics |
| GlaiveAI | 30K | First N | Non-math/code |

---

## Pre-sampling Scripts

| Script | Output | Size |
|:---|:---|:---|
| `scripts/sample_openo1_english.py` | `openo1_sft_english_20k.parquet` | 44.8 MB |
| `scripts/sample_cot_collection.py` | `cot_collection_10k.parquet` | 6.6 MB |
| `scripts/sample_glaiveai_continuation.py` | `glaiveai_continuation_100k.parquet` | 365.2 MB |

All scripts use `seed=42` for reproducibility.

---

## Training Configuration

### Full-Weight SFT vs LoRA

| Approach | Pros | Cons |
|:---|:---|:---|
| **Full SFT** | Maximum quality | Higher memory, slower |
| **LoRA** | Efficient, stable | May underfit |

**Current**: LoRA for speed and stability.

### Hyperparameters

```python
SFT_CONFIG = {
    "learning_rate": 2e-5,       # Session 1
    "continuation_lr": 5e-6,     # Session 2 (lower)
    "batch_size": 2,
    "gradient_accumulation": 16,  # Effective batch = 32
    "max_seq_length": 1024,
    "warmup_steps": 100,
}
```

---

## Format Standardization

All datasets use different tag formats:

| Dataset | Original Format | Standardized To |
|:---|:---|:---|
| Raiden | DeepSeek-R1 style | `<reasoning>...<answer>` |
| OpenO1 | `<Thought>...<Output>` | `<reasoning>...<answer>` |
| CoT-Collection | `rationale` column | `<reasoning>...<answer>` |
| GlaiveAI | `<think>` only | `<reasoning>...<answer>` |

**Target format**:
```
<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model
<reasoning>{trace}</reasoning>
<answer>{answer}</answer>
```

---

## Unrestricted Mode Strategy

### Session Allocation

| Session | Focus | Data Size | Source |
|:---|:---|:---:|:---|
| 1 | Core diverse reasoning | ~123K | tunix-sft-data |
| 2 | Extended coverage | 100K | tunix-sft-continuation-data |
| 3 | Polish (optional) | Variable | More GlaiveAI or GRPO |

### Fresh Data Principle

Continuation training uses **non-overlapping** samples:
- Session 1: GlaiveAI samples 1-30,000
- Session 2: GlaiveAI samples 30,001-130,000

This prevents overfitting and maximizes diversity.

---

## Fallback Strategies

If SFT underperforms:
1. **Add GRPO polish**: Light RL after SFT
2. **Increase epochs**: More passes over data
3. **Filter for quality**: Remove low-quality samples
4. **Revert to GRPO**: Documents preserved in `archive/`
