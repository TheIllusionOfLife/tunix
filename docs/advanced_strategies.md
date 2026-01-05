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

### Dataset Mix (100K target)

| Dataset | Samples | Strength |
|:---|:---:|:---|
| Raiden-DeepSeek-R1 | 62.9K | Creative/analytical - rare find |
| OpenO1-SFT | 20K | General reasoning diversity |
| General_Inquiry_Thinking | 6K | Philosophical depth |
| CoT-Collection | 10K | Commonsense/ethics |

---

## Training Configuration

### Full-Weight SFT vs LoRA

| Approach | Pros | Cons |
|:---|:---|:---|
| **Full SFT** | Maximum quality | Higher memory, slower |
| **LoRA** | Efficient, stable | May underfit |

**Recommendation**: Start with LoRA for speed, upgrade to full SFT if memory allows.

### Hyperparameters

```python
SFT_CONFIG = {
    "learning_rate": 2e-5,
    "batch_size": 2-4,  # Memory dependent
    "epochs": 2-3,
    "max_seq_length": 1024-2048,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
}
```

---

## Format Standardization

All datasets use different tag formats:

| Dataset | Format |
|:---|:---|
| Raiden | DeepSeek-R1 format |
| OpenO1 | `<Thought>...</Thought>` |
| General_Inquiry | `metadata.reasoning` column |
| CoT-Collection | `rationale` column |

**Target format**:
```
<start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
<reasoning>{trace}</reasoning>
<answer>{answer}</answer>
```

---

## Unrestricted Mode Strategy

### Session Allocation

| Session | Focus | Data Size |
|:---|:---|:---:|
| 1 | Core diverse reasoning | 100K |
| 2 | Extended coverage | 100K-500K |
| 3 | Polish or specialization | Variable |

### glaiveai/reasoning-v1-20m

With 22.2M samples, this dataset enables:
- Deep fine-tuning across multiple epochs
- Domain-specific filtering if needed
- Curriculum learning (easy → hard)

---

## Fallback Strategies

If SFT underperforms:
1. **Add GRPO polish**: Light RL after SFT
2. **Increase epochs**: More passes over data
3. **Filter for quality**: Remove low-quality samples
4. **Revert to GRPO**: Documents preserved in `archive/`
