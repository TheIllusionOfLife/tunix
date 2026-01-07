# Advanced Strategies (SFT Approach)

## Core Philosophy

> "For 2B parameter models, demonstration is more effective than exploration."

Instead of reinforcement learning on verifiable tasks, we teach reasoning through high-quality examples.

---

## Dataset Selection Strategy

### Selection Criteria
1. **Domain**: Non-math, non-code (high competition weight)
2. **Quality**: 2025 models (DeepSeek-R1-Distill-70B or equivalent)
3. **Format**: Explicit reasoning traces (`<think>` tags)
4. **License**: Apache 2.0
5. **Alignment**: Matches competition evaluation criteria

### Dataset Decision

| Dataset | Created | Decision | Reason |
|---------|---------|----------|--------|
| **GlaiveAI** | 2025 | ✅ **Use** | Non-math/code focus, 2025 quality |
| CoT-Collection | 2023 | ❌ Drop | Outdated model quality |
| Raiden-DeepSeek-R1 | 2025 | ❌ Drop | Unfiltered, infinite loops |
| OpenO1-SFT | 2025 | ❌ Drop | Math/code focus (deprioritized) |

---

## Training Configuration

### Hyperparameters

```python
SFT_CONFIG = {
    "learning_rate": 2e-5,       # Session 1
    "continuation_lr": 5e-6,     # Session 2 (lower)
    "batch_size": 8,
    "gradient_accumulation": 4,   # Effective batch = 32
    "max_seq_length": 2048,
    "warmup_steps": 200,
    "steps": 22500,               # 180K × 4 epochs
}
```

---

## Format Standardization

GlaiveAI uses `<think>` tags, standardized to competition format:

| Original | Standardized |
|----------|--------------|
| `<think>` | `<reasoning>` |
| (content after think) | `<answer>` |

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

| Session | GlaiveAI Slice | Samples |
|---------|----------------|---------|
| 1 (Single) | `train[:180000]` | 180K |
| 2 (Unrestricted) | `train[180000:280000]` | 100K |

Non-overlapping slices prevent overfitting.

---

## Why GlaiveAI-Only

1. **2025 Quality**: DeepSeek-R1-Distill-70B is state-of-the-art
2. **Competition-Aligned**: Non-math/code focus matches FAQ guidance
3. **Consistency**: Single source = no format standardization issues
4. **Scale**: 22M+ samples available for extension
