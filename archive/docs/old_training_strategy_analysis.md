# Training Strategy Analysis: 180K Samples × 4 Epochs

**Decision Date**: 2026-01-07
**Final Strategy**: 180K diverse samples, 4 epochs (~22,500 steps)

---

## Executive Summary

After analyzing training runs, evaluating output quality, and debating data diversity vs training depth, we concluded that **180K samples with 4 epochs** provides the optimal balance between learning diverse patterns and reinforcing format compliance.

---

## 1. Empirical Data from Training Runs

### Run 1: 4000 Steps (1 Epoch on 122K)
| Metric | Value |
|--------|-------|
| Training time | ~73 min |
| Format pass rate | 75% (3/4) |
| Key failure | Math (no format tags) |

### Run 2: 8000 Steps (2 Epochs on 122K)
| Metric | Value |
|--------|-------|
| Training time | ~146 min |
| Format pass rate | **90% (9/10)** |
| Key failure | Math (no format tags) |
| Improvement | +15% format compliance |

**Key Insight**: Doubling epochs from 1→2 yielded significant improvement (+15%). This suggests the model benefits from reinforcement.

---

## 2. LLM-as-a-Judge Quality Analysis

We evaluated 10 outputs from the 2-epoch run as an LLM-as-a-Judge would:

| Prompt | Score | Key Issue |
|--------|-------|-----------|
| Palindrome (code) | 8.5/10 | None |
| Sky blue (5yr old) | 5.5/10 | **Audience adaptation failure** |
| Photosynthesis | 6.5/10 | **Lazy answer: "I hope this helps!"** |
| AI healthcare ethics | 9.5/10 | None |
| AI rights debate | 9.0/10 | None |
| Math problem | 7.0/10 | Format failure (correct answer) |
| Creative writing | 8.5/10 | None |
| Haiku | 8.0/10 | None |
| AI in education | 8.5/10 | None |
| Renewable energy | 8.0/10 | None |
| **Average** | **7.7/10** | |

### Identified Quality Issues

1. **Audience Adaptation Failure** (Sky blue to 5-year-old)
   - Model explained Rayleigh scattering with "wavelength of 500nm"
   - A 5-year-old explanation should use simple analogies
   - **Root cause**: Lack of diverse audience-adapted examples in training data

2. **Lazy Answer Tags** (Photosynthesis)
   - Excellent reasoning, but answer was just "I hope this helps!"
   - Model learned this pattern from training data
   - **Root cause**: Some training examples have weak answer extraction

3. **Math Format Failure**
   - Correct mathematical answer, but no `<reasoning>` or `<answer>` tags
   - **Root cause**: Math domain may be undertrained for format
   - **Mitigation**: Competition FAQ explicitly deprioritizes math/code

---

## 3. Strategic Options Evaluated

### Option A: 5 Epochs on 122K (More Depth)
```
Samples: 122K
Epochs: 5
Steps: ~19,000
Time: ~6 hours
```

**Pros:**
- Deeper memorization of format patterns
- Simple, no new data preparation
- Lower risk of introducing noise

**Cons:**
- Won't fix audience adaptation (requires diverse examples)
- May reinforce lazy answer patterns
- Overfitting risk increases after epoch 3-4

### Option B: 3 Epochs on 243K (More Diversity)
```
Samples: 243K (+40K from each of 3 sources)
Epochs: 3
Steps: ~23,000
Time: ~7.2 hours
```

**Pros:**
- More diverse audience adaptation examples
- Better answer quality through variety
- Less overfitting risk

**Cons:**
- Variable data quality from larger samples
- Less reinforcement per pattern
- More data preparation work

### Option C: 4 Epochs on 180K (Compromise) ✓ SELECTED
```
Samples: 180K (+20K from each of 3 sources)
Epochs: 4
Steps: ~22,500
Time: ~7 hours
```

**Pros:**
- Balances diversity and reinforcement
- Addresses quality issues with new examples
- 4 epochs provides strong format learning
- Comfortable time buffer (~2 hours)

**Cons:**
- Requires new data preparation

---

## 4. Dataset Expansion Plan

### Current Dataset (122K)
| Dataset | Samples |
|---------|---------|
| Raiden-DeepSeek-R1 | 62,925 (full) |
| OpenO1-SFT (English) | 20,000 |
| CoT-Collection | 10,000 |
| GlaiveAI | 30,000 |
| **Total** | **122,925** |

### Expanded Dataset (180K)
| Dataset | Current | Add | New Total |
|---------|---------|-----|-----------|
| Raiden-DeepSeek-R1 | 62,925 | 0 (full) | 62,925 |
| OpenO1-SFT (English) | 20,000 | +20,000 | 40,000 |
| CoT-Collection | 10,000 | +20,000 | 30,000 |
| GlaiveAI | 30,000 | +20,000 | 50,000 |
| **Total** | **122,925** | **+60,000** | **~183,000** |

### Sampling Strategy
- **OpenO1-SFT**: Random sample from English-only filtered data (seed=42)
- **CoT-Collection**: Reservoir sampling from 1.8M pool (seed=42)
- **GlaiveAI**: Sequential slice `train[30000:50000]` (non-overlapping with original)

---

## 5. Time Budget Validation

```
Available: 9 hours = 540 min

Training estimate:
├── Setup + data loading: ~10 min
├── Model loading: ~5 min
├── Baseline eval: ~5 min
├── Training (22,500 steps): ~428 min (~7.1 hours)
├── Post-training eval: ~5 min
├── Checkpoint save: ~5 min
└── Total: ~458 min (~7.6 hours)

Buffer: ~82 min (~1.4 hours) ✓
```

---

## 6. Expected Improvements

### What More Data Fixes
| Issue | Before | After (Expected) |
|-------|--------|------------------|
| Audience adaptation | Fails on "explain to X" | Better with diverse examples |
| Lazy answers | "I hope this helps!" | More varied answer styles |
| Domain coverage | Good | Better |

### What More Epochs Fixes
| Issue | Before | After (Expected) |
|-------|--------|------------------|
| Format compliance | 90% | 95%+ |
| Tag consistency | Good | Better |
| Pattern reinforcement | Good | Strong |

---

## 7. Unrestricted Mode Strategy

The expanded dataset still leaves room for unrestricted mode bonus (+15 points):

| Mode | GlaiveAI Slice | Fresh Data |
|------|----------------|------------|
| Session 1 (Single) | `train[:50000]` | 50K |
| Session 2 (Unrestricted) | `train[50000:150000]` | 100K (non-overlapping) |

---

## 8. Competition Alignment

This strategy aligns with competition guidance:

1. **FAQ states math/code are deprioritized** → We focus on creative, analytical, philosophical domains
2. **Evaluation uses diverse domains** → More diverse training data matches this
3. **LLM-as-a-Judge evaluation** → Quality of reasoning matters more than format perfection
4. **Single-session worth 45pts vs 15pts unrestricted** → Maximize single-session quality

---

## 9. Action Items

- [ ] Create `openo1_sft_english_40k.parquet` (expand from 20K)
- [ ] Create `cot_collection_30k.parquet` (expand from 10K)
- [ ] Create `glaiveai_50k.parquet` (expand from 30K)
- [ ] Update notebook with `SFT_STEPS = 22500`
- [ ] Upload new parquets to Kaggle dataset
- [ ] Run training and evaluate

---

## 10. Conclusion

The **180K samples × 4 epochs** strategy represents the optimal balance:

- **Enough diversity** to fix audience adaptation and lazy answer issues
- **Enough epochs** to strongly reinforce format compliance
- **Enough buffer** to handle unexpected issues
- **Aligned with competition** evaluation criteria

This is the grandmaster approach: read the failure modes carefully, address root causes rather than symptoms, and optimize for what the judges actually measure.