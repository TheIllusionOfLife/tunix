# Training Strategy Analysis: GlaiveAI-Only (180K × 4 Epochs)

**Decision Date**: 2026-01-07 (Updated)
**Final Strategy**: GlaiveAI-only, ~180K samples, 4 epochs (Dynamic steps)

---

## Executive Summary

After analyzing training runs and dataset quality, we concluded that **GlaiveAI-only** provides the best alignment with competition goals. Other datasets were dropped due to quality/alignment concerns.

---

## 1. Dataset Evaluation

### Dropped Datasets

| Dataset | Created | Model | Issue | Decision |
|---------|---------|-------|-------|----------|
| **CoT-Collection** | Early 2023 | Pre-GPT4 | Outdated reasoning quality | ❌ Drop |
| **Raiden-DeepSeek-R1** | 2025 | R1 (685B) | Native R1 = infinite loops, unfiltered | ❌ Drop |
| **OpenO1-SFT** | 2025 | O1-style | Math/code focus misaligned | ❌ Drop |

### Selected Dataset

| Dataset | Created | Model | Focus | Decision |
|---------|---------|-------|-------|----------|
| **GlaiveAI** | 2025 | DeepSeek-R1-Distill-70B | Non-math/code (social science, creative) | ✅ Use |

---

## 2. Why GlaiveAI-Only

1. **Competition Alignment**: FAQ says math/code have "much lower weights"
2. **2025 Quality**: DeepSeek-R1-Distill-70B is state-of-the-art
3. **Curated Focus**: Specifically designed for non-code/math domains
4. **Massive Scale**: 22M+ samples available
5. **Consistency**: Single source = no format standardization issues

---

## 3. Training Configuration

| Setting | Value |
|---------|-------|
| Dataset | GlaiveAI 180K |
| Epochs | 4 |
| Steps | Dynamic (~21,400) |
| Effective Batch | 32 |
| Estimated Time | ~7 hours |

---

## 4. Time Budget

```
Available: 9 hours = 540 min

Training estimate:
├── Setup + data loading: ~10 min
├── Model loading: ~5 min
├── Baseline eval: ~5 min
├── Training (Dynamic steps): ~428 min (~7.1 hours)
├── Post-training eval: ~10 min
├── WandB extended eval: ~10 min
├── Checkpoint save: ~5 min
└── Total: ~473 min (~7.9 hours)

Buffer: ~67 min (~1.1 hours) ✓
```

---

## 5. Unrestricted Mode Strategy

| Mode | GlaiveAI Slice | Samples |
|------|----------------|---------|
| Session 1 (Single) | `train[:180000]` | 180K |
| Session 2 (Unrestricted) | `train[180000:280000]` | 100K (non-overlapping) |

---

## 6. Expected Outcomes

- **Format Compliance**: 95%+ (improved from 90%)
- **Reasoning Quality**: Consistent, 2025-quality responses
- **Domain Coverage**: Strong on creative, analytical, philosophical
- **Competition Alignment**: Matches evaluation criteria

---

## 7. Action Items

- [ ] Create `glaiveai_180k.parquet`
- [x] Update notebook to GlaiveAI-only
- [x] Set Dynamic Steps (Target 4 epochs)
- [ ] Upload to Kaggle and run training
