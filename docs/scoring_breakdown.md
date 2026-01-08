# Scoring Breakdown (GlaiveAI-Only Strategy)

## Competition Scoring Overview

| Category | Points | Our Approach |
|:---|:---:|:---|
| **Notebook Quality** | ~15 | Clean code, clear documentation |
| **Video Quality** | ~15 | Clear explanation of SFT strategy |
| **Model Quality** | 45 | SFT on GlaiveAI (2025 quality) |
| **Unrestricted Mode** | +15 | Extended SFT with 100K fresh samples |
| **Total** | **90** | |

---

## Model Quality Evaluation

From FAQ #6:
> "A private evaluation dataset will be constructed from scratch... covering a **range of domains**"
> "Verifiable tasks (math/coding) will have **much lower weights**"

### Domain Weight Predictions

| Domain | Weight | GlaiveAI Coverage |
|:---|:---:|:---|
| Creative Writing | High | ✅ Strong |
| Analytical Reasoning | High | ✅ Strong |
| Social Science | High | ✅ Strong |
| Philosophical/Ethics | Medium | ✅ Good |
| General Reasoning | Medium | ✅ Good |
| Math | Low | ⚠️ Not prioritized |
| Coding | Low | ⚠️ Not prioritized |

---

## Why GlaiveAI-Only

| Factor | GlaiveAI | Others |
|:---|:---|:---|
| **Model Year** | 2025 | Mixed |
| **Quality** | DeepSeek-R1-Distill-70B | Variable |
| **Focus** | Non-math/code (aligned) | Math/code heavy |
| **Format** | Consistent | Inconsistent |

---

## Our Dataset Strategy

| Dataset | Samples | Notes |
|:---|:---:|:---|
| GlaiveAI | 180,000 | Primary dataset |
| **Total** | **180K** | 4 epochs (Dynamic steps) |

---

## Risk Assessment

| Risk | Mitigation |
|:---|:---|
| Single source dependency | GlaiveAI is massive (22M+), high quality |
| Format compliance | Consistent `<think>` tag = easier standardization |
| Math performance drop | Acceptable - FAQ deprioritizes math/code |
| Overfitting | 4 epochs is moderate; 180K samples is large |
