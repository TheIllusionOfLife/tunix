# Scoring Breakdown (SFT Strategy)

## Competition Scoring Overview

| Category | Points | Our Approach |
|:---|:---:|:---|
| **Notebook Quality** | ~15 | Clean code, clear documentation |
| **Video Quality** | ~15 | Clear explanation of SFT strategy |
| **Model Quality** | 45 | SFT on diverse reasoning traces |
| **Unrestricted Mode** | +15 | Extended SFT across sessions |
| **Total** | **90** | |

---

## Model Quality Evaluation

From FAQ #6:
> "A private evaluation dataset will be constructed from scratch... covering a **range of domains**"
> "Verifiable tasks (math/coding) will have **much lower weights**"

### Domain Weight Predictions

| Domain | Weight | Our Coverage |
|:---|:---:|:---|
| Creative/Analytical | High | ✅ 62.9K Raiden samples |
| Philosophical/Ethics | High | ✅ 10K CoT |
| Commonsense | Medium | ✅ CoT-Collection |
| General Reasoning | Medium | ✅ 20K OpenO1 |
| Math | Low | ⚠️ Not prioritized |
| Coding | Low | ⚠️ Not prioritized |

### Evaluation Criteria (Predicted)

1. **Accuracy**: Does the answer address the question?
2. **Partial Accuracy**: Is reasoning partially correct?
3. **Format Accuracy**: Are `<reasoning>` and `<answer>` tags present?
4. **Reasoning Quality**: Is the thinking process coherent?
5. **Domain Coverage**: Can model handle diverse topics?

---

## Why SFT Over GRPO

| Factor | GRPO | SFT |
|:---|:---|:---|
| **Samples/9hr** | ~1,500 | ~100,000 |
| **Domain Coverage** | Math/Code only | Diverse |
| **Competition Alignment** | Low-weight domains | High-weight domains |
| **2B Model Suitability** | Questionable | Better |

---

## Risk Assessment

| Risk | Mitigation |
|:---|:---|
| Model overfits | Use diverse datasets, multiple sources |
| Format compliance drops | Data has explicit reasoning tags |
| Worse at math | Acceptable - low competition weight |
| Dataset quality varies | Multiple datasets for redundancy |
