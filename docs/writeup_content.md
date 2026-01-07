# Kaggle Competition Writeup

## Title
**Tunix Zero-Cost: Teaching Reasoning Through Demonstration**

## Subtitle
*High-quality SFT on 180K reasoning traces from DeepSeek-R1-Distill-70B*

---

## Project Description

### The Challenge
Small language models like Gemma 2B often answer questions too quickly without showing their reasoning process. The traditional approach uses reinforcement learning (GRPO) on math problems, but this has limitations:
- Only ~1,500 steps possible in 9 hours
- Focuses on math/code domains that have "much lower weights" per competition FAQ
- Requires verifiable reward signals

### Our Approach: SFT on Reasoning Traces
Instead of teaching the model to explore, we teach it to imitate high-quality thinking. Supervised Fine-Tuning (SFT) on explicit reasoning traces gives:
- **Dense supervision**: Every token gets feedback, not just final answers
- **Diverse domains**: Creative, analytical, philosophical reasoning
- **Competition-aligned**: Focuses on what judges actually evaluate

### Key Technical Decisions

1. **Single High-Quality Dataset**: We use GlaiveAI (glaiveai/reasoning-v1-20m), which features reasoning traces from DeepSeek-R1-Distill-Llama-70B. This 2025 dataset focuses on non-math/code domains like social science and creative writing.

2. **Format Standardization**: All responses are converted to `<reasoning>...</reasoning>` and `<answer>...</answer>` tags, teaching the model explicit structure.

3. **LoRA Training**: Low-rank adaptation enables efficient fine-tuning within the 9-hour constraint while achieving strong results.

4. **Training Configuration**:
   - 180K samples × 4 epochs
   - ~22,500 training steps
   - ~7 hours runtime

### Why This Works
The competition FAQ explicitly states that "verifiable tasks (math/coding) will have much lower weights." By training on high-quality reasoning from domains the competition actually values, we maximize evaluation performance.

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | Gemma 2B-IT |
| Method | LoRA (rank=64, alpha=64) |
| Dataset | GlaiveAI 180K |
| Steps | ~22,500 |
| Learning Rate | 2e-5 |
| Batch Size | 32 (effective) |
| Max Seq Length | 2048 |

---

## Learnings

1. **Quality > Quantity**: One curated 2025 dataset outperforms multiple older datasets
2. **Alignment Matters**: Train on what the competition measures, not what's easy to verify
3. **SFT Scales Better**: 180K samples in 7 hours vs ~1,500 GRPO steps

---

## Unrestricted Mode

For the +15 bonus points, we continue training with 100K fresh samples from the same source (train[180000:280000]), using a lower learning rate (5e-6) for refinement.

---

## Data Source

- **Dataset**: glaiveai/reasoning-v1-20m
- **License**: Apache 2.0
- **Samples Used**: 180K (single session) + 100K (unrestricted)
