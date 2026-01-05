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

### Introduction

Reasoning is the frontier of small language models. While large models naturally "think" before responding, smaller models like Gemma 2B often rush to provide answers without proper deliberation.

Our approach: **Teach reasoning through demonstration**, not trial-and-error. By fine-tuning on high-quality reasoning traces from diverse domains, we enable Gemma 2B to learn HOW to think, not just WHAT to answer.

### The Strategy: Supervised Fine-Tuning on Diverse Domains

We prioritized **non-verifiable domains** (creative, analytical, philosophical) over math/code because:
1. Competition evaluation emphasizes diverse reasoning quality
2. Smaller models benefit more from demonstration than exploration
3. SFT is more efficient, allowing 10x more training samples

#### Datasets Used
- **Raiden-DeepSeek-R1**: 62.9K creative & analytical samples
- **OpenO1-SFT**: 20K general reasoning samples
- **CoT-Collection**: 10K commonsense & ethics tasks
- **GlaiveAI-Reasoning**: 30K sampled math/code/general tasks

All datasets feature explicit reasoning traces (`<think>` tags) distilled from frontier models.

### Implementation Details

- **Library**: `google-tunix`, `flax`, `jax`
- **Hardware**: Kaggle TPU VM v5e-8
- **Method**: Full-weight SFT (not LoRA) for maximum quality
- **Runtime**: ~8 hours processing 100K+ samples

### Key Insight

> "For 2B parameter models, learning from demonstrations is more effective than reinforcement learning. SFT provides dense supervision at every token, while RL provides sparse rewards only at sequence end."

### Unrestricted Mode

For bonus points, we continue training using the remaining massive glaiveai/reasoning-v1-20m dataset (22.2M samples) across multiple sessions.

---

## 3. Learnings

- **Domain Matters More Than Method**: Training on diverse, high-weight domains (creative, analytical) outweighs technique sophistication
- **SFT Efficiency**: Processed 100K samples vs ~1,500 GRPO steps in same time
- **Reasoning Traces Are Key**: Explicit `<think>` traces teach structured problem-solving

---

## 4. Attachments

- **Video**: YouTube link (< 3 min)
- **Notebook**: `tunix_sft_train.ipynb`
