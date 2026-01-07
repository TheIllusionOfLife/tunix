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

## Our Journey & Failed Experiments

Our final solution is the result of a long journey through the Tunix ecosystem. We iterated through several approaches before arriving at the winning strategy.

### Phase 1: Exploration & Tutorials
*Artifacts: `my_tunix_note.ipynb`, `grpo-demo-gemma2-2b.ipynb`*

We started by exploring the Google DeepMind starter notebooks. We set up the Tunix environment on TPU v5e and successfully reproduced the GRPO pipeline on the GSM8K dataset.
- **Goal**: Understand Tunix `GRPOLearner` and `RLCluster`.
- **Result**: Functional pipeline, but realized that simple reproduction wouldn't win.

### Phase 2: The Hybrid SFT+GRPO Attempt
*Artifact: `cancelled_tunix-zero-cost-train.ipynb`*

We attempted to build a "Zero Cost" pipeline that chained SFT (using Magpie data) followed by GRPO (using GSM8K).
- **Hypothesis**: SFT would teach the format, and GRPO would optimize the reasoning.
- **Why it failed**: The complexities of managing two distinct training phases within the 9-hour Kaggle limit were too high. We had to mock parts of the trainer to make it fit, and ultimately **cancelled** this approach to find something more efficient.

### Phase 3: Pure GRPO on Math/Code
*Artifacts: `tunix_zero_cost_train.ipynb`, `tunix_continuation.ipynb`*

We pivoted to a "Pure GRPO" strategy, dropping the SFT phase entirely. We reasoned that `Gemma-2B-IT` was already instruction-tuned enough.
- **Dataset**: We combined `GSM8K` (Math) and `MBPP` (Code).
- **Rewards**: Implemented custom reward functions for structure (XML tags) and correctness (`math_correctness_reward`, `code_correctness_reward`).
- **Result**: We achieved ~99% format compliance!
- **Critical Flaw**: We realized too late that the competition **explicitly deprioritizes math and code**. We were optimizing a model for tasks that would have "much lower weights" in the final judging.

### Phase 4: The Final Pivot (SFT on GlaiveAI)
*Current Solution*

This led us to our final pivot. instead of *reinforcing* math/code (which we can verify but isn't valued), we chose to *imitate* high-quality general reasoning (which is valued but hard to verify).
- **Shift**: From GRPO (RL) → SFT (Supervised Learning).
- **Data**: From Math/Code → GlaiveAI (General Reasoning).
- **Outcome**: A model that "thinks" eloquently about philosophy, art, and science—exactly what the judges want.

---

## Learnings

1. **Read the Rules First**: We spent weeks optimizing for Math/Code (Phase 3) before noticing the FAQ deprioritized them.
2. **Quality > Quantity**: One curated 2025 dataset (GlaiveAI) outperforms a mix of older ones.
3. **Simplicity Wins**: Our "Cancelled" Phase 2 pipeline was too complex. The final pure SFT pipeline is robust and runs in 7 hours.

---

## Unrestricted Mode

For the +15 bonus points, we continue training with 100K fresh samples from the same source (train[180000:280000]), using a lower learning rate (5e-6) for refinement.

---

## Data Source

- **Dataset**: glaiveai/reasoning-v1-20m
- **License**: Apache 2.0
- **Samples Used**: 180K (single session) + 100K (unrestricted)
- **Reproducibility**: `python scripts/download_sft_datasets.py` creates the exact dataset
