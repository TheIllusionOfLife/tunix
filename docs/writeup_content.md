# Writeup for Kaggle Tunix competition

## Title
Tunix: Teach Reasoning Through Demonstration

## Subtitle
High-quality SFT on 180K reasoning traces from DeepSeek-R1-Distill-70B

## Project Description

### The Challenge
Small language models like Gemma2 2B often answer questions too quickly without showing their reasoning process. The traditional approach uses reinforcement learning (GRPO) on math problems, but this has limitations:
- Only ~1,500 steps possible in 9 hours
- Focuses on math/code domains that have "much lower weights" per competition FAQ
- Requires verifiable reward signals

### Our Approach: SFT on Reasoning Traces
Instead of teaching the model to explore, we teach it to imitate high-quality thinking. Supervised Fine-Tuning (SFT) on explicit reasoning traces gives:
- **Dense supervision**: Every token gets feedback, not just final answers
- **Diverse domains**: Creative, analytical, philosophical reasoning
- **Competition-aligned**: Focuses on what judges actually evaluate

### Key Technical Decisions

1. **Single High-Quality Dataset**: We use GlaiveAI (glaiveai/reasoning-v1-20m), which features reasoning traces from DeepSeek-R1-Distill-Llama-70B. This dataset focuses on non-math/code domains like social science and creative writing.

2. **Format Standardization**: All responses are converted to `<reasoning>...</reasoning>` and `<answer>...</answer>` tags, teaching the model explicit structure.

3. **LoRA Training**: Low-rank adaptation enables efficient fine-tuning within the 9-hour constraint while achieving strong results.

4. **Training Configuration**:
   - **SFT**: ~180k samples, 4 epochs (Dynamic steps)
   - ~7 hours runtime

### Why This Works
The competition FAQ explicitly states that "verifiable tasks (math/coding) will have much lower weights." By training on high-quality reasoning from domains the competition actually values, we maximize evaluation performance.

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | gemma2-2b-it |
| Method | LoRA (rank=64, alpha=64) |
| Dataset | 180k samples from glaiveai/reasoning-v1-20m |
| Steps | ~4 epochs (Dynamic steps) |
| Learning Rate | 2e-5 |
| Batch Size | 32 (effective) |
| Max Seq Length | 2048 |

---

## Our Journey & Failed Experiments

Our final solution is the result of a long journey through the Tunix ecosystem. We iterated through several approaches before arriving at our final strategy.

### Phase 1: Exploration & Tutorials

We started by exploring the Google DeepMind starter notebooks. We set up the Tunix environment on TPU v5e-8 and successfully reproduced the GRPO pipeline on the GSM8K dataset.
- **Goal**: Understand Tunix `GRPOLearner` and `RLCluster`.
- **Result**: Functional pipeline, but realized that simple reproduction wouldn't win.

### Phase 2: The Hybrid SFT+GRPO Attempt

We attempted to build a pipeline that chained SFT (using Magpie data) followed by GRPO (using GSM8K).
- **Hypothesis**: SFT would teach the format, and GRPO would optimize the reasoning.
- **Why it failed**: The complexities of managing two distinct training phases within the 9-hour Kaggle limit were too high. We had to mock parts of the trainer to make it fit, and ultimately cancelled this approach to find something more efficient.

### Phase 3: Pure GRPO on Math/Code

We pivoted to a "Pure GRPO" strategy, dropping the SFT phase entirely. We reasoned that `gemma2-2b-it` was already instruction-tuned enough.
- **Dataset**: We combined `gsm8k` (Math) and `mbpp` (Code).
- **Rewards**: Implemented custom reward functions for structure (XML tags) and correctness (`math_correctness_reward`, `code_correctness_reward`).
- **Result**: We achieved ~99% format compliance!
- **Critical Flaw**: We realized too late that the competition **explicitly deprioritizes math and code**. We were optimizing a model for tasks that would have "much lower weights" in the final judging.

### Phase 4: The Final Pivot (SFT on the glaiveai dataset)
*Current Solution*

This led us to our final pivot. instead of *reinforcing* math/code (which we can verify but isn't valued), we chose to *imitate* high-quality general reasoning (which is valued but hard to verify).
- **Shift**: From GRPO (RL) → SFT (Supervised Learning).
- **Data**: From Math/Code → General Reasoning (glaiveai/reasoning-v1-20m 180k samples).
- **Outcome**: A model that "thinks" eloquently about philosophy, art, and science, exactly what the judges want.

---

## Learnings

1. **Read the Rules First (Strategic)**
   We spent weeks optimizing for Math/Code (Phase 3) before noticing the FAQ deprioritized them. We learned that understanding the **evaluation criteria** is just as important as the model architecture.

2. **Imitation > Exploration for Small Models (Architectural)**
   We discovered that for a 2B parameter model, "Behavior Cloning" (SFT) on high-quality thoughts from a 70B teacher is far more effective than asking it to "discover" reasoning paths via RL (GRPO). The 2B model lacks the capacity to self-correct from scratch but is an excellent **student** of structured thought.

3. **Structure is Scaffolding (Data)**
   The `<reasoning>` tag isn't just XML; it's a cognitive scaffold. By forcing the model to output "Wait, I should check..." before the answer, we leverage the autoregressive nature of LLMs to condition the final answer on a "better state" of latent variables.

4. **Efficiency via Abstract Initialization (Technical)**
   Using Tunix's `nnx.eval_shape` allowed us to define the entire model and sharding layout on the TPU mesh **without allocating memory**. This "abstract initialization" was crucial for avoiding OOMs during the complex LoRA setup phases on TPU v5e-8.

5. **Quality > Quantity (Data)**
   One curated dataset (GlaiveAI) outperformed our mixtures of older datasets. The "freshness" and explicit reasoning style of the data mattered more than volume.

---

## Unrestricted Mode

For the multi session mode, we continue training with 100K fresh samples from the same source (train[180000:280000]), using a lower learning rate (5e-6) for refinement.

---

## Data Source

- **Dataset**: glaiveai/reasoning-v1-20m
- **License**: Apache 2.0
- **Samples Used**: 180K (single session) + 100K (unrestricted)
