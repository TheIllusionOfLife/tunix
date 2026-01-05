# Tunix Advanced Strategies & Evaluation Guide

## 1. DPO with UltraFeedback (The "Style" Polish)
**Concept**: SFT teaches "how to speak". GRPO teaches "how to solve". DPO teaches "what is preferred" (e.g., concise vs verbose).
**Implementation**:
1.  **Data**: `HuggingFaceH4/ultrafeedback_binarized` contains `(prompt, chosen, rejected)` triplets.
2.  **Pipeline Slot**: Insert *after* SFT and *before* GRPO.
3.  **Config**:
    ```python
    from tunix.dpo import dpo_trainer
    trainer = dpo_trainer.DPOTrainer(
        model=sft_model,             # Start from SFT checkpoint
        ref_model=sft_model,         # Reference usually same as init
        beta=0.1,                    # KL penalty
        train_dataset=dpo_dataset,   # Pairs of (chosen, rejected)
    )
    ```
4.  **Benefit**: Reduces "robot-speak" and improves specific stylistic constraints (e.g. "don't apologize repeatedly").

## 2. Rubric Rewards (For Creative Tasks)
**Concept**: In domains without a ground truth (Creative Writing), you can't use `exact_match`. You need a "Rubric".
**Implementation**: Define python functions that score specific properties.
```python
def rubric_reward_creative(prompts, completions, **kwargs):
    scores = []
    for c in completions:
        score = 0.0
        # 1. Length constraint (e.g. not too short)
        if len(c.split()) > 100: score += 0.2
        # 2. Formatting (e.g. uses bullet points)
        if "- " in c: score += 0.2
        # 3. No repetition
        if "As an AI language model" not in c: score += 0.6
        scores.append(score)
    return scores
```
**Usage**: Add this to the `reward_functions` list in `GRPOLearner`.

## 3. Ensembling (Adapter Swapping)
**Concept**: Train specialized LoRA adapters for different tasks, then swap them at inference time.
**Why**: A single 2B model struggles to be a master of Math AND Poetry.
**Strategy**:
1.  **Train Adapter A (Math)**: SFT(Magpie) -> GRPO(GSM8K). Save `adapter_math`.
2.  **Train Adapter B (Writing)**: SFT(UltraFeedback) -> DPO(UltraFeedback). Save `adapter_write`.
3.  **Inference**:
    ```python
    # Pseudo-code for inference loop
    if classify_prompt(prompt) == "MATH":
        model.load_adapter("adapter_math")
    else:
        model.load_adapter("adapter_write")
    ```
**Risk**: Classification overhead and loading time might eat into the 9-hour limit if switching frequently.

## 4. Process Reward Models (PRM) - The "Hard Mode"
**Concept**: Reward the model *per step* of reasoning, not just the final answer.
**Why**: Dense feedback learns faster than sparse feedback (GRPO/PPO).
**Difficulty**: Requires a dataset labeled at the step level (`Step 1: Correct`, `Step 2: Error`).
**Action**: Unless you manually label your reasoning traces, this is out of scope for Zero-Cost.

## 5. Iterative Self-Improvement (The Feedback Loop)
**Concept**: Use the model's own best outputs to retrain itself.
**Implementation (Zero-Cost)**:
1.  **Generate**: Run inference on 100 prompts.
2.  **Judge**: Use `scripts/evaluation_judge.py` (Human or LLM mode) to score them.
3.  **Filter**: Keep only samples with Score > 0.8.
4.  **Retrain**: Add high-scoring samples to your SFT dataset.
**Status**: Tool Implemented (`LocalLLMJudge`).

## 6. Agentic Reasoning (The "Thinking" Loop) - *Competitive Edge*
**Concept**: Use `AgenticGRPOLearner` (v0.1.4+) to enable multi-turn reasoning where the model can "act" (e.g., run code, search memory) and "observe" results before answering.
**Why**: Solves harder math/code problems by verifying intermediate steps. **This is how we beat models that only "guess" once.**
**Implementation**:
1.  **Learner**: Swap `GRPOLearner` for `tunix.rl.experimental.agentic_grpo_learner.AgenticGRPOLearner`.
2.  **Tools**: Define python functions (e.g., `calculator()`) the model can invoke.
3.  **Format**: Requires training heavily on tool-use trajectories (e.g., from `NuminaMath` dataset).

## 7. Throughput Optimization (SGLang & vLLM) - *Speed Edge*
**Concept**: Replace the vanilla JAX sampler with specialized inference engines provided in Tunix v0.1.4+.
**Features**:
*   **SGLang Sampler**: Drastically reduces sampling overhead during the "Act" phase of RL.
*   **vLLM Data Parallelism**: Allows running evaluations on much larger batch sizes.
**Goal**: Fit **2-3x more GRPO steps** into the 9-hour window, allowing convergence where others timeout.
**Status**: Requires manual installation of `vLLM` or `sglang` in the setup cell (see `project.optional-dependencies` in `pyproject.toml`).

## 8. Performance Profiling (The Diagnostics)
**Concept**: Use `tunix.perf.trace.PerfTracer` (v0.1.4+) to visualize exactly where TPU time is spent (e.g., Compilation vs Generation vs Training).
**Why**: In a constrained 9-hour run, knowing bottleneck allows targeted optimization (e.g., "Generation is slow -> Switch to SGLang" vs "Update is slow -> Increase Micro Batch").
**Implementation**:
```python
from tunix.perf import trace
tracer = trace.PerfTracer(devices=jax.devices())
# Wrap critical loops
with tracer.span("rollout"):
    # ... code ...
tracer.export() # Saves a chrome-tracing compatible JSON
```

---

# Evaluation & "Judgement" Strategy
You asked: *"How can we use extra TPU time to boost single session points?"*

**Answer**: Build a **"Judge" Notebook** to curate your *Next* Training Set.
Don't just run the submission notebook blindly. Use your extra sessions to **filter data**.

### The "Loop" Workflow
1.  **Session 1 (Data Generation)**: 
    *   Load Base Gemma 2B.
    *   Take your huge `Magpie` dataset (300k samples).
    *   Generate answers for 10k of them.
    *   **Filter**: Keep only the ones where the model output correct XML structure AND reasonable length.
    *   **Save**: `filtered_high_quality_magpie.jsonl`.
2.  **Session 2 (The Real Run)**:
    *   Train specifically on `filtered_high_quality_magpie.jsonl`.
    *   **Why**: Training on *clean* data is 10x more effective than training on noisy data.

### Categorized Evaluation Script
To categorize output, use keyword detection or a lightweight classifier (e.g. DistilBERT) to bucket prompts into [Math, Code, Creative].
*   If **Math**: distinct metric = `answer_correctness`.
*   If **Creative**: distinct metric = `length`, `vocabulary_richness`.
Look at the per-category scores to decide where to add more data.
