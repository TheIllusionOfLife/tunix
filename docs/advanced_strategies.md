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
**Cycle**:
1.  **Generate**: Model generates 10 answers per prompt.
2.  **Filter**: Use a "Judge" (e.g., Code Executor or Rubric) to find the single best answer.
3.  **Retrain**: Treat that best answer as the new "Ground Truth" for SFT.
4.  **Repeat**.

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
