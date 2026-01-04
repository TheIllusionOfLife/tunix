# Max-Score Winning Assessment

Goal: Achieve the highest possible points in every section by surpassing the "Naive Baseline" of typical Kaggle entries.

---

## 1. Notebook Quality (35 Points)
*Criteria: Clear writing, Training data, Hyperparams, Prompt, Strategy.*

| Metric | Naive Baseline (Average Score) | **Max Score Strategy (Our Edge)** |
| :--- | :--- | :--- |
| **Strategy** | "I trained a model." | **Narrative Diagram**: Show the "Format-Align-Reinforce" pipeline visually (ASCII Flowchart). Explain *why* 2B needs SFT (Format) before RL (Reasoning). |
| **Hyperparams** | Lists them. | **Justification**: "We selected `beta=0.08` to balance exploration vs mode collapse in creative tasks, based on [Reference Paper/Logic]." |
| **Reproducibility** | "Run all cells." | **Pinned Stability**: Explicit `google-tunix[prod]==0.1.5` install command. "We fixed version drift to guarantee this runs for judges." |

**Action Item**: Add ASCII Workflow Diagram to Notebook Intro in Phase 4.

---

## 2. Model Quality - Single Session (45 Points)
*Criteria: 9h Limit, XML Format, Reasoning Trace, Correctness, Domain Diversity (Math, Code, **Creative**, **Summarization**).*

| Metric | Naive Baseline (Average Score) | **Max Score Strategy (Our Edge)** |
| :--- | :--- | :--- |
| **XML Output** | Fails parsing 30% of time. | **Reward Enforcement**: Our `structure_reward` guarantees >95% parse rate. We survive the automated filter. |
| **Reasoning** | Tries to reason on everything. | **Domain Awareness**: Model learns (via UltraFeedback) that Creative tasks need *less* rigid structure than Math. |
| **Creativity** | **FAIL**: Fails "Write a poem" prompts because it's overfit to GSM8K. | **The "Creative Patch"**: Phase 2 explicitly adds `UltraFeedback` to ensure the model doesn't become a "Math Zombie". This is the difference between 30pts and 45pts. |
| **Throughput** | 500 GRPO Steps (Slow JAX sampler). | **SGLang Speedup**: Using `v0.1.4` SGLang sampler allows **2-3x steps** (1500+). More steps = Better Convergence within 9h. |
| **Depth** | Single-turn guessing. | **Agentic Reasoning**: (If enabled) Evaluation isn't just one guess; the model "thinks" then "verifies". We enable `AgenticGRPOLearner` for harder math. |

**Action Item**: Execute **Phase 2 (Domain Coverage)** task immediately to fix the Creative Gap.

---

## 3. Video Quality (20 Points)
*Criteria: Under 3 min, Instructional, High Production.*

| Metric | Naive Baseline (Average Score) | **Max Score Strategy (Our Edge)** |
| :--- | :--- | :--- |
| **Content** | Screen recording of code. | **Storytelling**: "The Hero's Journey". Show the Base Model failing (The Villain) -> Our Code (The Weapon) -> The Fine-Tuned Model Winning (The Hero). |
| **Visuals** | Static text. | **Dynamic Comparisons**: Split-screen showing "Before vs After" generation speed and quality. |

**Action Item**: Script the "Failure Demo" scenes now.

---

## 4. Unrestricted Mode (15 Points - Bonus)
*Criteria: Multi-session, Private Data.*

| Metric | Naive Baseline (Average Score) | **Max Score Strategy (Our Edge)** |
| :--- | :--- | :--- |
| **Strategy** | "I trained for 9h more." | **Curriculum Distillation**: "We used the first 9h for Basics, and the second 9h exclusively for the **Hardest 5%** of problems." |
| **Data** | Same dataset, more epochs. | **Private Hard Data**: Simulating a 70B teacher by filtering only the complex reasoning traces. |

**Action Item**: Execute **Phase 3** filtering script.

---
