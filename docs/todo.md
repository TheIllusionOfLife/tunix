# Tunix Zero-Cost Challenge: Master TODO

This file orchestrates the execution plan across all specific documentation. Use this as your command center.

## 📚 Documentation Index
## 📚 Documentation Index
*   **Strategy**: [Winning Assessment (Max Score)](scoring_breakdown.md) | [Advanced Strategies](advanced_strategies.md)
*   **Execution**: [Kaggle Run Memo](kaggle_memo.md) | [TPU Resource Plan](tpu_resource_plan.md)
*   **Submission**: [Unrestricted Guide](unrestricted_mode_guide.md) | [Writeup Content](writeup_content.md) | [Video Script](video_script.md)

---

## ✅ Phase 1: The Safety Net (Current)
*Status: Executing on Kaggle (TPU v5e-8)*
*> Competitive Edge: XML Structure Reward ensures >95% parse rate (vs. naive models failing format).*

- [ ] **Monitor Run**: Ensure `tunix_zero_cost_train.ipynb` completes (9h limit).
- [ ] **Quality Gate**: Verify `submission.csv` outputs STRICT `<reasoning>` tags. (If <90% valid, we fail).
- [ ] **Create Dataset**: Save output as `tunix-session1-checkpoint`.

## 🧠 Phase 2: Advanced Optimization (The "Veteran" Move)
*Goal: Outperform on throughput and reasoning depth.*
*> Competitive Edge: SGLang = 2x Training Steps. Agentic = Self-Correction.*

- [ ] **Domain Coverage**: Add `UltraFeedback` or `HelpSteer` to training mix.
    *   *Why*: Competition evaluates "Creative Writing" & "Summarization". Current model is Math/Code only.
- [ ] **Agentic Upgrade**: Swap `GRPOLearner` for `tunix.rl.experimental.agentic_grpo_learner.AgenticGRPOLearner`.
- [ ] **Speed Upgrade**: Enable `SGLang` sampler for faster rollouts.
- [ ] **Profiling**: Trigger `PerfTracer` to prove efficiency in Notebook Writeup.

## 🚀 Phase 3: Unrestricted Mode (The "Whale" Strategy)
*Goal: Use "Private Data" to simulate a 70B teacher.*
*> Competitive Edge: Training on "Hardest 5%" of Magpie data (Quality > Quantity).*

- [x] **Data Curation**: Upload "Private Hard" dataset (Done: `private_hard_reasoning`).
- [ ] **Chain Sessions**: Train Phase 3 model ON TOP of Phase 2 model.
- [ ] **Publish Model**: `yuyamukai/tunix-unrestricted-final`.
- [ ] **Update Notebook**:
    - [ ] Set `unrestricted_kaggle_model`.
    - [ ] Writeup: Explain "Curriculum Learning" (Easy -> Hard).

## 🎬 Phase 4: Final Submission
*Goal: Maximize Subjective Points (55pts Total).*

- [ ] **Agentic Verification**: Use `scripts/evaluation_judge.py` (or ask Antigravity) to audit 10 random samples.
- [ ] **Notebook Polish**: Add ASCII flowcharts to `tunix_zero_cost_train.ipynb` intro.
- [ ] **Video**: Focus on the *Failure vs Success* demo (Before/After).
