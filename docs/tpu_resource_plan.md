# TPU Resource & Timeline Strategy (11 Days Left)

**Core Principle**: Secure the **80 Points** (Single Session + Notebook Quality) before chasing the **15 Points** (Unrestricted).

## The Mathematics of Winning
*   **Single Session Checkpoint**: 45 pts
*   **Notebook Documentation**: 35 pts
*   **Video**: 20 pts
*   **Unrestricted Mode**: 15 pts (Bonus)

**Risk**: If you spend all time on Unrestricted and break your Single Session notebook, you lose the core competition.

---

## Period 1: The "Safety Net" (Current Status: Executing)
**Goal**: A submitted, valid Single Session entry.
*   **Activity**: Run `tunix_zero_cost_train.ipynb` **successfully** end-to-end.
*   **Why**: If you run out of time later or hit bugs, you need *something* on the leaderboard.
*   **Evaluation**:
    *   Download `submission.csv` (or output logs).
    *   Run `scripts/evaluation_judge.py` locally on the logs.
    *   Check: Does it *always* output XML?

## Period 2: "Exploration & Unrestricted" (Tomorrow + 6 Days)
**Quota**: 20 Hours.
**Goal**: Push performance (Unrestricted) and learn data quality.

*   **Run 1 (Unrestricted - Leg 1)**: 9 Hours.
    *   Train **SFT-Only** on full Magpie (or larger subset).
    *   Save dataset: `tuned-sft-checkpoint`.
*   **Run 2 (Unrestricted - Leg 2)**: 9 Hours.
    *   Load `tuned-sft-checkpoint`.
    *   Train **GRPO-Only** on GSM8K/MBPP + Private Hard Data.
    *   **Action**: Submit this Model ID for the 15 pts.
*   **Run 3 (Validation)**: 2 Hours (Remainder).
    *   Use this to test hyperparameter tweaks (learning rate, beta) for the Single Session.

**Analysis**: Compare the "Unrestricted" output vs "Safety Net" output using `evaluation_judge.py`.
*   If Unrestricted is MUCH better -> Try to squeeze more epochs into the Single Session config.
*   If Comparison is close -> Your Single Session is efficient!

## Period 3: "The Golden Run" (Final Days)
**Quota**: 20 Hours (Fresh).
**Goal**: The Final Submission & Video.

*   **Run 1 (Final Single Session)**: 9 Hours.
    *   Apply all learnings (best LR, best data mix).
    *   This is your **Main Track Entry**.
*   **Run 2 (Video Demo)**:
    *   Use this run to record the "Live Demo" part of your video.
    *   Generate specific cool examples (e.g. "Solve this complex riddle").

---

## Role of Colab TPU
**Warning**: Colab TPU environment (XLA version, JAX version) often differs from Kaggle.
**Do NOT** train weights on Colab to transfer to Kaggle. It might fail to load.

**USE Colab For**:
1.  **Data Processing**: Run `public_data_engine.py` to generate massive datasets (finding "Hard" examples) -> Upload to Kaggle.
2.  **Code Debugging**: Syntax check `tunix` imports or reward functions.
3.  **Evaluation**: Run `evaluation_judge.py` on outputs.
**Do NOT Use Colab For**:
1.  Final Training Runs.

---

## Summary
1.  **Today**: Lock in a valid Single Session run.
2.  **Week 2**: Win the Unrestricted points & learn what works.
3.  **Final Days**: Re-run the optimized Single Session & Record Video.
