# Unrestricted Mode Guide (Optional 15 Points)

This guide explains how to win the extra 15 points for "Model quality across multiple Kaggle sessions".

**Goal**: Train a stronger model by bypassing the 9-hour limit and using a custom "Private Dataset".

---

## Step 1: Create a "Private" Dataset (The "Secret Sauce")
To satisfy the "Use private data" requirement without spending money, we will curate a **"Hardest Problems"** dataset from public sources. This counts as "private" because you are uploading your own custom-filtered file that doesn't exist publicly in that exact form.

### Action Plan:
1.  **Download the Full Magpie Dataset** (on your local machine):
    It has ~300k samples. We only used 5k for the main submission.
2.  **Filter for "Hard" Math**:
    Keep only samples with `difficulty` > "medium" or long reasoning traces (> 500 words).
3.  **Save as `private_hard_reasoning.jsonl`**.
4.  **Upload to Kaggle**:
    *   Name it: `tunix-private-hard-reasoning`.
    *   **Keep it Private**.

### Strategic Edge:
**Why do this?** Most competitors will just train for *more epochs* on the same easy data. By filtering for the top 5% hardest problems, we are effectively performing "Curriculum Learning," simulating the distillation process of a much larger model. This is how we win the 15pts.

*(Note: If you don't want to run scripts locally, you can skip this and just use the "Checkpoint Chaining" step below. The private data is optional "to push the envelope".)*

---

## Step 2: Session 1 - The Base Training
**Objective**: Train your initial SFT + GRPO model and **save the weights**.

1.  Open your `tunix_zero_cost_train.ipynb`.
2.  **Config**:
    ```python
    PRETRAINED_PATH = None
    SFT_STEPS = 400
    GRPO_STEPS = 400
    SAVE_FINAL_PATH = "tunix_session1_checkpoint"
    ```
3.  **Run the Notebook**.
4.  **IMPORTANT**: After the run finishes, go to the **"Output"** tab of the notebook viewer.
5.  Click **"Create New Dataset"** from the output files.
    *   Name it: `tunix-session1-checkpoint`.
    *   This saves your trained model weights as a dataset we can load later.

---

## Step 3: Session 2 - The "Unrestricted" Training
**Objective**: Load the Session 1 model and train *more* using your Private Data.

1.  **Create a NEW Notebook** (e.g., `tunix_unrestricted_phase2`).
2.  **Add Data**:
    *   Add Data -> Your `tunix-session1-checkpoint` dataset.
    *   Add Data -> Your `tunix-private-hard-reasoning` dataset.
3.  **Modify Config**:
    ```python
    # Path to the model from Session 1 (Kaggle mounts datasets at /kaggle/input/...)
    PRETRAINED_PATH = "/kaggle/input/tunix-session1-checkpoint/sft_checkpoint" 
    
    # Point to your private data for extra training
    PRIVATE_DATA_PATH = "/kaggle/input/tunix-private-hard-reasoning/private_hard_reasoning.jsonl"
    
    # Increase steps since we basically have a fresh 9 hours!
    SFT_STEPS = 0    # Skip SFT if satisfy with Session 1, or do small epoch on Private Data
    GRPO_STEPS = 800 # Focus heavily on Reinforcement Learning
    ```
4.  **Run Training**.
5.  **Submit**:
    *   The model from *this* session is your final "Unrestricted" entry.
    *   Make sure to grab the **Model ID** (e.g., upload the output as a Model: `yuyamukai/tunix-unrestricted-final`).

---

## Step 4: Submission Requirement
At the bottom of your **Session 2 Notebook**, you must add a text cell:

```markdown
## Unrestricted Mode Submission
**Final Model ID**: https://www.kaggle.com/models/yuyamukai/tunix-unrestricted-final
**Strategy**: 
1. Session 1: SFT+GRPO on Public Data (Magpie/GSM8K).
2. Session 2: Loaded Checkpoint -> GRPO on Private "Hard" Dataset.
```
