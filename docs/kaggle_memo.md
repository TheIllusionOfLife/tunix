
# Kaggle Submission Memo

## 1. Prepare Data
1.  **Run Locally**: execute `python scripts/public_data_engine.py` (if not already done) to generate:
    *   `data/sft_magpie.jsonl`
    *   `data/grpo_gsm8k_train.jsonl`
2.  **Upload to Kaggle**:
    *   Go to Kaggle -> Datasets -> New Dataset.
    *   Title: `tunix-public-data`
    *   Upload the `data/` folder containing the `.jsonl` files.
    *   **Note the path**: It will likely be `/kaggle/input/tunix-public-data`. Update the `DATASET_PATH` variable in the notebook if different.

## 2. Prepare Notebook
1.  **Upload Notebook**:
    *   Go to Kaggle -> Code -> New Notebook.
    *   File -> Import Notebook -> Upload `tunix_zero_cost_train.ipynb`.
2.  **Configure Accelerator**:
    *   Session Options -> Accelerator -> **TPU VM v5e-8** (Required for Tunix).
    *   Session Options -> Persistence -> Files only (optional, good for saving checkpoints).

## 3. Run Training
1.  **Execute All Cells**:
    *   The notebook installs `tunix` from the repository (or you can add the repo as a dataset).
    *   It trains SFT on Magpie data (Format).
    *   It trains GRPO on GSM8K data (Math Correctness).
    *   It saves the final model.

## 4. Submission
1.  **Save Version**: Click "Save Version" -> "Save & Run All (Commit)".
    *   This ensures the notebook runs end-to-end within the 9-hour limit.
2.  **Submit**:
    *   Go to the competition page -> Submit.
    *   Select the notebook you just ran.

## Tips
*   **Debug Mode**: If debugging, reduce `SFT_STEPS` and `GRPO_STEPS` to 10 to check if it runs through quickly.
*   **Tunix Install**: If `!pip install git+...` fails due to internet off during submission, you must upload the `tunix` wheel or repo as a dataset and install from there.
