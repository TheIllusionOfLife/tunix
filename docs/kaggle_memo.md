# Kaggle Submission Memo (SFT Strategy)

## 1. Prepare Data

### Pre-upload Kaggle Dataset
1. Download raw data from HuggingFace locally
2. Keep original format (no preprocessing)
3. Upload to Kaggle Datasets with source documentation
4. Attach dataset to notebook

### Data Sources Documentation

Include a `DATA_SOURCES.md` in your Kaggle dataset:

```markdown
# Data Sources

| Dataset | Source | License |
|:---|:---|:---|
| Raiden-DeepSeek-R1 | huggingface.co/datasets/sequelbox/Raiden-DeepSeek-R1 | Apache 2.0 |
| OpenO1-SFT | huggingface.co/datasets/O1-OPEN/OpenO1-SFT | Apache 2.0 |
| General_Inquiry_Thinking | huggingface.co/datasets/moremilk/General_Inquiry_Thinking-Chain-Of-Thought | MIT |
| CoT-Collection | huggingface.co/datasets/pharaouk/CoT-Collection | CC-BY-4.0 |
| glaiveai/reasoning-v1-20m | huggingface.co/datasets/glaiveai/reasoning-v1-20m | Apache 2.0 |
```

---

## 2. Prepare Notebook

1. Upload `tunix_sft_train.ipynb`
2. Configure accelerator: **TPU VM v5e-8**
3. Attach: Data Kaggle Dataset
4. Enable persistence for checkpoints

---

## 3. Run Training

Notebook will:
1. Install Tunix and dependencies
2. Load & preprocess datasets
3. Initialize Gemma 2B-IT with LoRA (or full weights)
4. Run SFT for 2-3 epochs
5. Save final checkpoint

---

## 4. Submission

1. **Save Version**: "Save & Run All (Commit)"
2. **Verify**: Check output has checkpoint files
3. **Submit**: Link notebook on competition page

---

## Debugging Tips

- **Debug mode**: Set `MAX_STEPS=100` to quick test
- **Memory issues**: Reduce batch size or sequence length
- **Data errors**: Check preprocessing logs carefully
