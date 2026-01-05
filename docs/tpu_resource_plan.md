# TPU Resource Plan (SFT Strategy)

## Hardware Overview

| Resource | Specification |
|:---|:---|
| TPU Type | v5e-8 |
| HBM per core | 16 GB |
| Cores | 8 |
| Session limit | 9 hours |
| Weekly limit | 20 hours |

---

## SFT vs GRPO Resource Usage

| Metric | GRPO | SFT |
|:---|:---|:---|
| **Samples/hour** | ~150-200 | ~10,000-15,000 |
| **Memory/sample** | High (4x generations) | Low (1x forward+backward) |
| **Total in 9hr** | ~1,500 steps | ~100,000 samples |

SFT is dramatically more efficient for sample throughput.

---

## Session 1 Timeline (9 hours)

| Time | Activity | Notes |
|:---|:---|:---|
| 0:00-0:30 | Setup & imports | Install Tunix, load model |
| 0:30-1:00 | Data preprocessing | Load & format 100K samples |
| 1:00-8:00 | SFT Training | ~100K samples, 2-3 epochs |
| 8:00-8:30 | Checkpoint save | Save to output dir |
| 8:30-9:00 | Inference test | Generate sample outputs |

---

## Memory Optimization

### Batch Size Guidelines

| Model | LoRA | Full Weights |
|:---|:---:|:---:|
| Gemma-2B | 4-8 | 2-4 |
| Gemma-2B-IT | 4-8 | 2-4 |

### Sequence Length

| Max Length | Memory | Use Case |
|:---|:---|:---|
| 512 | Low | Short reasoning |
| 1024 | Medium | Standard |
| 2048 | High | Long traces |

**Recommendation**: Start at 1024, increase if stable.

---

## Unrestricted Mode Sessions

### Weekly Budget (20 hours)

| Session | Duration | Cumulative |
|:---|:---:|:---:|
| 1 (Main) | 9h | 9h |
| 2 | 8h | 17h |
| 3 | 3h | 20h |

### Session 2-3 Strategy

- **Session 2**: Load checkpoint, continue SFT on glaiveai data
- **Session 3**: Either more SFT or GRPO polish

---

## Checkpointing Strategy

Save checkpoints at:
- Every 1000 steps (during training)
- End of session (for continuation)

```python
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=1000,
    max_to_keep=3,
)
```

---

## Contingency Plans

| Issue | Solution |
|:---|:---|
| OOM during training | Reduce batch size or seq length |
| Session timeout | Checkpoint auto-saves every 1000 steps |
| Data preprocessing slow | Pre-upload processed data |
| Model diverges | Reduce learning rate, add warmup |
