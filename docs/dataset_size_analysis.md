# Dataset Size Analysis (Postmortem)

This document records the analysis and conclusions after a regression in model quality following a dataset strategy change. The regression was observed in `logs/2026-01-08-10-29-00.log` compared to `logs/2026-01-07-19-21-00.log`.

## Summary of What Went Wrong

The new run used a **GlaiveAI-only** dataset and an aggressive length filter:

- Filter rule: `len(response) > 4000` or `< 50` was excluded.
- GlaiveAI responses are typically much longer than 4,000 characters.
- Result: **~99% of samples were dropped**, leaving only **1,786 samples**.
- Training still ran **22,500 steps**, which is hundreds of epochs over a tiny dataset.
- Outcome: **format failures, incomplete answers, and quality regression**.

This is a classic overfitting failure caused by a **dataset collapse** due to over-aggressive filtering.

## Evidence from Logs

**Bad run (new):**
- `Final SFT dataset: 1786 samples`
- `> **Historical Note**: References to "22500 steps" below reflect early fixed-step planning. We now use dynamic steps.
> Starting Training for ~22500 steps...`
- `Format Validation: 7/10 passed`
  (`logs/2026-01-08-10-29-00.log`)

**Baseline run (old):**
- `Final SFT dataset: 122925 samples`
- `Starting Training for 8000 steps`
- `Format Validation: 9/10 passed`
  (`logs/2026-01-07-19-21-00.log`)

## Length Distribution (Full Files)

Script used: `scripts/dataset_distribution.py`

Full-file percentiles (response length in characters):

### GlaiveAI

| File | p1 | p5 | p10 | p50 | p90 | p95 | p96 | p97 | p98 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `glaiveai_90k_part1.parquet` | 4,020 | 4,880 | 5,343 | 7,078 | 9,398 | 10,362 | 10,694 | 11,227 | 12,181 | 15,731 |
| `glaiveai_90k_part2.parquet` | 3,993 | 4,871 | 5,332 | 7,086 | 9,420 | 10,376 | 10,717 | 11,260 | 12,253 | 16,160 |
| `glaiveai_continuation_100k.parquet` | 3,969 | 4,849 | 5,323 | 7,074 | 9,426 | 10,428 | 10,810 | 11,393 | 12,547 | 16,799 |

Observations:
- p95 is tightly clustered around **10.3k–10.4k**.
- p98 is around **12.2k–12.5k**.
- p99 reaches **15.7k–16.8k**.

### Archive Datasets (for reference)

| File (Column) | p1 | p5 | p10 | p50 | p90 | p95 | p96 | p97 | p98 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `archive/data/cot_collection_10k.parquet` (`target`) | 1 | 1 | 1 | 6 | 38 | 63 | 72 | 85 | 106 | 138 |
| `archive/data/cot_collection_10k.parquet` (`rationale`) | 32 | 63 | 84 | 238 | 563 | 722 | 755 | 783 | 818 | 867 |
| `archive/data/openo1_sft_english_20k.parquet` (`output`) | 558 | 1,518 | 2,012 | 4,120 | 7,575 | 8,920 | 9,342 | 9,886 | 10,617 | 12,173 |
| `archive/data/raiden_deepseek_r1.parquet` (`response`) | 2,854 | 4,488 | 5,908 | 9,769 | 22,547 | 28,791 | 30,352 | 32,119 | 34,163 | 37,339 |

## Conclusion

Filtering at **4,000 characters** is far too aggressive for GlaiveAI and collapses the dataset.

The safest approach is to:
- Keep GlaiveAI-only (as per `data/DATA_SOURCES.md`)
- Drop only the **longest tail** of samples (top 5% by length)
- Add a **cheap guardrail** so we never collapse the dataset again

## Agreed Filtering Policy (Option 2 + Guardrail)

We will **not compute percentiles at runtime** (too expensive), and instead use fixed thresholds with a guardrail:

**Definition (kept_ratio)**:
```
kept_ratio = kept_samples / total_samples_seen
```

1. Start with **10.4k** char cutoff (approx p95 for GlaiveAI).
2. If `kept_ratio < 0.8`, raise to **12.5k** (approx p98).
3. If `kept_ratio < 0.8`, raise to **15k** (approx p99).
4. If `kept_ratio < 0.8`, **disable filtering** for this run.

This keeps runtime cheap while preventing another dataset collapse.

## Notes

- This policy aligns with the competition expectation that output length should be <1k tokens **without destroying the dataset**.
- The main failure mode was not GlaiveAI itself, but the combination of **aggressive filtering + heavy epochs on a tiny dataset**.
