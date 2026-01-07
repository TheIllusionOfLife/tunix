#!/usr/bin/env python3
"""
SFT Notebook Generator for Tunix Competition
Strategy: Supervised Fine-Tuning on diverse domain reasoning traces

Datasets:
- Raiden-DeepSeek-R1 (62.9K creative/analytical)
- OpenO1-SFT (20K general reasoning)
- CoT-Collection (10K commonsense/ethics)
- GlaiveAI-Reasoning (30K math/code/general)
"""

import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # --- Cell 1: Title ---
    title_cell = nbf.v4.new_markdown_cell("""# Tunix SFT: Teaching Reasoning Through Demonstration

**Strategy**: Supervised Fine-Tuning on high-quality reasoning traces across diverse domains.

**Key Insight**: For 2B parameter models, learning from demonstrations is more effective than reinforcement learning. SFT provides dense supervision at every token, while RL provides sparse rewards only at sequence end.

**Datasets**: 
- Raiden-DeepSeek-R1 (Creative/Analytical)
- OpenO1-SFT (General Reasoning)
- CoT-Collection (Commonsense/Ethics)
- GlaiveAI-Reasoning (Math/Code/General)
""")

    # --- Strategy Cell ---
    strategy_cell = nbf.v4.new_markdown_cell("""
## Overall training and evaluation strategy

**Strategy: SFT on Diverse Domain Reasoning Traces**

Competition FAQ explicitly states that verifiable tasks (math/coding) have "much lower weights". Our strategy prioritizes non-verifiable domains:

1.  **Base Model**: We start with `Gemma-2-2b-it` for its instruction-following foundation.
2.  **SFT Training**: We fine-tune on ~100K reasoning traces from diverse domains (creative, analytical, philosophical, commonsense).
3.  **Format**: All data uses explicit `<reasoning>` and `<answer>` tags for structured outputs.

## 🗺️ Workflow Diagram
```mermaid
graph LR
    A[Gemma-2B-IT] --> B{SFT Training}
    B -->|Creative| C[Raiden-DeepSeek-R1]
    B -->|Reasoning| D[OpenO1-SFT]
    B -->|Ethics| E[CoT-Collection]
    B -->|General| F[GlaiveAI]
    C & D & E & F --> G[Trained Model]
    G --> H[Submission]
```
""")

    # --- Dataset Cell ---
    dataset_cell = nbf.v4.new_markdown_cell("""
## How your finetuning dataset is created

We employ a **Diverse Domain Strategy** using publicly available datasets with reasoning traces:

| Dataset | Source | Samples | Domain | License |
|:---|:---|:---:|:---|:---|
| Raiden-DeepSeek-R1 | HuggingFace | 62.9K | Creative/Analytical | Apache 2.0 |
| OpenO1-SFT | HuggingFace | 20K (English-only) | General Reasoning | Apache 2.0 |
| CoT-Collection | HuggingFace | 10K (pre-sampled) | Commonsense/Ethics | CC-BY-4.0 |
| GlaiveAI-Reasoning | HuggingFace | 30K | Non-math/code | Apache 2.0 |

All datasets are pre-processed and attached as parquet files for reproducibility.
""")

    # --- Finetuning Header ---
    finetuning_header = nbf.v4.new_markdown_cell("""## Tunix finetuning code""")

    # --- Variables Cell ---
    vars_cell = nbf.v4.new_code_cell("""
# Training parameters
TEMPERATURE=0.7
TOP_K=50
TOP_P=0.9
MAX_GENERATION_STEPS=768

# Output Tags
REASONING_START = "<reasoning>"
REASONING_END = "</reasoning>"
SOLUTION_START = "<answer>"
SOLUTION_END = "</answer>"

# Inference Params
INF_TEMPERATURE=0
INF_TOP_K=1
INF_TOP_P=None
SEED=42

# System prompt and template
SYSTEM_PROMPT = "You are a deep thinking AI. Think step by step about the problem and provide your reasoning between <reasoning> and </reasoning> tags. Then, provide the final answer between <answer> and </answer> tags."
TEMPLATE = f"<start_of_turn>user\\n{SYSTEM_PROMPT}\\n\\n{{question}}<end_of_turn>\\n<start_of_turn>model"

print("Template variables defined.")
""")

    # --- Setup Cell ---
    setup_cell = nbf.v4.new_code_cell("""
# --- Setup & Install ---
!pip install -q wandb==0.22.0
!pip install -q kagglehub
!pip install -q ipywidgets
!pip install -q tensorflow
!pip install -q tensorflow_datasets
!pip install -q tensorboardX
!pip install -q transformers
!pip install -q grain

# Tunix/Qwix Installation
# Check if we are offline (no internet), if so, assume wheels are attached
import socket
import os

def is_connected():
    try:
        # Check simple connectivity
        socket.create_connection(("1.1.1.1", 53))
        return True
    except OSError:
        pass
    return False

if is_connected():
    !pip install "google-tunix[prod]==0.1.5"
    !pip install git+https://github.com/google/qwix
else:
    print("Offline mode detected. Assuming dependencies are installed or wheels provided.")
    # Fallback: Try installing from local wheels if available
    if os.path.exists("/kaggle/input/tunix-wheels"):
        !pip install --no-index --find-links=/kaggle/input/tunix-wheels google-tunix
        !pip install --no-index --find-links=/kaggle/input/tunix-wheels qwix


# Fix Flax Version to 0.12.0 as required
!pip uninstall -q -y flax
!pip install flax==0.12.0

!pip install -q datasets==3.2.0 optax==0.2.4 chex==0.1.88

# --- Imports ---
import functools
import gc
import os
from pprint import pprint
import re
import csv
import shutil
import time

from flax import nnx
import grain
import humanize
import jax
import jax.numpy as jnp
import kagglehub
import optax
from orbax import checkpoint as ocp
from pathlib import Path
import qwix
import tensorflow_datasets as tfds
import datasets
from tqdm.auto import tqdm
import numpy as np

# Tunix Imports
from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma import model as gemma_lib
from tunix.models.gemma import params as params_lib
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer

# Transformers
from transformers import AutoTokenizer

# --- Stability Configs ---
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

print(f"JAX Devices: {jax.devices()}")

# --- Configuration Constants ---
MODEL_ID = "google/gemma-2-2b-it"
DATASET_PATH = "/kaggle/input/tunix-sft-data"
SFT_OUTPUT_DIR = "/kaggle/working/sft_checkpoint"

# Tuning Hyperparams - Adjust these for HP tuning
SFT_STEPS = 22500  # ~4 epochs with 180K samples, effective batch 32
TRAIN_BATCH_SIZE = 8 # Per-step batch size across all 8 TPU chips (1 sample/chip)
GRADIENT_ACCUMULATION = 4  # Effective batch = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION
EFFECTIVE_BATCH = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION  # 32

# Learning Rate - Key HP for tuning
LEARNING_RATE = 2e-5  # Try: 5e-5, 2e-5, 1e-5
WARMUP_STEPS = 200  # Warmup before reaching peak LR

# LoRA Hyperparams
RANK = 64
ALPHA = 64.0

# Sequence Length
MAX_SEQ_LEN = 2048  # Critical: increased from 1024 to avoid truncating reasoning

# Inference Hyperparams (shared across all evaluations)
INFERENCE_TEMPERATURE = 0.7
INFERENCE_TOP_K = 50
INFERENCE_TOP_P = 0.95
EVAL_MAX_TOKENS = 1024  # Max tokens for eval generation (less than MAX_SEQ_LEN to save memory)
""")

    # --- Model Utilities Cell ---
    model_utils_cell = nbf.v4.new_code_cell("""
# --- Model Utilities ---
MESH = [(8, 1), ("fsdp", "tp")]

def get_gemma_model(ckpt_path):
    mesh = jax.make_mesh(*MESH)
    model_config = gemma_lib.ModelConfig.gemma2_2b()
    abs_gemma: nnx.Module = nnx.eval_shape(
        lambda: gemma_lib.Transformer(model_config, rngs=nnx.Rngs(params=0))
    )
    abs_state = nnx.state(abs_gemma)
    abs_state = jax.tree.map(
        lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
        abs_state,
        nnx.get_named_sharding(abs_state, mesh),
    )
    checkpointer = ocp.StandardCheckpointer()
    restored_params = checkpointer.restore(ckpt_path, target=abs_state)

    graph_def, _ = nnx.split(abs_gemma)
    gemma = nnx.merge(graph_def, restored_params)
    return gemma, mesh, model_config

def get_lora_model(base_model, mesh):
    # LoRA config uses RANK and ALPHA from constants
    lora_provider = qwix.LoraProvider(
        module_path=(
            ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
            ".*attn_vec_einsum"
        ),
        rank=RANK,
        alpha=ALPHA,
    )

    model_input = base_model.get_model_input()
    lora_model = qwix.apply_lora_to_model(
        base_model, lora_provider, rngs=nnx.Rngs(params=0), **model_input
    )

    with mesh:
        state = nnx.state(lora_model)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(lora_model, sharded_state)

    return lora_model
""")

    # --- WandB Cell ---
    wandb_cell = nbf.v4.new_code_cell("""
# --- WandB Logging with Metrics Backend ---
WANDB_ENABLED = False

# Define WandB Backend for MetricsLogger
class WandbBackend:
    '''Custom backend to stream metrics to WandB during training'''
    def log_scalar(self, event: str, value, **kwargs):
        if WANDB_ENABLED:
            step = kwargs.get("step", 0)
            wandb.log({event: float(value)}, step=step)
    def close(self):
        pass

try:
    import wandb
    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()
    secret_value = user_secrets.get_secret("WANDB_API_KEY")

    if secret_value:
        wandb.login(key=secret_value)
        # Log hyperparameters to WandB config
        wandb.init(
            project="tunix-sft-diverse",
            name="sft-run-v2",
            anonymous="allow",
            config={
                "sft_steps": SFT_STEPS,
                "learning_rate": LEARNING_RATE,
                "warmup_steps": WARMUP_STEPS,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "gradient_accumulation": GRADIENT_ACCUMULATION,
                "effective_batch": EFFECTIVE_BATCH,
                "max_seq_len": MAX_SEQ_LEN,
                "lora_rank": RANK,
                "lora_alpha": ALPHA,
                "model_id": MODEL_ID,
            }
        )
        WANDB_ENABLED = True
        print("WandB Logging Enabled with hyperparameter tracking.")
    else:
        raise ValueError("Empty WANDB_API_KEY")

except Exception as e:
    print(f"WandB not enabled: {e}")
    os.environ["WANDB_MODE"] = "disabled"
    print("Proceeding without cloud logging (WANDB_MODE='disabled').")
""")

    # --- Data Preprocessing Cell ---
    data_preprocessing_cell = nbf.v4.new_code_cell("""
# --- Data Preprocessing (GlaiveAI-Only) ---
# Strategy: Single high-quality dataset aligned with competition goals

print("Loading GlaiveAI dataset...")

def standardize_glaive_format(prompt, response):
    '''Convert GlaiveAI <think> format to <reasoning>/<answer> tags'''
    
    # GlaiveAI uses <think>...</think> for reasoning
    text = response
    
    # Replace think tags with reasoning tags
    text = re.sub(r"<think>", "<reasoning>", text, flags=re.IGNORECASE)
    text = re.sub(r"</think>", "</reasoning>", text, flags=re.IGNORECASE)
    
    # Extract reasoning and answer parts
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL | re.IGNORECASE)
    
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
        # Get content after </reasoning> as answer
        remaining = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        
        if remaining:
            answer = remaining
        else:
            # No content after reasoning - use summary
            sentences = reasoning.split(".")
            answer = sentences[-1].strip() if sentences else reasoning[:200]
    else:
        # No think tags - use whole response
        reasoning = text[:500] if len(text) > 500 else text
        answer = text
    
    # Format for Gemma
    formatted = f"<start_of_turn>user\\n{SYSTEM_PROMPT}\\n\\n{prompt}<end_of_turn>\\n<start_of_turn>model\\n<reasoning>{reasoning}</reasoning>\\n<answer>{answer}</answer>"
    return formatted

# Load from Kaggle Dataset (pre-downloaded parquet)
all_texts = []

try:
    if os.path.exists(DATASET_PATH):
        import glob
        parquet_files = glob.glob(f"{DATASET_PATH}/*.parquet")
        
        if parquet_files:
            for parquet_file in parquet_files:
                ds = datasets.load_dataset("parquet", data_files=parquet_file, split="train")
                print(f"Loaded {len(ds)} samples from {os.path.basename(parquet_file)}")
                
                for sample in ds:
                    prompt = sample.get("prompt", "")
                    response = sample.get("response", "")
                    
                    # Filter: Skip very long outputs (>4000 chars ~1K tokens) or empty
                    if len(response) > 4000 or len(response) < 50:
                        continue
                    
                    formatted = standardize_glaive_format(prompt, response)
                    all_texts.append({"text": formatted})
        else:
            raise FileNotFoundError("No parquet files found")
    else:
        raise FileNotFoundError(f"Dataset path {DATASET_PATH} not found")

except Exception as e:
    print(f"Kaggle dataset not found, downloading from HuggingFace...")
    print(f"Warning: This may be slow. Pre-download recommended.")
    
    # Fallback: Stream from HuggingFace
    ds = datasets.load_dataset("glaiveai/reasoning-v1-20m", split="train", streaming=True)
    
    count = 0
    limit = 180000  # Match our target
    
    for sample in ds:
        prompt = sample.get("prompt", "")
        response = sample.get("response", "")
        
        if len(response) > 4000 or len(response) < 50:
            continue
            
        formatted = standardize_glaive_format(prompt, response)
        all_texts.append({"text": formatted})
        
        count += 1
        if count % 10000 == 0:
            print(f"  Downloaded {count} samples...")
        
        if count >= limit:
            break

print(f"Total samples after preprocessing: {len(all_texts)}")

# Create HuggingFace dataset
sft_dataset = datasets.Dataset.from_list(all_texts)
sft_dataset = sft_dataset.shuffle(seed=42)

print(f"Final SFT dataset: {len(sft_dataset)} samples")
print(f"Sample: {sft_dataset[0]['text'][:500]}...")
""")

    # --- Main Training Cell ---
    training_cell = nbf.v4.new_code_cell("""
# --- Main Training Logic ---

# 1. Download/setup Base Model
if "KAGGLE_USERNAME" not in os.environ:
    kagglehub.login()

# Download Gemma 2 (Flax)
model_path = { "gemma2": "google/gemma-2/flax/" }
model_family = "gemma2"
model_version = "gemma2-2b-it" 
kaggle_ckpt_path = kagglehub.model_download(f"{model_path[model_family]}{model_version}")

# Convert checkpoint format for Tunix/NNX
INTERMEDIATE_CKPT_DIR = "/tmp/content/intermediate_ckpt/"
CKPT_DIR = "/tmp/content/ckpts/"
!rm -rf {INTERMEDIATE_CKPT_DIR} {CKPT_DIR}

params = params_lib.load_and_format_params(os.path.join(kaggle_ckpt_path, "gemma2-2b-it"))
gemma = gemma_lib.Transformer.from_params(params, version="2-2b-it")
checkpointer = ocp.StandardCheckpointer()
_, state = nnx.split(gemma)
checkpointer.save(os.path.join(INTERMEDIATE_CKPT_DIR, "state"), state)
checkpointer.wait_until_finished()
del params, gemma, state
gc.collect()

# 2. Load Models
base_model, mesh, model_config = get_gemma_model(os.path.join(INTERMEDIATE_CKPT_DIR, "state"))
lora_model = get_lora_model(base_model, mesh=mesh)

# 3. Setup Tokenizer
tokenizer = tokenizer_lib.Tokenizer(
    tokenizer_path=os.path.join(kaggle_ckpt_path, "tokenizer.model")
)

# 4. Baseline Evaluation (Same prompts as post-training for comparison)
print("Running Baseline Evaluation...")
EVAL_PROMPTS = [
    # Creative writing
    "Write a short story about a robot learning to paint.",
    "Write a haiku about artificial intelligence.",
    # Creative ideation
    "Propose three innovative uses for AI in education.",
    # Summarization
    "Summarize the key benefits and risks of renewable energy in 3 paragraphs.",
    # Math (verifiable)
    "Solve step-by-step: If 2x + 5 = 15, what is x?",
    # Coding (verifiable)
    "Write a Python function to check if a string is a palindrome.",
    # Basic science
    "Explain why the sky is blue to a 5-year-old.",
    "Explain the process of photosynthesis step by step.",
    # Ethics/Reasoning
    "What are the ethical implications of AI in healthcare?",
    "Should AI systems have rights? Argue both sides.",
]

try:
    baseline_sampler = sampler_lib.Sampler(
        transformer=base_model,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=MAX_SEQ_LEN + 512,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )
    formatted = [TEMPLATE.format(question=p) for p in EVAL_PROMPTS]
    baseline_out = baseline_sampler(
        input_strings=formatted,
        max_generation_steps=EVAL_MAX_TOKENS,
        temperature=INFERENCE_TEMPERATURE,
        top_k=INFERENCE_TOP_K,
        top_p=INFERENCE_TOP_P,
        echo=False
    )
    print("--- Baseline Outputs (Before Training) ---")
    baseline_results = []
    for p, o in zip(EVAL_PROMPTS, baseline_out.text):
        print(f"Q: {p}")
        print(f"A: {o}")  # Full output
        has_reasoning = bool(re.search(r"<reasoning>.*?</reasoning>", o, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", o, re.DOTALL))
        baseline_results.append({"prompt": p, "output": o, "has_reasoning": has_reasoning, "has_answer": has_answer})
        print("-"*40)
except Exception as e:
    print(f"Baseline eval skipped: {e}")
    baseline_results = []
print("Baseline Done.")

# 5. SFT Training
print("\\n" + "="*50)
print("Starting SFT Training...")
print("="*50)

# Optimizer - Uses LEARNING_RATE from constants for HP tuning
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    decay_steps=SFT_STEPS,
    end_value=LEARNING_RATE / 20  # End at 5% of peak
)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01)
)

# Checkpointing
# Using Orbax options via TrainingConfig
checkpoint_options = ocp.CheckpointManagerOptions(
    save_interval_steps=500, max_to_keep=2
)

# Data Iterator\nfrom tunix.sft import utils as sft_utils

def create_data_iterator(dataset, batch_size, tokenizer):
    '''Create batches with tokenization and masking'''
    indices = np.random.permutation(len(dataset))
    
    # Infinite iterator matching steps
    while True:
        np.random.shuffle(indices)
        for i in range(0, len(dataset), batch_size):
            batch_indices = indices[i:i+batch_size]
            if len(batch_indices) < batch_size:
                continue # Skip incomplete batches
                
            texts = [dataset[int(idx)]['text'] for idx in batch_indices]
            
            # Tokenize
            # Tunix tokenizer returns list of ids
            batch_input_tokens = []
            batch_input_mask = []
            
            for text in texts:
                # Use Tunix Tokenizer.tokenize which handles BOS/EOS
                # tokenize returns np.array, convert to list for padding
                tokens = tokenizer.tokenize(text, add_eos=True).tolist()
                
                # Truncate / Pad
                if len(tokens) > MAX_SEQ_LEN:
                    tokens = tokens[:MAX_SEQ_LEN]
                    mask = [True] * MAX_SEQ_LEN
                else:
                    pad_len = MAX_SEQ_LEN - len(tokens)
                    mask = [True] * len(tokens) + [False] * pad_len
                    # Use pad_id if available, else 0
                    pad_id = getattr(tokenizer, 'pad_id', lambda: 0)()
                    tokens = tokens + [pad_id] * pad_len # 0 is usually pad, verify if needed
                
                batch_input_tokens.append(tokens)
                batch_input_mask.append(mask)
            
            # Convert to JAX arrays
            input_tokens = jnp.array(batch_input_tokens, dtype=jnp.int32)
            input_mask = jnp.array(batch_input_mask, dtype=jnp.bool_)
            
            # Create PEFT required inputs
            positions = sft_utils.build_positions_from_mask(input_mask)
            attention_mask = sft_utils.make_causal_attn_mask(input_mask)
            
            yield {
                "input_tokens": input_tokens,
                "input_mask": input_mask,
                "positions": positions,
                "attention_mask": attention_mask
            }

# Training Configuration with WandB Metrics Backend
from tunix.sft import metrics_logger as sft_metrics_logger

metrics_logging_options = sft_metrics_logger.MetricsLoggerOptions(
    log_dir="/kaggle/working/logs",
    backend_factories=[WandbBackend] if WANDB_ENABLED else []
)

training_config = peft_trainer.TrainingConfig(
    max_steps=SFT_STEPS,
    checkpoint_root_directory=SFT_OUTPUT_DIR,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    checkpointing_options=checkpoint_options,
    pbar_description="SFT Training",
    metrics_prefix="sft",
    metrics_logging_options=metrics_logging_options,
    eval_every_n_steps=10000, # Disable freq eval for speed or set high
)

# Initialize Trainer
# Note: we pass the optimizer, model, and config.
# Metrics logger defaults are fine.
trainer = peft_trainer.PeftTrainer(
    model=lora_model,
    optimizer=optimizer,
    training_config=training_config
)

# Create Iterator
train_iter = create_data_iterator(sft_dataset, TRAIN_BATCH_SIZE, tokenizer)

print(f"Starting Training for {SFT_STEPS} steps...")
with mesh:
    trainer.train(train_ds=train_iter, skip_jit=False)

print("SFT Training Completed.")
""")

    # --- Save Model Cell ---
    save_model_cell = nbf.v4.new_code_cell("""
# --- Save Final Model ---
FINAL_SAVE_DIR = "/kaggle/working/final_sft_model"
os.makedirs(FINAL_SAVE_DIR, exist_ok=True)

# Save the trained LoRA model checkpoint
checkpointer = ocp.StandardCheckpointer()
checkpointer.save(os.path.join(FINAL_SAVE_DIR, "checkpoint"), nnx.state(lora_model, nnx.LoRAParam))
checkpointer.wait_until_finished()

print(f"✅ Model saved to '{FINAL_SAVE_DIR}/'")
print("To submit for Unrestricted Mode:")
print("   1. Download the output folder after this notebook finishes.")
print("   2. Go to Kaggle -> Models -> New Model -> Upload the checkpoint files.")
print("   3. Set the Model ID below to match your upload.")

# Your Kaggle Model ID for Unrestricted Mode:
unrestricted_kaggle_model = "yuyamukai/tunix-gemma2-sft"
""")

    # --- Visual Evaluation Cell ---
    visual_eval_cell = nbf.v4.new_code_cell("""
# --- Visual Sanity Check & Validation ---
print("Running Post-Training Evaluation...")

try:
    inference_sampler = sampler_lib.Sampler(
        transformer=lora_model,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=MAX_SEQ_LEN + 512,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )

    test_prompts = [
        # Creative writing
        \"Write a short story about a robot learning to paint.\",
        \"Write a haiku about artificial intelligence.\",
        # Creative ideation
        \"Propose three innovative uses for AI in education.\",
        # Summarization
        \"Summarize the key benefits and risks of renewable energy in 3 paragraphs.\",
        # Math (verifiable)
        \"Solve step-by-step: If 2x + 5 = 15, what is x?\",
        # Coding (verifiable)
        \"Write a Python function to check if a string is a palindrome.\",
        # Basic science
        \"Explain why the sky is blue to a 5-year-old.\",
        \"Explain the process of photosynthesis step by step.\",
        # Ethics/Reasoning
        \"What are the ethical implications of AI in healthcare?\",
        \"Should AI systems have rights? Argue both sides.\",
    ]
    
    # Use same prompts as baseline for fair comparison
    test_prompts = EVAL_PROMPTS
    formatted_prompts = [TEMPLATE.format(question=p) for p in test_prompts]
    
    out_data = inference_sampler(
        input_strings=formatted_prompts,
        max_generation_steps=EVAL_MAX_TOKENS,
        temperature=INFERENCE_TEMPERATURE,
        top_k=INFERENCE_TOP_K,
        top_p=INFERENCE_TOP_P,
        echo=False
    )
    
    # Validation Logic
    print(\"--- Post-Training Outputs ---\")
    valid_format_count = 0
    results_for_wandb = []
    
    for p, o in zip(test_prompts, out_data.text):
        print(f\"Prompt: {p}\")
        print(f\"Output: {o}\")  # Full output, no truncation
        
        # Robust Regex Check
        has_reasoning = bool(re.search(r"<reasoning>.*?</reasoning>", o, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", o, re.DOTALL))
        
        is_valid = has_reasoning and has_answer
        if is_valid:
            valid_format_count += 1
            print("✅ Format Check: Passed")
        else:
            print(f"❌ Format Check: Failed (Reasoning: {has_reasoning}, Answer: {has_answer})")
            
        results_for_wandb.append([p, o, is_valid])
        print("-" * 50)
    
    print(f"Format Validation: {valid_format_count}/{len(test_prompts)} passed.")
    
    # Extended WandB Evaluation (25 prompts for statistical confidence)
    WANDB_EVAL_PROMPTS = [
        # Original 10 prompts
        *test_prompts,
        # Additional 15 prompts for diversity
        "Explain quantum entanglement to a high school student.",
        "Write a poem about the passage of time.",
        "What are the pros and cons of remote work?",
        "Describe how a compiler works step by step.",
        "Compare democracy and authoritarianism objectively.",
        "Write a short dialogue between a human and an AI about consciousness.",
        "Explain the greenhouse effect and its consequences.",
        "How would you teach a child about money management?",
        "What lessons can we learn from the fall of ancient Rome?",
        "Design a simple mobile app for tracking habits.",
        "Explain the difference between correlation and causation.",
        "Write a persuasive argument for learning a second language.",
        "How do vaccines work to protect against diseases?",
        "What ethical considerations arise with genetic engineering?",
        "Explain the concept of supply and demand with examples.",
    ]
    
    # Run extended evaluation for WandB
    try:
        if wandb.run is not None and WANDB_ENABLED:
            print("\\nRunning Extended WandB Evaluation (25 prompts in batches)...")
            extended_results = []
            extended_valid = 0
            BATCH_SIZE = 5  # Process in smaller batches to avoid OOM
            
            for batch_start in range(0, len(WANDB_EVAL_PROMPTS), BATCH_SIZE):
                batch_prompts = WANDB_EVAL_PROMPTS[batch_start:batch_start + BATCH_SIZE]
                batch_formatted = [TEMPLATE.format(question=p) for p in batch_prompts]
                batch_out = inference_sampler(
                    input_strings=batch_formatted,
                    max_generation_steps=EVAL_MAX_TOKENS,
                    temperature=INFERENCE_TEMPERATURE,
                    top_k=INFERENCE_TOP_K,
                    top_p=INFERENCE_TOP_P,
                    echo=False
                )
                
                for p, o in zip(batch_prompts, batch_out.text):
                    has_r = bool(re.search(r"<reasoning>.*?</reasoning>", o, re.DOTALL))
                    has_a = bool(re.search(r"<answer>.*?</answer>", o, re.DOTALL))
                    is_valid = has_r and has_a
                    if is_valid:
                        extended_valid += 1
                    extended_results.append([p, o[:1000], is_valid])  # Truncate for table
                print(f\"  Batch {batch_start//BATCH_SIZE + 1}/{(len(WANDB_EVAL_PROMPTS) + BATCH_SIZE - 1)//BATCH_SIZE} complete.\")
            
            # Log table
            tbl = wandb.Table(columns=["Prompt", "Output", "IsValid"], data=extended_results)
            wandb.log({"eval_results": tbl})
            
            # Log summary metrics
            format_compliance = extended_valid / len(WANDB_EVAL_PROMPTS) * 100
            wandb.log({
                "eval/format_compliance_pct": format_compliance,
                "eval/total_prompts": len(WANDB_EVAL_PROMPTS),
                "eval/valid_count": extended_valid,
            })
            print(f"Extended Evaluation: {extended_valid}/{len(WANDB_EVAL_PROMPTS)} ({format_compliance:.1f}%) passed.")
            print("Logged to WandB: eval_results table + summary metrics.")
    except Exception as w_err:
        print(f"Extended WandB eval skipped: {w_err}")

except Exception as e:
    print(f"Evaluation failed: {e}")
""")

    # --- Unrestricted Mode Cell ---
    unrestricted_header = nbf.v4.new_markdown_cell("""## [Optional 15pts] unrestricted mode""")
    unrestricted_code = nbf.v4.new_code_cell("""
# For Unrestricted Mode, upload the saved checkpoint as a Kaggle Model.
# Then update this variable with your Model ID:
unrestricted_kaggle_model = "yuyamukai/tunix-gemma2-sft"

print(f"Unrestricted Mode Model ID: {unrestricted_kaggle_model}")
""")

    # --- Other Info Cell ---
    other_info = nbf.v4.new_markdown_cell("""
## Other things I want the judges to know

### 1. Why GlaiveAI-Only
*   **Quality Over Quantity**: One 2025 dataset (DeepSeek-R1-Distill-70B) beats multiple older, mixed-quality sources.
*   **Competition Alignment**: GlaiveAI focuses on non-math/code (social science, creative writing) - exactly what FAQ says matters most.
*   **Consistency**: Single source = no format standardization issues across datasets.

### 2. Dataset Dropped (and Why)
*   ❌ CoT-Collection - Created 2023, outdated model quality
*   ❌ Raiden-DeepSeek-R1 - Unfiltered R1 outputs, infinite loops
*   ❌ OpenO1-SFT - Math/code focus, misaligned with competition

### 3. Data Source
*   ✅ glaiveai/reasoning-v1-20m (180K samples, Apache 2.0)
*   DeepSeek-R1-Distill-Llama-70B reasoning traces
*   Non-math/code focus: social science, creative writing, analytical reasoning

### 4. Training Config
*   **Steps**: 22,500 (~4 epochs)
*   **Runtime**: ~7 hours
*   **Method**: LoRA (efficient parameter updates)
""")

    # Assemble notebook
    nb.cells = [
        title_cell,
        strategy_cell,
        dataset_cell,
        finetuning_header,
        vars_cell,
        setup_cell,
        model_utils_cell,
        wandb_cell,
        data_preprocessing_cell,
        training_cell,
        save_model_cell,
        visual_eval_cell,
        unrestricted_header,
        unrestricted_code,
        other_info
    ]

    with open('tunix_sft_train.ipynb', 'w') as f:
        nbf.write(nb, f)
    
    print("Notebook 'tunix_sft_train.ipynb' created successfully.")

if __name__ == "__main__":
    create_notebook()
