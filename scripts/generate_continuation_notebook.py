#!/usr/bin/env python3
"""
SFT Continuation Notebook Generator for Tunix Competition
Strategy: Unrestricted Mode - Extended SFT on GlaiveAI

Usage:
    python scripts/generate_continuation_notebook.py
"""

import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # --- Cell 1: Title ---
    title_cell = nbf.v4.new_markdown_cell("""# Tunix SFT: Continuation Training (Unrestricted Mode)

**Strategy**: Continue Supervised Fine-Tuning on a larger dataset (GlaiveAI) starting from the Session 1 checkpoint.

**Prerequisites**:
1. Run Session 1 notebook (`tunix_sft_train.ipynb`) and save output.
2. Upload the Session 1 output as a Kaggle Dataset (e.g., `tunix-session1-checkpoint`).
3. Attach that dataset to this notebook.
""")

    # --- Configuration Cell ---
    config_cell = nbf.v4.new_code_cell("""
# --- Configuration ---
# Update these paths based on your Kaggle Dataset names

# Path to checkpoint from Session 1 (Uploaded as Dataset)
PREV_CHECKPOINT_PATH = "/kaggle/input/tunix-session1-checkpoint/final_sft_model/checkpoint"

# Path to continuation training data (Fresh GlaiveAI samples)
CONTINUATION_DATA_PATH = "/kaggle/input/tunix-sft-continuation-data"
# Note: Adaptive filtering (10.4k->15k) is applied during loading to ensure >80% retention.

# Training Hyperparams - Adjust for HP tuning
# Training Hyperparams - Adjust for HP tuning
# SFT_STEPS is now dynamic
# SFT_STEPS = 5000 (Removed)
LEARNING_RATE = 5e-6  # Lower LR for continuation (try: 1e-5, 5e-6, 2e-6)
WARMUP_STEPS = 100  # Warmup steps
TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 16
EFFECTIVE_BATCH = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION  # 32
MAX_SEQ_LEN = 2048

# LoRA Hyperparams (must match session 1)
RANK = 64
ALPHA = 64.0

# System Prompt (must match session 1)
SYSTEM_PROMPT = "You are a deep thinking AI. Think step by step about the problem and provide your reasoning between <reasoning> and </reasoning> tags. Then, provide the final answer between <answer> and </answer> tags."
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
import socket
import os

def is_connected():
    try:
        socket.create_connection(("1.1.1.1", 53))
        return True
    except OSError:
        return False

if is_connected():
    !pip install "google-tunix[prod]==0.1.5"
    !pip install git+https://github.com/google/qwix
else:
    print("Offline mode detected. Assuming legacy installation or wheels.")


# Fix Flax Version
!pip uninstall -q -y flax
!pip install flax==0.12.0
!pip install -q datasets==3.2.0 optax==0.2.4 chex>=0.1.90

# --- Imports ---
import functools
import gc
import os
import re
import time
from flax import nnx
import grain
import jax
import jax.numpy as jnp
import kagglehub
import optax
from orbax import checkpoint as ocp
import qwix
import datasets
from tqdm.auto import tqdm

# Tunix Imports
from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma import model as gemma_lib
from tunix.models.gemma import params as params_lib
from tunix.sft import peft_trainer

# Stability
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
print(f"JAX Devices: {jax.devices()}")

# Constants
MODEL_ID = "google/gemma-2-2b-it"
SFT_OUTPUT_DIR = "/kaggle/working/sft_continuation_checkpoint"
""")

    # --- WandB Cell ---
    wandb_cell = nbf.v4.new_code_cell("""
# --- WandB Logging with Metrics Backend ---
WANDB_ENABLED = False

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
        wandb.init(
            project="tunix-sft-continuation",
            name="sft-cont-v1",
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
            }
        )
        WANDB_ENABLED = True
        print("WandB Logging Enabled for continuation training.")
    else:
        raise ValueError("Empty WANDB_API_KEY")

except Exception as e:
    print(f"WandB not enabled: {e}")
    os.environ["WANDB_MODE"] = "disabled"
    print("Proceeding without cloud logging.")
""")

    # --- Model Utilities ---
    model_utils_cell = nbf.v4.new_code_cell("""
# --- Model Utilities ---
MESH = [(8, 1), ("fsdp", "tp")]

def get_gemma_model(ckpt_path):
    # Load Base Model Structure
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
    # Tunix LoRA Config
    RANK = 64
    ALPHA = 64.0
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

    # --- Load Checkpoint Logic ---
    load_ckpt_cell = nbf.v4.new_code_cell("""
# --- Load Checkpoint & Prepare Model ---

# 1. Download Base Model (for tokenizer & structure)
if "KAGGLE_USERNAME" not in os.environ:
    kagglehub.login()

kaggle_ckpt_path = kagglehub.model_download(f"google/gemma-2/flax/gemma2-2b-it")

# Prepare intermediate conversion
INTERMEDIATE_CKPT_DIR = "/tmp/content/intermediate_ckpt/"
if not os.path.exists(INTERMEDIATE_CKPT_DIR):
    print("Converting base model checkpoint...")
    params = params_lib.load_and_format_params(os.path.join(kaggle_ckpt_path, "gemma2-2b-it"))
    gemma = gemma_lib.Transformer.from_params(params, version="2-2b-it")
    checkpointer = ocp.StandardCheckpointer()
    _, state = nnx.split(gemma)
    checkpointer.save(os.path.join(INTERMEDIATE_CKPT_DIR, "state"), state)
    checkpointer.wait_until_finished()
    del params, gemma, state
    gc.collect()

# 2. Initialize Models
print("Initializing Base Model...")
base_model, mesh, model_config = get_gemma_model(os.path.join(INTERMEDIATE_CKPT_DIR, "state"))
lora_model = get_lora_model(base_model, mesh=mesh)

# 3. Load Previous Session State (LoRA weights)
print(f"Restoring Session 1 Checkpoint from: {PREV_CHECKPOINT_PATH}")

try:
    # Map structure for LoRA params
    abs_lora_params = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        nnx.state(lora_model, nnx.LoRAParam),
    )
    
    # Restore
    prev_checkpointer = ocp.StandardCheckpointer()
    restored_lora_params = prev_checkpointer.restore(PREV_CHECKPOINT_PATH, target=abs_lora_params)
    
    # Update model
    nnx.update(lora_model, restored_lora_params)
    print("✅ Successfully restored previous SFT state.")
    
except Exception as e:
    print(f"❌ Failed to restore checkpoint: {e}")
    print("Double check PREV_CHECKPOINT_PATH. If this is the first run, this is expected to fail.")
    print("CRITICAL: Continuing without loaded state means restarting training from scratch!")
    # raise e # Uncomment to enforce strict loading

# 4. Tokenizer
tokenizer = tokenizer_lib.Tokenizer(
    tokenizer_path=os.path.join(kaggle_ckpt_path, "tokenizer.model")
)
""")

    # --- Data Loading (Pre-sampled GlaiveAI Parquet) ---
    data_cell = nbf.v4.new_code_cell("""
# --- Load Continuation Dataset (Pre-sampled GlaiveAI) ---
# This parquet file contains 100K fresh samples from GlaiveAI
# (samples 30,001 - 130,000, NOT overlapping with single session)

import glob
import re

print(f"Loading continuation data from {CONTINUATION_DATA_PATH}...")

all_texts = []

def standardize_to_gemma_format(text, question=None):
    '''Standardize GlaiveAI format to Gemma conversation format'''
    # Replace GlaiveAI's <think> tags with our <reasoning> tags
    text = re.sub(r"<think>", "<reasoning>", text, flags=re.IGNORECASE)
    text = re.sub(r"</think>", "</reasoning>", text, flags=re.IGNORECASE)
    
    # Case 1: Has <reasoning> but no <answer> - extract answer from after </reasoning>
    if "<reasoning>" in text and "<answer>" not in text:
        parts = text.split("</reasoning>")
        if len(parts) > 1:
            reasoning_part = parts[0] + "</reasoning>"
            answer_part = parts[1].strip()
            if answer_part:
                text = f"{reasoning_part}\\n<answer>{answer_part}</answer>"
            else:
                # No content after reasoning - use last sentence as answer fallback
                reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
                if reasoning_match:
                    reasoning_text = reasoning_match.group(1).strip()
                    sentences = reasoning_text.split(".")
                    answer_fallback = sentences[-1].strip() if sentences and sentences[-1].strip() else reasoning_text[:200]
                    text = f"{text}\\n<answer>{answer_fallback}</answer>"
    
    # Case 2: No <reasoning> AND no <answer> - wrap entire text with both tags
    elif "<reasoning>" not in text and "<answer>" not in text:
        # Treat text as combined reasoning+answer
        # Use all but last paragraph as reasoning, last paragraph as answer
        paragraphs = text.strip().split("\\n\\n")
        if len(paragraphs) > 1:
            reasoning = "\\n\\n".join(paragraphs[:-1])
            answer = paragraphs[-1]
        else:
            # Single paragraph - use as both
            reasoning = text.strip()
            answer = text.strip()
        text = f"<reasoning>{reasoning}</reasoning>\\n<answer>{answer}</answer>"
    
    # Case 3: Has <answer> but no <reasoning> - wrap content before <answer> as reasoning
    elif "<answer>" in text and "<reasoning>" not in text:
        parts = text.split("<answer>")
        if len(parts) > 1:
            pre_answer = parts[0].strip()
            answer_content = "<answer>" + parts[1]
            if pre_answer:
                text = f"<reasoning>{pre_answer}</reasoning>\\n{answer_content}"
            else:
                # No content before answer - use answer content as reasoning too
                answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
                if answer_match:
                    text = f"<reasoning>{answer_match.group(1).strip()}</reasoning>\\n{answer_content}"
    
    # Build full conversation format
    if question:
        formatted = f"<start_of_turn>user\\n{SYSTEM_PROMPT}\\n\\n{question}<end_of_turn>\\n<start_of_turn>model\\n{text}"
        return formatted
    return text

# Load parquet files from continuation dataset
try:
    parquet_files = glob.glob(f"{CONTINUATION_DATA_PATH}/*.parquet")
    
    if parquet_files:
        # 1. Pass 1: Counting (No Memory Load)
        print("Pass 1: Scanning continuation dataset...")
        total_samples = 0
        threshold_counts = {t: 0 for t in [10400, 12500, 15000]}
        threshold_counts[None] = 0 # No filter

        for parquet_file in parquet_files:
            ds = datasets.load_dataset("parquet", data_files=parquet_file, split="train")
            print(f"Scanning {os.path.basename(parquet_file)} ({len(ds)} samples)...")
            
            for sample in ds:
                response_len = len(sample.get("response", ""))
                total_samples += 1 # Count ALL samples for ratio
                
                if response_len < 50: continue
                
                for t in threshold_counts:
                    if t is None or response_len <= t:
                        threshold_counts[t] += 1
        
        print(f"Total samples scanned: {total_samples}")
        
        # 2. Select Threshold
        selected_threshold = None
        thresholds_ordered = [10400, 12500, 15000, None]
        
        for t in thresholds_ordered:
            count = threshold_counts[t]
            ratio = count / total_samples if total_samples > 0 else 0
            print(f"Threshold: {t} -> Kept: {count}/{total_samples} ({ratio:.2%})")
            
            if ratio >= 0.8:
                selected_threshold = t
                print(f"Selected Threshold: {selected_threshold}")
                break
        else:
            print("All thresholds yielded < 80% data. Disabling length filter.")
            selected_threshold = None
        
        # 3. Pass 2: Loading & Formatting
        print("Pass 2: Loading selected continuation samples (Streaming to JSONL)...")
        
        # Stream directly to JSONL file to avoid holding all strings in RAM
        with open("continuation_data.jsonl", "w") as f:
            import json
            for parquet_file in parquet_files:
                ds = datasets.load_dataset("parquet", data_files=parquet_file, split="train")
                for sample in ds:
                    prompt = sample.get("prompt", "")
                    response = sample.get("response", "")
                    
                    # Apply filter
                    if len(response) < 50: continue
                    if selected_threshold is not None and len(response) > selected_threshold:
                        continue
                        
                    if prompt and response:
                        formatted = standardize_to_gemma_format(response, question=prompt)
                        f.write(json.dumps({"text": formatted}) + "\\n")
        
        # Load using memory-mapped Arrow dataset
        print("Loading dataset from JSONL (Memory Safe)...")
        sft_dataset = datasets.load_dataset("json", data_files="continuation_data.jsonl", split="train")
        dataset_size = len(sft_dataset)
        print(f"Final Continuation Dataset Size: {dataset_size}")
        
        # Reduce disk pressure
        if os.path.exists("continuation_data.jsonl"):
            os.remove("continuation_data.jsonl")
            print("Removed temporary continuation_data.jsonl")
        
        # --- Dynamic SFT Steps Calculation ---
        # Target: ~2 epochs for continuation
        # SAFETY: Use math.ceil and max(1, ...)
        import math
        TARGET_EPOCHS = 2
        SFT_STEPS = max(1, math.ceil((dataset_size * TARGET_EPOCHS) / EFFECTIVE_BATCH))
        print(f"Dynamic SFT Steps: {SFT_STEPS} (based on {dataset_size} samples, {TARGET_EPOCHS} epochs)")

    else:
        print(f"WARNING: No parquet files found in {CONTINUATION_DATA_PATH}")
        # Default fallback to prevent crash if files missing
        dataset_size = 0
        SFT_STEPS = 100 
        # Create empty dummy dataset to prevent NameError downstream
        sft_dataset = datasets.Dataset.from_dict({"text": []})
    
    print(f"Total continuation samples: {dataset_size}")
    
    # Shuffle & Show Sample (Only if data exists)
    if dataset_size > 0:
        sft_dataset = sft_dataset.shuffle(seed=42)
        print(f"\\nSample: {sft_dataset[0]['text'][:500]}...")
    
except Exception as e:
    print(f"CRITICAL: Failed to load continuation data: {e}")
    raise RuntimeError(f"Dataset loading failed: {e}")
""")

    # --- Training Loop ---
    train_cell = nbf.v4.new_code_cell("""
# --- Continuation Training ---

print("Starting SFT Continuation...")

# Imports for Training
import numpy as np
from tunix.sft import utils as sft_utils

# Optimizer (Lower LR)
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=LEARNING_RATE,
    warmup_steps=100,
    decay_steps=SFT_STEPS,
    end_value=1e-7
)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01)
)

# Training Config
checkpoint_options = ocp.CheckpointManagerOptions(
    save_interval_steps=500, max_to_keep=2
)

def create_data_iterator(dataset, batch_size, tokenizer):
    '''Create batches with tokenization and masking'''
    indices = np.random.permutation(len(dataset))
    
    # Infinite iterator matching steps
    while True:
        np.random.shuffle(indices)
        for i in range(0, len(dataset), batch_size):
            batch_indices = indices[i:i+batch_size]
            if len(batch_indices) < batch_size:
                continue
                
            texts = [dataset[int(idx)]['text'] for idx in batch_indices]
            
            # Tokenize
            batch_input_tokens = []
            batch_input_mask = []
            
            for text in texts:
                # Use Tunix Tokenizer.tokenize which handles BOS/EOS
                tokens = tokenizer.tokenize(text, add_eos=True).tolist()
                
                # Truncate / Pad
                if len(tokens) > MAX_SEQ_LEN:
                    tokens = tokens[:MAX_SEQ_LEN]
                    mask = [True] * MAX_SEQ_LEN
                else:
                    pad_len = MAX_SEQ_LEN - len(tokens)
                    mask = [True] * len(tokens) + [False] * pad_len
                    pad_id = getattr(tokenizer, 'pad_id', lambda: 0)()
                    tokens = tokens + [pad_id] * pad_len
                
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
    pbar_description="SFT Continuation",
    metrics_prefix="sft_cont",
    metrics_logging_options=metrics_logging_options,
    eval_every_n_steps=10000,
)

trainer = peft_trainer.PeftTrainer(
    model=lora_model,
    optimizer=optimizer,
    training_config=training_config
)

# Create Iterator
train_iter = create_data_iterator(sft_dataset, TRAIN_BATCH_SIZE, tokenizer)

print(f"Starting Continuation Training for {SFT_STEPS} steps...")
print(f"Learning Rate Peak: {LEARNING_RATE}")

with mesh:
    trainer.train(train_ds=train_iter, skip_jit=False)

print("Continuation Training Complete.")
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
        "What are the ethical implications of AI in healthcare?",
        "Write a short story about a robot learning to paint.",
        "Explain why the sky is blue to a 5-year-old.",
        "Solve this math problem step-by-step: If 2x + 5 = 15, what is x?"
    ]
    
    formatted_prompts = [TEMPLATE.format(question=p) for p in test_prompts]
    
    out_data = inference_sampler(
        input_strings=formatted_prompts,
        max_generation_steps=1024,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        echo=False
    )
    
    # Validation Logic
    print("--- Post-Training Outputs ---")
    valid_format_count = 0
    results_for_wandb = []
    
    for p, o in zip(test_prompts, out_data.text):
        print(f"Prompt: {p}")
        print(f"Output: {o[:500]}...")
        
        has_reasoning = bool(re.search(r"<reasoning>.*?</reasoning>", o, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", o, re.DOTALL))
        
        is_valid = has_reasoning and has_answer
        if is_valid:
            valid_format_count += 1
            print("✅ Format Check: Passed")
        else:
            print(f"❌ Format Check: Failed")
            
        results_for_wandb.append([p, o, is_valid])
        print("-" * 50)
    
    print(f"Format Validation: {valid_format_count}/{len(test_prompts)} passed.")
    
    # Safe WandB Logging
    try:
        if 'wandb' in locals() and wandb.run is not None:
            tbl = wandb.Table(columns=["Prompt", "Output", "IsValid"], data=results_for_wandb)
            wandb.log({"eval_results": tbl})
            print("Logged results to WandB.")
    except Exception as w_err:
        print(f"Post-training WandB logging failed: {w_err}")
    
    # Force sync before exit
    print("Syncing WandB...")
    try:
        if 'wandb' in locals() and wandb.run is not None:
             wandb.finish()
    except:
        pass
    
    import time
    time.sleep(5)
    print("Done.")

except Exception as e:
    print(f"Evaluation failed: {e}")
""")

    # --- Save Cell ---
    save_cell = nbf.v4.new_code_cell("""
# --- Save Continuation Model ---
FINAL_SAVE_DIR = "/kaggle/working/final_continuation_model"
os.makedirs(FINAL_SAVE_DIR, exist_ok=True)

checkpointer = ocp.StandardCheckpointer()
checkpointer.save(os.path.join(FINAL_SAVE_DIR, "checkpoint"), nnx.state(lora_model, nnx.LoRAParam))
checkpointer.wait_until_finished()

print(f"✅ Model saved to {FINAL_SAVE_DIR}")
print("1. Download output.")
print("2. Upload as Kaggle Model.")
print("3. Update Unrestricted Model ID.")

unrestricted_kaggle_model = "yuyamukai/tunix-gemma2-sft-unrestricted"
""")

    nb.cells = [
        title_cell,
        config_cell,
        setup_cell,
        wandb_cell,
        model_utils_cell,
        load_ckpt_cell,
        data_cell,
        train_cell,
        visual_eval_cell,
        save_cell
    ]

    with open('tunix_sft_continuation.ipynb', 'w') as f:
        nbf.write(nb, f)
    
    print("Notebook 'tunix_sft_continuation.ipynb' created successfully.")

if __name__ == "__main__":
    create_notebook()
