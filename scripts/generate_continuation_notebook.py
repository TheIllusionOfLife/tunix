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
# Format: /kaggle/input/{dataset-name}/{folder-structure}
PREV_CHECKPOINT_PATH = "/kaggle/input/tunix-session1-checkpoint/final_sft_model/checkpoint"

# Path to continuation training data (Fresh GlaiveAI samples - NOT overlapping with session 1)
# This is a NEW Kaggle dataset containing 100K samples from GlaiveAI (skipped first 30K used in session 1)
CONTINUATION_DATA_PATH = "/kaggle/input/tunix-sft-continuation-data"

# Training Config
SFT_STEPS = 5000  # More steps for extended training
LEARNING_RATE = 5e-6 # Lower LR for continuation
TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 16

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
!pip install -q datasets==3.2.0 optax==0.2.4 chex==0.1.88

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
    
    # Build full conversation format
    if question:
        formatted = f"<start_of_turn>user\\n{SYSTEM_PROMPT}\\n\\n{question}<end_of_turn>\\n<start_of_turn>model\\n{text}"
        return formatted
    return text

# Load parquet files from continuation dataset
try:
    for parquet_file in glob.glob(f"{CONTINUATION_DATA_PATH}/*.parquet"):
        ds = datasets.load_dataset("parquet", data_files=parquet_file, split="train")
        print(f"Loaded {len(ds)} samples from {parquet_file}")
        
        for sample in ds:
            q = sample.get("prompt", "")
            a = sample.get("response", "")
            
            if q and a:
                formatted = standardize_to_gemma_format(a, question=q)
                all_texts.append({"text": formatted})
    
    print(f"Total continuation samples: {len(all_texts)}")
    
    # Create HuggingFace dataset
    sft_dataset = datasets.Dataset.from_list(all_texts)
    sft_dataset = sft_dataset.shuffle(seed=42)
    
    # Show sample
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

MAX_SEQ_LEN = 1024

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

training_config = peft_trainer.TrainingConfig(
    max_steps=SFT_STEPS,
    checkpoint_root_directory=SFT_OUTPUT_DIR,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    checkpointing_options=checkpoint_options,
    pbar_description="SFT Continuation",
    metrics_prefix="sft_cont",
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
        print(f"WandB logging skipped: {w_err}")

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
