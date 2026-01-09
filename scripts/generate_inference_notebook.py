#!/usr/bin/env python3
"""
Tunix Inference Notebook Generator
Generates: tunix_inference_eval.ipynb

Usage:
    python scripts/generate_inference_notebook.py
"""

import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()

    # --- Cell 1: Title ---
    title_cell = nbf.v4.new_markdown_cell("""# Tunix Inference & Evaluation

This notebook loads a trained Tunix SFT checkpoint (Gemma 2 2B + LoRA) and runs inference on evaluation prompts.
Use this to verify model performance without re-running training.
""")

    # --- Cell 2: Setup ---
    setup_cell = nbf.v4.new_code_cell("""
# --- Setup & Install ---
!pip install -q kagglehub
!pip install -q ipywidgets
!pip install -q tensorflow
!pip install -q tensorflow_datasets
!pip install -q tensorboardX
!pip install -q transformers
!pip install -q grain

import socket
import os

def is_connected():
    try:
        socket.create_connection(("1.1.1.1", 53))
        return True
    except OSError:
        pass
    return False

if is_connected():
    !pip install -q -U chex==0.1.90
    !pip install -q -U google-tunix[prod]==0.1.5 distrax==0.1.7 optax==0.2.6
    !pip install git+https://github.com/google/qwix
else:
    print("Offline mode detected. Assuming dependencies are installed or wheels provided.")
    # Fallback: Try installing from local wheels if available
    if os.path.exists("/kaggle/input/tunix-wheels"):
        !pip install --no-index --find-links=/kaggle/input/tunix-wheels google-tunix
        !pip install --no-index --find-links=/kaggle/input/tunix-wheels qwix

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
import shutil
from pprint import pprint

from flax import nnx
import jax
import jax.numpy as jnp
import kagglehub
from orbax import checkpoint as ocp
import qwix
import numpy as np

# Tunix Imports
from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma import model as gemma_lib
from tunix.models.gemma import params as params_lib

# --- Config ---
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

print(f"JAX Devices: {jax.devices()}")

# Paths
CHECKPOINT_DIR = "/kaggle/input/tunix-sft-checkpoint-v5/sft_checkpoint"  # Updated for manual dataset upload
# CHECKPOINT_DIR = "/kaggle/working/sft_checkpoint"  # Default for training

RANK = 64
ALPHA = 64.0
MAX_SEQ_LEN = 2048

# Inference Params
INFERENCE_TEMPERATURE = 0.0 # Greedy decoding as per competition check
INFERENCE_TOP_K = 1
INFERENCE_TOP_P = None
EVAL_MAX_TOKENS = 2048
SEED = 42
""")

    # --- Cell 3: Model Utils ---
    model_utils_cell = nbf.v4.new_code_cell("""
# --- Model Loading Utilities ---
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
    # Restore base model params
    restored_params = checkpointer.restore(ckpt_path, target=abs_state)

    graph_def, _ = nnx.split(abs_gemma)
    gemma = nnx.merge(graph_def, restored_params)
    return gemma, mesh, model_config

def get_lora_model(base_model, mesh):
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

def restore_lora_checkpoint(lora_model, checkpoint_path):
    '''Restores LoRA adapter weights from Orbax checkpoint'''
    print(f"Restoring LoRA weights from {checkpoint_path}...")
    checkpointer = ocp.StandardCheckpointer()
    
    # We only need to restore the params structure
    abstract_state = nnx.state(lora_model, nnx.LoRAParam)
    restored_state = checkpointer.restore(checkpoint_path, target=abstract_state)
    
    # Update model with restored LoRA params
    nnx.update(lora_model, restored_state)
    print("LoRA weights restored.")
    return lora_model
""")

    # --- Cell 4: Load Base Model ---
    load_base_cell = nbf.v4.new_code_cell("""
# --- 1. Load Base Model ---
if "KAGGLE_USERNAME" not in os.environ:
    kagglehub.login()

# Download Base Gemma 2
model_path = { "gemma2": "google/gemma-2/flax/" }
model_version = "gemma2-2b-it" 
kaggle_ckpt_path = kagglehub.model_download(f"{model_path['gemma2']}{model_version}")

# Convert/Prepare Base Checkpoint
INTERMEDIATE_CKPT_DIR = "/tmp/content/intermediate_ckpt/"
if not os.path.exists(os.path.join(INTERMEDIATE_CKPT_DIR, "state")):
    print("Converting base checkpoint...")
    params = params_lib.load_and_format_params(os.path.join(kaggle_ckpt_path, "gemma2-2b-it"))
    gemma = gemma_lib.Transformer.from_params(params, version="2-2b-it")
    checkpointer = ocp.StandardCheckpointer()
    _, state = nnx.split(gemma)
    checkpointer.save(os.path.join(INTERMEDIATE_CKPT_DIR, "state"), state)
    checkpointer.wait_until_finished()
    del params, gemma, state
    gc.collect()

# Load Base Model
print("Loading Base Model...")
base_model, mesh, model_config = get_gemma_model(os.path.join(INTERMEDIATE_CKPT_DIR, "state"))
lora_model = get_lora_model(base_model, mesh=mesh)

# Setup Tokenizer
tokenizer = tokenizer_lib.Tokenizer(
    tokenizer_path=os.path.join(kaggle_ckpt_path, "tokenizer.model")
)
""")

    # --- Cell 5: Load Adapter ---
    # Robust loading logic here
    load_adapter_cell = nbf.v4.new_code_cell("""
# --- 2. Load Trained Adapters ---
# Find latest checkpoint
import glob
try:
    # Orbax checkpoints are typically directories named by step number, e.g., 20500
    # We need to find the step directories inside CHECKPOINT_DIR
    
    if not os.path.exists(CHECKPOINT_DIR):
        raise ValueError(f"Checkpoint directory {CHECKPOINT_DIR} does not exist.")
        
    # List subdirectories that are integers (steps)
    subdirs = [d for d in os.listdir(CHECKPOINT_DIR) if os.path.isdir(os.path.join(CHECKPOINT_DIR, d)) and d.isdigit()]
    if not subdirs:
        raise ValueError(f"No step checkpoints found in {CHECKPOINT_DIR}")
    
    latest_step = max([int(d) for d in subdirs])
    checkpoint_step_dir = os.path.join(CHECKPOINT_DIR, str(latest_step))
    
    print(f"Found latest checkpoint step: {latest_step}")
    print(f"Directory: {checkpoint_step_dir}")
    
    # --- Debug: List Directory Contents ---
    print("--- Directory Structure ---")
    for root, dirs, files in os.walk(checkpoint_step_dir):
        level = root.replace(checkpoint_step_dir, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        for f in files:
            print('{}{}'.format(indent + '    ', f))
    print("---------------------------")

    # Smart Path Detection
    # Orbax/Tunix might save under 'default', 'params', 'state', 'model_params', or directly in the step dir
    potential_subdirs = ["default", "params", "state", "model_params", "."]
    
    checkpoint_path = None
    
    for sub in potential_subdirs:
        path = os.path.join(checkpoint_step_dir, sub) if sub != "." else checkpoint_step_dir
        # Check if it looks like a checkpoint (contains msgpack or similar)
        # Or just try to restore from it
        if os.path.exists(path):
            # Basic check: does it contain files?
             if len(os.listdir(path)) > 0:
                 print(f"Attempting to restore from CANDIDATE path: {path}")
                 try:
                     restore_lora_checkpoint(lora_model, path)
                     checkpoint_path = path
                     print("✅ Restore successful!")
                     break
                 except Exception as restore_err:
                     print(f"       ⚠️ Failed to restore from {path}: {restore_err}")
    
    if checkpoint_path is None:
         raise RuntimeError("Could not find a valid checkpoint structure in any standard subdirectory.")

except Exception as e:
    print(f"CRITICAL: Failed to load checkpoint: {e}")
    # Raise error to STOP execution. Do not continue to inference with random weights.
    raise e
""")

    # --- Cell 6: Run Eval ---
    eval_cell = nbf.v4.new_code_cell("""
# --- 3. Run Inference ---
print("Running Evaluation...")

prompts = [
    "Write a short story about a robot learning to paint.",
    "Write a haiku about artificial intelligence.",
    "Propose three innovative uses for AI in education.",
    "Summarize the key benefits and risks of renewable energy in 3 paragraphs.",
    "Solve step-by-step: If 2x + 5 = 15, what is x?",
    "Write a Python function to check if a string is a palindrome.",
    "Explain why the sky is blue to a 5-year-old.",
    "Explain the process of photosynthesis step by step.",
    "What are the ethical implications of AI in healthcare?",
    "Should AI systems have rights? Argue both sides.",
]

SYSTEM_PROMPT = "You are a deep thinking AI. Think step by step about the problem and provide your reasoning between <reasoning> and </reasoning> tags. Then, provide the final answer between <answer> and </answer> tags."
TEMPLATE = f"<start_of_turn>user\\n{SYSTEM_PROMPT}\\n\\n{{question}}<end_of_turn>\\n<start_of_turn>model"
formatted_prompts = [TEMPLATE.format(question=p) for p in prompts]

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

print("--- Results & Validation (Greedy Decoding) ---")
valid_format_count = 0

# Process sequentially to avoid OOM with large context
for i, p in enumerate(prompts):
    print(f"\\nProcessing Prompt {i+1}/{len(prompts)}...")
    formatted_prompt = TEMPLATE.format(question=p)
    
    # Run single inference
    try:
        out_data = inference_sampler(
            input_strings=[formatted_prompt],
            max_generation_steps=EVAL_MAX_TOKENS,
            temperature=INFERENCE_TEMPERATURE,
            top_k=INFERENCE_TOP_K,
            top_p=INFERENCE_TOP_P,
            echo=False
        )
        output_text = out_data.text[0]
        
        print(f"Prompt: {p}")
        print(f"Output: {output_text}")
        
        # --- Format Validation ---
        has_reasoning = bool(re.search(r"<reasoning>.*?</reasoning>", output_text, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", output_text, re.DOTALL))
        
        is_valid = has_reasoning and has_answer
        if is_valid:
            valid_format_count += 1
            print("✅ Format Check: Passed")
        else:
            print(f"❌ Format Check: Failed (Reasoning: {has_reasoning}, Answer: {has_answer})")
            
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error generating response for prompt {i+1}: {e}")
    
    # Explicit Clean-up
    gc.collect() # Force garbage collection to prevent OOM

print(f"\\nFinal Score: {valid_format_count}/{len(prompts)} ({valid_format_count/len(prompts)*100:.1f}%) formatted correctly.")
print("Note: 'Baseline' models typically score 0% on this check as they lack the reasoning structure.")

""")

    nb.cells = [
        title_cell,
        setup_cell,
        model_utils_cell,
        load_base_cell,
        load_adapter_cell,
        eval_cell
    ]

    with open('tunix_inference_eval.ipynb', 'w') as f:
        nbf.write(nb, f)

    print("Notebook 'tunix_inference_eval.ipynb' created successfully.")

if __name__ == "__main__":
    create_notebook()
