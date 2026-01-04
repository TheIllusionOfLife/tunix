#!/usr/bin/env python3
"""
Generate tunix_continuation.ipynb for Unrestricted Mode (Session 2+).

This notebook:
1. Loads a checkpoint from a previous Kaggle session
2. Trains on harder/private data
3. Saves the final model for upload as a Kaggle Model

Usage:
    python scripts/generate_continuation_notebook.py
"""

import nbformat as nbf


def create_continuation_notebook():
    """Generate the continuation notebook for Unrestricted Mode."""
    nb = nbf.v4.new_notebook()

    # --- Title ---
    title_cell = nbf.v4.new_markdown_cell("""# Tunix Unrestricted Mode: Continuation Training

This notebook is designed for **Session 2+** of the Unrestricted Mode.

**Prerequisites**:
1. You have run `tunix_zero_cost_train.ipynb` (Session 1) and saved the checkpoint.
2. You have uploaded the checkpoint as a Kaggle Dataset.
3. You have uploaded `private_hard_reasoning.jsonl` as a separate private dataset.

**What this notebook does**:
1. Loads the checkpoint from your previous session.
2. Continues training on harder data (your private dataset).
3. Saves the final model for upload as a Kaggle Model.
""")

    # --- Configuration Cell ---
    config_cell = nbf.v4.new_code_cell("""
# ==============================================================================
# SESSION CONFIGURATION - UPDATE THESE FOR EACH RUN
# ==============================================================================

# Path to your previous session's checkpoint (uploaded as a Kaggle Dataset)
PREV_CHECKPOINT_DATASET = "/kaggle/input/tunix-session1-checkpoint/checkpoint"
# For Session 3, change to:
# PREV_CHECKPOINT_DATASET = "/kaggle/input/tunix-session2-checkpoint/checkpoint"

# Path to your training data (private hard reasoning dataset)
DATA_DATASET = "/kaggle/input/tunix-private-hard-reasoning"

# Training parameters
GRPO_STEPS = 800  # Increase for longer training
TRAIN_MICRO_BATCH_SIZE = 1

# Output directory for the final model
FINAL_SAVE_DIR = "final_continuation_model"

# Your Kaggle Model ID (set this after uploading)
unrestricted_kaggle_model = "yuyamukai/tunix-gemma2-2b-unrestricted"

print("Configuration loaded. Ready to continue training.")
""")

    # --- Setup Cell ---
    setup_cell = nbf.v4.new_code_cell("""
# --- Setup & Install ---
!pip install -q wandb==0.22.0
!pip install -q kagglehub
!pip install -q ipywidgets
!pip install -q tensorflow
!pip install -q tensorflow_datasets
!pip install -q 'google-tunix[prod]==0.1.5' --no-deps
!pip install -q orbax-checkpoint==0.11.4 grain==0.2.2 optax==0.2.4

import os
import sys
import gc
import re
import time
import json
import ast

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
import kagglehub
import datasets
from tqdm.auto import tqdm

# Tunix Imports
from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma import model as gemma_lib
from tunix.models.gemma import params as params_lib
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.rl.common import metrics_logger
from tunix.models.gemma import qwix

# JAX Config
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
print(f"JAX Devices: {jax.devices()}")
""")

    # --- Model Utilities ---
    model_utils_cell = nbf.v4.new_code_cell("""
# --- Model Utilities ---
MESH = [(8, 1), ("fsdp", "tp")]

def get_gemma_ref_model(ckpt_path):
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

    # --- Load Checkpoint ---
    load_checkpoint_cell = nbf.v4.new_code_cell("""
# --- Load Checkpoint from Previous Session ---
print(f"Loading checkpoint from: {PREV_CHECKPOINT_DATASET}")

# First, get the base Gemma model structure (for tokenizer)
if "KAGGLE_USERNAME" not in os.environ:
    kagglehub.login()

model_path = {"gemma2": "google/gemma-2/flax/"}
model_family = "gemma2"
model_version = "gemma2-2b-it"
kaggle_ckpt_path = kagglehub.model_download(f"{model_path[model_family]}{model_version}")

# Create intermediate directory for base model
INTERMEDIATE_CKPT_DIR = "/tmp/content/intermediate_ckpt/"
!rm -rf {INTERMEDIATE_CKPT_DIR}

params = params_lib.load_and_format_params(os.path.join(kaggle_ckpt_path, "gemma2-2b-it"))
gemma = gemma_lib.Transformer.from_params(params, version="2-2b-it")
checkpointer = ocp.StandardCheckpointer()
_, state = nnx.split(gemma)
checkpointer.save(os.path.join(INTERMEDIATE_CKPT_DIR, "state"), state)
checkpointer.wait_until_finished()
del params, gemma, state
gc.collect()

# Load models
ref_model, mesh, model_config = get_gemma_ref_model(os.path.join(INTERMEDIATE_CKPT_DIR, "state"))
lora_policy = get_lora_model(ref_model, mesh=mesh)

# Load previous session's LoRA weights
print("Loading LoRA weights from previous session...")
abs_lora_params = jax.tree.map(
    lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
    nnx.state(lora_policy, nnx.LoRAParam),
)
prev_checkpointer = ocp.StandardCheckpointer()
prev_lora_params = prev_checkpointer.restore(PREV_CHECKPOINT_DATASET, target=abs_lora_params)

# Update the policy with loaded weights
nnx.update(
    lora_policy,
    jax.tree.map(
        lambda a, b: b,
        nnx.state(lora_policy, nnx.LoRAParam),
        prev_lora_params,
    ),
)
print("✅ Previous checkpoint loaded successfully!")

# Setup Tokenizer
tokenizer = tokenizer_lib.Tokenizer(
    tokenizer_path=os.path.join(kaggle_ckpt_path, "tokenizer.model")
)
""")

    # --- Training Logic ---
    training_cell = nbf.v4.new_code_cell("""
# --- Continue GRPO Training on Hard Data ---
print("Loading private/hard training data...")

SYSTEM_PROMPT = "You are a deep thinking AI. Think step by step and provide reasoning between <reasoning> and </reasoning> tags. Then provide the final answer between <answer> and </answer> tags."
TEMPLATE = f"<start_of_turn>user\\n{SYSTEM_PROMPT}\\n\\n{{question}}<end_of_turn>\\n<start_of_turn>model"

# --- Reward Functions (same as Session 1) ---
def soft_structure_reward(prompts, completions, **kwargs):
    rewards = []
    for c in completions:
        score = 0.0
        if "<reasoning>" in c: score += 0.1
        if "</reasoning>" in c: score += 0.1
        if "<answer>" in c: score += 0.1
        if "</answer>" in c: score += 0.1
        if re.search(r"<reasoning>.*?</reasoning>", c, re.DOTALL): score += 0.3
        if re.search(r"<answer>.*?</answer>", c, re.DOTALL): score += 0.3
        rewards.append(min(1.0, score))
    return rewards

def structure_reward(prompts, completions, **kwargs):
    rewards = []
    for c in completions:
        has_reasoning = "<reasoning>" in c and "</reasoning>" in c
        has_answer = "<answer>" in c and "</answer>" in c
        score = 0.5 * has_reasoning + 0.5 * has_answer
        rewards.append(score)
    return rewards

def math_correctness_reward(prompts, completions, answer, **kwargs):
    rewards = []
    for c, gt in zip(completions, answer):
        try:
            match = re.search(r"<answer>(.*?)</answer>", c, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                if float(extracted) == float(gt):
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        except:
            rewards.append(0.0)
    return rewards

def code_correctness_reward(prompts, completions, answer, **kwargs):
    rewards = []
    for c, gt in zip(completions, answer):
        try:
            # First try to extract from <answer> tags
            ans_match = re.search(r"<answer>(.*?)</answer>", c, re.DOTALL)
            if ans_match:
                code_str = ans_match.group(1).strip()
            else:
                code_str = c.strip()
            
            # Check syntax validity
            ast.parse(code_str)
            # Check if ground truth is contained
            if gt.strip() in code_str or code_str in gt.strip():
                rewards.append(1.0)
            else:
                rewards.append(0.5)  # Valid syntax but different from GT
        except:
            rewards.append(0.0)
    return rewards

# Load Hard Dataset
try:
    hard_data_file = f"{DATA_DATASET}/private_hard_reasoning.jsonl"
    grpo_dataset = datasets.load_dataset("json", data_files=hard_data_file, split="train")
    print(f"Loaded {len(grpo_dataset)} hard reasoning samples.")
except Exception as e:
    print(f"CRITICAL: Failed to load dataset: {e}")
    print(f"Please ensure '{DATA_DATASET}' is attached with required files.")
    raise RuntimeError(f"Dataset loading failed. Cannot proceed without data. Error: {e}")

# Optimizer
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=3e-6,  # Lower LR for continuation
    warmup_steps=50,
    decay_steps=GRPO_STEPS,
    end_value=1e-7
)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.1)
)

# Configs
GRPO_OUTPUT_DIR = "grpo_continuation_checkpoint"
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=500, max_to_keep=2
)
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir="/tmp/grpo_logs", flush_every_n_steps=20
)

cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: mesh,
        rl_cluster_lib.Role.REFERENCE: mesh,
        rl_cluster_lib.Role.ROLLOUT: mesh,
    },
    rollout_engine='vanilla',
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=10,
        max_steps=GRPO_STEPS,
        mini_batch_size=TRAIN_MICRO_BATCH_SIZE,
        train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
        metrics_logging_options=metrics_logging_options,
        checkpoint_root_directory=GRPO_OUTPUT_DIR,
        checkpointing_options=checkpointing_options,
    ),
    rollout_config=base_rollout.RolloutConfig(
         max_tokens_to_generate=400,
         max_prompt_length=256,
         kv_cache_size=1024,
         temperature=0.9, top_p=1.0, top_k=50
    ),
)

grpo_config = GRPOConfig(
    num_generations=4,
    num_iterations=1,
    beta=0.08,
    epsilon=0.2,
)

# Create Cluster
rl_cluster = rl_cluster_lib.RLCluster(
    actor=lora_policy,
    reference=ref_model,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)

# Trainer
trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=[soft_structure_reward, structure_reward, math_correctness_reward, code_correctness_reward],
    algo_config=grpo_config,  # v0.1.5 API uses 'algo_config'
)

# Data Formatting & Training
with mesh:
    def format_fn(x):
        return {
            "prompts": TEMPLATE.format(question=x["question"]),
            "question": x["question"],
            "answer": x.get("answer", "")
        }
    
    train_ds = grpo_dataset.map(format_fn)
    
    import itertools
    import numpy as np

    def batched(iterable, n):
        it = iter(iterable)
        while True:
            chunk = list(itertools.islice(it, n))
            if not chunk: return
            batch = {k: np.array([d[k] for d in chunk]) for k in chunk[0]}
            yield batch

    def infinite_batch_generator(ds):
        while True:
            for batch in batched(ds.shuffle(seed=int(time.time())), TRAIN_MICRO_BATCH_SIZE):
                yield batch

    # Start Training
    print("Starting continuation GRPO training...")
    trainer.train(infinite_batch_generator(train_ds))

print("✅ Continuation GRPO Completed!")
""")

    # --- Save Model ---
    save_model_cell = nbf.v4.new_code_cell("""
# --- Save Final Model for Kaggle Upload ---
os.makedirs(FINAL_SAVE_DIR, exist_ok=True)

checkpointer = ocp.StandardCheckpointer()
checkpointer.save(os.path.join(FINAL_SAVE_DIR, "checkpoint"), nnx.state(lora_policy, nnx.LoRAParam))
checkpointer.wait_until_finished()

print(f"✅ Final model saved to '{FINAL_SAVE_DIR}/'")
print("")
print("=== NEXT STEPS ===")
print("1. Download the output folder after this notebook finishes.")
print("2. Go to Kaggle -> Models -> New Model -> Upload the checkpoint files.")
print(f"3. Set the Model ID to: {unrestricted_kaggle_model}")
print("")
print("Or, to continue for another session:")
print("1. Upload the output as a Dataset (e.g., 'tunix-session2-checkpoint')")
print("2. Update PREV_CHECKPOINT_DATASET in the config cell")
print("3. Run this notebook again")
""")

    # --- Unrestricted Mode Declaration ---
    unrestricted_cell = nbf.v4.new_code_cell("""
# --- Unrestricted Mode Submission ---
# This is the Kaggle Model ID for the 15 bonus points.
# Make sure you have uploaded the model files BEFORE submission.
# Note: unrestricted_kaggle_model is defined in the config cell above.

print(f"Unrestricted Mode Model ID: {unrestricted_kaggle_model}")
""")

    nb.cells = [
        title_cell,
        config_cell,
        setup_cell,
        model_utils_cell,
        load_checkpoint_cell,
        training_cell,
        save_model_cell,
        unrestricted_cell,
    ]

    with open('tunix_continuation.ipynb', 'w') as f:
        nbf.write(nb, f)
    
    print("Notebook 'tunix_continuation.ipynb' created successfully.")


if __name__ == "__main__":
    create_continuation_notebook()
