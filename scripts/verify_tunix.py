
import os
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from absl import app, logging

# Mock Model for testing
class MockModel(nnx.Module):
    def __init__(self, key):
        self.linear = nnx.Linear(32, 10, rngs=nnx.Rngs(params=key))
        self.config = "mock_config"

    def __call__(self, x):
        return self.linear(x)
    
    def get_model_input(self):
         return {}

# Verify SFT
def verify_sft():
    print("Verifying SFT...")
    try:
        from tunix.sft import peft_trainer
        
        config = peft_trainer.TrainingConfig(
            eval_every_n_steps=10,
            max_steps=10,
            checkpoint_root_directory="./tmp_checkpoints"
        )
        
        model = MockModel(jax.random.key(0))
        optimizer = optax.sgd(0.01)
        
        trainer = peft_trainer.PeftTrainer(
            model=model,
            optimizer=optimizer,
            training_config=config
        )
        
        # Mock Data
        train_ds = [{"input_tokens": jnp.ones((1, 32)), "input_mask": jnp.ones((1, 32))} for _ in range(20)]
        
        # Create a simple loss function compatible with PeftTrainer
        def loss_fn(model, **batch):
            # batch has 'input_tokens' etc but our mock model expects (B, 32)
            # We just ignore 'input_tokens' logic for this mock and call model with random data or reuse it
            inputs = batch.get("input_tokens", jnp.ones((1, 32)))
            logits = model(inputs)
            return jnp.mean(logits) # Dummy loss

        trainer.with_loss_fn(loss_fn)
        
        # Override gen_model_input_fn to just pass data through
        trainer.with_gen_model_input_fn(lambda x: x)
        
        print("SFT Trainer initialized successfully.")
        # Dry run train loop? trainer.train(train_ds) # Skip to avoid long run, init is enough for now
        
    except ImportError as e:
        print(f"SFT Verification Failed: {e}")
    except Exception as e:
        print(f"SFT Init Failed: {e}")

# Verify DPO
def verify_dpo():
    print("\nVerifying DPO...")
    try:
        from tunix.sft.dpo import dpo_trainer
        
        config = dpo_trainer.DPOTrainingConfig(
            eval_every_n_steps=10,
            max_steps=10,
            algorithm="dpo",
            beta=0.1
        )
        
        model = MockModel(jax.random.key(1))
        # Ref model can be same or None if not used in mock
        ref_model = MockModel(jax.random.key(1)) 
        optimizer = optax.sgd(0.01)
        
        trainer = dpo_trainer.DPOTrainer(
            model=model,
            ref_model=ref_model,
            optimizer=optimizer,
            training_config=config
        )
        
        print("DPO Trainer initialized successfully.")

    except ImportError as e:
        print(f"DPO Verification Failed: {e}")
    except Exception as e:
        print(f"DPO Init Failed: {e}")

# Verify GRPO
def verify_grpo():
    print("\nVerifying GRPO...")
    try:
        from tunix.rl.grpo import grpo_learner
        
        # GRPO requires a full RL cluster setup which is complex to mock entirely here
        # We just verify import and config init
        
        config = grpo_learner.GRPOConfig(
             num_generations=4,
             num_iterations=1,
             beta=0.04
        )
        
        print("GRPO Config and Import successful.")

    except ImportError as e:
        print(f"GRPO Verification Failed: {e}")
    except Exception as e:
        print(f"GRPO Init Failed: {e}")

def main(_):
    verify_sft()
    verify_dpo()
    verify_grpo()

if __name__ == "__main__":
    app.run(main)
