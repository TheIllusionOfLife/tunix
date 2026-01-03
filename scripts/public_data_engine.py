
import os
import json
import logging
from typing import List, Dict, Optional
from datasets import load_dataset
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for tags
REASONING_START = "<reasoning>"
REASONING_END = "</reasoning>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"

CHAT_TEMPLATE = """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""

class PublicDataEngine:
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def format_xml(self, reasoning: str, answer: str) -> str:
        return f"{REASONING_START}\n{reasoning}\n{REASONING_END}\n{ANSWER_START}\n{answer}\n{ANSWER_END}"

    def prepare_magpie_sft(self, num_samples: int = 5000):
        """
        Loads Magpie-Reasoning-V2 for SFT.
        Magpie output is often pure reasoning or mixed.
        We will try to parse if structure exists, or wrap the whole response as reasoning if it looks like a trace.
        For V2-250k-CoT-Deepseek-R1, it usually has 'response' with reasoning.
        """
        logger.info(f"Loading Magpie-Reasoning (subset: {num_samples})...")
        try:
            # Using a known high-quality subset or the main one
            # Note: We use streaming to avoid downloading massive 70B outputs if possible, or just take the first N
            ds = load_dataset("Magpie-Align/Magpie-Reasoning-V2-250k-CoT-Deepseek-R1-Llama-70B", split="train", streaming=True)
            
            data = []
            count = 0
            for item in ds:
                if count >= num_samples:
                    break
                
                instruction = item.get("instruction", "")
                response = item.get("response", "")
                
                # Check if response already has tags (unlikely for raw Magpie)
                # Heuristic: Magpie CoT often puts reasoning first. 
                # We will wrap the entire response in <reasoning> for now as "Thinking" data
                # And maybe extract the last line as answer? Or just treat it as a reasoning demo.
                # BETTER STRATEGY for Competition: 
                # Use dataset where we can separate R and A. 
                # GSM8K is better for this separation. Magpie might be good for 'pure reasoning' style.
                # Let's try to extract '####' style if present, else just put empty answer or wrap all.
                
                # Simple logic: If we can't extract answer, we put "See reasoning above" in answer tag
                # to satisfy the format requirement strictly.
                
                formatted_response = f"{REASONING_START}\n{response}\n{REASONING_END}\n{ANSWER_START}\n(See reasoning)\n{ANSWER_END}"
                
                full_text = CHAT_TEMPLATE.format(instruction=instruction) + formatted_response
                data.append({"text": full_text})
                count += 1
            
            output_path = os.path.join(self.output_dir, "sft_magpie.jsonl")
            with open(output_path, "w") as f:
                for entry in data:
                    f.write(json.dumps(entry) + "\n")
            logger.info(f"Saved {len(data)} Magpie SFT samples to {output_path}")
            
        except Exception as e:
            logger.error(f"Error loading Magpie: {e}")

    def prepare_gsm8k_grpo(self, split: str = "train"):
        """
        Loads GSM8K for GRPO (needs Question and Answer separated).
        Answer in GSM8K: '...reasoning... #### 42'
        We split this.
        """
        logger.info(f"Loading GSM8K ({split})...")
        try:
            ds = load_dataset("gsm8k", "main", split=split)
            
            data = []
            for item in tqdm(ds):
                q = item['question']
                a_raw = item['answer']
                
                if "####" in a_raw:
                    reasoning, answer = a_raw.split("####")
                    reasoning = reasoning.strip()
                    answer = answer.strip()
                else:
                    reasoning = a_raw
                    answer = ""
                
                # For GRPO, we usually filter by prompt only, but we need the ground truth for reward.
                # Tunix GRPO expects a dataset with 'prompt' and 'answer' (for reward fn).
                
                entry = {
                    "prompt": CHAT_TEMPLATE.format(instruction=q),
                    "answer": answer  # The ground truth for exact_match_reward
                }
                data.append(entry)
                
            output_path = os.path.join(self.output_dir, f"grpo_gsm8k_{split}.jsonl")
            with open(output_path, "w") as f:
                for entry in data:
                    f.write(json.dumps(entry) + "\n")
            logger.info(f"Saved {len(data)} GSM8K GRPO samples to {output_path}")

        except Exception as e:
            logger.error(f"Error loading GSM8K: {e}")

    def prepare_ultrafeedback_sft(self, num_samples: int = 1000):
        """
        Loads UltraFeedback to add diverse conversational/creative data to SFT.
        This helps with 'Creative writing', 'Summarization', etc.
        """
        logger.info(f"Loading UltraFeedback (subset: {num_samples})...")
        try:
            ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_sft")
            ds = ds.shuffle(seed=42).select(range(num_samples))
            
            output_file = os.path.join(self.output_dir, "sft_ultrafeedback.jsonl")
            
            with open(output_file, 'w') as f:
                for row in ds:
                    # UltraFeedback has 'messages' list: [{'role': 'user', 'content':...}, {'role': 'assistant', ...}]
                    messages = row['messages']
                    if len(messages) < 2: continue
                    
                    user_content = messages[0]['content']
                    # We use the 'chosen' response implicitly (train_sft split usually has high quality)
                    assist_content = messages[1]['content']
                    
                    # Wrap in our reasoning format (even if empty reasoning, to keep schema consistent)
                    # Or relying on the model to just output answer if no reasoning is needed for creative tasks.
                    # For consistency, we'll format it as a direct answer or wrap in tags if we had reasoning.
                    # Since UF doesn't have explicit reasoning traces, we just use standard chat format
                    # But wait, our SFT trainer expects <reasoning> tags? 
                    # Actually, for diversity, we might NOT want to force reasoning on everything (e.g. poetry).
                    # However, to keep the pipeline simple, we will treat the entire response as the <answer>.
                    
                    # Assuming CHAT_TEMPLATE is used for the prompt part
                    # And then the assistant's response is wrapped as an answer.
                    final_text = (
                        CHAT_TEMPLATE.format(instruction=user_content) +
                        f"{ANSWER_START}{assist_content}{ANSWER_END}"
                    )
                    
                    f.write(json.dumps({"text": final_text}) + "\n")
            
            logger.info(f"Saved {num_samples} UltraFeedback samples to {output_file}")
            
        except Exception as e:
            logger.error(f"Error preparing UltraFeedback: {e}")

    def prepare_mbpp_grpo(self, split: str = "train"):
        """
        Loads MBPP (Mostly Basic Python Problems) for Coding GRPO.
        """
        logger.info(f"Loading MBPP ({split})...")
        try:
            ds = load_dataset("google-research-datasets/mbpp", split=split)
            
            output_file = os.path.join(self.output_dir, f"grpo_mbpp_{split}.jsonl")
            
            with open(output_file, 'w') as f:
                for row in ds:
                    prompt = row['text']
                    code = row['code']
                    tests = row['test_list']
                    
                    entry = {
                        "prompt": CHAT_TEMPLATE.format(instruction=f"Write a Python function to solve: {prompt}"),
                        "answer": code,
                        "tests": tests
                    }
                    f.write(json.dumps(entry) + "\n")
                    
            logger.info(f"Saved {len(ds)} MBPP samples to {output_file}")
            
        except Exception as e:
            logger.error(f"Error preparing MBPP: {e}")

if __name__ == "__main__":
    engine = PublicDataEngine()
    engine.prepare_magpie_sft(num_samples=5000)
    engine.prepare_ultrafeedback_sft(num_samples=1000) # Add diversity (Creative/Summ)
    engine.prepare_gsm8k_grpo(split="train")
    engine.prepare_mbpp_grpo(split="train") # Verify Coding
