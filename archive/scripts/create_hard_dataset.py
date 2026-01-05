import json
import os
from datasets import load_dataset
from tqdm.auto import tqdm

OUTPUT_FILE = "data/private_hard_reasoning.jsonl"
os.makedirs("data", exist_ok=True)

# Tags
REASONING_START = "<reasoning>"
REASONING_END = "</reasoning>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"

CHAT_TEMPLATE = """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""

def create_hard_dataset():
    print("Loading Magpie-Reasoning-V2 (Full Stream)...")
    # Using the Llama-70B distilled version which has 'difficulty' metadata often
    ds = load_dataset("Magpie-Align/Magpie-Reasoning-V2-250k-CoT-Deepseek-R1-Llama-70B", split="train", streaming=True)
    
    hard_samples = []
    total_scanned = 0
    target_count = 5000 # Top 2% roughly
    
    print("Filtering for 'Very Hard' or 'Long Reasoning'...")
    
    for item in tqdm(ds):
        total_scanned += 1
        
        # Difficulty Heuristics
        # 1. Check Metadata if exists
        difficulty = item.get("difficulty")
        if difficulty is None: difficulty = "medium"
        difficulty = difficulty.lower()
        
        # 2. Check Reasoning Length (Proxy for complexity)
        response = item.get("response", "")
        instruction = item.get("instruction", "")
        
        # Filter Logic:
        # - Explicitly labelled 'very hard' OR
        # - Reasoning trace > 2048 chars (Deep Complexity)
        is_hard = "very hard" in difficulty or len(response) > 2048
        
        if is_hard:
            # Format nicely
            formatted_response = f"{REASONING_START}\n{response}\n{REASONING_END}\n{ANSWER_START}\n(Implicit)\n{ANSWER_END}"
            full_text = CHAT_TEMPLATE.format(instruction=instruction) + formatted_response
            
            hard_samples.append({"text": full_text, "difficulty": difficulty, "len": len(response)})
            
            if len(hard_samples) >= target_count:
                break
        
        if total_scanned > 100000 and len(hard_samples) < 100:
            print("Warning: Finding very few hard samples. Relaxing constraints...")
            # If strictly 'very hard' is too rare, fall back to length
    
    print(f"Scanned {total_scanned} items. Found {len(hard_samples)} HARD samples.")
    
    with open(OUTPUT_FILE, "w") as f:
        for entry in hard_samples:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    create_hard_dataset()
