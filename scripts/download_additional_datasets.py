import os
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import json

OUTPUT_DIR = "data/sft_datasets"
COT_JSON_PATH = os.path.join(OUTPUT_DIR, "CoT_collection_en.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_cot_collection():
    print("Processing local CoT-Collection...")
    try:
        if not os.path.exists(COT_JSON_PATH):
            print(f"File not found: {COT_JSON_PATH}. Please ensure curl download finishes.")
            return

        # Load local json
        # The file is likely a list of objects
        with open(COT_JSON_PATH, 'r') as f:
            data = json.load(f)
            
        print(f"Loaded CoT-Collection: {len(data)} samples")
        
        # Convert to DataFrame and save as parquet
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, "cot_collection.parquet")
        df.to_parquet(output_path)
        print(f"Saved to {output_path}")
        
    except Exception as e:
        print(f"Error processing CoT-Collection: {e}")

def download_glaiveai():
    print("Downloading glaiveai/reasoning-v1-20m (first 30k)...")
    try:
        # Stream dataset
        ds = load_dataset("glaiveai/reasoning-v1-20m", split="train", streaming=True)
        
        data = []
        count = 0
        limit = 30000
        
        for sample in tqdm(ds, total=limit):
            data.append(sample)
            count += 1
            if count >= limit:
                break
                
        print(f"Collected {len(data)} samples from GlaiveAI")
        
        # Convert to pandas and save
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, "glaiveai_30k.parquet")
        df.to_parquet(output_path)
        print(f"Saved to {output_path}")

    except Exception as e:
        print(f"Error downloading GlaiveAI: {e}")

if __name__ == "__main__":
    process_cot_collection()
    download_glaiveai()
