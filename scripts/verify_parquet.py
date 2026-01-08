
import pandas as pd
import glob
import os

# Define paths
parquet_files = glob.glob("data/glaiveai_90k_part*.parquet")
print(f"Found files: {parquet_files}")

# Analyze each file
for file in parquet_files:
    print(f"\n--- Analyzing {file} ---")
    try:
        df = pd.read_parquet(file)
        print(f"Columns: {df.columns.tolist()}")
        print(f"RowCount: {len(df)}")
        
        # Check for empty responses
        if 'response' in df.columns:
            empty_responses = df[df['response'].isna() | (df['response'] == "")]
            print(f"Empty/Null Responses: {len(empty_responses)}")
            
            # Length Distribution
            df['len'] = df['response'].fillna("").astype(str).apply(len)
            print(f"Mean Length: {df['len'].mean():.2f}")
            print(f"Min Length: {df['len'].min()}")
            print(f"Max Length: {df['len'].max()}")
            
            # Check filter impact
            filtered_out_long = len(df[df['len'] > 4000])
            filtered_out_short = len(df[df['len'] < 50])
            print(f"Samples > 4000 chars: {filtered_out_long} ({filtered_out_long/len(df)*100:.2f}%)")
            print(f"Samples < 50 chars: {filtered_out_short} ({filtered_out_short/len(df)*100:.2f}%)")
            
            # Language Check (Simple heuristic: check for common English words)
            sample_text = df['response'].iloc[0] if not df.empty else ""
            print(f"Sample response start: {sample_text[:100]}...")
            
        else:
            print("CRITICAL: 'response' column not found!")
            
    except Exception as e:
        print(f"Error reading {file}: {e}")
