
import os
import re
import datasets

# Mock Constants
SYSTEM_PROMPT = "You are a deep thinking AI. Think step by step about the problem and provide your reasoning between <reasoning> and </reasoning> tags. Then, provide the final answer between <answer> and </answer> tags."

# Use local pre-sampled parquet files
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def standardize_to_gemma_format(text, question=None):
    '''Convert various formats to Gemma chat template with <reasoning>/<answer> tags'''
    
    # Handle already formatted text
    if "<start_of_turn>" in text:
        text = re.sub(r"<think>", "<reasoning>", text, flags=re.IGNORECASE)
        text = re.sub(r"</think>", "</reasoning>", text, flags=re.IGNORECASE)
        text = re.sub(r"<thought>", "<reasoning>", text, flags=re.IGNORECASE)
        text = re.sub(r"</thought>", "</reasoning>", text, flags=re.IGNORECASE)
        
        # Case 1: Has <answer> but no <reasoning>
        if "<answer>" in text and "<reasoning>" not in text and "<start_of_turn>model" in text:
            match = re.search(r"<start_of_turn>model\n(.*)(<answer>.*</answer>)", text, re.DOTALL)
            if match:
                pre_answer = match.group(1).strip()
                answer_tag = match.group(2)
                if pre_answer:
                    new_content = f"<reasoning>{pre_answer}</reasoning>\n{answer_tag}"
                    text = re.sub(r"<start_of_turn>model\n.*(<answer>.*</answer>)", 
                                  f"<start_of_turn>model\n{new_content}", text, flags=re.DOTALL)
        
        # Case 2: Enforce <answer> tags if missing
        elif "<answer>" not in text and "<start_of_turn>model" in text:
            match = re.search(r"<start_of_turn>model\n(.*)$", text, re.DOTALL)
            if match:
                content = match.group(1).strip()
                if "<reasoning>" not in content:
                    text = text.replace(content, f"<answer>{content}</answer>")
                else:
                    parts = content.split("</reasoning>")
                    if len(parts) > 1 and parts[1].strip():
                        answer_part = parts[1].strip()
                        text = text.replace(content, f"{parts[0]}</reasoning>\n<answer>{answer_part}</answer>")
        return text
    
    # For raw question/response pairs
    if question:
        reasoning = ""
        answer = ""
        
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
        thought_match = re.search(r"<Thought>(.*?)</Thought>", text, re.DOTALL | re.IGNORECASE)
        reasoning_tag_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL | re.IGNORECASE)
        
        if think_match:
            reasoning = think_match.group(1).strip()
        elif thought_match:
            reasoning = thought_match.group(1).strip()
        elif reasoning_tag_match:
            reasoning = reasoning_tag_match.group(1).strip()
        else:
            reasoning = text.strip()
        
        ans_match = re.search(r"<Output>(.*?)</Output>", text, re.DOTALL | re.IGNORECASE)
        if ans_match:
            answer = ans_match.group(1).strip()
        else:
            answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
            if answer_match:
                answer = answer_match.group(1).strip()
            else:
                if reasoning_tag_match or think_match or thought_match:
                    remaining_text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL | re.IGNORECASE)
                    remaining_text = re.sub(r"<think>.*?</think>", "", remaining_text, flags=re.DOTALL | re.IGNORECASE)
                    remaining_text = re.sub(r"<Thought>.*?</Thought>", "", remaining_text, flags=re.DOTALL | re.IGNORECASE)
                    answer = remaining_text.strip()
                    if not answer and reasoning: 
                        answer = reasoning
                else:
                    paragraphs = text.strip().split("\n\n")
                    answer = paragraphs[-1] if paragraphs else text[:200]
        
        if not reasoning and answer:
            reasoning = answer
        elif not answer and reasoning:
            answer = reasoning
        elif not reasoning and not answer:
            reasoning = text.strip()
            answer = text.strip()

        formatted = f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{question}<end_of_turn>\n<start_of_turn>model\n<reasoning>{reasoning}</reasoning>\n<answer>{answer}</answer>"
        return formatted
    
    return text

print("Smoke Test: Using LOCAL pre-sampled parquet files...")
all_texts = []

try:
    # 1. Raiden
    print("\n1. Testing Raiden (Local Parquet)...")
    raiden_path = os.path.join(DATA_DIR, "raiden_deepseek_r1.parquet")
    if os.path.exists(raiden_path):
        ds = datasets.load_dataset("parquet", data_files=raiden_path, split="train[:50]")
        for sample in ds:
            prompt = sample.get("prompt", "")
            response = sample.get("response", sample.get("completion", ""))
            if len(response) > 8000 or len(response) < 50:
                continue
            if prompt and response:
                formatted = standardize_to_gemma_format(response, question=prompt)
                all_texts.append({"text": formatted})
        print(f"Raiden Pass: Loaded {len(all_texts)} samples.")
    else:
        print(f"Raiden SKIP: {raiden_path} not found")

    # 2. OpenO1
    print("\n2. Testing OpenO1 (Local Parquet)...")
    openo1_path = os.path.join(DATA_DIR, "openo1_sft_english_20k.parquet")
    if os.path.exists(openo1_path):
        before_count = len(all_texts)
        ds = datasets.load_dataset("parquet", data_files=openo1_path, split="train[:50]")
        for sample in ds:
            instruction = sample.get("instruction", "")
            output = sample.get("output", "")
            if instruction and output:
                formatted = standardize_to_gemma_format(output, question=instruction)
                all_texts.append({"text": formatted})
        print(f"OpenO1 Pass: Added {len(all_texts)-before_count} samples.")
    else:
        print(f"OpenO1 SKIP: {openo1_path} not found")

    # 3. CoT-Collection
    print("\n3. Testing CoT-Collection (Local Parquet)...")
    cot_path = os.path.join(DATA_DIR, "cot_collection_10k.parquet")
    if os.path.exists(cot_path):
        before_count = len(all_texts)
        ds = datasets.load_dataset("parquet", data_files=cot_path, split="train[:50]")
        for sample in ds:
            q = sample.get("source", "")
            r = sample.get("rationale", "")
            a = sample.get("target", "")
            if q and r and a:
                formatted = f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{q}<end_of_turn>\n<start_of_turn>model\n<reasoning>{r}</reasoning>\n<answer>{a}</answer>"
                all_texts.append({"text": formatted})
        print(f"CoT-Collection Pass: Added {len(all_texts)-before_count} samples.")
    else:
        print(f"CoT-Collection SKIP: {cot_path} not found")

    # 4. GlaiveAI
    print("\n4. Testing GlaiveAI (Local Parquet)...")
    glaive_path = os.path.join(DATA_DIR, "glaiveai_30k.parquet")
    if os.path.exists(glaive_path):
        before_count = len(all_texts)
        ds = datasets.load_dataset("parquet", data_files=glaive_path, split="train[:50]")
        for sample in ds:
            instruction = sample.get("instruction", sample.get("prompt", ""))
            output = sample.get("output", sample.get("response", ""))
            if instruction and output:
                formatted = standardize_to_gemma_format(output, question=instruction)
                all_texts.append({"text": formatted})
        print(f"GlaiveAI Pass: Added {len(all_texts)-before_count} samples.")
    else:
        print(f"GlaiveAI SKIP: {glaive_path} not found")

    print(f"\n✅ Smoke Test Passed. Total: {len(all_texts)} samples processed.")

except Exception as e:
    print(f"\n❌ Smoke Test Failed: {e}")
    exit(1)
