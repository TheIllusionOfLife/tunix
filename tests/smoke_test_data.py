
import os
import re
import datasets

# Mock Constants
SYSTEM_PROMPT = "You are a deep thinking AI. Think step by step about the problem and provide your reasoning between <reasoning> and </reasoning> tags. Then, provide the final answer between <answer> and </answer> tags."

def standardize_to_gemma_format(text, question=None):
    '''Convert various formats to Gemma chat template with <reasoning>/<answer> tags'''
    
    # Handle already formatted text
    if "<start_of_turn>" in text:
        # Just ensure we have our tags (case insensitive replacement)
        text = re.sub(r"<think>", "<reasoning>", text, flags=re.IGNORECASE)
        text = re.sub(r"</think>", "</reasoning>", text, flags=re.IGNORECASE)
        text = re.sub(r"<thought>", "<reasoning>", text, flags=re.IGNORECASE)
        text = re.sub(r"</thought>", "</reasoning>", text, flags=re.IGNORECASE)
        
        # Enforce <answer> tags if missing
        if "<answer>" not in text and "<start_of_turn>model" in text:
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
                     else:
                         reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", content, re.DOTALL)
                         if reasoning_match:
                             reasoning_text = reasoning_match.group(1).strip()
                             sentences = reasoning_text.split(".")
                             answer_fallback = sentences[-1].strip() if sentences else reasoning_text[:200]
                             text = text + f"\n<answer>{answer_fallback}</answer>"
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

print("Smoke Test: Starting Data Load Check...")
all_texts = []

try:
    print("1. Testing Raiden (Small Subset)...")
    # Using small split to test logic without downloading gigabytes
    raiden = datasets.load_dataset("sequelbox/Raiden-DeepSeek-R1", split="train[:50]") 
    for sample in raiden:
        prompt = sample.get("prompt", "")
        response = sample.get("response", sample.get("completion", ""))
        
        # Test Filter Logic
        if len(response) > 8000 or len(response) < 50:
            print(f"Filtered sample of length {len(response)}")
            continue
            
        if prompt and response:
            formatted = standardize_to_gemma_format(response, question=prompt)
            all_texts.append({"text": formatted})
    print(f"Raiden Pass: Loaded {len(all_texts)} valid samples.")

    print("\n2. Testing OpenO1 (Small Subset)...")
    openo1 = datasets.load_dataset("O1-OPEN/OpenO1-SFT", split="train[:50]")
    before_count = len(all_texts)
    for sample in openo1:
        instruction = sample.get("instruction", "")
        output = sample.get("output", "")
        # Chinese filter check (mocking inputs if needed, but using real data here)
        if any(u'\u4e00' <= c <= u'\u9fff' for c in instruction + output):
             continue
        if instruction and output:
            formatted = standardize_to_gemma_format(output, question=instruction)
            all_texts.append({"text": formatted})
    print(f"OpenO1 Pass: Added {len(all_texts)-before_count} samples.")

    print("\n✅ Smoke Test Passed. Logic does not crash.")

except Exception as e:
    print(f"\n❌ Smoke Test Failed: {e}")
    exit(1)
