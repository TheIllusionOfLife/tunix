
import re
import math
import logging
from typing import List, Dict

# --- Judge Configuration ---
# Simple keyword-based classification for Zero-Cost.
# In a real pipeline, use a Classifier Model (DistilBERT).
CATEGORIES = {
    "MATH": ["calculate", "solve", "math", "equation", "+", "-", "*", "/"],
    "CODE": ["python", "function", "code", "def ", "class ", "program"],
    "CREATIVE": ["story", "poem", "write", "describe", "imagine", "summary"]
}

def categorize_prompt(prompt: str) -> str:
    prompt_lower = prompt.lower()
    scores = {cat: 0 for cat in CATEGORIES}
    
    for cat, keywords in CATEGORIES.items():
        for kw in keywords:
            if kw in prompt_lower:
                scores[cat] += 1
                
    # Default to Creative if no strong signal
    best_cat = max(scores, key=scores.get)
    return best_cat if scores[best_cat] > 0 else "CREATIVE"

# --- Rubric Functions ---

def score_math(completion: str, expected_answer: str) -> float:
    # 1. Structure Check
    if "<reasoning>" not in completion: return 0.0
    
    # 2. Correctness
    # Extract <answer> block
    match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if not match: return 0.0
    
    pred = match.group(1).strip()
    return 1.0 if pred == expected_answer.strip() else 0.0

def score_code(completion: str, expected_code: str) -> float:
    score = 0.0
    # 1. Logic Recall
    if expected_code.strip() in completion: score += 0.5
    
    # 2. Syntax Check
    code_match = re.search(r"```python(.*?)```", completion, re.DOTALL)
    if code_match:
        try:
            compile(code_match.group(1), "<string>", "exec")
            score += 0.5
        except: pass
    return score

def score_creative(completion: str) -> float:
    score = 0.0
    # 1. Length (Reward substance)
    if len(completion.split()) > 50: score += 0.3
    if len(completion.split()) > 150: score += 0.2
    
    # 2. Formatting (Structure)
    if "<reasoning>" in completion: score += 0.3
    
    # 3. Tone (Negative constraints)
    if "As an AI" not in completion: score += 0.2
    
    return min(1.0, score)

class EvaluationJudge:
    def __init__(self):
        self.stats = {"MATH": [], "CODE": [], "CREATIVE": []}

    def judge_sample(self, prompt: str, completion: str, expected: str = None):
        category = categorize_prompt(prompt)
        score = 0.0
        
        if category == "MATH" and expected:
            score = score_math(completion, expected)
        elif category == "CODE" and expected:
            score = score_code(completion, expected)
        else:
            score = score_creative(completion)
            
        self.stats[category].append(score)
        return category, score

    def print_report(self):
        print("\n=== Evaluation Report ===")
        for cat, scores in self.stats.items():
            if not scores:
                print(f"{cat}: No samples")
                continue
            avg = sum(scores) / len(scores)
            print(f"{cat}: Avg Score = {avg:.2f} (n={len(scores)})")
            
        # Recommendation
        lowest_cat = min(self.stats, key=lambda k: sum(self.stats[k])/len(self.stats[k]) if self.stats[k] else 999)
        print("\n--- Strategy Recommendation ---")
        print(f"Your weakest area is {lowest_cat}. Increase dataset size or weight for {lowest_cat}.")

if __name__ == "__main__":
    judge = EvaluationJudge()
    
    # Mock Data Test
    samples = [
        ("Solve 2+2", "<reasoning>2+2 is 4</reasoning><answer>4</answer>", "4"),
        ("Write a python function", "def add(a,b): return a+b", "def add(a,b): return a+b"),
        ("Write a story", "Once upon a time there was a cat.", None)
    ]
    
    for p, c, e in samples:
        cat, s = judge.judge_sample(p, c, e)
        print(f"[{cat}] Score: {s}")
        
    judge.print_report()
