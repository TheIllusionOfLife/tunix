
import re
import logging
from typing import List, Optional
try:
    import sympy
except ImportError:
    sympy = None

logger = logging.getLogger(__name__)

# --- RegEx Patterns ---
reasoning_pattern = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

def structure_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """
    Reward for following the <reasoning>...<answer>... format.
    score = 1.0 if both tags present, 0.5 if partial, 0.0 if missing.
    """
    scores = []
    for completion in completions:
        # Check for tags
        has_reasoning_start = "<reasoning>" in completion
        has_reasoning_end = "</reasoning>" in completion
        has_answer_start = "<answer>" in completion
        has_answer_end = "</answer>" in completion
        
        score = 0.0
        if has_reasoning_start and has_reasoning_end: score += 0.4
        if has_answer_start and has_answer_end: score += 0.4
        
        # Check ordering: reasoning before answer
        if has_reasoning_end and has_answer_start:
            if completion.find("</reasoning>") < completion.find("<answer>"):
                score += 0.2
        
        # Penalize if answer is empty
        match = answer_pattern.search(completion)
        if match:
            ans_text = match.group(1).strip()
            if not ans_text:
                score -= 0.1 # Slight penalty for empty answer tags
        
        scores.append(min(1.0, max(0.0, score)))
    return scores

def strict_structure_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """
    Strict binary reward: 1.0 if perfectly formatted, 0.0 otherwise.
    """
    scores = []
    for completion in completions:
        match_r = reasoning_pattern.search(completion)
        match_a = answer_pattern.search(completion)
        if match_r and match_a:
             # Ensure reasoning comes before answer
            if completion.find("</reasoning>") < completion.find("<answer>"):
                scores.append(1.0)
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)
    return scores

def _is_numeric_equivalent(pred: str, truth: str) -> bool:
    """
    Checks if prediction and truth are numerically equivalent using SymPy or float casting.
    Handles '1,000' vs '1000', '1/2' vs '0.5', etc.
    """
    # Clean strings
    pred_clean = pred.replace(",", "").strip()
    truth_clean = truth.replace(",", "").strip()
    
    # 1. Simple string match
    if pred_clean == truth_clean:
        return True
        
    # 2. Float conversion
    try:
        if abs(float(pred_clean) - float(truth_clean)) < 1e-6:
            return True
    except ValueError:
        pass
        
    # 3. SymPy simplification (for fractions/expressions)
    if sympy:
        try:
            diff = sympy.simplify(f"({pred_clean}) - ({truth_clean})")
            if diff == 0:
                return True
        except Exception:
            pass # SymPy failed to parse
            
    return False

def correctness_reward(prompts: List[str], completions: List[str], answer: List[str], **kwargs) -> List[float]:
    """
    Reward for verifiable domains (Math).
    Extracts answer from <answer> tags and compares with ground truth `answer`.
    """
    scores = []
    for completion, true_answer in zip(completions, answer):
        match = answer_pattern.search(completion)
        if not match:
            scores.append(0.0)
            continue
        
        generated_ans = match.group(1).strip()
        
        # Check equivalence
        if _is_numeric_equivalent(generated_ans, true_answer):
            scores.append(1.0)
        else:
            scores.append(0.0)
    return scores

def code_correctness_reward(prompts: List[str], completions: List[str], answer: List[str], **kwargs) -> List[float]:
    """
    Reward for Coding (MBPP).
    1. Checks if the completion contains the ground truth code (Recall).
    2. Checks if the generated code is valid Python syntax.
    """
    scores = []
    for completion, true_code in zip(completions, answer):
        score = 0.0
        
        # 1. Recall Check (did they output the core logic?)
        if true_code.strip() in completion:
            score += 0.5
            
        # 2. Syntax Check
        # Extract code block
        code_match = re.search(r"```python(.*?)```", completion, re.DOTALL)
        if code_match:
            generated_code = code_match.group(1)
            try:
                compile(generated_code, "<string>", "exec")
                score += 0.5 # Valid syntax
            except SyntaxError:
                pass
        
        scores.append(score)
    return scores

# Map of reward functions
REWARD_FUNCTIONS = {
    "structure": structure_reward,
    "strict_structure": strict_structure_reward,
    "correctness": correctness_reward,
    "math": correctness_reward, # Alias
    "code": code_correctness_reward
}
