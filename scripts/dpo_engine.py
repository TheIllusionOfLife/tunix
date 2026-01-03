
import os
import time
import json
import logging
from typing import List, Dict, Optional
import google.generativeai as genai
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "gemini-3-flash-preview"
DPO_SYSTEM_PROMPT = """You are an expert dataset creator for DPO (Direct Preference Optimization).
Your task is to generate pairs of (chosen, rejected) reasoning traces for a given prompt.
You MUST follow the structural rules for reasoning traces: <reasoning>...</reasoning><answer>...</answer>.

Format your output exactly as follows:
<chosen>
<reasoning>
[Insert high-quality, step-by-step reasoning here.]
</reasoning>
<answer>
[Insert correct answer here.]
</answer>
</chosen>

<rejected>
<reasoning>
[Insert flawed, vague, or superficial reasoning here.]
</reasoning>
<answer>
[Insert answer here (can be correct or incorrect depending on the flaw).]
</answer>
</rejected>
"""

class DPOEngine:
    def __init__(self, api_key: Optional[str] = None, model_name: str = DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not set.")
        else:
            genai.configure(api_key=self.api_key)
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=DPO_SYSTEM_PROMPT
        )

    def generate_pair(self, prompt: str, domain: str = "General") -> Dict:
        if not self.api_key:
             return {"question": prompt, "error": "API Key missing"}

        try:
            response = self.model.generate_content(f"Domain: {domain}\nPrompt: {prompt}")
            text = response.text
            
            # Robust parsing
            try:
                chosen_part = text.split("<chosen>")[1].split("</chosen>")[0].strip()
                rejected_part = text.split("<rejected>")[1].split("</rejected>")[0].strip()
            except IndexError:
                # Fallback or retry logic could go here
                logger.warning(f"Failed to parse tags for prompt: {prompt[:20]}")
                return {"question": prompt, "error": "Parse Error", "raw_output": text}
            
            return {
                "question": prompt,
                "domain": domain,
                "chosen": chosen_part,
                "rejected": rejected_part,
                "model": self.model.model_name
            }
        except Exception as e:
            logger.error(f"Error generating pair: {e}")
            return {"question": prompt, "error": str(e)}

    def batch_generate(self, seeds: List[Dict], output_file: str, delay: float = 1.0):
        results = []
        logger.info(f"Starting DPO batch generation for {len(seeds)} items...")
        
        for item in tqdm(seeds):
            prompt = item.get("question")
            domain = item.get("domain", "General")
            
            result = self.generate_pair(prompt, domain)
            results.append(result)
            
            with open(output_file, 'a') as f:
                f.write(json.dumps(result) + "\n")
            
            time.sleep(delay)
            
        logger.info(f"DPO Batch generation complete. Saved to {output_file}")
        return results

if __name__ == "__main__":
    # Test run
    sample_seeds = [{"domain": "Creative", "question": "Write a story about a sad cloud."}]
    engine = DPOEngine()
    engine.batch_generate(sample_seeds, "test_dpo.jsonl")
