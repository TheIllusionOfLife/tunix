
import os
import time
import json
import random
import logging
from typing import List, Dict, Optional
import google.generativeai as genai
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "gemini-3-flash-preview"
REASONING_SYSTEM_PROMPT = """You are an expert reasoning assistant.
You are given a problem or a creative prompt.
You MUST format your response as follows:
1. First, think step-by-step about how to solve or address the prompt. Enclose this thinking process in <reasoning> and </reasoning> tags.
2. Then, provide the final output or answer. Enclose this final output in <answer> and </answer> tags.

For creative writing, the "answer" is the story/poem itself.
For code, the "answer" is the code block.
For math/logic, the "answer" is the final result.
"""

class DataEngine:
    def __init__(self, api_key: Optional[str] = None, model_name: str = DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not set. Generation will fail unless set.")
        else:
            genai.configure(api_key=self.api_key)
        
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=REASONING_SYSTEM_PROMPT
        )
        logger.info(f"Initialized DataEngine with model: {self.model_name}")

    def generate_trace(self, prompt: str, domain: str = "General") -> Dict:
        """
        Generates a reasoning trace and answer for a given prompt.
        """
        if not self.api_key:
            raise ValueError("API Key missing.")

        try:
            # Add explicit instruction in the user prompt as well to enforce format
            full_prompt = (
                f"Domain: {domain}\n"
                f"Prompt: {prompt}\n\n"
                "Remember: <reasoning>...trace...</reasoning><answer>...output...</answer>"
            )
            
            response = self.model.generate_content(full_prompt)
            return {
                "question": prompt,
                "domain": domain,
                "generated_text": response.text,
                "model": self.model_name,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error generating trace for prompt '{prompt[:30]}...': {e}")
            return {"question": prompt, "domain": domain, "error": str(e)}

    def batch_generate(self, seeds: List[Dict], output_file: str, delay: float = 1.0):
        """
        Process a list of seeds [{'question':Str, 'domain':Str}] and save to jsonl.
        """
        results = []
        logger.info(f"Starting batch generation for {len(seeds)} items...")
        
        for item in tqdm(seeds):
            prompt = item.get("question")
            domain = item.get("domain", "General")
            
            result = self.generate_trace(prompt, domain)
            results.append(result)
            
            # Append line-by-line to file
            with open(output_file, 'a') as f:
                f.write(json.dumps(result) + "\n")
            
            time.sleep(delay) # Rate formatting
            
        logger.info(f"Batch generation complete. Saved to {output_file}")
        return results

# Sample Seeds (if run directly)
SAMPLE_SEEDS = [
    {"domain": "Creative Writing", "question": "Write a haiku about a robot learning to love."},
    {"domain": "Logic", "question": "If A implies B, and B implies C, does C imply A? Explain why or why not."},
    {"domain": "Science", "question": "Explain quantum entanglement to a 5-year old."},
    {"domain": "Coding", "question": "Write a python function to reverse a string without using slicing."}
]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", help="Google API Key", default=None)
    parser.add_argument("--model", help="Model name", default=DEFAULT_MODEL)
    parser.add_argument("--output", help="Output JSONL file", default="synthetic_data.jsonl")
    args = parser.parse_args()

    engine = DataEngine(api_key=args.key, model_name=args.model)
    engine.batch_generate(SAMPLE_SEEDS, args.output)
