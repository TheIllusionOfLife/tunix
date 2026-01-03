
import os
import google.generativeai as genai

def list_models():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not found in environment variables.")
        print("Please export GOOGLE_API_KEY='your_key' and run again.")
        return

    genai.configure(api_key=api_key)
    
    print("Listing available Gemini models...")
    try:
        for m in genai.list_models():
            if 'gemini' in m.name:
                print(f"- {m.name} ({m.display_name})")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_models()
