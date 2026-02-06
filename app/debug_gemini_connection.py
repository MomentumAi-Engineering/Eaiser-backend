
import os
import asyncio
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
import time

# Load env
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("GEMINI_API_KEY")
print(f"Loaded API Key: {API_KEY[:5]}...{API_KEY[-4:] if API_KEY else 'None'}")

if not API_KEY:
    print("❌ API Key missing!")
    exit(1)

genai.configure(api_key=API_KEY)

async def test_gemini():
    print("Testing Gemini Connection...")
    models_to_try = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.0-pro",
        "gemini-pro",
        "gemini-2.0-flash-exp"
    ]
    
    for model_name in models_to_try:
        print(f"\nTrying model: {model_name}...")
        try:
            model = genai.GenerativeModel(model_name)
            start = time.time()
            response = await asyncio.to_thread(
                model.generate_content, 
                "Explain traffic lights in 5 words."
            )
            duration = time.time() - start
            print(f"✅ Success with {model_name}! Response time: {duration:.2f}s")
            print(f"Response: {response.text}")
            return
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_gemini())
