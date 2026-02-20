import os
import asyncio
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

async def test_model(client, model_id):
    print(f"\n--- Testing model: {model_id} ---")
    try:
        response = client.models.generate_content(
            model=model_id,
            contents="Hello, are you there?"
        )
        print(f"SUCCESS: {response.text[:50]}...")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

async def main():
    client = genai.Client(api_key=api_key)
    models_to_test = [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "models/gemini-2.0-flash"
    ]
    for m in models_to_test:
        if await test_model(client, m):
            print(f"Found working model: {m}")

if __name__ == "__main__":
    asyncio.run(main())
