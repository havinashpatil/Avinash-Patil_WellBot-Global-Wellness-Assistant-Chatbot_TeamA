import os
import asyncio
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
model_id = "gemini-2.0-flash-exp-image-generation"

async def test_live():
    client = genai.Client(api_key=api_key)
    print(f"Testing Gemini Live connection with model: {model_id}")
    try:
        async with client.aio.live.connect(model=model_id, config={"response_modalities": ["TEXT"]}) as session:
            print("Successfully connected to Gemini Live API!")
            await session.send("Hello!", end_of_turn=True)
            async for response in session.receive():
                if response.text:
                    print(f"Response: {response.text}")
                    break
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(test_live())
