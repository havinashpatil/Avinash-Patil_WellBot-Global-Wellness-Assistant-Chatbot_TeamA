import os
from google import genai
from dotenv import load_dotenv
import json

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

def dump_models():
    client = genai.Client(api_key=api_key)
    try:
        models = list(client.models.list())
        for model in models:
            print(f"--- {model.name} ---")
            # Try to print as a dict if possible
            try:
                # Many pydantic-based objects have .dict() or .model_dump()
                if hasattr(model, 'model_dump'):
                    print(json.dumps(model.model_dump(), indent=2))
                elif hasattr(model, 'dict'):
                    print(json.dumps(model.dict(), indent=2))
                else:
                    print(f"No dump method. Vars: {vars(model)}")
            except:
                print(f"Could not dump. Attributes: {dir(model)}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    dump_models()
