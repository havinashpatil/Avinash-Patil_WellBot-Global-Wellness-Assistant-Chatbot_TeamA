from fastapi import FastAPI
from typing import Optional

app = FastAPI()

@app.get("/")
async def home():
    return {"message": "Server is running! Visit /docs for the interactive UI."}

@app.get("/user/{name}")
async def greet(name: str, age: Optional[int] = None):
    # This logic processes your input and returns a structured response
    status = "an adult" if age and age >= 18 else "a minor (or age unknown)"
    
    return {
        "greeting": f"Hello, {name}!",
        "age_provided": age,
        "classification": status
    }

# Added a new route to show a functional output (Addition)
@app.get("/add/{num1}/{num2}")
async def add_numbers(num1: int, num2: int):
    sum_result = num1 + num2
    return {
        "operation": "addition",
        "inputs": [num1, num2],
        "result": sum_result
    }