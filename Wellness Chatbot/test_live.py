import requests

questions = [
    ("I feel very stressed about my exams", "Anxious"),
    ("I cannot sleep at night", "Sad"),
    ("How do I manage anxiety?", "Neutral"),
    ("I feel happy today!", "Happy"),
    ("I am feeling lonely", "Sad"),
]

print("Testing Llama responses via chatbot...\n")
for msg, mood in questions:
    try:
        r = requests.post("http://localhost:5000/chat",
                          json={"message": msg, "mood": mood}, timeout=90)
        data = r.json()
        print(f"Q [{mood}]: {msg}")
        print(f"A: {data['reply']}")
        print("-" * 60)
    except Exception as e:
        print(f"ERROR: {e}")
