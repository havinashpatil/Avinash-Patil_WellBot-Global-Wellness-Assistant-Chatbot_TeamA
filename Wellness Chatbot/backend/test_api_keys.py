"""
Test script to verify all API keys in .env are working.
Tests: Groq, Google Gemini, OpenAI, and Anthropic Claude APIs.
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

results = {}

# ── 1. GROQ API KEY ──
print("=" * 50)
print("1. Testing GROQ API Key...")
print("=" * 50)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("   ❌ GROQ_API_KEY not found in .env")
    results["Groq"] = "NOT SET"
else:
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=10
        )
        reply = res.choices[0].message.content.strip()
        print(f"   ✅ GROQ API is WORKING!")
        print(f"   Response: {reply}")
        results["Groq"] = "WORKING"
    except Exception as e:
        print(f"   ❌ GROQ API FAILED: {e}")
        results["Groq"] = f"FAILED: {e}"

# ── 2. GOOGLE GEMINI API KEY ──
print("\n" + "=" * 50)
print("2. Testing GOOGLE / GEMINI API Key...")
print("=" * 50)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GOOGLE_API_KEY:
    print(f"   Found GOOGLE_API_KEY in .env")
    api_key_to_test = GOOGLE_API_KEY
elif GEMINI_API_KEY:
    print(f"   Found GEMINI_API_KEY in .env")
    api_key_to_test = GEMINI_API_KEY
else:
    print("   ❌ Neither GOOGLE_API_KEY nor GEMINI_API_KEY found in .env")
    api_key_to_test = None
    results["Gemini"] = "NOT SET"

# Check for mismatch: app.py reads GEMINI_API_KEY but .env has GOOGLE_API_KEY
if GOOGLE_API_KEY and not GEMINI_API_KEY:
    print("   ⚠️  WARNING: .env has GOOGLE_API_KEY but app.py reads GEMINI_API_KEY!")
    print("   ⚠️  This means Gemini will NOT work in your app until you fix the key name.")

if api_key_to_test:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key_to_test.strip())
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Say hello in one word.")
        reply = response.text.strip()
        print(f"   ✅ GEMINI API is WORKING!")
        print(f"   Response: {reply}")
        results["Gemini"] = "WORKING"
    except Exception as e:
        print(f"   ❌ GEMINI API FAILED: {e}")
        results["Gemini"] = f"FAILED: {e}"

# ── 3. OPENAI API KEY ──
print("\n" + "=" * 50)
print("3. Testing OPENAI API Key...")
print("=" * 50)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("   ❌ OPENAI_API_KEY not found in .env")
    results["OpenAI"] = "NOT SET"
else:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=10
        )
        reply = res.choices[0].message.content.strip()
        print(f"   ✅ OPENAI API is WORKING!")
        print(f"   Response: {reply}")
        results["OpenAI"] = "WORKING"
    except Exception as e:
        print(f"   ❌ OPENAI API FAILED: {e}")
        results["OpenAI"] = f"FAILED: {e}"

# ── 4. ANTHROPIC (CLAUDE) API KEY ──
print("\n" + "=" * 50)
print("4. Testing ANTHROPIC (Claude) API Key...")
print("=" * 50)
ANTHROPIC_KEY = os.getenv("API_KEY")
if not ANTHROPIC_KEY:
    print("   ❌ API_KEY (Anthropic) not found in .env")
    results["Anthropic"] = "NOT SET"
else:
    try:
        import requests
        headers = {
            "x-api-key": ANTHROPIC_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Say hello in one word."}]
        }
        resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            reply = data["content"][0]["text"].strip()
            print(f"   ✅ ANTHROPIC API is WORKING!")
            print(f"   Response: {reply}")
            results["Anthropic"] = "WORKING"
        else:
            error_msg = resp.json().get("error", {}).get("message", resp.text)
            print(f"   ❌ ANTHROPIC API FAILED (HTTP {resp.status_code}): {error_msg}")
            results["Anthropic"] = f"FAILED (HTTP {resp.status_code}): {error_msg}"
    except Exception as e:
        print(f"   ❌ ANTHROPIC API FAILED: {e}")
        results["Anthropic"] = f"FAILED: {e}"

# ── SUMMARY ──
print("\n" + "=" * 50)
print("SUMMARY OF API KEY STATUS")
print("=" * 50)
for api, status in results.items():
    icon = "✅" if status == "WORKING" else "❌"
    print(f"   {icon} {api:12s} -> {status}")
print("=" * 50)
