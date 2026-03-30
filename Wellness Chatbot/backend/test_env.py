from dotenv import load_dotenv
import os

load_dotenv()

print("Environment Variable Loading Test")
print("=" * 50)
print(f"GOOGLE_CLIENT_ID: {os.getenv('GOOGLE_CLIENT_ID')}")
print(f"GOOGLE_CLIENT_SECRET: {os.getenv('GOOGLE_CLIENT_SECRET')}")
print(f"MONGO_URI: {os.getenv('MONGO_URI')}")
print("=" * 50)

if not os.getenv('GOOGLE_CLIENT_ID'):
    print("❌ GOOGLE_CLIENT_ID is NOT loaded!")
else:
    print("✅ GOOGLE_CLIENT_ID is loaded correctly")

if not os.getenv('GOOGLE_CLIENT_SECRET'):
    print("❌ GOOGLE_CLIENT_SECRET is NOT loaded!")
else:
    print("✅ GOOGLE_CLIENT_SECRET is loaded correctly")
