from pymongo import MongoClient
import hashlib
import os
from dotenv import load_dotenv

load_dotenv()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/wellbot")
client = MongoClient(MONGO_URI)
db = client.get_default_database()
users_col = db.users

admin_email = "avinashavinashgouda@gmail.com"
admin_pass = "CEOavinash@1"
hashed_pass = hash_password(admin_pass)

# Check if user exists
user = users_col.find_one({"email": admin_email})

if user:
    users_col.update_one(
        {"email": admin_email},
        {"$set": {"password": hashed_pass, "role": "admin", "name": "Admin Avinash"}}
    )
    print(f"User {admin_email} updated to Admin.")
else:
    users_col.insert_one({
        "name": "Admin Avinash",
        "email": admin_email,
        "password": hashed_pass,
        "role": "admin",
        "language": "English",
        "token": "admin-token-123",
        "created_at": None  # or datetime.now()
    })
    print(f"User {admin_email} created as Admin.")
