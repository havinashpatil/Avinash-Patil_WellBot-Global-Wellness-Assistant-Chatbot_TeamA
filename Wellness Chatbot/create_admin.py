from pymongo import MongoClient
import os
import secrets
from datetime import datetime
import hashlib
from dotenv import load_dotenv

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

load_dotenv()
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/wellbot")
client = MongoClient(mongo_uri)
# Get the database name from the URI if possible
db_name = mongo_uri.split('/')[-1] if '/' in mongo_uri else 'wellbot'
db = client[db_name]
users_col = db['users']

admin_email = "admin@healthcare.ai"
admin_pass = "Admin@123!"

# Delete existing admin in wellbot to be sure
# users_col.delete_many({"role": "admin"})

existing_admin = users_col.find_one({"email": admin_email})
if not existing_admin:
    token = secrets.token_hex(16)
    user_doc = {
        "name": "Super Admin",
        "email": admin_email,
        "password": hash_password(admin_pass),
        "language": "English",
        "role": "admin",
        "token": token,
        "created_at": datetime.now()
    }
    users_col.insert_one(user_doc)
    print(f"SUCCESS: Admin created in {db_name}. Email: {admin_email}, Password: {admin_pass}")
else:
    # Update existing one just in case
    users_col.update_one({"email": admin_email}, {"$set": {"password": hash_password(admin_pass), "role": "admin"}})
    print(f"INFO: Admin updated in {db_name}. Email: {admin_email}, Password: {admin_pass}")
