import os
import warnings
warnings.filterwarnings("ignore")
import hashlib
import secrets
from flask import Flask, request, jsonify, redirect, session, send_from_directory, url_for
from flask_cors import CORS
from datetime import datetime
from openai import OpenAI
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
from pymongo import MongoClient
import aiml

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))
CORS(app, supports_credentials=True)

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/wellbot")
client_db = MongoClient(MONGO_URI)
db = client_db.get_default_database()
users_col = db.users
chats_col = db.chats

# OpenAI client with environment variable
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
# USE_TEST_MODE = OPENAI_API_KEY is None

# if OPENAI_API_KEY:
#     client_openai = OpenAI(api_key=OPENAI_API_KEY)
# else:
#     client_openai = None
#     print("⚠️  WARNING: OPENAI_API_KEY not set. Running in TEST MODE with mock responses.")

# Initialize AIML (Traditional AI)
kernel = aiml.Kernel()
if os.path.exists("wellness.aiml"):
    kernel.learn("wellness.aiml")

# Initialize Gemini (Advanced AI)
from google import genai
GEMINI_API_KEY = "AIzaSyBK01r5171V8CI27jpo-siZfEYf12twyd4" # Provided by user
client_gemini = genai.Client(api_key=GEMINI_API_KEY)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Safety filter
def safety_check(message):
    urgent_keywords = ["suicide", "self harm", "kill myself", "end my life"]
    for word in urgent_keywords:
        if word in message.lower():
            return "I'm concerned about what you're sharing. Please reach out to a professional or a crisis helpline immediately."
    return None

@app.route('/')
def home():
    return send_from_directory('.', 'login.html')

@app.route('/login')
def login_page():
    return send_from_directory('.', 'login.html')

@app.route('/register')
def register_page():
    return send_from_directory('.', 'register.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

@app.route('/dashboard')
def dashboard():
    return send_from_directory('.', 'dashboard.html')

@app.route("/signup", methods=['POST'])
def signup():
    data = request.json
    try:
        # Check if email exists
        if users_col.find_one({"email": data['email']}):
            return jsonify({"success": False, "error": "Email already exists"}), 400
            
        token = secrets.token_hex(16)
        user_doc = {
            "name": data['name'],
            "email": data['email'],
            "password": hash_password(data['password']),
            "language": data['language'],
            "token": token,
            "created_at": datetime.now()
        }
        users_col.insert_one(user_doc)
        return jsonify({"success": True, "token": token, "name": data['name']})
    except Exception as e:
        print("Signup Error:", e)
        return jsonify({"success": False, "error": "Internal server error"}), 500

@app.route("/login", methods=['POST'])
def login():
    data = request.json
    user = users_col.find_one({
        "email": data['email'],
        "password": hash_password(data['password'])
    })
    
    if user:
        return jsonify({"success": True, "token": user['token'], "name": user['name']})
    return jsonify({"success": False, "error": "Invalid credentials"}), 401

@app.route('/auth/google')
def google_login():
    redirect_uri = 'http://localhost:5000/auth/google/callback'
    return google.authorize_redirect(redirect_uri, prompt='select_account')

@app.route('/auth/google/callback')
def google_callback():
    token = google.authorize_access_token()
    user_info = token.get('userinfo')
    
    user = users_col.find_one({"email": user_info['email']})
    
    if not user:
        user_token = secrets.token_hex(16)
        user_doc = {
            "name": user_info['name'],
            "email": user_info['email'],
            "password": "",  # Google users don't have a password
            "language": "English",
            "token": user_token,
            "created_at": datetime.now()
        }
        users_col.insert_one(user_doc)
    else:
        user_token = user['token']
    
    return redirect(f'/dashboard?token={user_token}&name={user_info["name"]}')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    user_mood = data.get('mood', 'Neutral')

    warning = safety_check(user_message)
    if warning:
        return jsonify({"reply": warning})

    try:
        # B. AIML Check (Check if there's a hard-coded rule first)
        aiml_response = kernel.respond(user_message.upper()) # AIML usually requires uppercase
        if aiml_response:
            bot_reply = aiml_response
        else:
            # C. LLM Generation (Gemini)
            try:
                prompt = f"You are an empathetic wellness assistant named WellBot. The user's current mood is {user_mood}. User says: {user_message}. Be supportive, concise, and professional."
                response = client_gemini.models.generate_content(
                    model='gemini-2.0-flash', contents=prompt
                )
                bot_reply = response.text
            except Exception as e:
                print(f"Gemini Error: {e}")
                bot_reply = f"I'm having trouble connecting right now, but I'm here for you. (Error: {str(e)})"

        chat_doc = {
            "user_message": user_message,
            "bot_response": bot_reply,
            "mood": user_mood,
            "timestamp": datetime.now()
        }
        chats_col.insert_one(chat_doc)

        return jsonify({"reply": bot_reply})

    except Exception as e:
        print("Chat Error:", e)
        return jsonify({"reply": "Server error. Try again later."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
