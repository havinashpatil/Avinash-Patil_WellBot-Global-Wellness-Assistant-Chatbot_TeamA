import os
import csv
import json as json_module
from io import StringIO
import warnings
warnings.filterwarnings("ignore")
import hashlib
import secrets
import requests as http_requests
from flask import Flask, request, jsonify, redirect, session, send_from_directory, url_for
from flask_cors import CORS
from datetime import datetime
from openai import OpenAI
from groq import Groq
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
from pymongo import MongoClient
import aiml
import google.generativeai as genai
from PIL import Image
import io
import base64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), 'frontend')

load_dotenv(os.path.join(BASE_DIR, '.env'))

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
feedback_col = db.feedback
issues_col = db.issues # New collection for login issues
error_logs_col = db.error_logs  # AI error tracking
admin_logs_col = db.admin_logs  # Admin action tracking

# Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client_groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Gemini setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    gemini_model = None

# Ollama setup
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava")

def ask_ollama(prompt):
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    response = http_requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["response"]

def ask_ollama_vision(prompt, image_base64):
    payload = {
        "model": OLLAMA_VISION_MODEL,
        "prompt": prompt,
        "stream": False,
        "images": [image_base64]
    }
    response = http_requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["response"]

# OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Initialize AIML
kernel = aiml.Kernel()
aiml_path = os.path.join(BASE_DIR, "wellness.aiml")
if os.path.exists(aiml_path):
    kernel.learn(aiml_path)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def safety_check(message):
    urgent_keywords = ["suicide", "self harm", "kill myself", "end my life"]
    for word in urgent_keywords:
        if word in message.lower():
            return "I'm concerned about what you're sharing. Please reach out to a professional or a crisis helpline immediately."
    return None

# ============================================================
# MEDICAL KNOWLEDGE BASE (WHO-aligned)
# ============================================================
MEDICAL_KB = {
    "fever": {
        "symptoms": ["high temperature", "chills", "sweating", "fatigue"],
        "precautions": ["drink plenty of fluids", "rest", "take paracetamol if needed", "use a cool compress"],
        "doctor": "If fever exceeds 103°F (39.4°C) or lasts more than 3 days",
        "source": "WHO"
    },
    "cold": {
        "symptoms": ["runny nose", "sneezing", "sore throat", "mild cough"],
        "precautions": ["rest", "stay hydrated", "warm soups", "avoid cold air"],
        "doctor": "If symptoms persist beyond 10 days or breathing becomes difficult",
        "source": "WHO"
    },
    "cough": {
        "symptoms": ["dry or wet cough", "chest discomfort", "throat irritation"],
        "precautions": ["honey with warm water", "steam inhalation", "stay hydrated", "avoid smoke"],
        "doctor": "If cough lasts more than 3 weeks or blood is present",
        "source": "WHO"
    },
    "headache": {
        "symptoms": ["pain in head or neck", "sensitivity to light", "nausea"],
        "precautions": ["rest in a quiet dark room", "drink water", "mild pain reliever", "avoid screen time"],
        "doctor": "If headache is sudden and severe or accompanied by vision changes",
        "source": "WHO"
    },
    "stress": {
        "symptoms": ["irritability", "fatigue", "difficulty concentrating", "muscle tension"],
        "precautions": ["deep breathing exercises", "regular exercise", "adequate sleep", "limit caffeine"],
        "doctor": "If stress interferes with daily functioning for more than 2 weeks",
        "source": "WHO"
    },
    "anxiety": {
        "symptoms": ["excessive worry", "rapid heartbeat", "shortness of breath", "restlessness"],
        "precautions": ["mindfulness meditation", "regular physical activity", "limit caffeine", "talk to someone"],
        "doctor": "If anxiety is severe, constant, or causing panic attacks",
        "source": "WHO"
    },
    "diabetes": {
        "symptoms": ["frequent urination", "excessive thirst", "blurred vision", "slow healing wounds"],
        "precautions": ["healthy balanced diet", "regular exercise", "monitor blood sugar", "limit sugar intake"],
        "doctor": "Consult regularly; seek immediate care if blood sugar is very high or low",
        "source": "WHO"
    },
    "hypertension": {
        "symptoms": ["headache", "dizziness", "shortness of breath", "nosebleeds"],
        "precautions": ["low sodium diet", "regular exercise", "maintain healthy weight", "avoid smoking"],
        "doctor": "If blood pressure is consistently above 140/90 mmHg",
        "source": "WHO"
    },
    "fatigue": {
        "symptoms": ["persistent tiredness", "lack of energy", "difficulty concentrating", "muscle weakness"],
        "precautions": ["get 7-9 hours of sleep", "balanced diet", "regular light exercise", "stay hydrated"],
        "doctor": "If fatigue is severe and unexplained for more than 2 weeks",
        "source": "WHO"
    },
    "nausea": {
        "symptoms": ["upset stomach", "urge to vomit", "dizziness", "loss of appetite"],
        "precautions": ["eat small bland meals", "stay hydrated", "ginger tea", "avoid strong smells"],
        "doctor": "If nausea is accompanied by severe abdominal pain or lasts more than 48 hours",
        "source": "WHO"
    }
}

def get_kb_response(query):
    """Check Medical Knowledge Base for a matching condition."""
    q = query.lower()
    for disease, data in MEDICAL_KB.items():
        if disease in q or any(sym in q for sym in data["symptoms"]):
            return (
                f"📋 **{disease.title()} Information** *(Source: {data['source']})*\n\n"
                f"**Common Symptoms:** {', '.join(data['symptoms'])}\n\n"
                f"**Precautions:** {', '.join(data['precautions'])}\n\n"
                f"**When to See a Doctor:** {data['doctor']}\n\n"
                f"⚠️ This is general information only. Always consult a qualified healthcare professional."
            ), disease, "kb"
    return None, None, None

def detect_intent(message):
    """Classify the intent of a user message."""
    msg = message.lower()
    symptom_words = ["fever", "pain", "ache", "cough", "cold", "headache", "nausea", "vomit",
                     "fatigue", "tired", "dizzy", "rash", "swelling", "bleed", "stress", "anxiety",
                     "diabetes", "hypertension", "sneeze", "runny nose"]
    mental_words = ["sad", "depressed", "lonely", "anxious", "worried", "mental", "emotion",
                    "mood", "stress", "overwhelmed", "hopeless", "unhappy"]
    nutrition_words = ["diet", "food", "nutrition", "calories", "vitamin", "protein", "carb",
                       "weight", "bmi", "eat", "drink", "meal", "supplement"]
    if any(w in msg for w in symptom_words):
        return "symptom"
    if any(w in msg for w in mental_words):
        return "mental"
    if any(w in msg for w in nutrition_words):
        return "nutrition"
    return "general"



@app.route('/')
def home():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/login', methods=['GET'])
def login_page():
    return send_from_directory(FRONTEND_DIR, 'login.html')

@app.route('/register', methods=['GET'])
def register_page():
    return send_from_directory(FRONTEND_DIR, 'register.html')

@app.route('/dashboard')
def dashboard():
    return send_from_directory(FRONTEND_DIR, 'dashboard.html')

@app.route('/chatbot')
def chatbot_page():
    return send_from_directory(FRONTEND_DIR, 'chatbot.html')

@app.route('/admin/dashboard')
def admin_dashboard_page():
    token = request.args.get('token')
    if not token or not users_col.find_one({"token": token, "role": "admin"}):
        return "<h1>Unauthorized</h1><p>You do not have permission to access this page.</p>", 403
    return send_from_directory(FRONTEND_DIR, 'admin_dashboard.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)

@app.route("/signup", methods=['POST'])
def signup():
    data = request.json
    try:
        if users_col.find_one({"email": data['email']}):
            return jsonify({"success": False, "error": "Email already exists"}), 400
        token = secrets.token_hex(16)
        user_doc = {
            "name": data['name'],
            "email": data['email'],
            "password": hash_password(data['password']),
            "language": data['language'],
            "role": "user",
            "token": token,
            "created_at": datetime.now()
        }
        users_col.insert_one(user_doc)
        return jsonify({"success": True, "token": token, "name": data['name'], "role": "user"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/login", methods=['POST'])
def login():
    try:
        data = request.json
        if not data or not data.get('email') or not data.get('password'):
            return jsonify({"success": False, "error": "Email and password are required"}), 400
        user = users_col.find_one({"email": data['email'], "password": hash_password(data['password'])})
        if user:
            return jsonify({"success": True, "token": user['token'], "name": user['name'], "role": user.get('role', 'user')})
        return jsonify({"success": False, "error": "Invalid credentials"}), 401
    except Exception as e:
        print(f"Login Error: {e}")
        return jsonify({"success": False, "error": "Server error during login. Please try again."}), 500

@app.route('/auth/google')
def google_auth():
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/auth/google/callback')
def google_callback():
    try:
        token = google.authorize_access_token()
        user_info = token.get('userinfo')
        if not user_info:
            return jsonify({"success": False, "error": "Failed to fetch user info"}), 400
        
        email = user_info.get('email')
        name = user_info.get('name', 'Google User')
        
        existing_user = users_col.find_one({"email": email})
        if not existing_user:
            user_token = secrets.token_hex(16)
            user_doc = {
                "name": name,
                "email": email,
                "password": "", 
                "language": "English",
                "role": "user",
                "token": user_token,
                "created_at": datetime.now(),
                "auth_provider": "google"
            }
            users_col.insert_one(user_doc)
            token_to_send = user_token
        else:
            token_to_send = existing_user['token']
            
        return redirect(f"/login?token={token_to_send}&name={name}")
    except Exception as e:
        print("Google Auth Error:", e)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    try:
        feedback_doc = {
            "user_email": data.get('email', 'Anonymous'),
            "rating": int(data.get('rating', 0)),
            "comment": data.get('comment', ''),
            "timestamp": datetime.now()
        }
        feedback_col.insert_one(feedback_doc)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/report-issue', methods=['POST'])
def report_issue():
    data = request.json
    try:
        issue_doc = {
            "email": data.get('email', 'Anonymous'),
            "issue": data.get('issue', ''),
            "status": "pending",
            "timestamp": datetime.now()
        }
        issues_col.insert_one(issue_doc)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/user_stats')
def user_stats():
    email = request.args.get('email')
    if not email:
        return jsonify({"success": False, "error": "Email required"}), 400
    try:
        chat_count = chats_col.count_documents({"user_email": email})
        # Get mood history (last 10)
        recent_chats = list(chats_col.find({"user_email": email}, {"mood": 1, "_id": 0}).sort("timestamp", -1).limit(10))
        moods = [c['mood'] for c in recent_chats if 'mood' in c]
        
        # Simple wellness tip based on most frequent mood
        tip = "Stay hydrated and take a 5-minute walk today!"
        if "Stressed" in moods or "Angry" in moods:
            tip = "Try a 2-minute deep breathing exercise to reset."
        elif "Tired" in moods:
            tip = "Ensure you're getting at least 7-8 hours of sleep."

        return jsonify({
            "success": True, 
            "chat_count": chat_count,
            "mood_history": moods,
            "daily_tip": tip
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate_api():
    data = request.json
    text = data.get('text')
    target_lang = data.get('language', 'English')
    
    if not text:
        return jsonify({"success": False, "error": "Text required"}), 400
        
    prompt = f"Translate the following healthcare-related text to {target_lang}. Return ONLY the translated text.\nText: {text}"
    
    try:
        # Use Groq for fast translation if available, else Gemini
        if client_groq:
            res = client_groq.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], max_tokens=500)
            translated = res.choices[0].message.content
        elif gemini_model:
            response = gemini_model.generate_content(prompt)
            translated = response.text
        else:
            translated = text # Fallback
            
        return jsonify({"success": True, "translated": translated.strip()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/admin/chat-logs')
def admin_chat_logs():
    token = request.args.get('token')
    if not token or not users_col.find_one({"token": token, "role": "admin"}):
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    try:
        logs = list(chats_col.find({}, {"_id": 0}).sort("timestamp", -1))
        return jsonify({"success": True, "logs": logs})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/admin/stats')
def admin_stats():
    token = request.args.get('token')
    if not token or not users_col.find_one({"token": token, "role": "admin"}):
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    try:
        total_users = users_col.count_documents({})
        total_chats = chats_col.count_documents({})
        all_feedback = list(feedback_col.find({}, {"_id": 0}))
        avg_rating = sum(f['rating'] for f in all_feedback) / len(all_feedback) if all_feedback else 0
        user_list = list(users_col.find({}, {"_id": 0, "password": 0, "token": 0}).sort("created_at", -1))
        reported_issues = list(issues_col.find({}, {"_id": 0}).sort("timestamp", -1))
        
        activity_data = list(chats_col.aggregate([
            {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}}, "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}, {"$limit": 7}
        ]))
        
        return jsonify({
            "success": True, "total_users": total_users, "total_questions": total_chats,
            "avg_rating": round(avg_rating, 1), "recent_feedback": all_feedback[-5:],
            "user_list": user_list, "activity_data": activity_data,
            "reported_issues": reported_issues
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/symptom_checker', methods=['POST'])
def symptom_checker():
    data = request.json
    symptom = data.get('symptom', '')
    language = data.get('language', 'English')

    if not symptom:
        return jsonify({"success": False, "error": "Please describe your symptoms."}), 400

    prompt = f"""
    A user reports the following symptoms:

    {symptom}

    Provide a structured response in {language} with:
    1. **Possible Conditions** (list 3-5 likely conditions)
    2. **Basic Precautions** (list practical self-care steps)
    3. **When to Consult a Doctor** (specific warning signs)

    End with this disclaimer:
    ⚠️ Disclaimer: This is NOT a medical diagnosis. Always consult a qualified healthcare professional for proper evaluation and treatment.
    """

    result = None

    # Priority 1: Groq
    if client_groq:
        try:
            res = client_groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant. Provide clear, structured health guidance."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600
            )
            result = res.choices[0].message.content
        except Exception as e:
            print(f"Groq Symptom Checker Error: {e}")

    # Priority 2: Gemini
    if result is None and gemini_model:
        try:
            response = gemini_model.generate_content(prompt)
            result = response.text
        except Exception as e:
            print(f"Gemini Symptom Checker Error: {e}")

    # Priority 3: Ollama
    if result is None:
        try:
            result = ask_ollama(prompt)
        except Exception as e:
            print(f"Ollama Symptom Checker Error: {e}")
            result = "Unable to analyze symptoms at this time. Please try again later."

    return jsonify({"success": True, "result": result})
    

@app.route('/api/diet-recommendation', methods=['POST'])
def diet_recommendation():
    data = request.json
    goal = data.get('goal', 'Balanced diet')
    language = data.get('language', 'English')

    prompt = f"""
    The user has the following health goal: {goal}

    Based on this goal, suggest a healthy, daily diet plan in {language}.
    Include the following sections:
    - **Breakfast**
    - **Lunch**
    - **Dinner**
    - **Snacks**
    - **Key Nutritional Tip**

    Focus on practical, healthy food options.
    Respond in {language}.
    """

    result = None

    # Priority 1: Groq
    if client_groq:
        try:
            res = client_groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a professional nutritionist expert. Provide clear, structured healthy diet advice."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600
            )
            result = res.choices[0].message.content
        except Exception as e:
            print(f"Groq Diet Error: {e}")

    # Priority 2: Gemini
    if result is None and gemini_model:
        try:
            response = gemini_model.generate_content(prompt)
            result = response.text
        except Exception as e:
            print(f"Gemini Diet Error: {e}")

    # Priority 3: Ollama
    if result is None:
        try:
            result = ask_ollama(prompt)
        except Exception as e:
            print(f"Ollama Diet Error: {e}")
            result = "Unable to generate diet recommendations at this time."

    return jsonify({"success": True, "recommendation": result})


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    user_mood = data.get('mood', 'Neutral')
    user_email = data.get('email', 'Anonymous')
    user_name = data.get('name', 'Guest')
    language = data.get('language', 'English')
    chat_mode = data.get('mode', 'wellness')
    image_data = data.get('image') # Base64 image data

    # Symptom keyword detection - suggest symptom checker
    symptom_keywords = ["fever", "cough", "headache", "pain", "sore throat", "nausea", "vomiting", "dizziness", "rash", "fatigue", "chest pain", "breathing"]
    suggest_checker = any(kw in user_message.lower() for kw in symptom_keywords)

    # AI Mood Detection (Simple sentiment override)
    detected_mood = user_mood
    negative_words = ["sad", "angry", "stressed", "unhappy", "pain", "bad", "depressed"]
    if any(word in user_message.lower() for word in negative_words):
        detected_mood = "Concerned"

    # Defaults for intent tracking (overridden in text chat branch)
    intent = "general"
    kb_match = None
    response_source = "llm"


    warning = safety_check(user_message)
    if warning: return jsonify({"reply": warning})

    try:
        ai_model_used = "Unknown"

        # Check for image (Vision Analysis)
        if image_data:
            response_source = "vision"
            # Priority 1: Ollama Vision (LLaVA/Molmo)
            try:
                temp_image_data = image_data
                if "," in temp_image_data:
                    temp_image_data = temp_image_data.split(",")[1]
                
                vision_prompt = "You are a professional medical assistant. Thoroughly read and analyze the provided medical document, lab report, or prescription image. Extract all key information including test names, results, reference ranges, diagnoses, medications, dosages, and any instructions or doctor's notes mentioned. Provide a comprehensive summary of the findings in easy-to-understand language. Read all the text in the report carefully. User message: " + user_message
                bot_reply = ask_ollama_vision(vision_prompt, temp_image_data)
                ai_model_used = "Ollama-Vision"
            except Exception as e:
                print(f"Ollama Vision Error: {e}")
                error_logs_col.insert_one({"model": "Ollama-Vision", "error": str(e), "timestamp": datetime.now()})
                bot_reply = None

            # Priority 2: Groq Vision
            if bot_reply is None and client_groq:
                try:
                    temp_image_data = image_data
                    if "," in temp_image_data:
                        temp_image_data = temp_image_data.split(",")[1]
                    
                    vision_prompt = "You are a professional medical assistant. Thoroughly read and analyze the provided medical document, lab report, or prescription image. Extract all key information including test names, results, reference ranges, diagnoses, medications, dosages, and any instructions or doctor's notes mentioned. Provide a comprehensive summary of the findings in easy-to-understand language. Read all the text in the report carefully. User message: " + user_message
                    
                    completion = client_groq.chat.completions.create(
                        model="llama-3.2-11b-vision-preview",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": vision_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{temp_image_data}",
                                        },
                                    },
                                ],
                            }
                        ],
                        max_tokens=1024,
                    )
                    bot_reply = completion.choices[0].message.content
                    ai_model_used = "Groq-Vision"
                except Exception as e:
                    print(f"Groq Vision Error: {e}")
                    error_logs_col.insert_one({"model": "Groq-Vision", "error": str(e), "timestamp": datetime.now()})
                    bot_reply = None

            # Priority 3: Gemini Vision (Fallback)
            if bot_reply is None and gemini_model:
                try:
                    if "," in image_data:
                        image_data = image_data.split(",")[1]
                    img_bytes = base64.b64decode(image_data)
                    img = Image.open(io.BytesIO(img_bytes))
                    vision_prompt = "You are a professional medical assistant. Thoroughly read and analyze the provided medical document, lab report, or prescription image. Extract all key information including test names, results, reference ranges, diagnoses, medications, dosages, and any instructions or doctor's notes mentioned. Provide a comprehensive summary of the findings in easy-to-understand language. Read all the text in the report carefully. User message: " + user_message
                    response = gemini_model.generate_content([vision_prompt, img])
                    bot_reply = response.text
                    ai_model_used = "Gemini-Vision"
                except Exception as e:
                    print(f"Gemini Vision Error: {e}")
                    error_logs_col.insert_one({"model": "Gemini-Vision", "error": str(e), "timestamp": datetime.now()})

            # Priority 4: OpenAI Vision (Fallback)
            if bot_reply is None and client_openai:
                try:
                    vision_prompt = "You are a professional medical assistant. Thoroughly read and analyze the provided medical document, lab report, or prescription image. Extract all key information including test names, results, reference ranges, diagnoses, medications, dosages, and any instructions or doctor's notes mentioned. Provide a comprehensive summary of the findings in easy-to-understand language. Read all the text in the report carefully. User message: " + user_message
                    response = client_openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": vision_prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                                ],
                            }
                        ],
                        max_tokens=500,
                    )
                    bot_reply = response.choices[0].message.content
                    ai_model_used = "OpenAI-Vision"
                except Exception as e:
                    error_logs_col.insert_one({"model": "OpenAI-Vision", "error": str(e), "timestamp": datetime.now()})

            if bot_reply is None:
                bot_reply = "Vision features are currently unavailable."
                ai_model_used = "None"
        else:
            # Detect intent
            intent = detect_intent(user_message)
            kb_match = None
            response_source = "llm"

            # Normal text chat — try AIML first
            aiml_response = kernel.respond(user_message.upper())
            if aiml_response:
                bot_reply = aiml_response
                ai_model_used = "AIML"
                response_source = "aiml"
            else:
                # Try Medical Knowledge Base before LLM
                kb_reply, kb_match, response_source = get_kb_response(user_message)
                if kb_reply:
                    bot_reply = kb_reply
                    ai_model_used = "KB"
                else:
                    response_source = "llm"
                    system_prompt = "You are an empathetic wellness assistant named WellBot."
                    if chat_mode == "mental":
                        system_prompt = "You are a supportive mental health assistant. Focus on emotional well-being and listening."
                    elif chat_mode == "nutrition":
                        system_prompt = "You are a professional nutrition expert. Focus on diet, vitamins, and healthy eating habits."
                    elif chat_mode == "fitness":
                        system_prompt = "You are an energetic fitness coach. Focus on exercise, movement, and physical strength."

                    full_prompt = f"{system_prompt} The user's mood is {detected_mood}. User prefers {language}. Respond in {language}. User says: {user_message}. Keep under 100 words."
                    bot_reply = None

                    if client_groq:
                        try:
                            res = client_groq.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": full_prompt}], max_tokens=200)
                            bot_reply = res.choices[0].message.content
                            ai_model_used = "Groq"
                        except Exception as e:
                            print(f"Groq Chat Error: {e}")
                            error_logs_col.insert_one({"model": "Groq", "error": str(e), "timestamp": datetime.now()})

                    # Priority 2: Gemini
                    if bot_reply is None and gemini_model:
                        try:
                            response = gemini_model.generate_content(full_prompt)
                            bot_reply = response.text
                            ai_model_used = "Gemini"
                        except Exception as e:
                            print(f"Gemini Error: {e}")
                            error_logs_col.insert_one({"model": "Gemini", "error": str(e), "timestamp": datetime.now()})

                    # Priority 3: Ollama
                    if bot_reply is None:
                        try:
                            bot_reply = ask_ollama(full_prompt)
                            ai_model_used = "Ollama"
                        except Exception as e:
                            error_logs_col.insert_one({"model": "Ollama", "error": str(e), "timestamp": datetime.now()})
                            bot_reply = "I'm having trouble connecting right now."
                            ai_model_used = "None"

        # Check for crisis content and flag it
        is_crisis = safety_check(user_message) is not None

        # Store intent, source, kb_match for admin monitoring
        chats_col.insert_one({
            "user_email": user_email, "user_name": user_name,
            "user_message": user_message, "bot_response": bot_reply,
            "mood": detected_mood, "mode": chat_mode, "language": language,
            "has_image": bool(image_data), "ai_model": ai_model_used,
            "is_crisis": is_crisis,
            "intent": intent if not image_data else "prescription",
            "response_source": response_source if not image_data else "vision",
            "kb_match": kb_match,
            "timestamp": datetime.now()
        })
        return jsonify({"reply": bot_reply, "suggest_symptom_checker": suggest_checker, "source": response_source if not image_data else "vision"})
    except Exception as e:
        print("Chat Error:", e)
        return jsonify({"reply": "Server error."}), 500


# ==========================================
# ADMIN API ENDPOINTS - Enterprise Dashboard
# ==========================================

def admin_auth_check():
    """Helper to verify admin token from query params."""
    token = request.args.get('token')
    if not token or not users_col.find_one({"token": token, "role": "admin"}):
        return False
    return True


@app.route('/api/admin/ai-decisions')
def admin_ai_decisions():
    if not admin_auth_check():
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    try:
        decisions = list(chats_col.find(
            {},
            {"_id": 0, "user_email": 1, "user_message": 1, "bot_response": 1,
             "intent": 1, "response_source": 1, "kb_match": 1,
             "ai_model": 1, "is_crisis": 1, "timestamp": 1}
        ).sort("timestamp", -1).limit(100))

        # Source distribution summary
        source_counts = {}
        intent_counts = {}
        for d in decisions:
            s = d.get("response_source", "llm")
            source_counts[s] = source_counts.get(s, 0) + 1
            i = d.get("intent", "general")
            intent_counts[i] = intent_counts.get(i, 0) + 1

        kb_hits = source_counts.get("kb", 0)
        llm_hits = source_counts.get("llm", 0) + source_counts.get("Groq", 0) + source_counts.get("Gemini", 0)

        return jsonify({
            "success": True,
            "decisions": decisions,
            "source_counts": source_counts,
            "intent_counts": intent_counts,
            "kb_hits": kb_hits,
            "llm_hits": llm_hits,
            "total": len(decisions)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/admin/chatbot-stats')
def admin_chatbot_stats():
    if not admin_auth_check():
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    try:
        total_chats = chats_col.count_documents({})
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_chats = chats_col.count_documents({"timestamp": {"$gte": today}})
        images_analyzed = chats_col.count_documents({"has_image": True})
        images_success = chats_col.count_documents({"has_image": True, "ai_model": {"$ne": "None"}})
        images_failed = images_analyzed - images_success
        crisis_count = chats_col.count_documents({"is_crisis": True})

        # Language usage
        lang_data = list(chats_col.aggregate([
            {"$group": {"_id": "$language", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]))

        return jsonify({
            "success": True,
            "total_chats": total_chats,
            "today_chats": today_chats,
            "images_analyzed": images_analyzed,
            "images_success": images_success,
            "images_failed": images_failed,
            "crisis_count": crisis_count,
            "language_stats": lang_data
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/admin/ai-usage')
def admin_ai_usage():
    if not admin_auth_check():
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    try:
        model_data = list(chats_col.aggregate([
            {"$group": {"_id": "$ai_model", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]))
        return jsonify({"success": True, "model_usage": model_data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/admin/health-queries')
def admin_health_queries():
    if not admin_auth_check():
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    try:
        health_keywords = ["headache", "fever", "stress", "anxiety", "depression", "cough", "cold",
                           "pain", "fatigue", "insomnia", "nausea", "dizziness", "allergy",
                           "diabetes", "blood pressure", "heart", "weight", "diet", "exercise"]
        keyword_counts = {}
        recent_messages = list(chats_col.find({}, {"user_message": 1, "_id": 0}).sort("timestamp", -1).limit(500))
        for msg_doc in recent_messages:
            msg = msg_doc.get("user_message", "").lower()
            for kw in health_keywords:
                if kw in msg:
                    keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

        sorted_queries = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        return jsonify({"success": True, "top_queries": [{"query": q, "count": c} for q, c in sorted_queries]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/admin/crisis-alerts')
def admin_crisis_alerts():
    if not admin_auth_check():
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    try:
        crisis_messages = list(chats_col.find(
            {"is_crisis": True},
            {"_id": 0, "user_email": 1, "user_message": 1, "timestamp": 1}
        ).sort("timestamp", -1).limit(20))
        return jsonify({"success": True, "alerts": crisis_messages})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/admin/system-health')
def admin_system_health():
    if not admin_auth_check():
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    services = []

    # MongoDB
    try:
        client_db.admin.command('ping')
        services.append({"name": "MongoDB", "status": "online", "icon": "database"})
    except:
        services.append({"name": "MongoDB", "status": "offline", "icon": "database"})

    # Groq
    if client_groq:
        try:
            client_groq.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": "ping"}], max_tokens=5)
            services.append({"name": "Groq API", "status": "online", "icon": "bolt"})
        except:
            services.append({"name": "Groq API", "status": "error", "icon": "bolt"})
    else:
        services.append({"name": "Groq API", "status": "not_configured", "icon": "bolt"})

    # Gemini
    if gemini_model:
        try:
            gemini_model.generate_content("ping")
            services.append({"name": "Gemini API", "status": "online", "icon": "gem"})
        except:
            services.append({"name": "Gemini API", "status": "error", "icon": "gem"})
    else:
        services.append({"name": "Gemini API", "status": "not_configured", "icon": "gem"})

    # Ollama
    try:
        r = http_requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            services.append({"name": "Ollama (Local)", "status": "online", "icon": "server"})
        else:
            services.append({"name": "Ollama (Local)", "status": "error", "icon": "server"})
    except:
        services.append({"name": "Ollama (Local)", "status": "offline", "icon": "server"})

    # OpenAI
    if client_openai:
        services.append({"name": "OpenAI API", "status": "configured", "icon": "brain"})
    else:
        services.append({"name": "OpenAI API", "status": "not_configured", "icon": "brain"})

    # Error logs count
    recent_errors = error_logs_col.count_documents({})

    return jsonify({"success": True, "services": services, "total_errors": recent_errors})


@app.route('/api/admin/error-logs')
def admin_error_logs():
    if not admin_auth_check():
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    try:
        logs = list(error_logs_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(50))
        return jsonify({"success": True, "logs": logs})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/admin/test-chat', methods=['POST'])
def admin_test_chat():
    if not admin_auth_check():
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    try:
        data = request.json
        message = data.get('message', '')
        if not message:
            return jsonify({"success": False, "error": "Message required"}), 400

        prompt = f"You are WellBot, an empathetic wellness assistant. User says: {message}. Keep under 100 words."
        reply = None
        model_used = "None"

        if client_groq:
            try:
                res = client_groq.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], max_tokens=200)
                reply = res.choices[0].message.content
                model_used = "Groq"
            except: pass

        if reply is None and gemini_model:
            try:
                response = gemini_model.generate_content(prompt)
                reply = response.text
                model_used = "Gemini"
            except: pass

        if reply is None:
            try:
                reply = ask_ollama(prompt)
                model_used = "Ollama"
            except:
                reply = "AI services are currently unavailable."

        admin_logs_col.insert_one({"action": "test_chat", "message": message, "timestamp": datetime.now()})
        return jsonify({"success": True, "reply": reply, "model": model_used})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/admin/user-activity')
def admin_user_activity():
    if not admin_auth_check():
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    try:
        users = list(users_col.find({}, {"_id": 0, "password": 0, "token": 0}))
        activity = []
        for user in users:
            email = user.get('email', '')
            chat_count = chats_col.count_documents({"user_email": email})
            image_count = chats_col.count_documents({"user_email": email, "has_image": True})
            last_chat = chats_col.find_one({"user_email": email}, sort=[("timestamp", -1)])
            last_active = last_chat["timestamp"] if last_chat else user.get("created_at")
            activity.append({
                "name": user.get("name", "Unknown"),
                "email": email,
                "role": user.get("role", "user"),
                "chat_count": chat_count,
                "image_count": image_count,
                "last_active": last_active,
                "created_at": user.get("created_at")
            })
        activity.sort(key=lambda x: x.get("chat_count", 0), reverse=True)
        return jsonify({"success": True, "users": activity})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/admin/export/<export_type>')
def admin_export(export_type):
    if not admin_auth_check():
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    fmt = request.args.get('format', 'json')

    try:
        if export_type == 'chats':
            data = list(chats_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(1000))
        elif export_type == 'users':
            data = list(users_col.find({}, {"_id": 0, "password": 0, "token": 0}))
        elif export_type == 'feedback':
            data = list(feedback_col.find({}, {"_id": 0}))
        else:
            return jsonify({"success": False, "error": "Invalid export type"}), 400

        # Convert datetime objects to strings
        for item in data:
            for key, value in item.items():
                if isinstance(value, datetime):
                    item[key] = value.isoformat()

        admin_logs_col.insert_one({"action": f"export_{export_type}", "format": fmt, "timestamp": datetime.now()})

        if fmt == 'csv':
            if not data:
                return jsonify({"success": True, "csv": ""})
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
            return jsonify({"success": True, "csv": output.getvalue(), "filename": f"{export_type}_export.csv"})
        else:
            return jsonify({"success": True, "data": data, "filename": f"{export_type}_export.json"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

