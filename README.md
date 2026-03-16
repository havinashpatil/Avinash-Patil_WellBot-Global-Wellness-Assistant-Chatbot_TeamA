# Avinash-Patil_WellBot-Global-Wellness-Assistant-Chatbot_TeamAProject Overview

WellBot combines AI chatbot technology with health analytics tools to help users manage their wellness effectively.

Core Capabilities

🤖 AI Wellness Chatbot

📸 Prescription Image Analysis

🩺 Symptom Checker

🍎 Nutrition Guidance

📊 Health Monitoring Dashboard

🧑‍💻 Admin Monitoring System

🏗 System Architecture
User Interface (HTML / CSS / JS)
        ↓
Frontend Dashboard
        ↓
Flask Backend API
        ↓
AI Model Router
        ↓
Groq | Gemini | Ollama | OpenAI
        ↓
MongoDB Database

This hybrid AI architecture ensures high reliability and fallback support.

💻 Technologies Used
Frontend

HTML5

CSS3

JavaScript (ES6)

Chart.js

Google Fonts (Inter / Poppins)

Frontend Pages:

index.html
login.html
register.html
dashboard.html
chatbot.html
admin_dashboard.html
⚙ Backend

Framework:

Python

Flask

Flask-CORS

Authlib (Google OAuth)

Backend Responsibilities:

API routing

Authentication

AI request handling

Image processing

Database operations

Main File:

app.py
🧠 AI Models Integrated

WellBot uses multi-tier AI orchestration.

Model	Purpose
AIML	Rule-based chatbot responses
Groq (Llama 3.3)	Primary LLM
Gemini 1.5 Flash	Cloud fallback
Ollama	Local AI model
OpenAI GPT-4o Vision	Prescription analysis
🗄 Database

Database:

MongoDB

Collections:

users
chats
feedback
reported_issues
prescriptions

MongoDB stores:

User accounts

Chat history

Feedback

Uploaded prescriptions

Issue reports

📦 Required Python Libraries

Install dependencies:

pip install -r requirements.txt

Main dependencies:

Flask
Flask-CORS
pymongo
python-dotenv
authlib
requests
Pillow
groq
openai
google-generativeai
🔐 Environment Variables

Create .env file:

MONGO_URI=your_mongodb_uri
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
🚀 Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/yourusername/wellbot.git
cd wellbot
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Start Backend Server
python app.py

Server will run at:

http://localhost:5000
👤 User Dashboard Features

User dashboard provides multiple health tools.

Health Tools

👣 Step Counter

😊 Mood Tracker

😴 Sleep Tracker

❤️ Heart Rate Monitor

⚖ BMI Calculator

🔥 Calories Tracker

🥗 AI Diet Planner

🍎 Nutrition Guide

Includes detailed nutrition information.

Vitamins

Vitamin A

Vitamin B

Vitamin C

Vitamin D

Vitamin E

Minerals

Iron

Zinc

Magnesium

Calcium

Potassium

Proteins

Eggs

Fish

Chicken

Lentils

Beans

Tofu

Milk

📸 AI Prescription Analysis

Users can upload a prescription image.

The system:

1️⃣ Reads prescription image
2️⃣ Extracts medicine names
3️⃣ Shows dosage instructions
4️⃣ Provides medical guidance

Supported vision models:

OpenAI Vision

Gemini Vision

Ollama Vision

🧑‍💻 Admin Dashboard

Admin dashboard provides system monitoring tools.

Admin Capabilities

👥 User Management

💬 Chat Monitoring

📊 Usage Analytics

⚠ Issue Reports

🤖 Floating Chatbot

🛡 System Health Monitor

🛡 Security Features

SHA-256 password hashing

Token-based authentication

Secure .env API keys

Crisis message detection

Protected admin routes

🌍 Accessibility Features
Multi-Language Support

English

Hindi

Kannada

Spanish

Theme System

Light Theme

Dark Theme

📊 System Monitoring

Admin dashboard tracks:

Active users

AI model usage

System health

Chat statistics

🎯 Project Goal

WellBot aims to provide AI-powered healthcare assistance that improves accessibility and understanding of health information.

🏁 Conclusion

WellBot demonstrates how AI technology can enhance digital healthcare systems by integrating:

AI chatbots

Prescription analysis

Wellness tracking

Intelligent admin monitoring

⭐ WellBot makes healthcare assistance faster, smarter, and accessible for everyone.

👥 Contributors

Avinash

Harika

Sameer

Sakshi

Tejaswini

Apeksha
