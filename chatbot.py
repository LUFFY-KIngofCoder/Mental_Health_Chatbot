import streamlit as st
import torch
import sqlite3
from collections import deque
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai
from dotenv import load_dotenv
import os
import gdown

folder_id = "1gxwtPi8rupeMSOU6UGg9GnV_c76Xq8kv"
SAVE_PATH = "mental_health_analysis"
# ✅ Download the folder if it doesn't exist
if not os.path.exists(SAVE_PATH):
    gdown.download_folder(id = folder_id, output=SAVE_PATH, quiet=False)

# ✅ Load RoBERTa Model
roberta_model_path = SAVE_PATH
roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_path)
roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_model_path)

# ✅ Configure Gemini API
load_dotenv()
API_KEY = os.getenv("API_KEY")
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# ✅ Emotion Labels
labels = {
    0: "Anxiety", 1: "Normal", 2: "Depression", 3: "Suicidal",
    4: "Stress", 5: "Bipolar", 6: "Personality Disorder"
}

# ✅ Emotion-Based Tone Adjustments
tone_prompts = {
    "Normal": "Respond casually as a friendly chatbot.",
    "Anxiety": "Be gentle and reassuring in your response.",
    "Depression": "Offer supportive and understanding words.",
    "Suicidal": "Encourage seeking professional help and offer crisis resources.",
    "Stress": "Provide relaxation techniques and encouragement.",
    "Bipolar": "Be neutral and offer balanced responses.",
    "Personality Disorder": "Be patient and supportive."
}

# ✅ Emotion Tracking Variables
emotion_history = deque(maxlen=3)  # Stores last 3 detected emotions
threshold_confidence = 0.7  # Confidence threshold to confirm emotion

# ✅ Database Configuration
DB_FILE = "chatbot_logs.db"


# ✅ Create Table for Chat Logs
def initialize_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_input TEXT,
            detected_emotion TEXT,
            confidence REAL,
            chatbot_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


# ✅ Function to Analyze Sentiment with Confidence
def analyze_sentiment(text):
    inputs = roberta_tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = roberta_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)  # Convert to probabilities
        confidence, predicted_label = torch.max(probs, dim=1)  # Get highest confidence score

    emotion = labels[predicted_label.item()]
    confidence_score = confidence.item()

    return emotion, confidence_score


# ✅ Emotion Confirmation Logic
def confirm_emotion(new_emotion, confidence_score):
    if confidence_score >= threshold_confidence:
        emotion_history.append(new_emotion)

        # Check if all last 3 emotions are the same
        if len(emotion_history) == emotion_history.maxlen:
            if all(e == new_emotion for e in emotion_history):
                return new_emotion
    return "Normal"


# ✅ Function to Log Chat Data to SQLite
def log_chat(user_input, detected_emotion, confidence, chatbot_response):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO chat_logs (user_input, detected_emotion, confidence, chatbot_response)
        VALUES (?, ?, ?, ?)
    """, (user_input, detected_emotion, confidence, chatbot_response))
    conn.commit()
    conn.close()


# ✅ Streamlit UI
st.title("🧠 Mental Health Chatbot")
st.markdown("Chat with an AI that adapts to your emotional state.")

# Initialize Database
initialize_database()

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
user_input = st.chat_input("Type a message...")

if user_input:
    # 🔹 Step 1: Analyze Sentiment
    detected_emotion, confidence = analyze_sentiment(user_input)
    confirmed_emotion = confirm_emotion(detected_emotion, confidence)

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 🔹 Step 2: Modify Prompt for Gemini
    prompt = f"{tone_prompts.get(confirmed_emotion, 'Respond in a neutral manner.')}\nUser: {user_input}\nChatbot:"

    # 🔹 Step 3: Generate Response
    response = gemini_model.generate_content(prompt).text

    # 🔹 Step 4: Log Data to Database
    log_chat(user_input, confirmed_emotion, confidence, response)

    # Display chatbot response
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# ✅ Display Chat Logs for Review
if st.button("📂 View Chat Log"):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT user_input, detected_emotion, confidence, chatbot_response, timestamp FROM chat_logs ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()

    if rows:
        st.write("### Chat History")
        for row in rows:
            st.write(f"**[{row[4]}] User:** {row[0]}")
            st.write(f"🧠 *Emotion:* {row[1]} (Confidence: {row[2]:.2f})")
            st.write(f"🤖 **Chatbot:** {row[3]}")
            st.write("---")
    else:
        st.warning("No chat logs found!")
