import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
# ==========================
# Load CSV Data
# ==========================
# Resolve CSV path relative to this file so the app works regardless of the
# current working directory when executed.
base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "health_data.csv"

# As a fallback, also check the current working directory.
cwd_fallback = Path.cwd() / "health_data.csv"
if not csv_path.exists() and cwd_fallback.exists():
    csv_path = cwd_fallback

if not csv_path.exists():
    # Streamlit-friendly error and stop execution early so user sees what's wrong.
    st.error(f"Could not find 'health_data.csv'. Searched: {base_dir} and {Path.cwd()}")
    st.stop()

# Use lowercase column names to match later code that expects 'symptoms' and 'advice'.
data = pd.read_csv(csv_path, header=0, names=["symptoms", "advice"], quoting=1, on_bad_lines='skip', encoding='utf-8')



# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🩺",
    layout="centered"
)

# ==========================
# CSS Styling - Health-tech theme
# ==========================
st.markdown("""
<style>
/* App background */
.stApp { 
    background-color: #ffffff; 
    font-family: 'Segoe UI', sans-serif; 
}

/* Header and subtitle */
.title { 
    text-align: center; 
    color: #1e40af; 
    font-size: 40px; 
    font-weight: bold; 
}
.subtitle { 
    text-align: center; 
    color: #3b82f6; 
    font-size: 20px; 
    margin-bottom: 25px; 
}

/* Buttons */
.stButton>button { 
    background-color: #3b82f6; 
    color: white; 
    border-radius: 10px; 
    height: 3em; 
    font-size: 18px; 
    font-weight: bold;
}
.stButton>button:hover { 
    background-color: #60a5fa; 
}

/* Result box */
.result-box { 
    background-color: #dbeafe; 
    border-left: 6px solid #1e40af;
    border-radius: 8px; 
    padding: 1.5em; 
    text-align: center; 
    font-weight: bold; 
    color: #1e40af; 
    box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
    margin-top: 20px; 
    font-size: 20px;
}

/* Sidebar */
.sidebar .sidebar-content {
    background-color: #f1f5f9;
    color: #1e40af;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# Header
# ==========================
st.markdown("<div class='title'>🧠 AI Health Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Get personalized health advice based on your symptoms</div>", unsafe_allow_html=True)
st.write("---")

# ==========================
# User Input
# ==========================
user_input = st.text_area("Enter your symptoms (comma-separated or full sentence)")

# ==========================
# Search & Display Advice
# ==========================
# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute embeddings for all symptoms
data['embedding'] = data['symptoms'].apply(lambda x: model.encode(x.lower()))

# ==========================
# Search & Display Advice
# ==========================
if st.button("Get Advice"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter your symptoms first.")
    else:
        # Encode user input
        user_embedding = model.encode(user_input.lower())

        # Compute cosine similarity between user input and all stored symptoms
        similarities = data['embedding'].apply(lambda x: cosine_similarity([user_embedding], [x])[0][0])

        # Get the most similar symptom
        best_match_index = np.argmax(similarities)
        best_match_score = similarities.iloc[best_match_index]

        if best_match_score > 0.65:  # threshold for similarity
            advice_found = data.iloc[best_match_index]['advice']
            matched_symptom = data.iloc[best_match_index]['symptoms']
            st.markdown(f"<div class='result-box'>💡 Matched Symptom: <b>{matched_symptom}</b><br>{advice_found}</div>", unsafe_allow_html=True)
        else:
            st.info("❗ No close match found. Please consult a doctor for proper advice.")
# ==========================
# Sidebar Info
# ==========================
st.sidebar.title("About")
st.sidebar.info(
    "🩺 AI Health Assistant\n"
    "Gives advice based on symptoms from health_data.csv\n"
    "Developed by TEAM INSANE | Velammal Vidyalaya Avadi"
) 