import streamlit as st
import pickle
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download('stopwords')

# Load resources
with open("best_lr_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

with open("label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)

# Preprocessing function
def preprocess(text):
    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text)
    text = re.sub(r'(http|https|ftp)://[a-zA-Z0-9./]+', '', text)
    text = BeautifulSoup(text, 'lxml').get_text()
    text = " ".join(text.split())
    text = " ".join([word for word in text.lower().split() if word not in stopwords.words('english')])
    return text

# App UI config
st.set_page_config(page_title="🎬 Movie Sentiment Analyzer", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        color: #1f77b4;
        text-align: center;
    }
    .footer {
        text-align: center;
        font-size: 12px;
        color: #888;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<h1 class='title'>🎥 Movie Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.write("Write a review for your favorite movie and check if the sentiment is **Positive** or **Negative**!")

# Movie selection
movies = [
    "Inception", "Titanic", "Interstellar", "The Godfather", "The Dark Knight",
    "Forrest Gump", "The Shawshank Redemption", "Fight Club", "Avengers: Endgame", "Joker"
]
selected_movie = st.selectbox("🎬 Select a Movie", movies)

# Review input
user_review = st.text_area(f"📝 Write your review for *{selected_movie}*", height=200)

# Predict button
if st.button("🔍 Analyze Sentiment"):
    if user_review.strip() == "":
        st.warning("🚨 Please enter a review before analyzing.")
    else:
        cleaned_review = preprocess(user_review)
        vectorized_review = vectorizer.transform([cleaned_review])
        prediction_encoded = model.predict(vectorized_review)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0] if hasattr(label_encoder, "inverse_transform") else prediction_encoded

        if prediction_label == 'pos':
            st.success("✅ Positive Sentiment! You seem to have liked the movie. 🎉")
        else:
            st.error("❌ Negative Sentiment! You didn’t enjoy the movie much. 😢")

# Footer
st.markdown("<div class='footer'>Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)
