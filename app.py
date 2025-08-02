
import streamlit as st
import joblib

# Load the trained pipeline (includes both TF-IDF vectorizer and classifier)
pipeline = joblib.load("news_classifier_pipeline.pkl")

# Mapping of class index to category name
category_map = {
    1: "World News",
    2: "Sports News",
    3: "Business News",
    4: "Science or Technology News"
}

st.set_page_config(page_title="News Classifier", layout="centered")
st.title("📰 News Classification App")
st.markdown("This app uses a machine learning model to classify news content into categories.")

# User input
news_input = st.text_area("✍️ Enter news article text here:")

if st.button("🔍 Classify"):
    if news_input.strip():
        prediction = pipeline.predict([news_input])[0]
        category = category_map.get(prediction, "Unknown")
        st.success(f"**Predicted Category:** {category}")
    else:
        st.warning("Please enter some news text.")
