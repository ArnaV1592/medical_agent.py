import streamlit as st
from transformers import pipeline
import google.generativeai as genai
import os

# Get the API key from Streamlit Secrets
try:
    GOOGLE_API_KEY = st.secrets["AIzaSyDq4f0donaT8xS8SRl82MLiM3bkVPwgGmQ"]
except KeyError:
    st.error("Please set the GOOGLE_API_KEY secret in Streamlit Cloud.")
    st.stop() # Stop the app if the API key is not set

genai.configure(api_key=GOOGLE_API_KEY)

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text):
    """Analyzes the sentiment of the given text."""
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']

def generate_personalized_response(symptoms, emotion, sentiment_label, sentiment_score):
    """Generates a personalized response using the LLM."""
    model = genai.GenerativeModel('gemini-pro')

    prompt = f"""You are an empathetic and helpful AI assistant designed to provide medical advice and support to patients.
    A patient is experiencing the following symptoms: {symptoms}.
    Their emotional state is described as: {emotion}. The sentiment analysis indicates {sentiment_label} with a score of {sentiment_score:.2f}.

    Provide a response that:
    1. Acknowledges their emotions.
    2. Offers potential explanations for their symptoms, but emphasize this is not a diagnosis.
    3. Suggests actionable steps they can take (e.g., relaxation techniques, over-the-counter medications, when to see a doctor).
    4. Uses empathetic and supportive language.

    Keep your response concise and easy to understand. Do not provide overly technical details.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the response: {e}"

# Streamlit App
st.title("Personalized AI Medical Assistant")

symptoms = st.text_area("Please describe your symptoms:")
emotion = st.text_area("Please describe how you are feeling emotionally:")

if st.button("Get Advice"):
    if symptoms and emotion:
        sentiment_label, sentiment_score = analyze_sentiment(emotion)
        response = generate_personalized_response(symptoms, emotion, sentiment_label, sentiment_score)
        st.write(response)
    else:
        st.warning("Please enter both your symptoms and your emotional state.")
