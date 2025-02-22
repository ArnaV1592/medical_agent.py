import streamlit as st
import google.generativeai as genai
from textblob import TextBlob

# Get the API key from Streamlit Secrets
try:
    GOOGLE_API_KEY = st.secrets["AIzaSyDRzf5wsfJ0CynApO_nkd-PvSHW9pjQdB8"]
except KeyError:
    st.error("Please set the GOOGLE_API_KEY secret in Streamlit Cloud.")
    st.stop()  # Stop the app if the API key is not set

genai.configure(api_key=GOOGLE_API_KEY)


def analyze_sentiment(text):
    """Analyzes the sentiment of the given text using TextBlob."""
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        sentiment_label = "POSITIVE"
        sentiment_score = polarity
    elif polarity < 0:
        sentiment_label = "NEGATIVE"
        sentiment_score = abs(polarity)  # Take absolute value for score
    else:
        sentiment_label = "NEUTRAL"
        sentiment_score = 0  # Neutral score

    return sentiment_label, sentiment_score


def generate_personalized_response(symptoms, emotion, sentiment_label, sentiment_score):
    """Generates a personalized response using the LLM."""
    model = genai.GenerativeModel('gemini-pro')

    prompt = f"""You are an empathetic and highly knowledgeable clinical assistant. Your task is to help patients understand their symptoms and provide structured, clear, and compassionate medical advice.

    User Query: I have been experiencing {symptoms}.
    My emotional state is: {emotion}. Sentiment analysis indicates {sentiment_label} with a score of {sentiment_score:.2f}.

    Please provide a structured response with the following sections:

    1.  **Possible Conditions:** List two possible conditions that could explain the symptoms. Be aware that these should be only possibilities, and not certainties.
    2.  **First Aid Medications/Interventions:** Suggest two immediate interventions or over-the-counter medications.
    3.  **Nutritional/Welfare Recommendations:** Suggest two nutritional or lifestyle recommendations that might help alleviate the symptoms.
    4.  **Additional Clinical Insights:** Include any extra information that may help in understanding the patient's condition, including factors that should prompt them to seek immediate medical attention.

    Ensure the tone is caring and human-like. Emphasize that you are an AI and the user should seek professional medical advice from their doctor.
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
        st.write("Disclaimer: This information is not a substitute for professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment.") # Added Disclaimer

    else:
        st.warning("Please enter both your symptoms and your emotional state.")
