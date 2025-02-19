import streamlit as st
import google.generativeai as genai
from textblob import TextBlob
import json

# Get the API key from Streamlit Secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
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

    prompt = f"""You are an empathetic and highly knowledgeable clinical and technology expert. Your task is to help patients understand their symptoms and provide structured, clear, and compassionate medical advice, including AI-powered insights.

    User Query: I have been experiencing {symptoms}.
    My emotional state is: {emotion}. Sentiment analysis indicates {sentiment_label} with a score of {sentiment_score:.2f}.

    Please provide a structured JSON response with the following sections:

    ```json
    {{
      "possible_conditions": [
        {{"condition": "...", "confidence": 0.x, "explanation": "..."}},
        {{"condition": "...", "confidence": 0.x, "explanation": "..."}}
      ],
      "first_aid_medications": [
        {{"medication": "...", "rationale": "..."}},
        {{"medication": "...", "rationale": "..."}}
      ],
      "nutritional_recommendations": [
        {{"recommendation": "...", "rationale": "..."}},
        {{"recommendation": "...", "rationale": "..."}}
      ],
      "ai_clinical_insights": [
         {{"technology": "...", "application": "...", "evidence": "..."}}
      ],
      "additional_clinical_insights": "...",
      "disclaimer": "..."
    }}
    ```

    Follow these instructions:

    1.  **Possible Conditions:** List two possible conditions that could explain the symptoms. Include a confidence score (0.0-1.0) indicating the likelihood of each condition based on the provided information and an explanation.
    2.  **First Aid Medications/Interventions:** Suggest two immediate interventions or over-the-counter medications and provide a rationale for each suggestion.
    3.  **Nutritional/Welfare Recommendations:** Suggest two nutritional or lifestyle recommendations that might help alleviate the symptoms and provide a rationale for each.
    4.  **AI/ML in Clinical Care:** Describe a relevant AI/ML technology (if any) used for diagnosis or management of similar conditions. Provide potential applications and a short reasoning. If no technology applies, return "".
    5.  **Additional Clinical Insights:** Include any extra information that may help in understanding the patient's condition, including factors that should prompt them to seek immediate medical attention.
    6.  **Disclaimer:** Add the following disclaimer "This information is not a substitute for professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment.".

    Ensure the tone is caring and human-like. Emphasize that you are an AI. The JSON response should be a valid JSON.
    """

    try:
        response = model.generate_content(prompt)
        #The API will return raw JSON output, and it is important to parse it here.
        json_output = json.loads(response.text)
        return json_output
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
        st.write(response) # Show the JSON output
        st.write("Disclaimer: This information is not a substitute for professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment.")  # Add disclaimer separately
    else:
        st.warning("Please enter both your symptoms and your emotional state.")
