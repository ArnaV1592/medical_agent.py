import streamlit as st
import google.generativeai as genai
from textblob import TextBlob
import json
from sentence_transformers import SentenceTransformer
import numpy as np

# Get the API key from Streamlit Secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Please set the GOOGLE_API_KEY secret in Streamlit Cloud.")
    st.stop()  # Stop the app if the API key is not set

genai.configure(api_key=GOOGLE_API_KEY)

# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define knowledge base
knowledge_base = {
    "headache": [
        "Tension headaches are often caused by stress and muscle tension.",
        "Migraines can cause intense throbbing pain and sensitivity to light and sound.",
        "Cluster headaches are severe headaches that occur in clusters, often with nasal congestion and eye tearing.",
        "AI can be used to predict migraine attacks based on physiological data.",
    ],
    "chest pain": [
        "Chest pain can be caused by heart problems, such as angina or a heart attack.",
        "It can also be caused by musculoskeletal issues, such as costochondritis.",
        "Anxiety and panic attacks can also cause chest pain.",
        "AI can analyze ECG data to detect heart abnormalities."
    ],
    "insomnia": [
        "Insomnia can be caused by stress, anxiety, depression, or poor sleep habits.",
        "Cognitive behavioral therapy for insomnia (CBT-I) is an effective treatment.",
        "Medications can also be used to treat insomnia, but they can have side effects.",
        "Wearable sensors can track sleep patterns and identify potential sleep disorders."
    ],
    "ai in medicine": [
        "AI can analyze medical images to detect diseases like cancer.",
        "AI can help predict patient outcomes and personalize treatment plans.",
        "AI can be used to develop new drugs and therapies.",
        "AI can monitor patients remotely and alert healthcare providers to potential problems."
    ]
}

# Embed knowledge base
embedded_knowledge = {}
for category, facts in knowledge_base.items():
    embedded_knowledge[category] = model.encode(facts)

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

def retrieve_relevant_knowledge(query):
    """Retrieves relevant knowledge from the knowledge base."""
    query_embedding = model.encode(query)
    scores = {}
    for category, embeddings in embedded_knowledge.items():
        scores[category] = np.max(np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)))

    most_relevant_category = max(scores, key=scores.get)
    relevant_facts = knowledge_base[most_relevant_category]
    return relevant_facts

def generate_personalized_response(symptoms, emotion, sentiment_label, sentiment_score):
    """Generates a personalized response using the LLM."""
    model = genai.GenerativeModel('gemini-pro')

    relevant_knowledge = retrieve_relevant_knowledge(symptoms + " " + emotion)

    prompt = f"""You are an empathetic and highly knowledgeable clinical and technology expert. Your task is to help patients understand their symptoms and provide structured, clear, and compassionate medical advice, including AI-powered insights. You will use chain of reasoning.

    First, reason step by step about the user's query.
    Second, use the following knowledge base: {relevant_knowledge}.
    Third, you will provide a structured JSON response example like the following:

    User Query: I have been experiencing {symptoms}.
    My emotional state is: {emotion}. Sentiment analysis indicates {sentiment_label} with a score of {sentiment_score:.2f}.

    ```json
    {{
      "possible_conditions": [
        {{"condition": "Common Cold", "confidence": 0.7, "explanation": "Symptoms include runny nose, sore throat, and cough."}},
        {{"condition": "Allergies", "confidence": 0.6, "explanation": "Symptoms include sneezing, itchy eyes, and nasal congestion."}}
      ],
      "first_aid_medications": [
        {{"medication": "Rest", "rationale": "Allows the body to recover."}},
        {{"medication": "Over-the-counter decongestant", "rationale": "Helps relieve nasal congestion."}}
      ],
      "nutritional_recommendations": [
        {{"recommendation": "Chicken soup", "rationale": "Provides hydration and nutrients."}},
        {{"recommendation": "Ginger tea", "rationale": "Soothes the throat and reduces inflammation."}}
      ],
      "ai_clinical_insights": [
         {{"technology": "AI-powered cough analysis", "application": "Detecting respiratory infections", "evidence": "Studies show AI can identify cough patterns indicative of specific illnesses."}}
      ],
      "additional_clinical_insights": "If symptoms worsen or persist, consult a doctor.",
      "disclaimer": "This information is not a substitute for professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment."
    }}
    ```

    Follow these instructions:

    1.  Reason about the user's query, linking it to the relevant medical knowledge.
    2.  **Possible Conditions:** List two possible conditions that could explain the symptoms. Include a confidence score (0.0-1.0) indicating the likelihood of each condition based on the provided information and an explanation.
    3.  **First Aid Medications/Interventions:** Suggest two immediate interventions or over-the-counter medications and provide a rationale for each suggestion.
    4.  **Nutritional/Welfare Recommendations:** Suggest two nutritional or lifestyle recommendations that might help alleviate the symptoms and provide a rationale for each.
    5.  **AI/ML in Clinical Care:** Describe a relevant AI/ML technology (if any) used for diagnosis or management of similar conditions. Provide potential applications and a short reasoning. If no technology applies, return an empty array "". Use knowledge base to guide it.
    6.  **Additional Clinical Insights:** Include any extra information that may help in understanding the patient's condition, including factors that should prompt them to seek immediate medical attention.
    7.  **Disclaimer:** Add the following disclaimer "This information is not a substitute for professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment.".

    Ensure the tone is caring and human-like. Emphasize that you are an AI. The JSON response should be a valid JSON.
    """

    try:
        response = model.generate_content(prompt)
        if response.text:
            try:
                json_output = json.loads(response.text)
                return json_output
            except json.JSONDecodeError as e:
                return f"Error decoding JSON: {e}. Raw response: {response.text}"
        else:
            return "The AI assistant returned an empty response. Please try again."
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
    else:
        st.warning("Please enter both your symptoms and your emotional state.")
