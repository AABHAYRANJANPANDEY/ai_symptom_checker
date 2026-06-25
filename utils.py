import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Initialize the official client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def extract_symptoms_from_text(user_input: str, valid_features: list) -> list:
    """Uses Gemini to parse conversational text into structured standard features."""
    prompt = f"""
    You are an expert clinical NLP parser. Analyze the user's input description of how they feel.
    Identify which of the following standard medical symptoms are explicitly mentioned or strongly implied.
    
    Allowed symptoms list: {valid_features}
    
    User Input: "{user_input}"
    
    Return ONLY a valid JSON array of strings matching the allowed symptoms list. Do not write markdown, code blocks, or explanations.
    Example Output: ["fever", "cough"]
    """
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        extracted = json.loads(response.text.strip())
        return [sym for sym in extracted if sym in valid_features]
    except Exception as e:
        print(f"Error extracting symptoms: {e}")
        return []

def generate_medical_explanation(user_input: str, top_predictions: list) -> str:
    """Uses Gemini to synthesize explanations and safety caveats."""
    prompt = f"""
    You are an empathetic, professional AI Triage Assistant. 
    The user described their condition as: "{user_input}"
    
    Our structural classification model calculated these likely conditions:
    {json.dumps(top_predictions, indent=2)}
    
    Provide an easy-to-understand explanation of these possibilities.
    - Translate any clinical terms into simple language.
    - Give 3 practical, safe, non-medicinal precautions (e.g., hydration, rest).
    - Provide a prominent warning if any high-severity conditions are listed.
    
    CRITICAL MANDATE: Start or finish with a clear disclaimer stating that you are an AI, not a doctor, and this is purely for informational triage.
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error gathering medical advice: {e}"
