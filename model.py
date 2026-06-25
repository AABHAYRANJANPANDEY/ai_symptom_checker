import pandas as pd

# Define a baseline map of symptoms -> conditions
# In production, this can be swapped out with an XGBoost/RandomForest .pkl file
DISEASE_DB = {
    "Influenza (Flu)": {
        "symptoms": ["fever", "cough", "headache", "fatigue", "body ache"],
        "severity": "Moderate",
        "urgency": "See doctor if symptoms worsen"
    },
    "Common Cold": {
        "symptoms": ["cough", "runny nose", "sore throat", "sneezing"],
        "severity": "Low",
        "urgency": "Rest at home"
    },
    "Acute Bronchitis": {
        "symptoms": ["cough", "chest pain", "fatigue", "shortness of breath"],
        "severity": "Moderate",
        "urgency": "See doctor if breathing becomes difficult"
    },
    "Angina / Cardiac Event": {
        "symptoms": ["chest pain", "shortness of breath", "sweating", "nausea"],
        "severity": "High",
        "urgency": "EMERGENCY: Seek immediate medical attention!"
    }
}

ALL_FEATURES = ["fever", "cough", "headache", "chest pain", "fatigue", "body ache", "runny nose", "sore throat", "sneezing", "shortness of breath", "sweating", "nausea"]

def text_to_features(extracted_symptoms: list) -> dict:
    """Converts a list of symptoms into a binary feature dictionary."""
    return {symptom: (1 if symptom in extracted_symptoms else 0) for symptom in ALL_FEATURES}

def predict_conditions(features: dict):
    """Calculates a simulated confidence score based on feature intersection."""
    active_symptoms = [k for k, v in features.items() if v == 1]
    
    if not active_symptoms:
        return []
        
    predictions = []
    for disease, data in DISEASE_DB.items():
        match_count = sum(1 for sym in active_symptoms if sym in data["symptoms"])
        if match_count > 0:
            # Simple probability calculation based on matched features
            confidence = round((match_count / len(data["symptoms"])), 2)
            predictions.append({
                "condition": disease,
                "confidence": confidence,
                "base_severity": data["severity"],
                "urgency": data["urgency"]
            })
            
    # Sort by highest confidence score
    predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)[:3]
    return predictions
