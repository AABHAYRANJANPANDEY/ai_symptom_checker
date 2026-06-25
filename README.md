# 🏥 AI-Powered Symptom Triage Dashboard

An intelligent, hybrid symptom-checking system that combines deterministic clinical rule processing with generative contextual reasoning. The system maps user-submitted descriptions to structured medical symptoms, evaluates them via a diagnostic matching core, and provides conversational, empathetic explanations using Google's Gemini-3.5-Flash model.

---

## 🚀 Key Features

* **Natural Language Parsing:** Users can describe how they feel in plain English. The AI automatically extracts valid medical features.
* **Deterministic Match Classification:** Prevents raw LLM hallucinations by passing extracted features through a structured diagnostic dataset to calculate confidence ranks for the Top 3 conditions.
* **Dynamic Patient Dashboard:** Interactive interface displaying progress bars for model confidence, risk level categorization (Low/Moderate/High), and automated traffic-light urgency warnings.
* **Contextual Rationale Generation:** Delivers structured, non-alarmist health explanations, safe non-medicinal precautions, and clear medical triage disclaimers.

---

## 🛠️ Tech Stack & Architecture

* **Frontend Framework:** Streamlit
* **Language Model Core:** Google Gen AI Python SDK (`gemini-3.5-flash`)
* **Data Processing:** Pandas, Python-dotenv
