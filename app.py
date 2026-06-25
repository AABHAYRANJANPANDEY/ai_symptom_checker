import streamlit as st
import model
import utils

st.set_page_config(page_title="AI Symptom Triage Dashboard", page_icon="🏥", layout="wide")

st.title("🏥 Intelligent Symptom Checker & Triage Engine")
st.markdown("Enter your symptoms below. Our hybrid processing core evaluates inputs using deterministic classifiers alongside generative context reasoning.")

# Layout Columns
col1, col2 = st.columns()

with col1:
    st.subheader("📋 Step 1: Tell Us What's Wrong")
    user_description = st.text_area(
        "Describe how you feel in your own words:", 
        placeholder="e.g., 'I have had a really bad cough and a burning fever for 2 days. My head hurts too, but no chest pain.'"
    )
    
    st.markdown("**OR select symptoms manually:**")
    manual_selections = []
    for symptom in model.ALL_FEATURES:
        if st.checkbox(symptom.replace("_", " ").title(), key=f"chk_{symptom}"):
            manual_selections.append(symptom)

    analyze_btn = st.button("Analyze Health Status", type="primary")

with col2:
    st.subheader("📊 Diagnostic Status Dashboard")
    
    if analyze_btn:
        with st.spinner("Analyzing parameters and prompting logic layers..."):
            # 1. Symptom Extraction Layer
            extracted = []
            if user_description.strip():
                extracted = utils.extract_symptoms_from_text(user_description, model.ALL_FEATURES)
            
            # Combine text-extracted and manual checkboxes
            final_symptoms = list(set(extracted + manual_selections))
            
            if not final_symptoms:
                st.warning("⚠️ No matching symptoms identified. Please clarify your inputs.")
            else:
                st.info(f"**Identified Features:** {', '.join([s.title() for s in final_symptoms])}")
                
                # 2. Match & Classify Layer
                feature_vector = model.text_to_features(final_symptoms)
                predictions = model.predict_conditions(feature_vector)
                
                if not predictions:
                    st.success("No critical conditions matched our analytical database. Continue to monitor your health.")
                else:
                    # Display Top predictions
                    st.write("### Predicted Conditions (Top Matches)")
                    
                    for pred in predictions:
                        # Color coding based on risk severity
                        if pred['base_severity'] == 'High':
                            color = "red"
                            alert_box = st.error
                        elif pred['base_severity'] == 'Moderate':
                            color = "orange"
                            alert_box = st.warning
                        else:
                            color = "green"
                            alert_box = st.success
                            
                        st.markdown(f"#### **{pred['condition']}**")
                        st.progress(float(pred['confidence']))
                        st.caption(f"Model Confidence: **{int(pred['confidence'] * 100)}%** | Base Risk: **{pred['base_severity']}**")
                        alert_box(f"🚦 **Action Recommended:** {pred['urgency']}")
                        st.markdown("---")
                    
                    # 3. LLM Reasoning Generation Layer
                    st.write("### 🧠 AI Medical Reasoning & Guidance")
                    explanation = utils.generate_medical_explanation(user_description, predictions)
                    st.markdown(explanation)
                    
    else:
        st.write("Submit an assessment on the left panel to populate diagnostic analytics.")
