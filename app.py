# app.py

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ✅ Load and prepare data
@st.cache_data
def load_model():
    data = pd.read_csv("dataset.csv")

    # Extract unique symptoms from Symptom_1 to Symptom_17
    all_symptoms = []
    for i in range(1, 18):
        all_symptoms.extend(data[f"Symptom_{i}"].dropna().unique())

    unique_symptoms = sorted(set([s for s in all_symptoms if str(s) != 'nan']))

    # Create binary features for each symptom
    for symptom in unique_symptoms:
        data[symptom] = data.apply(lambda row: 1 if symptom in row.values else 0, axis=1)

    # Drop original symptom columns
    data.drop(columns=[f'Symptom_{i}' for i in range(1, 18)], inplace=True)

    # Encode target
    label_encoder = LabelEncoder()
    data['Disease'] = label_encoder.fit_transform(data['Disease'])

    # Train model
    X = data.drop('Disease', axis=1)
    y = data['Disease']
    model = RandomForestClassifier()
    model.fit(X, y)

    return model, label_encoder, unique_symptoms

# Load everything
model, label_encoder, unique_symptoms = load_model()

# Load all diseases from label_encoder
all_diseases = label_encoder.classes_

# Simple default precautions (for demonstration)
default_precautions = [
    "Take prescribed medicine",
    "Drink plenty of water",
    "Get enough rest",
    "Consult a doctor if symptoms persist"
]

# Streamlit UI
st.set_page_config(page_title="Disease Prediction App")
st.title("🧠 Multi-Disease Prediction System with Precautions")
st.write("Select symptoms below to get a disease prediction and basic precautions.")

# Input: Symptom checkboxes
selected_symptoms = st.multiselect("Select Symptoms:", unique_symptoms)

# Predict
if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        # Create input vector
        input_data = np.zeros(len(unique_symptoms))
        for symptom in selected_symptoms:
            if symptom in unique_symptoms:
                input_data[unique_symptoms.index(symptom)] = 1

        prediction = model.predict([input_data])[0]
        disease_name = label_encoder.inverse_transform([prediction])[0]

        st.success(f"🩺 Predicted Disease: **{disease_name}**")

        st.markdown("### ✅ Suggested Precautions:")
        for precaution in default_precautions:
            st.markdown(f"- {precaution}")
