import streamlit as st
import joblib

# Load model and symptoms
model, symptoms = joblib.load("disease_model.pkl")

# UI
st.title("Disease Prediction System")
st.markdown("### Select symptoms from the list below:")

# Multiselect box
user_input = st.multiselect("Symptoms", symptoms)

# Predict button
if st.button("Predict Disease"):
    if user_input:
        input_vector = [1 if symptom in user_input else 0 for symptom in symptoms]
        prediction = model.predict([input_vector])[0]
        st.success(f"Predicted Disease: **{prediction}**")
    else:
        st.warning("Please select at least one symptom.")
