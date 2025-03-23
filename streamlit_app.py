import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
MODEL_PATH = "MDBonus1.pkl"
model = joblib.load(MODEL_PATH)

# Load dataset
DATA_PATH = "obesity_data.csv"
data = pd.read_csv(DATA_PATH)

# Sidebar title
st.sidebar.title("Obesity Classification App")

# Show raw data
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Raw Dataset")
    st.write(data)

# Data visualization
st.sidebar.subheader("Data Visualization")
plot_type = st.sidebar.selectbox("Select Plot Type", ["Histogram", "Boxplot"])

if plot_type == "Histogram":
    selected_column = st.sidebar.selectbox("Select Column", data.columns)
    st.subheader(f"Histogram of {selected_column}")
    st.hist_chart(data[selected_column])
    
elif plot_type == "Boxplot":
    selected_column = st.sidebar.selectbox("Select Column", data.columns)
    st.subheader(f"Boxplot of {selected_column}")
    st.box_chart(data[selected_column])

# User input for prediction
st.sidebar.subheader("Input Data for Prediction")
user_input = {}

# Numeric inputs (using sliders)
numeric_columns = data.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    user_input[col] = st.sidebar.slider(f"{col}", float(data[col].min()), float(data[col].max()), float(data[col].mean()))

# Categorical inputs (using selectbox)
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    user_input[col] = st.sidebar.selectbox(f"{col}", data[col].unique())

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Show user input data
st.subheader("User Input Data")
st.write(input_df)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display prediction
    st.subheader("Final Prediction")
    st.write(f"Predicted Class: **{prediction[0]}**")

    # Show probability per class
    st.subheader("Classification Probabilities")
    for i, prob in enumerate(prediction_proba[0]):
        st.write(f"Class {i}: {prob:.4f}")

