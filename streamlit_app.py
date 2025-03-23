import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
import matplotlib.pyplot as plt
import os
import joblib

if not os.path.exists('MDBonus1.pkl'):
    st.error("Model file 'MDBonus1.pkl' not found!")
if not os.path.exists('obesity_data.csv'):
    st.error("Data file 'obesity_data.csv' not found!")

try:
    with open('MDBonus1.pkl', 'rb') as file:
            content = pickle.load(file)
        if not os.path.exists('MDBonus1.pkl'):
    st.error("Model file 'MDBonus1.pkl' not found!")
if not os.path.exists('obesity_data.csv'):
    st.error("Data file 'obesity_data.csv' not found!")

try:
    with open('MDBonus1.pkl', 'rb') as file:
        content = pickle.load(file)
        if isinstance(content, tuple) and len(content) == 2:
            model, feature_names = content
        else:
            st.error("Unexpected content in 'MDBonus1.pkl'. Expected a tuple with two elements.")
except Exception as e:
    st.error(f"Error loading model: {e}")
except Exception as e:
    st.error(f"Error loading model: {e}")

ordinal_encoder = OrdinalEncoder(categories=[['no', 'Sometimes', 'Frequently', 'Always']])
label_encoders = {'Gender': LabelEncoder(), 'family_history_with_overweight': LabelEncoder()}  # Add other label encoders as needed
scaler = StandardScaler()

try:
    data = pd.read_csv('obesity_data.csv')
except Exception as e:
    st.error(f"Error loading data: {e}")

# Streamlit app
st.title("Obesity Prediction")

# Input fields
st.sidebar.header('Input Features')
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
family_history = st.sidebar.selectbox('Family History of Overweight', ['yes', 'no'])
calc = st.sidebar.selectbox('CALC', ['no', 'Sometimes', 'Frequently', 'Always'])
age = st.sidebar.slider('Age', 10, 90, 25)  # Example range

# Create input DataFrame
input_data = pd.DataFrame({
    'Gender': [gender],
    'family_history_with_overweight': [family_history],
    'CALC': [calc],
    'Age': [age]
})

try:
    input_data['CALC'] = ordinal_encoder.fit_transform(input_data[['CALC']])
    for col in ['Gender', 'family_history_with_overweight']:
        input_data[col] = label_encoders[col].fit_transform(input_data[col])
    input_data = pd.DataFrame(scaler.fit_transform(input_data), columns=input_data.columns)
except Exception as e:
    st.error(f"Error encoding or scaling input data: {e}")

try:
    input_data = input_data.reindex(columns=feature_names)
except Exception as e:
    st.error(f"Error reindexing input data: {e}")

try:
    prediction = model.predict(input_data)[0]
    st.write(f"The predicted obesity level is: {prediction}")
except Exception as e:
    st.error(f"Error making prediction: {e}")

try:
    st.sidebar.header('Data Visualization')
    selected_column = st.sidebar.selectbox('Select column for histogram', data.columns)
    st.write(f"Histogram of {selected_column}")
    st.bar_chart(data[selected_column].value_counts())

    st.write(f"Detailed Histogram of {selected_column}")
    fig, ax = plt.subplots()
    ax.hist(data[selected_column], bins=20, edgecolor='black')
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error in data visualization: {e}")
