import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

ordinal_encoder = OrdinalEncoder(categories=[['no', 'Sometimes', 'Frequently', 'Always']])
label_encoders = {'Gender': LabelEncoder(), 'family_history_with_overweight': LabelEncoder()}  # Add other label encoders as needed
scaler = StandardScaler()

st.title("Obesity Prediction")

st.sidebar.header('Input Features')
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
family_history = st.sidebar.selectbox('Family History of Overweight', ['yes', 'no'])
calc = st.sidebar.selectbox('CALC', ['no', 'Sometimes', 'Frequently', 'Always'])
age = st.sidebar.slider('Age', 10, 90, 25)  # Example range

input_data = pd.DataFrame({
    'Gender': [gender],
    'family_history_with_overweight': [family_history],
    'CALC': [calc],
    'Age': [age]
})

input_data['CALC'] = ordinal_encoder.fit_transform(input_data[['CALC']])
for col in ['Gender', 'family_history_with_overweight']:
    input_data[col] = label_encoders[col].fit_transform(input_data[col])
input_data = pd.DataFrame(scaler.fit_transform(input_data), columns=input_data.columns)

prediction = model.predict(input_data)[0]

st.write(f"The predicted obesity level is: {prediction}")