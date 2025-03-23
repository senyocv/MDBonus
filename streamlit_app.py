import os
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Load dataset to get feature names
DATA_FILE = "obesity_data.csv"
MODEL_FILE = "MDBonus1.pkl"

if not os.path.exists(DATA_FILE):
    st.error(f"Data file '{DATA_FILE}' not found!")
else:
    data = pd.read_csv(DATA_FILE)

# Extract feature names (excluding target)
target_column = "NObeyesdad"
feature_names = data.drop(columns=[target_column]).columns.tolist()

# Identify categorical columns
cat_cols = [col for col in feature_names if data[col].dtype == 'object']
num_cols = [col for col in feature_names if col not in cat_cols]

# Load trained model
if not os.path.exists(MODEL_FILE):
    st.error(f"Model file '{MODEL_FILE}' not found!")
else:
    try:
        with open(MODEL_FILE, "rb") as file:
            model = pickle.load(file)
        if not isinstance(model, RandomForestClassifier):
            st.error("Loaded model is not a RandomForestClassifier!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# --- Streamlit App ---
st.title("Obesity Prediction App")

# --- Dropdown: Show Raw Data ---
with st.expander("📊 Show Raw Data"):
    st.write(data)

# --- Dropdown: Show Data Visualization ---
with st.expander("📈 Data Visualization"):
    st.subheader("Numerical Features Distribution")
    
    fig, axes = plt.subplots(1, len(num_cols), figsize=(15, 5))
    
    for i, col in enumerate(num_cols):
        sns.histplot(data[col], ax=axes[i], kde=True)
        axes[i].set_title(col)
    
    st.pyplot(fig)

    st.subheader("Categorical Features Distribution")
    for col in cat_cols:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.countplot(data=data, x=col, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

# --- User Input Table ---
st.subheader("📝 Input Your Data")

# Store raw user input
user_data_raw = {}
for col in feature_names:
    if col in num_cols:  # Numerical input
        user_data_raw[col] = st.slider(f"Select {col}", float(data[col].min()), float(data[col].max()), float(data[col].mean()))
    else:  # Categorical input
        user_data_raw[col] = st.selectbox(f"Select {col}", data[col].unique())

# Convert user input into DataFrame (raw)
user_df_raw
