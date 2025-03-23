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
    fig, axes = plt.subplots(1, len(feature_names), figsize=(15, 5))
    
    for i, col in enumerate(feature_names):
        if data[col].dtype in ['int64', 'float64']:  # Numerical data
            sns.histplot(data[col], ax=axes[i], kde=True)
            axes[i].set_title(col)
    
    st.pyplot(fig)

    st.subheader("Categorical Features Distribution")
    cat_cols = [col for col in feature_names if data[col].dtype == 'object']
    
    for col in cat_cols:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.countplot(data=data, x=col, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

# --- User Input Table ---
st.subheader("📝 Input Your Data")

user_data = {}
for col in feature_names:
    if data[col].dtype in ['int64', 'float64']:  # Numerical input
        user_data[col] = st.slider(f"Select {col}", float(data[col].min()), float(data[col].max()), float(data[col].mean()))
    else:  # Categorical input
        user_data[col] = st.selectbox(f"Select {col}", data[col].unique())

# Convert user input into DataFrame
user_df = pd.DataFrame([user_data])

# --- Show User Input Data ---
st.subheader("🗂 User Input Data")
st.write(user_df)

# --- Make Predictions ---
if st.button("Predict"):
    try:
        prediction = model.predict(user_df)
        probabilities = model.predict_proba(user_df)
        
        # Display probability per class
        st.subheader("📊 Classification Probability")
        prob_df = pd.DataFrame(probabilities, columns=model.classes_)
        st.write(prob_df)

        # Display final prediction
        st.subheader("✅ Prediction Result")
        st.write(f"Predicted Obesity Class: **{prediction[0]}**")

    except Exception as e:
        st.error(f"Error making prediction: {e}")
