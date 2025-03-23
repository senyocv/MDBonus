import os
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder

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

# Load trained model and feature names
if not os.path.exists(MODEL_FILE):
    st.error(f"Model file '{MODEL_FILE}' not found!")
else:
    try:
        with open(MODEL_FILE, "rb") as file:
            model_data = pickle.load(file)
            if isinstance(model_data, tuple) and len(model_data) == 2:
                model, loaded_feature_names = model_data
                if not isinstance(model, RandomForestClassifier):
                    st.error("Loaded model is not a RandomForestClassifier!")
                elif loaded_feature_names != feature_names:
                    st.error("Feature names in the model do not match the dataset!")
            else:
                st.error("Unexpected content in 'MDBonus1.pkl'. Expected a tuple with model and feature names.")
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

# Display raw user input
st.write("### User Input Data (Raw)")
st.write(pd.DataFrame([user_data_raw]))

# Encoding and scaling
st.write("### Processed User Input Data")
try:
    # Convert user input into DataFrame
    user_df = pd.DataFrame([user_data_raw])

    # Ordinal encoding
    ord_enc = OrdinalEncoder(categories=[['no', 'Sometimes', 'Frequently', 'Always']])
    user_df['CALC'] = ord_enc.fit_transform(user_df[['CALC']])

    # Label encoding
    label_encoders = {col: LabelEncoder().fit(data[col]) for col in cat_cols if col != 'CALC'}
    for col, encoder in label_encoders.items():
        user_df[col] = encoder.transform(user_df[col])

    # Scaling
    scaler = StandardScaler().fit(data[num_cols])
    user_df[num_cols] = scaler.transform(user_df[num_cols])

    # Display processed user input
    st.write(user_df)
except Exception as e:
    st.error(f"Error processing user input: {e}")

# Make prediction
if st.button("Predict"):
    try:
        prediction = model.predict(user_df)[0]
        st.write(f"The predicted obesity level is: {prediction}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
