import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🩺 Diabetes Prediction App")
st.write("This app predicts whether a person is likely to have diabetes based on medical inputs.")

# -----------------------------
# Show available files
# -----------------------------
st.write("Files in current folder:", os.listdir())

# -----------------------------
# Load dataset with error handling
# -----------------------------
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("Diabetes.csv")
        return data
    except FileNotFoundError:
        st.error("Dataset file not found.")
        st.stop()

data = load_data()

# -----------------------------
# Dataset preview
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(data.head())

col1, col2, col3 = st.columns(3)
col1.metric("Rows", data.shape[0])
col2.metric("Columns", data.shape[1])
col3.metric("Missing Values", data.isnull().sum().sum())

# -----------------------------
# Dataset summary
# -----------------------------
with st.expander("Dataset Summary"):
    st.write(data.describe())

with st.expander("Check Missing Values"):
    st.write(data.isnull().sum())

# -----------------------------
# Class distribution
# -----------------------------
st.subheader("Outcome Distribution")
fig, ax = plt.subplots()
data["Outcome"].value_counts().plot(kind="bar", ax=ax)
ax.set_xlabel("Outcome")
ax.set_ylabel("Count")
ax.set_title("0 = No Diabetes, 1 = Diabetes")
st.pyplot(fig)

# -----------------------------
# Prepare data
# -----------------------------
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train model
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Enter Patient Details")

pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=300, value=120)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=79)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=33)

# -----------------------------
# BMI category function
# -----------------------------
def bmi_category(bmi_value):
    if bmi_value < 18.5:
        return "Underweight"
    elif bmi_value < 25:
        return "Normal weight"
    elif bmi_value < 30:
        return "Overweight"
    else:
        return "Obese"

# -----------------------------
# Create input dataframe
# -----------------------------
input_data = pd.DataFrame({
    "Pregnancies": [pregnancies],
    "Glucose": [glucose],
    "BloodPressure": [blood_pressure],
    "SkinThickness": [skin_thickness],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [dpf],
    "Age": [age]
})

# -----------------------------
# Display input data
# -----------------------------
st.subheader("User Input Data")
st.dataframe(input_data)

st.info(f"BMI Category: {bmi_category(bmi)}")

# -----------------------------
# Make prediction
# -----------------------------
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("The model predicts that the person is likely to have Diabetes.")
        st.warning("Advice: Please consult a doctor for proper medical diagnosis.")
    else:
        st.success("The model predicts that the person is not likely to have Diabetes.")
        st.info("Advice: Maintain a healthy lifestyle and regular check-up.")

    st.subheader("Prediction Probability")
    c1, c2 = st.columns(2)
    c1.metric("No Diabetes Probability", f"{prediction_proba[0]*100:.2f}%")
    c2.metric("Diabetes Probability", f"{prediction_proba[1]*100:.2f}%")

# -----------------------------
# Model performance
# -----------------------------
st.subheader("Model Performance")
st.write(f"Accuracy: **{accuracy * 100:.2f}%**")

# -----------------------------
# Feature importance
# -----------------------------
st.subheader("Feature Importance")
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.dataframe(feature_importance)

fig2, ax2 = plt.subplots()
ax2.bar(feature_importance["Feature"], feature_importance["Importance"])
ax2.set_title("Feature Importance")
ax2.set_xlabel("Features")
ax2.set_ylabel("Importance")
plt.xticks(rotation=45)
st.pyplot(fig2)

# -----------------------------
# Extra details
# -----------------------------
with st.expander("See Classification Details"):
    st.text("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual No Diabetes", "Actual Diabetes"],
        columns=["Predicted No Diabetes", "Predicted Diabetes"]
    )
    st.dataframe(cm_df)

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

# -----------------------------
# Footer note
# -----------------------------
st.caption("Note: This prediction is based on machine learning and should not replace medical advice.")