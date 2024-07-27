import streamlit as st
import pickle
import numpy as np
from joblib import load
# Load the model and preprocessing steps
# def load_model():
#     with open('saved_steps.pkl', 'rb') as file:
#         data = pickle.load(file)
#     return data

def load_model():
    data = load('saved_steps.pkl')
    return data

data = load_model()
regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

# Function to convert salary from USD to INR
def usd_to_inr(salary_usd):
    exchange_rate = 83  # 1 USD = 83 INR
    salary_inr = salary_usd * exchange_rate
    return salary_inr

# Function to show prediction page
def show_predict_page():
    st.title("Software Developer Salary Prediction")
    st.write("""### We need some information to predict the salary""")

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)

    experience = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, experience]])
        X[:, 0] = le_country.transform(X[:, 0])
        X[:, 1] = le_education.transform(X[:, 1])
        X = X.astype(float)

        salary_usd = regressor.predict(X)[0]
        salary_inr = usd_to_inr(salary_usd)
        st.subheader(f"The estimated salary is ₹{salary_inr:.2f}")
