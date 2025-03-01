import requests
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px

# Fetch COVID-19 data
url = "https://disease.sh/v3/covid-19/countries/usa"
r = requests.get(url)
data = r.json()

# Extract relevant fields
covid_data = {
    "cases": data["cases"],
    "todayCases": data["todayCases"],
    "deaths": data["deaths"],
    "todayDeaths": data["todayDeaths"],
    "recovered": data["recovered"],
    "active": data["active"],
    "critical": data["critical"],
    "casesPerMillion": data["casesPerOneMillion"],
    "deathsPerMillion": data["deathsPerOneMillion"],
}

# Convert the data into a Pandas DataFrame
df = pd.DataFrame([covid_data])

# Generate random historical data for simulation (last 30 days)
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Simulated last 30 days of cases
historical_deaths = np.random.randint(500, 2000, size=30)

# Create a DataFrame for historical data
df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)

# Prepare the data for SVM (SVR) regression model
X = df_historical[["day"]]  # Feature: Day
y = df_historical["cases"]  # Target: Cases

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM Regression Model for predicting the number of cases
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)

# Predict the number of cases for the next day (e.g., Day 31)
next_day = np.array([[31]])
predicted_cases_svm = svm_model.predict(next_day)

# For classification (whether the cases will exceed 50,000), use SVM with a classification approach
df_historical["high_case"] = (df_historical["cases"] > 50000).astype(int)  # 1 if cases > 50k, else 0
y_class = df_historical["high_case"]  # Classification target

# Split the data again for classification (ensure that it uses the same split as regression)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Train the SVM model for classification (predict if cases exceed 50k)
svm_classifier = SVR(kernel='rbf')  # SVR can also be used for classification tasks
y_train_class = y_train_class.astype(float)  # SVR requires float for target labels
svm_classifier.fit(X_train_class, y_train_class)

# Predict classification for Day 31 (whether cases exceed 50k)
pred_class = svm_classifier.predict(np.array([[31]]))

# Streamlit UI customizations
st.set_page_config(page_title="COVID-19 Prediction in USA", layout="wide", initial_sidebar_state="expanded")

# Title and Description
st.title("COVID-19 Prediction App")
st.markdown("""
    This app predicts the number of COVID-19 cases for the next day using SVM models.
    - **SVM Regression** predicts the number of cases for any given day.
    - **SVM Classification** predicts whether the cases will exceed 50,000.
""")

# Sidebar for Input
st.sidebar.header("Prediction Settings")
day_input = st.sidebar.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100, value=31)

# Add an Expander for the description of the models
with st.expander("What is SVM?"):
    st.write("""
        **SVM Regression** is used for predicting continuous values like the number of COVID-19 cases.
        **SVM Classification** is used for classifying data, here it's used to predict if cases will exceed 50,000.
    """)

# Display Historical Data Chart using Plotly
st.subheader("Historical COVID-19 Cases (Last 30 Days)")
st.write("The chart below shows the simulated historical data of COVID-19 cases for the last 30 days.")

fig = px.line(df_historical, x="day", y="cases", title="Simulated Historical COVID-19 Cases", labels={"day": "Day", "cases": "Number of Cases"})
st.plotly_chart(fig)

# Model Predictions Button
if st.button("Predict"):
    # Predict continuous cases using the SVM regression model
    prediction_svm = svm_model.predict([[day_input]])

    # Predict if cases will exceed 50k using the SVM classification model
    prediction_class = svm_classifier.predict(np.array([[day_input]]))

    # Display Predictions
    st.subheader(f"Prediction for Day {day_input}")

    st.write(f"### Predicted COVID-19 Cases (SVM Regression): {int(prediction_svm[0])} cases")
    if prediction_class[0] >= 0.5:
        st.write(f"### Prediction: **Cases will exceed 50,000** on Day {day_input} (SVM Classification).")
    else:
        st.write(f"### Prediction: **Cases will NOT exceed 50,000** on Day {day_input} (SVM Classification).")

    # Show a summary
    st.write("""
    The model predicts the number of COVID-19 cases for the selected day. It also predicts if the cases will exceed 50,000.
    """)

# Footer (can add any message or link)
st.markdown("---")
st.markdown("Created by [Your Name](https://yourwebsite.com)")

# Customize the theme (Optional)
st.markdown("""
    <style>
        .stApp {
            background-color: #f4f4f9;
        }
        .sidebar .sidebar-content {
            background-color: #f7f7f9;
        }
    </style>
""", unsafe_allow_html=True)
