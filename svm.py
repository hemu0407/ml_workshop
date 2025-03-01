import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
import streamlit as st

# Fetch COVID-19 data
url = "https://disease.sh/v3/covid-19/countries/usa"
r = requests.get(url)
data = r.json()

# Print the raw data (for debugging)
print(data)

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

# Convert to Pandas DataFrame
df = pd.DataFrame([covid_data])

# Print the DataFrame (for debugging)
print(df)

# Streamlit: Display the raw data in a table
st.write("COVID-19 Data for the USA:")
st.dataframe(df)

# Plot COVID-19 data for the USA
labels = ["Total Cases", "Active Cases", "Recovered", "Deaths"]
values = [data["cases"], data["active"], data["recovered"], data["deaths"]]

# Create bar plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
ax.set_xlabel("Category")
ax.set_ylabel("Count")
ax.set_title("COVID-19 Data for USA")

# Streamlit: Display the plot
st.pyplot(fig)

# Generate random historical data (last 30 days)
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Simulated last 30 days cases
historical_deaths = np.random.randint(500, 2000, size=30)

# Create a DataFrame for historical data
df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)

# Print historical data (for debugging)
print(df_historical.head())

# Prepare data for training
X = df_historical[["day"]]  # Feature: Day
y = df_historical["cases"]  # Target: Cases

# Create binary target for Logistic Regression
y_binary = (y > 50000).astype(int)  # Classify if cases exceed 50k

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Logistic Regression (used for classification, so we'll predict if cases exceed a threshold)
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)

# Support Vector Regression (SVR) for continuous prediction of cases
svr_model = SVR(kernel="rbf")
svr_model.fit(X_train, y)

# Predict next day's cases using Logistic Regression (binary prediction)
log_reg_pred = log_reg_model.predict(np.array([[31]]))
print(f"Logistic Regression prediction (if cases > 50k for Day 31): {log_reg_pred[0]}")

# Predict next day's cases using Support Vector Regression (SVR) for continuous prediction
svr_pred = svr_model.predict(np.array([[31]]))
print(f"SVR prediction for Day 31 (continuous cases): {int(svr_pred[0])}")

# Streamlit Web App
st.title("COVID-19 Cases Prediction in USA")
st.write("Predicting COVID-19 cases for the next day based on historical data.")

# User Input for prediction
day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

# Logistic Regression prediction button
if st.button("Predict (Logistic Regression)"):
    log_reg_prediction = log_reg_model.predict([[day_input]])
    if log_reg_prediction[0] == 1:
        st.write(f"Logistic Regression: Cases predicted to exceed 50k on Day {day_input}")
    else:
        st.write(f"Logistic Regression: Cases predicted to NOT exceed 50k on Day {day_input}")

# SVR prediction button
if st.button("Predict (SVR)"):
    svr_prediction = svr_model.predict([[day_input]])
    st.write(f"SVR: Predicted number of cases for Day {day_input}: {int(svr_prediction[0])}")
