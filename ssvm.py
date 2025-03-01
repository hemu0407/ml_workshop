import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

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

# Convert to Pandas DataFrame
df = pd.DataFrame([covid_data])

# Generate random historical data
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Last 30 days cases
historical_deaths = np.random.randint(500, 2000, size=30)
df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)

# Prepare data for regression
X = df_historical[["day"]]
y = df_historical["cases"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM Regression Model
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)

# Predict using SVM
next_day = np.array([[31]])
predicted_cases_svm = svm_model.predict(next_day)

# Streamlit Interface
st.title("COVID-19 Cases Prediction in USA")
st.write("Predicting COVID-19 cases for the next day using SVM.")

# User Input
day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

# SVM Prediction
if st.button("Predict"):
    prediction_svm = svm_model.predict([[day_input]])
    st.write(f"Predicted cases for day {day_input} using SVM: {int(prediction_svm[0])}")

# Plotting the historical data using Matplotlib
st.subheader("Historical COVID-19 Cases (Last 30 Days)")
st.write("The chart below shows the simulated historical data of COVID-19 cases for the last 30 days.")

# Plotting the historical data
plt.figure(figsize=(10, 6))
plt.plot(df_historical["day"], df_historical["cases"], marker='o', color='b', label="COVID-19 Cases")
plt.xlabel('Day')
plt.ylabel('Number of Cases')
plt.title('Simulated Historical COVID-19 Cases')
plt.grid(True)
plt.legend()

# Display the plot in Streamlit
st.pyplot(plt)
