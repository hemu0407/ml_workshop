import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Fetch detailed COVID-19 data from API
url = "https://disease.sh/v3/covid-19/countries/usa"
r = requests.get(url)
data = r.json()

# Extracting detailed COVID-19 data
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
    "tests": data["tests"],
    "testsPerMillion": data["testsPerMillion"],
    "population": data["population"],
    "continent": data["continent"],
    "flag": data["countryInfo"]["flag"]
}

# Convert to Pandas DataFrame
df = pd.DataFrame([covid_data])

# Displaying the extracted data in a tabular format
st.subheader("Detailed COVID-19 Data for USA")
st.write(df)

# Historical Data: Generate random simulated historical data for the last 30 days
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)
historical_deaths = np.random.randint(500, 2000, size=30)
historical_recovered = np.random.randint(10000, 50000, size=30)
df_historical = pd.DataFrame({
    "cases": historical_cases,
    "deaths": historical_deaths,
    "recovered": historical_recovered
})
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
st.write("Using SVM to predict COVID-19 cases for the next day.")

# User Input for prediction
day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

if st.button("Predict"):
    prediction_svm = svm_model.predict([[day_input]])
    st.write(f"Predicted cases for day {day_input} using SVM: {int(prediction_svm[0])}")

# Detailed Visualization: Historical Data
st.subheader("Historical COVID-19 Cases, Deaths, and Recoveries (Last 30 Days)")

# Plotting historical data
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_historical["day"], df_historical["cases"], marker='o', label="Cases", color='blue')
ax.plot(df_historical["day"], df_historical["deaths"], marker='x', label="Deaths", color='red')
ax.plot(df_historical["day"], df_historical["recovered"], marker='s', label="Recovered", color='green')
ax.set_xlabel("Day")
ax.set_ylabel("Count")
ax.set_title("Simulated Historical COVID-19 Data (Cases, Deaths, Recovered) over the Last 30 Days")
ax.legend()
ax.grid(True)

# Display the plot in Streamlit
st.pyplot(fig)

# Bar Chart: Current COVID-19 Statistics in the USA
st.subheader("Current COVID-19 Statistics (As of Today)")

labels = ["Total Cases", "Active Cases", "Recovered", "Deaths", "Tests Conducted"]
values = [data["cases"], data["active"], data["recovered"], data["deaths"], data["tests"]]

# Bar chart for the current data
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.bar(labels, values, color=['blue', 'orange', 'green', 'red', 'purple'])
ax2.set_xlabel("Category")
ax2.set_ylabel("Count")
ax2.set_title("Current COVID-19 Statistics for USA")
ax2.grid(True)

# Display the bar chart
st.pyplot(fig2)

# Additional Stats and Information
st.subheader("Additional COVID-19 Statistics and Information")
st.write(f"**Total Population**: {data['population']}")
st.write(f"**Tests Conducted**: {data['tests']}")
st.write(f"**Critical Cases**: {data['critical']}")
st.write(f"**Deaths per Million**: {data['deathsPerMillion']}")
st.write(f"**Cases per Million**: {data['casesPerMillion']}")
st.write(f"**Continent**: {data['continent']}")
st.image(data['flag'], caption="Country Flag", width=200)

