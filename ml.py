import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Get COVID-19 data for the USA
url = "https://disease.sh/v3/covid-19/countries/usa"
r = requests.get(url)
data = r.json()

# Print raw data (for debugging purposes)
print(data)

# Extract relevant fields!pip install requests pandas matplotlib scikit-learn numpy

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
print(df)

# Plot the COVID-19 data for the USA
labels = ["Total Cases", "Active Cases", "Recovered", "Deaths"]
values = [data["cases"], data["active"], data["recovered"], data["deaths"]]

plt.figure(figsize=(8,5))
plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("COVID-19 Data for USA")
plt.show()

# Generate random historical data for cases and deaths (this is just for testing)
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Last 30 days cases
historical_deaths = np.random.randint(500, 2000, size=30)

df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)

# Print historical data for verification
print(df_historical.head())

# Train a linear regression model on the historical data
X = df_historical[["day"]]
y = df_historical["cases"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predict the next day's cases
next_day = np.array([[31]])
predicted_cases = model.predict(next_day)
print(f"Predicted cases for Day 31: {int(predicted_cases[0])}")

# If you want to test with a different day input:
day_input = 31  # Example input
prediction = model.predict([[day_input]])
print(f"Predicted cases for day {day_input}: {int(prediction[0])}")

