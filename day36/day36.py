import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os 

# Load dataset
csv_path = os.path.join(os.path.dirname(__file__), "creditcard_small.csv")
data = pd.read_csv(csv_path)

# Create customer growth dataset
data = data.sort_values("Time")

growth = pd.DataFrame({
    "Day": range(1, len(data)+1),
    "Customers": np.arange(100, 100 + len(data))
})

print(growth.head())

# Train model
X = growth[["Day"]]
y = growth["Customers"]

model = LinearRegression()
model.fit(X, y)

# Predict next 30 days
future_days = np.arange(
    len(growth)+1,
    len(growth)+31
).reshape(-1, 1)

future_days_df = pd.DataFrame(
    future_days,
    columns=["Day"]
)

predictions = model.predict(future_days_df)

forecast = pd.DataFrame({
    "Future_Day": future_days.flatten(),
    "Predicted_Customers": predictions
})

print("\nNext 30 Days Forecast:")
print(forecast.head())

# Historical Trend
plt.plot(growth["Day"], growth["Customers"])
plt.title("Customer Growth Trend")
plt.xlabel("Day")
plt.ylabel("Customers")
plt.show()

# Forecast
plt.plot(future_days, predictions)
plt.title("Next 30 Days Forecast")
plt.xlabel("Future Day")
plt.ylabel("Predicted Customers")
plt.show()