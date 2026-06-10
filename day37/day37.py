import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -------------------------
# Create Revenue Data
# -------------------------
days = np.arange(1, 501)

# Assume revenue grows over time
revenue = 1000 + days * 5 + np.random.randint(-50, 50, size=500)

data = pd.DataFrame({
    "Day": days,
    "Revenue": revenue
})

print(data.head())

# -------------------------
# Train Model
# -------------------------
X = data[["Day"]]
y = data["Revenue"]

model = LinearRegression()
model.fit(X, y)

# -------------------------
# Forecast Next 30 Days
# -------------------------
future_days = np.arange(501, 531).reshape(-1, 1)

future_df = pd.DataFrame(future_days, columns=["Day"])

predictions = model.predict(future_df)

forecast = pd.DataFrame({
    "Future_Day": future_days.flatten(),
    "Predicted_Revenue": predictions
})

print("\nNext 30 Days Revenue Forecast:")
print(forecast.head())

# -------------------------
# Visualization
# -------------------------
plt.plot(data["Day"], data["Revenue"])
plt.title("Historical Revenue Trend")
plt.xlabel("Day")
plt.ylabel("Revenue")
plt.show()

plt.plot(future_days, predictions)
plt.title("Future Revenue Forecast")
plt.xlabel("Future Day")
plt.ylabel("Predicted Revenue")
plt.show()

print("\nRevenue Forecasting Completed!")