import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

# Load dataset
csv_path = os.path.join(os.path.dirname(__file__), "creditcard_small.csv")
data = pd.read_csv(csv_path)

# =====================
# KPI Module
# =====================

total_customers = len(data)

total_revenue = data["Amount"].sum()

avg_revenue = data["Amount"].mean()

# =====================
# Retention Module
# =====================

retained = len(
    data[data["Amount"] > avg_revenue]
)

retention_rate = (
    retained / total_customers
) * 100

clv = avg_revenue * 12

# =====================
# Forecasting Module
# =====================

future_revenue = total_revenue * 1.05

# =====================
# Risk Prediction
# =====================

data["Churn"] = (
    data["Amount"] < avg_revenue
).astype(int)

kmeans = KMeans(
    n_clusters=2,
    random_state=42
)

data["Segment"] = kmeans.fit_predict(
    data[["Amount", "Time"]]
)

X = data[["Amount", "Time", "Segment"]]
y = data["Churn"]

model = LogisticRegression(
    max_iter=1000
)

model.fit(X, y)

data["RiskScore"] = (
    model.predict_proba(X)[:,1]
)

high_risk = len(
    data[data["RiskScore"] > 0.8]
)

# =====================
# Dashboard
# =====================

print("===== CUSTOMER INTELLIGENCE SYSTEM =====")

print("Total Customers:",
      total_customers)

print("Total Revenue:",
      round(total_revenue,2))

print("Retention Rate:",
      round(retention_rate,2), "%")

print("Customer Lifetime Value:",
      round(clv,2))

print("Forecast Revenue:",
      round(future_revenue,2))

print("High Risk Customers:",
      high_risk)

print("\nSystem Integration Successful!")