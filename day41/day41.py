import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

# Load dataset
csv_path = os.path.join(os.path.dirname(__file__), "creditcard_small.csv")
data = pd.read_csv(csv_path)

# Feature Engineering
data["HighAmount"] = (
    data["Amount"] > data["Amount"].mean()
).astype(int)

# Simulated churn label
data["Churn"] = (
    data["Amount"] < data["Amount"].mean()
).astype(int)

# Segmentation
kmeans = KMeans(n_clusters=2, random_state=42)

data["Segment"] = kmeans.fit_predict(
    data[["Amount", "Time"]]
)

# Risk Prediction Model
X = data[["Amount", "Time", "Segment"]]
y = data["Churn"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Predict churn probability
data["Churn_Probability"] = model.predict_proba(X)[:, 1]

# High Risk Customers
high_risk = data.sort_values(
    "Churn_Probability",
    ascending=False
).head(10)

print("Top High-Risk Customers:")
print(high_risk[["Amount", "Time", "Churn_Probability"]])

# Visualization
plt.hist(data["Churn_Probability"], bins=10)

plt.title("Churn Risk Distribution")
plt.xlabel("Churn Probability")
plt.ylabel("Customers")

plt.show()

print("\nRisk Prediction Completed!")