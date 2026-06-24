import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
csv_path = os.path.join(
    os.path.dirname(__file__),
    "creditcard_small.csv"
)

data = pd.read_csv(csv_path)

# Feature Engineering
data["HighAmount"] = (
    data["Amount"] > data["Amount"].mean()
).astype(int)

# Customer Segmentation
kmeans = KMeans(
    n_clusters=2,
    random_state=42
)

data["Segment"] = kmeans.fit_predict(
    data[["Amount", "Time"]]
)

# Churn Label
data["Churn"] = (
    data["HighAmount"] == 0
).astype(int)

# Features
X = data[["Amount", "Time", "Segment"]]
y = data["Churn"]

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Feature Importance
importance = model.feature_importances_

results = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

results = results.sort_values(
    by="Importance",
    ascending=False
)

print("Feature Importance Analysis")
print(results)

# Visualization
plt.figure(figsize=(6,4))
plt.bar(
    results["Feature"],
    results["Importance"]
)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.tight_layout()
plt.show()

print("\nExplainable AI Analysis Completed!")