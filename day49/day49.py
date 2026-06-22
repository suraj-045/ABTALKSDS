import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load Dataset
csv_path = os.path.join(os.path.dirname(__file__), "creditcard_small.csv")
data = pd.read_csv(csv_path)

# -------------------------
# Feature Engineering
# -------------------------
data["HighAmount"] = (
    data["Amount"] > data["Amount"].mean()
).astype(int)

# -------------------------
# Scaling
# -------------------------
scaler = StandardScaler()

data[["Amount"]] = scaler.fit_transform(
    data[["Amount"]]
)

# -------------------------
# Customer Segmentation
# -------------------------
kmeans = KMeans(
    n_clusters=2,
    random_state=42
)

data["Segment"] = kmeans.fit_predict(
    data[["Amount", "Time"]]
)

# -------------------------
# Churn Label
# -------------------------
data["Churn"] = (
    data["HighAmount"] == 0
).astype(int)

# -------------------------
# Churn Prediction Model
# -------------------------
X = data[["Amount", "Time", "Segment"]]
y = data["Churn"]

model = LogisticRegression(
    max_iter=1000
)

model.fit(X, y)

data["ChurnProbability"] = (
    model.predict_proba(X)[:,1]
)

# -------------------------
# Results
# -------------------------
print("Customer Segments:")
print(data["Segment"].value_counts())

print("\nTop High Risk Customers:")
print(
    data[
        ["Amount",
         "Time",
         "ChurnProbability"]
    ].head()
)

print("\nBaseline Prototype Ready!")