import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
import os

# Load dataset
csv_path = os.path.join(os.path.dirname(__file__), "creditcard_small.csv")
data = pd.read_csv(csv_path)

# =====================
# Customer Segmentation
# =====================
X_cluster = data[['Amount', 'Time']]

kmeans = KMeans(n_clusters=2, random_state=42)

data['Cluster'] = kmeans.fit_predict(X_cluster)

print("Customer Segmentation Completed")

# =====================
# Recommendation System
# =====================
sample = data[['Amount', 'Time']].head(10)

similarity = cosine_similarity(sample)

scores = list(enumerate(similarity[0]))
scores = sorted(scores,
                key=lambda x: x[1],
                reverse=True)

print("\nTop Recommendations:")

for item in scores[1:4]:
    print("Customer", item[0],
          "Similarity:", round(item[1], 3))

# =====================
# Anomaly Detection
# =====================
iso = IsolationForest(
    contamination=0.05,
    random_state=42
)

data['Anomaly'] = iso.fit_predict(
    data[['Amount', 'Time']]
)

anomalies = len(
    data[data['Anomaly'] == -1]
)

print("\nAnomalies Detected:",
      anomalies)

# =====================
# Dashboard KPIs
# =====================
print("\n===== DASHBOARD =====")

print("Total Customers:",
      len(data))

print("Average Amount:",
      round(data['Amount'].mean(), 2))

print("Fraud Cases:",
      data['Class'].sum())

print("Anomalies:",
      anomalies)

print("\nSystem Integration Successful!")