import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
csv_path = os.path.join(os.path.dirname(__file__), "creditcard_small.csv")
data = pd.read_csv(csv_path)

# Select features
X = data[['Amount', 'Time']]

# -------------------------
# Elbow Method
# -------------------------
wcss = []

for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(2, 7), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# -------------------------
# Silhouette Scores
# -------------------------
print("K | Silhouette Score")

for k in range(2, 7):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )

    labels = kmeans.fit_predict(X)

    score = silhouette_score(X, labels)

    print(k, "|", round(score, 3))

# -------------------------
# Final Model (K=3)
# -------------------------
kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

data["Cluster"] = kmeans.fit_predict(X)

# -------------------------
# Visualization
# -------------------------
plt.scatter(
    data["Amount"],
    data["Time"],
    c=data["Cluster"]
)

plt.title("Customer Segments")
plt.xlabel("Amount")
plt.ylabel("Time")

plt.show()

print("\nCluster Sample:")
print(data[["Amount", "Time", "Cluster"]].head())