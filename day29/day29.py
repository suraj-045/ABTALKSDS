import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans

# Load dataset
csv_path = os.path.join(os.path.dirname(__file__), "creditcard_small.csv")
data = pd.read_csv(csv_path)

# Select features (simulate customer behavior)
X = data[['Amount', 'Time']]

# Try different cluster counts
wcss = []

for k in range(1, 6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method
plt.plot(range(1, 6), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Apply KMeans (choose k=3)
kmeans = KMeans(n_clusters=3, random_state=42)

clusters = kmeans.fit_predict(X)

# Add cluster column
data['Cluster'] = clusters

# Visualization
plt.scatter(
    data['Amount'],
    data['Time'],
    c=data['Cluster']
)

plt.xlabel("Amount")
plt.ylabel("Time")
plt.title("Customer Segmentation")

plt.show()

# Show cluster data
print(data[['Amount', 'Time', 'Cluster']].head())