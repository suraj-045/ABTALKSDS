import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

# Load dataset
csv_path = os.path.join(os.path.dirname(__file__), "creditcard_small.csv")
data = pd.read_csv(csv_path)

# Features
X = data[['Amount', 'Time']]

# Create 2 clusters (Best K from Day 30)
kmeans = KMeans(n_clusters=2, random_state=42)

data['Cluster'] = kmeans.fit_predict(X)

# Cluster Summary
summary = data.groupby('Cluster')[['Amount', 'Time']].mean()

print("Customer Persona Summary")
print(summary)

# Assign Persona Names
persona_names = {
    0: "Regular Customers",
    1: "Premium Customers"
}

data['Persona'] = data['Cluster'].map(persona_names)

print("\nSample Personas:")
print(data[['Amount', 'Time', 'Persona']].head())

# Visualization
plt.scatter(
    data['Amount'],
    data['Time'],
    c=data['Cluster']
)

plt.xlabel("Amount")
plt.ylabel("Time")
plt.title("Customer Personas")

plt.show()