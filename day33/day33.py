import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import IsolationForest

# Load dataset
csv_path = os.path.join(os.path.dirname(__file__), "creditcard_small.csv")
data = pd.read_csv(csv_path)

# Features
X = data[['Amount', 'Time']]

# Anomaly Detection
model = IsolationForest(
    contamination=0.05,
    random_state=42
)

data['Anomaly'] = model.fit_predict(X)

# Count anomalies
print("Normal Records:",
      len(data[data['Anomaly'] == 1]))

print("Anomalies:",
      len(data[data['Anomaly'] == -1]))

# Show anomaly samples
print("\nAnomaly Samples:")
print(
    data[data['Anomaly'] == -1]
    [['Amount', 'Time']]
    .head()
)

# Visualization
plt.scatter(
    data['Amount'],
    data['Time'],
    c=data['Anomaly']
)

plt.xlabel("Amount")
plt.ylabel("Time")
plt.title("Anomaly Detection")

plt.show()