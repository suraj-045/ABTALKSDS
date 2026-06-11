import pandas as pd
import matplotlib.pyplot as plt
import os

# Load dataset
csv_path = os.path.join(os.path.dirname(__file__), "creditcard_small.csv")
data = pd.read_csv(csv_path)

# -------------------------
# Retention Rate
# -------------------------
total_customers = len(data)

retained_customers = len(
    data[data["Amount"] > data["Amount"].mean()]
)

retention_rate = (
    retained_customers /
    total_customers
) * 100

print("Retention Rate:")
print(round(retention_rate, 2), "%")

# -------------------------
# Customer Lifetime Value
# -------------------------
average_purchase = data["Amount"].mean()

clv = average_purchase * 12

print("\nCustomer Lifetime Value:")
print(round(clv, 2))

# -------------------------
# High Value Customers
# -------------------------
high_value = data[
    data["Amount"] > average_purchase
]

print("\nHigh Value Customers:")
print(len(high_value))

# -------------------------
# Visualization
# -------------------------
plt.hist(
    data["Amount"],
    bins=10
)

plt.title("Customer Value Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")

plt.show()

print("\nRetention Analytics Completed!")