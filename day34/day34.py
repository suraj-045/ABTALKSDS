import pandas as pd
import matplotlib.pyplot as plt
import os

# Load dataset
csv_path = os.path.join(os.path.dirname(__file__), "creditcard_small.csv")
data = pd.read_csv(csv_path)

# KPI Metrics
total_customers = len(data)
total_amount = data["Amount"].sum()
average_amount = data["Amount"].mean()
fraud_cases = data["Class"].sum()

# KPI Cards
print("===== CUSTOMER DASHBOARD =====")
print("Total Customers:", total_customers)
print("Total Transaction Amount:", round(total_amount, 2))
print("Average Transaction Amount:", round(average_amount, 2))
print("Fraud Cases:", fraud_cases)

# -------------------------
# Chart 1: Amount Distribution
# -------------------------
plt.figure(figsize=(6,4))
plt.hist(data["Amount"], bins=10)

plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")

plt.show()

# -------------------------
# Chart 2: Fraud vs Normal
# -------------------------
counts = data["Class"].value_counts()

plt.figure(figsize=(5,4))
plt.bar(["Normal", "Fraud"], counts)

plt.title("Fraud vs Normal Transactions")

plt.show()

# -------------------------
# Chart 3: Transaction Trend
# -------------------------
plt.figure(figsize=(6,4))
plt.plot(data["Time"].head(50),
         data["Amount"].head(50))

plt.title("Transaction Trend")
plt.xlabel("Time")
plt.ylabel("Amount")

plt.show()

print("\nDashboard Created Successfully!")