import pandas as pd
import matplotlib.pyplot as plt
import os
# Load dataset
csv_path = os.path.join(os.path.dirname(__file__), "creditcard_small.csv")
data = pd.read_csv(csv_path)


# KPI Calculations
total_customers = len(data)

total_revenue = data["Amount"].sum()

average_revenue = data["Amount"].mean()

retention_rate = (
    len(data[data["Amount"] > average_revenue])
    / total_customers
) * 100

fraud_cases = data["Class"].sum()

# KPI Dashboard
print("===== KPI DASHBOARD =====")
print("Total Customers:", total_customers)
print("Total Revenue:", round(total_revenue, 2))
print("Average Revenue:", round(average_revenue, 2))
print("Retention Rate:", round(retention_rate, 2), "%")
print("Fraud Cases:", fraud_cases)

# =====================
# Revenue Distribution
# =====================

plt.figure(figsize=(6,4))
plt.hist(data["Amount"], bins=10)

plt.title("Revenue Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")

plt.show()

# =====================
# Fraud Monitoring
# =====================

fraud_counts = data["Class"].value_counts()

plt.figure(figsize=(5,4))
plt.bar(["Normal", "Fraud"], fraud_counts)

plt.title("Fraud Monitoring")

plt.show()

# =====================
# Revenue Trend
# =====================

plt.figure(figsize=(6,4))
plt.plot(data["Time"].head(50),
         data["Amount"].head(50))

plt.title("Revenue Trend")
plt.xlabel("Time")
plt.ylabel("Revenue")

plt.show()

print("\nKPI Monitoring System Completed!")