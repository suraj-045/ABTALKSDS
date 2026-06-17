import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
# -------------------------
# Title
# -------------------------
st.title("Customer Intelligence Dashboard")

# -------------------------
# Load Data
# -------------------------
csv_path = os.path.join(os.path.dirname(__file__), "creditcard_small.csv")
data = pd.read_csv(csv_path)

# -------------------------
# KPI Section
# -------------------------
st.header("Key Performance Indicators")

total_customers = len(data)
total_revenue = data["Amount"].sum()
avg_revenue = data["Amount"].mean()
fraud_cases = data["Class"].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Customers", total_customers)
col2.metric("Revenue", round(total_revenue, 2))
col3.metric("Avg Amount", round(avg_revenue, 2))
col4.metric("Fraud Cases", fraud_cases)

# -------------------------
# Filter Section
# -------------------------
st.sidebar.header("Filters")

min_amount = st.sidebar.slider(
    "Minimum Amount",
    float(data["Amount"].min()),
    float(data["Amount"].max()),
    float(data["Amount"].min())
)

filtered_data = data[data["Amount"] >= min_amount]

# -------------------------
# Revenue Distribution
# -------------------------
st.subheader("Revenue Distribution")

fig, ax = plt.subplots()
ax.hist(filtered_data["Amount"], bins=10)
st.pyplot(fig)

# -------------------------
# Fraud Analysis
# -------------------------
st.subheader("Fraud Analysis")

fraud_counts = filtered_data["Class"].value_counts()

fig2, ax2 = plt.subplots()
ax2.bar(["Normal", "Fraud"], fraud_counts)
st.pyplot(fig2)

# -------------------------
# Data Preview
# -------------------------
st.subheader("Customer Data")

st.dataframe(filtered_data.head(10))

st.success("Dashboard Loaded Successfully!")