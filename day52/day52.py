import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Customer Intelligence Platform",
    layout="wide"
)

st.title("📊 Customer Intelligence Platform")
st.write("Professional Business Analytics Dashboard")

# Load Dataset
csv_path = os.path.join(
    os.path.dirname(__file__),
    "creditcard_small.csv"
)

data = pd.read_csv(csv_path)

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard",
        "Customer Data",
        "Fraud Analysis"
    ]
)

st.sidebar.header("Filters")

min_amount = st.sidebar.slider(
    "Minimum Amount",
    float(data["Amount"].min()),
    float(data["Amount"].max()),
    float(data["Amount"].min())
)

filtered = data[data["Amount"] >= min_amount]

# ---------------- Dashboard ----------------

if page == "Dashboard":

    st.header("Business KPIs")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Customers", len(filtered))
    c2.metric("Revenue", round(filtered["Amount"].sum(),2))
    c3.metric("Average Amount", round(filtered["Amount"].mean(),2))
    c4.metric("Fraud Cases", int(filtered["Class"].sum()))

    st.subheader("Revenue Distribution")

    fig, ax = plt.subplots()
    ax.hist(filtered["Amount"], bins=10)
    st.pyplot(fig)

# ---------------- Customer Data ----------------

elif page == "Customer Data":

    st.header("Customer Records")
    st.dataframe(filtered)

# ---------------- Fraud Analysis ----------------

else:

    st.header("Fraud Analysis")

    fraud = filtered["Class"].value_counts()

    fig, ax = plt.subplots()

    ax.bar(
        ["Normal","Fraud"],
        fraud
    )

    st.pyplot(fig)

st.success("Customer Intelligence Platform Ready")