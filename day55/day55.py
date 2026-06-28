import streamlit as st
import pandas as pd
import os

from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Customer Intelligence Platform",
    layout="wide"
)

st.title("🚀 Customer Intelligence Platform")
st.subheader("Optimized Analytics Dashboard")

# Cache dataset for better performance
@st.cache_data
def load_data():
    csv_path = os.path.join(
        os.path.dirname(__file__),
        "creditcard_small.csv"
    )
    return pd.read_csv(csv_path)

data = load_data()

# Feature Engineering
data["HighAmount"] = (
    data["Amount"] > data["Amount"].mean()
).astype(int)

data["Segment"] = (
    data["Amount"] > data["Amount"].median()
).astype(int)

data["Churn"] = (
    data["HighAmount"] == 0
).astype(int)

X = data[["Amount", "Time", "Segment"]]
y = data["Churn"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Sidebar
st.sidebar.header("Customer Input")

amount = st.sidebar.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100.0
)

time = st.sidebar.number_input(
    "Transaction Time",
    min_value=0,
    value=100
)

segment = st.sidebar.selectbox(
    "Customer Segment",
    [0, 1]
)

# Validation
if st.sidebar.button("Predict"):

    if amount <= 0:
        st.error("Amount must be greater than zero.")

    else:
        try:
            prediction = model.predict(
                [[amount, time, segment]]
            )[0]

            probability = model.predict_proba(
                [[amount, time, segment]]
            )[0][1]

            st.header("Prediction Result")

            if prediction == 1:
                st.error("⚠ High Churn Risk")
            else:
                st.success("✅ Low Churn Risk")

            st.metric(
                "Churn Probability",
                round(probability,3)
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")

st.subheader("Dataset Preview")
st.dataframe(data.head())

st.success("Optimized Customer Intelligence Platform Running")