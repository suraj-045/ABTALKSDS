import streamlit as st
import pandas as pd
import os

from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Customer Intelligence Platform",
    layout="wide"
)

st.title("Customer Intelligence Platform")
st.subheader("Real-Time Customer Prediction")

# Load Dataset
csv_path = os.path.join(
    os.path.dirname(__file__),
    "creditcard_small.csv"
)

data = pd.read_csv(csv_path)

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

if st.sidebar.button("Predict"):

    prediction = model.predict(
        [[amount, time, segment]]
    )[0]

    probability = model.predict_proba(
        [[amount, time, segment]]
    )[0][1]

    st.header("Prediction Result")

    if prediction == 1:
        st.error("High Churn Risk")
    else:
        st.success("Low Churn Risk")

    st.write(
        "Churn Probability:",
        round(probability,3)
    )

st.subheader("Dataset Preview")

st.dataframe(data.head())