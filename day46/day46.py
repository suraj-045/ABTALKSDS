import logging
from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
import pandas as pd

# -------------------------
# Logging Setup
# -------------------------
logging.basicConfig(
    filename="logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

app = Flask(__name__)

# -------------------------
# Sample Model
# -------------------------
data = pd.DataFrame({
    "Amount": [100, 200, 300, 400, 500, 600],
    "Time": [10, 20, 30, 40, 50, 60],
    "Segment": [0, 0, 1, 1, 0, 1],
    "Churn": [1, 1, 1, 0, 0, 0]
})

X = data[["Amount", "Time", "Segment"]]
y = data["Churn"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# -------------------------
# Home Route
# -------------------------
@app.route("/")
def home():
    return "Production Monitoring API Running"

# -------------------------
# Prediction Route
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Validation
        if not all(k in data for k in ("Amount", "Time", "Segment")):
            logging.error("Invalid input: Missing fields")
            return jsonify({"error": "Missing fields"}), 400

        amount = float(data["Amount"])
        time = float(data["Time"])
        segment = int(data["Segment"])

        # Model Prediction
        features = [[amount, time, segment]]
        probability = model.predict_proba(features)[0][1]

        # Logging
        logging.info(f"Input: {data} | Prediction: {probability}")

        return jsonify({
            "Churn_Probability": round(float(probability), 3)
        })

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500


if __name__ == "__main__":
    app.run(debug=True)