from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
import pandas as pd

app = Flask(__name__)

data = pd.DataFrame({
    "Amount": [100, 200, 300, 400, 500, 600],
    "Time": [10, 20, 30, 40, 50, 60],
    "Segment": [0, 0, 1, 1, 0, 1],
    "Churn": [1, 1, 1, 0, 0, 0]
})

X = data[["Amount", "Time", "Segment"]]
y = data["Churn"]

model = LogisticRegression()
model.fit(X, y)

@app.route("/")
def home():
    return "Customer Risk Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():

    customer = request.json

    features = [[
        customer["Amount"],
        customer["Time"],
        customer["Segment"]
    ]]

    probability = model.predict_proba(features)[0][1]

    return jsonify({
        "Churn_Probability": round(float(probability), 3)
    })

if __name__ == "__main__":
    app.run(debug=True)