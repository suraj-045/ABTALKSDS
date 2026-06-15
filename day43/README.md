# Day 43 - Customer Intelligence API

## Features
- Flask API
- Real-time customer risk prediction
- JSON request and response

## Endpoint

POST /predict

## Sample Request

{
  "Amount": 150,
  "Time": 15,
  "Segment": 0
}

## Sample Response

{
  "Churn_Probability": 1.0
}

## Learning

Learned how to deploy a Machine Learning model as a REST API using Flask.
Customer Risk Prediction API

Overview

This API predicts customer churn probability using a Machine Learning model deployed with Flask.

Endpoint

POST /predict

Request Format

{
  "Amount": 150,
  "Time": 15,
  "Segment": 0
}

Response Format

{
  "Churn_Probability": 1.0
}

Workflow

1. User sends customer data to the API.
2. Flask receives the request.
3. The Machine Learning model processes the data.
4. Churn probability is calculated.
5. API returns prediction in JSON format.

Use Cases

* Customer churn prediction
* Customer analytics systems
* Business intelligence platforms
* Retention strategy planning

