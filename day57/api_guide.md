API Usage Guide

Endpoint

POST /predict

Request

{
“Amount”: 500,
“Time”: 20000,
“Segment”: 1
}

Success Response

{
“Churn_Probability”: 0.13
}

Error Response

{
“error”: “Invalid Input”
}

Validation

* Amount must be greater than 0.
* Segment must be either 0 or 1.
* Time cannot be negative.

Usage

The API predicts customer churn probability based on transaction information.