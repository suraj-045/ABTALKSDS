# Day 46 - Production Monitoring

## Features
- Logging system
- Input validation
- Error handling
- Prediction tracking

## API Endpoint

POST /predict

## Example Request

{
  "Amount": 150,
  "Time": 15,
  "Segment": 0
}

## Example Response

{
  "Churn_Probability": 0.85
}

## Monitoring Improvements
- Logs every prediction
- Detects invalid inputs
- Handles errors safely
- Improves reliability

## Outcome
Enhanced the ML system with production-level monitoring and validation.