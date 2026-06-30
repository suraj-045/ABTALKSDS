Customer Intelligence Platform

Overview

The Customer Intelligence Platform is an end-to-end Data Science application developed as the capstone project for the ABTalks 60 Days Data Science Challenge.

The platform helps businesses analyze customer behavior, predict churn risk, monitor KPIs, and generate actionable business insights through an interactive dashboard.

⸻

Business Objectives

* Predict customer churn risk
* Analyze customer behavior
* Segment customers into groups
* Forecast business trends
* Monitor business KPIs
* Support executive decision-making

⸻

Features

* Customer Segmentation
* Churn Prediction
* Revenue Forecasting
* KPI Dashboard
* Explainable AI (Feature Importance)
* Real-Time Prediction
* Production Monitoring
* Performance Optimization

⸻

Technology Stack

* Python
* Pandas
* Scikit-learn
* Streamlit
* Flask
* Matplotlib

⸻

Project Workflow

Customer Data
↓
Data Preprocessing
↓
Feature Engineering
↓
Machine Learning Models
↓
Business Analytics
↓
Interactive Dashboard
↓
Business Insights

⸻

Project Structure

day57/

* README.md
* architecture.md
* api_guide.md
* screenshots/

⸻

Installation

Clone the repository:

git clone https://github.com/suraj-045/ABTALKSDS.git

Install dependencies:

pip install -r requirements.txt

Run the dashboard:

streamlit run day55.py

⸻

API Example

Endpoint:

POST /predict

Example Request

{
“Amount”: 500,
“Time”: 20000,
“Segment”: 1
}

Example Response

{
“Churn_Probability”: 0.13
}

⸻

Screenshots

Include screenshots of:

* Dashboard
* Prediction Result
* KPI Dashboard

⸻

Future Improvements

* User Authentication
* Cloud Database
* Real-Time Streaming
* Advanced Forecasting
* Mobile Dashboard

⸻

Author

Suraj Sharma

ABTalks 60 Days Data Science Challenge