# ZenSutra
Real-time customer churn prediction API built on AWS with a robust MLOps pipeline. Predicts customer churn, enables proactive retention strategies, and showcases end-to-end ML lifecycle management.
## Real-time Customer Churn Prediction API
A scalable MLOps pipeline for real-time customer churn prediction, built entirely on AWS to enable proactive customer retention strategies.

## Project Overview
This project demonstrates the development and deployment of a **real-time customer churn prediction API**, leveraging industry-standard **Machine Learning Operations (MLOps)** practices on **Amazon Web Services (AWS)**. It focuses on creating a robust, scalable, and maintainable system that can be integrated into business applications to identify customers at high risk of churning.

## Problem Statement 
Customer churn, or the loss of customers, is a significant challenge across various industries, including telecommunications, SaaS, and e-commerce. Retaining existing customers is often more cost-effective than acquiring new ones. Without timely identification of at-risk customers, businesses miss opportunities to intervene with targeted retention strategies, leading to substantial revenue loss and decreased customer lifetime value.

The challenge lies in building a system that can:
1.  Accurately predict churn likelihood.
2.  Provide these predictions in **real-time** as new customer data becomes available.
3.  Operate reliably and scalably in a production environment.
4.  Be easily updated and maintained as data and models evolve.

## Solution & Architecture 
This project provides a comprehensive solution to the customer churn problem by implementing a serverless, real-time prediction API. The architecture is designed for scalability, reliability, and cost-efficiency.

At a high level, the solution involves:
1.  **Data Simulation & Storage:** Mock customer data is generated and stored in an AWS S3 bucket. In a real-world scenario, this would be integrated with operational databases or data lakes.
2.  **Model Training:** A machine learning model (e.g., Logistic Regression or XGBoost) is trained to predict churn based on customer features.
3.  **Model Deployment:** The trained model is deployed behind a FastAPI application, served as a **containerized AWS Lambda function**.
4.  **Real-time Inference API:** AWS API Gateway exposes a secure REST API endpoint for applications to send customer data and receive churn predictions instantly.
5.  **Automated MLOps Pipeline:** GitHub Actions orchestrates the entire CI/CD process, from code changes to infrastructure provisioning and model deployment.
6.  **Monitoring:** AWS CloudWatch is used to log API invocations and monitor the health and performance of the deployed service.
