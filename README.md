Global Attention Shift Predictor (GASP)
📌 Project Overview

Global Attention Shift Predictor (GASP) is a data-driven machine learning system designed to analyze, model, and predict how human digital attention shifts over time.

Instead of focusing on a single platform, this project integrates multiple behavioral signals to build a unified and interpretable representation of attention using a custom metric called the Human Attention Index (HAI).

The system performs end-to-end data preprocessing, exploratory data analysis (EDA), forecasting, trend classification, and user-driven prediction through an interactive web interface.

🎯 Problem Statement

Human digital attention is fragmented across platforms such as search engines, video platforms, social media, and mobile usage.
Analyzing these platforms independently gives an incomplete and biased understanding of attention behavior.

This project addresses the challenge of:

How can we combine multiple digital engagement signals to understand and predict where human attention is moving next?

🧠 Core Idea

The project introduces a Human Attention Index (HAI) — a unified metric that captures overall digital attention by combining normalized engagement signals from multiple platforms.

Each platform represents a different dimension of attention:

Platform	Represents
Google Search	Curiosity & intent
YouTube	Passive content consumption
TikTok	Viral engagement
Screen Time	Habitual digital behavior
🗂️ Datasets Used

Google Search Trends – Search interest over time

YouTube Dataset – Views and likes based on publish date

TikTok Dataset – Viral engagement metrics

Screen Time Dataset – Daily app usage duration

All datasets are cleaned, normalized, and aligned on a daily time scale before analysis.

⚙️ Methodology
1️⃣ Data Preprocessing (Backend Only)

Removal of irrelevant columns

Date normalization across datasets

Missing value handling (mean imputation)

Outlier removal using IQR method

Daily aggregation

Z-score normalization

Preprocessing is handled entirely in the backend and not exposed in the UI.

2️⃣ Feature Engineering

Construction of the Human Attention Index (HAI)

HAI = Average of normalized platform engagement signals

Negative values indicate below-average attention, not errors

3️⃣ Exploratory Data Analysis (EDA)

Performed through an interactive dashboard including:

Attention trends over time

Platform contribution analysis

Correlation heatmap

Behavioral patterns across days

Key summary KPIs

4️⃣ Machine Learning Models
Task	Model Used	Purpose
Forecasting	Linear Regression	Predict future attention
Classification	Logistic Regression	Rising vs Falling attention
User Prediction	Regression + Rules	Scenario-based prediction

Models are chosen for interpretability and academic clarity.

5️⃣ User-Driven Prediction

Users can input engagement values and instantly receive:

Predicted Human Attention Index

Interpreted attention level (Low / Moderate / High / Viral)

Trend direction (Rising or Falling)

🖥️ Application Interface

The Streamlit web application includes:

Home Page – Project objective and concept

Dashboard – EDA and KPIs

Trend Analysis – Daily attention movement

Forecast – Short-term attention projection

Prediction Module – Interactive user prediction

🛠️ Tools & Technologies

Programming Language: Python

Libraries: Pandas, NumPy, Scikit-learn

Visualization: Plotly

Web Framework: Streamlit

ML Models: Linear Regression, Logistic Regression
