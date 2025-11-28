📌 COVID-19 ICU Admission Prediction

This project focuses on predicting the likelihood of ICU admission for COVID-19 patients using machine learning and exploratory data analysis. The study uses a dataset of 200,000+ anonymized patient records, including symptoms, demographics, and comorbidity information.

The goal is to build a data-driven solution that highlights key risk factors, supports clinical decision-making, and segments patients into meaningful clusters.

🚀 Project Overview

This project was completed in two major phases:

Part 1 — Data Cleaning, EDA & Baseline Modeling

Performed aggressive data cleaning (handling missing values, inconsistent encodings, outliers).

Conducted exploratory data analysis to identify ICU risk patterns.

Built a baseline Logistic Regression model.

Improved performance using SMOTE to balance the dataset.

Part 2 — Improved Modeling, Clustering & PySpark

Applied imputation-based cleaning to retain more data.

Built improved models including Random Forest.

Performed K-Means clustering to segment patients by risk profile.

Visualized cluster separation via PCA.

Implemented a scalable pipeline using PySpark.

📊 Key Features & Analysis Steps
✔ Data Preprocessing

Cleaned inconsistent entries (e.g., "?", missing codes).

Converted categorical numerics and date fields.

Generated additional derived features such as DIED.

✔ Exploratory Data Analysis

ICU class distribution

ICU vs Age

ICU vs Comorbidities (obesity, hypertension, renal chronic, etc.)

Correlation heatmaps

PCA visualizations

✔ Predictive Modeling

Models built and evaluated:

Logistic Regression

Random Forest Classifier

Evaluation metrics:

Accuracy

Precision

Recall

F1 Score

ROC-AUC

✔ Class Imbalance Handling

Applied SMOTE to oversample the minority ICU class.

✔ Clustering & Segmentation

Performed K-Means (k=3).

Identified distinct patient groups with differing ICU likelihoods.

Visualized using PCA dimensionality reduction.

✔ Dashboarding & Visualization

Built interactive dashboards in Google Looker Studio.

Included risk trends, comorbidity charts, PCA plots, cluster distributions, and model summaries.

✔ PySpark Implementation

Built a data pipeline for large-scale processing.

Used MLlib for feature assembly, scaling, and logistic regression modeling.

🛠 Tools & Technologies

Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)

PySpark (Spark SQL + MLlib)

Looker Studio (Dashboarding)

Jupyter Notebook

SMOTE (imbalanced-learn)

KMeans, PCA for clustering & dimensionality reduction

📈 Results & Insights

ICU admission strongly correlated with age, pneumonia, hypertension, obesity, and renal chronic conditions.

Random Forest achieved the best predictive performance.

SMOTE significantly improved recall on high-risk patients.

Clustering revealed distinct patient groups with varying severity levels.

PySpark pipeline validated the scalability of the approach.
