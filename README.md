# 🏥 ICU Demand Forecasting & Patient Risk Stratification

A data analytics and machine learning project built using MIMIC-IV clinical data to predict ICU admissions and analyze patient risk patterns.

## 📌 Project Overview

This project focuses on:

Understanding hospital admission patterns
Identifying patients at risk of ICU admission
Building predictive models for ICU demand
Creating interactive dashboards for business insights

## 🎯 Problem Statement

Hospitals often struggle with:

Unexpected ICU demand
Resource allocation challenges
Late identification of high-risk patients

👉 This project aims to predict ICU admission risk early using patient-level data.

## 📊 Data Source
Dataset: MIMIC-IV v3.1
Source: PhysioNet
Data accessed via Google BigQuery

Includes:
Patient demographics
Hospital admissions
ICU stays

## 🛠️ Tech Stack
SQL (BigQuery) → Data extraction & joins
Python (Pandas, Scikit-learn) → Data processing & ML
Power BI → Dashboard & visualization

## ⚙️ Project Workflow
### 1️⃣ Data Extraction (BigQuery)
Joined:
patients
admissions
icustays
Created target variable:
ICU_FLAG (1 = ICU admission, 0 = No ICU)

### 2️⃣ Data Cleaning & Feature Engineering
Handled missing values
Removed data leakage (ICU_STAY_DAYS)
Created features:
HOSPITAL_STAY_DAYS
AGE_GROUP
Encoded categorical variables

### 3️⃣ Handling Imbalanced Data
Used SMOTE to balance ICU vs non-ICU classes

### 4️⃣ Model Building
🔹 Logistic Regression
Used for interpretability
Applied feature scaling  
🔹 Random Forest
Used for performance comparison  

### 5️⃣ Model Evaluation
Metric	Logistic Regression	Random Forest
ROC-AUC	~0.70	~0.75
Recall (ICU)	High	Moderate
Accuracy	Moderate	Higher

👉 Logistic Regression chosen for:
Better interpretability
Higher recall (important in healthcare)

### 6️⃣ Key Features Driving ICU Risk
Hospital Stay Duration
Age / Age Group
Gender

### 📈 Dashboard (Power BI)

The dashboard provides:  
🔹 KPIs
Total Patients
ICU Patients
ICU Rate
High-Risk Patients
Avg Hospital Stay  

🔹 Visual Insights
ICU Rate by Age Group
ICU Distribution by Gender
ICU Risk vs Hospital Stay Duration

### 📊 Key Insights
ICU admission likelihood increases with longer hospital stays
Certain age groups show higher ICU risk
Gender differences observed in ICU distribution
Dataset is highly imbalanced, requiring careful modeling

### 🚀 How to Run
Clone the repository
Install dependencies:
pip install pandas scikit-learn imbalanced-learn
Run the Python script:
python main.py
Open Power BI dashboard file (.pbix)

### 📂 Project Structure
📁 Project  
│  
├── 📄 main.py                # ML pipeline  
├── 📄 icu_prediction_data.csv  
├── 📊 dashboard.pbix        # Power BI dashboard  
├── 📄 README.md

### 💡 Future Improvements
Add more clinical features (labs, vitals)

### ⚠️ Disclaimer
Data is de-identified (MIMIC-IV)
For educational and research purposes only

### ⭐ Author
Pavan R V
