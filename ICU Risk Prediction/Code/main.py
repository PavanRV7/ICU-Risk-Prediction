# %%

# ==============================
# 1. Load Data
# ==============================

import pandas as pd

df = pd.read_csv(r"D:\R V Pavan\Projects\Healthcare ICU Demand Forecasting & Patient Risk Stratification\Dataset\icu_prediction_data.csv")

print(df.head())
print(df.info())


# %%

# ==============================
# 2. Data Cleaning & Feature Engineering
# ==============================

# Handle missing values
df["ICU_STAY_DAYS"] = df["ICU_STAY_DAYS"].fillna(0)

# Remove data leakage
df.drop(columns=["ICU_STAY_DAYS"], inplace=True)

# Drop unnecessary columns
df.drop(columns=["admittime", "dischtime", "subject_id", "hadm_id"], inplace=True)

# ----------------------------
# Create Age Groups
# ----------------------------
df["AGE_GROUP"] = pd.cut(
    df["anchor_age"],
    bins=[0, 30, 60, 100],
    labels=[0, 1, 2]
).astype(int)

# ----------------------------
# Create Dashboard Labels
# ----------------------------
df["AGE_GROUP_LABEL"] = df["AGE_GROUP"].map({
    0: "Young (0-30)",
    1: "Adult (30-60)",
    2: "Senior (60+)"
})

df["gender_label"] = df["gender"]  # keep original for dashboard

# ----------------------------
# Encode gender for ML
# ----------------------------
df["gender"] = df["gender"].map({"M": 1, "F": 0})


# %%

# ==============================
# SAVE CLEAN DATA FOR DASHBOARD
# ==============================

df_dashboard = df[[
    "gender_label",
    "AGE_GROUP_LABEL",
    "anchor_age",
    "HOSPITAL_STAY_DAYS",
    "ICU_FLAG"
]]

df_dashboard.to_csv(
    r"D:\R V Pavan\Projects\Healthcare ICU Demand Forecasting & Patient Risk Stratification\Dataset\cleaned_icu_dashboard.csv",
    index=False
)

print("\n Dashboard dataset saved!")


# %%

# ==============================
# 3. Split Features & Target (ML DATA)
# ==============================

X = df.drop(columns=["ICU_FLAG", "gender_label", "AGE_GROUP_LABEL"])
y = df["ICU_FLAG"]


# %%

# ==============================
# 4. Train-Test Split
# ==============================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# %%

# ==============================
# 5. Handle Imbalance (SMOTE)
# ==============================

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


# %%

# ==============================
# 6. Scaling (ONLY for Logistic Regression)
# ==============================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)


# %%

# ==============================
# 7. Train Models
# ==============================

# Logistic Regression
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(class_weight="balanced", max_iter=1000)
lr_model.fit(X_train_scaled, y_train_res)

# Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(class_weight="balanced", random_state=42)
rf_model.fit(X_train_res, y_train_res)


# %%

# ==============================
# 8. Evaluation
# ==============================

from sklearn.metrics import classification_report, roc_auc_score

# Logistic Regression
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
y_pred_lr = (lr_prob > 0.4).astype(int)

print("===== Logistic Regression =====")
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, lr_prob))


# Random Forest
rf_prob = rf_model.predict_proba(X_test)[:, 1]
y_pred_rf = (rf_prob > 0.4).astype(int)

print("\n===== Random Forest =====")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, rf_prob))


# %%

# ==============================
# 9. Feature Importance
# ==============================

import numpy as np

feature_importance = pd.Series(
    np.abs(lr_model.coef_[0]),
    index=X.columns
)

print("\nFeature Importance:")
print(feature_importance.sort_values(ascending=False))


# %%

# ==============================
# 10. Confusion Matrix
# ==============================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_lr)

print("\nConfusion Matrix (Logistic Regression):")
print(cm)