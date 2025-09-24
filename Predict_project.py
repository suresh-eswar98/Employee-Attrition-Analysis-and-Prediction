import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("Employee_Attrition_Employee_Attrition.csv")

# Keep only important features for simplicity
selected_features = ["Age", "MonthlyIncome", "JobRole", "OverTime", "Attrition"]
df = df[selected_features]

# Encode categorical columns
label_encoders = {}
for col in ["JobRole", "OverTime", "Attrition"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -----------------------------
# 2. Train-Test Split
# -----------------------------
X = df[["Age", "MonthlyIncome", "JobRole", "OverTime"]]
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 3. Train Models
# -----------------------------
# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
y_pred_prob_lr = lr_model.predict_proba(X_test)[:, 1]

# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.title("📊 Employee Attrition Prediction (HR Analytics Project)")
st.write("Using 4 features only (Age, Income, JobRole, OverTime).")

# Dataset Preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Random Forest Performance
# -----------------------------
st.subheader("🌲 Random Forest Performance")
st.write("✅ Accuracy:", round(accuracy_score(y_test, y_pred_rf), 3))
st.text("Classification Report:\n" + classification_report(y_test, y_pred_rf))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred_rf))

roc_auc_rf = roc_auc_score(y_test, y_pred_prob_rf)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)

fig, ax = plt.subplots()
ax.plot(fpr_rf, tpr_rf, label=f"Random Forest AUC = {roc_auc_rf:.3f}")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - Random Forest")
ax.legend()
st.pyplot(fig)

# -----------------------------
# Logistic Regression Performance
# -----------------------------
st.subheader("📉 Logistic Regression Performance")
st.write("✅ Accuracy:", round(accuracy_score(y_test, y_pred_lr), 3))
st.text("Classification Report:\n" + classification_report(y_test, y_pred_lr))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred_lr))

roc_auc_lr = roc_auc_score(y_test, y_pred_prob_lr)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_prob_lr)

fig, ax = plt.subplots()
ax.plot(fpr_lr, tpr_lr, label=f"LogReg AUC = {roc_auc_lr:.3f}", color="orange")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - Logistic Regression")
ax.legend()
st.pyplot(fig)

# -----------------------------
# Prediction Form (Simplified)
# -----------------------------
st.subheader("🔍 Predict Attrition for a New Employee")

age = st.number_input("Age", min_value=18, max_value=60, value=30)
income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)

jobrole = st.selectbox("Job Role", list(label_encoders["JobRole"].classes_))
overtime = st.selectbox("OverTime", list(label_encoders["OverTime"].classes_))

# Convert categorical inputs
jobrole_encoded = label_encoders["JobRole"].transform([jobrole])[0]
overtime_encoded = label_encoders["OverTime"].transform([overtime])[0]

# Input dataframe for prediction
input_data = pd.DataFrame([{
    "Age": age,
    "MonthlyIncome": income,
    "JobRole": jobrole_encoded,
    "OverTime": overtime_encoded
}])

if st.button("Predict"):
    # Random Forest
    pred_rf = rf_model.predict(input_data)[0]
    prob_rf = rf_model.predict_proba(input_data)[0][1]

    # Logistic Regression
    pred_lr = lr_model.predict(input_data)[0]
    prob_lr = lr_model.predict_proba(input_data)[0][1]

    st.write("### 🌲 Random Forest Result")
    if pred_rf == 1:
        st.error(f"⚠️ Likely to Leave | Probability: {prob_rf:.2f}")
    else:
        st.success(f"✅ Likely to Stay | Probability: {prob_rf:.2f}")

    st.write("### 📉 Logistic Regression Result")
    if pred_lr == 1:
        st.error(f"⚠️ Likely to Leave | Probability: {prob_lr:.2f}")
    else:
        st.success(f"✅ Likely to Stay | Probability: {prob_lr:.2f}")

# -----------------------------
# Download Predictions
# -----------------------------
st.subheader("📥 Download Predictions for Test Employees")

df_results = X_test.copy()
df_results["Actual Attrition"] = y_test.values
df_results["Pred_RF"] = y_pred_rf
df_results["Prob_RF"] = y_pred_prob_rf
df_results["Pred_LR"] = y_pred_lr
df_results["Prob_LR"] = y_pred_prob_lr

csv = df_results.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📩 Download Predictions as CSV",
    data=csv,
    file_name="employee_attrition_predictions.csv",
    mime="text/csv",
)
