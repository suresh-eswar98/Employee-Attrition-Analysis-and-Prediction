Employee Attrition Prediction (HR Analytics Project)

🔎 Overview
Employee attrition (when employees leave the company) is a major HR challenge.
This project analyzes employee data, identifies key drivers of attrition, and predicts whether an employee is likely to leave (Attrition = Yes) or stay (Attrition = No).

This project uses:
- Data Analytics & Preprocessing
- Classification Algorithms (Random Forest, Logistic Regression)
- Feature Engineering (encoding categorical variables)
- Model Evaluation (Accuracy, F1, AUC-ROC, Confusion Matrix)
- Streamlit Dashboard for interactive predictions

🛠️ Tech Stack
- Python
- pandas, numpy (Data processing)
- scikit-learn (Machine Learning models & metrics)
- matplotlib, seaborn (Visualization)
- Streamlit (Dashboard & interactive predictions)

📂 Project Structure
- attrition_app.py            # Main Streamlit application
- Employee_Attrition_Employee_Attrition.csv  # Dataset
- README.txt                   # Project documentation

⚙️ Installation & Setup
1. Clone this repository:
   git clone https://github.com/your-username/employee-attrition-prediction.git
   cd employee-attrition-prediction

2. Install dependencies:
   pip install -r requirements.txt

3. Run the Streamlit app:
   streamlit run attrition_app.py

📊 Features
1. Data Preprocessing
- Dropped irrelevant columns (EmployeeNumber, EmployeeCount, Over18, StandardHours)
- Encoded categorical columns using LabelEncoder
- Selected 4 important features: Age, MonthlyIncome, JobRole, OverTime

2. Classification Models
- Random Forest Classifier
- Logistic Regression

3. Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- AUC-ROC Curve

4. Streamlit Dashboard
- Dataset preview
- Model performance comparison
- ROC curves
- Feature importance (Random Forest)
- Interactive form to predict attrition for a new employee
- Download predictions for test employees as CSV

🚀 Business Use Cases
- Identify employees at risk of leaving
- Reduce hiring & training costs
- Improve retention strategies with data-driven insights
- Assist HR teams in workforce planning

📥 Example Prediction
Using the dashboard form:
Input:
- Age = 30
- Monthly Income = 5000
- JobRole = Sales Executive
- OverTime = Yes

Output:
- Random Forest → Likely to Leave (80% probability)
- Logistic Regression → Likely to Stay (65% probability)

📌 Future Improvements
- Add more features for better accuracy
- Try advanced models (XGBoost, LightGBM)
- Deploy app on Streamlit Cloud / Heroku
- Add SHAP values for model explainability


