import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Generate Synthetic Financial Dataset
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'income': np.random.normal(50000, 15000, n_samples),
    'debts': np.random.normal(15000, 5000, n_samples),
    'payment_history_score': np.random.uniform(300, 850, n_samples),
    'credit_lines': np.random.randint(1, 10, n_samples)
})

# Feature Engineering: Debt-to-Income Ratio
data['debt_to_income'] = data['debts'] / data['income']

# Target: 1 for Creditworthy, 0 for High Risk
data['creditworthy'] = np.where(
    (data['debt_to_income'] < 0.35) & (data['payment_history_score'] > 620), 1, 0
)

# 2. Features and Target
X = data.drop(columns=['creditworthy'])
y = data['creditworthy']

# 3. Train-Test Split & Feature Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Model Training (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Evaluation Metrics
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("--- Task 1: Credit Scoring Evaluation ---")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

