# train_pph_model.py
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
df = pd.read_csv("pph_cleaned_dataset.csv")  # Update path

# Separate features and target
X = df.drop('case', axis=1)  # Replace 'pph' with your target column name
y = df['case']

# -----------------------------
# Step 2: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 3: Handle Categorical Columns
# -----------------------------
# Find all object columns
categorical_cols = X_train.select_dtypes(include='object').columns
print("Categorical columns detected:", categorical_cols)

# Convert categorical columns using Label Encoding
le = LabelEncoder()
for col in categorical_cols:
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# -----------------------------
# Step 4: Train LightGBM Model
# -----------------------------
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=-1,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(100)]
)

# -----------------------------
# Step 5: Make Predictions
# -----------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# -----------------------------
# Step 6: Evaluate Model
# -----------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

# -----------------------------
# Step 7: Save the Model
# -----------------------------
import joblib
joblib.dump(model, "pph_model.pkl")
print("Model saved successfully!")