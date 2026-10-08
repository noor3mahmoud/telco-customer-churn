# ==============================
# Import Libraries
# ==============================
from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ==============================
# Load Data
# ==============================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" /"Telco-Customer-Churn.csv"
df = pd.read_csv(DATA_PATH)

# ==============================
# Data Preprocessing
# ==============================
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["SeniorCitizen"] = df["SeniorCitizen"].astype("category")

df = df.drop("customerID", axis=1)

# ==============================
# Define Features and Target
# ==============================
X = df.drop("Churn", axis=1)
y = df["Churn"]

y = y.map({"No": 0, "Yes": 1})

# ==============================
# Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ==============================
# Preprocessing Pipeline
# ==============================
numerical_columns = make_column_selector(dtype_exclude=["object", "category"])
cat_columns = make_column_selector(dtype_include=["object", "category"])

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipeline = Pipeline([
    ("OHE", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numerical_columns),
    ("cat", cat_pipeline, cat_columns)
])

# ==============================
# Final XGBoost Model
# ==============================
model = XGBClassifier(
    subsample=1.0,
    reg_lambda=30,
    reg_alpha=0.15,
    n_estimators=150,
    min_child_weight=5,
    max_depth=4,
    learning_rate=0.05,
    gamma=0.2,
    colsample_bytree=1.0,
    eval_metric="logloss",
    random_state=42
)
# ==============================
# Complete ML Pipeline
# ==============================
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# ==============================
# Train Model
# ==============================
pipeline.fit(X_train, y_train)

# ==============================
# Predictions
# ==============================
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# ==============================
# Model Evaluation
# ==============================
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
