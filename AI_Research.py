# -------------------------------
# IMPORTS
# -------------------------------
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

# -------------------------------
# LOAD DATA
# -------------------------------
info = pd.read_csv("studentInfo.csv", na_values="?")
assess = pd.read_csv("studentAssessment.csv", na_values="?")

# -------------------------------
# CLEAN DATA
# -------------------------------
assess["score"] = pd.to_numeric(assess["score"], errors="coerce")
assess = assess.dropna(subset=["score"])

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
student_features = (
    assess.groupby("id_student").agg({"score": ["mean", "std", "count"]}).reset_index()
)

student_features.columns = ["id_student", "avg_score", "score_std", "num_assessments"]

student_features["score_std"] = student_features["score_std"].fillna(0)

# -------------------------------
# MERGE
# -------------------------------
df = pd.merge(info, student_features, on="id_student", how="inner")

# -------------------------------
# SEE DATA
# -------------------------------
print("\nColumns:")
print(df.columns)

print("\nFirst rows:")
print(df.head())

print("\nShape:")
print(df.shape)

# -------------------------------
# ENCODING
# -------------------------------
cat_cols = df.select_dtypes(include=["object"]).columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# -------------------------------
# REMOVE DATA LEAKAGE
# -------------------------------
cols_to_remove = [col for col in df.columns if "final_result" in col]

# -------------------------------
# FEATURES AND TARGET
# -------------------------------
X = df.drop(columns=["id_student", "avg_score"] + cols_to_remove, errors="ignore")
y = df["avg_score"]

# -------------------------------
# CHECK (optional but recommended)
# -------------------------------
print("\nCheck leakage columns:")
print([col for col in X.columns if "final_result" in col])

# -------------------------------
# SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# MODEL
# -------------------------------
gb = GradientBoostingRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
)

gb.fit(X_train, y_train)
preds = gb.predict(X_test)

# -------------------------------
# RESULTS
# -------------------------------
print("\n=== RESULTS ===")
print("MSE:", round(mean_squared_error(y_test, preds), 2))
print("R2:", round(r2_score(y_test, preds), 4))
