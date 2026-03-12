import os
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

from xgboost import XGBClassifier

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "credit_card_fraud_10k.csv")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------
print("Loading dataset from:", DATA_PATH)

if not os.path.exists(DATA_PATH):
    raise Exception("Dataset file not found. Check dataset path.")

data = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully.")
print("Columns:", list(data.columns))

# ---------------------------------------------------
# Detect Target Column
# ---------------------------------------------------
possible_targets = ["is_fraud", "Class", "class", "label", "Label", "fraud", "target"]

target_column = None
for col in possible_targets:
    if col in data.columns:
        target_column = col
        break

if target_column is None:
    raise Exception("Target column not found in dataset.")

print("Target column detected:", target_column)

# ---------------------------------------------------
# Feature Engineering
# ---------------------------------------------------
X = data.drop(target_column, axis=1)
y = data[target_column]

# Encode categorical features (important)
X = pd.get_dummies(X, drop_first=True)

# ---------------------------------------------------
# Train-Test Split
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# Handle Class Imbalance
# ---------------------------------------------------
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print("Scale Pos Weight:", scale_pos_weight)

# ---------------------------------------------------
# XGBoost Model
# ---------------------------------------------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# ---------------------------------------------------
# Predictions
# ---------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nAccuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc)

# ---------------------------------------------------
# Save Model
# ---------------------------------------------------
os.makedirs(STATIC_DIR, exist_ok=True)
model_path = os.path.join(STATIC_DIR, "fraud_model.pkl")
joblib.dump(model, model_path)

print("Model saved at:", model_path)

# ---------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(STATIC_DIR, "confusion_matrix.png"))
plt.close()

# ---------------------------------------------------
# ROC Curve
# ---------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title(f"ROC Curve (AUC={roc_auc:.4f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig(os.path.join(STATIC_DIR, "roc_curve.png"))
plt.close()

# ---------------------------------------------------
# Feature Importance
# ---------------------------------------------------
importances = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

importance_df.to_csv(os.path.join(STATIC_DIR, "feature_importance.csv"), index=False)

plt.figure(figsize=(8, 6))
plt.barh(importance_df["Feature"][:15], importance_df["Importance"][:15])
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importance")
plt.savefig(os.path.join(STATIC_DIR, "feature_importance.png"))
plt.close()

# ---------------------------------------------------
# Fraud Distribution Pie Chart
# ---------------------------------------------------
counts = y.value_counts()

plt.figure()
plt.pie(counts, labels=["Legitimate", "Fraud"], autopct="%1.1f%%")
plt.title("Original Fraud Distribution")
plt.savefig(os.path.join(STATIC_DIR, "pie_chart.png"))
plt.close()

print("\nTraining Complete ✅")
print("All outputs saved in:", STATIC_DIR)