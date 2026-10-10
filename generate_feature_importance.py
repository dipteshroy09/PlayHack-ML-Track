"""Generate the Top 10 Feature Importance chart from the saved Task A classifier."""

from catboost import CatBoostClassifier
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

model = CatBoostClassifier()
model.load_model("submission_artifacts/final_task_a_classifier.cbm")

train = pd.read_csv("train_features.csv")
predictors = [
    c for c in train.columns
    if c not in ["athlete_id", "injured_in_risk_window",
                 "onset_day_offset", "recovery_duration"]
]

importances = model.get_feature_importance()
feat_imp = pd.Series(importances, index=predictors).sort_values(ascending=True)
top10 = feat_imp.tail(10)

print("Top 10 Features:")
for name, val in top10.items():
    print(f"  {val:6.2f}  {name}")

plt.figure(figsize=(10, 6))
top10.plot.barh(color="#3B82F6")
plt.title("Top 10 Feature Importances (Task A Classifier)",
          fontsize=14, fontweight="bold")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("\nSaved feature_importance.png")
