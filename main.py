# ============================================================
#   Student Performance Predictor
#   BYOP Project | AI/ML Course | Intermediate Level
#   Single File — run with:  python main.py
# ============================================================

# ── Imports ─────────────────────────────────────────────────
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.tree             import DecisionTreeClassifier
from sklearn.metrics          import (accuracy_score, classification_report,
                                       confusion_matrix, ConfusionMatrixDisplay)

os.makedirs("charts", exist_ok=True)
os.makedirs("model",  exist_ok=True)

print("=" * 55)
print("   STUDENT PERFORMANCE PREDICTOR")
print("=" * 55)


# ============================================================
# STEP 1 — CREATE DATASET
# ============================================================
print("\n[Step 1] Creating dataset...")

np.random.seed(42)
n = 300

study_hours = np.round(np.random.uniform(1, 9, n), 1)
attendance  = np.round(np.random.uniform(40, 100, n), 1)
prev_score  = np.round(np.random.uniform(20, 100, n), 1)

# Pass if weighted score crosses threshold (with small noise)
score  = 0.4 * (study_hours / 9) + 0.3 * (attendance / 100) + 0.3 * (prev_score / 100)
noise  = np.random.normal(0, 0.05, n)
result = np.where(score + noise >= 0.47, "Pass", "Fail")

df = pd.DataFrame({
    "study_hours" : study_hours,
    "attendance"  : attendance,
    "prev_score"  : prev_score,
    "result"      : result
})

df.to_csv("students.csv", index=False)
print(f"   Dataset created: {n} students")
print(f"   Pass: {sum(result == 'Pass')}  |  Fail: {sum(result == 'Fail')}")
print(df.head())


# ============================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS (3 charts)
# ============================================================
print("\n[Step 2] Generating EDA charts...")

# Chart 1 — Pass / Fail count
fig, ax = plt.subplots(figsize=(5, 4))
counts  = df["result"].value_counts()
colors  = ["#2ecc71", "#e74c3c"]
bars    = ax.bar(counts.index, counts.values, color=colors, width=0.4)
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 3,
            str(v), ha="center", fontweight="bold")
ax.set_title("Pass vs Fail Count")
ax.set_ylabel("Number of Students")
plt.tight_layout()
plt.savefig("charts/01_pass_fail.png", dpi=150)
plt.close()

# Chart 2 — Study hours vs Result
fig, ax = plt.subplots(figsize=(6, 4))
for label, color in [("Pass", "#2ecc71"), ("Fail", "#e74c3c")]:
    ax.hist(df[df["result"] == label]["study_hours"],
            bins=15, alpha=0.6, color=color, label=label)
ax.set_title("Study Hours by Result")
ax.set_xlabel("Study Hours / Day")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig("charts/02_study_hours.png", dpi=150)
plt.close()

# Chart 3 — Correlation heatmap
fig, ax = plt.subplots(figsize=(5, 4))
num_df  = df.copy()
num_df["result_num"] = (num_df["result"] == "Pass").astype(int)
sns.heatmap(num_df[["study_hours", "attendance",
                     "prev_score", "result_num"]].corr(),
            annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("charts/03_correlation.png", dpi=150)
plt.close()

print("   3 charts saved in ./charts/")


# ============================================================
# STEP 3 — PREPROCESSING
# ============================================================
print("\n[Step 3] Preprocessing...")

X = df[["study_hours", "attendance", "prev_score"]].values

le = LabelEncoder()
y  = le.fit_transform(df["result"])   # Fail=0, Pass=1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"   Train samples: {len(X_train)}")
print(f"   Test  samples: {len(X_test)}")


# ============================================================
# STEP 4 — TRAIN & COMPARE 3 MODELS
# ============================================================
print("\n[Step 4] Training and comparing models...")

models = {
    "Logistic Regression" : LogisticRegression(max_iter=500),
    "Decision Tree"       : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
print(f"\n   {'Model':<22}  {'Accuracy':>9}  {'CV Score':>9}")
print("   " + "-" * 44)

for name, model in models.items():
    model.fit(X_train, y_train)
    acc    = accuracy_score(y_test, model.predict(X_test))
    cv     = cross_val_score(model, X_train, y_train, cv=5).mean()
    results[name] = {"model": model, "acc": acc, "cv": cv}
    print(f"   {name:<22}  {acc*100:>8.2f}%  {cv*100:>8.2f}%")

# Pick best model by test accuracy
best_name  = max(results, key=lambda k: results[k]["acc"])
best_model = results[best_name]["model"]
print(f"\n   Best model: {best_name}")


# ============================================================
# STEP 5 — EVALUATE BEST MODEL
# ============================================================
print("\n[Step 5] Evaluating best model...")

y_pred = best_model.predict(X_test)
print(f"\n   Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\n   Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=le.classes_))

# Confusion matrix chart
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                       display_labels=le.classes_).plot(ax=ax, colorbar=False)
ax.set_title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig("charts/04_confusion_matrix.png", dpi=150)
plt.close()

# Feature importance chart (Random Forest only)
if best_name == "Random Forest":
    imp   = best_model.feature_importances_
    feats = ["Study Hours", "Attendance", "Prev Score"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(feats, imp * 100, color=["#3498db", "#9b59b6", "#e67e22"])
    ax.set_xlabel("Importance (%)")
    ax.set_title("Feature Importance")
    for i, v in enumerate(imp * 100):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center")
    plt.tight_layout()
    plt.savefig("charts/05_feature_importance.png", dpi=150)
    plt.close()
    print("   Feature importance chart saved.")

print("   Confusion matrix chart saved.")


# ============================================================
# STEP 6 — SAVE MODEL
# ============================================================
print("\n[Step 6] Saving model...")

with open("model/model.pkl",  "wb") as f:
    pickle.dump(best_model, f)
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("model/encoder.pkl","wb") as f:
    pickle.dump(le, f)

print("   model/model.pkl   saved")
print("   model/scaler.pkl  saved")
print("   model/encoder.pkl saved")


# ============================================================
# STEP 7 — PREDICT FOR NEW STUDENTS
# ============================================================
print("\n[Step 7] Predict for new students...")

def predict_student(study_hours, attendance, prev_score):
    """Predict Pass or Fail for one student."""
    features = np.array([[study_hours, attendance, prev_score]])
    features = scaler.transform(features)
    pred     = best_model.predict(features)[0]
    proba    = best_model.predict_proba(features)[0]
    label    = le.inverse_transform([pred])[0]
    conf     = dict(zip(le.classes_, (proba * 100).round(1)))
    return label, conf

# Example predictions
examples = [
    ("Student A", 7.0, 90.0, 80.0),
    ("Student B", 2.0, 55.0, 35.0),
    ("Student C", 4.5, 70.0, 60.0),
]

print(f"\n   {'Name':<12} {'Study':>6} {'Attend':>7} {'Score':>6}  Result")
print("   " + "-" * 48)
for name, sh, att, ps in examples:
    label, conf = predict_student(sh, att, ps)
    icon = "PASS" if label == "Pass" else "FAIL"
    print(f"   {name:<12} {sh:>6}  {att:>6}%  {ps:>5}   {icon}  "
          f"(Pass {conf.get('Pass',0)}%)")


# ============================================================
# INTERACTIVE PREDICTOR
# ============================================================
print("\n" + "=" * 55)
print("   INTERACTIVE PREDICTOR")
print("   Type student details to get a prediction.")
print("   Press Ctrl+C or type 'quit' to exit.")
print("=" * 55)

while True:
    try:
        print()
        val = input("   Study hours per day (1–9): ").strip()
        if val.lower() == "quit":
            break
        sh  = float(val)

        val = input("   Attendance % (40–100)   : ").strip()
        if val.lower() == "quit":
            break
        att = float(val)

        val = input("   Previous score (0–100)  : ").strip()
        if val.lower() == "quit":
            break
        ps  = float(val)

        label, conf = predict_student(sh, att, ps)
        icon = "✅ PASS" if label == "Pass" else "❌ FAIL"

        print(f"\n   Result     : {icon}")
        print(f"   Confidence : Pass = {conf.get('Pass',0)}%  |"
              f"  Fail = {conf.get('Fail',0)}%")

    except ValueError:
        print("   Please enter a valid number.")
    except KeyboardInterrupt:
        print("\n   Bye!")
        break

print("\n" + "=" * 55)
print("   Done! Check ./charts/ for all saved plots.")
print("=" * 55)
