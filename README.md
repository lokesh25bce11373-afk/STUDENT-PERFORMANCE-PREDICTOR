# STUDENT-PERFORMANCE-PREDICTOR
A Student Performance Predictor is an AI-driven tool that analyzes academic data like grades, attendance, and engagement to forecast future learning outcomes. It empowers educators to identify at-risk students early, personalize teaching strategies, and implement timely interventions, ultimately boosting overall student success and retention.
# Student Performance Predictor

A machine learning project that predicts whether a student will **Pass** or **Fail**
based on study hours, attendance, and previous exam score.

Built as a BYOP capstone project for an AI/ML course.

---

## What This Project Does

1. Creates a dataset of 300 students automatically
2. Generates 3 EDA charts (pass/fail count, study hours, correlation)
3. Trains and compares 3 ML models (Logistic Regression, Decision Tree, Random Forest)
4. Evaluates the best model and saves a confusion matrix chart
5. Saves the trained model to disk
6. Lets you predict Pass/Fail for any new student interactively

---

## How to Run

**Step 1 — Install libraries**
```
pip install -r requirements.txt
```

**Step 2 — Run the project**
```
python main.py
```

That's it. Everything runs in one command.

---

## Files

```
main.py            ← entire project (one file)
requirements.txt   ← libraries needed
students.csv       ← dataset (auto-created on run)
charts/            ← saved charts (auto-created on run)
model/             ← saved model files (auto-created on run)
```

---

## Example Output

```
[Step 4] Training and comparing models...

   Model                   Accuracy   CV Score
   --------------------------------------------
   Logistic Regression       86.67%     85.21%
   Decision Tree             88.33%     87.10%
   Random Forest             91.67%     90.43%

   Best model: Random Forest

   Accuracy : 91.67%
```

---

## Interactive Predictor

After training, you can enter any student's details:

```
Study hours per day (1–9): 6
Attendance % (40–100)   : 85
Previous score (0–100)  : 72

Result     : ✅ PASS
Confidence : Pass = 88.0%  |  Fail = 12.0%
```

---

## Technologies Used

- Python 3
- pandas, numpy — data handling
- scikit-learn — ML models
- matplotlib, seaborn — charts
