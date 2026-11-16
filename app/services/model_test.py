import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. DATA LOADING
# ============================================================

df = pd.read_csv('../../data/df_new.csv', delimiter=',')

target = df['No-show'].astype(int)
features = df.drop('No-show', axis=1)

# ============================================================
# 2. TRAIN, VALIDATION, AND TEST SPLIT
# ============================================================

# 70% train and 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    features,
    target,
    test_size=0.30,
    random_state=0,
    stratify=target
)

# 15% validation and 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=0,
    stratify=y_temp
)

print("Data split:")
print(f"Train: {len(X_train)} records")
print(f"Validation: {len(X_val)} records")
print(f"Test: {len(X_test)} records")


# ============================================================
# 3. LIGHTGBM CREATION AND TRAINING
# ============================================================

print("\nTraining Optimized LightGBM...")

model_lgbm = lgb.LGBMClassifier(
    num_leaves=15,
    max_depth=4,
    learning_rate=0.05,
    max_bin=100,
    n_estimators=200,
    class_weight='balanced',
    random_state=0,
    verbose=-1
)

model_lgbm.fit(X_train, y_train)


# ============================================================
# 4. THRESHOLD TESTING ON VALIDATION SET
# ============================================================

probabilities_val = model_lgbm.predict_proba(
    X_val
)[:, 1]

thresholds = [0.30, 0.40, 0.50, 0.60, 0.65, 0.70]

results = []

print("\n" + "=" * 65)
print("THRESHOLD COMPARISON - VALIDATION")
print("=" * 65)

for threshold in thresholds:

    predictions = (
        probabilities_val >= threshold
    ).astype(int)

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    results.append({
        'Threshold': threshold,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    })

    print(f"\nThreshold: {threshold:.2f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")


# ============================================================
# 5. BEST THRESHOLD SELECTION
# ============================================================

df_results = pd.DataFrame(results)

# Initial criterion: highest F1-score on validation
best_index = df_results['F1-score'].idxmax()

best_threshold = df_results.loc[
    best_index, 'Threshold'
]

print("\n" + "=" * 65)
print("BEST THRESHOLD ON VALIDATION")
print("=" * 65)

print(df_results.to_string(index=False))
print(f"\nChosen threshold: {best_threshold:.2f}")

# Save comparison for reference
df_results.to_csv(
    '../../data/threshold_comparison_validation.csv',
    index=False
)


# ============================================================
# 6. FINAL EVALUATION ON TEST SET
# ============================================================

# The test set is only used now, after choosing the threshold
probabilities_test = model_lgbm.predict_proba(
    X_test
)[:, 1]

predictions_test = (
    probabilities_test >= best_threshold
).astype(int)

print("\n" + "=" * 65)
print("FINAL RESULT - TEST SET")
print("=" * 65)

print(f"Used threshold: {best_threshold:.2f}")

print(
    "\nAccuracy: %.4f" %
    accuracy_score(y_test, predictions_test)
)

print(
    "Precision: %.4f" %
    precision_score(y_test, predictions_test, zero_division=0)
)

print(
    "Recall: %.4f" %
    recall_score(y_test, predictions_test, zero_division=0)
)

print(
    "F1-score: %.4f" %
    f1_score(y_test, predictions_test, zero_division=0)
)

print(
    "Average Precision (AP): %.4f" %
    average_precision_score(y_test, probabilities_test)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions_test))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions_test,
        zero_division=0
    )
)


# ============================================================
# 7. SAVING MODEL AND SETTINGS
# ============================================================

model_artifact = {
    'model': model_lgbm,
    'threshold': float(best_threshold),
    'features': features.columns.tolist()
}

joblib.dump(
    model_artifact,
    '../../models/model.pkl'
)

print("\nModel successfully saved!")
print("File: model.pkl")
print(f"Saved threshold: {best_threshold:.2f}")