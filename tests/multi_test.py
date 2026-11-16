import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. LOADING THE MODEL AND DATA
# ============================================================

df = pd.read_csv('../data/df_new.csv')

artifact = joblib.load('../models/model.pkl')

model = artifact['model']
threshold = artifact['threshold']
features = artifact['features']

# ============================================================
# 2. RECONSTRUCTION OF TRAIN / VALIDATION / TEST SPLIT
# ============================================================

from sklearn.model_selection import train_test_split

target = df['No-show'].astype(int)
features_df = df.drop('No-show', axis=1)

X_train, X_temp, y_train, y_temp = train_test_split(
    features_df,
    target,
    test_size=0.30,
    random_state=0,
    stratify=target
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=0,
    stratify=y_temp
)

print("Test set:", len(X_test), "records")

# ============================================================
# 3. SELECTION OF 100 RANDOM RECORDS
# ============================================================

sample = X_test.sample(
    n=100,
    random_state=None
)

correct_predictions = 0
total = 0

results = []

# ============================================================
# 4. PREDICTIONS
# ============================================================

for index, record in sample.iterrows():

    actual_result = int(y_test.loc[index])

    input_data = pd.DataFrame(
        [record.to_dict()],
        columns=features
    )

    probability = model.predict_proba(input_data)[0][1]

    prediction = int(probability >= threshold)

    is_correct = prediction == actual_result

    total += 1

    if is_correct:
        correct_predictions += 1

    results.append({
        'Record': index,
        'Probability': probability,
        'Prediction': prediction,
        'Actual_Result': actual_result
    })

    print(f"\nRecord: {index}")
    print(f"Probability of no-show: {probability:.2%}")
    print(f"Prediction: {prediction}")
    print(f"Actual result: {actual_result}")
    print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")

# ============================================================
# 5. SAMPLE METRICS
# ============================================================

df_results = pd.DataFrame(results)

y_actual = df_results['Actual_Result']
y_predicted = df_results['Prediction']

errors = total - correct_predictions

print("\n" + "=" * 40)
print("TEST SUMMARY")
print("=" * 40)

print(f"Total records: {total}")
print(f"Correct: {correct_predictions}")
print(f"Errors: {errors}")
print(f"Accuracy rate: {correct_predictions / total:.2%}")

print("\nMETRICS:")
print(f"Accuracy: {accuracy_score(y_actual, y_predicted):.4f}")
print(f"Precision: {precision_score(y_actual, y_predicted, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_actual, y_predicted, zero_division=0):.4f}")
print(f"F1-score: {f1_score(y_actual, y_predicted, zero_division=0):.4f}")

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_actual, y_predicted))

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    y_actual,
    y_predicted,
    zero_division=0
))

# ============================================================
# 6. SAVING THE RESULTS
# ============================================================

df_results.to_csv(
    '../data/test_results_100.csv',
    index=False
)

print("\nResults saved to test_results_100.csv")