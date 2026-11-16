import pandas as pd
import joblib

artifact = joblib.load('../models/model.pkl')

model = artifact['model']
threshold = artifact['threshold']
features = artifact['features']

# Dummy variable values
data = {
    'Gender': 0,
    'Age': 35,
    'Scholarship': 0,
    'Hipertension': 0,
    'Diabetes': 0,
    'Alcoholism': 0,
    'Handcap': 0,
    'Waiting_Days': 15,
    'Scheduled_Month': 5,
    'Scheduled_Day_of_the_Week': 2,
    'Appointment_Month': 5,
    'Appointment_Day_of_the_Week': 4,
    'Age_Group': 2,
    'Waiting_Category': 1
}

# Create all columns expected by the model
input_data = pd.DataFrame(0, index=[0], columns=features)

for column, value in data.items():
    if column in input_data.columns:
        input_data[column] = value

# Prediction
probability = model.predict_proba(input_data)[0][1]
prediction = int(probability >= threshold)

print(f'Estimated probability of no-show: {probability:.2%}')
print(f'Used threshold: {threshold:.2f}')
print(
    'Prediction:',
    'Possible no-show' if prediction == 1 else 'Possible attendance'
)