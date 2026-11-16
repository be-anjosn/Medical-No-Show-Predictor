import pandas as pd

# Original data
df = pd.read_csv('../../data/data_original_noshow.csv', encoding='iso-8859-1', sep=',')

df_novo = df.copy()

# Converting Datetimes
df_novo['ScheduledDay'] = pd.to_datetime(df_novo['ScheduledDay'])
df_novo['AppointmentDay'] = pd.to_datetime(df_novo['AppointmentDay'])

# Creating Waiting days
df_novo['Waiting_Days'] = (df_novo['AppointmentDay'] - df_novo['ScheduledDay'].dt.normalize()).dt.days

# Extract datetimes 
df_novo['Scheduled_Month'] = df_novo['ScheduledDay'].dt.month
df_novo['Scheduled_Day_of_the_Week'] = df_novo['ScheduledDay'].dt.day_of_week

df_novo['Appointment_Month'] = df_novo['AppointmentDay'].dt.month
df_novo['Appointment_Day_of_the_Week'] = df_novo['AppointmentDay'].dt.day_of_week


# Normalizing Age field
bins_idade = [-1, 12, 19, 59, 150]
labels_idade = [0, 1, 2, 3] # 0: Child, 1: Teen, 2: Adult, 3: Elderly
df_novo['Age_Group'] = pd.cut(df_novo['Age'], bins=bins_idade, labels=labels_idade, include_lowest=True).fillna(2).astype(int)

# Categorizing Wait days
bins_espera = [-1, 0, 3, 7, 30, 3000]
labels_espera = [0, 1, 2, 3, 4] # 0: Same day, 1: Up to 3 days, 2: Up to 1 week, 3: Up to 1 month, 4: More than 1 month
df_novo['Waiting_Category'] = pd.cut(df_novo['Waiting_Days'], bins=bins_espera, labels=labels_espera, include_lowest=True).fillna(0).astype(int)

# -----------------------------------------------------

# Treating errors and consistency
df_novo = df_novo[df_novo['Age'] >= 0]
df_novo = df_novo[df_novo['Waiting_Days'] >= 0]

# Removing Columns that do not enter on predictive calculation
df_novo = df_novo.drop(['ScheduledDay', 'AppointmentDay'], axis=1)
df_novo = df_novo.drop(['PatientId', 'AppointmentID', 'SMS_received'], axis=1)

# Saving a dataframe specially to test Catboost
df_novo.to_csv('../../data/df_catboost.csv', index=False)

# Transforming Categoric variables
df_novo['Gender'] = df_novo['Gender'].map({'F': 0, 'M': 1})
df_novo['No-show'] = df_novo['No-show'].map({'No': 0, 'Yes': 1})
df_novo = pd.get_dummies(df_novo, columns=['Neighbourhood'], dtype=int)

# Saving final versions
df_novo.to_csv('../../data/df_new.csv', index=False)