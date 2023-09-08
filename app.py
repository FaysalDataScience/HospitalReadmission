import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



df=pd.read_csv("diabetic_data.csv")
print(df.shape)

#removing Duplicates
# Drop rows with NaN values from df
df.dropna(inplace = True)
# Print the total number of rows in df
print('Total data = ', len(df))
# Import numpy library for using its unique function
import numpy as np
# Print the total number of unique 'patient_nbr' in df
print('Unique entries = ', len(np.unique(df['patient_nbr'])))
# Remove duplicates based on 'patient_nbr' and keep the first occurrence
df.drop_duplicates(['patient_nbr'], keep = 'first', inplace = True)
# Print the total number of rows in df after removing duplicates
print('Length after removing Duplicates:', len(df))


# Drop 'weight', 'payer_code' columns beacuse too many missing value
df = df.drop(['weight', 'payer_code'], axis=1)


# Select rows to drop based on conditions
drop_indices = set(
    df[(df['diag_1'] == '?') & (df['diag_2'] == '?') & (df['diag_3'] == '?')].index
)
drop_indices = drop_indices.union(
    set(df['diag_1'][df['diag_1'] == '?'].index)
)
drop_indices = drop_indices.union(
    set(df['diag_2'][df['diag_2'] == '?'].index)
)
drop_indices = drop_indices.union(
    set(df['diag_3'][df['diag_3'] == '?'].index)
)
drop_indices = drop_indices.union(
    set(df['race'][df['race'] == '?'].index)
)
drop_indices = drop_indices.union(
    set(df['gender'][df['gender'] == 'Unknown/Invalid'].index)
)

# Drop the selected rows
df = df.drop(drop_indices)


df = df.drop(['encounter_id', 'patient_nbr'], axis=1)
df=df.drop(["citoglipton","examide"],axis = 1)
df=df.drop(["glimepiride-pioglitazone","metformin-rosiglitazone"],axis = 1)
df = df.loc[~df.discharge_disposition_id.isin([11,13,14,19,20,21])]


# Print unique values in 'age' column
print(np.unique(df['age']))

# Define a dictionary for age range replacements
replaceDict = {
    '[0-10)' : 5,
    '[10-20)' : 15,
    '[20-30)' : 25,
    '[30-40)' : 35,
    '[40-50)' : 45,
    '[50-60)' : 55,
    '[60-70)' : 65,
    '[70-80)' : 75,
    '[80-90)' : 85,
    '[90-100)' : 95
}

# Replace age ranges with the mean value of each range using the replaceDict
df['age'] = df['age'].apply(lambda x : replaceDict[x])

# Print the first 5 rows of the 'age' column after the transformation
print(df['age'].head())

# Reclassify 'discharge_disposition_id'
df['discharge_disposition_id'] = df['discharge_disposition_id'].apply(lambda x : 1 if int(x) in [6, 8, 9, 13]
                                                                           else ( 2 if int(x) in [3, 4, 5, 14, 22, 23, 24]
                                                                           else ( 10 if int(x) in [12, 15, 16, 17]
                                                                           else ( 11 if int(x) in [19, 20, 21]
                                                                           else ( 18 if int(x) in [25, 26]
                                                                           else int(x) )))))

# Filter out rows with certain 'discharge_disposition_id' values
df = df[~df.discharge_disposition_id.isin([11,13,14,19,20,21])]


# Reclassify 'admission_type_id'
df['admission_type_id'] = df['admission_type_id'].apply(lambda x : 1 if int(x) in [2, 7]
                                                        else ( 5 if int(x) in [6, 8]
                                                        else int(x) ))


# admission_source_id'
df['admission_source_id'] = df['admission_source_id'].apply(lambda x : 1 if int(x) in [2, 3]
                                                            else ( 4 if int(x) in [5, 6, 10, 22, 25]
                                                            else ( 9 if int(x) in [15, 17, 20, 21]
                                                            else ( 11 if int(x) in [13, 14]
                                                            else int(x) ))))


high_frequency = ['InternalMedicine', 'Family/GeneralPractice', 'Cardiology', 'Surgery-General', 'Orthopedics',
                  'Orthopedics-Reconstructive', 'Emergency/Trauma', 'Urology','ObstetricsandGynecology','Psychiatry',
                  'Pulmonology','Nephrology','Radiologist']

low_frequency = ['Surgery-PlasticwithinHeadandNeck','Psychiatry-Addictive','Proctology','Dermatology','SportsMedicine',
                 'Speech','Perinatology','Neurophysiology','Resident','Pediatrics-Hematology-Oncology',
                 'Pediatrics-EmergencyMedicine','Dentistry','DCPTEAM','Psychiatry-Child/Adolescent',
                 'Pediatrics-Pulmonology','Surgery-Pediatric','AllergyandImmunology','Pediatrics-Neurology',
                 'Anesthesiology','Pathology','Cardiology-Pediatric','Endocrinology-Metabolism','PhysicianNotFound',
                 'Surgery-Colon&Rectal','OutreachServices','Surgery-Maxillofacial','Rheumatology',
                 'Anesthesiology-Pediatric','Obstetrics','Obsterics&Gynecology-GynecologicOnco']

pediatrics = ['Pediatrics','Pediatrics-CriticalCare','Pediatrics-EmergencyMedicine','Pediatrics-Endocrinology',
              'Pediatrics-Hematology-Oncology','Pediatrics-Neurology','Pediatrics-Pulmonology',
              'Anesthesiology-Pediatric','Cardiology-Pediatric','Surgery-Pediatric']

psychic = ['Psychiatry-Addictive', 'Psychology', 'Psychiatry','Psychiatry-Child/Adolescent',
           'PhysicalMedicineandRehabilitation','Osteopath']

neurology = ['Neurology', 'Surgery-Neuro','Pediatrics-Neurology','Neurophysiology']

surgery = ['Surgeon', 'Surgery-Cardiovascular', 'Surgery-Cardiovascular/Thoracic', 'Surgery-Colon&Rectal',
           'Surgery-General', 'Surgery-Maxillofacial', 'Surgery-Plastic', 'Surgery-PlasticwithinHeadandNeck',
           'Surgery-Thoracic','Surgery-Vascular', 'SurgicalSpecialty', 'Podiatry']

ungrouped = ['Endocrinology','Gastroenterology','Gynecology','Hematology','Hematology/Oncology','Hospitalist',
             'InfectiousDiseases','Oncology','Ophthalmology','Otolaryngology','Pulmonology','Radiology']

missing = ['?']

def categorize_specialty(val):
    if val in pediatrics:
        return 'pediatrics'
    elif val in psychic:
        return 'psychic'
    elif val in neurology:
        return 'neurology'
    elif val in surgery:
        return 'surgery'
    elif val in high_frequency:
        return 'high_freq'
    elif val in low_frequency:
        return 'low_freq'
    elif val in ungrouped:
        return 'ungrouped'
    elif val in missing:
        return 'missing'
    else:
        return val  # Keeps the original value if not found in any category

# Apply the function on the 'medical_specialty' column
df['medical_specialty'] = df['medical_specialty'].apply(categorize_specialty)


def fix_diag(row, icd9):
    """
    Function to categorize diagnosis codes using ICD9 standards.
    The input diagnosis codes are grouped into more generic categories.
    """
    code = row[icd9]

    if code[0] == "E" or code[0] == "V":
        return "Other"
    else:
        num = float(code)

        if 390 <= num <= 459 or num == 785:
            return "circulatory"
        elif 520 <= num <= 579 or num == 787:
            return "digestive"
        elif 580 <= num <= 629 or num == 788:
            return "genitourinary"
        elif np.trunc(num) == 250:
            return "diabetes"
        elif 800 <= num <= 999:
            return "injury"
        elif 710 <= num <= 739:
            return "musculoskeletal"
        elif 140 <= num <= 239:
            return "neoplasms"
        elif 460 <= num <= 519 or num == 786:
            return "respiratory"
        else:
            return "other"

# Apply the function to the columns 'diag_1', 'diag_2' and 'diag_3' of the DataFrame 'df'
df["diag1_norm"] = df.apply(fix_diag, axis=1, icd9="diag_1")
df["diag2_norm"] = df.apply(fix_diag, axis=1, icd9="diag_2")
df["diag3_norm"] = df.apply(fix_diag, axis=1, icd9="diag_3")

# Define the columns to be dropped
dropped_Cols=['diag_1', 'diag_2', 'diag_3']

# Drop the defined columns from the DataFrame 'df'
df.drop(columns=dropped_Cols, inplace=True)


#Create a new feature patient_service
df['patient_service'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']


# Given list of medication keys
keys = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'insulin', 'glyburide-metformin', 'tolazamide', 'metformin-pioglitazone',
         'glipizide-metformin','troglitazone', 'tolbutamide']

# Recoding medication use into a binary variable based on conditions and storing in new columns
for col in keys:
    col_name = str(col) + 'new'
    df[col_name] = df[col].apply(lambda x: 0 if (x == 'No' or x == 'Steady') else 1)

# Initializing 'med_change' column to 0
df['med_change'] = 0

# Summing up all the new columns to create the 'med_change' feature
for col in keys:
    col_name = str(col) + 'new'
    df['med_change'] = df['med_change'] + df[col_name]
    del df[col_name]  # Deleting the intermediate new columns

# Checking the status of the new feature 'med_change'
print(df['med_change'].value_counts())



# Given list of medication keys (assuming it's provided from the earlier part of our conversation)
keys = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'insulin', 'glyburide-metformin', 'tolazamide', 'metformin-pioglitazone',
          'glipizide-metformin',
        'troglitazone', 'tolbutamide']

# Convert medication indications to binary values (1 for used, 0 for not used)
for col in keys:
    df[col] = df[col].replace({'No': 0, 'Steady': 1, 'Up': 1, 'Down': 1})

# Calculate total number of medications used for each patient
df['num_med'] = df[keys].sum(axis=1)


# Define the list of numerical columns
num_col = ['age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
           'num_medications', 'number_outpatient', 'number_emergency',
           'number_inpatient', 'number_diagnoses', 'patient_service', 'med_change', 'num_med']

# Initialize a new DataFrame to store statistics
statdataframe = pd.DataFrame()

# Add the column names to the DataFrame
statdataframe['numeric_column'] = num_col

# Initialize lists to store the statistics
skew_before = []
skew_after = []
kurt_before = []
kurt_after = []
standard_deviation_before = []
standard_deviation_after = []
log_transform_needed = []
log_type = []

# For each column in the list of numerical columns
for i in num_col:
    # Compute skewness before transformation
    skewval = df[i].skew()
    skew_before.append(skewval)

    # Compute kurtosis before transformation
    kurtval = df[i].kurtosis()
    kurt_before.append(kurtval)

    # Compute standard deviation before transformation
    sdval = df[i].std()
    standard_deviation_before.append(sdval)

    # If skewness and kurtosis are high, transformation is needed
    if (abs(skewval) >2) & (abs(kurtval) >2):
        log_transform_needed.append('Yes')

        # If the proportion of 0 values is less than 2%, apply log transformation
        if len(df[df[i] == 0])/len(df) <=0.02:
            log_type.append('log')
            skewvalnew = np.log(pd.DataFrame(df[df[i] > 0])[i]).skew()
            skew_after.append(skewvalnew)

            kurtvalnew = np.log(pd.DataFrame(df[df[i] > 0])[i]).kurtosis()
            kurt_after.append(kurtvalnew)

            sdvalnew = np.log(pd.DataFrame(df[df[i] > 0])[i]).std()
            standard_deviation_after.append(sdvalnew)

        # If the proportion of 0 values is more than 2%, apply log1p transformation
        else:
            log_type.append('log1p')
            skewvalnew = np.log1p(pd.DataFrame(df[df[i] >= 0])[i]).skew()
            skew_after.append(skewvalnew)

            kurtvalnew = np.log1p(pd.DataFrame(df[df[i] >= 0])[i]).kurtosis()
            kurt_after.append(kurtvalnew)

            sdvalnew = np.log1p(pd.DataFrame(df[df[i] >= 0])[i]).std()
            standard_deviation_after.append(sdvalnew)

    # If skewness and kurtosis are not high, no transformation is needed
    else:
        log_type.append('NA')
        log_transform_needed.append('No')

        skew_after.append(skewval)
        kurt_after.append(kurtval)
        standard_deviation_after.append(sdval)

# Add all the computed statistics to the DataFrame
statdataframe['skew_before'] = skew_before
statdataframe['kurtosis_before'] = kurt_before
statdataframe['standard_deviation_before'] = standard_deviation_before
statdataframe['log_transform_needed'] = log_transform_needed
statdataframe['log_type'] = log_type
statdataframe['skew_after'] = skew_after
statdataframe['kurtosis_after'] = kurt_after
statdataframe['standard_deviation_after'] = standard_deviation_after

# Print the DataFrame
statdataframe


# If the log transform is needed according to our stats dataframe, apply the transformation
for i in range(len(statdataframe)):
    if statdataframe['log_transform_needed'][i] == 'Yes':
        # Get the column name
        colname = str(statdataframe['numeric_column'][i])

        # Apply the appropriate log transformation
        if statdataframe['log_type'][i] == 'log':
            df = df[df[colname] > 0]
            df[colname + "_log"] = np.log(df[colname])

        elif statdataframe['log_type'][i] == 'log1p':
            df = df[df[colname] >= 0]
            df[colname + "_log1p"] = np.log1p(df[colname])

# Drop some of the original columns that are not needed anymore
df = df.drop(['number_outpatient', 'number_inpatient', 'number_emergency', 'patient_service'], axis = 1)

print(df.shape)
df.head()

# Import the necessary library
import scipy as sp

# Select the numeric columns
num_cols = ['age', 'time_in_hospital', 'num_lab_procedures',
       'num_procedures', 'num_medications', 'number_diagnoses', 'med_change', 'num_med']

# Keep only the rows in the dataframe that have a z-score less than 3 (i.e., remove outliers that are 3 standard deviations away from the mean)
df = df[(np.abs(sp.stats.zscore(df[num_cols])) < 3).all(axis=1)]

# Print the updated shape and head of the dataframe
print(df.shape)
df.head()


# Convert 'readmitted' to binary
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)


import pandas as pd

# Assuming your dataframe is named df
df = df.drop(columns=['admission_source_id', 'change', 'diabetesMed', 'patient_service_log1p'])
# Without 'inplace=True', the 'drop' function would return a new data frame with the operation performed, leaving the original data frame unchanged.
df.drop('acetohexamide', axis=1, inplace=True)
print("rr", df.shape)
# Convert 'race' column into dummy/indicator variables
df = pd.get_dummies(df, columns = ["race"], prefix = "race", drop_first=True)
# Apply one-hot encoding to 'gender' column
df = pd.get_dummies(df, columns=['gender'], prefix = "race", drop_first=True)


def diagnose(df, diag):
    """
    Function to check if a particular diagnosis is present in any of the diag1_norm, diag2_norm, diag3_norm columns.
    If the diagnosis is present, it returns True, otherwise it returns False.
    """
    if (df["diag1_norm"] == diag) | (df["diag2_norm"] == diag) | (df["diag3_norm"] == diag):
        return True
    else:
        return False

# Check for the presence of certain diagnoses and create new columns for each diagnosis
for val in ['diabetes', 'other', 'circulatory', 'neoplasms', 'respiratory', 'injury', 'musculoskeletal', 'digestive', 'genitourinary']:
    name = val + "_diagnosis"
    df[name] = df.apply(diagnose, axis = 1, diag=val).astype(int)
# Define the columns to be dropped
dropped_Cols=['diag1_norm', 'diag2_norm', 'diag3_norm']

# Drop the defined columns from the DataFrame 'df'
df.drop(columns=dropped_Cols, inplace=True)


common_drugs = ['metformin', 'repaglinide', 'glimepiride', 'glipizide',
                'glyburide', 'pioglitazone', 'rosiglitazone', 'insulin']
rare_drugs = ["nateglinide", "chlorpropamide", "tolbutamide",
             "acarbose", "miglitol", "troglitazone", "tolazamide",
             "glyburide-metformin", "glipizide-metformin",
               "metformin-pioglitazone"]

# Combine common and rare drugs lists
drugs = common_drugs+ rare_drugs

# Apply binary  for each drug
for drug in drugs:
    name = "take_" + drug
    df[name] = df[drug].isin(["Down", "Steady", "Up"]).astype(int)

# Remove the previous drug columns
df = df.drop(drugs, axis=1)


df = pd.get_dummies(df, columns=['A1Cresult'], drop_first=False)
# Drop 'A1Cresult' and 'A1C_None' columns from DataFrame 'df'
df = df.drop(["A1Cresult_None"], axis = 1)



df = pd.get_dummies(df, columns=['max_glu_serum'], drop_first=False)
# Drop 'A1Cresult' and 'A1C_None' columns from DataFrame 'df'
df = df.drop(["max_glu_serum_None"], axis = 1)

df = pd.get_dummies(df, columns=['medical_specialty'], prefix=['med_spec'], drop_first=True)

df = df.drop(["med_spec_missing"], axis = 1)

print("result", df.shape)

target_counts = df['readmitted'].value_counts()
print(target_counts)



import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import pickle

# Assuming 'df' is your DataFrame and 'readmitted' is the target variable
# df = YOUR_DATAFRAME_HERE

# Use only the 10 specified features
selected_features = [
    'num_lab_procedures',
    'num_medications',
    'time_in_hospital',
    'age',
    'num_procedures',
    'number_diagnoses',
    'num_med',
    'discharge_disposition_id',
    'number_inpatient_log1p',
    'admission_type_id'
]
X = df[selected_features]
y = df['readmitted']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply Random Under-Sampling on the scaled data
rus = RandomUnderSampler()
X_resampled_scaled, y_resampled = rus.fit_resample(X_train_scaled, y_train)

# Train Random Forest Classifier
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
clf.fit(X_resampled_scaled, y_resampled)

# Save the trained model
with open('RandomForest_Undersampling_model_10_features.pkl', 'wb') as file:
    pickle.dump(clf.best_estimator_, file)

# Optionally, you can also save the scaler
with open('feature_scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
