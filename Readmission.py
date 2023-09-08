import streamlit as st
import numpy as np
import pickle

# Initialize session state for reset functionality
if 'reset' not in st.session_state:
    st.session_state.reset = False

# Load the trained model
try:
    with open('RandomForest_Undersampling_model_10_features.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Make sure the RandomForest_Undersampling_model_10_features.pkl file is in the same directory as this script.")

# Custom styles
st.markdown("""
    <style>
        .title-box {
            background-color: #900C3F;
            color: white;
            padding: 10px;
            border-radius: 5px;
        }
        .result-box {
            background-color: #C70039;
            color: white;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown("""
    <div class="title-box" style="display: inline-block;">
        <h1 style="color: white; display: inline;">🏥 Predicting Hospital Readmission For Diabetic Patients</h1>
    </div>
    """, unsafe_allow_html=True)
st.write("")

# Initialize a flag for valid input
valid_input = True

# Collect user input with manual validation
num_lab_procedures = st.number_input('Number of lab procedures:', value=1, help="Range: 1-132")
if not(1 <= num_lab_procedures <= 132):
    st.warning('Number of lab procedures should be between 1 and 132.')
    valid_input = False

num_medications = st.number_input('Number of medications:', value=1, help="Range: 1-81")
if not(1 <= num_medications <= 81):
    st.warning('Number of medications should be between 1 and 81.')
    valid_input = False

time_in_hospital = st.number_input('Time in hospital:', value=1, help="Range: 1-14")
if not(1 <= time_in_hospital <= 14):
    st.warning('Time in hospital should be between 1 and 14.')
    valid_input = False

age = st.number_input('Age:', value=15, help="Range: 15-95")
if not(15 <= age <= 95):
    st.warning('Age should be between 15 and 95.')
    valid_input = False

num_procedures = st.number_input('Number of procedures:', value=0, help="Range: 0-6")
if not(0 <= num_procedures <= 6):
    st.warning('Number of procedures should be between 0 and 6.')
    valid_input = False

number_diagnoses = st.number_input('Number of diagnoses:', value=1, help="Range: 1-16")
if not(1 <= number_diagnoses <= 16):
    st.warning('Number of diagnoses should be between 1 and 16.')
    valid_input = False

num_med = st.selectbox('Num Med:', ['Yes', 'No'])
if num_med not in ['Yes', 'No']:
    st.warning('Invalid value for Num Med.')
    valid_input = False

discharge_disposition_id = st.selectbox('Discharge disposition ID:', ['Home care', 'Transfer', 'Outpatients', 'Expired Home/Medical', 'Undefined'])
if discharge_disposition_id not in ['Home care', 'Transfer', 'Outpatients', 'Expired Home/Medical', 'Undefined']:
    st.warning('Invalid value for Discharge Disposition ID.')
    valid_input = False

number_inpatient_log1p = st.number_input('Number of inpatient events:', value=0, help="Range: 0-21")
if not(0 <= number_inpatient_log1p <= 21):
    st.warning('Number of inpatient events should be between 0 and 21.')
    valid_input = False

admission_type_id = st.selectbox('Admission Type ID:', ['Emergency', 'Undefined'])
if admission_type_id not in ['Emergency', 'Undefined']:
    st.warning('Invalid value for Admission Type ID.')
    valid_input = False

# Check if inputs are valid
if valid_input:
    if st.button("Submit", key='submit'):
        # Prepare data
        user_data = np.array([
            num_lab_procedures,
            num_medications,
            time_in_hospital,
            age,
            num_procedures,
            number_diagnoses,
            1 if num_med == 'Yes' else 0,
            1 if discharge_disposition_id == 'Home care' else 2 if discharge_disposition_id == 'Transfer' else 10 if discharge_disposition_id == 'Outpatients' else 11 if discharge_disposition_id == 'Expired Home/Medical' else 18,
            number_inpatient_log1p,
            1 if admission_type_id == 'Emergency' else 5
        ]).reshape(1, -1)

        # Make prediction
        prediction_proba = model.predict_proba(user_data)
        prob_of_readmission = prediction_proba[0][1]

        # Show prediction
        st.markdown(f"""
            <div class="result-box">
                <span>📊 The model predicts a {prob_of_readmission * 100:.2f}% probability of being readmitted.</span>
            </div>
            """, unsafe_allow_html=True)

        # Show reset button
        st.session_state.reset = True

# Insert space before the Reset button
if st.session_state.reset:
    st.write("")
    if st.button("Reset"):
        st.session_state.reset = False
        st.experimental_rerun()
