import streamlit as st
import pickle
import sqlite3
import numpy as np

# Load the trained LightGBM model
model_path = 'model/lgbmodel.pkl'  # Adjust the path accordingly
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define the database path
db_path = 'database.db'

# Initialize the database if necessary
conn = sqlite3.connect(db_path)
conn.close()  # Just to ensure the database is created

# Streamlit app
st.title('Health Prediction App')

# Input form using Streamlit widgets
st.write('Enter Patient Information:')
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=0, max_value=120)
hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'Children', 'Never_worked'])
residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
avg_glucose_level = st.number_input('Avg Glucose Level', min_value=0.0, max_value=300.0)
bmi = st.number_input('BMI', min_value=10.0, max_value=60.0)
smoking_status = st.selectbox('Smoking Status', ['Unknown', 'Never smoked', 'Formerly smoked', 'Smokes'])

submit_button = st.button('Predict')

if submit_button:
    # Convert categorical inputs to numerical for prediction
    gender = 1 if gender == 'Male' else 0
    hypertension = 1 if hypertension == 'Yes' else 0
    heart_disease = 1 if heart_disease == 'Yes' else 0
    work_type_mapping = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'Children': 3, 'Never_worked': 4}
    work_type = work_type_mapping.get(work_type, 0)
    residence_type = 1 if residence_type == 'Urban' else 0
    smoking_status_mapping = {'Unknown': 0, 'Never smoked': 1, 'Formerly smoked': 2, 'Smokes': 3}
    smoking_status = smoking_status_mapping.get(smoking_status, 0)

    # Prepare input data for prediction
    input_data = np.array([gender, age, hypertension, heart_disease, work_type, residence_type,
                           avg_glucose_level, bmi, smoking_status]).reshape(1, -1)

    # Predict using the pre-trained LightGBM model
    prediction = model.predict_proba(input_data)[:, 1][0]  # Probability of positive class

    # Determine the result based on the prediction probability
    if prediction >= 0.70:
        result = "Advice for check up"
    elif 0.40 <= prediction <= 0.69:
        result = "Be wary"
    else:
        result = "No"

    # Display prediction result
    st.write(f'Prediction Probability: {prediction:.2f}')
    st.write(f'Result: {result}')

    # Store the prediction in the database
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''INSERT INTO predictions (gender, age, hypertension, heart_disease, work_type, Residence_type,
                 avg_glucose_level, bmi, smoking_status, prediction, result)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (gender, age, hypertension, heart_disease, work_type, residence_type,
               avg_glucose_level, bmi, smoking_status, prediction, result))
    conn.commit()
    conn.close()
