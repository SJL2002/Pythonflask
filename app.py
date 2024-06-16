import streamlit as st
import pickle
import numpy as np
import sqlite3

# Load the trained LightGBM model
model_path = 'model/lgbmodel.pkl'  # Update with your actual path
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define the database path
db_path = 'database.db'  # Update with your actual path

def init_db():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY, gender INTEGER, age REAL, hypertension INTEGER, heart_disease INTEGER,
                 work_type INTEGER, Residence_type INTEGER, avg_glucose_level REAL, bmi REAL, smoking_status INTEGER,
                 prediction REAL, result TEXT)''')
    conn.commit()
    conn.close()

def alter_table():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
        c.execute('ALTER TABLE predictions ADD COLUMN result TEXT')
    except sqlite3.OperationalError as e:
        print(f"OperationalError: {e}")
    conn.commit()
    conn.close()

# Initialize the database and alter table if necessary
init_db()
alter_table()

def predict_stroke_probability(input_data):
    # Predict using the pre-trained LightGBM model
    prediction = model.predict_proba(input_data)[:, 1][0]  # Get the probability for the positive class

    # Determine the result based on the prediction probability
    if prediction >= 0.70:
        result = "Advice for check up"
    elif 0.40 <= prediction <= 0.69:
        result = "Be wary"
    else:
        result = "No"

    return prediction, result

# Main function to run the Streamlit app
def main():
    st.title("Brain Stroke Probability Prediction")

    # Sidebar inputs
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.slider("Age", 0, 100, 50)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.slider("Average Glucose Level", 50.0, 300.0, 100.0)
    bmi = st.slider("BMI", 10.0, 60.0, 25.0)
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

    # Predict button
    if st.button("Predict"):
        # Convert input data to the appropriate format
        input_data = np.array([gender, age, hypertension, heart_disease, work_type, Residence_type,
                               avg_glucose_level, bmi, smoking_status]).reshape(1, -1)

        # Predict stroke probability
        prediction, result = predict_stroke_probability(input_data)

        # Store the prediction in the database
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO predictions (gender, age, hypertension, heart_disease, work_type, Residence_type,
                     avg_glucose_level, bmi, smoking_status, prediction, result) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (*input_data[0], prediction, result))
        conn.commit()
        conn.close()

        # Display prediction result
        st.success(f"The probability of having a stroke is: {prediction:.2f}")
        st.info(f"Prediction saved to database with result: {result}")

# Run the main function
if __name__ == '__main__':
    main()
