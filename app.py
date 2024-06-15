from flask import Flask, request, render_template
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np

app = Flask(__name__)

# Load the trained LightGBM model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'lgbmodel.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define the database path
db_path = os.path.join(os.path.dirname(__file__), 'database.db')

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

# WTForms for input validation
class PredictionForm(Form):
    gender = TextAreaField('Gender', [validators.DataRequired()])
    age = TextAreaField('Age', [validators.DataRequired()])
    hypertension = TextAreaField('Hypertension', [validators.DataRequired()])
    heart_disease = TextAreaField('Heart Disease', [validators.DataRequired()])
    work_type = TextAreaField('Work Type', [validators.DataRequired()])
    Residence_type = TextAreaField('Residence Type', [validators.DataRequired()])
    avg_glucose_level = TextAreaField('Avg Glucose Level', [validators.DataRequired()])
    bmi = TextAreaField('BMI', [validators.DataRequired()])
    smoking_status = TextAreaField('Smoking Status', [validators.DataRequired()])

@app.route('/')
def index():
    form = PredictionForm(request.form)
    return render_template('index.html', form=form)

@app.route('/predict', methods=['POST'])
def predict():
    form = PredictionForm(request.form)
    if request.method == 'POST' and form.validate():
        data = request.form.to_dict()
        features = ['gender', 'age', 'hypertension', 'heart_disease', 'work_type', 'Residence_type',
                    'avg_glucose_level', 'bmi', 'smoking_status']

        # Convert input data to the appropriate format
        input_data = [float(data[feature]) for feature in features]
        input_data = np.array(input_data).reshape(1, -1)

        # Predict using the pre-trained LightGBM model
        prediction = model.predict_proba(input_data)[:, 1][0]  # Get the probability for the positive class

        # Determine the result based on the prediction probability
        if prediction >= 0.70:
            result = "Advice for check up"
        elif 0.40 <= prediction <= 0.69:
            result = "Be wary"
        else:
            result = "No"

        # Store the prediction in the database
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO predictions (gender, age, hypertension, heart_disease, work_type, Residence_type,
                     avg_glucose_level, bmi, smoking_status, prediction, result) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (*input_data[0], prediction, result))
        conn.commit()
        conn.close()

        return render_template('results.html', prediction=prediction, result=result)

    return render_template('index.html', form=form)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.form['feedback']
    # Store feedback in a database or handle it accordingly
    print(f"Feedback received: {feedback}")
    return render_template('thank_you.html')

if __name__ == '__main__':
    app.run(debug=True)
