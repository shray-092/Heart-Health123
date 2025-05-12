from flask import Flask, request, render_template
import pandas as pd
import joblib
import random
import os
import numpy as np

# app = Flask(__name__, static_folder='static')

# Load the trained model and data
heart_disease_model = joblib.load('heart_disease_model.sav')
recommendations_data = pd.read_csv('recommendations.csv')
symptoms_data = pd.read_csv('symptoms.csv')

# Exercise options
exercises = [
    "Brisk walking", "Cycling", "Swimming", "Running", "Dancing", 
    "Yoga", "Tai chi", "Pilates", "Jumping rope", "Rowing",
    "Hiking", "Strength training", "Circuit training", "Aerobic classes",
    "Kickboxing", "Zumba", "Rock climbing", "Squats", "Push-ups", "Sit-ups"
]

def generate_recommendations(age, sex, cholesterol, blood_pressure, diabetes, exercise, diet):
    recommendations = []
    
    # Basic health recommendations
    recommendations.append("Get regular check-ups with your healthcare provider.")
    recommendations.append("Aim for 7-9 hours of quality sleep each night.")
    
    # Age-specific
    if age > 50:
        recommendations.append("Consider more frequent heart health screenings due to age.")
    
    # Gender-specific
    if sex == 'Male':
        recommendations.append("Men should pay special attention to heart health indicators.")
    else:
        recommendations.append("Women should be aware of unique heart disease risk factors.")
    
    # Cholesterol
    if cholesterol > 200:
        recommendations.append(f"Your cholesterol level of {cholesterol} is elevated. Consider dietary changes.")
    
    # Blood pressure
    if blood_pressure > 130:
        recommendations.append(f"Your blood pressure of {blood_pressure} is elevated. Monitor regularly.")
    
    # Diabetes
    if diabetes == 'Yes':
        recommendations.append("Careful diabetes management is crucial for heart health.")
    
    # Exercise
    if exercise == 'No':
        rec_exercises = random.sample(exercises, 3)
        recommendations.append("Try incorporating exercise into your routine, such as:")
        recommendations.extend([f"- {ex}" for ex in rec_exercises])
    
    # Diet
    if diet == 'Poor':
        recommendations.append("Improve your diet by reducing processed foods and increasing vegetables.")
    elif diet == 'Fair':
        recommendations.append("Your diet could benefit from more whole foods and less sugar.")
    
    # Add random general recommendations
    general_recs = recommendations_data.sample(5)['Recommendation Description'].tolist()
    recommendations.extend(general_recs)
    
    return recommendations

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/home')
def home():
    return render_template('index.html', symptoms_data=symptoms_data)

@app.route('/predict', methods=['POST'])
def predict_heart_disease():
    try:
        # Prepare input features
        features = {
            'age': int(request.form['age']),
            'sex': 1 if request.form['sex'] == 'Male' else 0,
            'cp': int(request.form['cp']),
            'trestbps': int(request.form['trestbps']),
            'chol': int(request.form['chol']),
            'fbs': int(request.form['fbs']),
            'restecg': int(request.form['restecg']),
            'thalach': int(request.form['thalach']),
            'exang': int(request.form['exang']),
            'oldpeak': float(request.form['oldpeak']),
            'slope': int(request.form['slope']),
            'ca': int(request.form['ca']),
            'thal': int(request.form['thal'])
        }
        
        # Convert to numpy array in correct order
        feature_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        input_data = np.array([[features[col] for col in feature_order]])
        
        # Get prediction and probability
        prediction = heart_disease_model.predict(input_data)[0]
        probability = heart_disease_model.predict_proba(input_data)[0][1]
        
        # Generate recommendations
        recs = generate_recommendations(
            age=features['age'],
            sex=request.form['sex'],
            cholesterol=features['chol'],
            blood_pressure=features['trestbps'],
            diabetes=request.form['diabetes'],
            exercise=request.form['exercise'],
            diet=request.form['diet']
        )
        
        return render_template('result.html',
                            diagnosis='Positive' if prediction == 1 else 'Negative',
                            probability=f"{probability:.1%}",
                            recommendations=recs)
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)