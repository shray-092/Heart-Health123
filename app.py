from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import joblib
import random

app = Flask(__name__)


heart_disease_model = joblib.load('heart_disease_model.sav')


recommendations_data = pd.read_csv('recommendations.csv')


symptoms_data = pd.read_csv('symptoms.csv')

exercises = [
    "Brisk walking",
    "Cycling",
    "Swimming",
    "Running",
    "Dancing",
    "Yoga",
    "Tai chi",
    "Pilates",
    "Jumping rope",
    "Rowing",
    "Hiking",
    "Strength training",
    "Circuit training",
    "Aerobic classes",
    "Kickboxing",
    "Zumba",
    "Rock climbing",
    "Squats",
    "Push-ups",
    "Sit-ups"
]

def generate_recommendations(age, sex, cholesterol, blood_pressure, diabetes, exercise, diet):
    recommendations = []

    if age > 50:
        recommendations.append("Consider regular screenings for heart disease due to increased risk with age.")
    if sex == 'Male':
        recommendations.append("Men are at higher risk of heart disease; maintain a healthy lifestyle.")
    if cholesterol > 200:
        recommendations.append("Monitor cholesterol levels regularly and follow medical advice for management.")
    if blood_pressure > 120:
        recommendations.append("Keep blood pressure in check through diet, exercise, and medication if needed.")
    if diabetes == 'Yes':
        recommendations.append("Manage diabetes effectively to reduce the risk of cardiovascular complications.")
    if exercise == 'No':
        recommendations.append("Regular exercise is important for heart health; try to incorporate physical activity into your routine.")
    if diet == 'Poor':
        recommendations.append("A healthy diet is crucial for preventing heart disease; consider reducing intake of processed foods and sugar.")

    
        recommendations.append("Start by making small changes to your diet, such as swapping out sugary drinks for water or adding more vegetables to your meals.")
        if cholesterol > 200:
            recommendations.append("Choose heart-healthy fats like those found in avocados, nuts, and olive oil instead of saturated or trans fats.")
        if blood_pressure > 120:
            recommendations.append("Incorporate more potassium-rich foods like bananas, sweet potatoes, and leafy greens into your diet to help lower blood pressure.")
        if diabetes == 'Yes':
            recommendations.append("Focus on portion control and carbohydrate counting to manage blood sugar levels effectively, and consider meeting with a certified diabetes educator for personalized guidance.")
        recommendations.append("Include more fiber-rich foods in your diet such as whole grains, legumes, fruits, and vegetables to support digestive health and lower the risk of heart disease.")
        recommendations.append("Opt for lean protein sources like poultry, fish, tofu, and beans instead of processed and red meats, which can help reduce the risk of heart disease.")
        recommendations.append("Incorporate foods rich in antioxidants such as berries, leafy greens, and dark chocolate into your diet to help protect against oxidative stress and inflammation.")
        recommendations.append("Include foods high in omega-3 fatty acids such as flaxseeds, chia seeds, and walnuts, which can help reduce inflammation and lower the risk of heart disease.")
        recommendations.append("Swap out refined grains for whole grains whenever possible, such as choosing whole wheat bread instead of white bread and brown rice instead of white rice.")
        recommendations.append("Experiment with different herbs and spices to add flavor to your meals without relying on excess salt, which can contribute to high blood pressure.")

 
    if age >= 60:
        recommendations.append("Consider joining a local walking group or fitness class for seniors to stay active and socialize with others in your age group.")
    if sex == 'Male':
        recommendations.append("Stay connected with friends and family members for emotional support, and don't hesitate to reach out if you need help or someone to talk to.")
    elif sex == 'Female':
        recommendations.append("Take time for yourself and prioritize self-care activities such as journaling, meditation, or enjoying a warm bath.")
    if cholesterol > 220:
        recommendations.append("Work with a dietitian to develop personalized meal plans and recipes that meet your nutritional needs and support heart health.")
    if blood_pressure > 140:
        recommendations.append("Explore alternative therapies such as acupuncture, massage therapy, or biofeedback to help manage stress and lower blood pressure naturally.")
    if diabetes == 'Yes':
        recommendations.append("Stay up-to-date with the latest diabetes research and treatment options, and advocate for yourself to ensure you're receiving the best possible care.")
    if exercise == 'Yes' and diet == 'Fair':
        recommendations.append("Continue to prioritize regular exercise and a balanced diet, as these are essential components of heart disease prevention and management.")

    
    if exercise == 'No':
        num_exercises_to_recommend = 3 
        recommended_exercises = random.sample(exercises, num_exercises_to_recommend)
        recommendations.append("Consider trying the following exercises to improve your cardiovascular health:")
        for exercise in recommended_exercises:
            recommendations.append("- " + exercise)


    num_general_recommendations = 10  
    general_recommendations = recommendations_data.sample(n=num_general_recommendations, replace=True)['Recommendation Description'].tolist()
    recommendations.extend(general_recommendations)

    random.shuffle(recommendations)

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
        age = int(request.form['age'])
        sex = request.form['sex']
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])
        diabetes = request.form['diabetes']
        exercise = request.form['exercise']
        diet = request.form['diet']
    except ValueError:
        return "Invalid input data. Please make sure all input fields contain numeric values."

    try:
        
        sex_numeric = 1 if sex == 'Male' else 0
        diabetes_numeric = 1 if diabetes == 'Yes' else 0
        exercise_numeric = 1 if exercise == 'Yes' else 0
        diet_numeric = 1 if diet == 'Good' else 0
        
        prediction = heart_disease_model.predict([[age, sex_numeric, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        diagnosis = 'Positive' if prediction else 'Negative'
        recommendations = generate_recommendations(age, sex, chol, trestbps, diabetes, exercise, diet)
    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

    return render_template('result.html', diagnosis=diagnosis, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)