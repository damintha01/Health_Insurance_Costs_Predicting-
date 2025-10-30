from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the trained models and encoders
def load_models():
    try:
        # Load label encoders
        gender_encoder = joblib.load('gender_label_encoder.pkl')
        diabetic_encoder = joblib.load('diabetic_label_encoder.pkl')
        smoker_encoder = joblib.load('smoker_label_encoder.pkl')
        region_encoder = joblib.load('region_label_encoder.pkl')
        
        # Load scaler
        scaler = joblib.load('scaler.pkl')
        
        # Load the best model
        best_model = joblib.load('best_model.pkl')
        
        return {
            'encoders': {
                'gender': gender_encoder,
                'diabetic': diabetic_encoder,
                'smoker': smoker_encoder,
                'region': region_encoder
            },
            'scaler': scaler,
            'model': best_model
        }
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

models = load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if models is None:
            return jsonify({'error': 'Models not loaded properly'}), 500
        
        # Get form data
        age = float(request.form['age'])
        gender = request.form['gender']
        bmi = float(request.form['bmi'])
        bloodpressure = float(request.form['bloodpressure'])
        diabetic = request.form['diabetic']
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'bmi': [bmi],
            'bloodpressure': [bloodpressure],
            'diabetic': [diabetic],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        })
        
        # Encode categorical variables
        input_data['gender'] = models['encoders']['gender'].transform([gender])[0]
        input_data['diabetic'] = models['encoders']['diabetic'].transform([diabetic])[0]
        input_data['smoker'] = models['encoders']['smoker'].transform([smoker])[0]
        input_data['region'] = models['encoders']['region'].transform([region])[0]
        
        # Scale numerical features
        num_cols = ['age', 'bmi', 'bloodpressure', 'children']
        input_data[num_cols] = models['scaler'].transform(input_data[num_cols])
        
        # Make prediction
        prediction = models['model'].predict(input_data)[0]
        
        return render_template('result.html', 
                             prediction=round(prediction, 2),
                             age=age,
                             gender=gender,
                             bmi=bmi,
                             bloodpressure=bloodpressure,
                             diabetic=diabetic,
                             children=children,
                             smoker=smoker,
                             region=region)
    
    except Exception as e:
        return render_template('result.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        if models is None:
            return jsonify({'error': 'Models not loaded properly'}), 500
        
        data = request.get_json()
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [float(data['age'])],
            'gender': [data['gender']],
            'bmi': [float(data['bmi'])],
            'bloodpressure': [float(data['bloodpressure'])],
            'diabetic': [data['diabetic']],
            'children': [int(data['children'])],
            'smoker': [data['smoker']],
            'region': [data['region']]
        })
        
        # Encode categorical variables
        input_data['gender'] = models['encoders']['gender'].transform([data['gender']])[0]
        input_data['diabetic'] = models['encoders']['diabetic'].transform([data['diabetic']])[0]
        input_data['smoker'] = models['encoders']['smoker'].transform([data['smoker']])[0]
        input_data['region'] = models['encoders']['region'].transform([data['region']])[0]
        
        # Scale numerical features
        num_cols = ['age', 'bmi', 'bloodpressure', 'children']
        input_data[num_cols] = models['scaler'].transform(input_data[num_cols])
        
        # Make prediction
        prediction = models['model'].predict(input_data)[0]
        
        return jsonify({
            'prediction': round(float(prediction), 2),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
