from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os

# Flask setup
app = Flask(__name__, template_folder='templates')
CORS(app, resources={
    r"/predict": {"origins": "*"},
    r"/": {"origins": "*"}
})  

# Load models
try:
    with open('hallmark_model.pkl', 'rb') as f:
        hallmark_model, hallmark_scaler, hallmark_encoder, hallmark_label_encoders = pickle.load(f)

    with open('akorno_model.pkl', 'rb') as f:
        akorno_model, akorno_scaler, akorno_encoder, akorno_label_encoders = pickle.load(f)

    with open('munchies_model.pkl', 'rb') as f:
        munchies_model, munchies_scaler, munchies_encoder, munchies_label_encoders = pickle.load(f)

except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    exit()

# Mappings
day_mapping = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 
    'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
}
cafeteria_mapping = {
    "Akorno Services Ltd - Main Cafe": 0, 
    "Hallmark": 1, 
    "Munchies Services Ltd": 2
}
meal_period_mapping = {"Breakfast": 0, "Lunch": 1, "Dinner": 2}


@app.route('/')
def menu():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() if request.is_json else request.form
        if not data:
            return jsonify({'error': 'No data received'}), 400

        cafreteria_input = data.get('cafreteria')  
        day_of_week_input = data.get('day_of_week')
        meal_period_input = data.get('meal_period')

        if not all([cafreteria_input, day_of_week_input, meal_period_input]):
            return jsonify({'error': 'Missing required fields'}), 400

        cafreteria = cafeteria_mapping.get(cafreteria_input)
        day_of_week = day_mapping.get(day_of_week_input)
        meal_period = meal_period_mapping.get(meal_period_input)

        if None in (cafreteria, day_of_week, meal_period):
            return jsonify({'error': 'Invalid input values'}), 400

        model_dict = {
            "Hallmark": (hallmark_model, hallmark_scaler, hallmark_encoder, hallmark_label_encoders),
            "Akorno Services Ltd - Main Cafe": (akorno_model, akorno_scaler, akorno_encoder, akorno_label_encoders),
            "Munchies Services Ltd": (munchies_model, munchies_scaler, munchies_encoder, munchies_label_encoders)
        }

        model_data = model_dict.get(cafreteria_input)
        if not model_data:
            return jsonify({'error': 'Invalid cafeteria name'}), 400

        model, scaler, encoder, label_encoders = model_data
        input_data = pd.DataFrame({
            'cafetreria': [cafreteria],
            'day_of_week': [day_of_week], 
            'meal_period': [meal_period]
        })
        
        scaled_features = scaler.transform(input_data)
        prediction_probabilities = model.predict_proba(scaled_features)
        top_indices = np.argsort(prediction_probabilities[0])[-3:][::-1]
        
        label_encoder = label_encoders.get('product')
        if not label_encoder:
            return jsonify({'error': 'Label encoder not found'}), 500

        top_predictions = label_encoder.inverse_transform(top_indices)
        return jsonify({
            "predictions": top_predictions.tolist(),
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)