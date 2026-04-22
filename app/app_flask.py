"""
ChurnSense — Flask Backend
Run: python app_flask.py
Visit: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

# ── App setup ────────────────────────────────────────────────
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = os.path.join(CURR_DIR, 'static')
app = Flask(__name__, static_folder=STATIC_FOLDER)
CORS(app)

# ── Load models ───────────────────────────────────────────────
# Go up one level to project root
BASE_DIR = CURR_DIR
MODELS_DIR = os.path.join(BASE_DIR, 'models')

try:
    model = joblib.load(os.path.join(MODELS_DIR, 'best_xgb_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
    print("✅  Models loaded successfully.")
except FileNotFoundError as e:
    print(f"⚠️   Model file not found: {e}")
    print("    Run app.py first to train and save the models.")
    model, scaler, feature_names = None, None, None


# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(STATIC_FOLDER, 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Run app.py first.'}), 500

    data = request.get_json(force=True)

    # ── Build feature vector ──────────────────────────────────
    input_data = {col: 0 for col in feature_names}

    tenure = int(data.get('tenure', 0))
    monthly_charges = float(data.get('monthly_charges', 65))
    total_charges = float(data.get('total_charges', 780))

    # Core fields
    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = monthly_charges
    input_data['TotalCharges'] = total_charges
    input_data['SeniorCitizen'] = 1 if data.get(
        'senior_citizen') == 'Yes' else 0
    input_data['PaperlessBilling'] = 1 if data.get('paperless') == 'Yes' else 0

    # Services (only the ones we have in the form)
    input_data['OnlineSecurity'] = 1 if data.get(
        'online_security') == 'Yes' else 0
    input_data['TechSupport'] = 1 if data.get('tech_support') == 'Yes' else 0
    input_data['StreamingTV'] = 1 if data.get('streaming_tv') == 'Yes' else 0

    # One-hot encoding for contract
    contract = data.get('contract', 'Month-to-month')
    if contract == 'One year':
        input_data['Contract_One year'] = 1
    elif contract == 'Two year':
        input_data['Contract_Two year'] = 1

    # One-hot encoding for internet service
    internet = data.get('internet_service', 'DSL')
    if internet == 'Fiber optic':
        input_data['InternetService_Fiber optic'] = 1
    elif internet == 'No':
        input_data['InternetService_No'] = 1

    # One-hot encoding for payment method
    payment = data.get('payment_method', 'Electronic check')
    payment_map = {
        'Mailed check': 'PaymentMethod_Mailed check',
        'Bank transfer (automatic)': 'PaymentMethod_Bank transfer (automatic)',
        'Credit card (automatic)': 'PaymentMethod_Credit card (automatic)',
    }
    if payment in payment_map:
        input_data[payment_map[payment]] = 1

    # Engineered features
    input_data['avg_monthly_spend'] = total_charges / (tenure + 1)
    input_data['is_new_customer'] = 1 if tenure <= 12 else 0
    input_data['num_services'] = sum([
        1 if data.get('online_security') == 'Yes' else 0,
        1 if data.get('tech_support') == 'Yes' else 0,
        1 if data.get('streaming_tv') == 'Yes' else 0
    ])

    # ── Build DataFrame ───────────────────────────────────────
    df = pd.DataFrame([input_data])

    # Scale numerical columns
    scale_cols = ['tenure', 'MonthlyCharges',
                  'TotalCharges', 'avg_monthly_spend', 'num_services']
    df[scale_cols] = scaler.transform(df[scale_cols])

    # ── Predict ───────────────────────────────────────────────
    churn_prob = float(model.predict_proba(df)[0][1])
    churn_pred = int(model.predict(df)[0])

    # Prepare response
    if churn_pred == 1:
        message = "This customer is likely to leave. Take action now."
        recommendations = []
        if contract == 'Month-to-month':
            recommendations.append(
                "Offer a 20% discount to upgrade to annual contract")
        if tenure <= 12:
            recommendations.append(
                "Assign a customer success manager for onboarding support")
        if monthly_charges > 80:
            recommendations.append(
                "Review pricing plan — customer may feel overcharged")
    else:
        message = "This customer is likely to stay."
        recommendations = []

    return jsonify({
        'churn_probability': churn_prob * 100,
        'churn_prediction': churn_pred,
        'message': message,
        'recommendations': '. '.join(recommendations) if recommendations else None
    })


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'features': len(feature_names) if feature_names else 0
    })


# ── Run ────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🚀  ChurnSense server starting...")
    print("    Open http://localhost:7860 in your browser\n")
    app.run(debug=False, port=7860, host='0.0.0.0')
