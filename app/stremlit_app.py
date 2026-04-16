import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# ── Page configuration ────────────────────────────────────
st.set_page_config(
    page_title='Customer Churn Predictor',
    page_icon='📊',
    layout='wide'
)

# ── Load saved model and scaler ──────────────────────────


@st.cache_resource  # cache so model loads only once
def load_model():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, '../models')

    model = joblib.load(os.path.join(models_dir, 'best_xgb_model.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    features = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
    return model, scaler, features


model, scaler, feature_names = load_model()

# ── Title and description ─────────────────────────────────
st.title('Customer Churn Prediction System')
st.markdown('Enter customer details below to predict churn probability.')
st.divider()

# ── Input form — two columns ─────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('Account Info')
    tenure = st.slider('Tenure (months)', 0, 72, 12)
    contract = st.selectbox('Contract Type',
                            ['Month-to-month', 'One year', 'Two year'])
    payment_method = st.selectbox('Payment Method',
                                  ['Electronic check', 'Mailed check',
                                   'Bank transfer (automatic)', 'Credit card (automatic)'])

with col2:
    st.subheader('Services')
    internet_service = st.selectbox('Internet Service',
                                    ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security',  ['Yes', 'No'])
    tech_support = st.selectbox('Tech Support',     ['Yes', 'No'])
    streaming_tv = st.selectbox('Streaming TV',     ['Yes', 'No'])

with col3:
    st.subheader('Billing')
    monthly_charges = st.number_input('Monthly Charges ($)', 20.0, 120.0, 65.0)
    total_charges = st.number_input('Total Charges ($)', 0.0, 9000.0,
                                    monthly_charges * tenure)
    paperless = st.selectbox('Paperless Billing', ['Yes', 'No'])
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])

st.divider()

# ── Predict button ────────────────────────────────────────
if st.button('Predict Churn Risk', type='primary'):

    # Build input row matching the training feature columns
    input_data = {col: 0 for col in feature_names}
    # Start with all zeros, then fill in the customer's values

    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = monthly_charges
    input_data['TotalCharges'] = total_charges
    input_data['SeniorCitizen'] = 1 if senior_citizen == 'Yes' else 0
    input_data['PaperlessBilling'] = 1 if paperless == 'Yes' else 0
    input_data['OnlineSecurity'] = 1 if online_security == 'Yes' else 0
    input_data['TechSupport'] = 1 if tech_support == 'Yes' else 0
    input_data['StreamingTV'] = 1 if streaming_tv == 'Yes' else 0

    # One-hot encoding for contract
    if contract == 'One year':
        input_data['Contract_One year'] = 1
    if contract == 'Two year':
        input_data['Contract_Two year'] = 1

    # One-hot encoding for internet service
    if internet_service == 'Fiber optic':
        input_data['InternetService_Fiber optic'] = 1
    elif internet_service == 'No':
        input_data['InternetService_No'] = 1

    # Engineered features
    input_data['avg_monthly_spend'] = total_charges / (tenure + 1)
    input_data['is_new_customer'] = 1 if tenure <= 12 else 0
    input_data['num_services'] = sum([
        1 if online_security == 'Yes' else 0,
        1 if tech_support == 'Yes' else 0,
        1 if streaming_tv == 'Yes' else 0
    ])

    # Create DataFrame and scale numerical columns
    input_df = pd.DataFrame([input_data])
    scale_cols = ['tenure', 'MonthlyCharges',
                  'TotalCharges', 'avg_monthly_spend', 'num_services']
    input_df[scale_cols] = scaler.transform(input_df[scale_cols])

    # Get prediction and probability
    churn_prob = model.predict_proba(input_df)[0][1]
    churn_pred = model.predict(input_df)[0]

    # ── Display result ───────────────────────────────────
    st.subheader('Prediction Result')
    result_col1, result_col2 = st.columns(2)

    with result_col1:
        if churn_pred == 1:
            st.error(f'HIGH CHURN RISK: {churn_prob*100:.1f}%')
            st.write('This customer is likely to leave. Take action now.')
        else:
            st.success(f'LOW CHURN RISK: {churn_prob*100:.1f}%')
            st.write('This customer is likely to stay.')

    with result_col2:
        st.metric('Churn Probability', f'{churn_prob*100:.1f}%')
        st.progress(float(churn_prob))

    # ── Business recommendation ──────────────────────────
    st.subheader('Recommended Actions')
    if churn_pred == 1:
        if contract == 'Month-to-month':
            st.info('Offer a 20% discount to upgrade to annual contract')
        if tenure <= 12:
            st.info('Assign a customer success manager for onboarding support')
        if monthly_charges > 80:
            st.info('Review pricing plan — customer may feel overcharged')
