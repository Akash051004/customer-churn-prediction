# Customer Churn Prediction System

A machine learning web application that predicts customer churn probability using XGBoost and provides actionable business insights.

## Features

- **Interactive Web Interface**: Built with Streamlit for easy customer data input
- **Real-time Predictions**: Instant churn probability calculations
- **Business Recommendations**: Actionable insights based on customer profile
- **SHAP Explanations**: Feature importance analysis for model interpretability

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Git (optional)

### Installation

1. **Clone or download the project**
   ```bash
   cd customer-churn-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

**Option 1: Using the provided script (Recommended)**
```bash
# For Windows PowerShell
.\run_app.ps1

# Or for Windows Command Prompt
run_app.bat
```

**Option 2: Manual activation**
```bash
# Activate virtual environment
.\.venv\Scripts\activate

# Run the app
streamlit run app/stremlit_app.py
```

**Option 3: Direct execution**
```bash
.\.venv\Scripts\python.exe -m streamlit run app/stremlit_app.py
```

### Accessing the App

Once running, open your browser and go to:
- **Local URL**: http://localhost:8501
- **Network URL**: http://192.168.1.x:8501 (replace x with your IP)

## Usage

1. Enter customer details in the web interface
2. Click "Predict Churn Risk"
3. View the churn probability and risk assessment
4. Review recommended actions for high-risk customers

## Project Structure

```
customer-churn-prediction/
├── app/
│   ├── stremlit_app.py      # Main Streamlit application
│   └── app.py              # Alternative Flask app (if needed)
├── data/
│   └── churn_data.csv      # Training dataset
├── models/
│   ├── best_xgb_model.pkl  # Trained XGBoost model
│   ├── scaler.pkl          # Feature scaler
│   └── feature_names.pkl   # Feature names list
├── notebooks/
│   └── churn_analysis.ipynb # EDA and model training notebook
├── plots/                  # Generated visualizations
├── requirements.txt        # Python dependencies
├── run_app.ps1            # PowerShell runner script
├── run_app.bat            # Batch runner script
└── README.md              # This file
```

## Dependencies

- streamlit: Web application framework
- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: Machine learning utilities
- xgboost: Gradient boosting algorithm
- shap: Model interpretability
- matplotlib: Plotting library
- joblib: Model serialization

## Troubleshooting

**ModuleNotFoundError**: If you get import errors, make sure you're using the virtual environment:
```bash
.\.venv\Scripts\activate
pip install -r requirements.txt
```

**Port already in use**: If port 8501 is busy, Streamlit will automatically use the next available port.

**Model loading errors**: Ensure all model files exist in the `models/` directory.

## License

This project is for educational and demonstration purposes.