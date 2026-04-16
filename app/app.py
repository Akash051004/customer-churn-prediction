from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import precision_recall_curve
from imblearn.combine import SMOTEENN
import time

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

df = pd.read_csv('data/churn_data.csv')
'''
print('Shape:', df.shape)
print(df.head(5))
print('\nColumn names:')
print(df.columns.tolist())
'''
'''
print('\nData info:')
df.info()
'''
'''
print('\nNumerical summary:')
print(df.describe())

print('\nMissing values per column:')
print(df.isnull().sum())

print('\nChurn distribution:')
print(df['Churn'].value_counts())
print('Churn rate:', df['Churn'].value_counts(normalize=True).round(3))
'''

df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

'''
print('\nData types after cleaning:')
print(df.dtypes)
print('\nMissing values after cleaning:')
print(df.isnull().sum().sum())
print('\nChurn values:', df['Churn'].unique())

plt.figure(figsize=(6, 4))
df['Churn'].value_counts().plot(kind='bar', color=['steelblue', 'coral'],
                                edgecolor='black')
plt.xticks([0, 1], ['Stayed (0)', 'Churned (1)'], rotation=0)
plt.title('Customer Churn Distribution')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plots/churn_distribution.png', dpi=150)
plt.show()


plt.figure(figsize=(8, 5))
contract_churn = df.groupby('Contract')['Churn'].mean() * 100
contract_churn.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Churn Rate by Contract Type (%)')
plt.ylabel('Churn Rate %')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/contract_churn.png', dpi=150)
plt.show()


plt.figure(figsize=(10, 5))
df[df['Churn'] == 0]['tenure'].hist(
    bins=30, alpha=0.6, label='Stayed', color='steelblue')
df[df['Churn'] == 1]['tenure'].hist(
    bins=30, alpha=0.6, label='Churned', color='coral')
plt.xlabel('Tenure (months)')
plt.ylabel('Number of Customers')
plt.title('Tenure Distribution: Churned vs Stayed')
plt.legend()
plt.tight_layout()
plt.savefig('plots/tenure_distribution.png', dpi=150)
plt.show()


plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # hide upper triangle
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
df.boxplot(column='MonthlyCharges', by='Churn', figsize=(8, 5))
plt.suptitle('')
plt.title('Monthly Charges: Churned (1) vs Stayed (0)')
plt.xlabel('Churn')
plt.ylabel('Monthly Charges ($)')
plt.tight_layout()
plt.savefig('plots/monthly_charges_boxplot.png', dpi=150)
plt.show()
'''

categorical_cols = df.select_dtypes(include='object').columns.tolist()
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove('Churn')

'''
print('Categorical columns:', categorical_cols)
print('Numerical columns:', numerical_cols)

for col in categorical_cols:
    print(f'{col}: {df[col].unique()}')
'''

binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
               'PaperlessBilling', 'MultipleLines',
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
               'TechSupport', 'StreamingTV', 'StreamingMovies']

binary_map = {'Yes': 1, 'No': 0, 'Female': 1, 'Male': 0,
              'No phone service': 0, 'No internet service': 0}
for col in binary_cols:
    df[col] = df[col].map(binary_map)

multi_class_cols = ['InternetService', 'Contract', 'PaymentMethod']
df = pd.get_dummies(df, columns=multi_class_cols, drop_first=True)
df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'] + 1)
df['is_new_customer'] = (df['tenure'] <= 12).astype(int)
service_cols = ['PhoneService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['num_services'] = df[service_cols].sum(axis=1)
df['tenure_monthly_ratio'] = df['tenure'] / (df['MonthlyCharges'] + 1)
df['high_spender'] = (df['MonthlyCharges'] >
                      df['MonthlyCharges'].quantile(0.75)).astype(int)
'''
print('\nFinal dataset shape:', df.shape)
print('New features created: avg_monthly_spend, is_new_customer, num_services')
'''

X = df.drop('Churn', axis=1)
y = df['Churn']
'''
print('Features shape:', X.shape)
print('Target shape:',  y.shape)
print('Class distribution:', y.value_counts().to_dict())
'''

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
'''
print(f'Training: {X_train.shape[0]} samples')
print(f'Testing:  {X_test.shape[0]} samples')
'''
scaler = StandardScaler()
scale_cols = ['tenure', 'MonthlyCharges',
              'TotalCharges', 'avg_monthly_spend', 'num_services']
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])
# ADD THIS

smote = SMOTEENN(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
'''
print(f'\nAfter SMOTE:')
print(f'Training samples: {X_train_sm.shape[0]}')
print(
    f'Class distribution: {dict(zip(*np.unique(y_train_sm, return_counts=True)))}')
'''

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=3000,
        C=0.1,
        solver='saga',
        penalty='elasticnet',
        l1_ratio=0.5,
        tol=1e-5,
        random_state=42
    ),

    'Random Forest': RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=4,
        min_samples_split=10,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    ),

    'XGBoost': XGBClassifier(
        n_estimators=700,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        gamma=0.3,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_weight=5,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1
    )
}
results = {}

for name, model in models.items():
    start = time.time()
    model.fit(X_train_sm, y_train_sm)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    elapsed = round(time.time() - start, 2)

    results[name] = {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall':    recall_score(y_test, y_pred),
        'F1 Score':  f1_score(y_test, y_pred),
        'AUC-ROC':   roc_auc_score(y_test, y_proba),
        'Train Time (s)': elapsed
    }

    print(f'\n--- {name} ---')
    print(classification_report(y_test, y_pred))

results_df = pd.DataFrame(results).T.round(4)
print('\n=== MODEL COMPARISON ===')
print(results_df.to_string())

print('\n=== THRESHOLD TUNED RESULTS ===')
for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]

    y_pred_tuned = (y_proba >= best_thresh).astype(int)

    print(f'\n{name} (threshold={best_thresh:.2f})')
    print(f'  Accuracy : {accuracy_score(y_test, y_pred_tuned):.4f}')
    print(f'  Precision: {precision_score(y_test, y_pred_tuned):.4f}')
    print(f'  Recall   : {recall_score(y_test, y_pred_tuned):.4f}')
    print(f'  F1 Score : {f1_score(y_test, y_pred_tuned):.4f}')


param_dist = {
    'n_estimators':     randint(100, 400),
    'max_depth':        randint(3, 9),
    'learning_rate':    uniform(0.01, 0.2),
    'subsample':        uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_weight': randint(1, 10),
    'gamma':            uniform(0, 0.5),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rand_search = RandomizedSearchCV(
    estimator=XGBClassifier(random_state=42, use_label_encoder=False,
                            eval_metric='logloss'),
    param_distributions=param_dist,
    n_iter=50,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print('Running hyperparameter search (may take a few minutes)...')
rand_search.fit(X_train_sm, y_train_sm)

print('\nBest Parameters Found:')
print(rand_search.best_params_)
print(f'Best CV AUC-ROC: {rand_search.best_score_:.4f}')

best_xgb = rand_search.best_estimator_

y_pred_best = best_xgb.predict(X_test)
y_proba_best = best_xgb.predict_proba(X_test)[:, 1]

print(
    f'\nTuned XGBoost Test AUC-ROC: {roc_auc_score(y_test, y_proba_best):.4f}')
print(f'Tuned XGBoost Test F1:       {f1_score(y_test, y_pred_best):.4f}')
print('\nFull Classification Report:')
print(classification_report(y_test, y_pred_best))
