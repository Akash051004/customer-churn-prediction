import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

df = pd.read_csv('data/churn_data.csv')

print('Shape:', df.shape)
print(df.head(5))
print('\nColumn names:')
print(df.columns.tolist())

print('\nData info:')
df.info()

print('\nNumerical summary:')
print(df.describe())

print('\nMissing values per column:')
print(df.isnull().sum())

print('\nChurn distribution:')
print(df['Churn'].value_counts())
print('Churn rate:', df['Churn'].value_counts(normalize=True).round(3))


print('TotalCharges dtype:', df['TotalCharges'].dtype)
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print('Missing values in TotalCharges:', df['TotalCharges'].isnull().sum())
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
print('Missing values in TotalCharges after filling:',
      df['TotalCharges'].isnull().sum())
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print('\nData types after cleaning:')
print(df.dtypes)
print('\nMissing values after cleaning:')
print(df.isnull().sum().sum())
print('\nChurn values:', df['Churn'].unique())
