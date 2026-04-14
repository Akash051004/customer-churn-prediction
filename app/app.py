import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

df = pd.read_csv('data/churn_data.csv')

print('Shape:', df.shape)
print('\nFirst 5 rows:')
print(df.head())
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
