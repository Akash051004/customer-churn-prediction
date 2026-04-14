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
