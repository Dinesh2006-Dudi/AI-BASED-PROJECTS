import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data & preprocess
df = pd.read_csv('/home/heart disease data.csv', delimiter=';')
df = df.drop(columns=['id'])
df['age_years'] = (df['age'] / 365).round().astype(int)
df.drop(columns=['age'], inplace=True)

# Visualizations
plt.figure(figsize=(14,8))
df.hist(bins=15, figsize=(16,9))
plt.tight_layout()
plt.show()

sns.boxplot(data=df[['height','weight','ap_hi','ap_lo']])
plt.show()

sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.show()

# Split and scale
X = df.drop('cardio', axis=1)
y = df['cardio']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train logistic regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print('Logistic Regression Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
