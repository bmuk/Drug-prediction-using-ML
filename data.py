import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report



df = pd.read_csv('drug200.csv')

print("Dataset Info:")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())

print("\nDescriptive Statistics:")
print(df.describe())


label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['BP'] = label_encoder.fit_transform(df['BP'])
df['Cholesterol'] = label_encoder.fit_transform(df['Cholesterol'])

print("\nEncoded Data:")
print(df.head())

# Apply PCA
pca = PCA(n_components=2)
pca_features = pca.fit_transform(df.drop('Drug', axis=1))

# Create a DataFrame with PCA features
pca_df = pd.DataFrame(pca_features, columns=['PC1', 'PC2'])
pca_df['Drug'] = df['Drug']

# Display PCA DataFrame
print("\nPCA DataFrame:")
print(pca_df.head())

scaler = StandardScaler()
df[['Age', 'Na_to_K']] = scaler.fit_transform(df[['Age', 'Na_to_K']])

print("\nScaled Data:")
print(df.head())

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

rf = RandomForestClassifier()
rf.fit(df.drop('Drug', axis=1), df['Drug'])

plt.figure(figsize=(10, 6))
sns.barplot(x=rf.feature_importances_, y=df.drop('Drug', axis=1).columns)
plt.title("Feature Importance Using Random Forest")
plt.show()

X = df.drop('Drug', axis=1)
y = df['Drug']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'XGBoost': XGBClassifier()
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append([name, accuracy])

results_df = pd.DataFrame(results, columns=['Model', 'Accuracy'])
print("\nModel Accuracy:")
print(results_df.sort_values(by='Accuracy', ascending=False))

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("\nBest Parameters for Random Forest:")
print(grid_search.best_params_)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df.drop('Drug', axis=1))

print("\nClustered Data:")
print(df.head())

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Na_to_K', hue='Cluster', data=df, palette='viridis')
plt.title("K-Means Clustering")
plt.show()

import joblib

best_model = grid_search.best_estimator_
joblib.dump(best_model, 'drug_prediction_model.pkl')

loaded_model = joblib.load('drug_prediction_model.pkl')

y_pred = loaded_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

