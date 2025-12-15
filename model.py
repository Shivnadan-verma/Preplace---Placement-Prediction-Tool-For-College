# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("placement_data.csv")

# Handle Missing Values
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

categorical_columns = df.select_dtypes(include=['object', 'category']).columns
df[categorical_columns] = df[categorical_columns].apply(lambda col: col.fillna(col.mode()[0]))

# Aggregate AMCAT Sectional Scores
required_columns = ['Placement_status', 'sessional_marks', 'Class_attendance', 'project_marks',
                                    'LA2SS', 'LA3SS', 'LA4SS', 'LA5SS', 'LA6SS',
                                    'QA2SS', 'QA3SS', 'QA4SS', 'QA5SS', 'QA6SS',
                                    'EC2SS', 'EC3SS', 'EC4SS', 'EC5SS', 'EC6SS',
                                    'DS3S', 'DS4S', 'DS5S', 'DS6S',
                                    'AS5S', 'AS6S']

amcat_numeric_columns = df[required_columns].apply(pd.to_numeric, errors='coerce')
df['Logical_Ability'] = amcat_numeric_columns[['LA2SS', 'LA3SS', 'LA4SS', 'LA5SS', 'LA6SS']].mean(axis=1)
df['Quantitative_Ability'] = amcat_numeric_columns[['QA2SS', 'QA3SS', 'QA4SS', 'QA5SS', 'QA6SS']].mean(axis=1)
df['English_Comprehension'] = amcat_numeric_columns[['EC2SS', 'EC3SS', 'EC4SS', 'EC5SS', 'EC6SS']].mean(axis=1)
df['Domain_Knowledge'] = amcat_numeric_columns[['DS3S', 'DS4S', 'DS5S', 'DS6S']].mean(axis=1)
df['Automata_Programming'] = amcat_numeric_columns[['AS5S', 'AS6S']].mean(axis=1)

# Features and Target Variable
features = ['sessional_marks',  'Class_attendance', 
            'Logical_Ability', 'Quantitative_Ability', 
            'English_Comprehension', 'Domain_Knowledge', 'Automata_Programming', 'project_marks']
X = df[features]
y = df['Placement_status']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Validation-Test Split
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Hyperparameter Tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
grid_search = GridSearchCV(SVC(), param_grid, scoring='accuracy', cv=5, verbose=2)
grid_search.fit(X_train, y_train)

# Evaluate on Validation and Test Sets
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Validation Accuracy: {val_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save Model and Scaler
with open('final_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)
with open('final_scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Feature Importance for Linear Kernel
if grid_search.best_params_['kernel'] == 'linear':
    feature_importance = best_model.coef_[0]
    feature_names = features
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance, y=feature_names, palette="coolwarm")
    plt.title('Feature Importance for SVM (Linear Kernel)', fontsize=16)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.show()
else:
    print("Feature importance visualization is applicable only for linear kernel.")

# Visualize AMCAT Scores
aggregated_scores = df[['Logical_Ability', 'Quantitative_Ability', 'English_Comprehension', 'Domain_Knowledge', 'Automata_Programming']].mean()

plt.figure(figsize=(10, 6))
sns.barplot(x=aggregated_scores.index, y=aggregated_scores.values, palette="viridis")
plt.title('Average AMCAT Sectional Scores', fontsize=16)
plt.ylabel('Average Score')
plt.xlabel('AMCAT Sections')
plt.xticks(rotation=45)
plt.show()

# AMCAT Scores vs Placement Status
df_long = pd.melt(df, id_vars=['Placement_status'], value_vars=['Logical_Ability', 'Quantitative_Ability', 
                                                                'English_Comprehension', 'Domain_Knowledge', 
                                                                'Automata_Programming'], 
                  var_name='AMCAT Section', value_name='Score')

plt.figure(figsize=(12, 6))
sns.barplot(data=df_long, x='AMCAT Section', y='Score', hue='Placement_status', palette='Spectral')
plt.title('AMCAT Scores by Placement Status', fontsize=16)
plt.ylabel('Score')
plt.xlabel('AMCAT Section')
plt.xticks(rotation=45)
plt.legend(title='Placement Status', loc='upper right')
plt.show()

#Shivnandan verma 