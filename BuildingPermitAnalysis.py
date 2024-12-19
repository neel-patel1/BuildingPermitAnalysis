import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV

file_path = "C:/Users/Neel/Desktop/2023-2024/Spring 2024/Big Data and Forecasting/Building_Permits.csv"
data = pd.read_csv(file_path)
#print(data)

#Drop all observations where Current Status = expired or withdrawn
data = data[(data['Current Status'] != 'withdrawn')]

#Existing Units & Proposed Units (1 if value is same, 0 if value is not same)

bad = ['Block', 'Lot','Street Number Suffix', 'Permit Type Definition', 'Permit Number', 'Street Number', 'Street Name', 'Street Suffix', 'Unit', 'Unit Suffix', 'Description', 'Current Status', 'Current Status Date', 'Completed Date', 'Structural Notification', 'Voluntary Soft-Story Retrofit', 'Fire Only Permit', 'Permit Expiration Date', 'Existing Use', 'Proposed Use', 'TIDF Compliance', 'Existing Construction Type Description', 'Proposed Construction Type Description', 'Site Permit', 'Location', 'Record ID', 'Neighborhoods - Analysis Boundaries']
data = data.drop(columns = bad)
# print(data)

#Convert dates into datetime and subtract file date from issued date (and remove days)
data['Issued Date'] = pd.to_datetime(data['Issued Date'])
data['Filed Date']  = pd.to_datetime(data['Filed Date'])

data['datediff'] = (data['Issued Date'] - data['Filed Date']).dt.days

#Drop NA
data = data.dropna()

#Make dummies for Zipecode column, Proposed Construction type, and Plansets
data = pd.get_dummies(data, columns=['Zipcode'], drop_first= True, dtype = int)
data = pd.get_dummies(data, columns=['Proposed Construction Type'], drop_first= True, dtype = int)
data = pd.get_dummies(data, columns=['Plansets'], drop_first= True, dtype = int)

#Drop Date columns
date= ['Issued Date', 'Filed Date', 'Permit Creation Date', 'First Construction Document Date']
data = data.drop(columns=date)

# Assuming 'datediff' is your target variable, and other columns are features
X = data.drop(columns=['datediff'])
y = data['datediff']

# Create the Random Forest model
rf_model = RandomForestRegressor(n_estimators=50, random_state=42)

# Define a 10-fold cross-validation strategy
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and get MSE scores
mse_scores = cross_val_score(rf_model, X, y, cv=kf, scoring='neg_mean_squared_error')

# Convert negative MSE scores to positive
mse_scores = -mse_scores

# Display average MSE across folds
print(f'Average Mean Squared Error across 5 folds: {mse_scores.mean()}')

# Get feature importances from the model using the entire dataset
rf_model.fit(X, y)
feature_importances = rf_model.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the top features
print(feature_importance_df.head())

# Plot feature importances
plt.figure(figsize=(12, 6))
plt.barh(feature_importance_df['Feature'][:10], feature_importance_df['Importance'][:10])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances in Random Forest Model')
plt.show()

# SVM 
svm_model = SVR()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = cross_val_score(svm_model, X, y, cv=kf, scoring='neg_mean_squared_error')
mse_scores = -mse_scores
print(f'Average Mean Squared Error across 5 folds: {mse_scores.mean()}')
svm_model.fit(X, y)
y_pred = svm_model.predict(X)

# Calculate and display Mean Squared Error on the entire dataset
mse_full_data = mean_squared_error(y, y_pred)
print(f'Mean Squared Error on the entire dataset: {mse_full_data}')

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.scatter(y, y_pred)
plt.xlabel('Actual datediff')
plt.ylabel('Predicted datediff')
plt.title('Actual vs Predicted values for SVM')
plt.show()


# RidgeCV
ridge_model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=10)

# Fit the Ridge model on the entire dataset
ridge_model.fit(X, y)

# Get the optimal alpha (penalty term) selected by cross-validation
optimal_alpha = ridge_model.alpha_
print(f'Optimal Alpha selected by RidgeCV: {optimal_alpha}')

# Make predictions on the entire dataset
y_pred_ridge = ridge_model.predict(X)

# Calculate and display Mean Squared Error on the entire dataset
mse_full_data_ridge = mean_squared_error(y, y_pred_ridge)
print(f'Mean Squared Error on the entire dataset (Ridge): {mse_full_data_ridge}')

# Get the coefficients from the Ridge model
coefficients = pd.Series(ridge_model.coef_, index=X.columns)

# Create a DataFrame for the coefficients
coefficients_df = pd.DataFrame({'Feature': coefficients.index, 'Coefficient': coefficients.abs()})

# Sort the DataFrame by absolute coefficient values in descending order
coefficients_df = coefficients_df.sort_values(by='Coefficient', ascending=False)

# Display the sorted coefficients table
print(coefficients_df)
plt.figure(figsize=(10, 6))
plt.barh(coefficients_df['Feature'], coefficients_df['Coefficient'])
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Feature Importance in Ridge Regression')
plt.tight_layout()  # Adjust spacing
plt.show()







