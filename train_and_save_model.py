import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load and preprocess the data
df = pd.read_csv('SolarPrediction.csv')

# Filter out negative and zero Radiation values
df = df[df['Radiation'] > 0]

# Handling outliers using Interquartile Range (IQR)
for column in ['Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed']:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Feature Engineering: Add time-based features
df['Hour'] = pd.to_datetime(df['Time']).dt.hour
df['Minute'] = pd.to_datetime(df['Time']).dt.minute

# Cyclical Encoding of Time Features
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['Minute_sin'] = np.sin(2 * np.pi * df['Minute'] / 60)
df['Minute_cos'] = np.cos(2 * np.pi * df['Minute'] / 60)

# Define features and target variable
features = df[['UNIXTime', 'Temperature', 'Pressure', 'Humidity',
               'WindDirection(Degrees)', 'Speed', 'Hour_sin', 'Hour_cos',
               'Minute_sin', 'Minute_cos']]
target = df['Radiation']

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Train the XGBoost model with updated hyperparameters
xgb_model = XGBRegressor(
    n_estimators=1500,
    learning_rate=0.01,
    max_depth=8,
    min_child_weight=5,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    reg_alpha=2.0,
    random_state=42
)
xgb_model.fit(xtrain, ytrain)

# Make predictions on the test set
predictions = xgb_model.predict(xtest)

# Ensure non-negative predictions
predictions = np.maximum(predictions, 0)

# Evaluate model performance
mse = mean_squared_error(ytest, predictions)
print(f"Mean Squared Error: {mse}")

# Test cases for validation
test_cases = [
    [1475208022, 19.42, 55, 30.44, 57, 58.42, np.sin(2 * np.pi * 18 / 24), np.cos(2 * np.pi * 18 / 24),
     np.sin(2 * np.pi * 0 / 60), np.cos(2 * np.pi * 0 / 60)],
    [1475204125, 300.94, 58, 30.43, 55, 45.88, np.sin(2 * np.pi * 16 / 24), np.cos(2 * np.pi * 16 / 24),
     np.sin(2 * np.pi * 55 / 60), np.cos(2 * np.pi * 55 / 60)]
]

# Predicting for test cases
for case in test_cases:
    case_scaled = scaler.transform([case])
    predicted_radiance = xgb_model.predict(case_scaled)[0]
    predicted_radiance = max(predicted_radiance, 0)  # Ensure non-negativity
    print(f"Predicted Radiance: {predicted_radiance}")
