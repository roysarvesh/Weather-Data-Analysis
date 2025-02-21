import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

try:
    df = pd.read_csv('weather.csv')
except FileNotFoundError:
    print("The file 'weather.csv' was not found. Please check the file path.")
    exit()

# Data Cleaning and Preprocessing
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})

categorical_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Data Visualization
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Weather Data")
plt.show()

# Feature Selection and Data Splitting
X = df.drop(['Rainfall', 'RISK_MM', 'RainTomorrow', 'Date'], axis=1, errors='ignore')  # Drop Rainfall, RISK_MM, RainTomorrow
y = df['Rainfall']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}\n")

# Feature Importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.show()

# Real-time Data Prediction
def predict_rainfall(data, model, columns):
    try:
        # Ensure the data is a dictionary
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary.")
        
        # Convert the dictionary to a DataFrame with a single row
        input_df = pd.DataFrame([data])
        
        # Ensure RainToday is present in the input data
        if 'RainToday' not in input_df.columns:
            raise ValueError("RainToday is required in the input data.")

        # Impute any missing values with the same method
        for col in input_df.columns:
            if input_df[col].dtype == 'object':
                input_df[col] = input_df[col].fillna(input_df[col].mode()[0])
            else:
                input_df[col] = input_df[col].fillna(input_df[col].median())

        # Convert RainToday and RainTomorrow to numerical (0 and 1)
        input_df['RainToday'] = input_df['RainToday'].map({'Yes': 1, 'No': 0})

        # One-hot encode categorical features
        categorical_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
        input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
        
        # Add any missing columns and fill with 0
        missing_cols = set(columns) - set(input_df.columns)
        for c in missing_cols:
            input_df[c] = 0
        # Ensure the order of column are the same
        input_df = input_df[columns]
        
        # Ensure that the input data does not have any NAs
        if input_df.isnull().any().any():
            raise ValueError("Input data contains missing values after preprocessing.")
        
        # Make the prediction
        prediction = model.predict(input_df)
        return prediction[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# Example Usage:
real_time_data = {
    'MinTemp': 12.0,
    'MaxTemp': 25.0,
    'Evaporation': 5.0,
    'Sunshine': 8.0,
    'WindGustDir': 'NW',
    'WindGustSpeed': 40.0,
    'WindDir9am': 'SW',
    'WindDir3pm': 'NW',
    'WindSpeed9am': 10.0,
    'WindSpeed3pm': 25.0,
    'Humidity9am': 70.0,
    'Humidity3pm': 30.0,
    'Pressure9am': 1019.7,
    'Pressure3pm': 1015.0,
    'Cloud9am': 5.0,
    'Cloud3pm': 6.0,
    'Temp9am': 14.4,
    'Temp3pm': 23.6,
    'RainToday': 'No'
}

# Ensure that X_train columns
X_train_columns = X_train.columns

# Predict rainfall
predicted_rainfall = predict_rainfall(real_time_data, model, X_train_columns)

if predicted_rainfall is not None:
    print(f"Predicted Rainfall: {predicted_rainfall:.2f}")
