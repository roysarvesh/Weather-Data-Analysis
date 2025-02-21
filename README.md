# Weather Data Analyis Using Random Forest

## Overview
This project utilizes a Random Forest Regressor to predict rainfall based on historical weather data. The dataset is preprocessed to handle missing values, encode categorical variables, and generate useful features. A machine learning model is then trained and evaluated to make accurate predictions.

## Dataset
The project uses weather data stored in `weather.csv`. Ensure this file is placed in the correct directory before running the script.

### Features Used:
- Temperature (MinTemp, MaxTemp, Temp9am, Temp3pm)
- Humidity (Humidity9am, Humidity3pm)
- Wind (WindSpeed9am, WindSpeed3pm, WindGustSpeed, WindGustDir, WindDir9am, WindDir3pm)
- Atmospheric Pressure (Pressure9am, Pressure3pm)
- Cloud Cover (Cloud9am, Cloud3pm)
- Sunshine, Evaporation
- RainToday (Yes/No)

### Target Variable:
- Rainfall (amount of rain predicted)

## Installation and Dependencies
Ensure you have the following dependencies installed:

```bash
pip install pandas matplotlib seaborn scikit-learn numpy
```

## Running the Project
1. Place `weather.csv` in the project directory.
2. Run the script using:

```bash
python weather_prediction.py
```

The script will:
- Load and preprocess data
- Train a `RandomForestRegressor`
- Evaluate model performance
- Display feature importance
- Provide real-time predictions for new weather data

## Data Preprocessing
- Missing categorical values are replaced with the mode.
- Missing numerical values are replaced with the median.
- Categorical variables (`WindGustDir`, `WindDir9am`, `WindDir3pm`) are one-hot encoded.

## Model Training and Evaluation
- Data is split into training and testing sets using `train_test_split`.
- A `RandomForestRegressor` with 100 estimators is trained on the dataset.
- Performance is measured using:
  - Mean Squared Error (MSE)
  - R-squared (R²) Score
- Feature importance is visualized using a bar chart.

## Prediction Function
The `predict_rainfall` function allows real-time rainfall predictions based on user-input weather data.

### Example Input:
```python
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
```
### Example Output:
```
Predicted Rainfall: 2.45 mm
```

## Visualization
The script generates:
- A correlation heatmap to show relationships between features.
- A bar chart showing the top 10 most important features in predicting rainfall.

## Project Repository
The complete project can be found on GitHub: [Weather Data Analysis](https://github.com/roysarvesh/Weather-Data-Analysis.git)

## License
This project is open-source and available for modification and improvement.

## Author
Sarvesh Kumar Roy

