import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set up MLflow experiment
mlflow.set_experiment('RandomForest_Experiment0.0.0.5')

# Start MLflow run
with mlflow.start_run():

    # Load the dataset
    data = pd.read_csv('../../apps/123.csv')

    # One-hot encode the Android Ver column
    data = pd.get_dummies(data, columns=['Android Ver'])

    # Prepare the data
    data_predict_Ratings = data.drop(['App', 'Rating', 'Category', 'Last Updated'], axis=1)

    # Ensure all columns are numeric
    print(data_predict_Ratings.info())

    X = data_predict_Ratings.values
    y = data['Rating'].values

    print('x:', X.shape, 'y:', y.shape)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    # Initialize the Random Forest model
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model
    rf_reg.fit(X_train, y_train)

    # Predict
    y_pred = rf_reg.predict(X_test)
    print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))

    # Compute metrics
    r_squared = rf_reg.score(X_test, y_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Print the metrics
    print("R^2: {}".format(r_squared))
    print("RMSE: {}".format(rmse))

    # Log parameters, metrics, and model to MLflow
    mlflow.log_param('n_estimators', 100)
    mlflow.log_param('random_state', 42)
    mlflow.log_metric('r_squared', r_squared)
    mlflow.log_metric('rmse', rmse)

    # Log the model
    mlflow.sklearn.log_model(rf_reg, 'model')
