import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from preprocess import load_and_process_data
from utils import tune_random_forest

def sanitize_metric_name(name):
    return name.replace(" ", "_").replace("+", "plus")

def train_and_log_model(X, y, model, model_name, target_name):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Set up the MLflow experiment
    mlflow.set_experiment(f"{target_name}_prediction")

    # Start an MLflow run
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mse)

        # Log metrics to MLflow
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("rmse", rmse)

        # Log the model to MLflow
        mlflow.sklearn.log_model(model, model_name)
        mlflow.log_param("model_type", model_name)

        # If the model has feature importances, log them
        if hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_
            for i, feature_name in enumerate(X.columns):
                sanitized_name = sanitize_metric_name(feature_name)
                mlflow.log_metric(f"importance_{sanitized_name}", feature_importances[i])

        # Register the model
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/{model_name}", f"{model_name}_{target_name}_Model")

def main():
    filepath = 'apps/googleplaystore.csv'
    X, y_rating, y_installs = load_and_process_data(filepath)

    # Train and log models
    train_and_log_model(X, y_rating, LinearRegression(), "LinearRegression", "rating")
    tuned_forest_model_rating = tune_random_forest(X, y_rating)
    train_and_log_model(X, y_rating, tuned_forest_model_rating, "RandomForestRegressor", "rating")

    train_and_log_model(X, y_installs, LinearRegression(), "LinearRegression", "number_of_installs")
    tuned_forest_model_installs = tune_random_forest(X, y_installs)
    train_and_log_model(X, y_installs, tuned_forest_model_installs, "RandomForestRegressor", "number_of_installs")

if __name__ == "__main__":
    main()
