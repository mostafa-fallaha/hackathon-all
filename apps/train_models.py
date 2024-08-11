import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

def load_and_process_data(filepath):
    # Load data
    df = pd.read_csv(filepath)

    # Drop missing values (as an example of preprocessing)
    df = df.dropna()

    # Feature Engineering: Use selected features and potentially create new ones
    df['log_installs'] = np.log1p(df['Installs'])  # Log-transform the installs to handle skewness

    # Encoding categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Selecting Features for Rating and Installs
    feature_columns = ['Reviews', 'Size', 'Price', 'last_updated_year', 'last_updated_month', 'Category_encoded', 'Type_Free', 'Type_Paid']
    X = df[feature_columns]

# dekete
    y_installs = df['log_installs']


# 

    return X, y_installs

def train_and_log_model(X, y, model, model_name, target_name):
    print("Splitting the data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split complete: {len(X_train)} training samples and {len(X_test)} test samples.")

    # Scaling the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Setting up MLflow experiment...")
    mlflow.set_experiment(f"{target_name}_prediction")

    with mlflow.start_run(run_name=model_name):
        print(f"Training {model_name} model...")
        model.fit(X_train, y_train)
        print(f"{model_name} model training complete.")

        predictions = model.predict(X_test)
        print("Predictions complete.")

        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mse)

        print(f"MSE: {mse}, MAE: {mae}, R2: {r2}, RMSE: {rmse}")

        print("Logging metrics and model to MLflow...")
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(model, model_name, pip_requirements="requirements.txt")
        mlflow.log_param("model_type", model_name)

        if hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_
            for i, feature_name in enumerate(X.columns):
                mlflow.log_metric(f"importance_{feature_name}", feature_importances[i])

        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/{model_name}", f"{model_name}_{target_name}_Model")
        print("Model registered in MLflow.")

def tune_random_forest(X, y):
    print("Tuning Random Forest model...")
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    print(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the CSV file
    filepath = os.path.join(current_dir, 'googleplaystore.csv')

    print("Loading and processing data...")
    X, y_installs = load_and_process_data(filepath)
    print("Data loading and processing complete.")

    print("Training and logging Linear Regression model for rating prediction...")
    linear_model_rating = LinearRegression()
    train_and_log_model(X, y_rating, linear_model_rating, "LinearRegression", "rating")

    print("Training and logging Random Forest model for rating prediction...")
    tuned_forest_model_rating = tune_random_forest(X, y_rating)
    train_and_log_model(X, y_rating, tuned_forest_model_rating, "RandomForestRegressor", "rating")

    print("Training and logging Linear Regression model for installs prediction...")
    linear_model_installs = LinearRegression()
    train_and_log_model(X, y_installs, linear_model_installs, "LinearRegression", "number_of_installs")

    print("Training and logging Random Forest model for installs prediction...")
    tuned_forest_model_installs = tune_random_forest(X, y_installs)
    train_and_log_model(X, y_installs, tuned_forest_model_installs, "RandomForestRegressor", "number_of_installs")
