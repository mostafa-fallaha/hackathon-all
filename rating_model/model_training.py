import mlflow
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Start MLflow run
    with mlflow.start_run():
        rf_reg.fit(X_train, y_train)
        y_pred = rf_reg.predict(X_test)
        
        r_squared = rf_reg.score(X_test, y_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        
        # Log parameters, metrics, and model to MLflow
        mlflow.log_param('n_estimators', 100)
        mlflow.log_param('random_state', 42)
        mlflow.log_metric('r_squared', r_squared)
        mlflow.log_metric('rmse', rmse)
        
        # Save model using pickle
        with open('random_forest_model.pkl', 'wb') as model_file:
            pickle.dump(rf_reg, model_file)
        
        print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))
        print("R^2: {}".format(r_squared))
        print("RMSE: {}".format(rmse))

    return rf_reg
