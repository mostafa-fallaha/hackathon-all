import pandas as pd

def predict_new_entries(model, new_data):
    # One-hot encode new data with the same columns used during training
    new_data = pd.get_dummies(new_data, columns=['Android Ver'])
    # Ensure the new data has the same columns as the training data
    model_columns = model.feature_importances_.shape[0]
    new_data = new_data.reindex(columns=range(model_columns), fill_value=0)
    return model.predict(new_data)
#    heeaadse