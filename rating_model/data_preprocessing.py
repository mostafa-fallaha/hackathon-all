import pandas as pd

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = pd.get_dummies(data, columns=['Android Ver'])
    X = data.drop(['App', 'Rating', 'Category', 'Last Updated'], axis=1).values
    y = data['Rating'].values
    return X, y
