import pandas as pd
import numpy as np

def load_and_process_data(filepath):
    # Load data
    df = pd.read_csv(filepath)

    # Drop rows with missing values
    df = df.dropna()

    # Encode Android version to a numeric format
    def parse_android_version(version):
        try:
            return float(version.split()[0])
        except (ValueError, AttributeError):
            return np.nan

    df['Android_Ver'] = df['Android Ver'].apply(parse_android_version)
    df = df.dropna(subset=['Android_Ver'])

    # Log-transform installs to handle skewness
    def parse_installs(installs):
        if isinstance(installs, str):
            return int(installs.replace('+', '').replace(',', ''))
        elif isinstance(installs, int):
            return installs
        else:
            return np.nan

    df['log_installs'] = np.log1p(df['Installs'].apply(parse_installs))
    df = df.dropna(subset=['log_installs'])

    # Get dummies for categorical variables
    df = pd.get_dummies(df, columns=[
        'Content Rating_Adults only 18+',
        'Content Rating_Everyone',
        'Content Rating_Everyone 10+',
        'Content Rating_Mature 17+',
        'Content Rating_Teen',
        'Content Rating_Unrated'
    ], drop_first=True)

    # Selected features based on your JSON input
    feature_columns = [
        'Android_Ver', 'Size', 'Price', 'Category_encoded',
        'Type_Free', 'Type_Paid', 'last_updated_year', 'last_updated_month',
        'Content Rating_Adults only 18+',
        'Content Rating_Everyone',
        'Content Rating_Everyone 10+',
        'Content Rating_Mature 17+',
        'Content Rating_Teen',
        'Content Rating_Unrated'
    ]

    # Ensure all required columns exist in the dataframe
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_columns]
    y_rating = df['Rating']
    y_installs = df['log_installs']

    return X, y_rating, y_installs
