import pandas as pd
import numpy as np 

def load_data():
    df = pd.read_csv(f'data/data.csv')

    #split the data into training and testing sets
    df = df.sample(frac=1, random_state=42)
    split = int(len(df) * 0.8)

    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    #pandas dataframe -> numpy arrays
    X = train_df.iloc[:, 2:4].to_numpy()  
    Y = train_df.iloc[:, 1].map({
    "M": 1,
    "B": 0 
    }).to_numpy()

    X_test = test_df.iloc[:, 2:4].to_numpy()
    Y_test = test_df.iloc[:, 1].map({
    "M": 1,
    "B": 0 
    }).to_numpy()

    X_standardized, X_test_standardized, mean, std = standardize(X, X_test)
    #X_standardized = X
    #X_test_standardized = X_test

    return X_standardized, Y, X_test_standardized, Y_test, mean, std

def standardize(X, X_test):
    """
    Standardize the features in the training and testing sets.

    Parameters:
        X (np.ndarray): Training features.
        X_test (np.ndarray): Testing features.

    Returns:
        tuple: Standardized training and testing features.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_standardized = (X - mean) / (std + 1e-8)
    X_test_standardized = (X_test - mean) / (std + 1e-8)

    return X_standardized, X_test_standardized, mean, std
