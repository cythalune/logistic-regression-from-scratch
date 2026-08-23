import pandas as pd

def load_data():
    df = pd.read_csv(f'data/data.csv')

    #split the data into training and testing sets
    df = df.sample(frac=1, random_state=42)
    split = int(len(df) * 0.8)

    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    #pandas dataframe -> numpy arrays
    X = train_df.iloc[:, 2:4].to_numpy()  
    Y = train_df.iloc[:, 1].to_numpy() 

    X_test = test_df.iloc[:, 2:4].to_numpy()
    Y_test = test_df.iloc[:, 1].to_numpy()

    return X, Y, X_test, Y_test

