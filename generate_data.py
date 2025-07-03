import numpy as np
import pandas as pd

def generate_data(num_samples = 1000, num_features = 10, num_classes = 3):

    X = np.random.rand(num_samples, num_features)
    y = np.random.randint(0, num_classes, size = num_samples).astype('float64')

    return X, y

def save_data(X, y, filename_X = 'X.csv', filename_y = 'y.csv'):
    df_X = pd.DataFrame(X).astype('float64')
    df_y = pd.DataFrame(y, columns=['target'])

    raw_data = pd.concat([df_X, df_y], axis=1)
    raw_data.to_csv('data/raw_data.csv', index=False)

if __name__ == "__main__":
    X, y = generate_data()
    save_data(X, y)
    