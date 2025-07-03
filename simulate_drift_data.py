import numpy as np
import pandas as pd
from generate_data import generate_data

X, y = generate_data()


#introducing drift in a couple of features
X_drifted = X.copy()
X_drifted[:, 0] += np.random.normal(loc = 0.7, scale = 0.1, size = X.shape[0]) 
X_drifted[:, 1] += np.random.normal(loc = 0.3, scale = 0.05, size = X.shape[0])

df_X = pd.DataFrame(X_drifted)
df_y = pd.DataFrame(y, columns = ['target'])

df_production = pd.concat([df_X, df_y], axis = 1)

df_production.to_csv('production_data.csv', index = False)

