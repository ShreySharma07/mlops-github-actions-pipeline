

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

#mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment('LogisticRegressionExperiment')

with mlflow.start_run() as run:
    try:
        data = pd.read_csv('data/raw_data.csv')
        X = data.drop('target', axis=1)
        y = data['target']
    except FileNotFoundError as e:
        print(f'Error: {e}')
        exit()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Define model parameters and log them with mlflow
    solver = 'liblinear'
    c = 10.0
    #multi_class = 'multinomial'
    random_state_model = 42

    mlflow.log_param('solver', solver)
    mlflow.log_param('C', c)
    mlflow.log_param('random state', random_state_model)

    #training the model
    model = LogisticRegression(solver=solver, C=c, random_state=random_state_model)
    model.fit(X_train, y_train)

    #Evaluating the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # ---- Logging metrics with MLflow ----
    mlflow.log_metric('accuracy', accuracy)
    print(f'Accuracy: {accuracy}')

    # ---- Logging model as artifact ----
    mlflow.sklearn.log_model(
        sk_model = model,
        name = "logistic-model",
        input_example = X_train,
        registered_model_name = 'LogisticRegressionModel')
    print(f'Model registered and logged with MLflow {mlflow.active_run().info.run_id}')

    # Link DVC data version to MLflow run (conceptually, actual DVC integration is more advanced)
    # For simplicity, we'll just log the DVC commit hash if available
    try:
        import subprocess
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        mlflow.set_tag('git_commit', git_commit)
        #getting dvc hash data
        # dvc_hash = subprocess.check_output(['dvc', 'diff', 'data/raw_data.csv', '--json']).decode('utf-8')
        # mlflow.set_tag('dvc_data_hash', dvc_hash)
    except Exception as e:
        print(f'Error loggging DVC commit: {e}')


# try:
#     data = pd.read_csv('data/raw_data.csv')
#     X = data.drop('target', axis=1)
#     y = data['target']
# except FileNotFoundError as e:
#     print(f'Error: {e}')
#     exit()
    
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #Define model parameters and log them with mlflow
# solver = 'liblinear'
# c = 1.0
# random_state_model = 42

# model = LogisticRegression(solver=solver, C=c, random_state=random_state_model)
# model.fit(X_train, y_train)

# predictions = model.predict(X_test)
# accuracy = accuracy_score(y_test, predictions)

# joblib.dump(model, 'model.pkl')