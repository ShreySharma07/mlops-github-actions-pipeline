

import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature

# Use relative paths that work in both local and CI environments
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment('LogisticRegressionExperiment')

with mlflow.start_run() as run:
    try:
        # Use relative path that works in GitHub Actions
        data_path = os.path.join('data', 'raw_data.csv')
        data = pd.read_csv(data_path)
        X = data.drop('target', axis=1)
        y = data['target']
    except FileNotFoundError as e:
        print(f'Error: {e}')
        print(f'Current working directory: {os.getcwd()}')
        print(f'Files in current directory: {os.listdir(".")}')
        exit(1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.astype('float64')
    X_test = X_test.astype('float64')
    
    # Convert target to float64 to avoid MLflow schema warnings
    y_train = y_train.astype('float64')
    y_test = y_test.astype('float64')
    
    # Define model parameters and log them with mlflow
    solver = 'lbfgs'
    c = 10.0
    random_state_model = 42
    
    mlflow.log_param('solver', solver)
    mlflow.log_param('C', c)
    mlflow.log_param('random_state', random_state_model)
    
    # Training the model
    model = LogisticRegression(solver=solver, C=c, random_state=random_state_model)
    model.fit(X_train, y_train)
    
    # Evaluating the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # ---- Logging metrics with MLflow ----
    mlflow.log_metric('accuracy', accuracy)
    print(f'Accuracy: {accuracy}')
    
    # Create signature with float64 types to avoid schema warnings
    signature = infer_signature(X_train, predictions.astype('float64'))
    
    # ---- Logging model as artifact ----
    mlflow.sklearn.log_model(
        sk_model=model,
        name="logistic-model",
        input_example=X_train.head(5),
        signature=signature
    )
    print(f'Model registered and logged with MLflow {mlflow.active_run().info.run_id}')
    
    # Link Git commit to MLflow run (safer error handling for CI)
    try:
        import subprocess
        # Check if we're in a git repository
        if os.path.exists('.git'):
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                stderr=subprocess.DEVNULL
            ).strip().decode('utf-8')
            mlflow.set_tag('git_commit', git_commit)
        else:
            print('Not in a git repository, skipping git commit logging')
    except Exception as e:
        print(f'Error logging git commit: {e}')
        # Don't exit on git errors in CI environment

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