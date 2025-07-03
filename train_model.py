
import mlflow
import mlflow.sklearn
import pandas as pd
import os
import sys
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Suppress MLflow warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    """Main training function."""
    print("Starting MLflow training pipeline...")
    
    # Get current working directory
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Create mlruns directory in current workspace
    mlruns_dir = os.path.join(current_dir, 'mlruns')
    os.makedirs(mlruns_dir, exist_ok=True)
    
    # Set MLflow tracking URI to local directory
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")
    
    # Set experiment
    try:
        mlflow.set_experiment('LogisticRegressionExperiment')
    except Exception as e:
        print(f"Error setting experiment: {e}")
        # Create experiment if it doesn't exist
        mlflow.create_experiment('LogisticRegressionExperiment')
        mlflow.set_experiment('LogisticRegressionExperiment')
    
    print(f"MLflow tracking URI: file:{mlruns_dir}")
    
    with mlflow.start_run():
        try:
            # Load data
            data_path = os.path.join('data', 'raw_data.csv')
            
            if not os.path.exists(data_path):
                print(f'Error: Data file not found at {data_path}')
                print(f'Current directory contents: {os.listdir(".")}')
                if os.path.exists('data'):
                    print(f'Data directory contents: {os.listdir("data")}')
                sys.exit(1)
            
            print(f'Loading data from {data_path}')
            data = pd.read_csv(data_path)
            print(f'Data loaded successfully. Shape: {data.shape}')
            
            X = data.drop('target', axis=1)
            y = data['target']
            
        except Exception as e:
            print(f'Error loading data: {e}')
            sys.exit(1)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Convert to float64 to avoid schema warnings
        X_train = X_train.astype('float64')
        X_test = X_test.astype('float64')
        y_train = y_train.astype('float64')
        y_test = y_test.astype('float64')
        
        print(f'Training set shape: {X_train.shape}')
        print(f'Test set shape: {X_test.shape}')
        
        # Define model parameters
        solver = 'lbfgs'
        c = 10.0
        random_state_model = 42
        
        # Log parameters
        mlflow.log_param('solver', solver)
        mlflow.log_param('C', c)
        mlflow.log_param('random_state', random_state_model)
        mlflow.log_param('test_size', 0.2)
        
        # Training the model
        print("Training logistic regression model...")
        model = LogisticRegression(
            solver=solver, 
            C=c, 
            random_state=random_state_model,
            max_iter=1000
        )
        model.fit(X_train, y_train)
        
        # Evaluating the model
        print("Evaluating model...")
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Log metrics
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('train_samples', len(X_train))
        mlflow.log_metric('test_samples', len(X_test))
        
        print(f'Accuracy: {accuracy:.4f}')
        
        # Save model using joblib (simpler approach)
        try:
            import joblib
            model_path = os.path.join(mlruns_dir, 'model.pkl')
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path, 'model')
            print(f'Model saved to {model_path}')
        except Exception as e:
            print(f'Error saving model: {e}')
            # Try MLflow's built-in model logging as fallback
            try:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="logistic-model"
                )
                print("Model logged with MLflow sklearn")
            except Exception as e2:
                print(f'Error with MLflow model logging: {e2}')
        
        print("Training completed successfully!")
        return accuracy

if __name__ == "__main__":
    main()
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