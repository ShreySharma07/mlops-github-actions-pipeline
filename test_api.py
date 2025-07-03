import requests
import json

url = "http://localhost:5050/predict"
# Sample data matching your model's input features (e.g., 5 features)
data = {"features": [[0.1, 0.3, 0.1, 0.4, 0.99, 0.6, 0.7, 0.8, 0.9, 0.8]]} # Single prediction
# data = {"features": [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]]} # Multiple predictions

headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, data=json.dumps(data), headers=headers)
    response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
    print("Response JSON:", response.json())
except requests.exceptions.RequestException as e:
    print(f"Error making request: {e}")
    if response:
        print("Response Status Code:", response.status_code)
        print("Response Body:", response.text)