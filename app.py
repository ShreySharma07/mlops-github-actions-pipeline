import joblib
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import logging
import datetime

app = Flask(__name__)
LOG_FILE = 'app.log'

#--- configuring logging ---
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    handlers = [
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

MODEL_PATH = 'model.pkl'

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f'Error model not found on the given path: {MODEL_PATH}')
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json(force=True)
        features = pd.DataFrame(data['features'], columns=[str(i) for i in range(10)])
        predictions = model.predict(features).tolist()
        prediction_proba = model.predict_proba(features).tolist()
        return jsonify({
            'predictions':predictions,
            'prediction_proba':prediction_proba
        })
    except Exception as e:
        return jsonify({'error':str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    if model is not None:
        return jsonify({'status': 'healthy'}), 200
    else:
        return jsonify({'status': 'unhealthy'}), 500

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000, debug = True)