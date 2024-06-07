from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)

try:
    rf_classifier = joblib.load('random_forest_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
except Exception as e:
    logging.error(f"Error loading model or preprocessor: {e}")
    raise e

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  
        logging.info(f"Received data: {data}")

     
        if 'Hard/Smart worker' not in data:
            return jsonify({'error': "'Hard/Smart worker' field is missing"}), 400

        new_data_df = pd.DataFrame([data])
        new_data_df = pd.get_dummies(new_data_df, columns=['Hard/Smart worker'], drop_first=True)
        
        for col in preprocessor.feature_names_in_:
            if col not in new_data_df.columns:
                new_data_df[col] = 0
        
        new_data_df = new_data_df[preprocessor.feature_names_in_]
        new_data_scaled = preprocessor.transform(new_data_df)
        
        prediction = rf_classifier.predict(new_data_scaled)
        logging.info(f"Prediction: {prediction[0]}")

        return jsonify({'predicted_job_role': prediction[0]})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
