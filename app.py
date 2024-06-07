#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# 加載模型和預處理器
rf_classifier = joblib.load('random_forest_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    new_data_df = pd.DataFrame([data])
    new_data_df = pd.get_dummies(new_data_df, columns=['Hard/Smart worker'], drop_first=True)
    
    for col in preprocessor.feature_names_in_:
        if col not in new_data_df.columns:
            new_data_df[col] = 0
    
    new_data_df = new_data_df[preprocessor.feature_names_in_]
    new_data_scaled = preprocessor.transform(new_data_df)
    
    prediction = rf_classifier.predict(new_data_scaled)
    return jsonify({'predicted_job_role': prediction[0]},message='CORS enabled')

if __name__ == '__main__':
    app.run(debug=True)

