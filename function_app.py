import logging
from flask import Flask, request, jsonify
import joblib
import numpy as np
import azure.functions as func

# Flask 앱 생성
app = Flask(__name__)

# 모델 및 스케일러 로드
model = joblib.load("stress_model.pkl")
scaler = joblib.load("scaler.pkl")

def expand_to_60(data):
    if len(data) < 60:
        data = data + [data[-1]] * (60 - len(data))
    return data

def predict_stress(rolling_data, pitching_data):
    rolling_mean_change = np.mean(np.abs(np.diff(rolling_data)))
    rolling_std = np.std(rolling_data)
    pitching_mean_change = np.mean(np.abs(np.diff(pitching_data)))
    pitching_std = np.std(pitching_data)
    
    input_features = np.array([[rolling_mean_change, rolling_std, pitching_mean_change, pitching_std]])
    input_scaled = scaler.transform(input_features)
    stress_prediction = model.predict(input_scaled)
    return stress_prediction[0]

@app.route('/predict_stress', methods=['POST'])
def predict_stress_endpoint():
    try:
        data = request.get_json()
        rolling = data.get("rolling")
        pitching = data.get("pitching")
        
        if not rolling or not pitching:
            return jsonify({"error": "Missing rolling or pitching data."}), 400
        
        rolling = expand_to_60(rolling)
        pitching = expand_to_60(pitching)
        
        stress_level = predict_stress(rolling, pitching)
        return jsonify({"stress_level": round(stress_level, 2)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Azure Functions용 엔트리 포인트
def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    from azure.functions._http_wsgi import WsgiMiddleware
    return WsgiMiddleware(app).handle(req, context)
