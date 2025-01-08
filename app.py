from flask import Flask, request, jsonify
import numpy as np
import joblib

model = joblib.load('stress_model.pkl')
scaler = joblib.load('scaler.pkl')

def cal_pain(rolling, pitching):
    threshold = 0.3
    BL = np.sqrt(rolling ** 2 + pitching ** 2)
    turtle = abs(pitching) > threshold and abs(rolling) > threshold

    if turtle:
        pitching_score = max(0, (abs(pitching) - threshold) / (0.9 - threshold))
        rolling_score = max(0, (abs(rolling) - threshold) / (0.9 - threshold))
        weighted_pain = (pitching_score * 0.7 + rolling_score * 0.3)
        normalized_BL = BL / (np.sqrt(0.92 + 0.92))
        pain_level = (weighted_pain + normalized_BL) * 5
    else:
        pain_level = 0

    return round(pain_level, 1)

def cal_bmi(weight, height):
    return weight / ((height / 100) ** 2)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    rolling = data['rolling'][0]
    pitching = data['pitching'][0]
    weight = data['weight']
    height = data['height']
    
    pain = cal_pain(rolling, pitching)
    bmi = cal_bmi(weight, height)
    
    input_data = np.array([[pain, bmi]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0]

    return jsonify({"stress_level": round(prediction, 2)})

if __name__ == '__main__':
    app.run(host='0.0.0.0')