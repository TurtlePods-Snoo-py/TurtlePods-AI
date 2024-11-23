from flask import Flask, jsonify, request
import random

app = Flask(__name__)

@app.route('/predict_stress', methods=['POST'])
def predict_stress():
    stress_level = round(random.uniform(2, 6), 2)
    response = {
        "stress_level": stress_level
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
