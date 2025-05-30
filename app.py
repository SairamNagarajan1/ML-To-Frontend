from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("iris_model.pkl")

@app.route('/')
def home():
    return "Welcome to the Iris Classifier API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data['features']
    prediction = model.predict([features])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
