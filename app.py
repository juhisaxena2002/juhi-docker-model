from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model from the .pkl file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Welcome to the ML Model API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the POST request
    data = request.get_json()
    
    # Make a prediction using the model
    prediction = model.predict([np.array(data['features'])])
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=800)