from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

app = Flask(__name__)

# Train a simple linear regression model
def train_model():
    # Example data (X: feature, y: target variable)
    data = {
        'X': [1, 2, 3, 4, 5],
        'y': [1, 2, 3, 4, 5]
    }

    df = pd.DataFrame(data)
    X = df[['X']]  # Feature
    y = df['y']    # Target

    model = LinearRegression()
    model.fit(X, y)
    
    return model

# Create a global model instance
model = train_model()

@app.route('/')
def home():
    return "Linear Regression Flask API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json()  # {"feature": value}
        
        feature_value = data['feature']
        
        # Make prediction
        prediction = model.predict(np.array([[feature_value]]))
        
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
