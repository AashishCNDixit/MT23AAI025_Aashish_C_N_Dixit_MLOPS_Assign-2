from flask import Flask, jsonify, request
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('iris_model.pkl')

@app.route('/')
def home():
    return "Welcome to the Iris Classifier API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.get_json(force=True)
        
        # Extract the features for prediction
        features = [data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]
        
        # Make prediction
        prediction = model.predict([features])
        
        # Return the result as a JSON response
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
