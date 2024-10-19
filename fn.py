from flask import Flask, request, jsonify
import joblib  # Or your preferred library for loading the model
import pandas as pd



from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


app = Flask(__name__)

# Load your trained model (make sure to provide the correct path)
model = joblib.load('fake_news_model.pkl')

# Create a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = data['text']  # Get the user input from the request

    # Preprocess the input text if necessary (like vectorization)
    # For example, if you're using a vectorizer
    # X = vectorizer.transform([user_input])

    # Make prediction
    prediction = model.predict([user_input])  # Adjust based on your input shape

    # Prepare the response
    return jsonify({'prediction': 'Real' if prediction[0] == 0 else 'Fake'})  # Adjust labels as necessary

if __name__ == '__main__':
    app.run(debug=True)
