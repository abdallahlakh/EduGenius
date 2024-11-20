from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('framework_model.pkl')

from flask import Flask, request, jsonify
import numpy as np
import joblib

# Load the trained model
model = joblib.load('framework_model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the JSON request
        data = request.json
        # Ensure the input is properly structured
        input_data = np.array(data['features']).reshape(1, -1)
        
        # Perform prediction
        prediction = model.predict(input_data)
        
        # Return the prediction as a JSON response
        response = {
            "prediction": prediction.tolist()
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True,port=5001)


#$headers = @{"Content-Type" = "application/json"}; $body = '{"features": [3, 5, 20, 15, 10, 1, 0, 1, 1, 1]}'; Invoke-WebRequest -Uri http://127.0.0.1:5001/predict -Method POST -Headers $headers -Body $body
