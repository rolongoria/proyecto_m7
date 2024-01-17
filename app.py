from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS for the application
cors = CORS(app)

# Load the trained KNeighborsClassifier
# knn_clf = joblib.load('models/knn_clf.pkl')
rf_regressor = joblib.load('models/rf_regressor.pkl')

# Endpoint for receiving predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        print(data)

        # Convert the JSON data to a DataFrame
        new_data_point = pd.DataFrame([data])
        print(new_data_point)

        # Make Prediction
        # prediction = knn_clf.predict(new_data_point)
        prediction = rf_regressor.predict(new_data_point)

        # Return the prediction as JSON response
        print(prediction)
        return jsonify({'prediction': prediction.tolist()}), 200
        # return jsonify({'prediction': prediction}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
