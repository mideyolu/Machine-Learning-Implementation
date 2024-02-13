from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your trained model
model = joblib.load('wine_quality_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extracting feature values from the form and converting them to a numpy array
        features = np.array([
            float(request.form['fixed_acidity']),
            float(request.form['volatile_acidity']),
            float(request.form['citric_acid']),
            float(request.form['residual_sugar']),
            float(request.form['chlorides']),
            float(request.form['free_sulfur_dioxide']),
            float(request.form['total_sulfur_dioxide']),
            float(request.form['density']),
            float(request.form['pH']),
            float(request.form['sulphates']),
            float(request.form['alcohol'])
        ])

        # Reshape the features to a 2D array (necessary for the model prediction)
        features = features.reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        if prediction[0] == 0:
            result = 'Bad quality Wine'
        else:
            result = 'Good quality Wine'

        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
