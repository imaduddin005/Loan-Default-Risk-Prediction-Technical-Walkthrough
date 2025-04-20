from flask import Flask, render_template_string, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your trained VotingClassifier model
model_path = "C:/Users/PT/Downloads/credit_card_default/best_model_10_features.pkl"
model = joblib.load(model_path)

# HTML + CSS (basic, embedded in the same file)
template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Credit Default Risk Predictor</title>
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            font-family: 'Arial', sans-serif;
            background-color: #2c3e50;
            color: #ecf0f1;
            margin: 0;
            padding: 0;
        }
        .container {
            background-color: #34495e;
            max-width: 700px;
            margin: 50px auto;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        }
        h2 {
            text-align: center;
            font-size: 32px;
            margin-bottom: 30px;
        }
        label {
            display: block;
            font-size: 16px;
            margin-bottom: 8px;
            color: #ecf0f1;
        }
        input[type="number"] {
            width: 100%;
            padding: 15px;
            margin: 10px 0 20px;
            border-radius: 8px;
            border: 1px solid #7f8c8d;
            background-color: #34495e;
            color: #ecf0f1;
            font-size: 16px;
        }
        input[type="submit"] {
            width: 100%;
            padding: 15px;
            background-color: #3498db;
            color: white;
            font-size: 18px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        input[type="submit"]:hover {
            background-color: #2980b9;
        }
        .result {
            text-align: center;
            font-size: 20px;
            margin-top: 30px;
            color: #2ecc71;
        }
        footer {
            text-align: center;
            margin-top: 50px;
            color: #bdc3c7;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Credit Default Risk Prediction</h2>
        <form method="POST">
            {% for feature in feature_names %}
                <label for="{{ feature }}">{{ feature.replace('_', ' ').title() }}:</label>
                <input type="number" step="any" name="{{ feature }}" required>
            {% endfor %}
            <input type="submit" value="Predict">
        </form>
        {% if prediction %}
            <div class="result">
                <p><strong>Probability of Default:</strong> {{ prediction }}%</p>
            </div>
        {% endif %}
    </div>
    <footer>
        <p>&copy; 2025 Credit Risk Prediction. All rights reserved.</p>
    </footer>
</body>
</html>
"""

# Load feature names from training (update this list to match your model input)
feature_names = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
    'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'CNT_FAM_MEMBERS'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            input_values = [float(request.form[f]) for f in feature_names]
            input_array = np.array(input_values).reshape(1, -1)
            probability = model.predict_proba(input_array)[0][1]
            prediction = round(probability * 100, 2)
        except Exception as e:
            prediction = f"Error: {str(e)}"
    
    return render_template_string(template, feature_names=feature_names, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
