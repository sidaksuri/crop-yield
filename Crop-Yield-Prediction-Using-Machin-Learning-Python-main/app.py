from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn

# Print sklearn version (for debugging)
print(sklearn.__version__)

# Load trained model and preprocessor
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        # ---- Numerical inputs ----
        Year = int(request.form['Year'])
        average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
        pesticides_tonnes = float(request.form['pesticides_tonnes'])
        avg_temp = float(request.form['avg_temp'])

        # ---- Categorical inputs (normalize) ----
        Area = request.form['Area'].strip().title()
        Item = request.form['Item'].strip().title()

        # ---- Create feature array ----
        features = np.array(
            [[
                Year,
                average_rain_fall_mm_per_year,
                pesticides_tonnes,
                avg_temp,
                Area,
                Item
            ]],
            dtype=object
        )

        # ---- Transform & Predict ----
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features)[0]

        # ---- Render page with inputs + prediction ----
        return render_template(
            'index.html',
            prediction=round(prediction, 2),
            Year=Year,
            average_rain_fall_mm_per_year=average_rain_fall_mm_per_year,
            pesticides_tonnes=pesticides_tonnes,
            avg_temp=avg_temp,
            Area=Area,
            Item=Item
        )

if __name__ == "__main__":
    app.run(debug=True)
