import logging
from flask import Flask, request, render_template
import pandas as pd

from src.pipeline.predict_pipeline import train_and_save_model, predict_new_data
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.logger import logger  

application = Flask(__name__)
app = application

## Route for home page
@app.route('/')
def index():
    logger.info("Home page accessed")
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        logger.info("GET request received at /predictdata")
        return render_template('home.html')
    else:
        try:
            logger.info("POST request received at /predictdata")

            # Collect user input from form
            user_input = {
                "gender": request.form.get('gender'),
                "race_ethnicity": request.form.get('ethnicity'),
                "parental_level_of_education": request.form.get('parental_level_of_education'),
                "lunch": request.form.get('lunch'),
                "test_preparation_course": request.form.get('test_preparation_course'),
                "reading_score": float(request.form.get('reading_score')),
                "writing_score": float(request.form.get('writing_score'))
            }

            # Convert to DataFrame
            pred_df = pd.DataFrame([user_input])
            logger.info(f"Input DataFrame: \n{pred_df}")

            # Get prediction
            prediction = predict_new_data(pred_df)
            logger.info(f"Prediction result: {prediction[0]}")

            return render_template('home.html', results=prediction[0])

        except Exception as e:
            logger.error("Error during prediction", exc_info=True)
            return render_template('home.html', results="Error: Could not generate prediction")

if __name__ == "__main__":
    logger.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=8080)
