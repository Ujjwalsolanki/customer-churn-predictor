import sys
from flask import Flask, request, render_template

from logger import logging

from src.components.prediction import Prediction, CustomData

application = Flask(__name__)
app = application

# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
# @app.route('/post', methods=['POST'])
def predict_data():
    try:

        if request.method == 'GET':
            return render_template('index.html')
        else:
            logging.info("prediction post method called")

            print("Gender="+request.form.get('Gender'))
            print("SeniorCitizen="+request.form.get('SeniorCitizen'))
            print("Partner="+request.form.get('Partner'))
            print("Dependents="+request.form.get('Dependents'))
            print("Tenure="+request.form.get('Tenure'))
            print("PhoneService="+request.form.get('PhoneService'))
            print("MultipleLines="+request.form.get('MultipleLines'))
            print("InternetService="+request.form.get('InternetService'))
            print("OnlineSecurity="+request.form.get('OnlineSecurity'))
            print("OnlineBackup="+request.form.get('OnlineBackup'))
            print("DeviceProtection="+request.form.get('DeviceProtection'))
            print("TechSupport="+request.form.get('TechSupport'))
            print("StreamingTV="+request.form.get('StreamingTV'))
            print("StreamingMovies="+request.form.get('StreamingMovies'))
            print("Contract="+request.form.get('Contract'))
            print("PaperlessBilling="+request.form.get('PaperlessBilling'))
            print("PaymentMethod="+request.form.get('PaymentMethod'))
            print("TotalCharges ="+request.form.get('TotalCharges'))

            data=CustomData(
                Gender = request.form.get('Gender'),
                SeniorCitizen = request.form.get('SeniorCitizen'),
                Partner = request.form.get('Partner'),
                Dependents = request.form.get('Dependents'),
                Tenure = request.form.get('Tenure'),
                PhoneService = request.form.get('PhoneService'),
                MultipleLines = request.form.get('MultipleLines'),
                InternetService = request.form.get('InternetService'),
                OnlineSecurity = request.form.get('OnlineSecurity'),
                OnlineBackup = request.form.get('OnlineBackup'),
                DeviceProtection = request.form.get('DeviceProtection'),
                TechSupport = request.form.get('TechSupport'),
                StreamingTV = request.form.get('StreamingTV'),
                StreamingMovies = request.form.get('StreamingMovies'),
                Contract = request.form.get('Contract'),
                PaperlessBilling = request.form.get('PaperlessBilling'),
                PaymentMethod = request.form.get('PaymentMethod'),
                TotalCharges = request.form.get('TotalCharges')
            )
            pred_df=data.get_data_as_data_frame()

            logging.info("Before Prediction")
            logging.info(str(pred_df))

            prediction=Prediction()
            
            results=prediction.predict(pred_df)
            logging.info("Prediction: {0}".format(results))
            
            return render_template('index.html',results=results[0])

    except Exception as e:
        logging.exception(e)
        raise e


if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)