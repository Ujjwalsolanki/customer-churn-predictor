import os
import pickle
import pandas as pd
from logger import logging

from src.components.data_preprocessing import DataPreprocessing


class Prediction:

    def __init__(self) -> None:
        pass

    def predict(self, data):
            try:
                logging.info("Prediction started with predict method")
                model_path=os.path.join("artifacts/models","model.sav")

                model = self.load_object(file_path=model_path)

                data_preprocessing = DataPreprocessing()

                cleaned_data = data_preprocessing.convert_categorical_to_numeric(data)
                pred = model.predict(cleaned_data)

                return pred
            
            except Exception as e:
                logging.exception(e)
                raise e
            

    def load_object(self, file_path):
        try:
            with open(file_path, "rb") as file_obj:
                return pickle.load(file_obj)

        except Exception as e:
            raise e

class CustomData:
    def __init__(
        self,
        Gender,
        SeniorCitizen,
        Partner,
        Dependents,
        Tenure:int,
        PhoneService,
        MultipleLines,
        InternetService,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection,
        TechSupport,
        StreamingTV,
        StreamingMovies,
        Contract,
        PaperlessBilling,
        PaymentMethod,
        TotalCharges
    ):
        self.Gender = Gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.Tenure = Tenure
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.TotalCharges = TotalCharges

    def get_data_as_data_frame(self):
        try:
            logging.info("data converted in data frame")
            custom_data_input_dict = {
                "Gender": [self.Gender],
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "Tenure": [self.Tenure],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "TotalCharges": [self.TotalCharges]
            }
            logging.exception(custom_data_input_dict)
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise e
