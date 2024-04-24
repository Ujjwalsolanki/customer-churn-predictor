import os
import pandas as pd
import numpy as np
from logger import logging
from pathlib import Path

class DataPreprocessing:
    
    def __init__(self) -> None:
        pass

    def initiate_data_preprocessing(self):
        try:
            logging.info('data ingestion intiated')
            df = pd.read_csv(Path('training_files/data.csv'))
            df.dropna(how='any', inplace=True)
            
            column_names=['customerID','MonthlyCharges']

            df = self.drop_unwanted_columns(df, column_names)

            df = self.convert_categorical_to_numeric(df)

            df.to_csv(os.path.join('training_files/')+'cleaned_data.csv')
        except Exception as e:
            logging.exception(e)
            
    def drop_unwanted_columns(self, df, column_names):
        df = df.drop(columns=column_names)
        return df

    def convert_categorical_to_numeric(self, df):
        try:
            logging.info('converting categorical data to numeric')
            df.TotalCharges = pd.to_numeric(df.TotalCharges, errors="coerce")
            df.dropna(how='any', inplace=True)

            df.Churn = np.where(df.Churn == 'Yes', 1,0)

            df.gender = np.where(df.gender == 'Female', 1,0)
            df.Partner = np.where(df.Partner == 'Yes', 1,0)
            df.Dependents = np.where(df.Dependents == 'Yes', 1,0)
            df.tenure = df.tenure.apply(lambda x:x//12)
            df.PhoneService = np.where(df.Dependents == 'Yes', 1,0)
            df.MultipleLines = np.where(df.MultipleLines == 'Yes', 1,
                                        (np.where(df.MultipleLines == 'No', 0,2)))
            df.InternetService = np.where(df.InternetService == 'No', 0,
                                        (np.where(df.InternetService == 'DSL', 1,2)))
            df.OnlineSecurity = np.where(df.OnlineSecurity == 'Yes', 1,0)
            df.OnlineBackup = np.where(df.OnlineBackup == 'Yes', 1,0)
            df.DeviceProtection = np.where(df.DeviceProtection == 'Yes', 1,0)
            df.TechSupport = np.where(df.TechSupport == 'Yes', 1,0)
            df.StreamingTV = np.where(df.StreamingTV == 'Yes', 1,0)
            df.StreamingMovies = np.where(df.StreamingMovies == 'Yes', 1,0)
            df.Contract = np.where(df.Contract == 'Month-to-month', 0,
                                        (np.where(df.InternetService == 'One year', 1,2)))
            df.PaperlessBilling = np.where(df.PaperlessBilling == 'Yes', 1,0)
            df.PaymentMethod = np.where(df.PaymentMethod == 'Electronic check', 0,
                                        (np.where(df.InternetService == 'Mailed check', 1,
                                                (np.where(df.InternetService == 'Bank transfer (automatic)', 2,3)))))
            return df
        except Exception as e:
            logging.exception(e)