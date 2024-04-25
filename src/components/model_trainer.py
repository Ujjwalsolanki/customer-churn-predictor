import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from logger import logging
from src.utils.common import FileOperations

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

from sklearn.model_selection import GridSearchCV

class ModelTrainer:

    def __init__(self) -> None:
        pass

    def initiate_model_training(self):
        try:
            models = {
                "LogisticRegression": LogisticRegression(),
                "DecisionTree": DecisionTreeClassifier(),
                "RandomForest": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "GradientBoost": GradientBoostingClassifier(),
                "LightGradientBoost": LGBMClassifier(),
                "XGBClassifier":  XGBClassifier()
            }

            file_op = FileOperations()
            params = file_op.read_yaml(Path("params.yaml"))

            #get train and test data
            X_train, X_test, y_train, y_test = self.get_data()

            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                logging.info("Model trainer {0}".format(str(model)))

                param=params[list(models.keys())[i]]

                # gs = GridSearchCV(model,param,n_jobs=-1)
                gs = GridSearchCV(model,param,cv=5, n_jobs=-1)
                gs.fit(X_train,y_train)
                logging.info(gs.best_params_)

                model.set_params(**gs.best_params_)
                
                model.fit(X_train, y_train)  # Train model

                y_pred = model.predict(X_test)

                test_model_score = accuracy_score(y_test, y_pred)

                report[list(models.keys())[i]] = test_model_score

                logging.info(test_model_score)

            return report


        except Exception as e:
            logging.exception(e)
            raise e

    def get_data(self):
        try:
            data = pd.read_csv(os.path.join("training_files/cleaned_data.csv"))
            X = data.drop(columns='Churn')
            y = data['Churn']

            logging.info("train test split started")

            return train_test_split(X, y, test_size=0.2, random_state=30)

        except Exception as e:
            logging.exception(e)
            raise e
