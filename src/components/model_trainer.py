import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from pathlib import Path
from logger import logging
from src.utils.common import FileOperations

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

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

            best_model_score = max(sorted(report.values()))

            ## To get best model name from dict

            best_model_name = list(report.keys())[
                list(report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise Exception("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset")
            logging.info("Best model: {0}".format(best_model_name))

            file_path = os.path.join('artifacts/models','model.sav')
            file_name = ''


            logging.info('saving model')
            with open(file_path, "wb") as file_obj:
                pickle.dump(best_model, file_obj)

            predicted=best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            logging.info("accuracy {0}".format(accuracy))
            return accuracy


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
