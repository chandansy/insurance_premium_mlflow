import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        file_path = file_path.replace("\\","/")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:  
            pickle.dump(obj, file_obj)
            logging.info(f'The object has been saved')
    except Exception as e:
        logging.info("An error has occurred while saving the object")
        raise CustomException(e, sys)


def evaluate_model(x_train,y_train,y_test,x_test,models,param):
    try:
        logging.info("Evalutating models")
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(x_train)

            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            logging.info(f'The {model} model has score of {test_model_score}')

        return report
    except Exception as e:
        logging.info("An error as occured while evaluating the model")
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            loaded_obj = pickle.load(file_obj)
            logging.info('Loaded the object')
            return loaded_obj
    except Exception as e:
        logging.info("An error as occured while loading the pickle object")
        raise CustomException(e,sys)