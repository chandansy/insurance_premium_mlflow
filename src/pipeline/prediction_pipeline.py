import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor_path = preprocessor_path.replace("\\",'/')
            model_path = model_path.replace('\\','/')

            logging.info('Loading of objects initiated')
            preprocessor=load_object(preprocessor_path)
            logging.info('Loaded preporcessor object')
            model=load_object(model_path)
            logging.info('Loaded the model object')

            data_scaled=preprocessor.transform(features)
            # data_arr = features.to_numpy()

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                age:float,
                bmi:float,
                children:float,
                sex:str,
                smoker:str,
                region:str):
        
        self.age=age
        self.bmi=bmi
        self.children=children
        self.sex=sex
        self.smoker=smoker
        self.region=region

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'age':[self.age],
                'bmi':[self.bmi],
                'children':[self.children],
                'sex':[self.sex],
                'smoker':[self.smoker],
                'region':[self.region]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)