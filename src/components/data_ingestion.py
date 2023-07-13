import os
import pandas as pd
import sys
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def inititate_data_ingestion(self):
        logging.info("Data Ingestion Started")

        try:
            df = pd.read_csv(r'notebooks\data\insurance.csv')
            logging.info("Data set is read as a pandas DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Raw data is saved to artifacts')

            train_set,test_set = train_test_split(df,test_size=0.25,random_state=42)
            logging.info("Training and test data split completed")

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Test Train data saved')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logging.info('Exception has occured during Data Ingestion process')
            raise CustomException(e,sys)
        

if __name__ == '__main__':
    obj = DataIngestion()
    train,test = obj.inititate_data_ingestion()
