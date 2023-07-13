from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    ingestion_obj = DataIngestion()
    train,test = ingestion_obj.inititate_data_ingestion()
    transform_obj = DataTransformation()

    train_trans,test_trans,pre = transform_obj.initiate_data_transformation(train,test)

    train_obj = ModelTrainer()
    train_obj.model_trainer(train_trans,test_trans,pre)
