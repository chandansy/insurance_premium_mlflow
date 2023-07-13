from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from src.exception import CustomException
from src.logger import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.metrics import r2_score
from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def model_trainer(self,train_arr,test_arr,preprocessor_obj_file):

        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train = train_arr[:,:-1]
            Y_train = train_arr[:,-1]
            X_test  = test_arr[:,:-1]
            Y_test  = test_arr[:,-1]
            models={
                'Random_forest':RandomForestRegressor()
                }
            params={
                "Random_forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                
                    # 'max_features':['sqrt','log2',None],
                    'max_depth':[2,4,6,8,10],
                    'min_samples_leaf':[6,8,10],
                    'n_estimators': [8,16,25,32,64,128,256]
                }
                
            }
            
            model_report:dict=evaluate_model(x_train=X_train,y_train=Y_train,x_test=X_test,y_test=Y_test,
                                            models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model {best_model} on both training and testing dataset and it's accuracy is {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(Y_test, predicted)
            return r2_square
            
            
        except Exception as e:
            raise CustomException(e,sys)