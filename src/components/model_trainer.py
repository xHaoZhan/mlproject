import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_models, save_object
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info(f"Split training and test data.")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "AdaBoost": AdaBoostRegressor(),
                "Catboost": CatBoostRegressor(verbose=False),
                "K-Neighbours": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,
                                             X_test=X_test,y_test=y_test,models=models)

            print(model_report)

            #To get best model name & score from report dict
            best_model_name = sorted(model_report, key=model_report.get, reverse=True)[0]
            best_model_score = model_report.get(best_model_name)

            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model is found.")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=best_model
            )
            print(best_model_name)
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)

