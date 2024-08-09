import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as f:
            dill.dump(object, f)
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i] #loop through all models
            try:
                param = params[list(models.keys())[i]]
                #Hyperparameter tuning using GridSearchCV
                gs = GridSearchCV(model,param_grid=param,cv=5,n_jobs=3,verbose=2,refit=True)
                gs.fit(X_train,y_train)

                model.set_params(**gs.best_params_)
            except:
                #if no parameter found in params then pass
                pass

            model.fit(X_train,y_train) #Train model

            y_train_pred = model.predict(X_train) 
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred) 
            test_model_score = r2_score(y_test,y_test_pred) 

            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)