import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTranformationConfig:
    preprocessor_obj_filepath: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTranformation:
    def __init__(self):
        self.data_tranformation_config = DataTranformationConfig()

    def get_data_transformer_object(self):
        # Data transform different kind of data.
        try:
            numerical_features = ["writing score","reading score"]
            categorical_features = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]

            numerical_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(sparse_output=False)),
                    ("scaler",StandardScaler())
            ]    
            )
            
            logging.info(f"Numerical features: {numerical_features}'\n'Categorical features: {categorical_features}")

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_features),
                    ("categorical_pipeline", categorical_pipeline, categorical_features)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)   

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Complete reading train and test data.")

            logging.info("Obtaining preprocessing object.")
            preprocessing_object = self.get_data_transformer_object()

            target_feature = "math score"
            numerical_features = ["writing score","reading score"]

            input_feature_train_df = train_df.drop(columns=[target_feature],axis=1)
            target_feature_train_df = train_df[target_feature]

            input_feature_test_df = test_df.drop(columns=[target_feature],axis=1)
            target_feature_test_df = test_df[target_feature]

            logging.info("Applying preprocessing object on training and test dataframe.")

            input_feature_train_array = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_object.transform(input_feature_test_df)

            train_array = np.c_[
                input_feature_train_array, np.array(target_feature_train_df)
            ]
            test_array = np.c_[
                input_feature_test_array, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path = self.data_tranformation_config.preprocessor_obj_filepath,
                object = preprocessing_object
            )

            return(
                train_array, 
                test_array, 
                self.data_tranformation_config.preprocessor_obj_filepath, 
            )

        except Exception as e:
            raise CustomException(e,sys)