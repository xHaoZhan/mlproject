import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            scaled_data = preprocessor.transform(features)
            pred = model.predict(scaled_data)
            return pred
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self,
        gender: str,
        race_ethinicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender
        self.race_ethinicity = race_ethinicity
        self.parental_level_of_education = parental_level_of_education
        self.reading_score = reading_score
        self.writing_score = writing_score
        self.test_preparation_course = test_preparation_course
        self.lunch = lunch

    def get_data_as_df(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethinicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)