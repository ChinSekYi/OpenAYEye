"""
ROI_predict_pipeline.py

This module implements the prediction pipeline for bank marketing data analysis.
It includes preprocessing, model training, evaluation, and prediction functions.

"""

import os
import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    """
    Implements a pipeline for predicting financial metrics using various models.
    """

    def __init__(self):
        pass

    def predict(self, features):
        """
        Predicts the financial metrics based on the provided features.

        Args:
            features (pd.DataFrame): DataFrame containing the features for prediction.

        Returns:
            np.array: Predicted results.
        """
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            pred_result = model.predict(data_scaled)
            return pred_result

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Responsible for taking the inputs that we are giving in the HTML to the backend.
    """

    def __init__(self, **kwargs):
        """
        Initialize CustomData with the provided financial metrics.
        
        Accepts keyword arguments for the financial metrics.
        """
        self.__dict__.update(kwargs)

    def get_data_as_dataframe(self):
        """
        Converts the attributes of CustomData into a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame where each row represents a financial metric attribute.
        """
        try:
            custom_data_input = {
                "category": [self.category],
                "cost": [self.cost],
            }
            return pd.DataFrame(custom_data_input)

        except Exception as e:
            raise CustomException(e, sys)


# Example Usage
if __name__ == "__main__":
    customer_metrics = {
        "category": "social",
        "cost": "50000"
    }

    custom_data = CustomData(**customer_metrics)
    features_df = custom_data.get_data_as_dataframe()

    pipeline = PredictPipeline()
    predictions = pipeline.predict(features_df)
    print("Predicted Results:", predictions)
