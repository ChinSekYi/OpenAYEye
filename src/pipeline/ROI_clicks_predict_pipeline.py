"""
ROI_predict_pipeline.py

This module implements the prediction pipeline for bank marketing data analysis.
It includes preprocessing, model training, evaluation, and prediction functions.

"""

import os
import sys
import joblib

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object
from sklearn.preprocessing import OneHotEncoder

class PredictPipeline:
    """
    Implements a pipeline for predicting financial metrics using various models.
    """

    def __init__(self):
        pass

    def predict(self, features):
        """
        Predicts the clicks,leads,order based on the provided features.

        Args:
            features (pd.DataFrame): DataFrame containing the features for prediction.

        Returns:
            np.array: Predicted results.
        """
        try:
            model_path = os.path.join("artifacts", "clicks_model.pkl")
            encoder_path = os.path.join("artifacts", "onehot_encoder.pkl")

            model = load_object(file_path=model_path)
            encoder = joblib.load(encoder_path) # Load pre-fitted OneHotEncoder
            
            # Apply the pre-fitted encoder to the new data
            X_encoded = encoder.transform(features[['category']])  # Use transform, not fit_transform

            # Get the expected encoded feature names from the encoder
            encoded_columns = encoder.get_feature_names_out(['category'])
            
            # Concatenate the encoded category columns with the 'cost' column
            X_transformed = np.concatenate([X_encoded, features[['cost']].values], axis=1)
            
            # Create a DataFrame to ensure column names match the expected structure
            column_names = list(encoded_columns) + ['cost']
            X_transformed_df = pd.DataFrame(X_transformed, columns=column_names)
            
            # Predict using the trained model
            pred_result = model.predict(X_transformed_df)

            return pred_result[0]

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
    print(f'Input:\n {features_df}\n')
    print("Predicted Results (Number of clicks):\n", predictions)
