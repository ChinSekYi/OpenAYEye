"""
ROI_predict_pipeline.py

This module implements the prediction pipeline for ROI model.
It includes preprocessing, model training, evaluation, and prediction functions.

"""

import os
import sys
import joblib
import warnings 

import pandas as pd
import numpy as np

from collections import OrderedDict
from src.exception import CustomException
from src.utils import load_object

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')


class PredictPipeline:
    """
    Implements a pipeline for predicting number of clicks based on category and cost.
    """

    def __init__(self):
        pass

    def predict(self, features):
        """
        Predicts the clicks,leads, order based on the provided features.

        Args:
            features (pd.DataFrame): DataFrame containing the features for prediction.

        Returns:
            OrderedDict: Predicted results.
        """
        try:
            model_path = os.path.join("artifacts", "roi_trained_model.pkl")
            encoder_path = os.path.join("artifacts", "roi_onehot_encoder.pkl")

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
            target_names = ['clicks', 'leads', 'orders']
            pred_result = model.predict(X_transformed_df)

            results_dict = OrderedDict()
            for i, target in enumerate(target_names):
                results_dict[target] = pred_result[:, i][0]

            cost_of_credit_card = 200 #Assumption that it is a entry-level card
            orders = results_dict['orders'] 
            profit = round((orders * cost_of_credit_card) - float(features['cost'].values[0]), 2)
            results_dict['profit/loss'] = f"${profit}"

            return results_dict

        except Exception as e:
            raise CustomException(e, sys) from e


class CustomData:
    """
    Responsible for taking the inputs that we are giving in the HTML to the backend.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get_data_as_dataframe(self):
        """
        Converts the attributes of CustomData into a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame where each row represents a [category, cost].
        """
        try:
            custom_data_input = {
                "category": [self.category],
                "cost": [self.cost],
            }
            return pd.DataFrame(custom_data_input)

        except Exception as e:
            raise CustomException(e, sys) from e


# Example Usage
if __name__ == "__main__":
    user_input = {
        "category": "social",
        "cost": "500"
    }

    # Get input data in custom data format
    custom_data = CustomData(**user_input)
    features_df = custom_data.get_data_as_dataframe()

    # Run prediction pipeline
    pipeline = PredictPipeline()
    predictions = pipeline.predict(features_df)

    # Print input and output
    BOLD = "\033[1m"
    RESET = "\033[0m"
    ORANGE = "\033[38;5;214m"

    print(f'{BOLD}{ORANGE}Input:{RESET}\n {features_df}\n')
    print(f'{BOLD}{ORANGE}Predicted Results:{RESET}\n', predictions)
