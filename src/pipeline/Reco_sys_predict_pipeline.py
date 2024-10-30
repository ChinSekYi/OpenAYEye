"""
Reco_sys_predict_pipeline.py

This module implements the prediction pipeline for the recommendation system.
It includes preprocessing, model training, evaluation, and prediction functions.

"""

import os
import sys
import joblib

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    """
    Implements a pipeline for predicting the type of bank product a customer is likely to subscribe. 
    """

    def __init__(self):
        pass

    def predict(self, features):
        """
        Predicts the probability of a customer subscribing a particular bank product, using Logistic Regression.
        Available bank products: 'fixed_deposits', 'loan', 'credit_card_debit_card', 'account'

        Args:
            features (pd.DataFrame): DataFrame containing the features for prediction.

        Returns:
            np.array: Predicted results.
        """
        try:
            # Define the model paths
            account_model_path = os.path.join("artifacts", "reco_sys_account_model.pkl")
            card_model_path = os.path.join("artifacts", "reco_sys_credit_card_debit_card_model.pkl")
            fixed_model_deposits_path = os.path.join("artifacts", "reco_sys_fixed_deposits_model.pkl")
            loan_model_path = os.path.join("artifacts", "reco_sys_loan_model.pkl")

            # Load the models
            account_model = joblib.load(account_model_path)
            card_model = joblib.load(card_model_path)
            fixed_model_deposits = joblib.load(fixed_model_deposits_path)
            loan_model = joblib.load(loan_model_path)

            # Predict using the trained models
            account_pred_result = account_model.predict(features)
            credit_card_pred_result = card_model.predict(features)
            fixed_deposits_pred_result = fixed_model_deposits.predict(features)
            loan_pred_result = loan_model.predict(features)
            
            # Prepare the results dictionary
            pred_result = {
                "account": account_pred_result,
                "credit card & debit cardc": credit_card_pred_result,
                "fixed_deposits": fixed_deposits_pred_result,
                "loan": loan_pred_result
            }

            return pred_result

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get_data_as_dataframe(self):
        try:
            custom_data_input = {
                "age": [self.age],
                "contract_length": [self.contract_length],
                "seniority_months": [self.seniority_months],
                "gross_income": [self.gross_income],
                "primary_customer_status": [self.primary_customer_status],
                "residence_index": [self.residence_index],
                "customer_relation_type": [self.customer_relation_type],
                'customer_type_start_month': [self.customer_type_start_month]
            }

            country_values = ['ES', 'AT', 'NL', 'FR', 'GB', 'CL', 'CH', 'DE', 'DO', 'BE', 
                              'AR', 'VE', 'US', 'MX', 'BR', 'IT', 'EC', 'PE', 'CO', 
                              'HN', 'FI', 'SE', 'AL', 'PT', 'MZ', 'CN', 'TW', 'PL', 
                              'IN', 'CR', 'NI', 'CA', 'IE', 'AL'] 
            
            for country in country_values:
                custom_data_input[f"country_residence_{country}"] = [1 if self.country_residence == country else 0]

            regions = ["CENTRAL", "NORTH", "SOUTH", "EAST", "WEST"]

            for region in regions:
                custom_data_input[f"region_{region}"] = [
                    1 if self.region_name == region else 0
                ]

            customer_segment_values = ['03 - UNIVERSITARIO', '02 - PARTICULARES', '01 - TOP']  # List of unique customer segments
           
            for segment in customer_segment_values:
                custom_data_input[f"customer_segment_{segment}"] = [
                    1 if self.customer_segment == segment else 0
                ]

            join_channel_values = ['KAT', 'KAZ', 'KDH', 'KFA', 'KFC', 'KGN', 
                       'KHC', 'KHD', 'KHE', 'KHK', 'KHL', 'KHM', 
                       'KHO', 'RED']

            for channel in join_channel_values:
                custom_data_input[f"join_channel_{channel}"] = [1 if self.join_channel == channel else 0]

            customer_relation_type_values = ['I', 'A', 'P']  

            for relation_type in customer_relation_type_values:
                custom_data_input[f"customer_relation_type_{relation_type}"] = [
                    1 if self.customer_relation_type == relation_type else 0
                ]

            # List of unique deceased_index values
            deceased_index_values = ['N', 'S']  

            for index in deceased_index_values:
                custom_data_input[f"deceased_index_{index}"] = [
                    1 if self.deceased_index == index else 0
                ]

            # List of unique foreigner_index values
            foreigner_index_values = ['S', 'N']  

            for index in foreigner_index_values:
                custom_data_input[f"foreigner_index_{index}"] = [
                    1 if self.foreigner_index == index else 0
                ]

            # List of unique gender values
            gender_values = ['H', 'V']  

            for gender in gender_values:
                custom_data_input[f"gender_{gender}"] = [
                    1 if self.gender == gender else 0
                ]

            # List of unique new customer index values
            new_customer_index_values = [0.0, 1.0]  

            for index in new_customer_index_values:
                custom_data_input[f"new_customer_index_{index}"] = [
                    1 if self.new_customer_index == index else 0
                ]


            return pd.DataFrame(custom_data_input)
        except Exception as e:
            raise CustomException(e, sys)


# Example Usage
if __name__ == "__main__":
    user_input = {
    'age': 30,
    'contract_length': 12,
    'seniority_months': 24,
    'gross_income': 50000,
    'primary_customer_status': 'Active',
    'customer_segment': 'TOP',
    'region_name': 'CENTRAL',
    'join_channel': 'KAT',
    'country_residence': 'CA',
    'deceased_index': 0,
    'foreigner_index': 1,
    'residence_index': 1,
    'customer_relation_type': 0,
    'customer_type_start_month': 1,
    'gender': 1,
    'new_customer_index': 0,
    }


    custom_data = CustomData(**user_input)
    features_df = custom_data.get_data_as_dataframe()
    print(features_df.columns.tolist())

    pipeline = PredictPipeline()
    predictions = pipeline.predict(features_df)
    print(f'Input:\n {features_df}\n')
    print("Predicted Results (For top bank products recommended, and the prob of the customer subscribing to it):\n", predictions)