"""
Reco_sys_predict_pipeline.py

This module implements the prediction pipeline for the recommendation system.
It includes preprocessing, model training, evaluation, and prediction functions.

"""

import os
import sys
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.pipeline.reco_sys_custom_data import RecoSysCustomData  # Import the new class

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

class PredictPipeline:
    """
    Implements a pipeline for predicting the type of bank product a customer is likely to subscribe. 
    """

    def __init__(self):
        pass

    def predict_proba(self, new_data, models):
        """
        Calculates the probability of class 1 for each model on the new data.

        Args:
            new_data (pd.DataFrame): DataFrame containing the features for prediction.
            models (dict): Dictionary of models.

        Returns:
            dict: A dictionary with model names as keys and their predicted probabilities as values.
        """
        #scaler = StandardScaler()
        #scaler.fit(new_data)
        #new_data_normalized = scaler.transform(new_data)

        probabilities = {}
        for model_name, model in models.items():
            probabilities[model_name] = model.predict_proba(new_data)[:, 1][0]  # Probability of class 1
        return probabilities
    
    def rank_recommendations(self, probabilities):
        """
        Ranks the products by probability and maps model names to user-friendly names.

        Args:
            probabilities (dict): Dictionary with model names as keys and probabilities as values.

        Returns:
            list of tuples: Ranked list of products with user-friendly names and probabilities.
        """
        # Mapping from model keys to descriptive names
        model_name_mapping = {
            "account_model": "Account",
            "card_model": "Credit and Debit Card",
            "fixed_deposits_model": "Fixed Deposits",
            "loan_model": "Loan"
        }

        # Map model names to friendly names and sort by probability
        ranked_recommendations = sorted(
            [(model_name_mapping[name], prob) for name, prob in probabilities.items()],
            key=lambda x: x[1],
            reverse=True
        )
        return ranked_recommendations

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
            models = {
                "account_model": joblib.load(account_model_path),
                "card_model": joblib.load(card_model_path),
                "fixed_deposits_model": joblib.load(fixed_model_deposits_path),
                "loan_model": joblib.load(loan_model_path)
            }

            probabilities = self.predict_proba(features, models)
            ranked_recommendations = self.rank_recommendations(probabilities)
            
            # Format the ranked recommendations as a string
            recommendations_str = "Ranked Recommendations:\n"
            for rank, (name, prob) in enumerate(ranked_recommendations, start=1):
                recommendations_str += f"{rank}. {name}\n"

            return recommendations_str

        except Exception as e:
            raise CustomException(e, sys)


# Example Usage
if __name__ == "__main__":
    user_input = {
        'age': 92,
        'gender': 'H',
        'gross_income': 500,
        'customer_segment': '03 - TOP',
        'contract_length': 31,
        'seniority_months': 12,
        'primary_customer_status': '1',
        'new_customer_index': 0.0,
        'customer_type_start_month': 1,
        'country_residence': 'NI',
        'region': 'NORTH',
        'join_channel': 'KHA',
        'deceased_index': 'S',
        'foreigner_index': 'N',
        'residence_index': 'S',
        'customer_relation_type': 'I',
        }

    # Get input data in custom data format
    custom_data = RecoSysCustomData(**user_input)
    features_df = custom_data.get_data_as_dataframe()

    # Run prediction pipeline
    pipeline = PredictPipeline()
    predictions = pipeline.predict(features_df)

    # Print input and output
    BOLD = "\033[1m"
    RESET = "\033[0m"
    ORANGE = "\033[38;5;214m"

    print(f'{BOLD}{ORANGE}Input:{RESET}\n {features_df}\n')
    print(f"{BOLD}{ORANGE}Output:{RESET}\n", predictions)