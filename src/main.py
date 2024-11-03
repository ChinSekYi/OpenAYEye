"""
main.py

This is the entry point for the ROI prediction application and the recommendation system.
It initializes the respective prediction pipelines, processes user input,
and outputs the predicted results.

Instructions: Run `python3 -m src.main` in terminal.
"""

import sys

from src.exception import CustomException
from src.pipeline.reco_sys.Reco_sys_custom_data import RecoSysCustomData
from src.pipeline.reco_sys.Reco_sys_predict_pipeline import (
    PredictPipeline as RecoPredictPipeline,
)
from src.pipeline.roi.ROI_predict_pipeline import CustomData as ROICustomData
from src.pipeline.roi.ROI_predict_pipeline import PredictPipeline as ROIPredictPipeline


def run_roi_prediction(customer_metrics):
    """
    Runs the ROI prediction pipeline based on customer metrics.

    Args:
        customer_metrics (dict): Dictionary containing the metrics for ROI prediction.

    Returns:
        str: Predicted results for ROI.
    """
    try:
        # Get input data in custom data format
        custom_data = ROICustomData(**customer_metrics)
        features_df = custom_data.get_data_as_dataframe()

        # Initialize prediction pipeline
        roi_pipeline = ROIPredictPipeline()

        # Run prediction pipeline
        predictions = roi_pipeline.predict(features_df)

        return predictions, features_df

    except CustomException as e:
        raise CustomException(e, sys)


def run_recommendation_pipeline(user_input):
    """
    Runs the recommendation system prediction pipeline based on user input.

    Args:
        user_input (dict): Dictionary containing the features for prediction.

    Returns:
        str: Ranked recommendations as a formatted string.
    """
    try:
        # Get input data in custom data format
        custom_data = RecoSysCustomData(**user_input)
        features_df = custom_data.get_data_as_dataframe()

        # Run prediction pipeline
        reco_pipeline = RecoPredictPipeline()
        predictions = reco_pipeline.predict(features_df)

        return predictions, features_df

    except Exception as e:
        raise CustomException(e, sys)


def print_input_output(features_df, predictions, title):
    """Utility function to print formatted input and predicted results."""
    BOLD = "\033[1m"
    RESET = "\033[0m"
    ORANGE = "\033[38;5;214m"

    print(f"{BOLD}{ORANGE}{title}:{RESET}\n {features_df}\n")
    print(f"{BOLD}{ORANGE}Predicted Results:{RESET}\n", predictions)


def main():
    # Example customer metrics for ROI prediction
    customer_metrics = {
        "category": "social",  # Example category
        "cost": "50000",  # Example cost with dollar sign for user-friendly input
    }

    # Run ROI Prediction
    try:
        roi_predictions, roi_features = run_roi_prediction(customer_metrics)
        print_input_output(roi_features, roi_predictions, "ROI Prediction Input")
    except CustomException as e:
        print(f"An error occurred in ROI prediction: {e}")

    # Example user input for recommendation system
    user_input = {
        "age": 92,
        "gender": "Male",
        "gross_income": 500,
        "customer_segment": "Private",
        "contract_length": 31,
        "seniority_months": 12,
        "primary_customer_status": "primary customer",
        "new_customer_index": 'existing customer',
        "customer_type_start_month": "Dec",
        "country_residence": "NI",
        "region": "South",
        "join_channel": "Austria",
        "deceased_index": "no",
        "foreigner_index": "yes",
        "residence_index": "no",
        "customer_relation_type": "Associated",
    }

    # Run Recommendation Prediction
    try:
        reco_predictions, reco_features = run_recommendation_pipeline(user_input)
        print("\n")
        print_input_output(
            reco_features, reco_predictions, "Recommendation Prediction Input"
        )
    except CustomException as e:
        print(f"An error occurred in Recommendation prediction: {e}")


if __name__ == "__main__":
    main()
