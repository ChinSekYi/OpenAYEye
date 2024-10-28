"""
Module for training and evaluating a Random Forest Regressor on a multi-output regression problem.

This module contains:
- A configuration class for model training parameters.
- A ModelTrainer class that handles the model training, evaluation, and saving of the best performing model.

The model is trained on features derived from a dataset, including categorical and numerical inputs. The targets include multiple numerical outputs (clicks, leads, orders).

Usage:
    >>> from model_trainer import ModelTrainer
    >>> trainer = ModelTrainer()
    >>> predicted_values, r2_scores = trainer.initiate_model_trainer(train_array, test_array)
"""

import os
import sys
import joblib

import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    """
    Configuration class for model training operations.

    Attributes:
    - trained_model_file_path (str): File path to save the trained model.
    """

    trained_model_file_path = os.path.join("artifacts", "leads_model.pkl")
    encoder_path = os.path.join("artifacts", "onehot_encoder.pkl")

class ModelTrainer:
    """
    ModelTrainer class handles training of the Random Forest Regressor
    and evaluation of its performance on a multi-output regression problem.

    Methods:
    - __init__(): Initializes a ModelTrainer instance with default configuration.
    - initiate_model_trainer(df): Initiates model training and evaluation,
      saving the best model and returning predictions and performance metrics.

    """

    def __init__(self):
        """
        Initializes a ModelTrainer instance with default configuration.
        """
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, df):
        """
        Initiates model training, evaluation, and saves the trained model.

        Args:
        - df (pd.DataFrame): DataFrame containing the dataset with features and targets.

        Returns:
        - dict: A dictionary containing predicted values, R-squared scores, and MSE for each target variable.

        Raises:
        - CustomException: If an error occurs during model training or evaluation.

        Usage:
        >>> trainer = ModelTrainer()
        >>> output = trainer.initiate_model_trainer(df)
        """
        try:
            logging.info("Selecting features and targets from the dataset.")
            X = df[['category', 'cost']]  # Input features
            y = df['leads']  # Targets

            # One-hot encoding for 'category'
            encoder = OneHotEncoder(sparse_output=False)
            X_encoded = encoder.fit_transform(X[['category']])
            
            # Get feature names for the one-hot encoded columns
            encoded_columns = encoder.get_feature_names_out(['category'])
            cost_column = ['cost']

            # Combine column names
            column_names = list(encoded_columns) + cost_column

            # Concatenate the encoded category with cost
            X_transformed = np.concatenate([X_encoded, X[['cost']].values], axis=1)

            # Convert to DataFrame with proper column names
            X_transformed_df = pd.DataFrame(X_transformed, columns=column_names)

            # Split the data into train and test sets
            logging.info("Train test split initiated in Model trainer")
            X_train, X_test, y_train, y_test = train_test_split(X_transformed_df, y, test_size=0.2, random_state=42)

            # Define and train the RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Predict on the test set
            y_pred = model.predict(X_test)

            # Evaluate the model using Mean Squared Error and R^2 score
            mse_value = mean_squared_error(y_test, y_pred, multioutput='raw_values')
            r2_value = r2_score(y_test, y_pred, multioutput='raw_values')

            # Round MSE and R² scores to three decimal places
            mse_values_rounded = np.round(mse_value, 3)
            r2_values_rounded = np.round(r2_value, 3)

            # Structure the output for frontend consumption
            output = {
                #'predictions': y_pred.tolist(),  # Convert to list for JSON serialization
                'r2_scores': mse_values_rounded.tolist(),  # Convert to list for JSON serialization
                'mse_values': r2_values_rounded.tolist()  # Convert to list for JSON serialization
            }

            # Print the accuracy for each target variable
            logging.info(f"{'leads'} - MSE: {mse_values_rounded}, R^2: {r2_values_rounded}")

            logging.info(f"Saving the trained model to {self.model_trainer_config.trained_model_file_path}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model,
            )

            # Save the fitted OneHotEncoder to a .pkl file
            joblib.dump(encoder, self.model_trainer_config.encoder_path)  # Save the encoder

            return output  # Return structured output

        except Exception as e:
            raise CustomException(e, sys) from e

if __name__ == "__main__":
    """
    Main function to test the ModelTrainer.
    """
    try:
        # Load your dataset (adjust the path to your actual dataset)
        df = pd.read_csv('artifacts/roi_leads_train_data.csv')  # Replace with your dataset path

        # Initialize the ModelTrainer
        trainer = ModelTrainer()

        # Train the model and get predictions and R^2 scores
        output = trainer.initiate_model_trainer(df)

        # Print the structured output for 'leads'
        print("Output:")
        print(output)

    except Exception as e:
        print(f"An error occurred: {e}")
