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

    trained_model_file_path = os.path.join("artifacts", "model.pkl")

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
            y = df[['clicks', 'leads', 'orders']]  # Targets

            # One-hot encoding for 'category'
            encoder = OneHotEncoder(sparse_output=False)
            X_encoded = encoder.fit_transform(X[['category']])
            
            # Concatenate the encoded category with cost
            X_transformed = np.concatenate([X_encoded, X[['cost']].values], axis=1)

            # Split the data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

            # Define and train the RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Predict on the test set
            y_pred = model.predict(X_test)

            # Evaluate the model using Mean Squared Error and R^2 score for each target
            mse_values = mean_squared_error(y_test, y_pred, multioutput='raw_values')
            r2_scores = r2_score(y_test, y_pred, multioutput='raw_values')

            # Structure the output for frontend consumption
            output = {
                #'predictions': y_pred.tolist(),  # Convert to list for JSON serialization
                'r2_scores': r2_scores.tolist(),  # Convert to list for JSON serialization
                'mse_values': mse_values.tolist()  # Convert to list for JSON serialization
            }

            # Print the accuracy for each target variable
            target_names = ['clicks', 'leads', 'orders']
            for i, target in enumerate(target_names):
                logging.info(f"{target} - MSE: {mse_values[i]}, R^2: {r2_scores[i]}")

            logging.info(f"Saving the trained model to {self.model_trainer_config.trained_model_file_path}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model,
            )

            return output  # Return structured output

        except Exception as e:
            raise CustomException(e, sys) from e

if __name__ == "__main__":
    """
    Main function to test the ModelTrainer.
    """
    try:
        # Load your dataset (adjust the path to your actual dataset)
        df = pd.read_csv('artifacts/roi_model_data.csv')  # Replace with your dataset path

        # Initialize the ModelTrainer
        trainer = ModelTrainer()

        # Train the model and get predictions and R^2 scores
        output = trainer.initiate_model_trainer(df)

        # Print the structured output for 'clicks', 'leads', 'orders'
        print("Output:")
        print(output)

    except Exception as e:
        print(f"An error occurred: {e}")
