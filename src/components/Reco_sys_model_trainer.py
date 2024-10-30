"""
Module for training and evaluating Logistic Regression models for multi-output regression tasks.

This module includes:
- A configuration class for model training parameters.
- A `ModelTrainer` class that handles model training, evaluation, and the saving of the best-performing models.

The models are trained on features derived from a dataset that includes both categorical and numerical inputs. The targets consist of multiple numerical outputs (e.g., clicks, leads, orders).

Usage:
    >>> from model_trainer import ModelTrainer
    >>> trainer = ModelTrainer()
    >>> metrics, final_models = trainer.initiate_model_trainer(train_array, test_array)
"""

import os
import sys
import joblib

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm 
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    """
    Configuration class for model training operations.

    Attributes:
        trained_fixed_deposits_model_file_path (str): File path to save the trained fixed deposits model.
        trained_loan_model_file_path (str): File path to save the trained loan model.
        trained_credit_card_debit_card_file_path (str): File path to save the trained credit card/debit card model.
        trained_account_model_file_path (str): File path to save the trained account model.
    """

    trained_fixed_deposits_model_file_path = os.path.join("artifacts", "reco_sys_fixed_deposits_model.pkl")
    trained_loan_model_file_path = os.path.join("artifacts", "reco_sys_loan_model.pkl")
    trained_credit_card_debit_card_file_path = os.path.join("artifacts", "reco_sys_credit_card_debit_card_model.pkl")
    trained_account_model_file_path = os.path.join("artifacts", "reco_sys_account_model.pkl")

class ModelTrainer:
    """
    Class to handle the training and evaluation of Logistic Regression models for multi-output regression.

    Methods:
        __init__(): Initializes a ModelTrainer instance with the default configuration.
        train_models(X_train, y_train): Trains multiple Logistic Regression models on the training data.
        evaluate_model(model, X_test, y_test, column_name): Evaluates a trained model on the test set and logs metrics.
        evaluate_all_models(X_test, y_test): Evaluates all trained models on the test set.
        save_models(): Saves each trained model to its respective file path.
        initiate_model_trainer(df_train, df_test): Orchestrates model training, evaluation, and saving of models.
    """

    def __init__(self):
        """
        Initializes a ModelTrainer instance with default configuration.
        """
        self.model_trainer_config = ModelTrainerConfig()
        self.models = {}
        self.output_metrics = {}
        logging.info("ModelTrainer initialized with configuration.")

    def train_models(self, X_train, y_train):
        """
        Trains Logistic Regression models for each target variable.

        Args:
            X_train (pd.DataFrame): Feature matrix for training.
            y_train (pd.DataFrame): Target variables for training.

        Returns:
            dict: A dictionary containing trained models for each target.
        """
        # Define the logistic regression model
        lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')

        # Set up hyperparameters for tuning
        param_dist = {
            'C': [0.001, 0.01, 0.1, 1],    # Reduced number of hyperparameters
            'penalty': ['l1', 'l2'],       # Only two regularization types
        }

        # Use StratifiedKFold for cross-validation
        logging.info("Use StratifiedKFold for cross-validation.")
        skf = StratifiedKFold(n_splits=3)  # Reduced number of folds

        # Perform random search with cross-validation
        logging.info("Perform random search with cross-validation.")
        random_search = RandomizedSearchCV(lr_model, param_dist, n_iter=8, cv=skf, scoring='f1_macro', n_jobs=-1)

        for i in tqdm(range(y_train.shape[1]), desc="Training Models", unit="model"):
            logging.info(f"Training model for target column {y_train.columns[i]}")
            
            # Train the model
            random_search.fit(X_train, y_train.iloc[:, i])
            self.models[y_train.columns[i]] = random_search.best_estimator_
            logging.info(f"Best parameters for {y_train.columns[i]}: {random_search.best_params_}")

        return self.models

    def evaluate_model(self, model, X_test, y_test, column_name):
        """
        Evaluates a trained model on the test data and logs the results.

        Args:
            model: The trained Logistic Regression model to evaluate.
            X_test (pd.DataFrame): Feature matrix for testing.
            y_test (pd.DataFrame): True values for target variables.
            column_name (str): The name of the target variable to evaluate.
        """
        y_pred = model.predict(X_test)
        confusion = confusion_matrix(y_test[column_name], y_pred)
        classification = classification_report(y_test[column_name], y_pred, output_dict=True)

        # Store confusion matrix and classification report in output metrics
        self.output_metrics[column_name] = {
            'confusion_matrix': confusion.tolist(),  # Convert to list for JSON serialization
            'classification_report': classification
        }

        logging.info(f"Confusion Matrix for {column_name}:\n{confusion}")
        logging.info(f"Classification Report for {column_name}:\n{classification}")
        
    def evaluate_all_models(self, X_test, y_test):
        """
        Evaluates all trained models against the test dataset.

        Args:
            X_test (pd.DataFrame): Feature matrix for testing.
            y_test (pd.DataFrame): True values for target variables.
        """
        for column_name, model in self.models.items():
            logging.info(f"Evaluating model for target column {column_name}.")
            self.evaluate_model(model, X_test, y_test, column_name)

    def save_models(self):
        """
        Saves each trained model to its respective file path.
        """
        model_paths = {
            'fixed_deposits': self.model_trainer_config.trained_fixed_deposits_model_file_path,
            'loan': self.model_trainer_config.trained_loan_model_file_path,
            'credit_card_debit_card': self.model_trainer_config.trained_credit_card_debit_card_file_path,
            'account': self.model_trainer_config.trained_account_model_file_path
        }
        
        for target, model in self.models.items():
            file_path = model_paths[target]
            joblib.dump(model, file_path)
            logging.info(f"Saved {target} model to {file_path}")

    def initiate_model_trainer(self, df_train, df_test):
        """
        Orchestrates the training and evaluation of models.

        Args:
            df_train (pd.DataFrame): DataFrame containing training data with features and targets.
            df_test (pd.DataFrame): DataFrame containing test data with features and targets.

        Returns:
            tuple: A tuple containing output metrics and trained models.

        Raises:
            CustomException: If an error occurs during model training or evaluation.

        Usage:
            >>> trainer = ModelTrainer()
            >>> output = trainer.initiate_model_trainer(df_train, df_test)
        """
        try:
            logging.info("Selecting features and targets from the dataset.")
            prediction_columns = ['fixed_deposits', 'loan', 'credit_card_debit_card', 'account']
            X_train = df_train.drop(columns=prediction_columns)  # First 86 columns (features)
            y_train = df_train[prediction_columns]

            X_test = df_test.drop(columns=prediction_columns)  # First 86 columns (features)
            y_test = df_test[prediction_columns]

            print(len(X_train.columns))
            # Normalize the feature columns
            #scaler = StandardScaler()
            #X_normalized = scaler.fit_transform(X)

            logging.info("Start model training.")
            self.train_models(X_train, y_train)

            logging.info("Start model evaluation for all 4 logistic regression models.")
            self.evaluate_all_models(X_test, y_test)

            # Save each model to its corresponding file
            self.save_models()

            logging.info("Model training and evaluation completed successfully.")
            return self.output_metrics , self.models

        except Exception as e:
            print(e)
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    """
    Main function to test the ModelTrainer.
    """
    try:
        # Load your dataset (adjust the path to your actual dataset)
        df_train = pd.read_csv('artifacts/reco_sys_train_data.csv') 
        df_test = pd.read_csv('artifacts/reco_sys_test_data.csv') 

        # Initialize the ModelTrainer
        trainer = ModelTrainer()

        # Train the model and get predictions and R^2 scores
        output = trainer.initiate_model_trainer(df_train, df_test)

        # Define color codes
        BOLD = "\033[1m"
        RESET = "\033[0m"
        ORANGE = "\033[38;5;214m"  # An approximation of orange
                
        # Print the structured output
        print(f"\n{BOLD}{ORANGE}Metrics for each model:{RESET} \n {output[0]}\n")
        print(f"{BOLD}{ORANGE}Final trained models:{RESET} \n {output[1]}")

    except Exception as e:
        print(f"An error occurred: {e}")
