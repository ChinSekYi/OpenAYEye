"""
roi_data_pipeline.py

This pipeline is used when new data is fed into the frontend.
"""


"""
Module for data transformation operations including preprocessing and saving the 
preprocessor object.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


# Custom imports
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation operations.
    """

    preprocessor_ob_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    DataTransformation class handles data preprocessing and transformation operations.

    Methods:
    - __init__(): Initializes a DataTransformation instance with default configuration.
    - get_data_transformer_object(): Returns the preprocessing object.
    - initiate_data_transformation(train_path, test_path): Initiates data transformation,
      performs preprocessing on train and test datasets, and saves the preprocessor object.
    """

    def __init__(self):
        """
        Initializes a DataTransformation instance with default configuration.
        """
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Returns the preprocessing object.
        """
        try:
            categorical_columns = ["category"]  # for category
            numerical_columns = ["cost"]         # for cost

            num_pipeline = Pipeline(
                steps=[
                    ("scaler", MinMaxScaler()),
                ]
            )

            # Create an empty categorical pipeline (no transformations)
            cat_pipeline = Pipeline(steps=[])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            print(e)
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self, train_path, test_path):
        """
        Initiates the data transformation process.
        Reads train and test datasets, applies preprocessing, and saves the preprocessor object.

        Args:
        - train_path (str): File path to the training dataset.
        - test_path (str): File path to the testing dataset.

        Returns:
        - Tuple: Transformed train and test datasets and the file path of the preprocessor object.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            input_feature_columns = ['category', 'cost']  # Input features
            target_columns = ['clicks', 'leads', 'orders']  # Targets

            # Separate the features and target variables
            input_feature_train_df = train_df[input_feature_columns]
            target_feature_train_df = train_df[target_columns]

            input_feature_test_df = test_df[input_feature_columns]
            target_feature_test_df = test_df[target_columns]

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys) from e

if __name__ == "__main__":
    train_data_path = 'artifacts/roi_model_training_data.csv'
    test_data_path = 'artifacts/roi_model_test_data.csv'

    # Initialize the DataTransformation class
    data_transformation = DataTransformation()

    try:
        # Call the initiate_data_transformation method and capture the output
        train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        
        # Print the results
        print("Training Data Array:\n", train_array)
        print("Testing Data Array:\n", test_array)
        print("Preprocessor object saved at:\n", preprocessor_path)

    except CustomException as ce:
        print(f"Custom Exception occurred: {ce}")
    except Exception as e:
        print(f"An error occurred: {e}")
