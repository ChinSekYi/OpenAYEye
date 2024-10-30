"""
Reco_sys_data_pipeline.py

This module helps perform data ingestion. 

Methods:
- __init__(): Initializes with default configuration.
- initiate_data_ingestion(): Reads, cleans, and combines datasets for further processing.

Output:
- Train and test data file path 

Note:
- Imports necessary components for data transformation, model training, exception handling, logging, and printing outcomes.
- Utilizes standard libraries and third-party modules like sklearn for data handling.
"""


import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import read_column_mapping

# Setting the project root path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


@dataclass
class DataIngestionConfig:
    """
    Configuration class for data paths used in data ingestion processes.

    Attributes:
    - train_data_path (str): Path to training data file.
    - test_data_path (str): Path to testing data file.
    - raw_data_path (str): Path to raw data file.
    """

    train_data_path: str = os.path.join("artifacts", "reco_sys_train_data.csv")
    test_data_path: str = os.path.join("artifacts", "reco_sys_test_data.csv")
    raw_data_path: str = os.path.join("data", "santander_train_small.csv")


class DataIngestion:
    """
    DataIngestion class handles the ingestion of raw data,
    performs train-test split, and saves the split datasets.

    Attributes:
    - ingestion_config (DataIngestionConfig): Configuration object for data paths.

    Methods:
    - __init__(): Initializes a new instance of DataIngestion with default configuration.
    - initiate_data_ingestion(): Initiates the data ingestion process, reads a CSV file,
      performs train-test split, and saves the datasets to specified paths.
    """

    def __init__(self):
        """
        Initializes a DataIngestion instance with default configuration.
        """
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process.

        Reads a CSV file from a specified path, performs train-test split,
        and saves the split datasets to predefined paths.

        Returns:
        - Tuple containing paths to the training and testing datasets.

        Raises:
        - CustomException: If an error occurs during data ingestion process.
        """
        logging.info("Entered the data ingestion component for Recommendation system.")
        try:
            logging.info("Read the dataset as dataframe")
            df = pd.read_csv(self.ingestion_config.raw_data_path)

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)

            train_output_path = self.ingestion_config.train_data_path
            train_set.to_csv(
                train_output_path, index=False, header=True
            )

            test_output_path = self.ingestion_config.test_data_path
            test_set.to_csv(
                test_output_path, index=False, header=True
            )

            logging.info("Ingestion of the data is completed")
            logging.info(f"Training and test data saved at: {train_output_path} and {test_output_path}")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys) from e



if __name__ == "__main__":
    ingestion = DataIngestion()
    output_data_path = ingestion.initiate_data_ingestion()
    print(f"Training and test data saved at: {output_data_path}")
