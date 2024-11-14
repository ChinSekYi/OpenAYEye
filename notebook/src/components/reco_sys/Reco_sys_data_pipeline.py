"""
Reco_sys_data_pipeline.py

This module performs data ingestion, cleaning, and transformation for a recommendation system dataset.

Classes:
- DataIngestionConfig: Configuration class defining paths for raw, train, test data, and column mapping.
- DataIngestion: Handles ingestion, data cleaning, train-test split, and transformation.

Main Methods:
- DataIngestion.__init__(): Initializes the data ingestion with default configuration paths.
- DataIngestion.initiate_data_ingestion(): Loads data, applies transformations, splits data, and performs SMOTE on training data.

Execution:
- Running this script directly initiates the data ingestion process, performs transformations, and outputs the paths for train and test data files.

Output:
- File paths for transformed training and test data in 'artifacts' directory.

Note:
- This module depends on several external components for data processing, logging, and exception handling.
- Standard libraries and third-party modules like pandas and sklearn are utilized for efficient data handling.
"""


import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.reco_sys.data_processing_utils import *
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

    train_data_path: str = os.path.join("src/artifacts", "reco_sys_train_data.csv")
    test_data_path: str = os.path.join("src/artifacts", "reco_sys_test_data.csv")
    raw_data_path: str = os.path.join("../data/src/data", "recodataset.csv")
    column_mapping_path: str = os.path.join(
        "src", "components", "reco_sys", "column_mapping.json"
    )


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
        logging.info("Initializing DataIngestion class.")
        self.ingestion_config = DataIngestionConfig()
        logging.debug(f"Ingestion configuration: {self.ingestion_config}")

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
            logging.info("Read the dataset as dataframe.")
            df = pd.read_csv(self.ingestion_config.raw_data_path)

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            column_mapping = read_column_mapping(
                self.ingestion_config.column_mapping_path
            )

            logging.info("Perform data cleaning required before SMOTE.")
            df = process_csv(self.ingestion_config.raw_data_path, column_mapping)
            df = create_additional_columns(df)
            df = prepare_data_for_ml(df)

            logging.info("Train test split initiated.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            logging.info("Perform SMOTE on training dataset only.")
            train_set_ = perform_SMOTE(train_set)

            logging.info("Ingestion and transformation of the data is completed.")
            train_output_path = self.ingestion_config.train_data_path
            train_set_.to_csv(train_output_path, index=False, header=True)

            test_output_path = self.ingestion_config.test_data_path
            test_set.to_csv(test_output_path, index=False, header=True)

            logging.info(
                f"Training and test data saved at: {train_output_path} and {test_output_path}."
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    ingestion = DataIngestion()
    output_data_path = ingestion.initiate_data_ingestion()
    print(
        f"Data ingestion, cleaning and transformation completed for: {ingestion.ingestion_config.raw_data_path}"
    )
    print(f"Training and test data saved at: {output_data_path}")
