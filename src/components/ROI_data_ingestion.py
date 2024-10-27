"""
roi_data_pipeline.py

This module handles the configuration, ingestion, and transformation of training, testing, and raw data files.
It defines a DataIngestionConfig class for setting default paths and imports necessary components
for data transformation, model training, exception handling, logging, and printing bankruptcy 
outcomes.

Classes:
- DataIngestionConfig: Configuration class for data paths.
- DataIngestion: Class for ingesting and cleaning data.

Methods:
- __init__(): Initializes with default configuration.
- clean_dataset1(df): Cleans dataset 1 by removing duplicates and calculating the click-to-revenue ratio.
- clean_dataset2(df): Cleans dataset 2 by dropping null columns and calculating leads.
- initiate_data_ingestion(): Reads, cleans, and combines datasets for further processing.

Note:
- This module imports components from src.components for data transformation (DataTransformation,
  DataTransformationConfig), model training (ModelTrainer, ModelTrainerConfig), exception handling
  (CustomException), logging (logging), and utility functions for printing bankruptcy outcomes
  (print_bankruptcy_outcome).
- It uses standard library modules such as os, sys, dataclasses, and pathlib, as well as third-party
  modules like sklearn.model_selection.train_test_split.
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


# Setting the project root path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


@dataclass
class DataIngestionConfig:
    """
    Configuration class for data paths used in data ingestion processes.
    """
    dataset1_path: str = os.path.join("data", "Marketing.csv")
    dataset2_path: str = os.path.join("data", "online_advertising_performance_data.csv")
    combined_data_path: str = os.path.join("artifacts", "roi_model_data.csv")


class DataIngestion:
    """
    DataIngestion class handles the ingestion and cleaning of raw data,
    and prepares it for downstream tasks like model training.

    Methods:
    - __init__(): Initializes with default configuration.
    - initiate_data_ingestion(): Reads, cleans, and combines datasets for further processing.
    """

    def __init__(self):
        """
        Initializes a DataIngestion instance with default configuration.
        """
        self.ingestion_config = DataIngestionConfig()

    def clean_dataset1(self, df):
        """
        Cleans dataset 1 by dropping duplicates and null values, and calculating the click-to-revenue ratio.
        """
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        df['click_to_revenue_ratio'] = df['revenue'] / df['clicks']
        df_cleaned = df.drop(columns=["id", "c_date", "campaign_name", "campaign_id"])
        return df_cleaned.rename(columns={"mark_spent": "cost"})

    def clean_dataset2(self, df):
        """
        Cleans dataset 2 by dropping all-null columns, calculating leads, and renaming columns.
        """
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_cleaned = df.dropna(axis=1, how='all').copy()

        base_conversion_rate = 0.025  # 2.5% of clicks turn into leads
        conversion_adjustment = np.random.uniform(0.9, 1.1, len(df_cleaned))  # Random adjustment between 90% to 110%

        # Create a synthetic 'leads' column based on clicks, budget, and random noise
        df_cleaned['leads'] = (df_cleaned['clicks'] * base_conversion_rate * conversion_adjustment).astype(int)

        # Drop specified columns
        df_cleaned = df_cleaned.drop(columns=["month", "day", "campaign_number", "user_engagement", "banner"])

        df_cleaned = df_cleaned.rename(columns={"placement": "category",
                                                "displays": "impressions",
                                                "post_click_conversions": "orders"})

        # Replace 'abc' with 'jkl' in the 'category' column
        df_cleaned['category'] = df_cleaned['category'].replace('abc', 'jkl')

        # Remove the 'revenue' column
        df_cleaned = df_cleaned.drop(columns=["revenue"])
        
        # Rename columns
        df_cleaned = df_cleaned.rename(columns={
            "post_click_sales_amount": "revenue",
        })

        df_cleaned['click_to_revenue_ratio'] = df_cleaned.apply(
            lambda row: row['revenue'] / row['clicks'] if row['clicks'] != 0 else 0, axis=1
        )

        # Replace category elements with new values
        df_cleaned['category'] = df_cleaned['category'].replace({
            'mno': 'social',
            'def': 'search',
            'ghi': 'influencer',
            'abc': 'media'
        })

        return df_cleaned

    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process, cleaning both datasets and combining them into one.
        """
        try:
            # Read datasets
            df1 = pd.read_csv(self.ingestion_config.dataset1_path)
            logging.info(f"Dataset 1 loaded from {self.ingestion_config.dataset1_path}. Shape: {df1.shape}")

            df2 = pd.read_csv(self.ingestion_config.dataset2_path)
            logging.info(f"Dataset 2 loaded from {self.ingestion_config.dataset2_path}. Shape: {df2.shape}")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            # Clean datasets
            df1_cleaned = self.clean_dataset1(df1)
            df2_cleaned = self.clean_dataset2(df2)

            # Combine datasets
            common_columns = ["category", "impressions", "cost", "clicks", "leads", "orders", "revenue", "click_to_revenue_ratio"]

            df_combined = pd.concat([
                df1_cleaned[common_columns].reset_index(drop=True), 
                df2_cleaned[common_columns].reset_index(drop=True)
            ], axis=0, ignore_index=True)

            # Save the combined data
            output_path = self.ingestion_config.combined_data_path 
            df_combined.to_csv(output_path, index=False)
            logging.info("Combined data saved successfully.")
            logging.info(f"Data ingestion and transformation for ROI model completed successfully. Combined data saved at: {output_path}")
            
            logging.info("Selecting features and targets from the dataset.")
            X = df_combined[['category', 'cost']]  # Input features
            y = df_combined[['clicks', 'leads', 'orders']]  # Targets

            # One-hot encoding for 'category'
            encoder = OneHotEncoder(sparse_output=False)
            X_encoded = encoder.fit_transform(X[['category']])
            
            # Concatenate the encoded category with cost
            X_transformed = np.concatenate([X_encoded, X[['cost']].values], axis=1)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df_combined, test_size=0.2, random_state=42)
            
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,
            )
        
        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise CustomException(e, sys)





if __name__ == "__main__":
    ingestion = DataIngestion()
    combined_data_path = ingestion.initiate_data_ingestion()
    print(f"Combined data saved at: {combined_data_path}")
