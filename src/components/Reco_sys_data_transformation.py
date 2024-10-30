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

from notebook.data_cleaning import AsDiscrete, map_class_labels
# Custom imports
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, read_column_mapping

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation operations.
    """

    preprocessor_ob_file_path = os.path.join("artifacts", "reco_sys_preprocessor.pkl")
    clean_train_file_path = os.path.join("artifacts", "cleaned_santander_train.csv")
    clean_test_file_path = os.path.join("artifacts", "cleaned_santander_test.csv")




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
            numerical_columns = list(range(0, 64))
            categorical_columns = [64]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", MinMaxScaler()),
                ]
            )
            cat_pipeline = Pipeline(
                ("label_mapping", FunctionTransformer(map_class_labels)),
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e

    def process_csv(csv_file_path, output_file_path, column_mapping, columns_to_drop):
    # Read the original CSV file
    df = pd.read_csv(csv_file_path)

    # Drop columns if they exist in the DataFrame
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Rename the columns to English
    df.rename(columns=column_mapping, inplace=True)

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_file_path, index=False)
    print(f"CSV file successfully relabelled and saved to {output_file_path}")
    return df

    # Load column mapping from JSON
    column_mapping = read_column_mapping("column_mapping.json")

    # List of columns to drop if they exist
    columns_to_drop = ['ult_fec_cli_1t', 'ind_actividad_cliente', 'cod_prov', 'conyuemp', 'tipodom']

    # Process the training and testing datasets
    train_df = process_csv(train_file, cleaned_train_file, column_mapping, columns_to_drop)
    test_df = process_csv(test_file, cleaned_test_file, column_mapping, columns_to_drop)

    # In[16]:

    # Making changes to the Spanish data file to suit our needs, the simplified dataset is the one we assume is collected by companies.
    def create_additional_columns(df):
        # Define new column names
        fixed_deposits_col = 'fixed_deposits'
        loan_col = 'loan'
        credit_card_debit_card_col = 'credit_card_debit_card'
        account_col = 'account'

        # Check and create new columns as needed
        if fixed_deposits_col not in df.columns:
            deposit_columns = [
                "short_term_deposits",  # ind_deco_fin_ult1
                "medium_term_deposits",  # ind_deme_fin_ult1
                "long_term_deposits"    # ind_dela_fin_ult1
            ]
            df[fixed_deposits_col] = df[deposit_columns].any(axis=1).astype(int)

        if loan_col not in df.columns:
            loan_columns = [
                "loans",                # ind_pres_fin_ult1
                "pensions"             # ind_plan_fin_ult1
            ]
            df[loan_col] = df[loan_columns].any(axis=1).astype(int)

        if credit_card_debit_card_col not in df.columns:
            credit_card_columns = [
                "credit_card",         # ind_tjcr_fin_ult1
                "direct_debit"        # ind_recibo_ult1
            ]
            df[credit_card_debit_card_col] = df[credit_card_columns].any(axis=1).astype(int)

        if account_col not in df.columns:
            account_columns = [
                "saving_account",      # ind_ahor_fin_ult1
                "current_account",     # ind_cco_fin_ult1
                "derivada_account",    # ind_cder_fin_ult1
                "payroll_account",     # ind_cno_fin_ult1
                "junior_account",      # ind_ctju_fin_ult1
                "more_particular_account",  # ind_ctma_fin_ult1
                "particular_account",   # ind_ctop_fin_ult1
                "particular_plus_account", # ind_ctpp_fin_ult1
                "e_account",           # ind_ecue_fin_ult1
                "funds",               # ind_fond_fin_ult1
                "home_account",        # ind_viv_fin_ult1
            ]
            df[account_col] = df[account_columns].any(axis=1).astype(int)
        
        # List of columns to drop for simplicity
        columns_to_drop = [
            'saving_account', 'guarantee', 'current_account', 'derivada_account', 'payroll_account', 
            'junior_account', 'more_particular_account', 'particular_account', 'particular_plus_account', 
            'short_term_deposits', 'medium_term_deposits', 'long_term_deposits', 'e_account', 'funds', 
            'mortgage', 'pensions', 'loans', 'taxes', 'credit_card', 'securities', 'home_account', 
            'payroll', 'pensions_payments', 'direct_debit'
        ]
        # Drop the columns if they exist
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        return df

    # Apply to both train and test DataFrames
    train_df = create_additional_columns(train_df)
    test_df = create_additional_columns(test_df)

    # ### Prepare the data for machine learning, all columns are converted to numerical

    # In[10]:

    def prepare_data_for_ml(df):
        date_columns = ['report_date', 'contract_start_date']  # Add any other date columns if necessary

        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

        # Calculate the difference in days
        df['contract_length'] = (df['report_date'] - df['contract_start_date']).dt.days

        # Insert 'contract_length' in the same spot as 'contract_start_date'
        start_date_index = df.columns.get_loc('contract_start_date')
        df.insert(start_date_index, 'contract_length', df.pop('contract_length'))

        # Drop any remaining unnecessary columns or further process as needed
        # ...

        return df

    # Prepare both train and test DataFrames for ML
    train_df = prepare_data_for_ml(train_df)
    test_df = prepare_data_for_ml(test_df)
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
            
            #logging.info("Obtaining preprocessing object")
            #preprocessing_obj = self.get_data_transformer_object()

            logging.info("Drop columns")
            columns_to_drop = ['ult_fec_cli_1t', 'ind_actividad_cliente', 'cod_prov', 'conyuemp', 'tipodom']
            train_df.drop(columns=[col for col in columns_to_drop if col in train_df.columns], inplace=True)
            test_df.drop(columns=[col for col in columns_to_drop if col in test_df.columns], inplace=True)

            logging.info("Rename columns")
            column_mapping = read_column_mapping("column_mapping.json")
            train_df.rename(columns=column_mapping)
            test_df.rename(columns=column_mapping)

            logging.info("Group products into 4 main groups")
            # Fixed deposits
            if 'fixed_deposits' not in train_df.columns:
                train_df['fixed_deposits'] = train_df[
                    ["short_term_deposits", "medium_term_deposits", "long_term_deposits"]
                ].any(axis=1).astype(int)

            if 'fixed_deposits' not in test_df.columns:
                test_df['fixed_deposits'] = test_df[
                    ["short_term_deposits", "medium_term_deposits", "long_term_deposits"]
                ].any(axis=1).astype(int)

            # Loans
            if 'loan' not in train_df.columns:
                train_df['loan'] = train_df[["loans", "pensions"]].any(axis=1).astype(int)

            if 'loan' not in test_df.columns:
                test_df['loan'] = test_df[["loans", "pensions"]].any(axis=1).astype(int)

            # Credit/debit cards
            if 'credit_card_debit_card' not in train_df.columns:
                train_df['credit_card_debit_card'] = train_df[["credit_card", "direct_debit"]].any(axis=1).astype(int)

            if 'credit_card_debit_card' not in test_df.columns:
                test_df['credit_card_debit_card'] = test_df[["credit_card", "direct_debit"]].any(axis=1).astype(int)

            # Accounts
            if 'account' not in train_df.columns:
                train_df['account'] = train_df[
                    [
                        "saving_account", "current_account", "derivada_account",
                        "payroll_account", "junior_account", "more_particular_account",
                        "particular_account", "particular_plus_account", "e_account", "funds", "home_account"
                    ]
                ].any(axis=1).astype(int)

            if 'account' not in test_df.columns:
                test_df['account'] = test_df[
                    [
                        "saving_account", "current_account", "derivada_account",
                        "payroll_account", "junior_account", "more_particular_account",
                        "particular_account", "particular_plus_account", "e_account", "funds", "home_account"
                    ]
                ].any(axis=1).astype(int)

            # List of columns to drop for simplicity
            columns_to_drop = [
                'saving_account', 'guarantee', 'current_account', 'derivada_account', 'payroll_account', 
                'junior_account', 'more_particular_account', 'particular_account', 'particular_plus_account', 
                'short_term_deposits', 'medium_term_deposits', 'long_term_deposits', 'e_account', 'funds', 
                'mortgage', 'pensions', 'loans', 'taxes', 'credit_card', 'securities', 'home_account', 
                'payroll', 'pensions_payments', 'direct_debit'
            ]

            # Drop the original columns used to create new features
            train_df.drop(columns=[
                "saving_account", "current_account", "derivada_account", "payroll_account",
                "junior_account", "more_particular_account", "particular_account",
                "particular_plus_account", "short_term_deposits", "medium_term_deposits",
                "long_term_deposits", "e_account", "funds", "mortgage", "pensions", "loans",
                "taxes", "credit_card", "direct_debit"
            ], inplace=True)

            test_df.drop(columns=[
            "saving_account", "current_account", "derivada_account", "payroll_account",
            "junior_account", "more_particular_account", "particular_account",
            "particular_plus_account", "short_term_deposits", "medium_term_deposits",
            "long_term_deposits", "e_account", "funds", "mortgage", "pensions", "loans",
            "taxes", "credit_card", "direct_debit"
            ], inplace=True)










            logging.info("Cleaned train and test data saved to CSV files")
            train_df.to_csv(self.data_transformation_config.clean_train_file_path, index=False)
            test_df.to_csv(self.data_transformation_config.clean_test_file_path, index=False)

            # Save the cleaned DataFrames to CSV files (optional)








            input_feature_train_df = train_df.drop(
                train_df.columns[target_column_index], axis=1, inplace=False
            )
            target_feature_train_df = train_df.iloc[:, target_column_index]

            input_feature_test_df = test_df.drop(
                train_df.columns[target_column_index], axis=1, inplace=False
            )
            target_feature_test_df = test_df.iloc[:, target_column_index]

            
            
            
            
            
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
