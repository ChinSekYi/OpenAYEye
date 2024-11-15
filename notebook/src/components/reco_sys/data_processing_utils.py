"""
data_preprocessing_utils.py

This module contains helper functions for preprocessing data in the Reco_sys_data_pipeline.

Functions:
    - prepare_data_for_ml: Preprocesses data by handling date conversion, missing values, and encoding.
    - perform_SMOTE: Balances the dataset classes using Synthetic Minority Over-sampling (SMOTE).
    - create_additional_columns: Adds aggregated product indicators (e.g., loan, account).
    - process_csv: Loads, cleans, and renames columns in the input CSV according to a mapping.

Usage:
    df = prepare_data_for_ml(df)
    df = perform_SMOTE(df)
    df = create_additional_columns(df)
    df = process_csv("file.csv", column_mapping)
"""

from statistics import median

import pandas as pd
from imblearn.over_sampling import SMOTE
from src.logger import logging

column_order = [
    "age",
    "gender_H",
    "gender_V",
    "gross_income",
    "customer_segment_03 - UNIVERSITARIO",
    "customer_segment_02 - PARTICULARES",
    "customer_segment_01 - TOP",
    "contract_length",
    "seniority_months",
    "primary_customer_status",
    "new_customer_index_0.0",
    "new_customer_index_1.0",
    "customer_type_start_month",
    "country_residence_ES",
    "country_residence_AT",
    "country_residence_NL",
    "country_residence_FR",
    "country_residence_GB",
    "country_residence_CL",
    "country_residence_CH",
    "country_residence_DE",
    "country_residence_DO",
    "country_residence_BE",
    "country_residence_AR",
    "country_residence_VE",
    "country_residence_US",
    "country_residence_MX",
    "country_residence_BR",
    "country_residence_IT",
    "country_residence_EC",
    "country_residence_PE",
    "country_residence_CO",
    "country_residence_HN",
    "country_residence_FI",
    "country_residence_SE",
    "country_residence_AL",
    "country_residence_PT",
    "country_residence_MZ",
    "country_residence_CN",
    "country_residence_TW",
    "country_residence_PL",
    "country_residence_IN",
    "country_residence_CR",
    "country_residence_NI",
    "region_CENTRAL",
    "region_NORTH",
    "region_SOUTH",
    "region_EAST",
    "region_WEST",
    "join_channel_KHE",
    "join_channel_KHD",
    "join_channel_KFA",
    "join_channel_KHC",
    "join_channel_KAT",
    "join_channel_KFC",
    "join_channel_KAZ",
    "join_channel_RED",
    "join_channel_KDH",
    "join_channel_KHK",
    "join_channel_KEH",
    "join_channel_KAD",
    "join_channel_KBG",
    "join_channel_KHL",
    "join_channel_KGC",
    "join_channel_KHF",
    "join_channel_KFK",
    "join_channel_KHN",
    "join_channel_KHA",
    "join_channel_KHM",
    "join_channel_KAF",
    "join_channel_KGX",
    "join_channel_KFD",
    "join_channel_KAG",
    "join_channel_KFG",
    "join_channel_KAB",
    "join_channel_KCC",
    "join_channel_KAE",
    "join_channel_KAH",
    "join_channel_KAR",
    "join_channel_KFJ",
    "join_channel_KFL",
    "join_channel_KAI",
    "join_channel_KFU",
    "join_channel_KAQ",
    "join_channel_KFS",
    "join_channel_KAA",
    "join_channel_KFP",
    "join_channel_KAJ",
    "join_channel_KFN",
    "join_channel_KGV",
    "join_channel_KGY",
    "join_channel_KFF",
    "join_channel_KAP",
    "deceased_index_N",
    "deceased_index_S",
    "foreigner_index_S",
    "foreigner_index_N",
    "residence_index_S",
    "residence_index_N",
    "customer_relation_type_I",
    "customer_relation_type_A",
    "fixed_deposits",
    "loan",
    "credit_card_debit_card",
    "account",
]


def prepare_data_for_ml(df):
    # Specify date columns for conversion
    logging.info(f"Preparing data for machine learning. {len(df.columns)}")
    date_columns = [
        "report_date",
        "contract_start_date",
    ]  # Add any other date columns if necessary

    # Convert date columns to datetime
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        logging.debug(f"Converted {col} to datetime.")

    # Calculate the difference in days for contract length
    df["contract_length"] = (df["report_date"] - df["contract_start_date"]).dt.days
    logging.debug("Calculated contract_length.")

    # Insert 'contract_length' in the same spot as 'contract_start_date'
    start_date_index = df.columns.get_loc("contract_start_date")
    df.insert(start_date_index, "contract_length", df.pop("contract_length"))

    # Drop the original date columns and 'customer_id'
    df = df.drop(["contract_start_date", "report_date", "customer_id"], axis="columns")
    logging.debug("Dropped unnecessary columns.")

    # Replace missing values in 'gross_income' with the median
    df["gross_income"] = df["gross_income"].fillna(df["gross_income"].median())
    logging.debug("Replaced missing values in 'gross_income'.")

    # Handling 'age' column
    df["age"] = pd.to_numeric(df["age"].replace(" NA", None), errors="coerce")
    med_age = df["age"].median()
    df["age"] = df["age"].fillna(med_age).astype(int)
    logging.debug(f"Replaced missing values in 'age' with median: {med_age}.")

    # Function to strip leading and trailing spaces from string columns
    def strip_spaces(column):
        if column.dtype == "object":  # Check if the column is of string type
            return column.str.strip()  # Strip leading and trailing spaces
        return column

    # Apply the strip_spaces function to all columns in the DataFrame
    df = df.apply(strip_spaces)

    columns_with_na = [
        "country_residence",
        "gender",
        "customer_relation_type",
        "residence_index",
        "foreigner_index",
        "join_channel",
        "deceased_index",
        "customer_segment",
        "new_customer_index",
    ]

    # Drop rows with NA values in the specified columns
    df = df.dropna(subset=columns_with_na)
    logging.debug("Dropped rows with missing values.")

    # Convert 'seniority_months' to integers
    df["seniority_months"] = df["seniority_months"].astype(int)
    logging.debug("Converted 'seniority_months' to integer.")

    # Create a new 'region' column based on 'province_name'
    region = []
    for province in df["province_name"]:
        if province in [
            "CIUDAD REAL",
            "SALAMANCA",
            "TOLEDO",
            "SEGOVIA",
            "MADRID",
            "GUADALAJARA",
            "ALBACETE",
            "SORIA",
            "CUENCA",
            "AVILA",
        ]:
            region.append("CENTRAL")
        elif province in [
            "ALAVA",
            "GIPUZKOA",
            "PALENCIA",
            "BURGOS",
            "NAVARRA",
            "CANTABRIA",
            "BIZKAIA",
            "RIOJA, LA",
            "ZARAGOZA",
            "TARRAGONA",
            "LERIDA",
            "HUESCA",
        ]:
            region.append("NORTH")
        elif province in [
            "CADIZ",
            "JAEN",
            "SEVILLA",
            "PALMAS, LAS",
            "CORDOBA",
            "GRANADA",
            "SANTA CRUZ DE TENERIFE",
            "MELILLA",
            "CEUTA",
            "MALAGA",
        ]:
            region.append("SOUTH")
        elif province in [
            "VALENCIA",
            "TERUEL",
            "BALEARS, ILLES",
            "CASTELLON",
            "ALICANTE",
            "MURCIA",
            "ALMERIA",
            "BARCELONA",
            "GIRONA",
        ]:
            region.append("EAST")
        elif province in [
            "ZAMORA",
            "CACERES",
            "HUELVA",
            "BADAJOZ",
            "ASTURIAS",
            "LEON",
            "LUGO",
            "CORUÑA, A",
            "OURENSE",
            "VALLADOLID",
            "PONTEVEDRA",
        ]:
            region.append("WEST")
        else:
            region.append(None)  # Append None for unmatched provinces

    # Assign the new region list to the DataFrame
    df["region"] = region
    df = df.drop(columns=["province_name"])  # Drop the original 'province_name' column
    logging.debug(f"Dropped original 'province_name' column.{len(df.columns)}")

    # One-hot encode categorical variables
    df = pd.get_dummies(
        df,
        columns=["customer_segment", "region", "join_channel", "country_residence"],
        drop_first=False,
        dtype=int,
    )
    df = pd.get_dummies(
        df,
        columns=[
            "deceased_index",
            "foreigner_index",
            "residence_index",
            "customer_relation_type",
            "gender",
            "new_customer_index",
        ],
        drop_first=False,
        dtype=int,
    )
    logging.debug("Performed one-hot encoding on categorical variables.")

    # Reorder columns if needed (maintain original logic)
    cols = df.columns.tolist()
    cols = cols[:5] + cols[9:] + cols[5:9]  # Adjust indexing as needed
    df = df[cols]
    logging.debug("Reordered columns.")

    # Move 'account' column to the last position
    account_column = df.pop("account")
    df["account"] = account_column
    logging.debug("Moved 'account' column to the last position.")

    df = df[column_order]

    return df


def perform_SMOTE(df):
    logging.info("Performing SMOTE to balance the dataset.")

    # Initialize the new balanced DataFrame
    balanced_df = pd.DataFrame()

    # Define the label columns to balance
    label_columns = ["fixed_deposits", "loan", "credit_card_debit_card", "account"]

    # Set the maximum number of samples to take from any class
    number_of_each_class = (
        15000  # Change according to desired size of original dataset,
    )
    max_samples = number_of_each_class  # Adjust as needed

    # Loop over each label column to balance it individually
    for label in label_columns:
        logging.debug(f"Balancing class for label: {label}")
        # Separate the current label and the features
        y = df[label]
        X = df.drop(columns=label_columns)  # Keep all features but exclude other labels

        # Prepare the data to balance the 0 and 1 classes for the current label
        class_0 = df[df[label] == 0]
        class_1 = df[df[label] == 1]

        # Take a max of 'max_samples' or the available samples for each class
        sampled_class_0 = class_0.sample(
            n=min(len(class_0), max_samples), random_state=42
        )
        sampled_class_1 = class_1.sample(
            n=min(len(class_1), max_samples), random_state=42
        )

        # Combine the samples to form the data for SMOTE
        df_to_balance = pd.concat([sampled_class_0, sampled_class_1], ignore_index=True)

        # Separate features and the label for SMOTE
        X_balance = df_to_balance.drop(columns=label)
        y_balance = df_to_balance[label]

        # Apply SMOTE to balance the current label
        smote = SMOTE(sampling_strategy="auto", random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_balance, y_balance)

        # Create a DataFrame from the resampled data
        resampled_df = pd.DataFrame(X_resampled, columns=X_balance.columns)
        # resampled_df[label] = y_resampled  # Add the resampled label back

        # Combine the features and the resampled label into one DataFrame
        resampled_df = pd.concat(
            [resampled_df, pd.Series(y_resampled, name=label)], axis=1
        )

        # Append the resampled data to the balanced_df
        balanced_df = pd.concat([balanced_df, resampled_df], ignore_index=True)

    # Shuffle the final balanced DataFrame
    balanced_df = (
        balanced_df.sample(frac=1, random_state=42)
        .reset_index(drop=True)
        .head(number_of_each_class * 2 * len(label_columns))
    )

    # Define the target columns to move to the front
    columns_to_move = ["fixed_deposits", "loan", "credit_card_debit_card", "account"]

    # Remove the target columns from the DataFrame
    remaining_columns = [
        col for col in balanced_df.columns if col not in columns_to_move
    ]

    # Add the target columns back to the front
    balanced_df = balanced_df[columns_to_move + remaining_columns]
    logging.info("SMOTE completed and dataset balanced.")
    return balanced_df


def create_additional_columns(df):
    # Define new column names
    fixed_deposits_col = "fixed_deposits"
    loan_col = "loan"
    credit_card_debit_card_col = "credit_card_debit_card"
    account_col = "account"

    # Check and create new columns as needed
    if fixed_deposits_col not in df.columns:
        deposit_columns = [
            "short_term_deposits",  # ind_deco_fin_ult1
            "medium_term_deposits",  # ind_deme_fin_ult1
            "long_term_deposits",  # ind_dela_fin_ult1
        ]
        df[fixed_deposits_col] = df[deposit_columns].any(axis=1).astype(int)

    if loan_col not in df.columns:
        loan_columns = ["loans", "pensions"]  # ind_pres_fin_ult1  # ind_plan_fin_ult1
        df[loan_col] = df[loan_columns].any(axis=1).astype(int)

    if credit_card_debit_card_col not in df.columns:
        credit_card_columns = [
            "credit_card",  # ind_tjcr_fin_ult1
            "direct_debit",  # ind_recibo_ult1
        ]
        df[credit_card_debit_card_col] = df[credit_card_columns].any(axis=1).astype(int)

    if account_col not in df.columns:
        account_columns = [
            "saving_account",  # ind_ahor_fin_ult1
            "current_account",  # ind_cco_fin_ult1
            "derivada_account",  # ind_cder_fin_ult1
            "payroll_account",  # ind_cno_fin_ult1
            "junior_account",  # ind_ctju_fin_ult1
            "more_particular_account",  # ind_ctma_fin_ult1
            "particular_account",  # ind_ctop_fin_ult1
            "particular_plus_account",  # ind_ctpp_fin_ult1
            "e_account",  # ind_ecue_fin_ult1
            "funds",  # ind_fond_fin_ult1
            "home_account",  # ind_viv_fin_ult1
        ]
        df[account_col] = df[account_columns].any(axis=1).astype(int)

    # List of columns to drop for simplicity
    columns_to_drop = [
        "saving_account",
        "guarantee",
        "current_account",
        "derivada_account",
        "payroll_account",
        "junior_account",
        "more_particular_account",
        "particular_account",
        "particular_plus_account",
        "short_term_deposits",
        "medium_term_deposits",
        "long_term_deposits",
        "e_account",
        "funds",
        "mortgage",
        "pensions",
        "loans",
        "taxes",
        "credit_card",
        "securities",
        "home_account",
        "payroll",
        "pensions_payments",
        "direct_debit",
        "employee_index",
    ]

    # Drop the columns if they exist
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    return df


def process_csv(csv_file_path, column_mapping):
    df = pd.read_csv(csv_file_path)

    # List of columns to drop if they exist
    columns_to_drop = [
        "ult_fec_cli_1t",
        "ind_actividad_cliente",
        "cod_prov",
        "conyuemp",
        "tipodom",
    ]

    # Drop columns if they exist in the DataFrame
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Rename the columns to English
    df.rename(columns=column_mapping, inplace=True)

    return df
