#!/usr/bin/env python
# coding: utf-8

# ### Renaming the columns of the csv file to english readeable names, in addition the dataset used here is a subset of the original dataset, the original dataset train is 13,647,000 rows and test is 900,000 rows, loading it in will crash your computer. 
# ### Currently the train file in spanish is named sandenter_train_small and test is sandenter_test_small. The code uses sandenter_train_small as it has enough rows to do both train and testing

# In[13]:


#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from statistics import median


# In[14]:


## Set the names of the files to clean and the name of the cleaned files here:
old_file = "nuclear.csv"
cleaned_file = "cleaned_nuclear.csv"


# In[15]:


# Creating a new file to for the translated version, this is solely for visualisation purposes and is not used. 
# The original spanish verison will be used.

csv_file_path = old_file  # input CSV file
output_file_path = cleaned_file  # output CSV file

# Define the mapping of original Spanish column names to English column names
column_mapping = {
    "fecha_dato": "report_date",
    "ncodpers": "customer_id",
    "ind_empleado": "employee_index",
    "pais_residencia": "country_residence",
    "sexo": "gender",
    "age": "age",
    "fecha_alta": "contract_start_date",
    "ind_nuevo": "new_customer_index",
    "antiguedad": "seniority_months",
    "indrel": "primary_customer_status",
    "ult_fec_cli_1t": "last_primary_customer_date",
    "indrel_1mes": "customer_type_start_month",
    "tiprel_1mes": "customer_relation_type",
    "indresi": "residence_index",
    "indext": "foreigner_index",
    "conyuemp": "spouse_employee_index",
    "canal_entrada": "join_channel",
    "indfall": "deceased_index",
    "tipodom": "address_type",
    "cod_prov": "province_code",
    "nomprov": "province_name",
    "ind_actividad_cliente": "activity_index",
    "renta": "gross_income",
    "segmento": "customer_segment",
    "ind_ahor_fin_ult1": "saving_account",
    "ind_aval_fin_ult1": "guarantee",
    "ind_cco_fin_ult1": "current_account",
    "ind_cder_fin_ult1": "derivada_account",
    "ind_cno_fin_ult1": "payroll_account",
    "ind_ctju_fin_ult1": "junior_account",
    "ind_ctma_fin_ult1": "more_particular_account",
    "ind_ctop_fin_ult1": "particular_account",
    "ind_ctpp_fin_ult1": "particular_plus_account",
    "ind_deco_fin_ult1": "short_term_deposits",
    "ind_deme_fin_ult1": "medium_term_deposits",
    "ind_dela_fin_ult1": "long_term_deposits",
    "ind_ecue_fin_ult1": "e_account",
    "ind_fond_fin_ult1": "funds",
    "ind_hip_fin_ult1": "mortgage",
    "ind_plan_fin_ult1": "pensions",
    "ind_pres_fin_ult1": "loans",
    "ind_reca_fin_ult1": "taxes",
    "ind_tjcr_fin_ult1": "credit_card",
    "ind_valo_fin_ult1": "securities",
    "ind_viv_fin_ult1": "home_account",
    "ind_nomina_ult1": "payroll",
    "ind_nom_pens_ult1": "pensions_payments",
    "ind_recibo_ult1": "direct_debit"
}

# Read the original CSV file
df = pd.read_csv(csv_file_path)

#Drop the useless columns (correlation matrix used https://medium.com/@samarthjoelram/santander-recommendation-system-cab6b40596b5)
# List of columns to drop if they exist
columns_to_drop = ['ult_fec_cli_1t', 'ind_actividad_cliente', 'cod_prov', 'conyuemp', 'tipodom']
# Drop columns if they exist in the DataFrame
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Rename the columns to english
df.rename(columns=column_mapping, inplace=True)

# Save the updated DataFrame to a new CSV file
df.to_csv(output_file_path, index=False)

print(f"CSV file successfully relabelled and saved to {output_file_path}")


# In[16]:


#Making changes to spanish datafile to suit our needs, the simplified dataset is the one we assume is collected by 
#companies.
import pandas as pd

# Define the path to the renamed CSV file
csv_file_path = cleaned_file

# Read the renamed CSV file
df = pd.read_csv(csv_file_path)

# Define new column names
fixed_deposits_col = 'fixed_deposits'
loan_col = 'loan'
credit_card_debit_card_col = 'credit_card_debit_card'
account_col = 'account'

# Check and create a new column for fixed deposits, if it doesn't exist
if fixed_deposits_col not in df.columns:
    deposit_columns = [
        "short_term_deposits",  # ind_deco_fin_ult1
        "medium_term_deposits",  # ind_deme_fin_ult1
        "long_term_deposits"    # ind_dela_fin_ult1
    ]
    df[fixed_deposits_col] = df[deposit_columns].any(axis=1).astype(int)

# Check and create a new column for loans, if it doesn't exist
if loan_col not in df.columns:
    loan_columns = [
        "loans",                # ind_pres_fin_ult1
        "pensions"             # ind_plan_fin_ult1
    ]
    df[loan_col] = df[loan_columns].any(axis=1).astype(int)

# Check and create a new column for credit and debit cards, if it doesn't exist
if credit_card_debit_card_col not in df.columns:
    credit_card_columns = [
        "credit_card",         # ind_tjcr_fin_ult1
        "direct_debit"        # ind_recibo_ult1
    ]
    df[credit_card_debit_card_col] = df[credit_card_columns].any(axis=1).astype(int)

# Check and create a new column for all accounts combined, if it doesn't exist
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
    
# List of columns to drop, we merged these columns for to keep it simple to present 
# The model used for commercial purposes does not merge the products together
columns_to_drop = [
    'saving_account', 'guarantee', 'current_account', 'derivada_account', 'payroll_account', 
    'junior_account', 'more_particular_account', 'particular_account', 'particular_plus_account', 
    'short_term_deposits', 'medium_term_deposits', 'long_term_deposits', 'e_account', 'funds', 
    'mortgage', 'pensions', 'loans', 'taxes', 'credit_card', 'securities', 'home_account', 
    'payroll', 'pensions_payments', 'direct_debit'
]

# Drop the columns if they exist
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])


# ### Prepare the data for machine learning, all columns are converted to numerical

# In[10]:


date_columns = ['report_date', 'contract_start_date']  # Add any other date columns if necessary

for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Calculate the difference in days
df['contract_length'] = (df['report_date'] - df['contract_start_date']).dt.days

# Insert 'contract_length' in the same spot as 'contract_start_date'
start_date_index = df.columns.get_loc('contract_start_date')
df.insert(start_date_index, 'contract_length', df.pop('contract_length'))

# Drop the original 'contract_start_date' and 'report_date' columns
df = df.drop(['contract_start_date', 'report_date', 'customer_id'], axis='columns')
#replace missing values in gross income and age with median of distribution
count = df['gross_income'].isna().sum()
df['gross_income'] = df['gross_income'].fillna(df['gross_income'].median())

age_new = df.iloc[:,3]

age_new[1020:1040]
new = []
for i in age_new:
    if i != ' NA':
        new.append(int(i))

med = median(new)
med
age = []
for i in age_new:
    if i !=' NA':
        age.append(int(i))
    else:
        age.append(med)

age[1020:1040]
df['age'] = age

#has NA and many countries one hot encode
count = df['country_residence'].isna().sum()

#dropping row if column country_residenceis NA
df = df.dropna(subset=["country_residence"])


#conversion from string to int
new = []
for i in df['seniority_months']:
    new.append(int(i))
df['seniority_months'] = new


df = df.drop('employee_index', axis='columns')
df = pd.get_dummies(df, columns=['customer_segment','province_name','join_channel','country_residence'], dtype = int)
#one hot encoding for many variables
df = pd.get_dummies(df, columns=['deceased_index','foreigner_index','residence_index','customer_relation_type','gender','new_customer_index' ], drop_first=True, dtype = int)
# binary one hot encode
df
cols = df.columns.tolist()
cols
cols = cols[:5]+cols[9:] + cols[5:9]
df = df[cols]
# Move 'account' column to the last position
account_column = df.pop('account')  # Remove the column and save it
df['account'] = account_column

# Save the updated DataFrame back to the renamed CSV file, overwriting it
df.to_csv(csv_file_path, index=False)

print(f"CSV file successfully updated with new columns in {csv_file_path}")


