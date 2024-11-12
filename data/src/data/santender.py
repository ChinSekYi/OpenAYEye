import re
import time
import random
from datetime import datetime, timedelta

import pandas as pd
import numpy as np


def to_lowercase(df):
	df = df.copy()
	df.rename({i:i.lower() for i in df.columns.values}, axis=1, inplace=True)
	return df

def to_snakecase(df):
	snakecase = {i: re.sub(r"[,.;@#?!&$]+\ *", "", i.strip()).replace(" ", "_") for i in df}
	df.rename(columns=snakecase, inplace=True)
	return df
	

def get_Products(path="data/recodataset.csv", 
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
	):
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
			'payroll', 'pensions_payments', 'direct_debit', 'employee_index'
		]

		# Drop the columns if they exist
		df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
		
		return df
	santender = pd.read_csv(path)
	santender = santender.rename(columns=column_mapping)
	santender['customer_id'] = santender['customer_id'].apply(lambda x: str(x).zfill(4))
	santender = to_snakecase(to_lowercase(santender))
	santender['report_date'] = pd.to_datetime(santender['report_date'])
	santender['contract_start_date'] = pd.to_datetime(santender['contract_start_date'])
	santender['last_primary_customer_date'] = pd.to_datetime(santender['last_primary_customer_date'])
	str_cols = santender.select_dtypes(include='object').columns
	santender[str_cols] = santender[str_cols].apply(lambda x: x.str.strip(), axis=1)
	santender[str_cols] = santender[str_cols].replace(regex=[r'NA'], value=None)
	santender['gender'] = santender['gender'].map({'H': 'Male', 'V':'Female'}).fillna(np.random.choice(['Male', 'Female'])).astype('category')
	santender['age'] = pd.to_numeric(santender['age'].replace(' NA', None), errors='coerce')
	med_age = santender['age'].median()
	santender['age'] = santender['age'].fillna(med_age).astype(int)
	santender['gross_income'] = pd.to_numeric(santender['gross_income'].replace(' NA', None), errors='coerce')
	med_age = santender['gross_income'].median()
	santender['gross_income'] = santender['gross_income'].fillna(med_age).astype(int)
	santender = create_additional_columns(santender)
	return santender

santender = get_Products()
