# %%
import re
import time
import random
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")
from data import campaign, engagement, santender

random.seed(10)

# %% [markdown]
# # Helper Functions
# 

# %%
def to_lowercase(df):
	df = df.copy()
	df.rename({i:i.lower() for i in df.columns.values}, axis=1, inplace=True)
	return df

def to_snakecase(df):
	snakecase = {i: re.sub(r"[,.;@#?!&$]+\ *", "", i.strip()).replace(" ", "_") for i in df}
	df.rename(columns=snakecase, inplace=True)
	return df

# from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.linear_model import LogisticRegression

def get_est(df, y_col='fixed_deposits'):
	X = df[['age', 'gross_income']]
	y = df[y_col]
	# clf = XGBRFClassifier(random_state=42, enable_categorical=True)
	# clf = XGBRFClassifier(random_state=42, n_estimators=10, max_depth=3, enable_categorical=True)
	clf = LogisticRegression(random_state=42)
	clf.fit(X, y)
	return clf

	
def get_users(path="data/users.csv", date_format = "%Y-%m"):
	users = to_snakecase(to_lowercase(pd.read_csv(path)))
	users['user'] = users['user'].astype(str).str.pad(width=4, side='left', fillchar='0')
	users['birth_year'] = users['birth_year'].astype(str) + '-' + users['birth_month'].astype(str)
	users['birth_year'] = pd.to_datetime(users['birth_year'], format=date_format).astype('datetime64[ns]')

	users = users.drop(columns=['birth_month'])
	users = users.rename(columns={'user':'customer_id', 'birth_year': 'birth_year_month'})
	users['gender'] = users['gender'].astype('category')
	for i in ['fixed_deposits', 'credit_card_debit_card', 'account']:
		est = get_est(santender, y_col=i)
		test = users[['current_age', 'yearly_income']].rename(columns={'current_age': 'age', 'yearly_income':'gross_income'})
		users[i] = est.predict(test)
	users['loan'] = [random.choices([0,1], weights=[1-0.015, 0.015])[0] for _ in range(len(users))]
	users = users.rename(columns={'fixed_deposits':'deposits', 'credit_card_debit_card': 'cards'})
	users = users.drop(columns=['birth_year_month'])
	return users.loc[1:, :]


def get_creditcards(path="data/credit_cards.csv", date_format = "%m/%Y"):
	credit_cards = to_snakecase(to_lowercase(pd.read_csv(path)))
	credit_cards['user'] = credit_cards['user'].astype(str).str.pad(width=4, side='left', fillchar='0')
	credit_cards['expires'] = pd.to_datetime(credit_cards['expires'], format=date_format)
	credit_cards['acct_open_date'] = pd.to_datetime(credit_cards['expires'], format=date_format)
	credit_cards['year_pin_last_changed'] = pd.to_datetime(credit_cards['year_pin_last_changed'], format="%Y")
	return credit_cards

def get_transactions(path="data/transactions.csv", date_format = "%m/%Y"):
	transactions = to_snakecase(to_lowercase(pd.read_csv(path)))
	transactions.insert(0, 'identifier', transactions.index + 1) 
	transactions['user'] = transactions['user'].astype(str).str.pad(width=4, side='left', fillchar='0')
	transactions = transactions.rename(columns={'card':'card_index'})
	hour_min = transactions['time'].str.split(":", expand=True).rename(columns={0:'hour', 1:'minute'})
	transactions = pd.concat([transactions, hour_min], axis=1)

	date_cols = ['year', 'month', 'day', 'hour', 'minute']
	transactions['date'] = pd.to_datetime(transactions[date_cols])

	cc_no = get_creditcards()[['user', 'card_index', 'card_number']]
	card_no = transactions.merge(cc_no, how='inner', on=['user', 'card_index'])['card_number'].astype(str).str.pad(width=4, side='right', fillchar='0')
	transactions.insert(1, 'card_number', card_no) 
	transactions = transactions.drop(columns= ['card_index', 'time'] + date_cols)
	transactions = transactions.rename(columns={'user':'customer_id'})
	return transactions

def get_churn(users, 
        start_date = datetime(2023, 1, 1),
        end_date = datetime(2024, 10, 31)
    ):
    customer_ids = users["customer_id"].tolist()
    # Helper function to generate random churn date
    def random_churn_date():
        return start_date + timedelta(days=random.randint(0, (end_date - start_date).days))

    # Generate churn data
    churn_data = {
        "customer_id": [],
        # "has_churned": [],
        "churn_date": []
    }

    for customer_id in customer_ids:
        has_churned = random.random() < 0.2  # 10% churn rate
        if has_churned:
            churn_data["customer_id"].append(customer_id) 
            # churn_data["has_churned"].append(has_churned)
            churn_data["churn_date"].append(random_churn_date())

    # Create DataFrame
    churn = pd.DataFrame(churn_data)
    return churn




users = get_users()
transactions = get_transactions()
churn = get_churn(users)
# %% [markdown]
# # Connect to Database

# %%
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine


def create_db(user="root", password="msql1234", server="db", port="3306", database="transact"):
    SQLALCHEMY_DATABASE_URL = "mysql+pymysql://{}:{}@{}:{}/{}".format(
        user, password, server, port, database
    )
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    return engine, SessionLocal, Base

engine, SessionLocal, Base = create_db(server="db")

print(engine)
# %% [markdown]
# # Insert into Database

# %%
with engine.connect() as db:
	dct = {'users': users, 
		'transactions':transactions,
		'churn':churn, 
		'campaign': campaign,
		'engagement': engagement,
		}
	for k,v in dct.items():
		try:
			v.to_sql(k, con=engine, if_exists='append', index=False)
			db.commit()
			print("{} Ok".format(k))
		except:
			db.rollback()
			print("{} Failed".format(k))
	db.close()
# users.to_sql('users', con=engine, if_exists='append', index=False)
# with engine.connect() as db:
# 	query = sqlalchemy.text('''SELECT * FROM users ''')
# 	fetched = db.execute(query).fetchall()
# 	print(pd.DataFrame(fetched))
# 	db.close()