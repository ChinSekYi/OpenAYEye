import pandas as pd
import numpy as np

import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


class Churn():
    def __init__(self, engine):
        self.engine = engine  
        query="""
            SELECT u.*, c.churn_date
            FROM users u
            LEFT JOIN churn c
            ON u.customer_id = c.customer_id;
            """
        with engine.connect() as db:
            query_string = sqlalchemy.text(query)
            fetched = pd.DataFrame(db.execute(query_string).fetchall())
            db.close()
        self.df = fetched

    def preprocess(self):
        data = self.df.copy()
        data = data.drop(['customer_id', 'person', 'retirement_age', 'birth_year_month',
                   'address', 'apartment', 'zipcode',
                   'latitude', 'longitude'], axis=1)
        data['churn'] = np.where(data['churn_date'].isna(), 0, 1)
        return data
    
    def get_num_cols(self):
        data = self.preprocess()
        num_cols = data.select_dtypes([np.number]).columns
        return num_cols
    
    def get_cat_cols(self):
        data = self.preprocess()
        cat_cols = data.select_dtypes(['category', 'object']).columns
        return cat_cols
    
    def get_dat_cols(self):
        data = self.preprocess()
        dat_cols = data.select_dtypes(['datetime64']).columns
        return dat_cols
    
    def get_X(self, col_type = 'market'):
        data = self.preprocess()
        X = data.drop(['churn', 'churn_date'], axis = 1)
        return X
    def get_y(self):
        data = self.preprocess()
        return data[['churn']]

def create_db(user="root", password="Chenlu1974", server="localhost", database="transact"):
    SQLALCHEMY_DATABASE_URL = "mysql+pymysql://{}:{}@{}/{}".format(
        user, password, server, database
    )
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    return engine, SessionLocal, Base

# engine, SessionLocal, Base = create_db()

# churn = Churn(engine, col_type='demo')
# churn = Churn(engine)
# print(churn.get_X())
# print(churn.get_y())
