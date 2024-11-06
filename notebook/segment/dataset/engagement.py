import pandas as pd
import numpy as np

import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class Engagement():
    def __init__(self, engine, task, col_type = 'market', y_col = 'conversion'):
        self.engine = engine
        query = """
            SELECT c.*, e.customer_id, e.engagement_date, 
            e.action_type, e.device_type, e.feedback_score,
            e.conversion_value
            FROM campaign c, engagement e
            WHERE c.campaign_id = e.campaign_id;
            """
        with engine.connect() as db:
            query_string = sqlalchemy.text(query)
            fetched = pd.DataFrame(db.execute(query_string).fetchall())
            db.close()
        self.df = fetched
    
    def preprocess(self):
        data = self.df.copy()
        data = data.drop(['campaign_name', 'budget', 'customer_id',
                          'displays'], axis = 1)
        return data
    
    def get_num_cols(self):
        data = self.preprocess()
        num_cols = data.select_dtypes([np.number]).columns
        return num_cols
    
    def get_cat_col(self):
        data = self.preprocess()
        cat_cols = data.select_dtypes(['category', 'object']).columns
        return cat_cols
    
    def get_dat_cols(self):
        data = self.preprocess()
        dat_cols = data.select_dtypes(['datetime64']).columns
        return dat_cols
    
    def get_X(self, col_type ='market'):
        data = self.preprocess()
        X = data.drop(['action_type'], axis = 1)
        return X
    
    def get_y(self):
        data = self.preprocess()
        return data[['action_type']]

# def create_db(user="root", password="Chenlu1974", server="localhost", database="transact"):
#     SQLALCHEMY_DATABASE_URL = "mysql+pymysql://{}:{}@{}/{}".format(
#         user, password, server, database
#     )
#     engine = create_engine(SQLALCHEMY_DATABASE_URL)

#     SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
#     Base = declarative_base()

#     return engine, SessionLocal, Base

# engine, SessionLocal, Base = create_db()

# eg = Engagement(engine, task='classification', col_type = 'market', y_col = 'conversion')
# data = eg.preprocess()
# print(eg.get_X())
# print(eg.get_y())
