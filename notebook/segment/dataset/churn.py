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
    def __init__(self, engine, col_type):
        self.engine = engine
        self.col_type = col_type
        if self.col_type == 'demo':
            self.cols = ['age', 'gender', 'income', 'geography']
        elif self.col_type == 'market':
            self.cols = ['isactivemember', 'numofproducts', 'balance',
                         'hascrcard', 'tenure', 'creditscore']
        
        query='''SELECT * FROM churn;'''
        with engine.connect() as db:
            query_string = sqlalchemy.text(query)
            fetched = pd.DataFrame(db.execute(query_string).fetchall())
            fetched = fetched.drop(['estimatedsalary'], axis = 1)
            db.close()
        self.df = self.preprocess(fetched)

    def get_X(self):
        if self.col_type == None:
            res = self.df.drop(['exited', 'surname', 'customerid'], axis = 1)
            return res
        else:
            return self.df[self.cols]

    def get_y(self):
        return self.df[['exited']]
    
    def preprocess(self, X, y):
        # Get all the categorical data
        categorical_cols = X.select_dtypes(include=['category', 'object']).columns

        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col])

        y['exited'] = le.fit_transform(y)
        return X, y

    def get_smote(self, X_train, X_test, y_train, y_test):
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        X_test, y_test = smote.fit_resample(X_test, y_test)
        return X_train, X_test, y_train, y_test

    def get_split(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def get_dataset(self):
        X = self.get_X()
        y = self.get_y()
        # X, y = self.recode(X, y)
        # print(self.get_split(X, y))
        X_train, X_test, y_train, y_test = self.get_split(X, y)
        X_train, X_test, y_train, y_test = self.get_smote(X_train, X_test, y_train, y_test)
        return X_train, X_test, y_train, y_test

    def get_data(self):
        return self.df
# def create_db(user="root", password="Chenlu1974", server="localhost", database="transact"):
#     SQLALCHEMY_DATABASE_URL = "mysql+pymysql://{}:{}@{}/{}".format(
#         user, password, server, database
#     )
#     engine = create_engine(SQLALCHEMY_DATABASE_URL)

#     SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
#     Base = declarative_base()

#     return engine, SessionLocal, Base

# engine, SessionLocal, Base = create_db(password='msql1234')

# # churn = Churn(engine, col_type='demo')
# churn = Churn(engine, col_type='market')
# X_train, X_test, y_train, y_test = churn.get_dataset()
# print(X_train)
# print(y_train)