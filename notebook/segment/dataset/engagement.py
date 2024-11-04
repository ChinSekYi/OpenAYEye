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
        self.task = task
        self.col_type = 'market'
        self.y_col = y_col
        
        self.x_scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))

        if self.col_type == 'demo':
            self.col = ['age', 'gender', 'income']
        elif self.col_type == 'market':
            self.col = ['adspend']
        elif self.col_type == 'engage':
            self.col = ['conversion', 'clickthroughrate', 'websitevisits', 'timeonsite', 'socialshares', 'emailopens', 'emailclicks']

        query = '''SELECT * FROM engagement;'''
        with engine.connect() as db:
            query_string = sqlalchemy.text(query)
            fetched = pd.DataFrame(db.execute(query_string).fetchall())
            db.close()
        fetched['gender'] = fetched['gender'].replace({'Male':'M', 'Female':'F'})
        self.df = fetched
        # self.df = self.preprocess(fetched)

    def get_X(self, data):
        if self.col_type == None:
            res = self.df.drop(['customerid', 'conversionrate'], axis = 1)
            return res
        else:
            return data[self.col]
    
    def get_y(self, data):
        return data[[self.y_col]]

    def preprocess(self, data):
        categorical_cols = ['campaignchannel', 'campaigntype']
        numerical_cols = ['adspend', 'clickthroughrate', 'websitevisits', 'timeonsite', 'socialshares', 'emailopens', 'emailclicks']
        for col in categorical_cols:
            new_cols = pd.get_dummies(data[col], dtype=int)
            self.col += list(new_cols.columns)
            # print(self.col)
            # print(list(new_cols.columns))
            data = data.join(pd.get_dummies(data[col], dtype=int))
        data = data.drop(categorical_cols, axis=1)

        for col in numerical_cols:
            data[col] = self.x_scaler.fit_transform(data[[col]])
        return data

    def get_smote(self, X_train, X_test, y_train, y_test):
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        X_test, y_test = smote.fit_resample(X_test, y_test)
        return X_train, X_test, y_train, y_test

    def get_split(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def get_dataset(self, data): 
        data = self.preprocess(self.df)
        X = self.get_X(data)
        y = self.get_y(data)
        X_train, X_test, y_train, y_test = self.get_split(X, y)
        if self.task == 'classification': 
            X_train, X_test, y_train, y_test = self.get_smote(X_train, X_test, y_train, y_test)
        return X_train, X_test, y_train, y_test

    def get_data(self):
        return self.df


def create_db(user="root", password="Chenlu1974", server="localhost", database="transact"):
    SQLALCHEMY_DATABASE_URL = "mysql+pymysql://{}:{}@{}/{}".format(
        user, password, server, database
    )
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    return engine, SessionLocal, Base

engine, SessionLocal, Base = create_db()

eg = Engagement(engine, task='classification', col_type = 'market', y_col = 'conversion')
data = eg.get_data()
processed_data = eg.preprocess(data)
X = eg.get_X(processed_data)
y = eg.get_y(processed_data)
print(y)
# X_train, X_test, y_train, y_test = eg.get_dataset()
# print(y_train.value_counts())
# print(pd.concat([X_train, y_train], axis=1))

