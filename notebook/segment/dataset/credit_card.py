import pandas as pd
import numpy as np

import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


class CreditCard():
    def __init__(self, engine, col_type = None): # Assume we take in a config
        self.engine = engine
        self.col_type = col_type # Add in the col type
        if self.col_type == 'demo':
            self.col = ['age', 'gender', 'income']
        elif self.col_type == 'market':
            self.col = ['campaigntype', 'campaignchannel', 'clickthroughrate', 'adspend', 'conversionrate']
        elif self.col_type == 'engage':
            self.col = ['pagespervisit', 'socialshares', 'timeonsite', 'emailopens', 'websitevisits', 'emailclicks']
        
        query1 = '''SELECT age, gender, gross_income, credit_card FROM santender;'''
        query2 = '''SELECT * FROM engagement;'''
        with engine.connect() as db:
            query_string = sqlalchemy.text(query1)
            fetched = pd.DataFrame(db.execute(query_string).fetchall())
            db.close()
        self.santender = fetched

        with engine.connect() as db:
            query_string = sqlalchemy.text(query2)
            fetched = pd.DataFrame(db.execute(query_string).fetchall())
            db.close()
        self.engagement = fetched

        self.preprocess()
        self.merge_df()
            


    def preprocess(self):
        # Recode gender
        self.santender['gender'] = self.santender['gender'].replace({'H': 'M', 'V': 'F'})
        self.engagement['gender'] = self.engagement['gender'].replace({'Male':'M', 'Female':'F'})

        # Remove columns
        self.santender.rename(columns={'gross_income':'income'}, inplace = True)

        # Change income into categorical
        bins = [0, 30000, 100000, 500000, 6000000]
        labels = [1,2,3,4]
        
        # Drop NA
        self.santender.dropna(subset=['age', 'gender', 'income'], inplace= True)
        self.engagement.dropna(subset=['age', 'gender', 'income'], inplace = True)
        self.santender['income_category'] = pd.cut(self.santender['income'], bins=bins, labels=labels, include_lowest=True)
        self.engagement['income_category'] = pd.cut(self.engagement['income'], bins=bins, labels=labels, include_lowest=True)

        # Change datatype
        self.santender['age'] = self.santender['age'].apply(int) 

    def merge_df(self):
        
        # Merge both datasets together based on demographics
        res = self.engagement.merge(self.santender, how = 'inner', on =['age', 'gender', 'income_category'])
        # print(res)
        # Keep distinct customerid
        res.drop_duplicates(subset=['customerid'], inplace = True)
        res.drop(['income_y'], axis=1)        
        res.rename(columns={'income_x':'income'}, inplace = True)
        self.df = res
        res.reset_index(inplace = True)
        # print(self.df)

    def get_X(self):
        if self.col_type == None:
            res = self.df.drop(['index', 'customerid', 'income_y', 'credit_card'], axis = 1)
            return res
        else:
            # dat = dat.rename(columns={'income_x':'income'}, inplace = True)
            return self.df[self.col]
    
    def get_y(self):
        return self.df[['credit_card']]
    
    def recode(self, X, y):
        # Get all the categorical data
        categorical_cols = X.select_dtypes(include=['category', 'object']).columns

        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col])

        y = le.fit_transform(y)
        return X, y

    def get_smote(self, X_train, X_test, y_train, y_test):
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        X_test, y_test = smote.fit_resample(X_test, y_test)
        return X_train, X_test, y_train, y_test

    def get_split(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

    def get_Dataset(self):
        X = self.get_X()
        y = self.get_y()
        X, y = self.recode(X, y)
        # print(self.get_split(X, y))
        X_train, X_test, y_train, y_test = self.get_split(X, y)
        X_train, X_test, y_train, y_test = self.get_smote(X_train, X_test, y_train, y_test)
        return X_train, X_test, y_train, y_test


# def create_db(user="root", password="Chenlu1974", server="localhost", database="transact"):
#     SQLALCHEMY_DATABASE_URL = "mysql+pymysql://{}:{}@{}/{}".format(
#         user, password, server, database
#     )
#     engine = create_engine(SQLALCHEMY_DATABASE_URL)

#     SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
#     Base = declarative_base()

#     return engine, SessionLocal, Base


# engine, SessionLocal, Base = create_db(password='msql1234')

# cc = CreditCard(engine, col_type='demo')
# cc = CreditCard(engine, col_type='market')
# cc = CreditCard(engine, col_type='engage')
# X_train, X_test, y_train, y_test = cc.get_Dataset()
# print(X_train)
# print(y_train)