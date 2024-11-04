import pandas as pd
import numpy as np

import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE




class RFM():
    def __init__(self, engine, col_type, y_col):
        self.engine = engine   
        self.col_type = col_type   
        self.y_col = y_col
        self.x_scaler = MinMaxScaler(feature_range=(0, 1))

        if self.col_type == 'demo':
            self.col = ['age', 'gender', 'income']
        elif self.col_type == 'market':
            self.col = ['adspend', 'income', 'age']
        elif self.col_type == 'engage':
            self.col = ['conversionrate', 'pagespervisit', 'socialshares', 'timeonsite', 'emailopens', 'websitevisits', 'emailclicks']
        with engine.connect() as db:
            query_string = sqlalchemy.text(
                """SELECT u.*, t.date,
                t.amount, t.errors,
                t.use_chip, t.zip
                FROM users u, transactions t
                WHERE u.user = t.user
                ORDER BY t.user; """
            )
            fetched = pd.DataFrame(db.execute(query_string).fetchall())
            
            db.close()

        self.df = self.format(fetched)

        with engine.connect() as db:
            query_string = sqlalchemy.text(
                """
                SELECT *
                FROM engagement;
                """
            )
            fetched = pd.DataFrame(db.execute(query_string).fetchall())
            
            db.close()
        # print(fetched)
        self.df = self.get_RFM().merge(fetched, how='inner', on=['age', 'gender'])
        self.df = self.df.drop(['user'],axis=1)
        # print(self.df.columns)
        self.df = self.preprocess(self.df)


    def format(self, data):
        data['day'] = data['date'].dt.day
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year
        # data = data.drop(columns=['user', 'person'], axis=1)
        data.rename(columns={'current_age':'age'}, inplace = True)
        data = data[['user', 'age', 'gender', 'date', 'day', 'month', 'year', 'amount']]
        # categorical_cols = data.select_dtypes(include=['category', 'object']).columns

        return data

    def get_RFM(self, group=["age", "gender"], 
            seg_map = {
                r'[1-2][1-2]': 'Hibernating',
                r'[1-2][3-4]': 'At Risk',
                r'[1-2]5': 'Cannot Lose',
                r'3[1-2]': 'About to Sleep',
                r'33': 'Need Attention',
                r'[3-4][4-5]': 'Loyal Customers',
                r'41': 'Promising',
                r'51': 'New Customers',
                r'[4-5][2-3]': 'Potential Loyalists',
                r'5[4-5]': 'Champions'
            }
        ):
        data = self.df
        data = data.sort_values(by=group)
        recency = (data['date'].max() - data.groupby(['user'] +  group).agg({"date":"max"})).rename(columns = {"date":"recency"})
        recency['recency'] = recency['recency'].apply(lambda x: x.days)
        recency = recency.reset_index()
        frequency = (data.groupby(['user'] + group).agg({"date":"nunique"})).rename(columns = {"date":"frequency"}).reset_index()
        # print(frequency)
        monetary = data.groupby(["user"] + group).agg({"amount":"sum"}).rename(columns = {"amount":"monetary"}).reset_index()

        rfm = recency.merge(frequency, how='outer', on=(['user'] +  group)).merge(monetary, how='outer', on=(['user'] +  group))

        rfm['recency_score'] = pd.qcut(rfm["recency"], 5, labels = [5,4,3,2,1])
        rfm['frequency_score'] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])
        rfm['monetary_score'] = pd.qcut(rfm['monetary'], 5, labels = [1, 2, 3, 4, 5])
        rfm['rfm_score'] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))

        rfm['segment'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)
        rfm['segment'] = rfm['segment'].replace(seg_map, regex=True)
        rfm = rfm.drop(columns=['recency', 'frequency', 'monetary'], axis=1)
        return rfm

    def get_X(self):
        if self.col_type == None:
            res = self.df.drop(['index', 'customerid', 'credit_card'] + self.y_col, axis = 1)
            return res
        else:
            # dat = dat.rename(columns={'income_x':'income'}, inplace = True)
            return self.df[self.col]

    def get_y(self):
        return self.df[[self.y_col]]

    def get_inverse_y(self, y):
        return self.le.inverse_transform(y)
    
    def preprocess(self, data):
        # print(data.columns)
        categorical_cols = ['campaignchannel', 'campaigntype', 'gender']
        numerical_cols = ['adspend', 'income', 'age']
        for col in categorical_cols:
            new_cols = pd.get_dummies(data[col], dtype=int)
            self.col += list(new_cols.columns)
            data = data.join(pd.get_dummies(data[col], dtype=int))
        data = data.drop(categorical_cols, axis=1)

        for col in numerical_cols:
            data[col] = self.x_scaler.fit_transform(data[[col]])

        # data['segment'] = self.le.fit_transform(data['segment'])
        new_cols = pd.get_dummies(data['segment'], dtype=int)
        data = data.join(new_cols)


        return data

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

# # segments = [
# #     'Hibernating', 'At Risk', 
# #     'Cannot Lose', 'About to Sleep', 
# #     'Need Attention', 'Loyal Customers', 
# #     'Promising', 'New Customers',
# #     'Potential Loyalists', 'Champions'
# # ]
# rfm = RFM(engine, col_type='market', y_col='Cannot Lose')
# X_train, X_test, y_train, y_test = rfm.get_dataset()
# print(pd.concat([X_train, y_train], axis=1).head())