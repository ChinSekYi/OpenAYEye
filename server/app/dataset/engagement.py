import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from .data import Data
# from engine import engine

class Engagement(Data):
    def __init__(self, engine, 
            query_string = """
            SELECT c.*, e.customer_id, e.engagement_date, 
            e.action_type, e.device_type, e.feedback_score,
            e.conversion_value
            FROM campaign c, engagement e
            WHERE c.campaign_id = e.campaign_id;
            """
        ):
        super().__init__(engine, query_string)
    
    def preprocess(self):
        data = self.df.copy()
        data = data.drop(['campaign_id', 'campaign_name', 
            'customer_id', 'device_type',
            'feedback_score','conversion_value'], axis = 1)
        return data
    
    def get_num_cols(self):
        return super().get_num_cols()
    
    def get_cat_cols(self):
        return super().get_cat_cols()
    
    def get_dat_cols(self):
        return super().get_dat_cols()
    
    def get_X(self, col_type ='market'):
        data = self.preprocess()
        X = data.drop(['action_type'], axis = 1)
        return X
    
    def get_y(self):
        data = self.preprocess()
        return data[['action_type']]


# eg = Engagement(engine)
# print(eg.df)
# print(eg.get_num_cols())
# print(eg.get_cat_cols())
# print(eg.get_dat_cols())
# data = eg.preprocess()
# print(eg.get_X())
# print(eg.get_y())

