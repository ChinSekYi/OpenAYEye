import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from .data import Data

class ROI(Data):
    def __init__(self, engine, query_string="""
            SELECT c.campaign_id,
            SUM(c.budget) AS mark_spent,
            c.channel as category,
            c.start_date as c_date,
            c.displays,
            SUM(CASE WHEN e.action_type = 'clicked' THEN 1 ELSE 0 END) AS clicks,
            SUM(CASE WHEN e.action_type = 'credentials' THEN 1 ELSE 0 END) AS leads, 
            SUM(CASE WHEN e.action_type = 'converted' THEN 1 ELSE 0 END) AS orders
            FROM campaign c, engagement e
            WHERE c.campaign_id = e.campaign_id
            GROUP BY c.campaign_id, c.channel;
            """
        ):
        super().__init__(engine, query_string)

    def preprocess(self):
        data = self.df.copy()
        data = data.drop(["campaign_id"], axis=1)
        # data.rename(columns={'current_age':'age'}, inplace = True)
        # cat_cols = self.get_cat_cols()
        data['mark_spent'] = data['mark_spent'].astype(np.float64)
        data['displays'] = data['displays'].astype(np.int64)
        data['clicks'] = data['clicks'].astype(np.int64)
        data['leads'] = data['leads'].astype(np.int64)
        data['orders'] = data['orders'].astype(np.int64)
        data['c_date'] = data['c_date'].astype(np.int64)
        data = pd.concat([data, pd.get_dummies(data['category'], dtype=int)], axis=1)
        data = data.drop(['category'], axis=1)
        return data

    def get_X(self):
        data = self.get_dataset()
        X = data.drop(['clicks', 'leads', 'orders'], axis = 1)
        return X

    def get_y(self):
        data = self.get_dataset()
        return data[['clicks', 'leads', 'orders']]
# from engine import engine
# print(engine)
# reco = Reco(engine)
# print(reco.df)