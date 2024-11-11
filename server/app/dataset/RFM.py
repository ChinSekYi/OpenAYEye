import pandas as pd
import numpy as np

from .data import Data
# from engine import engine

class RFM(Data):
    def __init__(self, engine, 
            query_string = """
            SELECT u.*, t.amount, t.date
            FROM users u, transactions t
            WHERE u.customer_id = t.customer_id;
            """
        ):
        super().__init__(engine, query_string)

    def preprocess(self):
        data = self.df.copy()
        data = data.drop(['person', 'retirement_age',
            'birth_year_month', 'address', 'apartment', 'zipcode', 
            'city', 'latitude', 'longitude'], axis=1)

        data.rename(columns={'current_age':'age'}, inplace = True)
        return data

    def get_num_cols(self):
        data = self.get_dataset()
        cols = data.select_dtypes([np.number]).columns
        return cols

    def get_cat_cols(self):
        data = self.get_dataset()
        cols = data.select_dtypes(["category", "object"]).columns
        return cols

    def get_dat_cols(self):
        data = self.get_dataset()
        cols = data.select_dtypes([np.datetime64]).columns
        return cols

    def get_dataset(self):
        data = super().get_dataset().drop(["amount", "date"], axis=1).drop_duplicates(['customer_id'])
        rfm = self.get_RFM()[['customer_id', 'segment']]
        data = data.merge(rfm, on=['customer_id'])
        data = data.drop(["customer_id"], axis=1)
        return data

    def get_RFM(self, group=["customer_id", "age", "gender"], 
            seg_map = {
                r'[1-3][1-3]': 'Hibernating',
                r'[1-3][4-5]': 'At Risk',
                r'[4-5][4-5]': 'Loyal Customers',
                r'[4-5][1-3]': 'New Customers',
            }
            # seg_map = {
            #     r'[1-2][1-2]': 'Hibernating',
            #     r'[1-2][3-4]': 'At Risk',
            #     r'[1-2]5': 'Cannot Lose',
            #     r'3[1-2]': 'About to Sleep',
            #     r'33': 'Need Attention',
            #     r'[3-4][4-5]': 'Loyal Customers',
            #     r'41': 'Promising',
            #     r'51': 'New Customers',
            #     r'[4-5][2-3]': 'Potential Loyalists',
            #     r'5[4-5]': 'Champions'
            # }
        ):
        data = self.preprocess()
        data['day'] = data['date'].dt.day
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year
        # data = data.sort_values(by=group)
        recency = (data['date'].max() - data.groupby(group).agg({"date":"max"})).rename(columns = {"date":"recency"})
        recency['recency'] = recency['recency'].apply(lambda x: x.days)
        # recency = recency.reset_index()

        frequency = (data.groupby(group).agg({"date":"nunique"})).rename(columns = {"date":"frequency"}).reset_index()

        monetary = data.groupby(group).agg({"amount":"sum"}).rename(columns = {"amount":"monetary"}).reset_index()

        rfm = recency.merge(frequency, how='outer', on=(group)).merge(monetary, how='outer', on=(group))

        rfm['recency_score'] = pd.qcut(rfm["recency"], 5, labels = [5,4,3,2,1])
        rfm['frequency_score'] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])
        rfm['monetary_score'] = pd.qcut(rfm['monetary'], 5, labels = [1, 2, 3, 4, 5])

        rfm['rfm_score'] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))

        rfm['segment'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)
        rfm['segment'] = rfm['segment'].replace(seg_map, regex=True)

        # rfm = rfm.drop(columns=['recency', 'frequency', 'monetary'], axis=1)
        return rfm

    def get_X(self):
        data = self.get_dataset()
        X = data.drop(['segment'], axis = 1)
        return X

    def get_y(self):
        data = self.get_dataset()
        return data[['segment']]

# segments = [
#     'Hibernating', 'At Risk', 
#     'Cannot Lose', 'About to Sleep', 
#     'Need Attention', 'Loyal Customers', 
#     'Promising', 'New Customers',
#     'Potential Loyalists', 'Champions'
# ]
# rfm = RFM(engine)

# print(rfm.df.columns)
# print(rfm.get_num_cols())
# print(rfm.get_cat_cols())
# print(rfm.get_dat_cols())

# print(rfm.get_dataset()['segment'])