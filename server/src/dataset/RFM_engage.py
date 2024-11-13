import pandas as pd
import numpy as np

from .data import Data

class RFM_engage(Data):
    def __init__(self, RFM, engage, segment='Loyal Customers'):
        
        rfm = RFM.get_dataset()[['customer_id', 'segment']]
        engage = engage.get_dataset()

        self.segment = segment
        self.df = engage.merge(rfm, how='inner', on=['customer_id'])

    def preprocess(self):
        data = self.df.drop(['customer_id'], axis=1)
        data = data[data['segment'] == self.segment].drop(['segment'], axis=1)
        return data

    def get_X(self, ):
        data = self.preprocess()
        X = data.drop(['action_type'], axis=1)
        return X

    def get_y(self, segment='Loyal Customers'):
        data = self.preprocess()
        y = data[['action_type']]
        return y
        