import numpy as np
import pandas as pd

from .data import Data


class RFM_churn(Data):
    def __init__(self, RFM, churn, segment="Loyal Customers"):
        rfm = RFM.get_dataset()[["customer_id", "segment"]]
        churn = churn.get_dataset()

        self.segment = segment
        self.df = churn.merge(rfm, how="inner", on=["customer_id"])

    def preprocess(self):
        data = self.df.drop(["customer_id"], axis=1)
        data = data[data["segment"] == self.segment].drop(["segment"], axis=1)
        return data

    def get_X(self):
        data = self.preprocess()
        X = data.drop(["churn"], axis=1)
        return X

    def get_y(self, segment="Loyal Customers"):
        data = self.preprocess()
        y = data[["churn"]]
        return y
