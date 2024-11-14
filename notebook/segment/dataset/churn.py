import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .data import Data

# from engine import engine


class Churn(Data):
    def __init__(
        self,
        engine,
        query_string="""
            SELECT u.*, c.churn_date
            FROM users u
            LEFT JOIN churn c
            ON u.customer_id = c.customer_id;
            """,
    ):
        super().__init__(engine, query_string)

    def preprocess(self):
        data = self.df.copy()
        data = data.drop(
            [
                "customer_id",
                "person",
                "retirement_age",
                "birth_year_month",
                "address",
                "apartment",
                "zipcode",
                "city",
                "state",
                "latitude",
                "longitude",
            ],
            axis=1,
        )
        data["churn"] = np.where(data["churn_date"].isna(), 0, 1)
        return data

    def get_num_cols(self):
        return super().get_num_cols()

    def get_cat_cols(self):
        return super().get_cat_cols()

    def get_dat_cols(self):
        return super().get_dat_cols()

    def get_X(self):
        data = self.preprocess()
        X = data.drop(["churn", "churn_date"], axis=1)
        return X

    def get_y(self):
        data = self.preprocess()
        return data[["churn"]]


# churn = Churn(engine)
# print(churn.get_X())
# print(churn.get_y())
