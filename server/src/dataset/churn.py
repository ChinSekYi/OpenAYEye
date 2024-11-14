import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .data import Data


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
                "person",
                "retirement_age",
                "address",
                "apartment",
                "zipcode",
                "per_capita_income",
                "num_credit_cards",
                "fico_score",
                "state",
                "city",
                "latitude",
                "longitude",
            ],
            axis=1,
        )
        data = data.rename(columns={"current_age": "age", "churn_date": "churn"})
        data["yearly_income"] = data["yearly_income"].astype(np.float64)
        data["total_debt"] = data["total_debt"].astype(np.float64)
        # data['age'] = pd.qcut(data['age'], [0, .33, .67, 1.], labels=[0, 1, 2])
        # data['yearly_income'] = pd.qcut(data['yearly_income'].astype(np.float64), [0, 0.33, .67, 1.], labels=[0, 1, 2])
        # data['total_debt'] = pd.qcut(data['total_debt'].astype(np.float64), [0, 0.33, .67, 1.], labels=[0, 1, 2])
        data["churn"] = data["churn"].apply(lambda x: "no" if pd.isnull(x) else "yes")
        # print(data['churn'])
        # data = data.drop(["customer_id"], axis=1)
        return data

    def get_num_cols(self):
        return super().get_num_cols()

    def get_cat_cols(self):
        return super().get_cat_cols()

    def get_dat_cols(self):
        return super().get_dat_cols()

    def get_X(self):
        data = self.preprocess()
        X = data.drop(["churn"], axis=1)
        return X

    def get_y(self):
        data = self.preprocess()
        return data[["churn"]]


# from engine import engine
# print(engine)
# churn = Churn(engine)
# print(churn.get_dataset())
