import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from .data import Data

# from engine import engine


class Engagement(Data):
    def __init__(
        self,
        engine,
        query_string="""
            SELECT c.*, e.customer_id, e.engagement_date, 
            e.action_type, e.device_type, e.feedback_score,
            e.conversion_value
            FROM campaign c, engagement e
            WHERE c.campaign_id = e.campaign_id;
            """,
    ):
        super().__init__(engine, query_string)

    def preprocess(self):
        data = self.df.copy()
        data = data.drop(
            [
                "campaign_id",
                "campaign_name",
                "displays",
                "end_date",
                "device_type",
                "conversion_value",
            ],
            axis=1,
        )
        data["budget"] = data["budget"].astype(np.float64)
        data["feedback_score"] = data["feedback_score"].astype(np.float64)

        # data['budget'] = pd.qcut(data['budget'].astype(np.float64), [0, .33, .67, 1.], labels=[0, 1, 2])
        data["engagement_date"] = data["engagement_date"].dt.month.astype("category")
        data["start_date"] = data["start_date"].dt.month.astype("category")

        data = data.rename(
            columns={"engagement_date": "engage_month", "start_date": "campaign_month"}
        )
        obj_cols = data.select_dtypes(["object"]).columns
        data[obj_cols] = data[obj_cols].astype("category")

        return data

    def get_num_cols(self):
        return super().get_num_cols()

    def get_cat_cols(self):
        return super().get_cat_cols()

    def get_dat_cols(self):
        return super().get_dat_cols()

    def get_X(self, col_type="market"):
        data = self.preprocess()
        X = data.drop(["action_type"], axis=1)
        return X

    def get_y(self):
        data = self.preprocess()
        return data[["action_type"]]


# eg = Engagement(engine)
# print(eg.df)
# print(eg.get_num_cols())
# print(eg.get_cat_cols())
# print(eg.get_dat_cols())
# data = eg.preprocess()
# print(eg.get_X())
# print(eg.get_y())
