import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .data import Data


class Reco(Data):
    def __init__(
        self,
        engine,
        query_string="""
            SELECT *
            FROM users u
            """,
    ):
        super().__init__(engine, query_string)


# from engine import engine
# print(engine)
# reco = Reco(engine)
# print(reco.df)
