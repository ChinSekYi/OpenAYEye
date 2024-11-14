import numpy as np
import pandas as pd
import sqlalchemy


class Data:
    def __init__(self, engine, query_string):
        self.engine = engine
        with engine.connect() as db:
            query_string = sqlalchemy.text(query_string)
            fetched = pd.DataFrame(db.execute(query_string).fetchall())
            db.close()
        self.df = fetched

    def preprocess(self):
        pass

    def get_num_cols(self):
        data = self.preprocess()
        cols = data.select_dtypes([np.number]).columns
        return cols

    def get_cat_cols(self):
        data = self.preprocess()
        cols = data.select_dtypes(["category", "object"]).columns
        return cols

    def get_dat_cols(self):
        data = self.preprocess()
        cols = data.select_dtypes([np.datetime64]).columns
        return cols

    def get_dataset(self):
        data = self.preprocess()
        return data

    def get_X(self):
        pass

    def get_y(self):
        pass
