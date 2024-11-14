import os
import sys
import warnings

import numpy as np
import pandas as pd

# from dateutil.relativedelta import relativedelta
# import matplotlib.pyplot as plt
# import seaborn as sns

warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesse

import shap
from dataset import RFM, ROI, Churn, Engagement, Reco, RFM_churn, RFM_engage, engine
from models import (
    CLFSwitcher,
    Pipe,
    RecoSystem,
    REGSwitcher,
    Transform,
    parameters,
    reg_parameters,
)
from sklearn.model_selection import GridSearchCV

rfm = RFM(engine)

engage = Engagement(engine)

churn = Churn(engine)



def train_roi(data):
    X = data.get_X()
    y = data.get_y()
    # ct = Transform(data)
    # X, y = ct.get_Xy()

    pipeline = Pipe(task="REG").get_pipeline()

    def train_roi(X, y, pipeline, parameters):
        grid_search = GridSearchCV(
            pipeline,
            reg_parameters,
            cv=5,
            n_jobs=12,
            return_train_score=True,
            verbose=1,
        )
        # grid_search = pipeline
        grid_search.fit(X, y)
        return grid_search, grid_search.best_estimator_[-1]

    _, best_est = train_roi(X, y, pipeline, parameters)
    # print(best_est.predict(X))
    return best_est


roi = ROI(engine)
roi_est = train_roi(roi)

print("Fitting for recomendations, totalling 1 fits")
reco = Reco(engine)
recosys = RecoSystem(reco)
reco_df = recosys.recommend()


def train(data):
    X = data.get_X()
    y = data.get_y()
    ct = Transform(data)
    X, y = ct.get_Xy()

    pipeline = Pipe(ct).get_pipeline()

    def train(X, y, pipeline, parameters):
        grid_search = GridSearchCV(
            pipeline, parameters, cv=5, n_jobs=12, return_train_score=True, verbose=1
        )
        # grid_search = pipeline
        grid_search.fit(X, y)
        return grid_search, grid_search.best_estimator_[-1]

    _, best_est = train(X, y, pipeline, parameters)
    best_est.explain(data, ct)

    return best_est

customer_lst = ["Hibernating", "At Risk", "Loyal Customers", "New Customers"]
# customer_lst = ['Loyal Customers']
explained_dct = {}
for customer in customer_lst:
    engage_explain = train(RFM_engage(rfm, engage, customer))
    churn_explain = train(RFM_churn(rfm, churn, customer))
    explained_dct["engage " + customer] = engage_explain
    explained_dct["churn " + customer] = churn_explain
