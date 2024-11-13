import pandas as pd
import numpy as np

# from dateutil.relativedelta import relativedelta
# import matplotlib.pyplot as plt
# import seaborn as sns

import os
import sys
import warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesse
 
import shap
from dataset import engine, RFM, Churn, Engagement, RFM_engage, RFM_churn, Reco
from models import CLFSwitcher, Transform, Pipe, parameters, RecoSystem

from sklearn.model_selection import GridSearchCV

rfm = RFM(engine)

engage = Engagement(engine)

churn = Churn(engine)

def train(data):
    X = data.get_X()
    y = data.get_y()
    ct = Transform(data)
    X, y = ct.get_Xy()

    pipeline = Pipe(ct).get_pipeline()
    def train(X, y, pipeline, parameters):
            grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=12, return_train_score=True, verbose=2)
            grid_search.fit(X, y)
            return grid_search, grid_search.best_estimator_[-1]
    _, best_est = train(X, y, pipeline, parameters)
    best_est.explain(data, ct)

    return best_est

# customer_lst = ['Hibernating','At Risk','Loyal Customers','New Customers']
customer_lst = ['Loyal Customers']
explained_dct = {}
for customer in customer_lst:
    engage_explain = train(RFM_engage(rfm, engage, customer))
    churn_explain = train(RFM_churn(rfm, churn, customer))
    explained_dct['engage ' + customer] = engage_explain
    explained_dct['churn ' + customer] = churn_explain

def recommend(dataset=Reco(engine)):
    print('Modelling Recommendations...')
    df = dataset.df[['customer_id', 'deposits', 'cards', 'account', 'loan']]
    df['customer_id'] = df['customer_id'].astype(int)
    recosys = RecoSystem(df)
    reco = [recosys.hybrid(i) for i in df['customer_id']]
    data = pd.concat([df, pd.DataFrame(reco)], axis=1)
    data['customer_id'] = data['customer_id'].apply(lambda x: str(x).zfill(4))
    print('Recommendations Done...')
    return data

# from pathlib import Path

# root_dir = Path(__file__).resolve().parent
# env_path = os.path.join(str(root_dir),  "dataset/reco_df.csv")

# reco_df = pd.read_csv("dataset/reco_df.csv")
# reco_df['customer_id'] = reco_df['customer_id'].apply(lambda x: str(x).zfill(4))

reco_df = recommend()