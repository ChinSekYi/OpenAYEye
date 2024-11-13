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
from dataset import engine, RFM, Churn, Engagement, RFM_engage, RFM_churn
from models import CLFSwitcher, Transform, Pipe, parameters
from sklearn.model_selection import GridSearchCV

rfm = RFM(engine)

engage = Engagement(engine)

churn = Churn(engine)
# churn = churn.get_dataset()[['customer_id', 'churn']]

def train(data):
    X = data.get_X()
    y = data.get_y()
    ct = Transform(data)
    X, y = ct.get_Xy()
    # ct.inverse_transform(pd.concat([X, y], axis=1))
    # y

    pipeline = Pipe(ct).get_pipeline()
    def train(X, y, pipeline, parameters):
            grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=12, return_train_score=True, verbose=2)
            grid_search.fit(X, y)
            return grid_search, grid_search.best_estimator_[-1]
    _, best_est = train(X, y, pipeline, parameters)
    best_est.explain(data, ct)

    return best_est

customer_lst = ['Hibernating','At Risk','Loyal Customers','New Customers']
explained_dct = {}
for customer in customer_lst:
    engage_explain = train(RFM_engage(rfm, engage, customer))
    churn_explain = train(RFM_churn(rfm, churn, customer))
    explained_dct['engage ' + customer] = engage_explain
    explained_dct['churn ' + customer] = churn_explain
    # print(explained_dct)

# shap_df = best_est.get_shap(X_col='engage_month', y_col='action_type', y_val='converted')
# shap_df


# plt.scatter(shap_df['engage_month__converted'], shap_df['shap'])


