
import pandas as pd
import numpy as np

import os
import sys
import warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesse
 
from dataset import engine, RFM, Churn, Engagement
from models import CLFSwitcher, Transform, Pipe, parameters
from sklearn.model_selection import GridSearchCV

def train(X, y, pipeline, parameters):
        grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=12, return_train_score=True, verbose=2)
        grid_search.fit(X, y)
        return grid_search, grid_search.best_estimator_[-1]

def main():
    rfm = RFM(engine)
    churn = Churn(engine)
    engage = Engagement(engine)

    datasets = [rfm, churn, engage]

    best_estimators = {}

    for dt in datasets:

        ct = Transform(churn)
        X, y = ct.get_Xy()
        pipeline = Pipe(ct).get_pipeline()
        _, best_estimator = train(X, y, pipeline, parameters)
        print(best_estimator.class_report(X, y))
        best_estimators[dt.__class__.__name__] = best_estimator    
        # print(best_estimator.feature_importance())

    return best_estimators


if __name__ == "__main__":
    main()