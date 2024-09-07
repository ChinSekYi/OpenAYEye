import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.stats import boxcox
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler



# function to separate features and target
def get_Xy(df):
    pass
    return X, y


# function to handle missing values
def med_impute(X, y):
    pass
    return X, y


# function to normalise numerical columns to remove effect of inconsistent scales
def normalise(df):
    pass
    return X_scaled