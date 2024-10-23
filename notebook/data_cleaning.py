import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

"""
functions starting with df_ can generate a processed dataframe directly
"""

# function to separate features and target
def get_Xy(df):
    X = df.iloc[:, 0 : len(df.columns) - 1]
    y = df.iloc[:, -1]
    return X, y


# function to handle missing values
def med_impute(X, y):
    # Remove columns with more than 40% null values
    thd1 = X.shape[0] * 0.4
    cols = X.columns[X.isnull().sum() < thd1]
    X = X[cols]

    # Remove rows with more than 50% null values
    thd2 = X.shape[1] * 0.5
    y = y[X.isnull().sum(axis=1) <= thd2]
    X = X[X.isnull().sum(axis=1) <= thd2]

    # Median imputation for remaining null values
    X = X.fillna(X.median())
    return X, y


# function to normalise numerical columns to remove effect of inconsistent scales
def normalise(df):
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return X_scaled


def check_skewness(df):
    X, y = get_Xy(df)
    skewness = X.skew()
    skewed_features = skewness[abs(skewness) > 0.5]
    num_skewed_features = len(skewed_features)
    return num_skewed_features

# function for feature selection
def drop_high_corr(X, threshold=0.7):
    correlation_matrix = X.corr()
    high_cor = []
    dropped_features = []

    # Iterate through the correlation matrix to find highly correlated pairs
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                if correlation_matrix.columns[j] != correlation_matrix.columns[i]:
                    high_cor.append(
                        [
                            correlation_matrix.columns[i],
                            correlation_matrix.columns[j],
                            correlation_matrix.iloc[i, j],
                        ]
                    )

    # Iterate through the list of highly correlated pairs
    for pair in high_cor:
        feature1, feature2, correlation = pair

        # Check if either of the features in the pair has already been dropped
        if feature1 not in dropped_features and feature2 not in dropped_features:
            # Check if the feature exists in the DataFrame before attempting to drop it
            if feature2 in X.columns:
                # Drop one of the correlated features from the dataset
                # Here, we arbitrarily choose to drop the second feature in the pair
                X.drop(feature2, axis=1, inplace=True)
                dropped_features.append(feature2)
            else:
                print("Feature '" + feature2 + "' not found in the DataFrame.")

    X.reset_index(drop=True, inplace=True)
    return X, dropped_features


def drop_corr_columns_from_test(X_test, dropped_features):
    X_test_dropped = X_test.drop(columns=dropped_features, errors="ignore")
    return X_test_dropped


def evaluate_model(model, params, *args):
    """
    Evaluate multiple models using GridSearchCV and return their R-squared scores.

    Args:
        x_train (array-like): Training input samples.
        y_train (array-like): Target values for training.
        x_test (array-like): Test input samples.
        y_test (array-like): Target values for testing.
        models (dict): Dictionary of models to evaluate.
        params (dict): Dictionary of parameter grids for GridSearchCV.

    Returns:
        dict: Dictionary containing model names as keys and their R-squared scores as values.
    """

    X_train = args[0]
    X_test = args[1]
    y_train = args[2]
    y_test = args[3]

    report = {}

    gs = GridSearchCV(model, params, cv=3)
    gs.fit(X_train, y_train)

    model.set_params(**gs.best_params_)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)

    y_test_pred = model.predict(X_test)

    train_model_r2score = r2_score(y_train, y_train_pred)

    test_model_r2score = r2_score(y_test, y_test_pred)

    return train_model_r2score, test_model_r2score


def train_and_evaluate_model(model, *args):
    """
    Train and evaluate a model using the provided data.
    Specifically used for ANOVA test
    """
    X_train = args[0]
    X_test = args[1]
    y_train = args[2]
    y_test = args[3]

    model.fit(X_train, y_train)

    # Calculate the accuracy of the model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    return train_accuracy, test_accuracy
