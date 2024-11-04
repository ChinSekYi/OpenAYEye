from xgboost import XGBRFClassifier, XGBRFRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

from sklearn.base import BaseEstimator

# import sklearn

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KDTree
# from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRFClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

class CLFSwitcher(BaseEstimator):

	def __init__(
			self, 
			estimator = SGDClassifier(),
		):
		"""
		A Custom BaseEstimator that can switch between classifiers.
		:param estimator: sklearn object - The classifier
		""" 
		self.linear_lst = [i.__name__ for i in [
			SGDClassifier,
		]]
		self.NB_lst = [i.__name__ for i in [
			MultinomialNB,
		]]
		self.tree_lst = [i.__name__ for i in [
			XGBClassifier,
			XGBRFClassifier,
		]]
		# self.nn_lst = [i.__name__ for i in [
		# 	MLPClassifier,
		# ]]
		self.non_lin_lst = [i.__name__ for i in[
			KDTree,
		]]
		self.estimator = estimator


	def fit(self, X, y=None, **kwargs):
		self.estimator.fit(X, y)
		return self


	def predict(self, X, y=None):
		return self.estimator.predict(X)


	def predict_proba(self, X):
		return self.estimator.predict_proba(X)

	def class_report(self, X, y):
		return classification_report(y, self.predict(X))

	def score(self, X, y):
		return f1_score(y, self.predict(X), average='micro')

	def feature_importance(self):
		model_name = self.estimator.__class__.__name__
		columns = self.estimator.feature_names_in_
		# is_linear = model_name in self.linear_lst
		# print(is_linear)
		if model_name in self.linear_lst:
			coef = self.estimator.coef_[0]
			# return list(zip(coef, columns))
		elif model_name in self.NB_lst:
			coef = self.estimator.feature_log_prob_[0]
			# return list(zip(coef, columns))
		elif model_name in self.tree_lst:
			coef = self.estimator.feature_importances_
			# return list(zip(coef, columns))
		return pd.DataFrame({"features": columns, "significance": coef}).sort_values(by=['significance'], ascending=False)
		