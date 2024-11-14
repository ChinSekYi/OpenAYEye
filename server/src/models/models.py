import pandas as pd

from sklearn.base import BaseEstimator

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KDTree
# from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRFClassifier, XGBRegressor

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

import shap

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
		# return f1_score(y, self.predict(X), average='micro')
		return self.estimator.score(X, y)

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
		
	def explain(self, data, ct):
		self.data = data
		self.ct = ct
		X, y = ct.get_Xy()
		instance = X.sample(n=100, random_state=1)
		# print(instance)	
		explainer = shap.Explainer(self.estimator, instance)
		shap_values = explainer(instance)
		self.shap_values = shap_values
		# shap_val = self.get_shap(data, shap_values, ct, X_col, y_col, y_val)
		return self.shap_values

	def get_shap(self, X_col='engage_month', y_col='action_type', y_val='converted'):
		cat_dict = {v:k for k, v in zip(self.ct.ct['cat_preprocess'].categories_, self.data.get_cat_cols())}
		# print(cat_dict)
		y_col = {k: v for v, k in enumerate(cat_dict[y_col])}
		if len(self.shap_values.shape) == 3:
			shap_val = self.shap_values[:, X_col,  y_col[y_val]]
		else:
			shap_val = self.shap_values[:, X_col]
		if X_col in cat_dict.keys():
			# print(self.shap_values.shape)
			dat = [cat_dict[X_col][int(i)] for i in shap_val.data]
		else:
			dat = shap_val.data
			
		val = shap_val.values
			
		df = pd.DataFrame({'shap': val, (X_col + "__" +  y_val): dat})
		return df

class REGSwitcher(BaseEstimator):

	def __init__(
			self, 
			estimator = XGBRegressor(),
		):
		"""
		A Custom BaseEstimator that can switch between Regressors.
		:param estimator: sklearn object - The Regressors
		""" 
		# self.linear_lst = [i.__name__ for i in [
		# 	SGDClassifier,
		# ]]
		# self.NB_lst = [i.__name__ for i in [
		# 	MultinomialNB,
		# ]]
		# self.tree_lst = [i.__name__ for i in [
		# 	XGBClassifier,
		# 	XGBRFClassifier,
		# ]]
		# # self.nn_lst = [i.__name__ for i in [
		# # 	MLPClassifier,
		# # ]]
		# self.non_lin_lst = [i.__name__ for i in[
		# 	KDTree,
		# ]]
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
		return self.estimator.score(X, y)