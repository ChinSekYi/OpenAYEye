import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler


class Transform():
	def __init__(self, dataset):
		self.df = dataset.get_dataset()
		self.cat = dataset.get_cat_cols()
		self.num = dataset.get_num_cols()
		self.ct = ColumnTransformer(
			[
				("num_preprocess", MinMaxScaler(), self.num),
				("cat_preprocess", OrdinalEncoder(), self.cat)
			]
		)
		self.X = list(dataset.get_X().columns)
		self.y = list(dataset.get_y().columns)

	def fit_transform(self):
		# df = self.df
		# df[self.num] = self.ct.fit_transform(self.df)
		df = self.ct.fit_transform(self.df)
		# print(self.cat, self.num)
		# print(df.columns)
		df = pd.DataFrame(df, columns=list(self.num) + list(self.cat))
		return df

	def inverse_transform(self, df):
		# df = pd.DataFrame(df, columns=list(self.num) + list(self.cat))
		df[self.num] = self.ct.named_transformers_['num_preprocess'].inverse_transform(df[self.num])
		df[self.cat] = self.ct.named_transformers_['cat_preprocess'].inverse_transform(df[self.cat])
		return df
	
	def get_Xy(self):
		data = self.fit_transform()
		X = data[self.X]
		y = data[self.y]
		return X, y