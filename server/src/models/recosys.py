import pandas as pd
import numpy as np

from collections import defaultdict
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics.pairwise import pairwise_distances 
from sklearn.preprocessing import MinMaxScaler

from imblearn.over_sampling  import SMOTE
from imblearn.pipeline import Pipeline

# from tqdm import tqdm
# from tqdm.auto import tqdm
# from tqdm_loggable.auto import tqdm

class RecoSystem():
	def __init__(self, dataset):
		self.dataset = dataset
		self.df = dataset.df
		self.df['customer_id'] = self.df['customer_id'].astype(int)

		self.models = self.modeltrain()

	def recommend(self):
		reco = []
		df = self.df.sample(frac=0.2, random_state=10).reset_index(drop=True)
		max_iter = len(df)
		for i in df['customer_id']:
			# if i % (max_iter//10) == 0:
				# print("Inferring : {}/{}".format(i, max_iter))
			reco += [self.hybrid(i) ]
		# reco = [self.hybrid(i) for i in self.df['customer_id']]
		# reco = self.df['customer_id'].progress_apply(lambda x: self.hybrid(x))
		data = pd.concat([df, pd.DataFrame(reco)], axis=1)
		data['customer_id'] = data['customer_id'].apply(lambda x: str(x).zfill(4))
		# data = data.sample(frac=0.2, random_state=10)
		return data

	def popularity_based(self):
		"""
		Function that calculates the probability of a product occurring. 
		Probability range is <0, 1>.
		"""
		df = self.df[['customer_id', 'deposits', 'cards', 'account', 'loan']]
		top_col = {}
		for col in df.columns[1:]:
			top_col[col] = df[col].value_counts()[1]

		for k, v in top_col.items():
			top_col[k] = np.around(v / df.shape[0], decimals=4)
			
		return top_col

	def get_sim(self):
		# create the user-item similarity matrix
		# removes index names
		df = self.df[['customer_id', 'deposits', 'cards', 'account', 'loan']].set_index('customer_id')
		cosine_sim = 1 - pairwise_distances(df, metric="cosine")
		return cosine_sim

	def useritem(self, user_id):
		"""
		Function that calculates recommendations for a given user.
		It uses cosine similarity to calculate the most similar users.
		Returns the probability of products for a given user based on similar users.
		Probability range is <0, 1>.
		"""
		sim_matrix = self.get_sim()
		df = self.df[['customer_id', 'deposits', 'cards', 'account', 'loan']].set_index('customer_id')
		# computes the index in the user-item similarity matrix for a given user_id
		cos_id = list(df.index).index(user_id) 
		
		# number of similar users
		k = 0
		sim_min = 0.79
		user_sim_k = {}
		
		while k < 20:
			# creates the dictionary {'similar user':'similarity'}
			for user in range(len(df)):
				
				# 0.99 because I don`t want the same user as user_id
				if sim_min < sim_matrix[cos_id, user] < 0.99:
					user_sim_k[user] = sim_matrix[cos_id, user]
					k+=1
					
			sim_min -= 0.025
			
			# if there are no users with similarity at least 0.65, the recommendation probability will be set to 0 
			if sim_min < 0.65:
				break
				
		# sorted k most similar users
		user_sim_k = dict(sorted(user_sim_k.items(), key=lambda item: item[1], reverse=True))
		user_id_k = list(user_sim_k.keys()) 
		
		# dataframe with k most similar users
		df_user_k = df.iloc[user_id_k]
		df_user_k_T = df_user_k.T
		
		# change the user index to the cosine index
		df_user_k_T.columns = user_id_k
		
		# mean of ownership by k similar users
		ownership = []
		usit = {}
		for row_name, row in df_user_k_T.iterrows():
			
			for indx, own in row.items():
				
				ownership.append(own) 
			# print(row_name)
			usit[row_name] = np.mean(ownership)
			ownership = []
			
		# if there are no users with similarity at least 0.65, the recommendation probability is 0 
		if pd.isna(list(usit.values())[0]) == True:
			
			usit = {key : 0 for (key, value) in usit.items()}
				
		return usit

	def modeltrain(self):
		df = self.df.set_index('customer_id')
		df = df.drop(['person', 'retirement_age', 'address',
			'apartment', 'zipcode', 'per_capita_income','num_credit_cards',
			'fico_score', 'state', 'city', 'latitude', 'longitude', 'gender'], axis=1)
		df.rename(columns={'current_age':'age'}, inplace = True)
		df['yearly_income'] = df['yearly_income'].astype(np.float64)
		df['total_debt'] = df['total_debt'].astype(np.float64)
		# print(df)
		mdbs = {}
		
		model_dct = {}
		for c in df[['deposits', 'cards', 'account', 'loan']].columns:
			model = Pipeline([
				('scaler', MinMaxScaler()),
				('smote', SMOTE(sampling_strategy='auto', k_neighbors=3)),
				('clf', XGBRFClassifier(max_depth=5, n_estimators=100, random_state=10)),
			])
			y_train = df[c].astype('int')
			x_train = df.drop([c], axis = 1)
			model.fit(x_train, y_train)
			model_dct[c + "_model"] = model
		return model_dct

	def modelbased(self, user_id):
		"""
		Function that calculates recommendations for a given user.
		It uses machine learning model to calculate the probability of products.
		Probability range is <0, 1>.   
		"""
		df = self.df.set_index('customer_id')
		df = df.drop(['person', 'retirement_age', 'address',
			'apartment', 'zipcode', 'per_capita_income','num_credit_cards',
			'fico_score', 'state', 'city', 'latitude', 'longitude', 'gender'], axis=1)
		df.rename(columns={'current_age':'age'}, inplace = True)
		df['yearly_income'] = df['yearly_income'].astype(np.float64)
		df['total_debt'] = df['total_debt'].astype(np.float64)

		mdbs = {}
		
		model_dct = self.models
		for c in df[['deposits', 'cards', 'account', 'loan']].columns:
			model = model_dct[c + "_model"]
			y_train = df[c].astype('int')
			x_train = df.drop([c], axis = 1)
			p_train = model.predict_proba(x_train[x_train.index == user_id])[:,1]
			mdbs[c] = p_train[0]
		
		return mdbs
		

	def hybrid(self, user_id, f1=0.2, f2=0.60, f3=0.2):
		"""
		Function that calculates weighted hybrid recommendations for a given user.
		It uses weights to calculate the probability of products. 
		"""

		pb_h = self.popularity_based()
		ui_h = self.useritem(user_id)
		mb_h =  self.modelbased(user_id)

		hybrid = {}
		for k, v in pb_h.items():
			hybrid[k + '_reco'] = (v * f1) + (ui_h[k] * f2) + (mb_h[k] * f3)
		
		return hybrid

# recosys = RecoSystem(df)