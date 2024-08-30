import time
import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler

from copy import deepcopy

class Sliding():
    def __init__(self, dataset, X_period=7, y_period=1):
        self.dataset = dataset
        self.X_period = X_period
        self.y_period = y_period
        self.scaler = MinMaxScaler(feature_range=(0,1))

    def get_Dataset(self):
        formatted = self.format_dataset(self.dataset)
        slided = self.slide(formatted)
        slided = slided.to_numpy()
        scaled = self.scaler.fit_transform(slided)

        X = deepcopy(np.flip(scaled[:, 1:], axis=1))
        y = scaled[:, 0]

        X_split = self.tt_split(X)
        y_split = self.tt_split(y)

        X_split = tuple(map(lambda x: self.to_tensor(x), X_split))
        y_split = tuple(map(lambda x: self.to_tensor(x), y_split))

        train, val, test = zip(X_split, y_split)

        train_set = SlideDataset(train)
        val_set = SlideDataset(val)
        test_set = SlideDataset(test)
        return train_set, val_set, test_set

    def format_dataset(self, dataset):
        return dataset[['Close']]

    def tt_split(self, data):
        train_size = int(len(data)*0.8)
        val_split = int(train_size*0.9)
        train = data[:train_size]
        test = data[train_size - self.X_period:]

        valid = train[val_split:]
        train = train[:val_split]
        return train, valid, test

    def to_tensor(self, data):
        return torch.tensor(data).unsqueeze(-1)

    def slide(self, data):
        data = deepcopy(data)
        for i in range(1, self.X_period + 1):
            data[f'Close (t-{i})'] = data['Close'].shift(i)
        data.dropna(inplace=True)

        return data
    
    def unscale(self, config, pred):
        dummies = np.zeros((pred.shape[0], config.WINDOW_SIZE+1))
        dummies[:, 0] = pred.flatten()
        dummies = self.scaler.inverse_transform(dummies)
        pred = dummies[:, 0]
        return pred

def yh_get_company_dat(ticker="GOOGL", 
        period1 = int(time.mktime(datetime.datetime(2000, 1, 1, 23, 59).timetuple())), 
        period2 = int(time.mktime(datetime.datetime(2023, 12, 31, 23, 59).timetuple())),
        interval = '1d'
    ):
	# Get the date, open, high, low, adj close of a company
	# Input of the date is (year, month, day, hour, min) in the form of datetime object
	query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history'

	df = pd.read_csv(query_string)

	# Format date column
	date_format = "%Y-%m-%d"
	df['Date'] = pd.to_datetime(df['Date']).dt.strftime(date_format)

	# Remove the irrelevant rows
	df['Date'] = df['Date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
	df.set_index('Date', inplace=True)
	df.sort_index(inplace=True)	

	return df

class SlideDataset(Dataset):
    def __init__(self, data, transform=None):
        self.X, self.y = data

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def get_X(self):
        return self.X.detach().cpu().numpy().flatten()
    
    def get_y(self):
        return self.y.detach().cpu().numpy().flatten()